"""
technical_route_bw_svg_generator agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于模板 PNG 生成黑白技术路线图 SVG
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)


@register("technical_route_bw_svg_generator")
class TechnicalRouteBWSvgGenerator(BaseAgent):
    """参考模板 PNG 生成黑白技术路线图 SVG 的 Agent"""

    @property
    def role_name(self) -> str:
        return "technical_route_bw_svg_generator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_technical_route_bw_svg_generator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_technical_route_bw_svg_generator"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paper_idea": pre_tool_results.get("paper_idea", ""),
            "template_svg_code": pre_tool_results.get("template_svg_code", ""),
            "validation_feedback": pre_tool_results.get("validation_feedback", ""),
            "lang": self.state.request.language,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "paper_idea": "",
            "template_svg_code": "",
            "validation_feedback": "",
        }

    def get_react_validators(self) -> List:
        """返回 ReAct 模式下使用的验证器列表"""

        # 获取用户要求的语言（用于语言验证）
        required_lang = getattr(getattr(self.state, "request", None), "language", "en").lower()

        def _extract_svg_fragment(svg_code: str) -> str:
            """提取干净的 SVG 片段"""
            if not svg_code:
                return ""
            text = svg_code.strip()
            if text.startswith("```"):
                lines = [line for line in text.splitlines() if line.strip("`").strip()]
                for i, line in enumerate(lines):
                    if "<svg" in line or "<SVG" in line:
                        text = "\n".join(lines[i:])
                        break
            start = text.find("<svg")
            end = text.rfind("</svg>")
            if start == -1 or end == -1:
                return text
            end += len("</svg>")
            return text[start:end].strip()

        def _inject_chinese_font_for_validation(svg_code: str) -> str:
            """为验证注入中文字体（与 workflow 中的逻辑一致）"""
            import re

            # 检查是否包含中文字符
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', svg_code))
            if not has_chinese:
                return svg_code

            # 中文友好字体列表
            chinese_fonts = 'Noto Sans CJK SC, Microsoft YaHei, SimHei, SimSun, WenQuanYi Zen Hei, sans-serif'

            # 替换所有 font-family 属性
            svg_code = re.sub(
                r'font-family="[^"]*"',
                f'font-family="{chinese_fonts}"',
                svg_code
            )

            # 注入全局样式
            idx = svg_code.find(">")
            if idx != -1:
                style_block = f"""
  <style type="text/css">
    text, tspan {{
      font-family: {chinese_fonts} !important;
    }}
  </style>
"""
                svg_code = svg_code[:idx + 1] + style_block + svg_code[idx + 1:]

            return svg_code

        def validate_svg_structure(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """验证 SVG 基本结构"""
            import xml.etree.ElementTree as ET

            svg_code = parsed_result.get("svg_code", "")
            if not svg_code:
                return False, "缺少 svg_code 字段或内容为空"

            fragment = _extract_svg_fragment(svg_code)
            if "<svg" not in fragment.lower():
                return False, "未检测到 <svg> 根标签"

            if "viewBox" not in fragment and "viewbox" not in fragment:
                return False, "缺少 viewBox 属性，请添加 viewBox"

            try:
                ET.fromstring(fragment)
            except Exception as e:
                return False, f"SVG XML 解析失败: {e}"

            return True, ""

        def validate_chinese_font(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """验证中文字体设置"""
            import re

            svg_code = parsed_result.get("svg_code", "")
            if not svg_code:
                return True, ""  # 如果没有 SVG，跳过此验证

            # 检查是否包含中文字符
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', svg_code))
            if not has_chinese:
                return True, ""  # 没有中文，跳过验证

            # 检查是否使用了不支持中文的字体
            bad_fonts = ["Arial", "Helvetica", "Times", "Courier"]
            for font in bad_fonts:
                if f'font-family="{font}"' in svg_code or f"font-family='{font}'" in svg_code:
                    return False, (
                        f"检测到中文文本但使用了不支持中文的字体 {font}。"
                        f"请使用中文友好字体，如：'Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'SimSun', sans-serif"
                    )

            return True, ""

        def validate_language_requirement(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """验证 SVG 文本语言是否符合用户要求"""
            import re

            svg_code = parsed_result.get("svg_code", "")
            if not svg_code:
                return True, ""  # 如果没有 SVG，跳过此验证

            # 检查是否包含中文字符
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', svg_code))

            # 如果用户要求英文，但 SVG 包含中文，则验证失败
            if required_lang in ["en", "english"] and has_chinese:
                return False, (
                    f"用户要求使用英文（language={required_lang}），但生成的 SVG 包含中文字符。"
                    f"请严格按照用户要求的语言生成 SVG，所有文本标签必须使用英文。"
                    f"例如：'Data Processing', 'Model Training', 'Feature Extraction' 等。"
                )

            # 如果用户要求中文，但 SVG 不包含中文，给出提示（警告而非错误）
            if required_lang in ["zh", "chinese", "cn", "中文"] and not has_chinese:
                # 这里不返回失败，因为可能是纯图形的 SVG
                # 但可以在日志中记录
                log.warning(f"用户要求使用中文（language={required_lang}），但生成的 SVG 不包含中文字符")

            return True, ""

        def validate_svg_renderable(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """验证 SVG 是否可以成功渲染"""
            import tempfile
            import os

            svg_code = parsed_result.get("svg_code", "")
            if not svg_code:
                return True, ""  # 如果没有 SVG，跳过此验证

            # 提取 SVG 片段
            fragment = _extract_svg_fragment(svg_code)

            # 注入中文字体（模拟 workflow 的行为）
            fragment_with_font = _inject_chinese_font_for_validation(fragment)

            # 尝试渲染 SVG
            try:
                from dataflow_agent.toolkits.multimodaltool.bg_tool import local_tool_for_svg_render

                # 创建临时文件用于渲染测试
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    local_tool_for_svg_render({
                        "svg_code": fragment_with_font,
                        "output_path": tmp_path,
                    })
                    # 渲染成功，删除临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return True, ""
                except Exception as e:
                    # 渲染失败，删除临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return False, f"SVG 渲染失败: {str(e)}。请检查 SVG 代码的格式和结构，确保所有属性值都被正确引号包裹。"
            except Exception as e:
                return False, f"渲染验证过程出错: {str(e)}"

        return [validate_svg_structure, validate_chinese_font, validate_language_requirement, validate_svg_renderable]

    def update_state_result(
        self,
        state: Paper2FigureState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        svg_code = None
        if isinstance(result, dict):
            svg_code = result.get("svg_code")
        state.figure_tec_svg_bw_content = svg_code or ""
        super().update_state_result(state, result, pre_tool_results)


def create_technical_route_bw_svg_generator(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    parser_type: str = "json",
    **kwargs,
) -> TechnicalRouteBWSvgGenerator:
    """
    创建技术路线图黑白 SVG 生成器。

    注意: 不再使用 VLM (视觉语言模型),而是通过 pre_tool 提供 SVG 模板代码作为文本输入。
    """
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()

    return TechnicalRouteBWSvgGenerator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        parser_type=parser_type,
        use_vlm=False,  # 不再使用 VLM
        **kwargs,
    )
