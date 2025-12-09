"""
SvgBgCleaner agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
参考 TechnicalRouteDescGenerator，实现对 SVG 代码的“去文本”清洗。

职责：
- 从 MainState 中读取原始 SVG 源码（通常在 figure_tec_svg_content 字段）；
- 调用 LLM，根据 prompt 清洗掉所有文本元素，仅保留图形相关元素；
- 将结果写入 state.agent_results["svg_bg_cleaner"]["svg_bg_code"]，供 workflow 使用；
- 提供严格的 ReAct 验证器：
  1. JSON 结构正确，包含 svg_bg_code；
  2. svg_bg_code 是合法 SVG（XML 解析通过）；
  3. svg_bg_code 中不再包含 <text>、<tspan>、<title> 标签。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)


@register("svg_bg_cleaner")
class SvgBgCleaner(BaseAgent):
    """对 SVG 代码进行“去文本”清洗的 Agent"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "svg_bg_cleaner"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_svg_bg_cleaner"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_svg_bg_cleaner"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将当前 state 中的原始 SVG 代码传入 prompt。
        约定：
        - 上游已将 SVG 源码写入 state.figure_tec_svg_content。
        """
        return {
            "svg_code": self.state.figure_tec_svg_content
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    # ---------- ReAct 验证器 ----------
    def get_react_validators(self) -> List:
        """
        验证器签名：
            validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]

        这里对 parsed_result["svg_bg_code"] 做两类检查：
        1) XML 层面：是否为合法 SVG；
        2) 语义层面：是否不再包含文本相关标签（text / tspan / title）。
        """

        def _extract_svg_fragment(svg_code: str) -> str:
            """
            从模型返回的字符串中，提取出干净的 <svg>...</svg> 片段。

            步骤：
            1. 去掉首尾空白；
            2. 去掉可能的 ``` / ```svg 代码块包裹；
            3. 截取第一个 <svg ...> 到最后一个 </svg> 之间的部分。
            """
            if not svg_code:
                return ""

            text = svg_code.strip()

            # 处理 ```svg ... ``` 或 ``` ... ``` 代码块
            if text.startswith("```"):
                lines = [line for line in text.splitlines() if line.strip("`").strip()]
                for i, line in enumerate(lines):
                    if "<svg" in line or "<SVG" in line or "<Svg" in line or "<svg" in line or "<SVG" in line:
                        text = "\n".join(lines[i:])
                        break

            # 既兼容已转义的 <svg>，也兼容原始 <svg>
            candidates = ["<svg", "<SVG", "<Svg", "<svg", "<SVG", "<Svg"]
            start = -1
            for c in candidates:
                start = text.find(c)
                if start != -1:
                    break

            end = -1
            for c in ["</svg>", "</svg>"]:
                pos = text.rfind(c)
                if pos != -1:
                    end = pos + len(c)
                    break

            if start == -1 or end == -1:
                return text

            return text[start:end].strip()

        def validate_svg_bg(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """
            SvgBgCleaner 的 SVG 验证器：
            1. 检查 parsed_result 中存在 svg_bg_code；
            2. 检查 svg_bg_code XML well-formed；
            3. 检查不包含 text/tspan/title 标签（大小写不敏感）。
            """
            import re
            import xml.etree.ElementTree as ET

            if not isinstance(parsed_result, dict):
                return False, "解析结果不是字典，无法找到 svg_bg_code 字段"

            svg_bg_code = parsed_result.get("svg_bg_code")
            if not svg_bg_code:
                return False, "缺少 svg_bg_code 字段或内容为空"

            fragment = _extract_svg_fragment(svg_bg_code)
            if "svg" not in fragment.lower():
                return False, "返回内容中未检测到 <svg> 根标签"

            # 1) XML 合法性检查
            try:
                # 如果是 HTML 实体转义过的 <svg> 形式，先简单还原
                xml_text = fragment.replace("<", "<").replace(">", ">")
                ET.fromstring(xml_text)
            except Exception as e:
                log.warning(f"svg_bg_cleaner.validate_svg_bg: SVG XML 解析失败: {e}")
                return False, f"svg_bg_code 不是合法 XML: {e}"

            # 2) 文本标签检查：不应再包含 text / tspan / title
            lowered = fragment.lower()
            text_like = ["<text", "<text", "<tspan", "<tspan", "<title", "<title"]
            if any(t in lowered for t in text_like):
                return False, "svg_bg_code 中仍包含 text/tspan/title 文本相关标签"

            return True, "SVG 背景清洗结果验证通过"

        return [validate_svg_bg]

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        期望 LLM 返回：
        {"svg_bg_code": "<svg ...>...</svg>"}
        """
        svg_bg_code = None
        if isinstance(result, dict):
            svg_bg_code = result.get("svg_bg_code")

        state.svg_bg_code = svg_bg_code

        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def svg_bg_cleaner(
    state: MainState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_vlm: bool = False,
    vlm_config: Optional[Dict[str, Any]] = None,
    use_agent: bool = False,
    **kwargs,
) -> MainState:
    """
    svg_bg_cleaner 的异步入口
    """
    agent = SvgBgCleaner(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_svg_bg_cleaner(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_vlm: bool = False,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> SvgBgCleaner:
    """
    工厂函数，便于在其他模块中创建 svg_bg_cleaner agent。
    """
    return SvgBgCleaner.create(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
        **kwargs,
    )
