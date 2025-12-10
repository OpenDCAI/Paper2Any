"""
TechnicalRouteDescGenerator agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-08 00:44:59
生成位置: dataflow_agent/agentroles/common_agents/technical_route_desc_generator_agent.py

本文件由 `dfa create --agent_name technical_route_desc_generator` 自动生成。
1. 填写 prompt-template 名称
2. 根据需要完成 get_task_prompt_params / update_state_result
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("technical_route_desc_generator")
class TechnicalRouteDescGenerator(BaseAgent):
    """TODO: 描述 technical_route_desc_generator 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "technical_route_desc_generator"

    @property
    def system_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "system_prompt_for_technical_route_desc_generator"

    @property
    def task_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "task_prompt_for_technical_route_desc_generator"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        return {
            'text2img_prompt': pre_tool_results.get('prompt', ''),
            'image_size': pre_tool_results.get('size', '512x512'),
            'num_images': pre_tool_results.get('num_images', 1),
        }
        """
        # TODO: 按需补充
        return {
            "paper_idea": pre_tool_results.get("paper_idea", "")
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """若调用方未显式传入，返回默认前置工具结果"""
        return {}

    # ---------- ReAct 验证器 ----------
    def get_react_validators(self) -> List:
        """
        返回 ReAct 模式下使用的验证器列表。

        验证器签名固定为:
            validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]
        这里基于 parsed_result["svg_code"] 做 SVG XML 合法性检查。
        """

        def _extract_svg_fragment(svg_code: str) -> str:
            """
            从模型返回的字符串中，提取出干净的 <svg>...</svg> 片段。

            会做几件事：
            1. 去掉首尾空白
            2. 去掉可能的 ``` 或 ```svg 代码块包裹
            3. 只保留从第一个 <svg 开始到最后一个 </svg> 结束的部分
            """
            if not svg_code:
                return ""

            text = svg_code.strip()

            # 处理 ```svg ... ``` 或 ``` ... ``` 代码块
            if text.startswith("```"):
                lines = [line for line in text.splitlines() if line.strip("`").strip()]
                # 找到第一行包含 <svg 的
                for i, line in enumerate(lines):
                    if "<svg" in line or "<SVG" in line or "<Svg" in line:
                        text = "\n".join(lines[i:])
                        break

            # 定位 <svg ...> 到 </svg> 的区间
            start = text.find("<svg")
            end = text.rfind("</svg>")
            if start == -1 or end == -1:
                # 找不到明确片段时，直接返回原始文本，交给 XML 解析报错
                return text

            end += len("</svg>")
            return text[start:end].strip()

        def validate_svg(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """
            technical_route_desc_generator 的 SVG 验证器。

            使用解析后的结果 parsed_result["svg_code"] 做 XML 级合法性检查，
            与 base_agent._run_validators(content, parsed_result) 的调用约定保持一致。
            """
            import xml.etree.ElementTree as ET

            svg_code = None
            if isinstance(parsed_result, dict):
                svg_code = parsed_result.get("svg_code")

            if not svg_code:
                return False, "缺少 svg_code 字段或内容为空"

            fragment = _extract_svg_fragment(svg_code)
            if "<svg" not in fragment and "<SVG" not in fragment:
                return False, "返回内容中未检测到 <svg> 根标签"

            try:
                # XML 解析只关心 well‑formed，不关心命名空间等
                ET.fromstring(fragment)
            except Exception as e:
                log.warning(f"technical_route_desc_generator.validate_svg: SVG XML 解析失败: {e}")
                return False, f"SVG 不是合法 XML: {e}"

            return True, "SVG 验证通过"

        return [validate_svg]

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推理结果写回 MainState，可按需重写

        期望 LLM 返回形如：
        {"svg_code": "<svg ...>...</svg>"}
        """
        # 解析 LLM 返回的 JSON 结果，取出 svg_code
        svg_code = None
        if isinstance(result, dict):
            svg_code = result.get("svg_code", None)
        state.figure_tec_svg_content = svg_code
        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def technical_route_desc_generator(
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
    """technical_route_desc_generator 的异步入口
    
    Args:
        state: 主状态对象
        model_name: 模型名称，如 "gpt-4"
        tool_manager: 工具管理器实例
        temperature: 采样温度，控制随机性 (0.0-1.0)
        max_tokens: 最大生成token数
        tool_mode: 工具调用模式 ("auto", "none", "required")
        react_mode: 是否启用ReAct推理模式
        react_max_retries: ReAct模式下最大重试次数
        parser_type: 解析器类型 ("json", "xml", "text")，这个允许你在提示词中定义LLM不同的返回，xml还是json，还是直出；
        parser_config: 解析器配置字典（如XML的root_tag）
        use_vlm: 是否使用视觉语言模型，使用了视觉模型，其余的参数失效；
        vlm_config: VLM配置字典
        use_agent: 是否使用agent模式
        **kwargs: 其他传递给execute的参数
        
    Returns:
        更新后的MainState对象
    """
    agent = TechnicalRouteDescGenerator(
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


def create_technical_route_desc_generator(
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
) -> TechnicalRouteDescGenerator:
    return TechnicalRouteDescGenerator.create(
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
