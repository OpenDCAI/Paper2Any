"""
TechRouteReferenceAnalyzer agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用 VLM 分析技术路线图参考图，提取布局、风格、配色等信息
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)


@register("tech_route_reference_analyzer")
class TechRouteReferenceAnalyzer(BaseAgent):
    """使用 VLM 分析技术路线图参考图的 Agent"""

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "tech_route_reference_analyzer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_tech_route_reference_analyzer"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_tech_route_reference_analyzer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数"""
        return {
            "reference_image_path": pre_tool_results.get("reference_image_path", ""),
            "lang": self.state.request.language if hasattr(self.state, "request") else "zh",
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"reference_image_path": ""}

    def get_react_validators(self) -> List:
        """返回 ReAct 模式下使用的验证器列表"""

        def validate_svg_code(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, str]:
            """验证 SVG 代码的完整性"""
            if not isinstance(parsed_result, dict):
                return False, "返回结果不是有效的 JSON 对象"

            svg_code = parsed_result.get("svg_code")
            if not svg_code:
                return False, "缺少 svg_code 字段"

            if "<svg" not in svg_code.lower():
                return False, "svg_code 中未检测到 <svg> 标签"

            return True, "验证通过"

        return [validate_svg_code]

    def update_state_result(
        self,
        state: Paper2FigureState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将 VLM 生成的 SVG 代码写回 State"""
        if isinstance(result, dict):
            if not hasattr(state, "temp_data"):
                state.temp_data = {}
            svg_code = result.get("svg_code", "")
            state.temp_data["reference_svg_code"] = svg_code
            log.info(f"[TechRouteReferenceAnalyzer] 参考图 SVG 生成完成，长度: {len(svg_code)}")
        super().update_state_result(state, result, pre_tool_results)


async def tech_route_reference_analyzer(
    state: Paper2FigureState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    use_vlm: bool = True,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Paper2FigureState:
    """tech_route_reference_analyzer 的异步入口"""
    agent = TechRouteReferenceAnalyzer(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
    )
    return await agent.execute(state, **kwargs)


def create_tech_route_reference_analyzer(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    use_vlm: bool = True,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TechRouteReferenceAnalyzer:
    """创建 TechRouteReferenceAnalyzer 实例"""
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()

    return TechRouteReferenceAnalyzer(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        use_vlm=use_vlm,
        vlm_config=vlm_config,
        **kwargs,
    )
