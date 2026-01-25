"""
technical_route_colorize_svg_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于黑白 SVG + 色卡上色生成彩色技术路线图 SVG
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)


@register("technical_route_colorize_svg")
class TechnicalRouteColorizeSvgAgent(BaseAgent):
    """对黑白 SVG 进行配色的 Agent"""

    @property
    def role_name(self) -> str:
        return "technical_route_colorize_svg"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_technical_route_colorize_svg"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_technical_route_colorize_svg"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "bw_svg_code": pre_tool_results.get("bw_svg_code", ""),
            "palette_json": pre_tool_results.get("palette_json", ""),
            "validation_feedback": pre_tool_results.get("validation_feedback", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "bw_svg_code": "",
            "palette_json": "",
            "validation_feedback": "",
        }

    def update_state_result(
        self,
        state: Paper2FigureState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        svg_code = None
        if isinstance(result, dict):
            svg_code = result.get("svg_code")
        state.figure_tec_svg_color_content = svg_code or ""
        super().update_state_result(state, result, pre_tool_results)


def create_technical_route_colorize_svg_agent(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    parser_type: str = "json",
    **kwargs,
) -> TechnicalRouteColorizeSvgAgent:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()

    return TechnicalRouteColorizeSvgAgent(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        parser_type=parser_type,
        **kwargs,
    )
