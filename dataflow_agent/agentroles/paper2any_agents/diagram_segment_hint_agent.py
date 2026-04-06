"""
Diagram segment hint agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use a VLM pass to propose extra SAM3 image prompts for diagram segmentation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register


@register("diagram_segment_hint_agent")
class DiagramSegmentHintAgent(BaseAgent):
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "diagram_segment_hint_agent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_diagram_segment_hint_agent"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_diagram_segment_hint_agent"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        temp_data = getattr(self.state, "temp_data", {}) or {}
        return {
            "ocr_text_lines_json": temp_data.get("diagram_segment_hint_text_lines_json", "[]"),
            "base_image_prompts_json": temp_data.get("diagram_segment_hint_base_prompts_json", "[]"),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        if not hasattr(state, "temp_data") or state.temp_data is None:
            state.temp_data = {}
        state.temp_data["sam3_segment_hints_raw"] = result
        super().update_state_result(state, result, pre_tool_results)


async def diagram_segment_hint_agent(
    state: MainState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
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
    agent = DiagramSegmentHintAgent(
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


def create_diagram_segment_hint_agent(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_vlm: bool = False,
    vlm_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> DiagramSegmentHintAgent:
    return DiagramSegmentHintAgent.create(
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
