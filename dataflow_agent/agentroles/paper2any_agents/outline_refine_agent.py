"""
OutlineRefineAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Refines an existing PPT outline based on user feedback.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)


@register("outline_refine_agent")
class OutlineRefineAgent(BaseAgent):
    """Refine existing outline content while keeping page order and count."""

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:  # noqa: D401
        return "outline_refine_agent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_outline_refine_agent"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_outline_refine_agent"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pagecontent": pre_tool_results.get("pagecontent", "[]"),
            "outline_feedback": pre_tool_results.get("outline_feedback", ""),
            "minueru_output": pre_tool_results.get("minueru_output", ""),
            "text_content": pre_tool_results.get("text_content", ""),
            "page_count": self.state.request.page_count,
            "language": self.state.request.language,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        original = pre_tool_results.get("pagecontent_raw")
        if not isinstance(original, list):
            original = getattr(state, "pagecontent", []) or []

        if not isinstance(result, list) or len(result) != len(original):
            log.warning("[outline_refine_agent] Invalid result, fallback to original pagecontent.")
            state.pagecontent = original
            super().update_state_result(state, original, pre_tool_results)
            return

        merged_pages = []
        for idx, item in enumerate(result):
            base = original[idx] if idx < len(original) and isinstance(original[idx], dict) else {}
            merged = dict(base)

            if isinstance(item, dict):
                title = item.get("title")
                if isinstance(title, str):
                    merged["title"] = title

                layout = item.get("layout_description")
                if isinstance(layout, str):
                    merged["layout_description"] = layout

                key_points = item.get("key_points")
                if isinstance(key_points, list):
                    merged["key_points"] = key_points

                asset_ref = item.get("asset_ref")
                if asset_ref:
                    merged["asset_ref"] = asset_ref

            merged_pages.append(merged)

        state.pagecontent = merged_pages
        log.info(f"[outline_refine_agent] refined {len(merged_pages)} pages")
        super().update_state_result(state, merged_pages, pre_tool_results)


async def outline_refine_agent(
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
    """Async entry for outline_refine_agent."""
    agent = OutlineRefineAgent(
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


def create_outline_refine_agent(
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
) -> OutlineRefineAgent:
    return OutlineRefineAgent.create(
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
