from __future__ import annotations
from typing import Any, Dict, Optional
from dataflow_agent.state import DFState, Paper2FigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent

log = get_logger(__name__)

@register("paper_idea_extractor")
class PaperIdeaExtractor(BaseAgent):
    @property
    def role_name(self) -> str:
        return "paper_idea_extractor"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_paper_idea_extractor"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_paper_idea_extractor"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        paper_content = pre_tool_results.get("paper_content")
        return {
            "paper_content": paper_content,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "paper_content": "",
        }

    def update_state_result(self, state: Paper2FigureState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):

        state.paper_idea = result.get("paper_idea", "") if result.get("paper_idea", "") != "" else result
        
        return super().update_state_result(state, result, pre_tool_results)


# Function to extract paper ideas
async def paper_idea_extractor(
    state: Paper2FigureState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
) -> Paper2FigureState:
    inst = create_paper_idea_extractor(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_paper_idea_extractor(tool_manager: Optional[ToolManager] = None, **kwargs) -> PaperIdeaExtractor:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return PaperIdeaExtractor(tool_manager=tool_manager, **kwargs)
