from __future__ import annotations
from typing import Any, Dict, Optional
from dataflow_agent.state import DFState, Paper2FigureState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
log = get_logger(__name__)
from dataflow_agent.agentroles.cores.base_agent import BaseAgent

@register("figure_desc_generator")
class FigureDescGenerator(BaseAgent):
    @property
    def role_name(self) -> str:
        return "figure_desc_generator"

    @property
    def system_prompt_template_name(self) -> str:
        if getattr(self.state.request, "figure_complex", "") == "easy":
            return "system_prompt_for_figure_desc_generator"
        elif getattr(self.state.request, "figure_complex", "") == "hard":
            return  "system_prompt_for_figure_desc_generator_free"
        else:
            return  "system_prompt_for_figure_desc_generator_mid"

    @property
    def task_prompt_template_name(self) -> str:
        if getattr(self.state.request, "figure_complex", "") == "easy":
            return "task_prompt_for_figure_desc_generator"
        elif getattr(self.state.request, "figure_complex", "") == "hard":
            return  "task_prompt_for_figure_desc_generator_free"
        else:
            return "task_prompt_for_figure_desc_generator_mid"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        paper_idea = pre_tool_results.get("paper_idea")
        style = self.state.request.style
        figure_complex = getattr(self.state.request, "figure_complex", "mid")
        if not figure_complex or figure_complex == "":
            figure_complex = "easy"

        params = {
            "paper_idea": paper_idea,
            "style": style
        }

        # 根据复杂度注入风格配置
        if figure_complex in ["easy", "mid"]:
            try:
                from dataflow_agent.promptstemplates.resources.pt_technical_route_desc_generator_repo import FIGURE_STYLE_CONFIGS
                style_config = FIGURE_STYLE_CONFIGS.get(style, FIGURE_STYLE_CONFIGS.get("cartoon", {}))

                if figure_complex == "easy":
                    params.update(style_config)
                else:
                    params.update({
                        "style_desc": style_config.get("style_desc", ""),
                        "color_palette": style_config.get("color_palette", ""),
                        "rendering": style_config.get("rendering", ""),
                        "font": style_config.get("font", "")
                    })

            except Exception as e:
                log.error(f"[get_task_prompt_params] Failed to load style config: {e}")
        else:
            log.info(f"[get_task_prompt_params] Hard complexity, no style config injection")


        return params

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "paper_idea": "",
        }

    def update_state_result(self, state: Paper2FigureState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        try:
            figure_desc = result.get("figure_desc", "") if isinstance(result, dict) else ""
            if figure_desc:
                state.fig_desc = figure_desc
        except Exception:
            pass
        return super().update_state_result(state, result, pre_tool_results)


# Function to generate figure description
async def figure_desc_generator(
    state: Paper2FigureState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
) -> Paper2FigureState:
    inst = create_figure_desc_generator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)


def create_figure_desc_generator(tool_manager: Optional[ToolManager] = None, **kwargs) -> FigureDescGenerator:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return FigureDescGenerator(tool_manager=tool_manager, **kwargs)
