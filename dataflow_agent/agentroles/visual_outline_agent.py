from typing import Any, Dict, Optional
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.state import DFState
from dataflow import get_logger

log = get_logger()


class VisualOutlineAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "visual_outline_generator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_visual_outline"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_visual_outline"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 中的占位符：
            {{ text_outline }}   – 需要生成视觉大纲的结构化文本
        """
        return {
            "text_outline": pre_tool_results.get("text_outline", ""),
            "color_palette": pre_tool_results.get('color_palette', ''),
            'illustration_style': pre_tool_results.get('illustration_style', '')
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "text_outline": {},
            "color_pallete": {},
            'illustraion_style': {}
        }

    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：包含视觉大纲信息的 JSON 数据
        """
        state.visual_outline = result
        super().update_state_result(state, result, pre_tool_results)


# ------------------------------------------------------------------ #
#                    对外统一调用入口（函数封装）                   #
# ------------------------------------------------------------------ #
async def generate_visual_outline(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[Any] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """
    单步调用：执行 VisualOutlineAgent 并将结果写回 DFState
    """
    agent = VisualOutlineAgent(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_visual_outline_agent(
    tool_manager: Optional[Any] = None,
    **kwargs,
) -> VisualOutlineAgent:
    return VisualOutlineAgent(tool_manager=tool_manager, **kwargs)
