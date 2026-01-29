"""
DiagramPlanner agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分析内容并规划图表结构

用于 Paper2Drawio 工作流
"""

from __future__ import annotations

from typing import Any, Dict

from dataflow_agent.state import Paper2DrawioState
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent

log = get_logger(__name__)


@register("diagram_planner")
class DiagramPlanner(BaseAgent):
    """规划图表结构的 Agent"""

    @property
    def role_name(self) -> str:
        return "diagram_planner"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_diagram_planner"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_diagram_planner"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """从 pre_tool_results 中获取 prompt 参数"""
        paper_content = pre_tool_results.get("paper_content", "")
        text_content = pre_tool_results.get("text_content", "")
        diagram_type = pre_tool_results.get("diagram_type", "auto")
        language = pre_tool_results.get("language", "")

        return {
            "paper_content": paper_content,
            "text_content": text_content,
            "diagram_type": diagram_type,
            "language": language,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认的 pre_tool_results"""
        return {
            "paper_content": "",
            "text_content": "",
            "diagram_type": "auto",
            "language": "",
        }

    def update_state_result(
        self,
        state: Paper2DrawioState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将规划结果写入 state"""
        try:
            if isinstance(result, dict):
                plan = result.get("diagram_plan", "") or result.get("plan", "")
            elif isinstance(result, str):
                plan = result
            else:
                plan = str(result)

            if plan:
                state.diagram_plan = plan
                log.info(f"[DiagramPlanner] 规划完成，长度: {len(plan)}")
            else:
                log.warning("[DiagramPlanner] 未生成有效的规划")

        except Exception as e:
            log.error(f"[DiagramPlanner] 更新状态失败: {e}")
