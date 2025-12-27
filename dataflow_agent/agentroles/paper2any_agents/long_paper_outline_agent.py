"""
LongPaperOutlineAgent agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Description: 专门用于处理长文档分批生成 PPT 大纲的 Agent。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

log = get_logger(__name__)

# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("long_paper_outline_agent")
class LongPaperOutlineAgent(BaseAgent):
    """
    LongPaperOutlineAgent: 负责接收分批次的长文本，生成对应的 PPT 大纲页面。
    """

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:
        return "long_paper_outline_agent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_outline_agent"

    @property
    def task_prompt_template_name(self) -> str:
        if getattr(self.state, "is_first", False):
            return "task_prompt_for_long_paper_outline_agent_first"
        if getattr(self.state, "is_last", False):
            return "task_prompt_for_long_paper_outline_agent_last"
        return "task_prompt_for_long_paper_outline_agent_middle"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造 Prompt 参数。
        需要 Workflow 传入:
        - current_chunk: 当前批次的文本内容
        - batch_info: 批次信息 (index, total, etc.)
        """
        batch_info = pre_tool_results.get("batch_info", {})
        
        return {
            "current_chunk": self.state.current_chunk,
            "batch_index": batch_info.get("batch_index", 1),
            "total_batches": batch_info.get("total_batches", 1),
            "pages_to_generate": batch_info.get("pages_to_generate", 10),
            "is_first": batch_info.get("is_first", False),
            "is_last": batch_info.get("is_last", False),
            "page_count" : self.state.request.page_count,
            "language": self.state.request.language,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "current_chunk": "",
            "batch_info": {}
        }

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        将生成的结果（JSON List）写回 State。
        注意：在 Workflow 的 generate_outline_for_batch 中，
        会从返回的 State 中读取 pagecontent。
        """
        # 结果预期是一个 List[Dict] (页面列表)
        state.pagecontent = result
        log.info(f"[long_paper_outline_agent] 生成了 {len(result) if isinstance(result, list) else 0} 页内容")
        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def long_paper_outline_agent(
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
    agent = LongPaperOutlineAgent(
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


def create_long_paper_outline_agent(
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
) -> LongPaperOutlineAgent:
    return LongPaperOutlineAgent.create(
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
