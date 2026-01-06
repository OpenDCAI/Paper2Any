"""
DeepResearchAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
用于根据 Topic 生成详细的长篇研究报告/大纲内容。
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
@register("deep_research_agent")
class DeepResearchAgent(BaseAgent):
    """
    DeepResearchAgent: 接收 Topic，输出长篇研究报告。
    """

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:
        return "deep_research_agent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_deep_research_agent"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_deep_research_agent"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text_content": self.state.text_content or "",
            "language": getattr(self.state.request, "language", "zh"),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        Deep Research 的结果通常是长文本。
        如果是 parser_type="text"，result 就是 str。
        如果是 parser_type="json"，result 是 dict。
        这里假设我们使用 text parser，或者 json 中有个 'content' 字段。
        """
        # 假设 LLM 直接返回文本内容，或者 JSON 中包含 content
        content = ""
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict):
            # 尝试获取常见字段
            content = result.get("content") or result.get("report") or result.get("research_result") or str(result)
        
        # 将生成的长文本写回 state.text_content，供后续 outline_agent 使用
        if content:
            state.text_content = content
            log.info(f"[deep_research_agent]: Generated content length: {len(content)}")
        
        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
def create_deep_research_agent(
    tool_manager: Optional[ToolManager] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    parser_type: str = "text", # 默认为 text，直接输出长文
    **kwargs,
) -> DeepResearchAgent:
    return DeepResearchAgent.create(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        parser_type=parser_type,
        **kwargs,
    )
