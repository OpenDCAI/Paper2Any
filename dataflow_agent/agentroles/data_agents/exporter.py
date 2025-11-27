from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DFState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

from dataflow_agent.agentroles.cores.base_agent import BaseAgent

class DataExporter(BaseAgent):
    """数据导出器 - 继承自BaseAgent"""

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "exporter"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_nodes_export"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_nodes_export"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """数据导出器特有的提示词参数"""
        return {
            'nodes_info': pre_tool_results.get('nodes_info', '[]'),
            'sample': pre_tool_results.get('sample','')
            # 'export_fields': pre_tool_results.get('export_fields', '[]'),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """数据导出器的默认前置工具结果"""
        return {
            'nodes_info': '',
        }
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 保持向后兼容"""
        state.nodes_info = result
        super().update_state_result(state, result, pre_tool_results)

async def data_export(
    state: DFState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    exporter = DataExporter(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await exporter.execute(state, use_agent=use_agent, **kwargs)

def create_exporter(tool_manager: Optional[ToolManager] = None, **kwargs) -> DataExporter:
    return DataExporter(tool_manager=tool_manager, **kwargs)