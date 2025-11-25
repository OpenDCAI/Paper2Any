from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DFState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

from .base_agent import BaseAgent


class TargetParser(BaseAgent):
    """目标意图理解 - 将用户的 target 拆解为多个算子描述"""
    
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "target_parser"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_target_parsing"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_target_parsing"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """目标解析器特有的提示词参数"""
        return {
            'target': pre_tool_results.get('target', ''),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """目标解析器的默认前置工具结果"""
        return {
            'target': ''
        }
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 将算子描述列表保存到 temp_data"""
        operator_descriptions = result.get('operator_descriptions', [])
        state.temp_data['operator_descriptions'] = operator_descriptions
        log.info(f"[TargetParser] 拆解出 {len(operator_descriptions)} 个算子描述")
        super().update_state_result(state, result, pre_tool_results)

async def target_parsing(
    state: DFState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    parser = TargetParser(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await parser.execute(state, use_agent=use_agent, **kwargs)

def create_target_parser(tool_manager: Optional[ToolManager] = None, **kwargs) -> TargetParser:
    return TargetParser(tool_manager=tool_manager, **kwargs)
