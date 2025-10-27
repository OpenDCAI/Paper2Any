from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import MainState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.workflow.registry import register

log = get_logger(__name__)

from .base_agent import BaseAgent

@register("icon_prompt_generator")
class IconPromptGenerator(BaseAgent):
    """图标提示词生成器 - 根据用户关键词生成text2img prompt"""
    
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "icon_prompt_generator"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_icon_prompt_generation"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_icon_prompt_generation"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """图标提示词生成器特有的提示词参数"""
        return {
            'user_keywords': pre_tool_results.get('keywords', ''),
            'style_preferences': pre_tool_results.get('style', 'kartoon, minimalist'),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """图标提示词生成器的默认前置工具结果"""
        return {
            'keywords': '',
            'style': 'modern, minimalist'
        }
    
    def update_state_result(self, state: MainState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 保存生成的prompt"""
        state.icon_prompt = result.get('prompt', result) if isinstance(result, dict) else result
        super().update_state_result(state, result, pre_tool_results)


async def icon_prompt_generation(
    state: MainState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> MainState:
    """生成图标提示词的入口函数"""
    generator = IconPromptGenerator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await generator.execute(state, use_agent=use_agent, **kwargs)


def create_icon_prompt_generator(tool_manager: Optional[ToolManager] = None, **kwargs) -> IconPromptGenerator:
    """创建图标提示词生成器实例"""
    return IconPromptGenerator(tool_manager=tool_manager, **kwargs)