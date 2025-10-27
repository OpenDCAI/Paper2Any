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
from dataflow_agent.workflow.registry import register

log = get_logger(__name__)

from .base_agent import BaseAgent

@register("icon_generator")
class IconGenerator(BaseAgent):
    """图标生成器 - 根据text2img prompt生成图标图片"""
    
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "icon_generator"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_icon_generation"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_icon_generation"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """图标生成器特有的提示词参数"""
        return {
            'text2img_prompt': pre_tool_results.get('prompt', ''),
            'image_size': pre_tool_results.get('size', '512x512'),
            'num_images': pre_tool_results.get('num_images', 1),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """图标生成器的默认前置工具结果"""
        return {
            'prompt': '',
            'size': '512x512',
            'num_images': 1
        }
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 保存生成的图片"""
        # 假设 result 包含图片URL或base64数据
        if isinstance(result, dict):
            state.icon_image = result.get('image_url', result.get('image_data', result))
        else:
            state.icon_image = result
        super().update_state_result(state, result, pre_tool_results)


async def icon_generation(
    state: DFState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """生成图标的入口函数"""
    generator = IconGenerator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await generator.execute(state, use_agent=use_agent, **kwargs)


def create_icon_generator(tool_manager: Optional[ToolManager] = None, **kwargs) -> IconGenerator:
    """创建图标生成器实例"""
    return IconGenerator(tool_manager=tool_manager, **kwargs)