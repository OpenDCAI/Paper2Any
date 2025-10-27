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
# from 

log = get_logger(__name__)

from .base_agent import BaseAgent


class IconEditor(BaseAgent):
    """图标编辑器 - 根据图片和提示词进行二次编辑"""
    
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "icon_editor"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_icon_editing"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_icon_editing"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """图标编辑器特有的提示词参数"""
        return {
            'original_image': pre_tool_results.get('image', ''),
            'edit_instructions': pre_tool_results.get('instructions', ''),
            'edit_strength': pre_tool_results.get('strength', 0.8),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """图标编辑器的默认前置工具结果"""
        return {
            'image': '',
            'instructions': '',
            'strength': 0.8
        }
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """自定义状态更新 - 保存编辑后的图片"""
        # 假设 result 包含编辑后的图片URL或base64数据
        if isinstance(result, dict):
            state.icon_image = result.get('edited_image_url', result.get('edited_image_data', result))
            # 可选：保存编辑历史
            if not hasattr(state, 'icon_edit_history'):
                state.icon_edit_history = []
            state.icon_edit_history.append({
                'original': pre_tool_results.get('image', ''),
                'instructions': pre_tool_results.get('instructions', ''),
                'result': state.icon_image
            })
        else:
            state.icon_image = result
        super().update_state_result(state, result, pre_tool_results)


async def icon_editing(
    state: DFState, 
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """编辑图标的入口函数"""
    editor = IconEditor(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await editor.execute(state, use_agent=use_agent, **kwargs)


def create_icon_editor(tool_manager: Optional[ToolManager] = None, **kwargs) -> IconEditor:
    """创建图标编辑器实例"""
    return IconEditor(tool_manager=tool_manager, **kwargs)