"""
TopicWriter agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-27 02:40:27
生成位置: dataflow_agent/agentroles/common_agents/topic_writer_agent.py

本文件由 `dfa create --agent_name topic_writer` 自动生成。
1. 填写 prompt-template 名称
2. 根据需要完成 get_task_prompt_params / update_state_result
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
@register("topic_writer")
class TopicWriter(BaseAgent):
    """TODO: 描述 topic_writer 的职责"""

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "topic_writer"

    @property
    def system_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "system_prompt_for_topic_writer"

    @property
    def task_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "task_prompt_for_topic_writer"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数"""
        target_pages = getattr(self.state, "target_pages", 60)
        
        # 动态计算目标字符数
        is_en = self._is_english_text(self.state.current_text) if self.state.current_text else False
        chars_per_page = 3000 if is_en else 800
        target_chars = target_pages * chars_per_page
        
        return {
            'text_content': self.state.text_content,
            'generation_round': pre_tool_results.get('generation_round', 0),
            'language': getattr(self.state.request, "language", "中文"),
            'target_pages': target_pages,
            'target_chars': target_chars,
        }

    def _is_english_text(self, text: str) -> bool:
        """判断文本是否为英文"""
        if not text:
            return False
        sample = text[:5000]
        ascii_count = sum(1 for c in sample if ord(c) < 128)
        return (ascii_count / len(sample)) > 0.8

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """若调用方未显式传入，返回默认前置工具结果"""
        return {}

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将生成的文本写回 state.text_content"""
        state.text_content = result['text']
        super().update_state_result(state, result, pre_tool_results)
# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def topic_writer(
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
    """topic_writer 的异步入口
    
    Args:
        state: 主状态对象
        model_name: 模型名称，如 "gpt-4"
        tool_manager: 工具管理器实例
        temperature: 采样温度，控制随机性 (0.0-1.0)
        max_tokens: 最大生成token数
        tool_mode: 工具调用模式 ("auto", "none", "required")
        react_mode: 是否启用ReAct推理模式
        react_max_retries: ReAct模式下最大重试次数
        parser_type: 解析器类型 ("json", "xml", "text")，这个允许你在提示词中定义LLM不同的返回，xml还是json，还是直出；
        parser_config: 解析器配置字典（如XML的root_tag）
        use_vlm: 是否使用视觉语言模型，使用了视觉模型，其余的参数失效；
        vlm_config: VLM配置字典
        use_agent: 是否使用agent模式
        **kwargs: 其他传递给execute的参数
        
    Returns:
        更新后的MainState对象
    """
    agent = TopicWriter(
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


def create_topic_writer(
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
) -> TopicWriter:
    return TopicWriter.create(
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
