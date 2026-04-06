"""
P2vRefineSubtitleAndCursor agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2026-02-02 15:32:55
生成位置: dataflow_agent/agentroles/common_agents/p2v_refine_subtitle_and_cursor_agent.py

本文件由 `dfa create --agent_name p2v_refine_subtitle_and_cursor` 自动生成。
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
import json
import re

log = get_logger(__name__)

def parse_subtitle_and_cursor_result(result: Dict[str, Any]) -> Optional[str]:
    """从 LLM 返回中提取 refine subtitle/cursor 文本，兼容历史 key。"""
    accepted_keys = ("refine_subtitle_and_cursor", "subtitle_and_cursor")
    if not result:
        return None
    for key in accepted_keys:
        value = result.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    raw = result.get("raw")
    if not isinstance(raw, str):
        return None
    raw_text = raw.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            for key in accepted_keys:
                value = parsed.get(key)
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
    except json.JSONDecodeError:
        pass
    match = re.search(
        r'"(?:refine_subtitle_and_cursor|subtitle_and_cursor)"\s*:\s*"(.*?)"\s*}',
        raw_text,
        re.DOTALL,
    )
    if match:
        value = match.group(1).strip()
        if value:
            return value
    return None


# ----------------------------------------------------------------------
# Agent Definition
# ----------------------------------------------------------------------
@register("p2v_refine_subtitle_and_cursor")
class P2vRefineSubtitleAndCursor(BaseAgent):
    """描述 p2v_refine_subtitle_and_cursor 的职责
        对subtitle_and_cursor文件中的内容进行细化
        在subtitle的基础上增加cursor 的信息
    """
    

    # ---------- 工厂 ----------
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:  # noqa: D401
        return "p2v_refine_subtitle_and_cursor"

    @property
    def system_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "system_prompt_for_p2v_refine_subtitle_and_cursor"

    @property
    def task_prompt_template_name(self) -> str:
        # TODO: 修改为真实的模板 id
        return "task_prompt_for_p2v_refine_subtitle_and_cursor"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """根据前置工具结果构造 prompt 参数
        提示词中的占位符：
        """
        # TODO: 按需补充
        return {
            "sentence": pre_tool_results.get("tmp_sentence", ""),
            "language": pre_tool_results.get("video_language", "English"),
        }

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
        """解析 LLM 返回的 refine_subtitle_and_cursor，追加到 state；格式错误时不 append，由上层重试。"""
        subtitle_and_cursor_info = parse_subtitle_and_cursor_result(result)
        if subtitle_and_cursor_info is not None:
            log.info("获取了单张 slide 的 Refine Subtitle and Cursor 信息: %s", subtitle_and_cursor_info[:80] if len(subtitle_and_cursor_info) > 80 else subtitle_and_cursor_info)
            state.subtitle_and_cursor.append(subtitle_and_cursor_info)
        else:
            log.warning("LLM 返回格式不符合 {\"refine_subtitle_and_cursor\": \"...\"}，未写入 state，上层可重试")
        super().update_state_result(state, result, pre_tool_results)


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
async def p2v_refine_subtitle_and_cursor(
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
    """p2v_refine_subtitle_and_cursor 的异步入口
    
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
    agent = P2vRefineSubtitleAndCursor(
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


def create_p2v_refine_subtitle_and_cursor(
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
) -> P2vRefineSubtitleAndCursor:
    return P2vRefineSubtitleAndCursor.create(
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
