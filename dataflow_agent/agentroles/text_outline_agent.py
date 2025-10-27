# dataflow/dataflowagent/agentroles/text_outline_generator.py
# -*- coding: utf-8 -*-
"""
TextOutlineGenerator —— 从输入文本生成结构化大纲的 Agent

前置工具（可选，按需配置 ToolManager）：
    - input_text : 需要结构化的大段文本

后置工具（可选，例如让 LLM 调工具自动修改大纲）：
    - outline_tool : 自定义 Tool，将 LLM 给出的结构化大纲应用到文件或其它地方

本 Agent 仅负责：
    1. 读取 "input_text"；
    2. 让 LLM 给出『文本结构化大纲』的 JSON 结果，并更新 `state.outline`。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.state import DFState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()


class TextOutlineGenerator(BaseAgent):
    @property
    def role_name(self) -> str:
        return "text_outline_generator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_text_outline_generation"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_text_outline_generation"

    # -------------------- Prompt 参数 -------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 中的占位符：
            {{ input_text }}   – 需要结构化的文本
        """
        return {
            "input_text": pre_tool_results.get("input_text", ""),
            "num_of_blocks": pre_tool_results.get("num_of_blocks", "")
        }

    # -------------------- 前置工具默认值 -----------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "input_text": "",
            "num_of_blocks": "",
        }

    # -------------------- 结果写回 DFState --------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        约定 LLM 输出格式：
            outline: dict      – 结构化大纲
        """
        state.text_outline = result
        super().update_state_result(state, result, pre_tool_results)


# ------------------------------------------------------------------
#                    对外统一调用入口（函数封装）
# ------------------------------------------------------------------
async def text_outline_generate(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    """
    单步调用：执行 TextOutlineGenerator 并将结果写回 DFState
    """
    outline_generator = TextOutlineGenerator(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await outline_generator.execute(state, use_agent=use_agent, **kwargs)


def create_text_outline_generator(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> TextOutlineGenerator:
    return TextOutlineGenerator(tool_manager=tool_manager, **kwargs)
