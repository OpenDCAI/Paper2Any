from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from .base import BaseLLMCaller   # 如果你建了 base.py
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class TextLLMCaller(BaseLLMCaller):
    def __init__(self, state, *, model_name, temperature,
                 max_tokens, tool_mode, tool_manager):
        self.state          = state
        self.model_name     = model_name
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.tool_mode      = tool_mode
        self.tool_manager   = tool_manager

    async def call(self, messages: List[BaseMessage], *,
                   bind_post_tools: bool = False):
        actual_model = self.model_name or self.state.request.model
        log.info(f"[TextLLMCaller] use model: {actual_model}")

        llm = ChatOpenAI(
            openai_api_base=self.state.request.chat_api_url,
            openai_api_key = self.state.request.api_key,
            model_name     = actual_model,
            temperature    = self.temperature,
            # max_tokens     = self.max_tokens,
        )

        # 绑定 Tool（可选）
        if bind_post_tools and self.tool_manager:
            tools = self.tool_manager.get_post_tools(self.state.current_role)
            if tools:
                llm = llm.bind_tools(tools, tool_choice=self.tool_mode)

        return await llm.ainvoke(messages)