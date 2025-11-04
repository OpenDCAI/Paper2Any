from typing import List
from langchain_core.messages import BaseMessage,AIMessage
from langchain_openai import ChatOpenAI

from .base import BaseLLMCaller
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class TextLLMCaller(BaseLLMCaller):
    """文本LLM调用器 - 原有实现"""
    
    async def call(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        log.info(f"TextLLM调用，模型: {self.model_name}")
        
        llm = ChatOpenAI(
            openai_api_base=self.state.request.chat_api_url,
            openai_api_key=self.state.request.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
        )
        
        # 绑定工具（如果需要）
        if bind_post_tools and self.tool_manager:
            from langchain_core.tools import Tool
            tools = self.tool_manager.get_post_tools("current_role")  # 需要传入角色名
            if tools:
                llm = llm.bind_tools(tools, tool_choice=self.tool_mode)
                log.info(f"为LLM绑定了 {len(tools)} 个工具")
        
        response = await llm.ainvoke(messages)
        return response