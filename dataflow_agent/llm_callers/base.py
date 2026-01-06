from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
import base64
import os

log = get_logger(__name__)


class BaseLLMCaller(ABC):
    """LLM调用器基类"""
    
    def __init__(self, 
                 state: MainState,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 tool_mode: str = "auto",
                 tool_manager: Optional[ToolManager] = None):
        self.state = state
        self.model_name = model_name or state.request.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.tool_manager = tool_manager
    
    @abstractmethod
    async def call(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        """调用LLM"""
        pass