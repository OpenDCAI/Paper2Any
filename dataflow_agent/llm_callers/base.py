from abc import ABC, abstractmethod
from typing import List
from langchain_core.messages import BaseMessage

class BaseLLMCaller(ABC):
    """所有 LLMCaller 的共同接口"""

    @abstractmethod
    async def call(
        self,
        messages: List[BaseMessage],
        *,
        bind_post_tools: bool = False,
    ) -> BaseMessage: ...