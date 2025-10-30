from typing import List
from langchain_core.messages import AIMessage, BaseMessage
import aiohttp

from .base import BaseLLMCaller
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class ImageLLMCaller(BaseLLMCaller):
    """调用 Gemini / DALLE3 等，返回 base64 图片字符串"""

    def __init__(self, state, *, model_name, temperature, **_):
        self.api_url     = state.request.chat_api_url
        self.api_key     = state.request.api_key
        self.model       = model_name or "gemini-2.5-flash-image-preview"
        self.temperature = temperature

    async def call(self, messages: List[BaseMessage], **_):
        prompt = messages[-1].content
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_modalities": ["Image"],
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{self.api_url}/chat/completions",
                                 headers=headers, json=payload, timeout=120) as resp:
                resp.raise_for_status()
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        log.info("[ImageLLMCaller] get image content (base64 length=%s)", len(content))
        return AIMessage(content=content)