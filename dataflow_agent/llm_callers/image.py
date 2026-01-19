import base64
from typing import Any, Dict, List, Optional
from dataflow_agent.state import MainState
from langchain_core.messages import AIMessage, BaseMessage

from dataflow_agent.llm_callers.base import BaseLLMCaller
from dataflow_agent.logger import get_logger

# Import new tools
from dataflow_agent.toolkits.multimodaltool.req_ocr import call_ocr_async
from dataflow_agent.toolkits.multimodaltool.req_understanding import call_image_understanding_async
from dataflow_agent.toolkits.multimodaltool.req_videos import call_video_understanding_async
from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async

log = get_logger(__name__)

class VisionLLMCaller(BaseLLMCaller):
    """
    视觉LLM调用器 - 统一入口
    支持模式:
    1. understanding       (通用图像理解)
    2. generation / edit   (图像生成/编辑)
    3. video_understanding (视频理解)
    4. ocr                 (OCR专用)
    """
    
    def __init__(self, 
                 state: MainState,
                 vlm_config: Dict[str, Any],
                 **kwargs):
        """
        Args:
            vlm_config: VLM配置，包含：
                - mode: "generation" | "edit" | "understanding" | "video_understanding" | "ocr"
                - input_image: 输入图像路径
                - input_video: 输入视频路径 (video_understanding模式)
                - output_image: 输出图像保存路径 (generation/edit模式)
                - response_format: "image" | "text" (默认根据mode自动判断)
        """
        super().__init__(state, **kwargs)
        self.vlm_config = vlm_config
        self.mode       = vlm_config.get("mode", "understanding")
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 4096)
    
    async def call(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        """调用VLM"""
        log.info(f"VisionLLM调用，模型: {self.model_name}, 模式: {self.mode}")
        
        # 1. 图像生成/编辑
        if self.mode in ["generation", "edit"]:
            return await self._call_image_output(messages)
            
        # 2. 视频理解
        elif self.mode == "video_understanding":
             return await self._call_video_understanding(messages)
             
        # 3. OCR (显式模式 或 隐式检测)
        elif self.mode == "ocr" or ("qwen-vl-ocr" in self.model_name.lower() and "apiyi" in self.state.request.chat_api_url):
             return await self._call_ocr(messages)
             
        # 4. 通用图像理解 (默认)
        else:
            return await self._call_image_understanding(messages)
        
    async def _call_ocr(self, messages: List[BaseMessage]) -> AIMessage:
        """调用 OCR 模块"""
        # 转换 LangChain 消息为 list[dict]
        msgs = self._convert_messages(messages)
        image_path = self.vlm_config.get("input_image")
        
        content = await call_ocr_async(
            model=self.model_name,
            messages=msgs,
            api_url=self.state.request.chat_api_url,
            api_key=self.state.request.api_key,
            image_path=image_path,
            max_tokens=self.max_tokens,
            temperature=0.01, # OCR usually needs low temp
            timeout=self.vlm_config.get("timeout", 120)
        )
        return AIMessage(content=content)

    async def _call_video_understanding(self, messages: List[BaseMessage]) -> AIMessage:
        """调用视频理解模块"""
        msgs = self._convert_messages(messages)
        # 支持 input_video 或 input_image (兼容性)
        video_path = self.vlm_config.get("input_video") or self.vlm_config.get("input_image")
        
        if not video_path:
            raise ValueError("video_understanding mode requires 'input_video' in vlm_config")

        content = await call_video_understanding_async(
            model=self.model_name,
            messages=msgs,
            api_url=self.state.request.chat_api_url,
            api_key=self.state.request.api_key,
            video_path=video_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.vlm_config.get("timeout", 300)
        )
        return AIMessage(content=content)

    async def _call_image_understanding(self, messages: List[BaseMessage]) -> AIMessage:
        """调用通用图像理解模块"""
        msgs = self._convert_messages(messages)
        image_path = self.vlm_config.get("input_image")
        
        content = await call_image_understanding_async(
            model=self.model_name,
            messages=msgs,
            api_url=self.state.request.chat_api_url,
            api_key=self.state.request.api_key,
            image_path=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.vlm_config.get("timeout", 120)
        )
        return AIMessage(content=content)
    
    async def _call_image_output(self, messages: List[BaseMessage]) -> AIMessage:
        """图像生成/编辑模式 - 输出图像"""
        # 提取prompt（最后一条用户消息）
        prompt = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                prompt = msg.content
                break
        
        # 调用图像生成函数
        save_path = self.vlm_config.get("output_image", "./generated_image.png")
        image_path = self.vlm_config.get("input_image") if self.mode == "edit" else None
        aspect_ratio = self.vlm_config.get("aspect_ratio", "16:9")
        
        b64 = await generate_or_edit_and_save_image_async(
            prompt=prompt,
            save_path=save_path,
            api_url=self.state.request.chat_api_url,
            api_key=self.state.request.api_key,
            model=self.model_name,
            image_path=image_path,
            use_edit=(self.mode == "edit"),
            timeout=self.vlm_config.get("timeout", 120),
            aspect_ratio = aspect_ratio 
        )
        
        content = f"图像已生成并保存至: {save_path}"
        return AIMessage(content=content, additional_kwargs={
            "image_path": save_path,
            "image_base64": b64,
        })

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Helper: Convert LangChain messages to dict format"""
        processed_messages = []
        for msg in messages:
            role = "user"
            if hasattr(msg, "type"):
                if msg.type == "human": role = "user"
                elif msg.type == "ai": role = "assistant"
                elif msg.type == "system": role = "system"
                elif msg.type == "tool": role = "tool"
            
            processed_messages.append({"role": role, "content": msg.content})
        return processed_messages

# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    import os
    import sys
    import asyncio
    from types import SimpleNamespace
    from pathlib import Path
    from langchain_core.messages import HumanMessage

    async def _quick_test(img_path: str):
        api_url = os.getenv("DF_API_URL")
        api_key = os.getenv("DF_API_KEY")
        if not api_url or not api_key:
            print("❌  请先设置环境变量 DF_API_URL / DF_API_KEY")
            sys.exit(1)

        img_path = Path(img_path).expanduser().resolve()
        if not img_path.exists():
            print(f"❌  图片不存在: {img_path}")
            sys.exit(1)

        request = SimpleNamespace(chat_api_url=api_url.rstrip("/"), api_key=api_key, model="gemini-2.5-flash-image-preview")
        state = SimpleNamespace(request=request)

        # 1. Test Understanding
        print("\n[TEST] Understanding Mode...")
        caller_und = VisionLLMCaller(
            state=state,
            vlm_config={"mode": "understanding", "input_image": str(img_path)}
        )
        res = await caller_und.call([HumanMessage(content="Describe this image in 10 words.")])
        print(f"Understanding Result: {res.content}")

        # 2. Test Generation (Requires prompt, ignores input image usually, but config needs to be valid)
        # Note: Generation writes to file, we skip if we don't want side effects or provide a test path
        
    if len(sys.argv) >= 2:
        asyncio.run(_quick_test(sys.argv[1]))
