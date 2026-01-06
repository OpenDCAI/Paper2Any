import base64
from typing import Any, Dict, List
from dataflow_agent.state import MainState
from langchain_core.messages import AIMessage, BaseMessage
import aiohttp

from dataflow_agent.llm_callers.base import BaseLLMCaller
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class VisionLLMCaller(BaseLLMCaller):
    """视觉LLM调用器 - 支持图像输入/输出"""
    
    def __init__(self, 
                 state: MainState,
                 vlm_config: Dict[str, Any],
                 **kwargs):
        """
        Args:
            vlm_config: VLM配置，包含：
                - mode: "generation" | "edit" | "understanding"
                - input_image: 输入图像路径（edit/understanding模式）
                - output_image: 输出图像保存路径（generation/edit模式）
                - response_format: "image" | "text" (默认根据mode自动判断)
        """
        super().__init__(state, **kwargs)
        self.vlm_config = vlm_config
        self.mode       = vlm_config.get("mode", "understanding")
        self.temperature = kwargs.get("temperature", 0.1)
        # self.max_tokens = kwargs.get("max_tokens", 16384)
    
    async def call(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        """调用VLM"""
        log.info(f"VisionLLM调用，模型: {self.model_name}, 模式: {self.mode}")
        
        if self.mode in ["generation", "edit"]:
            return await self._call_image_output(messages)
        else:
            return await self._call_image_understanding(messages)
        
    async def _call_image_understanding(self, messages: List[BaseMessage]) -> AIMessage:
        """图像理解模式 - 输入图像，输出文本"""

        ROLE_MAP = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        processed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            lc_role = getattr(msg, "type", "human")     # human / ai / system / tool
            role     = ROLE_MAP.get(lc_role, "user")    # 兜底用 "user"

            processed_messages.append({
                "role": role,
                "content": msg.content                  # str 或 multimodal list 都行
            })

        # --------  如有 input_image，把图贴到最后一条 user 消息里 --------
        if "input_image" in self.vlm_config and processed_messages:
            b64, fmt = self._encode_image(self.vlm_config["input_image"])
            last_msg = processed_messages[-1]
            if last_msg["role"] == "user":
                last_msg["content"] = [
                    {"type": "text",  "text": last_msg["content"]},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                ]

        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response_data = await self._post_chat_completions(payload)
        # log.critical(f"VisionLLM理解模式响应: {response_data}")
        content = response_data["choices"][0]["message"]["content"]
        return AIMessage(content=content)
    
    # async def _call_image_understanding(self, messages: List[BaseMessage]) -> AIMessage:
    #     """图像理解模式 - 输入图像，输出文本"""
    #     # 这个还有bug！！！


    #     import httpx
        
    #     # 构建包含图像的消息
    #     processed_messages = []
    #     for msg in messages:
    #         if hasattr(msg, 'content') and isinstance(msg.content, str):
    #             processed_messages.append({
    #                 "role": msg.type if hasattr(msg, 'type') else "user",
    #                 "content": msg.content
    #             })
        
    #     # 如果配置了输入图像，添加到最后一条消息
    #     if "input_image" in self.vlm_config:
    #         b64, fmt = self._encode_image(self.vlm_config["input_image"])
            
    #         # 修改最后一条用户消息为多模态格式
    #         last_msg = processed_messages[-1]
    #         if last_msg["role"] == "user":
    #             last_msg["content"] = [
    #                 {"type": "text", "text": last_msg["content"]},
    #                 {"type": "image_url", 
    #                  "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
    #             ]
        
    #     # 调用API
    #     payload = {
    #         "model": self.model_name,
    #         "messages": processed_messages,
    #         "temperature": self.temperature,
    #         "max_tokens": self.max_tokens,
    #     }
        
    #     response_data = await self._post_chat_completions(payload)
    #     content = response_data["choices"][0]["message"]["content"]
        
    #     return AIMessage(content=content)
    
    async def _call_image_output(self, messages: List[BaseMessage]) -> AIMessage:
        """图像生成/编辑模式 - 输出图像"""
        from dataflow_agent.toolkits.imtool.req_img import generate_or_edit_and_save_image_async
        
        # 提取prompt（最后一条用户消息）
        prompt = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                prompt = msg.content
                break
        
        # 调用图像生成函数
        save_path = self.vlm_config.get("output_image", "./generated_image.png")
        image_path = self.vlm_config.get("input_image") if self.mode == "edit" else None
        aspect_ratio = self.vlm_config.get("aspect_ratio", "16:9")  # <-- 新增：默认 16:9
        
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
        
        # 返回图像路径作为内容
        content = f"图像已生成并保存至: {save_path}"
        
        return AIMessage(content=content, additional_kwargs={
            "image_path": save_path,
            "image_base64": b64,
        })
    
    def _encode_image(self, image_path: str) -> tuple:
        """编码图像为base64"""
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        
        ext = image_path.rsplit(".", 1)[-1].lower()
        if ext in {"jpg", "jpeg"}:
            fmt = "jpeg"
        elif ext == "png":
            fmt = "png"
        else:
            raise ValueError(f"不支持的图像格式: {ext}")
        
        return b64, fmt
    
    # async def _post_chat_completions(self, payload: dict) -> dict:
    #     """调用chat completions API"""
    #     import httpx
        
    #     url = f"{self.state.request.chat_api_url}/chat/completions".rstrip("/")

    #     log.critical(f"[_post_chat_completions]>调用URL: {url}")
    #     headers = {
    #         "Authorization": f"Bearer {self.state.request.api_key}",
    #         "Content-Type": "application/json",
    #     }
        
    #     timeout = self.vlm_config.get("timeout", 120)
    #     async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
    #         resp = await client.post(url, headers=headers, json=payload)
    #         resp.raise_for_status()
    #         return resp.json()

    async def _post_chat_completions(self, payload: dict) -> dict:
        """调用chat completions API"""
        import httpx
        
        # 构建URL
        base_url = self.state.request.chat_api_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        
        # 日志记录
        log.critical(f"[调用URL]: {url}")
        log.debug(f"[请求payload]: {payload}")
        
        headers = {
            "Authorization": f"Bearer {self.state.request.api_key}",
            "Content-Type": "application/json",
        }
        
        # 处理timeout参数
        timeout_value = self.vlm_config.get("timeout", 120)
        if isinstance(timeout_value, str):
            try:
                timeout = float(timeout_value)
            except ValueError:
                log.warning(f"timeout配置'{timeout_value}'无效，使用默认值120")
                timeout = 120
        else:
            timeout = float(timeout_value)
        
        log.info(f"[请求配置] timeout={timeout}s")
        
        try:
            # 创建客户端时显式配置，避免环境变量干扰
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                proxy=None,        # 禁用代理
                trust_env=False,     # 不读取环境变量
                follow_redirects=True
            ) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                log.debug(f"[响应]: {data}")
                return data
                
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            log.error(f"请求错误: {e}")
            raise
        except Exception as e:
            log.error(f"未知错误: {e}")
            raise
        
# ======================================================================
# 快速自测：python vision.py <image_path>
# ======================================================================
if __name__ == "__main__":
    """
    用法:
        python vision.py /path/to/your/image.png
    """
    import os
    import sys
    import asyncio
    from types import SimpleNamespace
    from pathlib import Path
    from langchain_core.messages import HumanMessage

    async def _quick_test(img_path: str):
        # 1. 环境变量检查
        api_url = os.getenv("DF_API_URL")
        api_key = os.getenv("DF_API_KEY")
        if not api_url or not api_key:
            print("❌  请先设置环境变量 DF_API_URL / DF_API_KEY")
            sys.exit(1)

        # 2. 检查图片
        img_path = Path(img_path).expanduser().resolve()
        if not img_path.exists():
            print(f"❌  图片不存在: {img_path}")
            sys.exit(1)

        # 3. 构造极简 MainState
        request = SimpleNamespace(chat_api_url=api_url.rstrip("/"), api_key=api_key, model = "gemini-2.5-flash-image-preview")
        state = SimpleNamespace(request=request)

        # 4. 实例化并调用
        caller = VisionLLMCaller(
            state=state,
            vlm_config={
                "mode": "generation",
                "input_image": str(img_path),
                "timeout": 60,
            },
            
        )
        print("🚀 正在请求模型，请稍候 …")
        ai_msg = await caller.call([HumanMessage(content="水果风格猫！")])

        print("\n================  结果  ================")
        print(ai_msg.content)
        print("========================================")

    # -------- 入口 --------
    if len(sys.argv) < 2:
        print("用法: python vision.py <image_path>")
        sys.exit(0)

    asyncio.run(_quick_test(sys.argv[1]))