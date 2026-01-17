import os
import httpx
from typing import List, Dict, Any, Optional

from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.utils import (
    encode_image_to_base64
)
from dataflow_agent.toolkits.multimodaltool.providers import get_provider

log = get_logger(__name__)

async def _post_raw(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """Helper for POST request"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    log.info(f"[Understanding] POST {url}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

async def call_image_understanding_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    image_path: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    调用通用图像理解模型
    """
    
    # 1. 准备消息
    processed_messages = [msg.copy() for msg in messages]

    # 2. 处理图像
    if image_path:
        b64, fmt = encode_image_to_base64(image_path)
        
        target_msg = None
        if processed_messages:
            last_msg = processed_messages[-1]
            if last_msg["role"] == "user":
                target_msg = last_msg

        if target_msg:
            original_content = target_msg["content"]
            
            if isinstance(original_content, str):
                target_msg["content"] = [
                    {"type": "text", "text": original_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                ]
            elif isinstance(original_content, list):
                target_msg["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                )
        else:
             # 如果没有 user 消息或列表为空，追加一条
            processed_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                ]
            })

    # 3. 使用 Provider 构造请求
    provider = get_provider(api_url, model)
    url, payload = provider.build_chat_request(
        api_url=api_url,
        model=model,
        messages=processed_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    # 4. 发送请求
    data = await _post_raw(url, api_key, payload, timeout)
    
    # 5. 解析响应
    return provider.parse_chat_response(data)

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from PIL import Image

    load_dotenv()
    
    def create_dummy_image(path: str):
        if not os.path.exists(path):
            img = Image.new('RGB', (100, 100), color='green')
            img.save(path)
        return path

    async def _test():
        API_URL = os.getenv("DF_API_URL", "http://127.0.0.1:3000/v1")
        API_KEY = os.getenv("DF_API_KEY", "sk-xxx")
        MODEL = os.getenv("DF_IMG_MODEL", "gemini-2.5-flash") # Use a chat/vision model

        print(f"--- Understanding Config ---")
        print(f"URL: {API_URL}")
        print(f"Model: {MODEL}")
        print(f"----------------------------")

        img_path = create_dummy_image("./test_understanding.png")
        
        try:
            print("[1] Testing Image Understanding...")
            result = await call_image_understanding_async(
                model=MODEL,
                messages=[{"role": "user", "content": "What is the dominant color in this image?"}],
                api_url=API_URL,
                api_key=API_KEY,
                image_path=img_path
            )
            print(">> Understanding Result:", result)
        except Exception as e:
            print(f">> Understanding Failed: {e}")

    asyncio.run(_test())
