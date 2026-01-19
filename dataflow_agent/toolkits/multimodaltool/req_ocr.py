import os
import httpx
from typing import List, Dict, Any, Optional
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.utils import encode_image_to_base64
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
    
    log.info(f"[OCR] POST {url}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error(f"OCR Request failed: {e.response.text}")
            raise
        except Exception as e:
            log.error(f"OCR Error: {e}")
            raise

async def call_ocr_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.01,
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    调用OCR模型 (例如 Qwen-VL-OCR)
    使用 Provider 策略进行请求构建
    """
    
    # 1. 准备消息列表 (深拷贝以避免修改原列表)
    processed_messages = [msg.copy() for msg in messages]
    
    # 2. 处理图像注入 (这部分逻辑通常是通用的，可以在这里保留，也可以移到 Provider)
    # 目前保持在这里，因为这是业务层面的“如何组合消息”
    if image_path:
        b64, fmt = encode_image_to_base64(image_path)
        
        # 找到最后一条 user 消息注入图片
        target_msg = None
        for m in reversed(processed_messages):
            if m["role"] == "user":
                target_msg = m
                break
        
        if target_msg:
            original_text = target_msg.get("content", "")
            if isinstance(original_text, str):
                target_msg["content"] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
                    {"type": "text", "text": original_text}
                ]
            elif isinstance(original_text, list):
                target_msg["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                )
        else:
            processed_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
                    {"type": "text", "text": "Describe this image."}
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

    # --- 辅助函数：创建测试图片 ---
    def create_text_image(path: str, text="Hello World"):
        if not os.path.exists(path):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (300, 100), color='white')
                d = ImageDraw.Draw(img)
                d.text((10,10), text, fill=(0,0,0))
                img.save(path)
                print(f"Created text image: {path}")
            except ImportError:
                print("PIL not available, skipping image creation")
        return path

    async def _test():
        API_URL = os.getenv("DF_API_URL", "http://127.0.0.1:3000/v1")
        API_KEY = os.getenv("DF_API_KEY", "sk-xxx")
        MODEL = os.getenv("DF_OCR_MODEL", "qwen-vl-ocr-2025-11-20")

        print(f"--- OCR Config ---")
        print(f"URL: {API_URL}")
        print(f"Model: {MODEL}")
        print(f"----------------")

        img_path = "/data/users/liuzhou/dev/DataFlow-Agent/tests/test_02.png"
        if not os.path.exists(img_path):
            print("Image creation failed, skipping test.")
            return

        try:
            print("[1] Testing OCR...")
            result = await call_ocr_async(
                model=MODEL,
                messages=[{"role": "user", "content": "Extract text from this image, return JSON. "}],
                api_url=API_URL,
                api_key=API_KEY,
                image_path=img_path
            )
            print(">> OCR Result:", result)
        except Exception as e:
            print(f">> OCR Failed: {e}")

    asyncio.run(_test())
