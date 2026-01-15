import os
import httpx
from typing import List, Dict, Any, Optional

from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.imtool.utils import (
    encode_image_to_base64
)

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

    # 3. 发送请求
    url = f"{api_url.rstrip('/')}/chat/completions"
    
    payload = {
        "model": model,
        "messages": processed_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    data = await _post_raw(url, api_key, payload, timeout)
    
    if "choices" not in data or not data["choices"]:
        log.error(f"Invalid API response structure: {data}")
        if "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
        raise RuntimeError(f"Unknown API response format: {data}")
        
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    import asyncio
    import argparse
    from dotenv import load_dotenv
    
    # Load env vars from .env file if present
    load_dotenv()

    async def main():
        parser = argparse.ArgumentParser(description="Test Image Understanding API")
        parser.add_argument("--image", required=True, help="Path to image file")
        parser.add_argument("--model", default="gemini-2.5-flash", help="Model name")
        parser.add_argument("--api_url", default=os.getenv("DF_API_URL", "https://api.apiyi.com/v1"), help="API URL")
        parser.add_argument("--api_key", default=os.getenv("DF_API_KEY"), help="API Key")
        
        args = parser.parse_args()
        
        if not args.api_key:
            print("Error: API Key is required (via --api_key or DF_API_KEY env var)")
            return

        print(f"Testing with image: {args.image}")
        print(f"Model: {args.model}")
        print(f"API URL: {args.api_url}")
        
        try:
            result = await call_image_understanding_async(
                model=args.model,
                messages=[{"role": "user", "content": "Describe this image in detail."}],
                api_url=args.api_url,
                api_key=args.api_key,
                image_path=args.image
            )
            print("\nResult:")
            print("-" * 40)
            print(result)
            print("-" * 40)
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())
