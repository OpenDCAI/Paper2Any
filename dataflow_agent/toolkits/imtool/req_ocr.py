import os
import httpx
from typing import List, Dict, Any, Optional
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.imtool.utils import encode_image_to_base64

log = get_logger(__name__)

async def call_ocr_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.01,
    timeout: int = 120,
) -> str:
    """
    调用OCR模型 (例如 Qwen-VL-OCR)
    """
    
    # 1. 准备消息列表 (深拷贝以避免修改原列表)
    processed_messages = [msg.copy() for msg in messages]
    
    # 2. 处理图像注入
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
            # 确保 content 是 text 格式再处理，或者是 list
            if isinstance(original_text, str):
                target_msg["content"] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
                    {"type": "text", "text": original_text}
                ]
            elif isinstance(original_text, list):
                # 如果已经是 list，追加 image_url
                target_msg["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                )
        else:
            # 如果没有 user 消息，强行加一条
            processed_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
                    {"type": "text", "text": "Describe this image."}
                ]
            })

    # 3. 构造请求
    url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": processed_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    log.info(f"[OCR Call] Model: {model}, URL: {url}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            log.error(f"OCR Request failed: {e.response.text}")
            raise
        except Exception as e:
            log.error(f"OCR Error: {e}")
            raise
