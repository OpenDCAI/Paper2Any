import os
import base64
import re
from typing import Tuple, Optional

import httpx

from dataflow_agent.logger import get_logger
<<<<<<< HEAD

=======
import asyncio
import os
>>>>>>> 9ceb602 (debug prompt agent)
log = get_logger(__name__)

_B64_RE = re.compile(r"[A-Za-z0-9+/=]+")  # 匹配 Base64 字符

def extract_base64(s: str) -> str:
    """
    从任意字符串中提取最长连续 Base64 串
    """
    s = "".join(s.split())                # 去掉所有空白
    matches = _B64_RE.findall(s)          # 提取候选段
    return max(matches, key=len) if matches else ""

def _encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    读取本地图片并编码为 Base64，同时返回图片格式（jpeg / png）
    """
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    if ext in {"jpg", "jpeg"}:
        fmt = "jpeg"
    elif ext == "png":
        fmt = "png"
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    return b64, fmt

async def _post_chat_completions(
    api_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """
    统一的 /chat/completions POST
    """
    url = f"{api_url}/chat/completions".rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log.info(f"POST {url}")
    log.debug(f"payload: {payload}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), http2=False) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            log.info(f"status={resp.status_code}")
            log.debug(f"resp.text[:500]={resp.text[:500]}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error(f"HTTPError {e}")
            log.error(f"Response body: {e.response.text}")
            raise


async def call_gemini_image_generation_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
) -> str:
    """
    纯文生图
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # Gemini 系列要求显式指定返回图片
        "response_format": {"type": "image"},
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    data = await _post_chat_completions(api_url, api_key, payload, timeout)
    return data["choices"][0]["message"]["content"]


async def call_gemini_image_edit_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout: int = 120,
) -> str:
    """
    图像 Edit（输入文本 + 原图 -> 返回新图）
    """
    b64, fmt = _encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
            ],
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "image"},
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    data = await _post_chat_completions(api_url, api_key, payload, timeout)
    return data["choices"][0]["message"]["content"]


# -------------------------------------------------
# 对外主接口
# -------------------------------------------------
async def generate_or_edit_and_save_image_async(
    prompt: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str,
    *,
    image_path: Optional[str] = None,
    use_edit: bool = False,
    timeout: int = 120,
) -> str:
    """
    根据开关选择生图或编辑，并将返回的 Base64 图片保存到本地。

    参数说明
    ----------
    prompt      : 提示词
    save_path   : 保存生成图片的路径
    api_url     : OpenAI /v1 兼容地址
    api_key     : API Key
    model       : 模型名称（如 gemini-2.5-flash-image-preview）
    image_path  : 当进行 Edit 时传入原图路径
    use_edit    : True => Edit；False => 纯生图
    timeout     : 请求超时（秒）

    返回值
    ----------
    返回生成结果中的 Base64 字符串；若解析失败则抛异常
    """
    if use_edit or image_path:
        if not image_path:
            raise ValueError("use_edit=True 时必须提供 image_path")
        raw = await call_gemini_image_edit_async(
            api_url, api_key, model, prompt, image_path, timeout
        )
    else:
        raw = await call_gemini_image_generation_async(
            api_url, api_key, model, prompt, timeout
        )

    b64 = extract_base64(raw)
    if not b64:
        raise RuntimeError(f"未找到 Base64 字符串，原始响应前 200 字符: {raw[:200]}")

    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))

    log.info(f"图片已保存至 {save_path}")
    return b64


# -------------------------------------------------
# 当以脚本运行时做个简单示例
# -------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _demo():
        API_URL = "http://123.129.219.111:3000/v1"
        API_KEY = os.getenv("DF_API_KEY")
        MODEL = "gemini-2.5-flash-image-preview"

        # 1) 纯文生图
        await generate_or_edit_and_save_image_async(
            prompt="一只霓虹风格的赛博朋克猫头像",
            save_path="./gen_cat.png",
            api_url=API_URL,
            api_key=API_KEY,
            model=MODEL,
            use_edit=False, 
        )

        # 2) Edit 模式
        await generate_or_edit_and_save_image_async(
            prompt="请把这只猫改成蒸汽朋克风格",
            image_path="./gen_cat.png",
            save_path="./edited_cat.png",
            api_url=API_URL,
            api_key=API_KEY,
            model=MODEL,
            use_edit=True, 
        )

    asyncio.run(_demo())

