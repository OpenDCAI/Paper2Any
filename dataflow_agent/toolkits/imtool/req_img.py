import os
import base64
import re
from typing import Tuple, Optional
import httpx

from dataflow_agent.logger import get_logger

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

def _is_dalle_model(model: str) -> bool:
    """
    判断是否为DALL-E系列模型
    """
    return model.lower().startswith(('dall-e', 'dall-e-2', 'dall-e-3'))

def _is_gemini_model(model: str) -> bool:
    """
    判断是否为Gemini系列模型
    """
    return 'gemini' in model.lower()

async def call_dalle_image_generation_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    response_format: str = "b64_json",
    timeout: int = 120,
) -> str:
    """
    DALL-E 图像生成
    """
    url = f"{api_url}/images/generations".rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
    }

    # 仅DALL-E-3支持quality和style参数
    if model.lower() == "dall-e-3":
        payload["quality"] = quality
        payload["style"] = style

    log.info(f"POST {url}")
    log.debug(f"payload: {payload}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            log.info(f"status={resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            
            if response_format == "b64_json":
                return data["data"][0]["b64_json"]
            else:
                # 如果是URL格式，下载图片并返回base64
                image_url = data["data"][0]["url"]
                image_resp = await client.get(image_url)
                image_resp.raise_for_status()
                return base64.b64encode(image_resp.content).decode("utf-8")
                
        except httpx.HTTPStatusError as e:
            log.error(f"HTTPError {e}")
            log.error(f"Response body: {e.response.text}")
            raise

async def call_dalle_image_edit_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    mask_path: Optional[str] = None,
    size: str = "1024x1024",
    response_format: str = "b64_json",
    timeout: int = 120,
) -> str:
    """
    DALL-E 图像编辑
    """
    url = f"{api_url}/images/edits".rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # 准备multipart/form-data数据
    files = {}
    data = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
    }

    # 读取图像文件
    with open(image_path, "rb") as f:
        files["image"] = (os.path.basename(image_path), f.read(), "image/png")

    # 如果有mask，添加mask文件
    if mask_path and os.path.exists(mask_path):
        with open(mask_path, "rb") as f:
            files["mask"] = (os.path.basename(mask_path), f.read(), "image/png")

    log.info(f"POST {url}")
    log.debug(f"data: {data}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        try:
            resp = await client.post(url, headers=headers, data=data, files=files)
            log.info(f"status={resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            
            if response_format == "b64_json":
                return data["data"][0]["b64_json"]
            else:
                # 如果是URL格式，下载图片并返回base64
                image_url = data["data"][0]["url"]
                image_resp = await client.get(image_url)
                image_resp.raise_for_status()
                return base64.b64encode(image_resp.content).decode("utf-8")
                
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
    纯文生图 - Gemini
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
    图像 Edit（输入文本 + 原图 -> 返回新图）- Gemini
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
    mask_path: Optional[str] = None,
    use_edit: bool = False,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    response_format: str = "b64_json",
    timeout: int = 1200,
) -> str:
    """
    根据模型类型选择不同的API进行图像生成/编辑

    参数说明
    ----------
    prompt      : 提示词
    save_path   : 保存生成图片的路径
    api_url     : OpenAI /v1 兼容地址
    api_key     : API Key
    model       : 模型名称
    image_path  : 当进行 Edit 时传入原图路径
    mask_path   : DALL-E编辑时的mask路径（可选）
    use_edit    : True => Edit；False => 纯生图
    size        : 图像尺寸（DALL-E专用）
    quality     : 图像质量（DALL-E-3专用）
    style       : 图像风格（DALL-E-3专用）
    response_format : 返回格式（DALL-E专用）
    timeout     : 请求超时（秒）

    返回值
    ----------
    返回生成结果中的 Base64 字符串；若解析失败则抛异常
    """
    # 根据模型类型选择不同的API
    if _is_dalle_model(model):
        if use_edit:
            if not image_path:
                raise ValueError("DALL-E Edit模式必须提供image_path")
            raw = await call_dalle_image_edit_async(
                api_url, api_key, model, prompt, image_path, mask_path, 
                size, response_format, timeout
            )
        else:
            raw = await call_dalle_image_generation_async(
                api_url, api_key, model, prompt, size, quality, style, 
                response_format, timeout
            )
    elif _is_gemini_model(model):
        if use_edit :
            if not image_path:
                raise ValueError("Gemini Edit模式必须提供image_path")
            raw = await call_gemini_image_edit_async(
                api_url, api_key, model, prompt, image_path, timeout
            )
        else:
            raw = await call_gemini_image_generation_async(
                api_url, api_key, model, prompt, timeout
            )
    else:
        raise ValueError(f"不支持的模型: {model}")

    # 处理返回结果
    if _is_dalle_model(model):
        # DALL-E直接返回base64，无需提取
        b64 = raw
    else:
        # Gemini需要从响应中提取base64
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
        
        # 测试Gemini模型
        MODEL_GEMINI = "gemini-2.5-flash-image-preview"
        
        # 测试DALL-E模型
        MODEL_DALLE = "dall-e-3"

        # 1) Gemini纯文生图
        # await generate_or_edit_and_save_image_async(
        #     prompt="一只霓虹风格的赛博朋克猫头像",
        #     save_path="./gen_cat_gemini.png",
        #     api_url=API_URL,
        #     api_key=API_KEY,
        #     model=MODEL_GEMINI,
        #     use_edit=False, 
        # )

        # 2) DALL-E纯文生图
        await generate_or_edit_and_save_image_async(
            prompt="一只可爱的小海獭",
            save_path="./gen_otter_dalle.png",
            api_url=API_URL,
            api_key=API_KEY,
            model=MODEL_DALLE,
            use_edit=False,
            quality="standard",
            style="vivid"
        )

        # 3) Gemini Edit 模式
        await generate_or_edit_and_save_image_async(
            prompt="请把这只猫改成蒸汽朋克风格",
            image_path="./gen_cat_gemini.png",
            save_path="./edited_cat_gemini.png",
            api_url=API_URL,
            api_key=API_KEY,
            model=MODEL_GEMINI,
            use_edit=True, 
        )

        # 4) DALL-E Edit 模式（需要mask）
        # await generate_or_edit_and_save_image_async(
        #     prompt="一只戴着贝雷帽的可爱小海獭",
        #     image_path="./otter.png",
        #     mask_path="./mask.png",
        #     save_path="./edited_otter_dalle.png",
        #     api_url=API_URL,
        #     api_key=API_KEY,
        #     model=MODEL_DALLE,
        #     use_edit=True,
        # )

    asyncio.run(_demo())