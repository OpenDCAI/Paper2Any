import os
import base64
import re
from typing import Tuple, Optional, List, Union
import httpx
from enum import Enum

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class Provider(str, Enum):
    APIYI = "apiyi"
    LOCAL_123 = "local_123"
    OTHER = "other"

_B64_RE = re.compile(r"[A-Za-z0-9+/=]+")  # 匹配 Base64 字符


def detect_provider(api_url: str) -> Provider:
    """
    根据 api_url 粗略识别服务商
    """
    if "api.apiyi.com" in api_url:
        return Provider.APIYI
    if "123.129.219.111" in api_url:
        return Provider.LOCAL_123
    return Provider.OTHER

def extract_base64(s: str) -> str:
    """
    从任意字符串中提取最长连续 Base64 串
    """
    s = "".join(s.split())                # 去掉所有空白
    # log.info(f"raw response: {s}")
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

async def _post_raw(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """
    统一的 POST，不拼接路径，由调用方传入完整 URL
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log.info(f"POST {url}")
    # log.debug(f"payload: {payload}")  # avoid logging full payload (may contain base64 image data)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), http2=False) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            log.info(f"status={resp.status_code}")
            # log.info(f"resp[:500]={resp.text}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            # NOTE: 不同兼容层（尤其是本地 123）可能对 payload 格式非常严格，
            # 这里把响应体打印出来便于定位，同时将异常继续抛出由上层决定是否重试。
            log.error(f"HTTPError {e}")
            log.error(f"Response body: {e.response.text}")
            raise


async def _post_chat_completions(
    api_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """
    统一的 /chat/completions POST（用于 local_123 的 gemini-2.5 OpenAI 兼容实现）
    """
    url = f"{api_url}/chat/completions".rstrip("/")
    return await _post_raw(url, api_key, payload, timeout)

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


def is_gemini_25(model: str) -> bool:
    """
    是否为 Gemini 2.5 系列（例如 gemini-2.5-flash-image-preview）
    """
    return "gemini-2.5" in model.lower()


def is_gemini_3_pro(model: str) -> bool:
    """
    是否为 Gemini 3 Pro 系列（例如 gemini-3-pro-image-preview）
    """
    return "gemini-3-pro" in model.lower()

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

def build_gemini_generation_request(
    api_url: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    resolution: str = "2K",
) -> tuple[str, dict]:
    """
    根据服务商 + 模型 构造 文生图 请求的 (url, payload)
    """
    provider = detect_provider(api_url)
    base = api_url.rstrip("/")

    # 1) apiyi + gemini-2.5-flash-image-preview => generateContent + aspectRatio
    if provider is Provider.APIYI and is_gemini_25(model):
        url = "https://api.apiyi.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                },
            },
        }
        return url, payload

    # 2) apiyi + gemini-3-pro-image-preview => generateContent + aspectRatio + imageSize
    if provider is Provider.APIYI and is_gemini_3_pro(model):
        url = "https://api.apiyi.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
        }
        return url, payload

    # 3) 123.129.219.111 + gemini-3-pro-image-preview => chat/completions + generationConfig
    if provider is Provider.LOCAL_123 and is_gemini_3_pro(model):
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
            "generationConfig": {
                "imageConfig": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": "4K",
                }
            },
        }
        return url, payload

    # 4) 123.129.219.111 + gemini-2.5-flash-image-preview
    #    该兼容层对入参格式可能更偏 OpenAI（messages[].content）或 Gemini 原生（contents[].parts），
    #    先尝试 OpenAI 兼容格式，失败时在 call_gemini_image_generation_async 里做 fallback 重试。
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "generationConfig": {
                "width": 1920,
                "height": 1080,
                "quality": "high",
            },
        }
        return url, payload

    # 5) 其他服务商 => 保持最初的 OpenAI 兼容文生图逻辑
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "image"},
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    return url, payload


async def call_gemini_image_generation_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
) -> dict:
    """
    纯文生图 - Gemini

    兼容策略：
    - APIYI + gemini-2.5：走 generateContent（candidates 结构），由 build_gemini_generation_request 构造 url/payload；
    - APIYI + gemini-3-pro：走 generateContent（candidates 结构），支持 aspectRatio 和 imageSize；
    - LOCAL_123 + gemini-2.5：按已验证可用的 OpenAI chat/completions 兼容实现（choices 结构）。
    """
    provider = detect_provider(api_url)

    # local_123 + gemini-2.5：强制走 OpenAI chat/completions 兼容 payload（与 online 版本保持一致）
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
            "generationConfig": {
                "imageConfig": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": "4K",
                }
            },
        }
        # log.info(payload)  # avoid logging full payload (may contain base64 image data)
        return await _post_chat_completions(api_url, api_key, payload, timeout)

    # 其它情况：沿用 build_gemini_generation_request
    url, payload = build_gemini_generation_request(api_url, model, prompt, aspect_ratio, resolution)
    # log.info(payload)  # avoid logging full payload (may contain base64 image data)
    return await _post_raw(url, api_key, payload, timeout)

def build_gemini_edit_request(
    api_url: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    b64: str,
    fmt: str,
    resolution: str = "2K",
) -> tuple[str, dict]:
    """
    根据服务商 + 模型 构造 图像编辑 请求的 (url, payload)
    """
    provider = detect_provider(api_url)
    base = api_url.rstrip("/")

    # 1) apiyi + 2.5 => generateContent + inlineData + aspectRatio
    if provider is Provider.APIYI and is_gemini_25(model) and aspect_ratio != "1:1":
        url = "https://api.apiyi.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                },
            },
        }
        return url, payload

    # 2) apiyi + 3 Pro => generateContent + inline_data + aspectRatio + imageSize
    if provider is Provider.APIYI and is_gemini_3_pro(model):
        url = "https://api.apiyi.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
        }
        return url, payload

    # 3) 123 + 3 Pro => 保持现有 chat/completions + text + image_url
    if provider is Provider.LOCAL_123 and is_gemini_3_pro(model):
        log.critical(f'走 Local 3000 + Gemini3 Pro 路径')
        url = f"{base}/chat/completions"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{fmt};base64,{b64}",
                        },
                    },
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "image"},
            # "max_tokens": 1024,
            # "temperature": 0.7,
        }
        return url, payload

    # 4) 123 + 2.5 => messages.parts + inline_data + width/height/quality
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "width": 1920,
                "height": 1080,
                "quality": "high",
            },
        }
        return url, payload

    # 5) 其他服务商 => 原始 OpenAI 兼容编辑逻辑
    url = f"{base}/chat/completions"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{fmt};base64,{b64}",
                    },
                },
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
    return url, payload


async def gemini_multi_image_edit_async(
    prompt: str,
    image_paths: List[str],
    save_path: str,
    api_url: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
    timeout: int = 300,
) -> str:
    """
    专门针对 Gemini (APIYI / Google 原生格式) 的多图编辑
    
    参数:
        image_paths: 本地图片路径列表，支持多张图片合成/编辑
        resolution: "1K", "2K", "4K" (仅 gemini-3-pro 支持 2K/4K)
    """
    # 1. 构造 parts 列表：Text + Images
    parts = [{"text": prompt}]
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        b64, fmt = _encode_image_to_base64(img_path)
        parts.append({
            "inline_data": {
                "mime_type": f"image/{fmt}",
                "data": b64,
            }
        })
        
    # 2. 构造 URL (强制使用 Google 原生格式端点)
    # 假设 api_url 可能是 "https://api.apiyi.com/v1" 或 "https://api.apiyi.com"
    # 我们需要构造类似 "https://api.apiyi.com/v1beta/models/{model}:generateContent"
    
    base_url = api_url.rstrip("/")
    # 如果用户传的是 v1 结尾，尝试去掉它以回到根域名，或者直接替换
    # 这里做个简单的处理：如果包含 /v1，则替换为 /v1beta/models/...
    # 也可以直接假定用户传入的是 host base
    
    if "/v1" in base_url:
        base_url = base_url.split("/v1")[0]
        
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    
    # 3. 构造 Payload
    image_config = {"aspectRatio": aspect_ratio}
    
    # 只有 gemini-3-pro 支持 imageSize
    if is_gemini_3_pro(model):
        image_config["imageSize"] = resolution
        
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": image_config
        }
    }
    
    # 4. 根据分辨率动态调整超时
    if is_gemini_3_pro(model):
        timeout_map = {"1K": 180, "2K": 300, "4K": 360}
        timeout = max(timeout, timeout_map.get(resolution, 300))
        
    # 5. 发送请求
    log.info(f"[Gemini Multi-Image] POST {url} (resolution={resolution}, images={len(image_paths)})")
    resp_json = await _post_raw(url, api_key, payload, timeout)
    
    # 6. 解析结果 (Google 格式)
    # candidates[0].content.parts[0].inlineData.data
    try:
        candidates = resp_json.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned. Response: {str(resp_json)[:200]}")
            
        content = candidates[0].get("content", {})
        res_parts = content.get("parts", [])
        if not res_parts:
            raise RuntimeError("No parts in response content")
            
        inline_data = res_parts[0].get("inlineData", {})
        b64_res = inline_data.get("data")
        
        if not b64_res:
             raise RuntimeError("No inlineData.data found in response")
             
    except Exception as e:
        log.error(f"Failed to parse Gemini response: {e}")
        log.error(f"Full response: {resp_json}")
        raise
        
    # 7. 保存
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64_res))
        
    log.info(f"Multi-image edit saved to {save_path}")
    return b64_res


async def call_gemini_image_edit_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout: int = 120,
    aspect_ratio: str = "1:1",
    resolution: str = "2K",
) -> dict:
    """
    图像 Edit（输入文本 + 原图 -> 返回新图）- Gemini

    兼容策略：
    - APIYI + gemini-2.5：走 generateContent（candidates 结构），由 build_gemini_edit_request 构造 url/payload；
    - APIYI + gemini-3-pro：走 generateContent（candidates 结构），支持 aspectRatio 和 imageSize；
    - LOCAL_123 + gemini-2.5：按已验证可用的 OpenAI chat/completions 兼容实现（choices 结构）。
    """
    provider = detect_provider(api_url)

    # local_123 + gemini-2.5：强制走 OpenAI chat/completions 兼容 payload（与 online 版本保持一致）
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        b64, fmt = _encode_image_to_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{fmt};base64,{b64}"},
                    },
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
        # log.info(payload)  # avoid logging full payload (contains data:image/...;base64,...)
        return await _post_chat_completions(api_url, api_key, payload, timeout)

    b64, fmt = _encode_image_to_base64(image_path)
    url, payload = build_gemini_edit_request(api_url, model, prompt, aspect_ratio, b64, fmt, resolution)
    # log.info(payload)  # avoid logging full payload (may contain base64 image data)
    return await _post_raw(url, api_key, payload, timeout)

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
    aspect_ratio: str = '16:9',
    resolution: str = "2K",
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
    aspect_ratio: 图像宽高比（Gemini专用，如 16:9, 1:1, 9:16 等）
    resolution  : 图像分辨率（Gemini-3 Pro专用，可选: 1K, 2K, 4K）
    quality     : 图像质量（DALL-E-3专用）
    style       : 图像风格（DALL-E-3专用）
    response_format : 返回格式（DALL-E专用）
    timeout     : 请求超时（秒）

    返回值
    ----------
    返回生成结果中的 Base64 字符串；若解析失败则抛异常
    """
    # 根据分辨率动态调整超时时间（仅对 Gemini-3 Pro 生效）
    if _is_gemini_model(model) and is_gemini_3_pro(model):
        timeout_map = {"1K": 180, "2K": 300, "4K": 360}
        timeout = timeout_map.get(resolution, 300)
    
    log.info(f"aspect_ratio: {aspect_ratio} \n resolution: {resolution} \n use_edit: {use_edit} \n model: {model} \n api_url: {api_url} \n timeout: {timeout} \n api_key: {api_key}")
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
        # 针对 apiyi + 非 1:1 比例的特殊处理：
        # - 如果是 gemini 且 api_url 包含 api.apiyi.com 且 aspect_ratio != "1:1"
        #   则强制走 Nano Banana 的 generateContent 端点；
        # - 否则维持原有行为。

        if use_edit:
            if not image_path:
                raise ValueError("Gemini Edit模式必须提供image_path")
            log.critical(f'正在执行 Gemini 的编辑模式......')
            raw_data = await call_gemini_image_edit_async(
                api_url, api_key, model, prompt, image_path, timeout, aspect_ratio, resolution
            )
        else:
            raw_data = await call_gemini_image_generation_async(
                api_url, api_key, model, prompt, timeout, aspect_ratio, resolution
            )
    else:
        raise ValueError(f"不支持的模型: {model}")

    # 处理返回结果
    if _is_dalle_model(model):
        # DALL-E直接返回base64，无需提取
        b64 = raw
    elif _is_gemini_model(model):
        # Gemini：根据不同 provider / 模型解析 base64
        data = raw_data

        # 1) Nano Banana / Google Gemini 风格：candidates[0].content.parts[0].inlineData.data
        if "candidates" in data:
            try:
                candidates = data["candidates"]
                if not candidates:
                    raise RuntimeError("candidates 为空")

                content = candidates[0]["content"]
                parts = content["parts"]
                inline_data = parts[0]["inlineData"]
                b64 = inline_data["data"]
            except Exception as e:
                log.error(f"解析 Gemini candidates 结构失败: {e}")
                log.error(f"响应结构可能变化，完整响应如下（截断）: {str(data)[:2000]}")
                raise

        # 2) OpenAI 兼容 chat/completions 风格：choices[0].message.content 中嵌入 base64
        elif "choices" in data:
            try:
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    # 内容里直接是 base64 或包含 base64 片段
                    b64 = extract_base64(content)
                elif isinstance(content, list):
                    # OpenAI 风格 content 为 list[{"type": "...", ...}]
                    joined = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                    b64 = extract_base64(joined)
                else:
                    raise RuntimeError(f"不支持的 content 类型: {type(content)}")

                if not b64:
                    raise RuntimeError("从 choices[0].message.content 中未提取到 Base64")
            except Exception as e:
                log.error(f"解析 OpenAI 兼容 Gemini 响应失败: {e}")
                log.error(f"响应结构可能变化，完整响应如下（截断）: {str(data)[:2000]}")
                raise
        else:
            # 未知结构，直接报错并输出前一部分响应便于调试
            log.error("未知的 Gemini 响应结构：既没有 candidates 也没有 choices")
            log.error(f"响应内容（截断）: {str(data)[:2000]}")
            raise RuntimeError("未知的 Gemini 响应结构：缺少 candidates / choices 字段")
    else:
        raise ValueError(f"不支持的模型: {model}")

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
        # API_URL = "https://api.apiyi.com/v1"
        API_KEY = os.getenv("DF_API_KEY")
        
        # 测试Gemini模型
        MODEL_GEMINI = "gemini-2.5-flash-image-preview"
        # gemini-2.5-flash-image-preview gemini-3-pro-image-preview
        
        # 测试DALL-E模型
        # MODEL_DALLE = "dall-e-3"

        # 1) Gemini纯文生图
        await generate_or_edit_and_save_image_async(
            prompt="一只霓虹风格的赛博朋克猫头像",
            save_path="./gen_cat_gemini.png",
            api_url=API_URL,
            api_key=API_KEY,
            model=MODEL_GEMINI,
            use_edit=False, 
        )

        # 2) DALL-E纯文生图
        # await generate_or_edit_and_save_image_async(
        #     prompt="一只可爱的小海獭",
        #     save_path="./gen_otter_dalle.png",
        #     api_url=API_URL,
        #     api_key=API_KEY,
        #     model=MODEL_DALLE,
        #     use_edit=False,
        #     quality="standard",
        #     style="vivid"
        # )

        # 3) Gemini Edit 模式
        # await generate_or_edit_and_save_image_async(
        #     prompt="请把这只猫改成蒸汽朋克风格",
        #     # image_path=f"{get_project_root()}/tests/cat_icon.png",
        #     save_path="./edited_cat_gemini.png",
        #     api_url=API_URL,
        #     api_key=API_KEY,
        #     model=MODEL_GEMINI,
        #     # use_edit=True, 
        # )
        
        # 3.5) Gemini Multi-Image Edit (Manual Call)
        # await gemini_multi_image_edit_async(
        #     prompt="Merge these images into a sci-fi poster",
        #     image_paths=["", ""],
        #     save_path="./merged_gemini.png",
        #     api_url=API_URL,
        #     api_key=API_KEY,
        #     model="gemini-2.5-flash-image",
        #     resolution="2K"
        # )

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
