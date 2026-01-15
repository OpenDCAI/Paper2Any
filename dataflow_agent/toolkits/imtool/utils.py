import os
import re
import base64
from enum import Enum
from io import BytesIO
from typing import Tuple
from PIL import Image
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class Provider(str, Enum):
    APIYI = "apiyi"
    LOCAL_123 = "local_123"
    OTHER = "other"

_B64_RE = re.compile(r"[A-Za-z0-9+/=]+")

def detect_provider(api_url: str) -> Provider:
    """
    根据 api_url 粗略识别服务商
    """
    if "apiyi" in api_url:
        return Provider.APIYI
    if "123.129.219.111" in api_url:
        return Provider.LOCAL_123
    return Provider.OTHER

def extract_base64(s: str) -> str:
    """
    从任意字符串中提取最长连续 Base64 串
    """
    s = "".join(s.split())                # 去掉所有空白
    matches = _B64_RE.findall(s)          # 提取候选段
    return max(matches, key=len) if matches else ""

def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    读取本地图片并编码为 Base64，同时返回图片格式（jpeg / png）。
    如果图片过大（>3MB），则自动进行压缩/Resize以避免 413 错误。
    """
    MAX_SIZE = 6 * 1024 * 1024  # 6MB
    MAX_DIM = 2048              # 最大边长 2048

    if not os.path.exists(image_path):
         raise FileNotFoundError(f"Image not found: {image_path}")

    file_size = os.path.getsize(image_path)
    ext = image_path.rsplit(".", 1)[-1].lower()
    fmt = "jpeg" if ext in {"jpg", "jpeg"} else "png"

    # 如果文件小于 3MB 且是常见格式，直接读取
    if file_size < MAX_SIZE and fmt in ["jpeg", "png"]:
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        return b64, fmt

    # 否则进行压缩处理
    log.info(f"[utils] Image {os.path.basename(image_path)} too large ({file_size/1024/1024:.2f}MB), compressing...")
    try:
        with Image.open(image_path) as img:
            # 1. Resize if too large
            if max(img.size) > MAX_DIM:
                scale = MAX_DIM / max(img.size)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 2. Convert to RGB if needed (for JPEG)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # 3. Save to buffer as JPEG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            raw = buffer.getvalue()
            
            log.info(f"[utils] Compressed size: {len(raw)/1024/1024:.2f}MB")
            b64 = base64.b64encode(raw).decode("utf-8")
            return b64, "jpeg"
            
    except Exception as e:
        log.warning(f"[utils] Compression failed: {e}, falling back to original.")
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        return b64, fmt

def is_gemini_model(model: str) -> bool:
    """判断是否为Gemini系列模型"""
    return 'gemini' in model.lower()

def is_gemini_25(model: str) -> bool:
    """是否为 Gemini 2.5 系列"""
    return "gemini-2.5" in model.lower()

def is_gemini_3_pro(model: str) -> bool:
    """是否为 Gemini 3 Pro 系列"""
    return "gemini-3-pro" in model.lower()
