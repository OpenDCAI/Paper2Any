import os
import base64
import httpx
import subprocess
import uuid
from typing import List, Dict, Any, Optional
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.providers import get_provider

log = get_logger(__name__)

def _compress_video(input_path: str) -> str:
    """
    使用 ffmpeg 压缩视频。
    返回压缩后的临时文件路径，如果失败返回原路径。
    """
    output_path = f"/tmp/compressed_{uuid.uuid4()}.mp4"
    
    # 压缩策略：缩放至720p，CRF 28 (平衡画质与大小)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", "28", "-preset", "faster",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    
    log.info(f"Compressing video > 20MB: {input_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(output_path):
            new_size = os.path.getsize(output_path)
            log.info(f"Compression success. Size: {new_size/1024/1024:.2f}MB")
            return output_path
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg compression failed: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        log.error(f"Compression error: {e}")
    
    return input_path

def _encode_video_to_base64(video_path: str) -> tuple[str, str]:
    """
    读取视频文件并编码为Base64。如果视频大于20MB，尝试自动压缩。
    返回: (base64_str, mime_type)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    ext = video_path.rsplit(".", 1)[-1].lower()
    
    # 默认 MIME 类型处理
    mime_map = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "quicktime": "video/quicktime",
        "avi": "video/x-msvideo",
        "mpeg": "video/mpeg",
        "wmv": "video/x-ms-wmv"
    }
    mime_type = mime_map.get(ext, "video/mp4")
    if ext not in mime_map:
        log.warning(f"Unknown video extension {ext}, defaulting to video/mp4")

    # 检查文件大小 (20MB)
    file_size = os.path.getsize(video_path)
    final_path = video_path
    is_compressed = False

    if file_size > 20 * 1024 * 1024:
        log.warning(f"Video size {file_size/1024/1024:.2f}MB > 20MB, attempting compression...")
        compressed_path = _compress_video(video_path)
        if compressed_path != video_path:
            final_path = compressed_path
            mime_type = "video/mp4" # ffmpeg 输出总是 mp4
            is_compressed = True
    
    try:
        with open(final_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
    finally:
        # 清理临时压缩文件
        if is_compressed and os.path.exists(final_path):
            try:
                os.remove(final_path)
                log.info(f"Removed temp compressed video: {final_path}")
            except Exception as e:
                log.warning(f"Failed to remove temp video: {e}")

    return b64, mime_type

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
    
    log.info(f"[Video] POST {url}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

async def call_video_understanding_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    video_path: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    timeout: int = 300, # Video processing might take longer
    **kwargs,
) -> str:
    """
    调用视频理解模型
    """
    b64, mime_type = _encode_video_to_base64(video_path)
    log.info(f"[Video] Encoded video {video_path}, mime={mime_type}, size={len(b64)/1024/1024:.2f}MB")

    # 1. Prepare Messages
    processed_messages = [msg.copy() for msg in messages]
    
    # Inject video into the last user message
    target_msg = None
    for m in reversed(processed_messages):
        if m["role"] == "user":
            target_msg = m
            break
            
    # OpenAI Vision Format / Gemini OpenAI-Compat Format
    video_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64}"
        },
        "mime_type": mime_type
    }
            
    if target_msg:
        original_content = target_msg["content"]
        if isinstance(original_content, str):
            target_msg["content"] = [
                {"type": "text", "text": original_content},
                video_content
            ]
        elif isinstance(original_content, list):
            target_msg["content"].append(video_content)
    else:
         processed_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video."},
                video_content
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

    load_dotenv()
    
    # 创建一个空的 dummy 视频文件是不容易被 ffmpeg 处理的
    # 所以视频测试仅当用户提供了有效路径时才尝试，或者尝试查找
    def find_any_mp4():
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith(".mp4"):
                    return os.path.join(root, f)
        return None

    async def _test():
        API_URL = os.getenv("DF_API_URL", "http://127.0.0.1:3000/v1")
        API_KEY = os.getenv("DF_API_KEY", "sk-xxx")
        MODEL = os.getenv("DF_IMG_MODEL", "gemini-2.5-flash") # Use a chat/vision model

        print(f"--- Video Understanding Config ---")
        print(f"URL: {API_URL}")
        print(f"Model: {MODEL}")
        print(f"----------------------------------")

        # 尝试使用环境变量指定视频，否则查找
        video_path = os.getenv("TEST_VIDEO_PATH")
        if not video_path:
            video_path = find_any_mp4()
        
        if not video_path or not os.path.exists(video_path):
            print("No video found for testing. Set TEST_VIDEO_PATH env var.")
            return

        print(f"Using video: {video_path}")
        
        try:
            print("[1] Testing Video Understanding...")
            result = await call_video_understanding_async(
                model=MODEL,
                messages=[{"role": "user", "content": "Describe what happens in this video."}],
                api_url=API_URL,
                api_key=API_KEY,
                video_path=video_path
            )
            print(">> Video Result:", result)
        except Exception as e:
            print(f">> Video Failed: {e}")

    asyncio.run(_test())
