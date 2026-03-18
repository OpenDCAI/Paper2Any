import os
import re
import wave
import base64
from typing import Optional, List
import httpx
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.providers import get_provider, CosyVoiceProvider
from dataflow_agent.toolkits.multimodaltool.req_img import _post_raw

log = get_logger(__name__)


class TTSFallbackToF5Error(RuntimeError):
    """云 TTS（CosyVoice）在指定次数内均失败，调用方可用 F5-TTS 回退生成。"""


async def _post_raw_bytes(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> bytes:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log.info(f"POST {url} (binary)")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), http2=False) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            log.info(f"status={resp.status_code}")
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            log.error(f"HTTPError {e}")
            log.error(f"Response body: {e.response.text}")
            raise

def split_tts_sentences(content: str) -> List[str]:
    content = content.replace("\r", "")
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content.strip()]

    sentence_splitter = re.compile(r"(?<=[。！？.!?;；])\s*")
    sentences: List[str] = []
    for para in paragraphs:
        if not para:
            continue
        parts = [s.strip() for s in sentence_splitter.split(para) if s.strip()]
        if not parts:
            sentences.append(para.strip())
        else:
            sentences.extend(parts)
    return sentences

def split_tts_text_by_bytes(content: str, limit_bytes: int) -> List[str]:
    if limit_bytes is None or limit_bytes <= 0:
        return [content]
    sentences = split_tts_sentences(content)
    if not sentences:
        return [content]

    chunks: List[str] = []
    buf = ""
    buf_bytes = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_bytes = len(sent.encode("utf-8"))
        if sent_bytes > limit_bytes:
            if buf:
                chunks.append(buf)
                buf = ""
                buf_bytes = 0
            # Fallback: hard split by bytes while keeping order
            part = ""
            part_bytes = 0
            for ch in sent:
                ch_b = len(ch.encode("utf-8"))
                if part_bytes + ch_b > limit_bytes and part:
                    chunks.append(part)
                    part = ch
                    part_bytes = ch_b
                else:
                    part += ch
                    part_bytes += ch_b
            if part:
                chunks.append(part)
            continue

        if not buf:
            buf = sent
            buf_bytes = sent_bytes
            continue
        if buf_bytes + 1 + sent_bytes <= limit_bytes:
            buf = f"{buf} {sent}"
            buf_bytes += 1 + sent_bytes
        else:
            chunks.append(buf)
            buf = sent
            buf_bytes = sent_bytes

    if buf:
        chunks.append(buf)
    return chunks

def split_tts_text(content: str, limit: int) -> List[str]:
    if limit is None or limit <= 0:
        return [content]
    if len(content) <= limit:
        return [content]
    # Normalize whitespace
    content = content.replace("\r", "")
    parts: List[str] = []
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content.strip()]

    for para in paragraphs:
        if len(para) <= limit:
            parts.append(para)
            continue
        sentences = split_tts_sentences(para)
        if not sentences:
            sentences = [para]
        buf = ""
        for sent in sentences:
            if not buf:
                buf = sent
                continue
            if len(buf) + 1 + len(sent) <= limit:
                buf = f"{buf} {sent}"
            else:
                parts.append(buf)
                buf = sent
        if buf:
            parts.append(buf)

    # Hard split if any chunk is still too large
    final_parts: List[str] = []
    for p in parts:
        if len(p) <= limit:
            final_parts.append(p)
        else:
            for i in range(0, len(p), limit):
                final_parts.append(p[i:i + limit])
    return final_parts

async def generate_speech_bytes_async(
    text: str,
    api_url: str,
    api_key: str,
    model: str = "cosyvoice-v3-flash",
    voice_name: str = "",
    timeout: int = 120,
    max_attempts: int = 10,
    **kwargs,
) -> bytes:
    """
    生成单段 TTS 音频字节。失败时重试最多 max_attempts 次；
    若仍失败且 max_attempts <= 5 则抛 TTSFallbackToF5Error，否则抛 RuntimeError。
    支持 CosyVoice（DashScope SDK）；失败可抛 TTSFallbackToF5Error 回退 F5。
    """
    import asyncio
    provider = get_provider(api_url, model)
    log.info(f"TTS using Provider: {provider.__class__.__name__}")

    # CosyVoice：走 DashScope SDK（同步），在 executor 中执行，重试后失败抛 TTSFallbackToF5Error
    if isinstance(provider, CosyVoiceProvider):
        last_error = None
        for attempt in range(max_attempts):
            try:
                loop = asyncio.get_event_loop()
                audio_bytes = await loop.run_in_executor(
                    None,
                    lambda: provider.synthesize_to_bytes(
                        api_key=api_key,
                        text=text,
                        model=model,
                        voice_name=voice_name,
                        **kwargs,
                    ),
                )
                if audio_bytes:
                    return audio_bytes
            except Exception as e:
                last_error = e
                log.warning(
                    "[cosyvoice] attempt %s/%s failed: %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts - 1:
                    # 限流（Throttling.RateQuota）时延长等待再重试
                    err_str = str(e).lower()
                    if "ratequota" in err_str or "rate limit" in err_str or "throttling" in err_str:
                        delay = 3.0 + attempt * 2.0  # 3s, 5s, 7s, ...
                        log.warning("[cosyvoice] rate limit, wait %.1fs before retry", delay)
                        await asyncio.sleep(delay)
                    continue
        msg = "CosyVoice TTS 重试后仍失败，将回退 F5"
        err = TTSFallbackToF5Error(msg)
        if last_error is not None:
            err.__cause__ = last_error
        log.error("CosyVoice TTS failed after %s attempts: %s", max_attempts, last_error)
        raise err

    # HTTP 类 Provider（Gemini / OpenAI TTS 等）
    def _build_payload(use_text: str):
        try:
            return provider.build_tts_request(
                api_url=api_url,
                model=model,
                text=use_text,
                voice_name=voice_name,
                **kwargs,
            )
        except NotImplementedError:
            log.error(f"Provider {provider.__class__.__name__} does not support TTS")
            raise

    audio_bytes = None
    last_error = None
    resp_data = None
    for attempt in range(max_attempts):
        url, payload, is_stream = _build_payload(text)
        response_type = payload.pop("__response_type__", "json")
        if response_type == "binary":
            resp_data = await _post_raw_bytes(url, api_key, payload, timeout)
        else:
            resp_data = await _post_raw(url, api_key, payload, timeout)
        try:
            audio_bytes = provider.parse_tts_response(resp_data)
            break
        except Exception as e:
            last_error = e
            if "No parts in content" in str(e) or "No parts" in str(e):
                if attempt < max_attempts - 1:
                    log.warning(
                        f"TTS returned empty (attempt {attempt + 1}/{max_attempts}), retrying: {e}"
                    )
                    continue
            log.error(f"Failed to parse TTS response: {e}")
            log.error(f"Response: {resp_data}")
    if audio_bytes is None:
        msg = "内容存在问题，审查不过关，需要修改"
        if max_attempts <= 5:
            err = TTSFallbackToF5Error(msg)
        else:
            err = RuntimeError(msg)
        if last_error is not None:
            err.__cause__ = last_error
        log.error(f"TTS failed after {max_attempts} attempts: {msg}; last_error={last_error}")
        raise err
    return audio_bytes


async def generate_speech_and_save_async(
    text: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str = "cosyvoice-v3-flash",
    voice_name: str = "",
    timeout: int = 120,
    max_chars: int = 1500,
    max_attempts: int = 10,
    **kwargs,
) -> str:
    """
    生成语音并保存为 WAV 文件。单段失败时重试最多 max_attempts 次，
    若 max_attempts <= 5 且仍失败则抛 TTSFallbackToF5Error 供调用方用 F5 回退。
    """
    chunks = split_tts_text(text, max_chars)
    log.info(f"TTS split into {len(chunks)} chunk(s) with max_chars={max_chars}")

    audio_chunks: List[bytes] = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            audio_bytes = await generate_speech_bytes_async(
                text=chunk,
                api_url=api_url,
                api_key=api_key,
                model=model,
                voice_name=voice_name,
                timeout=timeout,
                max_attempts=max_attempts,
                **kwargs,
            )
        except Exception as e:
            log.error(f"Failed to generate speech (chunk {idx}/{len(chunks)}): {e}")
            raise
        audio_chunks.append(audio_bytes)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # CosyVoice 返回完整 WAV（带 44 字节头），多段需剥掉头再拼接；Gemini 为 raw PCM
    if audio_chunks and audio_chunks[0][:4] == b"RIFF":
        pcm = b"".join(c[44:] for c in audio_chunks)
    else:
        pcm = b"".join(audio_chunks)

    with wave.open(save_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)

    log.info(f"Audio saved to {save_path}")
    return save_path

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    
    async def _test():
        url = os.getenv("DF_API_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/inference")
        key = os.getenv("COSYVOICE_KEY") or os.getenv("DF_API_KEY", "")
        model = os.getenv("DF_TTS_MODEL", "cosyvoice-v3-flash")
        log.info(f"Testing TTS with URL: {url}, Model: {model}")
        voice_name = "longanhuan"
        if not key:
            raise RuntimeError("Please set COSYVOICE_KEY or DF_API_KEY before running req_tts.py directly")
        try:
            path = await generate_speech_and_save_async(
                "Hello this is a test",
                # f"/data/users/ligang/Paper2Any/frontend-workflow/public/paper2video/cosyvoice/v3-plus/{voice_name}.wav",
                "test.wav",
                url, key, model, voice_name=voice_name,
            )
            log.info(f"Success: {path}")
        except Exception as e:
            log.error(f"Error: {e}")
            
    asyncio.run(_test())
