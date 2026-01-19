import os
import wave
import base64
from typing import Optional
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.providers import get_provider
from dataflow_agent.toolkits.multimodaltool.req_img import _post_raw

log = get_logger(__name__)

async def generate_speech_and_save_async(
    text: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str = "gemini-2.5-pro-preview-tts",
    voice_name: str = "Kore", #Aoede, Charon, Fenrir, Kore, Puck, Orbit, Orus, Trochilidae, Zephyr
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    生成语音并保存为WAV文件
    """
    provider = get_provider(api_url, model)
    log.info(f"TTS using Provider: {provider.__class__.__name__}")
    
    try:
        url, payload, is_stream = provider.build_tts_request(
            api_url=api_url,
            model=model,
            text=text,
            voice_name=voice_name,
            **kwargs
        )
    except NotImplementedError:
        log.error(f"Provider {provider.__class__.__name__} does not support TTS")
        raise

    # TTS usually returns full audio in one go
    resp_data = await _post_raw(url, api_key, payload, timeout)
    
    try:
        audio_bytes = provider.parse_tts_response(resp_data)
    except Exception as e:
        log.error(f"Failed to parse TTS response: {e}")
        log.error(f"Response: {resp_data}")
        raise
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save as WAV (assuming 24kHz, 16bit, Mono as per user doc)
    with wave.open(save_path, "wb") as wav_file:
        wav_file.setnchannels(1)        # 1 Channel
        wav_file.setsampwidth(2)        # 16 bit = 2 bytes
        wav_file.setframerate(24000)    # 24kHz
        wav_file.writeframes(audio_bytes)
        
    log.info(f"Audio saved to {save_path}")
    return save_path

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    
    async def _test():
        url = os.getenv("DF_API_URL", "https://api.apiyi.com/v1")
        key = os.getenv("DF_API_KEY", "")
        model = os.getenv("DF_TTS_MODEL", "gemini-2.5-pro-preview-tts")
        
        print(f"Testing TTS with URL: {url}, Model: {model}")
        try:
            path = await generate_speech_and_save_async(
                "Data governance ensures data quality, security, and compliance through policies and standards—a critical foundation for scaling modern AI development. Recently, Large Language Models (LLMs) have emerged as a promising solution for automating data governance by translating user intent into executable transformation code. However, existing benchmarks for automated data science often emphasize snippet-level coding or high-level analytics, failing to capture the unique challenge of data governance: ensuring the correctness and quality of the data itself. To bridge this gap, we introduce DataGovBench, a benchmark featuring 150 diverse tasks grounded in real-world scenarios, built on data from actual cases. DataGovBench employs a novel ``reversed-objective'' methodology to synthesize realistic noise and utilizes rigorous metrics to assess end-to-end pipeline reliability. Our analysis on DataGovBench reveals that current models struggle with complex, multi-step workflows and lack robust error-correction mechanisms. Consequently, we propose DataGovAgent, a framework utilizing a Planner-Executor-Evaluator architecture that integrates constraint-based planning, retrieval-augmented generation, and sandboxed feedback-driven debugging. Experimental results show that DataGovAgent significantly boosts the Average Task Score (ATS) on complex tasks from 39.7 to 54.9 and reduces debugging iterations by over 77.9 compared to general-purpose baselines. ",
                "test_tts.wav",
                url, key, model
            )
            print(f"Success: {path}")
        except Exception as e:
            print(f"Error: {e}")
            
    asyncio.run(_test())
