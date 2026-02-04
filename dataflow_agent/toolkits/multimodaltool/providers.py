import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List
from dataflow_agent.toolkits.multimodaltool.utils import (
    Provider, detect_provider, extract_base64, 
    is_gemini_model, is_gemini_25, is_gemini_3_pro
)
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class AIProviderStrategy(ABC):
    """
    通用 AI 服务商策略基类
    支持：
    1. 图像生成 (Generation)
    2. 多模态理解 (Chat/Vision/Video/OCR)
    """
    
    @abstractmethod
    def match(self, api_url: str, model: str) -> bool:
        """判断当前策略是否适用"""
        pass
        
    # --- Generation Interface ---
    
    @abstractmethod
    def build_generation_request(
        self, 
        api_url: str, 
        model: str, 
        prompt: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        构造文生图请求
        Returns: (url, payload, is_stream)
        """
        pass

    def build_edit_request(
        self, 
        api_url: str, 
        model: str, 
        prompt: str, 
        image_b64: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        构造图生图/编辑请求
        Returns: (url, payload, is_stream)
        
        注意：如果返回的 payload 包含 "__is_multipart__": True，
        则 payload 应包含 "files" 和 "data" 字段，用于 multipart/form-data 上传。
        """
        raise NotImplementedError("Edit not supported by this provider")

    def build_multi_image_edit_request(
        self,
        api_url: str,
        model: str,
        prompt: str,
        image_b64_list: List[Tuple[str, str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        构造多图编辑请求
        Returns: (url, payload, is_stream)
        """
        raise NotImplementedError("Multi-image edit not supported by this provider")
        
    @abstractmethod
    def parse_generation_response(self, response_data: Dict[str, Any]) -> str:
        """
        解析生图响应，返回图片 Base64 字符串
        """
        pass

    # --- TTS Interface ---

    def build_tts_request(self, api_url: str, model: str, text: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        """
        构造TTS请求
        Returns: (url, payload, is_stream)
        """
        raise NotImplementedError("TTS not supported by this provider")
    
    def parse_tts_response(self, response_data: Dict[str, Any]) -> bytes:
        """
        解析TTS响应，返回音频二进制数据
        """
        raise NotImplementedError("TTS not supported by this provider")

    # --- Understanding / Chat Interface ---

    def build_chat_request(
        self,
        api_url: str,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        构造对话/理解请求 (OCR, Image Understanding, Video Understanding)
        Returns: (url, payload)
        
        Default implementation: OpenAI Standard Format
        """
        url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        return url, payload

    def parse_chat_response(self, response_data: Dict[str, Any]) -> str:
        """
        解析对话/理解响应，返回文本内容
        
        Default implementation: OpenAI Standard Format
        """
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        if "error" in response_data:
             raise RuntimeError(f"API Error: {response_data['error']}")
        raise RuntimeError(f"Unknown API response format: {str(response_data)[:200]}")


class ApiYiGeminiProvider(AIProviderStrategy):
    """
    APIYI 服务商针对 Gemini 模型的特殊处理
    """
    def match(self, api_url: str, model: str) -> bool:
        return detect_provider(api_url) is Provider.APIYI and is_gemini_model(model)

    def _get_base_url(self, api_url: str) -> str:
        base = api_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base

    # --- Generation ---

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = self._get_base_url(api_url)
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        resolution = kwargs.get("resolution", "2K")

        if is_gemini_25(model):
            url = f"{base}/v1beta/models/gemini-2.5-flash-image:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            }
            return url, payload, False

        if is_gemini_3_pro(model):
            url = f"{base}/v1beta/models/gemini-3-pro-image-preview:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": resolution,
                    },
                },
            }
            return url, payload, False
        
        raise ValueError(f"Unsupported Gemini model for APIYI Generation: {model}")

    def build_edit_request(self, api_url: str, model: str, prompt: str, image_b64: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = self._get_base_url(api_url)
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        resolution = kwargs.get("resolution", "2K")
        fmt = kwargs.get("image_fmt", "png")

        if is_gemini_25(model) and aspect_ratio != "1:1":
             url = f"{base}/v1beta/models/gemini-2.5-flash-image:generateContent"
             payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": f"image/{fmt}", "data": image_b64}}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            }
             return url, payload, False

        if is_gemini_3_pro(model):
            url = f"{base}/v1beta/models/gemini-3-pro-image-preview:generateContent"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": f"image/{fmt}", "data": image_b64}}
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
            return url, payload, False
            
        raise ValueError(f"Unsupported Gemini Edit combination for APIYI: {model}")

    def build_multi_image_edit_request(
        self,
        api_url: str,
        model: str,
        prompt: str,
        image_b64_list: List[Tuple[str, str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        base = self._get_base_url(api_url)
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        resolution = kwargs.get("resolution", "2K")

        parts = [{"text": prompt}]
        for b64, fmt in image_b64_list:
            parts.append({
                "inline_data": {
                    "mime_type": f"image/{fmt}",
                    "data": b64
                }
            })

        url = f"{base}/v1beta/models/{model}:generateContent"
        
        image_config = {"aspectRatio": aspect_ratio}
        if is_gemini_3_pro(model):
            image_config["imageSize"] = resolution
            
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": image_config
            }
        }
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("candidates is empty")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            inline_data = parts[0].get("inlineData", {})
            return inline_data.get("data")
        except Exception as e:
            log.error(f"Failed to parse APIYI Gemini response: {e}")
            log.error(f"Response preview: {str(data)[:500]}")
            raise

    def build_tts_request(self, api_url: str, model: str, text: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = self._get_base_url(api_url)
        url = f"{base}/v1beta/models/{model}:generateContent"
        
        voice_name = kwargs.get("voice_name", "Kore")
        
        payload = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name
                        }
                    }
                }
            }
        }
        return url, payload, False

    def parse_tts_response(self, data: Dict[str, Any]) -> bytes:
        if "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
            
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in response: {str(data)[:200]}")
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise RuntimeError("No parts in content")
            
        inline_data = parts[0].get("inlineData", {})
        b64 = inline_data.get("data")
        
        if not b64:
             raise RuntimeError("No inlineData.data found")
             
        import base64
        return base64.b64decode(b64)


class Local123GeminiProvider(AIProviderStrategy):
    """
    Local 123 服务商针对 Gemini 模型的特殊处理
    """
    def match(self, api_url: str, model: str) -> bool:
        return detect_provider(api_url) is Provider.LOCAL_123 and is_gemini_model(model)

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = api_url.rstrip("/")
        aspect_ratio = kwargs.get("aspect_ratio", "")
        resolution = kwargs.get("resolution", "2K")

        # Logic from original req_img.py
        if aspect_ratio:
            prompt = f"{prompt} 生成比例：{aspect_ratio}, 4K 分辨率"

        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "group": "default",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "generationConfig": {
                "imageConfig": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": resolution
                }
            }
        }
        return url, payload, True

    def build_edit_request(self, api_url: str, model: str, prompt: str, image_b64: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = api_url.rstrip("/")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        resolution = kwargs.get("resolution", "2K")
        fmt = kwargs.get("image_fmt", "png")

        if is_gemini_3_pro(model):
            url = f"{base}/chat/completions"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{fmt};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ]
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "generationConfig": {
                    "imageConfig": {
                        "aspect_ratio": aspect_ratio, 
                        "image_size": resolution
                    }
                }
            }
            return url, payload, True

        if is_gemini_25(model):
            url = f"{base}/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": f"image/{fmt}", "data": image_b64}},
                        ],
                    }
                ],
                "generationConfig": {
                    "width": 1920,
                    "height": 1080,
                    "quality": "high",
                },
            }
            return url, payload, False

        raise ValueError(f"Unsupported Gemini Edit model for Local123: {model}")

    def build_multi_image_edit_request(
        self,
        api_url: str,
        model: str,
        prompt: str,
        image_b64_list: List[Tuple[str, str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        base = api_url.rstrip("/")
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        resolution = kwargs.get("resolution", "2K")
        
        content_parts = [{"type": "text", "text": prompt}]
        for b64, fmt in image_b64_list:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{fmt};base64,{b64}"
                }
            })

        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "stream": True,
            "temperature": 0.7,
            "generationConfig": {
                "imageConfig": {
                    "aspect_ratio": aspect_ratio, 
                    "image_size": resolution
                }
            }
        }
        return url, payload, True

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        # Local 123 returns OpenAI-like format
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                b64 = extract_base64(content)
            elif isinstance(content, list):
                joined = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
                b64 = extract_base64(joined)
            else:
                raise RuntimeError(f"Unsupported content type: {type(content)}")
            
            if not b64:
                raise RuntimeError("Failed to extract base64 from Local123 response")
            return b64
        raise RuntimeError("Unknown Local123 response structure")


class ApiYiSeeDreamProvider(AIProviderStrategy):
    """
    APIYI SeeDream 系列模型支持 (兼容 OpenAI Image API)
    """
    def match(self, api_url: str, model: str) -> bool:
        return model.lower().startswith("seedream")

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/images/generations"
        
        size = kwargs.get("size", "2048x2048")
        quality = kwargs.get("quality", "standard")
        response_format = kwargs.get("response_format", "b64_json")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": response_format,
        }
        
        # 合并额外参数 (如 output_format)
        for k, v in kwargs.items():
            if k not in payload and k not in ["api_key", "timeout"]:
                payload[k] = v
                
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "b64_json" in item:
                return item["b64_json"]
            if "url" in item:
                return item["url"]
        raise RuntimeError(f"Failed to parse SeeDream response: {str(data)[:200]}")


class ApiYiGPTImageProvider(AIProviderStrategy):
    """
    APIYI GPT-Image 系列模型支持 (兼容 OpenAI Image API)
    """
    def match(self, api_url: str, model: str) -> bool:
        return model.lower().startswith("gpt-image")

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/images/generations"
        
        size = kwargs.get("size", "1024x1024")
        # 映射 quality 参数: DALL-E 的 standard/hd -> GPT-Image 的 low/medium/high/auto
        quality = kwargs.get("quality", "auto")
        if quality == "standard":
            quality = "medium"
        elif quality == "hd":
            quality = "high"
            
        payload = {
            "model": model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": size,
            "quality": quality,
        }
        
        # 白名单过滤：仅传递 GPT-Image 文档支持的参数
        # 移除 style, aspect_ratio, resolution 等不支持的参数
        # 移除 response_format (API 报错不支持)
        supported_params = [
            "output_format", 
            "output_compression", 
            "background", 
            "user"
        ]
        
        for k in supported_params:
            if k in kwargs:
                payload[k] = kwargs[k]
                
        return url, payload, False

    def build_edit_request(
        self, 
        api_url: str, 
        model: str, 
        prompt: str, 
        image_b64: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        构造 APIYI GPT-Image 系列模型的图像编辑请求 (Multipart 格式)
        
        参数:
            api_url (str): API 基础地址
            model (str): 模型名称 (如 gpt-image-1)
            prompt (str): 文本提示词，描述想要生成的编辑效果
            image_b64 (str): 原始图像的 Base64 编码字符串
            **kwargs: 其他可选参数
                - mask_path (str): 遮罩图像的文件路径 (如果存在)
                - n (int): 生成图像数量，默认为 1
                - size (str): 输出图像尺寸 (如 1024x1024)
                - response_format (str): 返回格式 (url 或 b64_json)，注意 GPT-Image-1 可能不支持此参数
                - user (str): 用户标识符
        
        返回:
            Tuple[str, Dict[str, Any], bool]: (请求URL, 请求载荷, 是否流式)
            
        注意:
            返回的 payload 包含特殊标记 "__is_multipart__": True。
            "files": 包含 'image' 和可选的 'mask' 文件数据 (bytes)。
            "data": 包含其他表单字段 (prompt, n, size 等)。
        """
        import base64
        import os
        
        url = f"{api_url.rstrip('/')}/images/edits"
        
        # 1. 解码图片 Base64 为二进制
        image_bytes = base64.b64decode(image_b64)
        
        files = {
            "image": ("image.png", image_bytes, "image/png")
        }
        
        # 2. 处理遮罩 (Mask)
        mask_path = kwargs.get("mask_path")
        if mask_path and os.path.exists(mask_path):
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()
            files["mask"] = (os.path.basename(mask_path), mask_bytes, "image/png")
            
        # 3. 构造表单数据 (Data)
        data = {
            "model": model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
        }
        
        # 添加可选参数 (白名单过滤)
        supported_params = ["response_format", "user"] # 尽管 Generation 不支持 response_format，但 Edit 标准通常支持，保留以防万一或稍后测试
        # 如果 GPT-Image Edit 同样不支持 response_format，稍后也应移除。
        # 安全起见，为了和 Generation 保持一致，这里暂时不包含 response_format，除非文档明确说 Edit 支持。
        # 文档确实提到了 response_format 参数在 Edit API 中。
        # 但鉴于 Generation 报错，我们先尝试不传，或仅在 kwargs 明确有的时候传。
        
        if "user" in kwargs:
            data["user"] = kwargs["user"]
            
        # 构造特殊返回 Payload
        payload = {
            "__is_multipart__": True,
            "files": files,
            "data": data
        }
        
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "b64_json" in item:
                return item["b64_json"]
            if "url" in item:
                return item["url"]
        raise RuntimeError(f"Failed to parse GPT-Image response: {str(data)[:200]}")


class OpenAIDalleProvider(AIProviderStrategy):
    """
    OpenAI DALL-E 系列 (images/generations)
    注意：DALL-E 仅支持生成，不支持理解/Chat
    """
    def match(self, api_url: str, model: str) -> bool:
        return model.lower().startswith(('dall-e', 'dall-e-2', 'dall-e-3'))

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/images/generations"
        
        size = kwargs.get("size", "1024x1024")
        quality = kwargs.get("quality", "standard")
        style = kwargs.get("style", "vivid")
        response_format = kwargs.get("response_format", "b64_json")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": response_format,
        }
        
        if model.lower() == "dall-e-3":
            payload["quality"] = quality
            payload["style"] = style
            
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        if "data" in data and len(data["data"]) > 0:
            if "b64_json" in data["data"][0]:
                return data["data"][0]["b64_json"]
        raise RuntimeError("Failed to parse DALL-E response")


class OpenAITTSProvider(AIProviderStrategy):
    """
    OpenAI TTS (/audio/speech)
    """
    _OPENAI_VOICES = {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad",
        "ash", "sage", "marin", "cedar", "amuch", "aster", "brook", "clover", "dan",
        "elan", "marilyn", "meadow", "jazz", "rio", "breeze", "cove", "ember", "fathom",
        "glimmer", "harp", "juniper", "maple", "orbit", "vale",
        "megan-wetherall", "jade-hardy",
        "megan-wetherall-2025-03-07", "jade-hardy-2025-03-07",
    }
    _FALLBACK_VOICE = "alloy"

    def match(self, api_url: str, model: str) -> bool:
        model_l = model.lower()
        return model_l.startswith("gpt-4o-mini-tts") or model_l in {"tts-1", "tts-1-hd"}

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        raise NotImplementedError("Generation not supported by OpenAITTSProvider")

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError("Generation not supported by OpenAITTSProvider")

    def build_tts_request(self, api_url: str, model: str, text: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/audio/speech"
        voice_raw = kwargs.get("voice_name") or kwargs.get("voice") or self._FALLBACK_VOICE
        voice = str(voice_raw).strip().lower()
        if voice not in self._OPENAI_VOICES:
            log.warning(f"OpenAI TTS voice '{voice_raw}' not supported, fallback to '{self._FALLBACK_VOICE}'")
            voice = self._FALLBACK_VOICE
        response_format = kwargs.get("response_format", "pcm")
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        # Optional OpenAI params
        if "speed" in kwargs:
            payload["speed"] = kwargs["speed"]
        if "instructions" in kwargs:
            payload["instructions"] = kwargs["instructions"]

        payload["__response_type__"] = "binary"
        return url, payload, False

    def parse_tts_response(self, data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
        raise RuntimeError("Unexpected TTS response format")


class OpenAICompatGeminiProvider(AIProviderStrategy):
    """
    通用 OpenAI 兼容格式
    生图：chat/completions (image response)
    理解：chat/completions (text response)
    """
    def match(self, api_url: str, model: str) -> bool:
        # Always True as fallback if no others match
        return True

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        return url, payload, False

    def build_edit_request(self, api_url: str, model: str, prompt: str, image_b64: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/chat/completions"
        fmt = kwargs.get("image_fmt", "png")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{fmt};base64,{image_b64}",
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
        return url, payload, False

    def build_multi_image_edit_request(
        self,
        api_url: str,
        model: str,
        prompt: str,
        image_b64_list: List[Tuple[str, str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        base = api_url.rstrip("/")
        
        content_parts = [{"type": "text", "text": prompt}]
        for b64, fmt in image_b64_list:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{fmt};base64,{b64}"
                }
            })
            
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                b64 = extract_base64(content)
            elif isinstance(content, list):
                joined = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
                b64 = extract_base64(joined)
            else:
                raise RuntimeError(f"Unsupported content type: {type(content)}")
            
            if not b64:
                raise RuntimeError("Failed to extract base64 from OpenAI-compat response")
            return b64
        raise RuntimeError("Unknown OpenAI-compat response structure")


class GoogleNativeProvider(AIProviderStrategy):
    """
    Google 官方 Gemini API 
    """
    def match(self, api_url: str, model: str) -> bool:
        return "googleapis.com" in api_url and is_gemini_model(model)

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        base = api_url.rstrip("/")
        if "v1" not in base and "v1beta" not in base:
            base = f"{base}/v1beta"
        url = f"{base}/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        return url, payload, False

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["inlineData"]["data"]
        raise RuntimeError("No candidates in Google response")

    # Native Chat Implementation can also be added here (e.g., converting messages to contents)


class ComflyProvider(AIProviderStrategy):
    """
    Comfly AI Provider
    图生图：https://ai.comfly.chat/v1/images/edits
    文生图：https://ai.comfly.chat/v1/images/generations
    修改图片/文本处理：https://ai.comfly.chat/v1/chat/completions

    Model Mapping:
    - gemini-2.5-flash-image -> nano-banana
    - gemini-3-pro-image-preview -> nano-banana-2-2k
    """

    def match(self, api_url: str, model: str) -> bool:
        """Match if the URL contains comfly.chat"""
        return "comfly.chat" in api_url.lower()

    def _translate_model_name(self, model: str) -> str:
        """
        Translate Gemini model names to Comfly-specific model names
        """
        model_mapping = {
            "gemini-2.5-flash-image": "nano-banana",
            "gemini-3-pro-image-preview": "nano-banana-2-2k",
        }
        translated = model_mapping.get(model, model)
        if translated != model:
            log.info(f"[ComflyProvider] Translated model: {model} -> {translated}")
        return translated

    def _aspect_ratio_to_size(self, aspect_ratio: str, resolution: str = "2K") -> str:
        """
        Convert aspect_ratio (e.g., "16:9") to OpenAI-style size (e.g., "1920x1080")

        Args:
            aspect_ratio: Aspect ratio string like "16:9", "1:1", "9:16"
            resolution: Resolution hint like "1K", "2K", "4K" (default: "2K")

        Returns:
            Size string like "1920x1080"
        """
        # Resolution base dimensions
        resolution_map = {
            "1K": 1024,
            "2K": 2048,
            "4K": 4096,
        }
        base = resolution_map.get(resolution, 2048)

        # Parse aspect ratio
        if ":" in aspect_ratio:
            try:
                w_ratio, h_ratio = map(int, aspect_ratio.split(":"))

                # Calculate dimensions based on aspect ratio
                if w_ratio > h_ratio:
                    # Landscape (e.g., 16:9)
                    width = base
                    height = int(base * h_ratio / w_ratio)
                elif w_ratio < h_ratio:
                    # Portrait (e.g., 9:16)
                    height = base
                    width = int(base * w_ratio / h_ratio)
                else:
                    # Square (1:1)
                    width = height = base

                return f"{width}x{height}"
            except (ValueError, ZeroDivisionError):
                log.warning(f"Invalid aspect_ratio: {aspect_ratio}, using default 1024x1024")
                return "1024x1024"

        return "1024x1024"

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        """Build text-to-image generation request"""
        url = f"{api_url.rstrip('/')}/images/generations"

        # Translate model name for Comfly API
        translated_model = self._translate_model_name(model)

        # Handle aspect_ratio conversion to size
        aspect_ratio = kwargs.get("aspect_ratio", "")
        resolution = kwargs.get("resolution", "2K")

        if aspect_ratio:
            # Convert aspect_ratio to size format
            size = self._aspect_ratio_to_size(aspect_ratio, resolution)
            log.info(f"[ComflyProvider] Converted aspect_ratio {aspect_ratio} ({resolution}) to size {size}")
        else:
            # Use explicit size if provided, otherwise default
            size = kwargs.get("size", "1024x1024")

        quality = kwargs.get("quality", "standard")
        response_format = kwargs.get("response_format", "b64_json")

        payload = {
            "model": translated_model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": response_format,
        }

        return url, payload, False

    def build_edit_request(
        self,
        api_url: str,
        model: str,
        prompt: str,
        image_b64: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Build image-to-image edit request (multipart format)"""
        import base64
        import os

        url = f"{api_url.rstrip('/')}/images/edits"

        # Translate model name for Comfly API
        translated_model = self._translate_model_name(model)

        # Decode base64 image to binary
        image_bytes = base64.b64decode(image_b64)

        files = {
            "image": ("image.png", image_bytes, "image/png")
        }

        # Handle mask if provided
        mask_path = kwargs.get("mask_path")
        if mask_path and os.path.exists(mask_path):
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()
            files["mask"] = (os.path.basename(mask_path), mask_bytes, "image/png")

        # Handle aspect_ratio vs size
        aspect_ratio = kwargs.get("aspect_ratio", "")
        resolution = kwargs.get("resolution", "2K")

        data = {
            "model": translated_model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
        }
        if aspect_ratio:
            # Prefer Comfly-native fields when aspect_ratio is provided
            data["aspect_ratio"] = aspect_ratio
            data["image_size"] = resolution
        else:
            # Use explicit size if provided, otherwise default
            data["size"] = kwargs.get("size", "1024x1024")

        # Construct multipart payload
        payload = {
            "__is_multipart__": True,
            "files": files,
            "data": data
        }

        return url, payload, False

    def _fix_base64_padding(self, b64_string: str) -> str:
        """
        Fix Base64 padding and remove Data URL prefix if present

        Handles formats like: data:image/png;base64,<base64data>
        """
        # Remove any whitespace
        b64_string = b64_string.strip()

        # Check for Data URL format: data:image/png;base64,<base64data>
        if b64_string.startswith('data:'):
            comma_index = b64_string.find(',')
            if comma_index != -1:
                b64_string = b64_string[comma_index + 1:]
                log.info(f"[ComflyProvider] Removed Data URL prefix")

        # Calculate padding needed
        padding_needed = (4 - len(b64_string) % 4) % 4

        if padding_needed > 0:
            b64_string += '=' * padding_needed
            log.info(f"[ComflyProvider] Added {padding_needed} padding characters to Base64 string")

        return b64_string

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        """Parse generation response and return base64 image"""
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]

            # Debug: log response structure
            log.info(f"[ComflyProvider] Response item keys: {list(item.keys())}")

            if "b64_json" in item:
                b64_string = item["b64_json"]
                log.info(f"[ComflyProvider] b64_json length: {len(b64_string)}, first 100 chars: {b64_string[:100]}")
                # Fix Base64 padding if needed
                return self._fix_base64_padding(b64_string)
            if "url" in item:
                log.info(f"[ComflyProvider] Returning URL: {item['url']}")
                return item["url"]
        raise RuntimeError(f"Failed to parse Comfly response: {str(data)[:200]}")

# 注册顺序
STRATEGIES = [
    ApiYiGeminiProvider(),
    ApiYiSeeDreamProvider(),
    ApiYiGPTImageProvider(),
    Local123GeminiProvider(),
    OpenAIDalleProvider(),
    OpenAITTSProvider(),
    ComflyProvider(),
    # Add GoogleNativeProvider() here if needed
    OpenAICompatGeminiProvider(), # Default Fallback
]

def get_provider(api_url: str, model: str) -> AIProviderStrategy:
    for strategy in STRATEGIES:
        if strategy.match(api_url, model):
            return strategy
    return OpenAICompatGeminiProvider()


# --- Text-only LLM client (rebuttal / chat workflows) ---
# URL 与 API key 均由前端传入，不再使用本地 PROVIDER_CONFIGS。


class TokenUsageTracker:
    """Tracks token usage across LLM calls; can export to file and print summary."""
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.usage_records: List[Dict] = []
        self.total_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0,
        }
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    def add_record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        agent_name: str = "unknown",
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "agent_name": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.usage_records.append(record)
        self.total_stats["total_prompt_tokens"] += prompt_tokens
        self.total_stats["total_completion_tokens"] += completion_tokens
        self.total_stats["total_tokens"] += total_tokens
        self.total_stats["total_calls"] += 1

    def export_to_file(self, file_path: Optional[str] = None) -> Optional[str]:
        output_file = file_path or self.log_file
        if not output_file:
            return None
        export_data = {
            "export_time": datetime.now().isoformat(),
            "summary": self.total_stats,
            "records": self.usage_records,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        log.info(f"Token usage statistics exported to: {output_file}")
        return output_file

    def print_summary(self) -> None:
        summary = (
            "\n"
            + "=" * 60
            + "\nToken Usage Summary\n"
            + "=" * 60
            + f"\nTotal API calls: {self.total_stats['total_calls']}"
            + f"\nTotal input tokens: {self.total_stats['total_prompt_tokens']:,}"
            + f"\nTotal output tokens: {self.total_stats['total_completion_tokens']:,}"
            + f"\nTotal tokens: {self.total_stats['total_tokens']:,}"
            + "\n"
            + "=" * 60
            + "\n"
        )
        log.info(summary)


class LLMClient:
    """
    Text-only LLM client for rebuttal and other chat workflows.
    URL 与 API key 均由调用方（如前端）传入；支持 OpenAI 兼容 API 与 Gemini 原生 SDK。
    """
    def __init__(
        self,
        api_key: str,
        provider: str = "openrouter",
        base_url: Optional[str] = None,
        default_model: str = "google/gemini-3-flash-preview",
        request_timeout: int = 600,
        token_tracker: Optional[TokenUsageTracker] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.provider = provider.lower()
        self.default_model = default_model
        self.request_timeout = request_timeout
        self.token_tracker = token_tracker
        self.current_agent_name = "unknown"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = None
            self._http_client = None
        else:
            if not base_url:
                raise ValueError("base_url is required for non-Gemini providers (URL and key are passed from frontend).")
            import httpx
            from openai import OpenAI
            self._genai = None
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": site_url or "http://localhost",
                    "X-Title": site_name or "Rebuttal Assistant",
                }
            self._http_client = httpx.Client(
                trust_env=True,
                timeout=request_timeout,
                headers=extra_headers,
            )
            self._client = OpenAI(
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                http_client=self._http_client,
            )

    def _log_output(self, model_name: str, final_text: str, reasoning_text: str = "") -> None:
        if reasoning_text:
            message = (
                "[LLM Output]\n"
                f"api_key={self.api_key}\n"
                f"provider={self.provider}\n"
                f"model={model_name}\n"
                f"agent={self.current_agent_name}\n"
                f"output=\n{final_text}\n"
                f"reasoning=\n{reasoning_text}"
            )
        else:
            message = (
                "[LLM Output]\n"
                f"api_key={self.api_key}\n"
                f"provider={self.provider}\n"
                f"model={model_name}\n"
                f"agent={self.current_agent_name}\n"
                f"output=\n{final_text}"
            )
        log.info(message)

    def generate(
        self,
        instructions: Optional[str],
        input_text: str,
        model: Optional[str] = None,
        enable_reasoning: bool = True,
        temperature: float = 0.6,
        agent_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        model_name = model or self.default_model
        if agent_name:
            self.current_agent_name = agent_name
        final_text = ""
        reasoning_text = ""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "gemini":
                    final_text, reasoning_text = self._generate_gemini(
                        instructions, input_text, model_name, temperature
                    )
                else:
                    final_text, reasoning_text = self._generate_openai_compatible(
                        instructions, input_text, model_name, temperature
                    )
                rate_limit_keywords = ["并发", "rate limit", "too many requests", "quota exceeded", "限流"]
                if any(kw in (final_text or "").lower() for kw in rate_limit_keywords):
                    raise RuntimeError(f"Rate limit detected in response: {(final_text or '')[:100]}...")
                self._log_output(model_name, final_text or "", reasoning_text or "")
                return final_text or "", reasoning_text or ""
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    log.warning(
                        "[Retry]\n"
                        f"api_key={self.api_key}\n"
                        f"provider={self.provider}\n"
                        f"model={model_name}\n"
                        f"agent={self.current_agent_name}\n"
                        f"attempt={attempt + 1}/{self.max_retries}\n"
                        f"error={type(e).__name__}: {e}\n"
                        f"waiting={wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    log.error(
                        "[LLM Error]\n"
                        f"api_key={self.api_key}\n"
                        f"provider={self.provider}\n"
                        f"model={model_name}\n"
                        f"agent={self.current_agent_name}\n"
                        f"attempts={self.max_retries + 1}\n"
                        f"error={type(e).__name__}: {e}"
                    )
        return f"Error calling {self.provider} after {self.max_retries + 1} attempts: {str(last_error)}", ""

    def _generate_gemini(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions or "You are a helpful AI assistant.",
            generation_config={"temperature": temperature},
        )
        response = model.generate_content(input_text)
        final_text = response.text or ""
        if self.token_tracker and hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", 0)
            self.token_tracker.add_record(
                provider="gemini",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                agent_name=self.current_agent_name,
            )
            log.info(
                "[Token]\n"
                f"api_key={self.api_key}\n"
                f"provider=gemini\n"
                f"model={model_name}\n"
                f"agent={self.current_agent_name}\n"
                f"in={prompt_tokens}\n"
                f"out={completion_tokens}\n"
                f"total={total_tokens}"
            )
        return final_text, ""

    def _generate_openai_compatible(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        messages = [
            {"role": "system", "content": (instructions or "You are a helpful AI assistant.")},
            {"role": "user", "content": input_text},
        ]
        response = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        final_text = ""
        if getattr(response, "choices", None):
            choice0 = response.choices[0]
            message = getattr(choice0, "message", None)
            if message is not None:
                final_text = getattr(message, "content", None) or ""
        if self.token_tracker and hasattr(response, "usage"):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            self.token_tracker.add_record(
                provider=self.provider,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                agent_name=self.current_agent_name,
            )
            log.info(
                "[Token]\n"
                f"api_key={self.api_key}\n"
                f"provider={self.provider}\n"
                f"model={model_name}\n"
                f"agent={self.current_agent_name}\n"
                f"in={prompt_tokens}\n"
                f"out={completion_tokens}\n"
                f"total={total_tokens}"
            )
        return final_text, ""
