import json
from abc import ABC, abstractmethod
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


# 注册顺序
STRATEGIES = [
    ApiYiGeminiProvider(),
    ApiYiSeeDreamProvider(),
    ApiYiGPTImageProvider(),
    Local123GeminiProvider(),
    OpenAIDalleProvider(),
    # Add GoogleNativeProvider() here if needed
    OpenAICompatGeminiProvider(), # Default Fallback
]

def get_provider(api_url: str, model: str) -> AIProviderStrategy:
    for strategy in STRATEGIES:
        if strategy.match(api_url, model):
            return strategy
    return OpenAICompatGeminiProvider()
