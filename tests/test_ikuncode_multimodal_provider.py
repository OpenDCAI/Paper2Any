from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.toolkits.multimodaltool.providers import (  # noqa: E402
    ApiYiGPTImageAllProvider,
    ApiYiGPTImageProvider,
    IkunCodeGeminiProvider,
    get_provider,
)
from dataflow_agent.toolkits.multimodaltool.utils import Provider, detect_provider  # noqa: E402


def test_detect_provider_identifies_ikuncode() -> None:
    assert detect_provider("https://api.ikuncode.cc") is Provider.IKUNCODE
    assert detect_provider("https://api.ikuncode.cc/v1beta") is Provider.IKUNCODE


def test_get_provider_returns_ikuncode_strategy_for_supported_image_models() -> None:
    provider = get_provider("https://api.ikuncode.cc", "gemini-3-pro-image-preview")
    assert isinstance(provider, IkunCodeGeminiProvider)


def test_ikuncode_generation_payload_uses_documented_fields() -> None:
    provider = IkunCodeGeminiProvider()
    url, payload, is_stream = provider.build_generation_request(
        api_url="https://api.ikuncode.cc",
        model="gemini-3-pro-image-preview",
        prompt="test prompt",
        aspect_ratio="21:9",
        resolution="4K",
    )
    assert not is_stream
    assert url == "https://api.ikuncode.cc/v1beta/models/gemini-3-pro-image-preview:generateContent"
    image_config = payload["generationConfig"]["imageConfig"]
    assert image_config["aspectRatio"] == "21:9"
    assert image_config["image_size"] == "4K"
    assert "imageSize" not in image_config


def test_ikuncode_edit_payload_uses_inline_data_schema() -> None:
    provider = IkunCodeGeminiProvider()
    _, payload, _ = provider.build_edit_request(
        api_url="https://api.ikuncode.cc/v1beta",
        model="gemini-3.1-flash-image-preview",
        prompt="edit prompt",
        image_b64="ZmFrZQ==",
        image_fmt="png",
        aspect_ratio="4:5",
        resolution="2K",
    )
    parts = payload["contents"][0]["parts"]
    assert "inlineData" in parts[0]
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    assert parts[0]["inlineData"]["data"] == "ZmFrZQ=="
    assert parts[1]["text"] == "edit prompt"
    assert payload["generationConfig"]["imageConfig"]["aspectRatio"] == "4:5"
    assert payload["generationConfig"]["imageConfig"]["image_size"] == "2K"


def test_ikuncode_parse_generation_response_handles_text_then_image_parts() -> None:
    provider = IkunCodeGeminiProvider()
    data = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Here is your edited slide."},
                        {"inlineData": {"mimeType": "image/png", "data": "ZmFrZS1pbWFnZQ=="}},
                    ],
                }
            }
        ]
    }
    assert provider.parse_generation_response(data) == "ZmFrZS1pbWFnZQ=="


def test_get_provider_returns_apiyi_gpt_image_all_strategy_for_special_model() -> None:
    provider = get_provider("https://api.apiyi.com/v1", "gpt-image-2-all")
    assert isinstance(provider, ApiYiGPTImageAllProvider)


def test_apiyi_gpt_image_all_payload_omits_unsupported_generation_fields() -> None:
    provider = ApiYiGPTImageAllProvider()
    url, payload, is_stream = provider.build_generation_request(
        api_url="https://api.apiyi.com/v1",
        model="gpt-image-2-all",
        prompt="test prompt",
        size="2048x1152",
        quality="medium",
        n=2,
    )
    assert not is_stream
    assert url == "https://api.apiyi.com/v1/images/generations"
    assert payload == {
        "model": "gpt-image-2-all",
        "prompt": "test prompt",
        "response_format": "b64_json",
    }


def test_apiyi_gpt_image_all_parser_strips_data_url_prefix() -> None:
    provider = ApiYiGPTImageAllProvider()
    data = {
        "data": [
            {
                "b64_json": "data:image/png;base64,ZmFrZS1pbWFnZQ==",
            }
        ]
    }
    assert provider.parse_generation_response(data) == "ZmFrZS1pbWFnZQ=="


def test_apiyi_gpt_image_2_payload_keeps_supported_fields() -> None:
    provider = ApiYiGPTImageProvider()
    _, payload, _ = provider.build_generation_request(
        api_url="https://api.apiyi.com/v1",
        model="gpt-image-2",
        prompt="test prompt",
        size="2048x1152",
        quality="medium",
        output_format="png",
    )
    assert payload["model"] == "gpt-image-2"
    assert payload["size"] == "2048x1152"
    assert payload["quality"] == "medium"
    assert payload["output_format"] == "png"
