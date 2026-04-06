from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any


def _request_get(request: Any, key: str) -> Any:
    if isinstance(request, Mapping):
        return request.get(key)
    return getattr(request, key, None)


def get_request_image_api_url(request: Any) -> str:
    value = _request_get(request, "image_api_url") or _request_get(request, "chat_api_url") or ""
    return str(value).strip()


def get_request_image_api_key(request: Any) -> str:
    value = (
        _request_get(request, "image_api_key")
        or _request_get(request, "chat_api_key")
        or _request_get(request, "api_key")
        or os.getenv("DF_IMAGE_API_KEY")
        or os.getenv("DF_API_KEY")
        or ""
    )
    return str(value).strip()


def get_request_text_api_url(request: Any) -> str:
    value = _request_get(request, "chat_api_url") or ""
    return str(value).strip()


def get_request_text_api_key(request: Any) -> str:
    value = (
        _request_get(request, "chat_api_key")
        or _request_get(request, "api_key")
        or os.getenv("DF_API_KEY")
        or ""
    )
    return str(value).strip()
