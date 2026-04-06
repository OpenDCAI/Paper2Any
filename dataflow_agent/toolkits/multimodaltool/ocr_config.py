from __future__ import annotations

import os
from typing import Tuple


def get_ocr_api_credentials() -> Tuple[str, str]:
    """
    Return unified OCR endpoint credentials.

    Priority:
    1) Environment (`PAPER2DRAWIO_OCR_API_URL/_KEY`)
    2) `fastapi_app.config.settings` compatibility fallback

    If still empty, fail fast with explicit error.
    """
    api_url = (os.getenv("PAPER2DRAWIO_OCR_API_URL") or "").strip()
    api_key = (os.getenv("PAPER2DRAWIO_OCR_API_KEY") or "").strip()

    try:
        from fastapi_app.config.settings import settings

        settings_url = (getattr(settings, "PAPER2DRAWIO_OCR_API_URL", "") or "").strip()
        settings_key = (getattr(settings, "PAPER2DRAWIO_OCR_API_KEY", "") or "").strip()
        if settings_url:
            api_url = settings_url
        if settings_key:
            api_key = settings_key
    except Exception:
        pass

    if not api_url or not api_key:
        raise ValueError(
            "OCR endpoint/key not configured. "
            "Please set PAPER2DRAWIO_OCR_API_URL and PAPER2DRAWIO_OCR_API_KEY in fastapi_app/.env."
        )

    return api_url, api_key
