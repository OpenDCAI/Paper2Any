from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .contracts import EditablePPTInputRunRequest, EditablePPTRunRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "fastapi_app" / ".env"


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_manually(env_file: Path) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_env_value(value))


def load_runtime_env() -> None:
    """Load fastapi_app/.env for local Agent runtime calls without exposing values."""
    if not ENV_FILE.is_file():
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=False)
        return

    _load_env_manually(ENV_FILE)


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _setting(name: str, default: str = "") -> str:
    try:
        from fastapi_app.config import settings
    except Exception:
        return default
    return str(getattr(settings, name, default) or "").strip()


def _text_api_url(explicit: Optional[str]) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_TEXT_API_URL"),
        os.getenv("PAPER2PPT_MANAGED_API_URL"),
        os.getenv("DF_API_URL"),
        _setting("SIMPLE_TEXT_API_URL"),
        _setting("PAPER2PPT_MANAGED_API_URL"),
        _setting("DF_API_URL"),
        _setting("DEFAULT_LLM_API_URL"),
        "https://api.openai.com/v1",
    )


def _text_api_key(explicit: Optional[str]) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_TEXT_API_KEY"),
        os.getenv("PAPER2PPT_MANAGED_API_KEY"),
        os.getenv("DF_API_KEY"),
        _setting("SIMPLE_TEXT_API_KEY"),
        _setting("PAPER2PPT_MANAGED_API_KEY"),
        _setting("DF_API_KEY"),
    )


def _text_model(explicit: Optional[str], *, role: str = "content", fallback: Optional[str] = None) -> str:
    role_setting = "PAPER2PPT_OUTLINE_MODEL" if role == "outline" else "PAPER2PPT_CONTENT_MODEL"
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_TEXT_MODEL"),
        os.getenv(role_setting),
        os.getenv("PAPER2PPT_DEFAULT_MODEL"),
        _setting("SIMPLE_TEXT_MODEL"),
        _setting(role_setting),
        _setting("PAPER2PPT_DEFAULT_MODEL"),
        fallback,
        "gpt-4o",
    )


def _image_api_url(explicit: Optional[str], *, fallback_url: str) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_IMAGE_API_URL"),
        os.getenv("PAPER2PPT_MANAGED_IMAGE_API_URL"),
        os.getenv("DF_IMAGE_API_URL"),
        _setting("SIMPLE_IMAGE_API_URL"),
        _setting("PAPER2PPT_MANAGED_IMAGE_API_URL"),
        _setting("DF_IMAGE_API_URL"),
        fallback_url,
    )


def _image_api_key(explicit: Optional[str], *, fallback_key: str) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_IMAGE_API_KEY"),
        os.getenv("PAPER2PPT_MANAGED_IMAGE_API_KEY"),
        os.getenv("DF_IMAGE_API_KEY"),
        _setting("SIMPLE_IMAGE_API_KEY"),
        _setting("PAPER2PPT_MANAGED_IMAGE_API_KEY"),
        _setting("DF_IMAGE_API_KEY"),
        fallback_key,
    )


def _image_model(explicit: Optional[str]) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_IMAGE_MODEL"),
        os.getenv("PAPER2PPT_IMAGE_GEN_MODEL"),
        os.getenv("PAPER2PPT_DEFAULT_IMAGE_MODEL"),
        _setting("SIMPLE_IMAGE_MODEL"),
        _setting("PAPER2PPT_IMAGE_GEN_MODEL"),
        _setting("PAPER2PPT_DEFAULT_IMAGE_MODEL"),
    )


def _vlm_model(explicit: Optional[str], *, fallback_model: str) -> str:
    return _first_non_empty(
        explicit,
        os.getenv("SIMPLE_VLM_MODEL"),
        os.getenv("PAPER2PPT_VLM_MODEL"),
        _setting("SIMPLE_VLM_MODEL"),
        _setting("PAPER2PPT_VLM_MODEL"),
        fallback_model,
    )


def resolve_runtime_request(req: EditablePPTRunRequest) -> EditablePPTRunRequest:
    """Fill empty Agent runtime credentials/models from fastapi_app/.env."""
    load_runtime_env()
    api_url = _text_api_url(req.api_url)
    api_key = _text_api_key(req.api_key)
    model = _text_model(req.model, role="content")
    image_api_url = _image_api_url(req.image_api_url, fallback_url=api_url)
    image_api_key = _image_api_key(req.image_api_key, fallback_key=api_key)

    return req.model_copy(
        update={
            "api_url": api_url,
            "api_key": api_key,
            "model": model,
            "vlm_api_url": _first_non_empty(req.vlm_api_url, api_url),
            "vlm_api_key": _first_non_empty(req.vlm_api_key, api_key),
            "vlm_model": _vlm_model(req.vlm_model, fallback_model=model),
            "image_api_url": image_api_url,
            "image_api_key": image_api_key,
            "image_model": _image_model(req.image_model),
        }
    )


def resolve_runtime_input_request(req: EditablePPTInputRunRequest) -> EditablePPTInputRunRequest:
    """Fill empty parsing/runtime credentials before Paper2PPT pagecontent is invoked."""
    load_runtime_env()
    api_url = _text_api_url(req.api_url)
    api_key = _text_api_key(req.api_key)
    model = _text_model(req.model, role="content")
    outline_model = _text_model(req.outline_model, role="outline", fallback=model)
    image_api_url = _image_api_url(req.image_api_url, fallback_url=api_url)
    image_api_key = _image_api_key(req.image_api_key, fallback_key=api_key)
    image_model = _image_model(req.image_model)

    return req.model_copy(
        update={
            "api_url": api_url,
            "api_key": api_key,
            "model": model,
            "outline_model": outline_model,
            "vlm_api_url": _first_non_empty(req.vlm_api_url, api_url),
            "vlm_api_key": _first_non_empty(req.vlm_api_key, api_key),
            "vlm_model": _vlm_model(req.vlm_model, fallback_model=model),
            "image_api_url": image_api_url,
            "image_api_key": image_api_key,
            "image_model": image_model,
            "gen_fig_model": _first_non_empty(req.gen_fig_model, image_model),
        }
    )
