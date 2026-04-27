from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.dependencies import AuthUser
from fastapi_app.services.billing_service import BillingService
from fastapi_app.services import image_playground_service as image_playground_service_module
from fastapi_app.services.image_playground_service import ImagePlaygroundService


def _build_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/image-playground/generate",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)
    request.state.guest_id = "guest:test"
    return request


def test_generation_kwargs_accepts_custom_gemini_controls() -> None:
    service = ImagePlaygroundService()
    kwargs = service._generation_kwargs(
        "gemini-3.1-flash-image-preview",
        aspect_ratio="1:8",
        resolution="4K",
        size=None,
        quality=None,
    )
    assert kwargs["aspect_ratio"] == "1:8"
    assert kwargs["resolution"] == "4K"


def test_generation_kwargs_rejects_unsupported_gemini_aspect_ratio() -> None:
    service = ImagePlaygroundService()
    with pytest.raises(HTTPException):
        service._generation_kwargs(
            "gemini-3-pro-image-preview",
            aspect_ratio="1:8",
            resolution="2K",
            size=None,
            quality=None,
        )


def test_generation_kwargs_accepts_custom_gpt_image_controls() -> None:
    service = ImagePlaygroundService()
    kwargs = service._generation_kwargs(
        "gpt-image-2",
        aspect_ratio=None,
        resolution=None,
        size="1536x1024",
        quality="high",
    )
    assert kwargs["size"] == "1536x1024"
    assert kwargs["quality"] == "high"


def test_generation_kwargs_ignores_extra_controls_for_gpt_image_2_all() -> None:
    service = ImagePlaygroundService()
    kwargs = service._generation_kwargs(
        "gpt-image-2-all",
        aspect_ratio="16:9",
        resolution="4K",
        size="2048x1152",
        quality="high",
    )
    assert kwargs == {"response_format": "b64_json", "timeout": 120}


def test_generate_batch_creates_previews_and_bills_only_successes(tmp_path: Path, monkeypatch) -> None:
    service = ImagePlaygroundService()
    service.outputs_root = tmp_path
    captured_charge: dict[str, object] = {}

    async def fake_generate_image(**kwargs):
        save_path = Path(kwargs["save_path"])
        if save_path.name == "image_02.png":
            raise RuntimeError("upstream timeout")
        Image.new("RGB", (1400, 900), color=(50, 80, 120)).save(save_path)
        return str(save_path)

    def fake_consume(self, **kwargs):
        captured_charge.update(kwargs)
        return {"success": True, "amount": kwargs.get("amount", 0)}

    monkeypatch.setattr(
        image_playground_service_module,
        "resolve_image_generation_credentials",
        lambda *_args, **_kwargs: ("https://api.example.com/v1", "dummy-key"),
    )
    monkeypatch.setattr(
        image_playground_service_module,
        "generate_or_edit_and_save_image_async",
        fake_generate_image,
    )
    monkeypatch.setattr(BillingService, "consume_workflow", fake_consume)

    user = AuthUser(user_id="user-123", email="demo@example.com", phone=None, is_anonymous=False)
    result = asyncio.run(
        service.generate(
            prompt="draw a scientific workflow",
            model="gpt-image-2-all",
            chat_api_url=None,
            api_key=None,
            template_key="research",
            domain_key="research",
            aspect_ratio="16:9",
            resolution="2K",
            size=None,
            quality=None,
            batch_count=4,
            request=_build_request(),
            user=user,
        )
    )

    assert result["success"] is True
    assert result["batch_count"] == 4
    assert result["success_count"] == 3
    assert result["failed_count"] == 1
    assert len(result["images"]) == 3
    assert result["zip_path"].endswith(".zip")
    assert result["image_url"] == result["images"][0]["image_url"]
    assert captured_charge["workflow_type"] == "image_playground"
    assert captured_charge["amount"] == 6
    assert captured_charge["user"].id == "user-123"

    first_image = Path(result["images"][0]["file_path"])
    first_preview = Path(result["images"][0]["preview_path"])
    assert first_image.exists()
    assert first_preview.exists()
    assert first_preview.parent.name == "previews"
    assert (first_image.parent / "result.json").exists()
    assert (first_image.parent / "request.json").exists()


def test_generate_batch_all_failures_raise_http_error(tmp_path: Path, monkeypatch) -> None:
    service = ImagePlaygroundService()
    service.outputs_root = tmp_path

    async def fake_generate_image(**_kwargs):
        raise RuntimeError("provider 429")

    monkeypatch.setattr(
        image_playground_service_module,
        "resolve_image_generation_credentials",
        lambda *_args, **_kwargs: ("https://api.example.com/v1", "dummy-key"),
    )
    monkeypatch.setattr(
        image_playground_service_module,
        "generate_or_edit_and_save_image_async",
        fake_generate_image,
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            service.generate(
                prompt="draw a scientific workflow",
                model="gpt-image-2-all",
                chat_api_url=None,
                api_key=None,
                template_key="research",
                domain_key="research",
                aspect_ratio="16:9",
                resolution="2K",
                size=None,
                quality=None,
                batch_count=2,
                request=_build_request(),
                user=AuthUser(user_id="user-123", email="demo@example.com", phone=None, is_anonymous=False),
            )
        )

    assert exc.value.status_code == 502
    assert "All image generation attempts failed" in str(exc.value.detail)
