from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.config.pricing import get_workflow_cost
from fastapi_app.dependencies import AuthUser
from fastapi_app.main import create_app
from fastapi_app.middleware import api_key as api_key_module
from fastapi_app.services import image_playground_service as image_playground_service_module
from fastapi_app.services.billing_service import BillingService


def test_image_playground_generate_route_handles_batch_billing_after_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_key_module, "API_KEY", "test-key")

    async def fake_resolve_user(_request):
        return AuthUser(user_id="user-123", email="demo@example.com", phone=None, is_anonymous=False)

    def fake_get_quota(self, **_kwargs):
        return {
            "used": 0,
            "limit": 100,
            "remaining": 100,
            "is_authenticated": True,
            "billing_mode": "free",
            "user_id": "user-123",
        }

    charges: list[dict] = []

    def fake_consume(self, **kwargs):
        charges.append(kwargs)
        return {"success": True, "amount": kwargs.get("amount", 0)}

    async def fake_generate_image(**kwargs):
        save_path = Path(kwargs["save_path"])
        Image.new("RGB", (1024, 768), color=(20, 50, 90)).save(save_path)
        return str(save_path)

    monkeypatch.setattr(api_key_module, "_resolve_user", fake_resolve_user)
    monkeypatch.setattr(BillingService, "get_quota", fake_get_quota)
    monkeypatch.setattr(BillingService, "consume_workflow", fake_consume)
    monkeypatch.setattr(image_playground_service_module, "get_outputs_root", lambda: tmp_path)
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

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/image-playground/generate",
        headers={
            "X-API-Key": "test-key",
            "Authorization": "Bearer fake-token",
            "X-Workflow-Amount": "4",
        },
        json={
            "prompt": "draw a research workflow figure",
            "model": "gpt-image-2-all",
            "template_key": "research_general",
            "domain_key": "computer_science",
            "batch_count": 2,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert payload["workflow_type"] == "image_playground"
    assert payload["model"] == "gpt-image-2-all"
    assert payload["batch_count"] == 2
    assert payload["success_count"] == 2
    assert payload["failed_count"] == 0
    assert len(payload["images"]) == 2
    assert payload["zip_path"].endswith(".zip")

    assert len(charges) == 1
    assert charges[0]["workflow_type"] == "image_playground"
    assert charges[0]["amount"] == 4
    assert charges[0]["user"].id == "user-123"


def test_image_playground_workflow_cost_defaults_to_two_points() -> None:
    assert get_workflow_cost("image_playground", default=1) == 2
