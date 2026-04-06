from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.main import create_app
from fastapi_app.middleware import api_key as api_key_module
from fastapi_app.services.billing_service import BillingService
from fastapi_app.services.paper2ppt_frontend_service import Paper2PPTFrontendService
from fastapi_app.services.paper2ppt_service import Paper2PPTService


def test_paper2ppt_generate_multipart_survives_middleware(monkeypatch):
    monkeypatch.setattr(api_key_module, "API_KEY", "test-key")

    charges: list[dict] = []
    captured: dict[str, object] = {}

    def fake_consume(self, **kwargs):
        charges.append(kwargs)
        return {"success": True, "amount": kwargs.get("amount", 0)}

    async def fake_generate(self, req, reference_img=None, request=None):
        captured["req"] = req
        captured["reference_img_name"] = getattr(reference_img, "filename", None)
        return {
            "success": True,
            "result_path": req.result_path,
            "all_output_files": [],
        }

    monkeypatch.setattr(BillingService, "consume_workflow", fake_consume)
    monkeypatch.setattr(Paper2PPTService, "generate_ppt", fake_generate)

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/paper2ppt/generate",
        headers={"X-API-Key": "test-key"},
        data={
            "img_gen_model_name": "gemini-3-pro-image-preview",
            "result_path": "/tmp/paper2ppt-test",
            "pagecontent": '[{"ppt_img_path":"/outputs/demo/page_000.png"}]',
            "get_down": "false",
            "model": "gpt-5.1",
            "language": "en",
        },
        files={"reference_img": ("ref.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert captured["reference_img_name"] == "ref.png"
    assert captured["req"].img_gen_model_name == "gemini-3-pro-image-preview"
    assert captured["req"].result_path == "/tmp/paper2ppt-test"
    assert len(charges) == 1
    assert charges[0]["workflow_type"] == "paper2ppt"
    assert charges[0]["amount"] == 1


def test_paper2ppt_frontend_generate_bills_after_binding(monkeypatch):
    monkeypatch.setattr(api_key_module, "API_KEY", "test-key")

    charges: list[dict] = []
    captured: dict[str, object] = {}

    def fake_consume(self, **kwargs):
        charges.append(kwargs)
        return {"success": True, "amount": kwargs.get("amount", 0)}

    async def fake_generate_slides(self, req, request=None):
        captured["req"] = req
        return {
            "success": True,
            "slides": [],
            "result_path": req.result_path,
        }

    monkeypatch.setattr(BillingService, "consume_workflow", fake_consume)
    monkeypatch.setattr(Paper2PPTFrontendService, "generate_slides", fake_generate_slides)

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/paper2ppt/frontend/generate",
        headers={"X-API-Key": "test-key"},
        data={
            "result_path": "/tmp/paper2ppt-frontend-test",
            "pagecontent": '[{"title":"Slide 1"}]',
            "include_images": "false",
            "model": "gpt-5.1",
            "language": "zh",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert captured["req"].result_path == "/tmp/paper2ppt-frontend-test"
    assert len(charges) == 1
    assert charges[0]["workflow_type"] == "paper2ppt"
    assert charges[0]["amount"] == 1


def test_normalize_ppt_response_exposes_failed_pages():
    service = Paper2PPTService()

    normalized = service.normalize_ppt_response(
        {
            "success": True,
            "result_path": "/tmp/paper2ppt-partial",
            "pagecontent": [
                {
                    "page_idx": 0,
                    "generated_img_path": "/tmp/paper2ppt-partial/ppt_pages/page_000.png",
                    "mode": "origin_gen",
                },
                {
                    "page_idx": 1,
                    "generated_img_path": None,
                    "mode": "origin_gen_failed",
                    "error": "api failed",
                },
            ],
        },
        request=None,
    )

    assert normalized["partial_success"] is True
    assert normalized["failed_page_indices"] == [1]
    assert normalized["failed_pages"] == [
        {
            "page_idx": 1,
            "reason": "api failed",
            "mode": "origin_gen_failed",
            "error": "api failed",
        }
    ]
