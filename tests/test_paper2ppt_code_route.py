from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

fake_pydantic_settings = types.ModuleType("pydantic_settings")
fake_pydantic_settings.BaseSettings = BaseModel
sys.modules.setdefault("pydantic_settings", fake_pydantic_settings)

fake_yaml = types.ModuleType("yaml")
fake_yaml.safe_load = lambda *args, **kwargs: {}
sys.modules.setdefault("yaml", fake_yaml)

fake_utils = types.ModuleType("dataflow_agent.utils")
fake_utils.get_project_root = lambda: PROJECT_ROOT
sys.modules.setdefault("dataflow_agent.utils", fake_utils)

fake_logger = types.ModuleType("dataflow_agent.logger")


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


fake_logger.get_logger = lambda *args, **kwargs: _FakeLogger()
sys.modules.setdefault("dataflow_agent.logger", fake_logger)

from fastapi_app.services.editable_ppt_service import EditablePPTService

ROUTER_PATH = PROJECT_ROOT / "fastapi_app" / "routers" / "paper2ppt_code.py"
spec = importlib.util.spec_from_file_location("paper2ppt_code_router_under_test", ROUTER_PATH)
assert spec is not None and spec.loader is not None
paper2ppt_code_router = importlib.util.module_from_spec(spec)
sys.modules.setdefault("paper2ppt_code_router_under_test", paper2ppt_code_router)
spec.loader.exec_module(paper2ppt_code_router)
router = paper2ppt_code_router.router


def test_paper2ppt_code_generate_accepts_form_and_returns_success(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEditablePPTService:
        async def generate_editable_ppt(self, req, request=None):
            captured["req"] = req
            return {
                "success": True,
                "result_path": req.result_path,
                "ppt_pptx_path": "/outputs/demo/paper2ppt_code_editable.pptx",
                "ppt_pdf_path": "",
                "ir_path": "/outputs/demo/deck_ir.json",
                "render_log_path": "/outputs/demo/run.log",
                "all_output_files": [],
                "error": "",
            }

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[paper2ppt_code_router.get_editable_ppt_service] = lambda: FakeEditablePPTService()

    client = TestClient(app)
    response = client.post(
        "/api/v1/paper2ppt/code/generate",
        data={
            "result_path": "outputs/service-test/editable-ppt-run",
            "pagecontent": '[{"title":"Slide 1","ppt_img_path":"/outputs/service-test/assets/hero.png"}]',
            "model": "gpt-5.1",
            "language": "en",
            "style": "clean",
            "include_pdf_preview": "true",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert captured["req"].result_path == "outputs/service-test/editable-ppt-run"
    assert captured["req"].pagecontent == '[{"title":"Slide 1","ppt_img_path":"/outputs/service-test/assets/hero.png"}]'


def test_generate_task_endpoint_accepts_form_and_returns_task_id(monkeypatch):
    class FakeTaskService:
        async def submit_generate_slides_task(self, req, request=None):
            return {"task_id": "abc123", "status": "pending", "progress": {}}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[paper2ppt_code_router.get_code_task_service] = lambda: FakeTaskService()

    client = TestClient(app)
    response = client.post(
        "/api/v1/paper2ppt/code/generate-task",
        data={
            "result_path": "outputs/test/generate-task-run",
            "pagecontent": '[{"title":"Slide 1"}]',
            "language": "en",
            "style": "",
            "include_pdf_preview": "false",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_id"] == "abc123"
    assert payload["status"] == "pending"


def test_assemble_task_endpoint_returns_task(monkeypatch):
    class FakeTaskService:
        async def submit_assemble_task(self, req, request=None):
            return {"task_id": "def456", "status": "pending", "progress": {}}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[paper2ppt_code_router.get_code_task_service] = lambda: FakeTaskService()

    client = TestClient(app)
    response = client.post(
        "/api/v1/paper2ppt/code/assemble-task",
        data={
            "result_path": "outputs/test/assemble-task-run",
            "include_pdf_preview": "true",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_id"] == "def456"
    assert payload["status"] == "pending"


def test_code_tasks_get_endpoint(monkeypatch):
    class FakeTaskService:
        def get_task(self, task_id, request=None):
            return {"task_id": task_id, "status": "completed", "progress": {}}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[paper2ppt_code_router.get_code_task_service] = lambda: FakeTaskService()

    client = TestClient(app)
    response = client.get("/api/v1/paper2ppt/code/tasks/abc123")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_id"] == "abc123"
    assert payload["status"] == "completed"
