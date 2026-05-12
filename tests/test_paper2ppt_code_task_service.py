from __future__ import annotations

import sys
from pathlib import Path
import types
import asyncio

import pytest
from fastapi import Request
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- fake modules bootstrap ----

fake_pydantic_settings = types.ModuleType("pydantic_settings")
fake_pydantic_settings.BaseSettings = BaseModel
sys.modules.setdefault("pydantic_settings", fake_pydantic_settings)

fake_yaml = types.ModuleType("yaml")
fake_yaml.safe_load = lambda *args, **kwargs: {}
sys.modules.setdefault("yaml", fake_yaml)

fake_utils = types.ModuleType("dataflow_agent.utils")
fake_utils.get_project_root = lambda: PROJECT_ROOT
sys.modules.setdefault("dataflow_agent.utils", fake_utils)

fake_utils_common = types.ModuleType("dataflow_agent.utils_common")
fake_utils_common.robust_parse_json = lambda text, **kwargs: text
sys.modules.setdefault("dataflow_agent.utils_common", fake_utils_common)

fake_wa_paper2ppt = types.ModuleType("fastapi_app.workflow_adapters.wa_paper2ppt")
fake_wa_paper2ppt.run_paper2page_content_wf_api = lambda *args, **kwargs: None
fake_wa_paper2ppt.run_paper2ppt_full_pipeline = lambda *args, **kwargs: None
fake_wa_paper2ppt.run_paper2ppt_wf_api = lambda *args, **kwargs: None
sys.modules.setdefault("fastapi_app.workflow_adapters.wa_paper2ppt", fake_wa_paper2ppt)

fake_prompt_templates = types.ModuleType("dataflow_agent.promptstemplates")
fake_prompt_templates.PromptsTemplateGenerator = object
sys.modules.setdefault("dataflow_agent.promptstemplates", fake_prompt_templates)

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

# ---- imports under test ----

from fastapi_app.schemas import AssembleEditablePPTRequest, EditablePPTGenerationRequest
from fastapi_app.services.editable_ppt_service import EditablePPTService
from fastapi_app.services.paper2ppt_code_task_service import Paper2PPTCodeTaskService


# ---- helpers ----

def _make_result_path(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- tests ----

@pytest.mark.anyio
async def test_submit_generate_slides_task_writes_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """generate task completes with progress containing slideArtifacts from simulated events."""
    result_path = _make_result_path(tmp_path, "gen-task")

    captured_callback = {}

    async def fake_generate_editable_ppt_slides(self_inner, req, request, progress_callback=None):
        # Simulate progress events that the real runtime emits
        loop = asyncio.get_running_loop()

        def _emit(event, params):
            if progress_callback is not None:
                # Simulate what the service does: call_soon_threadsafe or direct
                progress_callback(event, params)

        _emit("planning_done", {
            "planned_ir_path": str(result_path / "code_runtime/ir/planned/final_ir.json"),
            "materials_manifest_path": str(result_path / "code_runtime/materials/manifest.json"),
            "material_resolution_path": str(result_path / "code_runtime/materials/resolution.json"),
        })
        _emit("final_ir_done", {
            "final_ir_path": str(result_path / "code_runtime/ir/final/final_ir.json"),
        })
        _emit("slide_rendered", {
            "slide_done": 1,
            "slide_total": 2,
            "artifact": {"index": 0, "slide_id": "s0", "pptx_path": str(result_path / "slides/s0.pptx"), "preview_png_path": ""},
        })
        _emit("slide_rendered", {
            "slide_done": 2,
            "slide_total": 2,
            "artifact": {"index": 1, "slide_id": "s1", "pptx_path": str(result_path / "slides/s1.pptx"), "preview_png_path": ""},
        })
        _emit("rendering_done", {
            "slide_artifacts": [
                {"index": 0, "slide_id": "s0", "pptx_path": str(result_path / "slides/s0.pptx"), "preview_png_path": ""},
                {"index": 1, "slide_id": "s1", "pptx_path": str(result_path / "slides/s1.pptx"), "preview_png_path": ""},
            ],
        })

        return {
            "success": True,
            "result_path": str(result_path),
            "ppt_pptx_path": str(result_path / "paper2ppt_code_editable.pptx"),
            "ppt_pdf_path": "",
            "slide_artifacts": [
                {"index": 0, "slide_id": "s0"},
                {"index": 1, "slide_id": "s1"},
            ],
        }

    monkeypatch.setattr(
        EditablePPTService,
        "generate_editable_ppt_slides",
        fake_generate_editable_ppt_slides,
    )

    # Override TASK_ROOT to use tmp_path for isolation
    import fastapi_app.services.paper2ppt_code_task_service as svc_module
    original_task_root = svc_module.TASK_ROOT
    svc_module.TASK_ROOT = tmp_path / "tasks"
    try:
        svc = Paper2PPTCodeTaskService()
        req = EditablePPTGenerationRequest(
            result_path=str(result_path),
            pagecontent='[{"title":"S1"},{"title":"S2"}]',
            model="test-model",
            language="en",
        )
        response = await svc.submit_generate_slides_task(req, request=None)
        task_id = response["task_id"]

        # Poll until done (background task runs in same event loop)
        for _ in range(100):
            await asyncio.sleep(0.05)
            rec = svc.get_task(task_id)
            if rec["status"] in ("done", "failed"):
                break

        final = svc.get_task(task_id)
        assert final["status"] == "done", f"Expected done, got {final['status']}: {final.get('error')}"
        progress = final["progress"]
        assert progress is not None
        assert progress["stage"] == "done"
        assert len(progress["slideArtifacts"]) == 2
        assert progress["slideArtifacts"][0]["slide_id"] == "s0"
        assert progress["slideDone"] == 2
        assert progress["slideTotal"] == 2
        assert "final_ir.json" in progress["finalIrPath"]
        assert final["result"] is not None
        assert final["result"]["success"] is True
    finally:
        svc_module.TASK_ROOT = original_task_root


@pytest.mark.anyio
async def test_submit_assemble_task_calls_assemble_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """assemble task completes and result contains ppt_pptx_path."""
    result_path = _make_result_path(tmp_path, "assemble-task")

    called = {}

    async def fake_assemble_editable_ppt(self_inner, req, request, progress_callback=None):
        called["result_path"] = req.result_path
        called["include_pdf_preview"] = req.include_pdf_preview
        return {
            "success": True,
            "result_path": str(result_path),
            "ppt_pptx_path": str(result_path / "paper2ppt_code_editable.pptx"),
            "ppt_pdf_path": "",
            "error": "",
        }

    monkeypatch.setattr(
        EditablePPTService,
        "assemble_editable_ppt",
        fake_assemble_editable_ppt,
    )

    import fastapi_app.services.paper2ppt_code_task_service as svc_module
    original_task_root = svc_module.TASK_ROOT
    svc_module.TASK_ROOT = tmp_path / "tasks2"
    try:
        svc = Paper2PPTCodeTaskService()
        req = AssembleEditablePPTRequest(
            result_path=str(result_path),
            include_pdf_preview=False,
        )
        response = await svc.submit_assemble_task(req, request=None)
        task_id = response["task_id"]

        for _ in range(100):
            await asyncio.sleep(0.05)
            rec = svc.get_task(task_id)
            if rec["status"] in ("done", "failed"):
                break

        final = svc.get_task(task_id)
        assert final["status"] == "done", f"Expected done, got {final['status']}: {final.get('error')}"
        assert called["result_path"] == str(result_path)
        assert called["include_pdf_preview"] is False
        assert final["result"] is not None
        assert final["result"]["ppt_pptx_path"].endswith("paper2ppt_code_editable.pptx")
    finally:
        svc_module.TASK_ROOT = original_task_root
