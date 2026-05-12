from __future__ import annotations

import sys
from pathlib import Path
import types

import pytest
from fastapi import Request
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

from fastapi_app.schemas import EditablePPTGenerationRequest
from fastapi_app.services.editable_ppt_service import EditablePPTService
from fastapi_app.services.paper2ppt_service import Paper2PPTService
from vendor.presentagent_runtime.contracts import EditablePPTRunArtifacts, EditablePPTRunRequest


@pytest.mark.anyio
async def test_generate_editable_ppt_parses_pagecontent_and_normalizes_runtime_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = EditablePPTService(paper2ppt_service=Paper2PPTService())
    outputs_root = PROJECT_ROOT / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    hero_path = (outputs_root / "service-test" / "assets" / "hero.png").resolve()
    hero_path.parent.mkdir(parents=True, exist_ok=True)
    hero_path.write_bytes((PROJECT_ROOT / "tests" / "test_02.png").read_bytes())
    result_path = (outputs_root / "service-test" / "editable-ppt-run").resolve()
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final").mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final" / "final_ir.json").write_text("{}", encoding="utf-8")

    req = EditablePPTGenerationRequest(
        chat_api_url="https://llm.example.com/v1",
        api_key="user-key",
        credential_scope="paper2ppt",
        model="gpt-test",
        language="en",
        style="clean",
        result_path=str(result_path),
        pagecontent=f'[{{"title":"Slide 1","ppt_img_path":"/outputs/service-test/assets/hero.png"}}]',
        include_pdf_preview=True,
    )
    captured: dict[str, object] = {}

    def fake_resolve_llm_credentials(chat_api_url, api_key, *, scope=None):
        captured["resolved_credentials_input"] = {
            "chat_api_url": chat_api_url,
            "api_key": api_key,
            "scope": scope,
        }
        return "https://resolved-llm.example.com/v1", "resolved-key"

    def fake_run_from_pagecontent(runtime_req: EditablePPTRunRequest, *, progress_callback=None) -> EditablePPTRunArtifacts:
        captured["runtime_req"] = runtime_req
        run_dir = Path(runtime_req.result_path)
        return EditablePPTRunArtifacts(
            run_dir=str(run_dir / "code_runtime"),
            ir_path=str(run_dir / "code_runtime" / "ir" / "deck_ir.json"),
            recipe_path=str(run_dir / "code_runtime" / "recipes" / "render_recipe.json"),
            pptx_path=str(run_dir / "code_runtime" / "exports" / "paper2ppt_code_editable.pptx"),
            pdf_path=str(run_dir / "code_runtime" / "exports" / "paper2ppt_code_preview.pdf"),
            log_path=str(run_dir / "code_runtime" / "logs" / "run.log"),
        )

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_llm_credentials",
        fake_resolve_llm_credentials,
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_model_name",
        lambda requested_model, *, managed_default, fallback_default=None: requested_model or managed_default or fallback_default or "",
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.run_from_pagecontent",
        fake_run_from_pagecontent,
    )

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.assemble_deck_from_final_ir",
        lambda final_ir_path, output_pptx, *, include_pdf_preview=True, progress_callback=None: {"pptx_path": str(output_pptx), "pdf_path": ""},
    )
    response = await service.generate_editable_ppt(req=req, request=None)

    assert captured["resolved_credentials_input"] == {
        "chat_api_url": "https://llm.example.com/v1",
        "api_key": "user-key",
        "scope": "paper2ppt",
    }
    assert captured["runtime_req"] == EditablePPTRunRequest(
        result_path=str(result_path),
        pagecontent=[{"title": "Slide 1", "ppt_img_path": str(hero_path)}],
        language="en",
        style="clean",
        model="gpt-test",
        api_url="https://resolved-llm.example.com/v1",
        api_key="resolved-key",
        include_pdf_preview=False,
    )
    assert response["success"] is True
    assert response["result_path"] == str(result_path)
    assert response["ppt_pptx_path"].endswith("paper2ppt_code_editable.pptx")
    assert response["ir_path"] == str(
        result_path / "code_runtime" / "ir" / "deck_ir.json"
    )
    assert response["render_log_path"] == str(
        result_path / "code_runtime" / "logs" / "run.log"
    )


@pytest.mark.anyio
async def test_generate_editable_ppt_resolves_llm_vlm_and_image_credentials_before_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = EditablePPTService(paper2ppt_service=Paper2PPTService())
    outputs_root = PROJECT_ROOT / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    result_path = (outputs_root / "service-test" / "editable-ppt-agent-run").resolve()
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final").mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final" / "final_ir.json").write_text("{}", encoding="utf-8")

    req = EditablePPTGenerationRequest(
        chat_api_url="https://llm.example.com/v1",
        api_key="user-key",
        credential_scope="paper2ppt",
        model="gpt-test",
        language="en",
        style="clean",
        result_path=str(result_path),
        pagecontent='[{"title":"Slide 1","key_points":["Point 1"]}]',
        include_pdf_preview=True,
    )
    captured: dict[str, object] = {}

    def fake_resolve_llm_credentials(chat_api_url, api_key, *, scope=None):
        captured.setdefault("llm_calls", []).append(
            {
                "chat_api_url": chat_api_url,
                "api_key": api_key,
                "scope": scope,
            }
        )
        call_index = len(captured["llm_calls"])
        if call_index == 1:
            return "https://resolved-llm.example.com/v1", "resolved-llm-key"
        return "https://resolved-vlm.example.com/v1", "resolved-vlm-key"

    def fake_resolve_image_generation_credentials(chat_api_url, api_key, *, scope=None):
        captured["image_call"] = {
            "chat_api_url": chat_api_url,
            "api_key": api_key,
            "scope": scope,
        }
        return "https://resolved-image.example.com/v1", "resolved-image-key"

    def fake_run_from_pagecontent(runtime_req: EditablePPTRunRequest, *, progress_callback=None) -> EditablePPTRunArtifacts:
        captured["runtime_req"] = runtime_req
        run_dir = Path(runtime_req.result_path)
        return EditablePPTRunArtifacts(
            run_dir=str(run_dir / "code_runtime"),
            materials_manifest_path=str(run_dir / "code_runtime" / "materials" / "material_manifest.json"),
            material_resolution_path=str(run_dir / "code_runtime" / "materials" / "material_resolution.json"),
            planned_ir_path=str(run_dir / "code_runtime" / "ir" / "planned" / "final_ir.json"),
            final_ir_path=str(run_dir / "code_runtime" / "ir" / "final" / "final_ir.json"),
            ir_path=str(run_dir / "code_runtime" / "ir" / "final" / "final_ir.json"),
            recipe_path=str(run_dir / "code_runtime" / "code" / "generated" / "render_recipe.json"),
            pptx_path=str(run_dir / "code_runtime" / "exports" / "paper2ppt_code_editable.pptx"),
            pdf_path=str(run_dir / "code_runtime" / "exports" / "paper2ppt_code_preview.pdf"),
            log_path=str(run_dir / "code_runtime" / "logs" / "run.log"),
        )

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_llm_credentials",
        fake_resolve_llm_credentials,
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_image_generation_credentials",
        fake_resolve_image_generation_credentials,
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_model_name",
        lambda requested_model, *, managed_default, fallback_default=None: requested_model or managed_default or fallback_default or "",
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.run_from_pagecontent",
        fake_run_from_pagecontent,
    )

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.assemble_deck_from_final_ir",
        lambda final_ir_path, output_pptx, *, include_pdf_preview=True, progress_callback=None: {"pptx_path": str(output_pptx), "pdf_path": ""},
    )
    await service.generate_editable_ppt(req=req, request=None)

    assert captured["llm_calls"] == [
        {
            "chat_api_url": "https://llm.example.com/v1",
            "api_key": "user-key",
            "scope": "paper2ppt",
        },
        {
            "chat_api_url": "https://llm.example.com/v1",
            "api_key": "user-key",
            "scope": "paper2ppt",
        },
    ]
    assert captured["image_call"] == {
        "chat_api_url": "https://llm.example.com/v1",
        "api_key": "user-key",
        "scope": "paper2ppt",
    }
    assert captured["runtime_req"] == EditablePPTRunRequest(
        result_path=str(result_path),
        pagecontent=[{"title": "Slide 1", "key_points": ["Point 1"]}],
        language="en",
        style="clean",
        model="gpt-test",
        api_url="https://resolved-llm.example.com/v1",
        api_key="resolved-llm-key",
        vlm_model="gpt-test",
        vlm_api_url="https://resolved-vlm.example.com/v1",
        vlm_api_key="resolved-vlm-key",
        image_model="gpt-test",
        image_api_url="https://resolved-image.example.com/v1",
        image_api_key="resolved-image-key",
        enable_agent_planner=True,
        enable_material_resolution=True,
        enable_llm_codegen=False,
        include_pdf_preview=False,
    )


@pytest.mark.anyio
async def test_generate_editable_ppt_externalizes_artifact_paths_when_request_is_present(
    monkeypatch,
) -> None:
    outputs_root = PROJECT_ROOT / "outputs"
    result_path = (outputs_root / "service-test" / "editable-ppt-run-url").resolve()
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final").mkdir(parents=True, exist_ok=True)
    (result_path / "code_runtime" / "ir" / "final" / "final_ir.json").write_text("{}", encoding="utf-8")
    hero_path = (outputs_root / "service-test" / "assets" / "hero-url.png").resolve()
    hero_path.parent.mkdir(parents=True, exist_ok=True)
    hero_path.write_bytes((PROJECT_ROOT / "tests" / "test_02.png").read_bytes())

    service = EditablePPTService(paper2ppt_service=Paper2PPTService())
    req = EditablePPTGenerationRequest(
        chat_api_url="https://llm.example.com/v1",
        api_key="user-key",
        credential_scope="paper2ppt",
        model="gpt-test",
        language="en",
        style="clean",
        result_path=str(result_path),
        pagecontent=f'[{{"title":"Slide 1","ppt_img_path":"/outputs/service-test/assets/{hero_path.name}"}}]',
        include_pdf_preview=True,
    )

    def fake_run_from_pagecontent(runtime_req: EditablePPTRunRequest, *, progress_callback=None) -> EditablePPTRunArtifacts:
        run_dir = Path(runtime_req.result_path)
        pptx_path = run_dir / "code_runtime" / "exports" / "paper2ppt_code_editable.pptx"
        pdf_path = run_dir / "code_runtime" / "exports" / "paper2ppt_code_preview.pdf"
        ir_path = run_dir / "code_runtime" / "ir" / "deck_ir.json"
        log_path = run_dir / "code_runtime" / "logs" / "run.log"
        for path in [pptx_path, pdf_path, ir_path, log_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact", encoding="utf-8")
        return EditablePPTRunArtifacts(
            run_dir=str(run_dir / "code_runtime"),
            materials_manifest_path=str(run_dir / "code_runtime" / "materials" / "material_manifest.json"),
            material_resolution_path=str(run_dir / "code_runtime" / "materials" / "material_resolution.json"),
            planned_ir_path=str(run_dir / "code_runtime" / "ir" / "planned" / "final_ir.json"),
            final_ir_path=str(ir_path),
            ir_path=str(ir_path),
            recipe_path=str(run_dir / "code_runtime" / "recipes" / "render_recipe.json"),
            pptx_path=str(pptx_path),
            pdf_path=str(pdf_path),
            log_path=str(log_path),
        )

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_llm_credentials",
        lambda chat_api_url, api_key, *, scope=None: (chat_api_url, api_key),
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_model_name",
        lambda requested_model, *, managed_default, fallback_default=None: requested_model or managed_default or fallback_default or "",
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.run_from_pagecontent",
        fake_run_from_pagecontent,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "https",
        "path": "/api/v1/paper2ppt/code/generate",
        "headers": [(b"host", b"example.com")],
    }
    request = Request(scope)

    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.assemble_deck_from_final_ir",
        lambda final_ir_path, output_pptx, *, include_pdf_preview=True, progress_callback=None: {"pptx_path": str(output_pptx), "pdf_path": ""},
    )
    response = await service.generate_editable_ppt(req=req, request=request)

    assert "/outputs/" in response["ppt_pptx_path"]
    assert "/outputs/" in response["ppt_pdf_path"]
    assert "/outputs/" in response["ir_path"]
    assert "/outputs/" in response["render_log_path"]
    assert response["all_output_files"]


@pytest.mark.anyio
async def test_generate_editable_ppt_slides_returns_slide_artifacts_and_skips_assembly(
    monkeypatch, tmp_path: Path,
) -> None:
    from vendor.presentagent_runtime.contracts import SlideArtifact
    service = EditablePPTService(paper2ppt_service=Paper2PPTService())
    outputs_root = PROJECT_ROOT / "outputs"
    result_path = (outputs_root / "service-test" / "slides-only").resolve()
    result_path.mkdir(parents=True, exist_ok=True)

    req = EditablePPTGenerationRequest(
        chat_api_url="https://llm.example.com/v1", api_key="k",
        credential_scope="paper2ppt", model="gpt-test",
        language="en", style="",
        result_path=str(result_path),
        pagecontent='[{"title":"S1"},{"title":"S2"}]',
        include_pdf_preview=True,
    )

    captured: dict[str, object] = {}
    def fake_run(req, *, progress_callback=None):
        captured["callback"] = progress_callback
        run_dir = Path(req.result_path) / "code_runtime"
        return EditablePPTRunArtifacts(
            run_dir=str(run_dir),
            materials_manifest_path=str(run_dir / "materials" / "material_manifest.json"),
            material_resolution_path=str(run_dir / "materials" / "material_resolution.json"),
            planned_ir_path=str(run_dir / "ir" / "planned" / "final_ir.json"),
            final_ir_path=str(run_dir / "ir" / "final" / "final_ir.json"),
            ir_path=str(run_dir / "ir" / "final" / "final_ir.json"),
            recipe_path=str(run_dir / "recipes" / "recipe.json"),
            pptx_path=str(run_dir / "exports" / "paper2ppt_code_editable.pptx"),
            pdf_path="",
            log_path=str(run_dir / "logs" / "run.log"),
            slides_dir=str(run_dir / "slides"),
            slide_artifacts=[
                SlideArtifact(index=0, slide_id="s1", title="S1",
                              pptx_path=str(run_dir / "slides/slide_000.pptx"),
                              preview_png_path=str(run_dir / "slides/slide_000.png")),
                SlideArtifact(index=1, slide_id="s2", title="S2",
                              pptx_path=str(run_dir / "slides/slide_001.pptx"),
                              preview_png_path=""),
            ],
        )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_llm_credentials",
        lambda url, key, *, scope=None: (url, key),
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_image_generation_credentials",
        lambda url, key, *, scope=None: (url, key),
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.resolve_model_name",
        lambda m, *, managed_default, fallback_default=None: m or managed_default or fallback_default or "",
    )
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.run_from_pagecontent", fake_run,
    )

    events: list[tuple[str, dict]] = []
    result = await service.generate_editable_ppt_slides(
        req=req, request=None,
        progress_callback=lambda n, p: events.append((n, p)),
    )
    assert callable(captured["callback"])
    assert result["success"] is True
    assert len(result["slide_artifacts"]) == 2
    assert result["slide_artifacts"][0]["pptx_path"].endswith("slide_000.pptx")
    assert result["planned_ir_path"].endswith("final_ir.json")
    assert result.get("ppt_pdf_path", "") == ""


@pytest.mark.anyio
async def test_assemble_editable_ppt_invokes_assemble_deck_from_final_ir(
    monkeypatch, tmp_path: Path,
) -> None:
    from fastapi_app.schemas import AssembleEditablePPTRequest
    service = EditablePPTService(paper2ppt_service=Paper2PPTService())
    outputs_root = PROJECT_ROOT / "outputs"
    result_path = (outputs_root / "service-test" / "assemble").resolve()
    (result_path / "code_runtime" / "ir" / "final").mkdir(parents=True, exist_ok=True)
    final_ir_path = result_path / "code_runtime" / "ir" / "final" / "final_ir.json"
    final_ir_path.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}
    def fake_assemble(final_ir_path, output_pptx, *, include_pdf_preview, progress_callback=None):
        captured["final_ir_path"] = str(final_ir_path)
        captured["output_pptx"] = str(output_pptx)
        captured["include_pdf_preview"] = include_pdf_preview
        Path(output_pptx).parent.mkdir(parents=True, exist_ok=True)
        Path(output_pptx).write_text("fake-pptx", encoding="utf-8")
        return {"pptx_path": str(output_pptx), "pdf_path": ""}
    monkeypatch.setattr(
        "fastapi_app.services.editable_ppt_service.assemble_deck_from_final_ir",
        fake_assemble,
    )

    req = AssembleEditablePPTRequest(result_path=str(result_path), include_pdf_preview=False)
    result = await service.assemble_editable_ppt(req=req, request=None)
    assert captured["final_ir_path"] == str(final_ir_path)
    assert captured["include_pdf_preview"] is False
    assert result["success"] is True
    assert result["ppt_pptx_path"].endswith("paper2ppt_code_editable.pptx")
