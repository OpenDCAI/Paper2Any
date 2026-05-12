from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .coder.pptx_code_agent import PPTXCodeAgent, execute_build_deck, write_code_artifacts
from .coder.pptx_recipe_renderer import render_pptx, render_pptx_per_slide, render_recipe
from .contracts import EditablePPTInputRunRequest, EditablePPTRunArtifacts, EditablePPTRunRequest, SlideArtifact
from .env import resolve_runtime_input_request, resolve_runtime_request
from .llm.client import LLMClient
from .materials.material_pipeline import collect_materials, resolve_materials
from .materials.vlm_descriptor import VLMDescriptor
from .patcher.slide_patcher import patch_slide_ir
from .planner.ir_artifact_writer import IRArtifactWriter
from .planner.brief_adapter import pagecontent_to_slide_briefs
from .planner.llm_planner import PagecontentDeckPlanner
from .planner.content_enricher import enrich_deck_ir_from_markdown
from .planner.ir_models import DeckIR
from .planner.pagecontent_adapter import pagecontent_to_deck_ir
from .refiner.noop_refiner import refine_ir
from .renderer.artifact_store import build_artifacts, resolve_artifact_base_dir, write_json, write_log


ProgressCallback = Callable[[str, dict], None]


def _emit(callback: ProgressCallback | None, name: str, payload: dict | None = None) -> None:
    if callback is None:
        return
    try:
        callback(name, dict(payload or {}))
    except Exception:
        pass


def _build_slide_briefs(deck_ir) -> list[dict[str, object]]:
    briefs = []
    for slide in deck_ir.slides:
        briefs.append(
            {
                "brief_id": slide.brief_id or f"brief-{slide.page_num:03d}",
                "slide_id": slide.slide_id,
                "type": slide.type,
                "section_id": slide.section_id,
                "section_title": slide.section_title,
                "title": slide.title,
                "core_message": slide.core_message,
                "objective": slide.objective,
                "content_points": list(slide.points),
                "source_chunk_ids": list(slide.source_chunk_ids),
                "source_headings": [slide.section_title or slide.title],
                "source_excerpt": slide.core_message or slide.objective,
            }
        )
    return briefs


def _build_deck_stage(deck_ir) -> dict[str, object]:
    return {
        "metadata": deck_ir.metadata.model_dump(),
        "storyline": deck_ir.storyline.model_dump(),
        "subtitle": deck_ir.subtitle,
        "theme": deck_ir.theme.model_dump(),
        "slide_manifest": list(deck_ir.slide_manifest),
        "source_asset_index": dict(deck_ir.source_asset_index),
    }


def _build_llm_client(req: EditablePPTRunRequest) -> LLMClient | None:
    if not req.api_url or not req.api_key or not req.model:
        return None
    return LLMClient(
        api_key=req.api_key,
        api_base=req.api_url,
        model=req.model,
        client_type="llm",
    )


def _build_vlm_client(req: EditablePPTRunRequest) -> LLMClient | None:
    if not req.vlm_api_url or not req.vlm_api_key or not req.vlm_model:
        return None
    return LLMClient(
        api_key=req.vlm_api_key,
        api_base=req.vlm_api_url,
        model=req.vlm_model,
        client_type="vlm",
    )


def plan_deck_ir_and_materials(
    req: EditablePPTRunRequest,
    *,
    artifacts: EditablePPTRunArtifacts,
    log_lines: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
):
    """Agent step 2: build DeckIR/SlideIR, collect materials, and resolve slide assets."""
    log_lines = log_lines if log_lines is not None else []
    artifact_writer = IRArtifactWriter(artifacts.run_dir)
    llm_client = _build_llm_client(req) if (req.enable_agent_planner or req.enable_llm_codegen) else None
    vlm_client = _build_vlm_client(req) if req.enable_material_resolution else None

    base_deck_ir = pagecontent_to_deck_ir(
        req.pagecontent,
        language=req.language,
        style=req.style,
        asset_base_dir=req.result_path,
    )
    log_lines.append(f"pagecontent normalized into {len(base_deck_ir.slides)} slides")
    if not base_deck_ir.metadata.deck_id:
        base_deck_ir.metadata.deck_id = Path(req.result_path).name or "paper2ppt-code-runtime"
    base_deck_ir.metadata.stage = "planned"

    markdown_path = Path(req.result_path) / "auto" / "full.md"
    if markdown_path.exists():
        base_deck_ir = enrich_deck_ir_from_markdown(base_deck_ir, markdown_path)
        log_lines.append("enriched slide content from full.md")

    material_manifest = collect_materials(base_deck_ir)

    slide_briefs_payload = pagecontent_to_slide_briefs(
        req.pagecontent,
        language=req.language,
        style=req.style,
    )
    planned_deck_ir = base_deck_ir
    if req.enable_agent_planner and llm_client is not None:
        try:
            planned_deck_ir = PagecontentDeckPlanner(llm_client).plan_deck(
                slide_briefs=slide_briefs_payload,
                base_deck_ir=base_deck_ir,
                materials=material_manifest,
            )
            planned_deck_ir.metadata.deck_id = base_deck_ir.metadata.deck_id
            planned_deck_ir.metadata.stage = "planned"
            material_manifest = collect_materials(planned_deck_ir)
            log_lines.append("llm planner produced planned deck ir")
        except Exception as exc:  # noqa: BLE001
            planned_deck_ir = base_deck_ir
            log_lines.append(f"planner fallback: {exc}")

    material_resolution_stub = {
        "summary": {
            "request_count": len(planned_deck_ir.material_requests),
            "resolved_count": 0,
            "unresolved_count": len(planned_deck_ir.material_requests),
        },
        "requests": [],
    }
    artifact_writer.write_materials(
        material_manifest=material_manifest,
        material_resolution=material_resolution_stub,
    )
    log_lines.append("wrote material manifest artifacts")

    slide_briefs = slide_briefs_payload.get("slide_briefs") or _build_slide_briefs(planned_deck_ir)
    deck_stage = _build_deck_stage(planned_deck_ir)
    artifact_writer.write_planned_ir(
        slide_briefs=slide_briefs,
        deck_stage=deck_stage,
        deck_ir=planned_deck_ir,
    )
    log_lines.append("wrote planned ir artifacts")
    _emit(
        progress_callback,
        "planning_done",
        {
            "planned_ir_path": artifacts.planned_ir_path,
            "materials_manifest_path": artifacts.materials_manifest_path,
            "material_resolution_path": artifacts.material_resolution_path,
        },
    )

    descriptor = VLMDescriptor(vlm_client) if vlm_client is not None else None
    resolved_deck_ir, material_resolution = resolve_materials(
        planned_deck_ir,
        material_manifest,
        descriptor=descriptor,
    )
    artifact_writer.write_materials(
        material_manifest=material_manifest,
        material_resolution=material_resolution,
    )

    needs_material_resolution = bool(planned_deck_ir.material_requests) or any(
        slide.asset_paths for slide in planned_deck_ir.slides
    )
    refine_input_deck = resolved_deck_ir if needs_material_resolution else planned_deck_ir
    refined_deck_ir = refine_ir(refine_input_deck)
    refined_deck_ir.metadata.stage = "final"
    artifact_writer.write_final_ir(refined_deck_ir)
    log_lines.append("deck ir refined")
    log_lines.append("wrote final ir artifacts")
    _emit(progress_callback, "final_ir_done", {"final_ir_path": artifacts.final_ir_path})

    return refined_deck_ir, material_manifest, material_resolution


def generate_slide_code_from_ir(
    req: EditablePPTRunRequest,
    *,
    artifacts: EditablePPTRunArtifacts,
    deck_ir,
    material_manifest: dict,
    material_resolution: dict,
    log_lines: list[str] | None = None,
) -> tuple[Path, bool]:
    """Agent step 3: generate per-slide Python code from final SlideIR."""
    log_lines = log_lines if log_lines is not None else []
    llm_client = _build_llm_client(req) if req.enable_llm_codegen else None
    code_agent = PPTXCodeAgent(llm_client) if llm_client is not None else None
    build_deck_path = write_code_artifacts(
        artifacts.run_dir,
        deck_ir,
        code_agent=code_agent,
        material_manifest=material_manifest,
        material_resolution=material_resolution,
    )
    if code_agent is not None:
        log_lines.append("wrote llm slide code artifacts from final slide ir")
    else:
        log_lines.append("wrote fallback slide code artifacts from final slide ir")
    return build_deck_path, code_agent is not None


async def parse_pagecontent_from_input(req: EditablePPTInputRunRequest) -> tuple[list[dict[str, object]], str]:
    """Reuse Paper2PPT's PDF/TEXT/TOPIC/PPTX parser as the Agent runtime first step."""
    req = resolve_runtime_input_request(req)
    from fastapi_app.schemas import Paper2PPTRequest
    from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2page_content_wf_api

    input_type = str(req.input_type or "PDF").upper().strip()
    if input_type == "PPTX":
        input_type = "PPT"

    outline_req = Paper2PPTRequest(
        language=req.language,
        chat_api_url=req.api_url or "",
        credential_scope=req.credential_scope,
        chat_api_key=req.api_key or "",
        api_key=req.api_key or "",
        image_api_url=req.image_api_url or "",
        image_api_key=req.image_api_key or "",
        model=req.outline_model or req.model,
        gen_fig_model=req.gen_fig_model,
        input_type=input_type,
        input_content=req.input_content,
        render_dpi=req.render_dpi,
        aspect_ratio=req.aspect_ratio,
        style=req.style,
        ref_img=req.ref_img,
        email=req.email,
        page_count=req.page_count,
        use_long_paper=bool(req.use_long_paper),
    )
    pagecontent_resp = await run_paper2page_content_wf_api(outline_req, result_path=Path(req.result_path))
    pagecontent = pagecontent_resp.pagecontent or []
    if not pagecontent:
        backend_error = str(getattr(pagecontent_resp, "error", "") or "").strip()
        if backend_error:
            raise RuntimeError(f"paper2page_content returned no pagecontent: {backend_error}")
        raise RuntimeError("paper2page_content returned no pagecontent")
    return pagecontent, pagecontent_resp.result_path or req.result_path


async def run_from_input(req: EditablePPTInputRunRequest) -> EditablePPTRunArtifacts:
    """Run the complete Agent flow, starting with Paper2PPT's unified pagecontent parser."""
    req = resolve_runtime_input_request(req)
    pagecontent, result_path = await parse_pagecontent_from_input(req)
    return run_from_pagecontent(
        EditablePPTRunRequest(
            result_path=result_path,
            pagecontent=pagecontent,
            language=req.language,
            style=req.style,
            model=req.model,
            api_url=req.api_url,
            api_key=req.api_key,
            vlm_model=req.vlm_model,
            vlm_api_url=req.vlm_api_url,
            vlm_api_key=req.vlm_api_key,
            image_model=req.image_model,
            image_api_url=req.image_api_url,
            image_api_key=req.image_api_key,
            enable_agent_planner=req.enable_agent_planner,
            enable_material_resolution=req.enable_material_resolution,
            enable_llm_codegen=req.enable_llm_codegen,
            include_pdf_preview=req.include_pdf_preview,
        )
    )


def run_from_pagecontent(
    req: EditablePPTRunRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> EditablePPTRunArtifacts:
    req = resolve_runtime_request(req)
    log_lines = ["run_from_pagecontent started"]
    artifacts = build_artifacts(req.result_path)
    final_artifacts = artifacts.model_copy(update={"pdf_path": ""})

    try:
        refined_deck_ir, material_manifest, material_resolution = plan_deck_ir_and_materials(
            req,
            artifacts=artifacts,
            log_lines=log_lines,
            progress_callback=progress_callback,
        )

        write_json(Path(artifacts.ir_path), refined_deck_ir.model_dump())
        log_lines.append(f"wrote deck ir to {artifacts.ir_path}")

        recipe = render_recipe(refined_deck_ir)
        write_json(Path(artifacts.recipe_path), recipe)
        log_lines.append(f"wrote render recipe to {artifacts.recipe_path}")

        rendered_via_codegen = False
        build_deck_path, used_llm = generate_slide_code_from_ir(
            req,
            artifacts=artifacts,
            deck_ir=refined_deck_ir,
            material_manifest=material_manifest,
            material_resolution=material_resolution,
            log_lines=log_lines,
        )
        if req.enable_llm_codegen and used_llm:
            try:
                execute_build_deck(build_deck_path, artifacts.pptx_path)
                rendered_via_codegen = True
                log_lines.append(f"executed build_deck.py to {artifacts.pptx_path}")
            except Exception as exc:  # noqa: BLE001
                log_lines.append(f"codegen execution fallback: {exc}")

        if not rendered_via_codegen:
            render_pptx(refined_deck_ir, Path(artifacts.pptx_path))
            log_lines.append(f"wrote editable pptx to {artifacts.pptx_path}")
        else:
            log_lines.append(f"wrote editable pptx via codegen to {artifacts.pptx_path}")

        # --- Per-slide rendering (render one, emit, then next — true progressive) ---
        slides_dir = Path(artifacts.run_dir) / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)
        from .contracts import SlideArtifact
        from .coder.pptx_recipe_renderer import (
            library_create_presentation,
            _apply_theme_defaults,
            render_slide_scaffold,
            _build_library_materials,
            SLIDE_WIDTH_INCHES,
            SLIDE_HEIGHT_INCHES,
        )
        slide_artifacts: list[SlideArtifact] = []
        total = len(refined_deck_ir.slides)
        deck_payload_base = refined_deck_ir.model_dump()
        for idx, slide_ir in enumerate(refined_deck_ir.slides):
            single_path = (slides_dir / f"slide_{idx:03d}.pptx").resolve()
            pres = library_create_presentation(SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES)
            _apply_theme_defaults(pres, refined_deck_ir)
            render_slide_scaffold(
                pres,
                deck_payload_base,
                slide_ir.model_dump(),
                _build_library_materials(refined_deck_ir, slide_ir),
            )
            pres.save(single_path)

            preview_path = ""
            if idx == 0 and total > 0:
                png_target = slides_dir / f"slide_{idx:03d}.png"
                preview_path = _render_slide_preview(single_path, png_target)

            sa = SlideArtifact(
                index=idx,
                slide_id=slide_ir.slide_id,
                title=slide_ir.title or "",
                pptx_path=str(single_path),
                preview_png_path=preview_path,
            )
            slide_artifacts.append(sa)
            _emit(
                progress_callback,
                "slide_rendered",
                {
                    "index": sa.index,
                    "total": total,
                    "slide_id": sa.slide_id,
                    "title": sa.title,
                    "pptx_path": sa.pptx_path,
                    "preview_png_path": sa.preview_png_path,
                },
            )
            log_lines.append(f"rendered slide {idx} to {single_path}")

        # Batch-fill previews for remaining slides (index > 0)
        for sa in slide_artifacts:
            if sa.preview_png_path:
                continue
            png_target = slides_dir / f"slide_{sa.index:03d}.png"
            sa.preview_png_path = _render_slide_preview(Path(sa.pptx_path), png_target)

        log_lines.append(f"rendered {total} per-slide pptx files to {slides_dir}")
        _emit(
            progress_callback,
            "rendering_done",
            {
                "total": total,
                "slide_artifacts": [sa.model_dump() for sa in slide_artifacts],
            },
        )

        pdf_path = ""
        preview_pdf_path = Path(artifacts.pptx_path).with_name("paper2ppt_code_preview.pdf")
        if req.include_pdf_preview:
            pdf_path = _export_pdf_preview(Path(artifacts.pptx_path), preview_pdf_path, log_lines)
        else:
            log_lines.append("pdf preview export skipped by request")

        final_artifacts = artifacts.model_copy(update={
            "pdf_path": pdf_path,
            "slides_dir": str(slides_dir),
            "slide_artifacts": slide_artifacts,
        })
        return final_artifacts
    except Exception as exc:
        log_lines.append(f"run failed: {exc.__class__.__name__}: {exc}")
        raise
    finally:
        write_log(Path(final_artifacts.log_path), log_lines)


def _export_pdf_preview(pptx_path: Path, pdf_path: Path, log_lines: list[str]) -> str:
    soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_bin:
        log_lines.append("pdf preview unavailable: libreoffice/soffice not found")
        return ""

    try:
        result = subprocess.run(
            [
                soffice_bin,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(pdf_path.parent),
                str(pptx_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log_lines.append(f"pdf preview export failed: {exc}")
        return ""

    generated_pdf = pdf_path.parent / f"{pptx_path.stem}.pdf"
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        log_lines.append(f"pdf preview export failed with code {result.returncode}: {stderr}")
        return ""

    if not generated_pdf.exists():
        log_lines.append("pdf preview export reported success but no pdf was created")
        return ""

    generated_pdf.replace(pdf_path)
    log_lines.append(f"wrote pdf preview to {pdf_path}")
    return str(pdf_path)


def _render_slide_preview(slide_pptx: Path, preview_png: Path) -> str:
    """Convert a single-slide pptx to a PNG preview via libreoffice + pdf2image.

    Returns the png path on success; returns "" on any failure (missing libreoffice,
    timeout, pdf conversion error, pdf2image failure, save failure). Cleans up the
    intermediate PDF and any partial PNG on failure.
    """
    soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_bin:
        return ""

    preview_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [soffice_bin, "--headless", "--convert-to", "pdf",
             "--outdir", str(preview_png.parent), str(slide_pptx)],
            check=False, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""

    pdf_path = preview_png.parent / f"{slide_pptx.stem}.pdf"
    if not pdf_path.exists():
        return ""

    try:
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), dpi=120, first_page=1, last_page=1)
        except Exception:
            return ""
        if not images:
            return ""
        try:
            images[0].save(preview_png, format="PNG")
        except Exception:
            try:
                preview_png.unlink(missing_ok=True)
            except Exception:
                pass
            return ""
        return str(preview_png)
    finally:
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass


def assemble_deck_from_final_ir(
    final_ir_path: Path,
    output_pptx: Path,
    *,
    include_pdf_preview: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Re-render a full deck pptx from a persisted final_ir.json.
    Does NOT merge single-slide pptx files; deterministic re-rendering via render_pptx.
    """
    import json as _json
    from .planner.ir_models import DeckIR

    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    payload = _json.loads(Path(final_ir_path).read_text(encoding="utf-8"))
    deck_ir = DeckIR.model_validate(payload)
    render_pptx(deck_ir, output_pptx)

    pdf_path = ""
    if include_pdf_preview:
        preview_pdf = output_pptx.with_name("paper2ppt_code_preview.pdf")
        pdf_path = _export_pdf_preview(output_pptx, preview_pdf, log_lines=[])

    _emit(
        progress_callback,
        "exporting_done",
        {"pptx_path": str(output_pptx), "pdf_path": pdf_path},
    )
    return {"pptx_path": str(output_pptx), "pdf_path": pdf_path}


def repatch_single_slide(
    result_path: str,
    slide_index: int,
    feedback: str,
    feedback_type: str = "auto",
    *,
    api_url: str = "",
    api_key: str = "",
    model: str = "",
    image_api_url: str = "",
    image_api_key: str = "",
    image_model: str = "",
    progress_callback: ProgressCallback | None = None,
) -> SlideArtifact:
    """Patch a single slide: re-enrich SlideIR, regenerate code, re-render PPTX + PNG.

    Reads final_ir.json from the existing run, patches the target slide in-place,
    regenerates its code (if LLM available), re-renders the single-slide PPTX and
    PNG preview, then writes the updated final_ir.json back to disk.
    """
    base_dir = Path(result_path).expanduser().resolve()
    artifact_base = resolve_artifact_base_dir(base_dir)
    run_dir = artifact_base / "code_runtime"
    final_ir_path = run_dir / "ir" / "final" / "final_ir.json"
    slides_dir = run_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    if not final_ir_path.exists():
        raise FileNotFoundError(f"final_ir.json not found at {final_ir_path}")

    payload = json.loads(final_ir_path.read_text(encoding="utf-8"))
    deck_ir = DeckIR.model_validate(payload)

    if slide_index < 0 or slide_index >= len(deck_ir.slides):
        raise IndexError(f"slide_index {slide_index} out of range (deck has {len(deck_ir.slides)} slides)")

    _emit(progress_callback, "patch_analyzing", {"slide_index": slide_index, "feedback": feedback})

    # Build LLM client (used for both SlideIR regeneration and code generation)
    llm_client = None
    if api_url and api_key and model:
        try:
            llm_client = LLMClient(api_key=api_key, api_base=api_url, model=model, client_type="llm")
        except Exception:
            pass

    # Build optional image client
    image_client = None
    if image_api_url and image_api_key and image_model:
        try:
            image_client = LLMClient(
                api_key=image_api_key,
                api_base=image_api_url,
                model=image_model,
                client_type="image",
            )
        except Exception:
            pass

    # Patch the SlideIR (LLM-based regeneration when llm_client is available)
    original_slide = deck_ir.slides[slide_index]
    patched_slide = patch_slide_ir(
        original_slide,
        deck_ir,
        feedback,
        feedback_type,
        result_path=artifact_base,
        llm_client=llm_client,
        image_client=image_client,
    )

    # Write patched slide back into deck_ir and persist
    updated_slides = list(deck_ir.slides)
    updated_slides[slide_index] = patched_slide
    updated_deck_ir = deck_ir.model_copy(update={"slides": updated_slides})
    write_json(final_ir_path, updated_deck_ir.model_dump())

    _emit(progress_callback, "patch_ir_done", {"slide_index": slide_index})

    if llm_client is not None:
        _emit(progress_callback, "patch_codegen", {"slide_index": slide_index})
        try:
            code_agent = PPTXCodeAgent(llm_client)
            function_name = f"build_slide_{slide_index:03d}"
            material_manifest = collect_materials(updated_deck_ir)
            slide_materials = material_manifest.get("slides", {}).get(patched_slide.slide_id, {})
            deck_payload = {k: v for k, v in updated_deck_ir.model_dump().items() if k != "slides"}
            code_agent.generate_slide_code_with_feedback(
                deck_payload,
                patched_slide.model_dump(),
                {"current_slide": slide_materials},
                function_name=function_name,
                artifact_dir=run_dir / "code" / "generated",
            )
        except Exception:
            pass  # codegen failure is non-fatal; fall through to recipe renderer

    # Re-render single-slide PPTX
    _emit(progress_callback, "patch_rendering", {"slide_index": slide_index})
    from .coder.pptx_recipe_renderer import (
        library_create_presentation,
        _apply_theme_defaults,
        render_slide_scaffold,
        _build_library_materials,
        SLIDE_WIDTH_INCHES,
        SLIDE_HEIGHT_INCHES,
    )

    single_path = (slides_dir / f"slide_{slide_index:03d}.pptx").resolve()
    pres = library_create_presentation(SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES)
    _apply_theme_defaults(pres, updated_deck_ir)
    deck_payload_base = updated_deck_ir.model_dump()
    render_slide_scaffold(
        pres,
        deck_payload_base,
        patched_slide.model_dump(),
        _build_library_materials(updated_deck_ir, patched_slide),
    )
    pres.save(single_path)

    # Re-render PNG preview
    png_target = slides_dir / f"slide_{slide_index:03d}.png"
    preview_path = _render_slide_preview(single_path, png_target)

    sa = SlideArtifact(
        index=slide_index,
        slide_id=patched_slide.slide_id,
        title=patched_slide.title or "",
        pptx_path=str(single_path),
        preview_png_path=preview_path,
        status="rendered",
    )
    _emit(
        progress_callback,
        "patch_done",
        {
            "index": sa.index,
            "slide_id": sa.slide_id,
            "title": sa.title,
            "pptx_path": sa.pptx_path,
            "preview_png_path": sa.preview_png_path,
        },
    )
    return sa
