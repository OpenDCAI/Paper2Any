from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import HTTPException, Request

from fastapi_app.config import settings
from fastapi_app.schemas import AssembleEditablePPTRequest, EditablePPTGenerationRequest
from fastapi_app.services.managed_api_service import (
    resolve_image_generation_credentials,
    resolve_llm_credentials,
    resolve_model_name,
)
from fastapi_app.utils import _to_outputs_url
from vendor.presentagent_runtime.contracts import EditablePPTRunRequest
from vendor.presentagent_runtime.runner import (
    assemble_deck_from_final_ir,
    run_from_pagecontent,
)


def _create_paper2ppt_service():
    from fastapi_app.services.paper2ppt_service import Paper2PPTService
    return Paper2PPTService()


class EditablePPTService:
    def __init__(self, paper2ppt_service=None) -> None:
        self._paper2ppt_service = paper2ppt_service or _create_paper2ppt_service()

    @staticmethod
    def _requires_agent_runtime(pagecontent: list[dict[str, Any]]) -> bool:
        for item in pagecontent or []:
            if not isinstance(item, dict):
                continue
            direct = item.get("asset_paths") or item.get("ppt_img_path")
            if direct:
                continue
            return True
        return False

    @staticmethod
    def _pick_requested_runtime_model(explicit_model: str, default_model: str, fallback_model: str) -> str:
        normalized = str(explicit_model or "").strip()
        if normalized and normalized != default_model:
            return normalized
        return fallback_model

    def _build_runtime_request(
        self,
        req: EditablePPTGenerationRequest,
        base_dir: Path,
        *,
        include_pdf_preview: bool,
    ) -> EditablePPTRunRequest:
        pagecontent = self._paper2ppt_service.parse_pagecontent_json(req.pagecontent)
        pagecontent = self._paper2ppt_service.normalize_pagecontent_asset_paths_for_runtime(pagecontent)
        requires_agent_runtime = self._requires_agent_runtime(pagecontent)

        credential_scope = self._paper2ppt_service.resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url, req.api_key, scope=credential_scope,
        )

        vlm_model = ""
        resolved_vlm_api_url = None
        resolved_vlm_api_key = None
        image_model = ""
        resolved_image_api_url = None
        resolved_image_api_key = None
        if requires_agent_runtime:
            resolved_vlm_api_url, resolved_vlm_api_key = resolve_llm_credentials(
                req.chat_api_url, req.api_key, scope=credential_scope,
            )
            resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
                req.chat_api_url, req.api_key, scope=credential_scope,
            )
            vlm_model = resolve_model_name(
                self._pick_requested_runtime_model(req.vlm_model, settings.PAPER2PPT_VLM_MODEL, req.model),
                managed_default=settings.PAPER2PPT_VLM_MODEL,
                fallback_default=settings.PAPER2PPT_CONTENT_MODEL,
            )
            image_model = resolve_model_name(
                self._pick_requested_runtime_model(req.image_model, settings.PAPER2PPT_IMAGE_GEN_MODEL, req.model),
                managed_default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
            )

        return EditablePPTRunRequest(
            result_path=str(base_dir),
            pagecontent=pagecontent,
            language=req.language,
            style=req.style,
            model=resolve_model_name(
                req.model,
                managed_default=settings.PAPER2PPT_CONTENT_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
            ),
            api_url=resolved_chat_api_url,
            api_key=resolved_api_key,
            vlm_model=vlm_model,
            vlm_api_url=resolved_vlm_api_url,
            vlm_api_key=resolved_vlm_api_key,
            image_model=image_model,
            image_api_url=resolved_image_api_url,
            image_api_key=resolved_image_api_key,
            enable_agent_planner=True,
            enable_material_resolution=True,
            enable_llm_codegen=False,
            include_pdf_preview=include_pdf_preview,
        )

    async def generate_editable_ppt_slides(
        self,
        req: EditablePPTGenerationRequest,
        request: Request | None,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        runtime_req = self._build_runtime_request(req, base_dir, include_pdf_preview=False)
        artifacts = await asyncio.to_thread(
            run_from_pagecontent, runtime_req, progress_callback=progress_callback,
        )
        return self._paper2ppt_service.normalize_editable_ppt_response(
            result_path=str(base_dir), artifacts=artifacts, request=request,
        )

    async def assemble_editable_ppt(
        self,
        req: AssembleEditablePPTRequest,
        request: Request | None,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        from vendor.presentagent_runtime.renderer.artifact_store import resolve_artifact_base_dir
        artifact_base = resolve_artifact_base_dir(base_dir)
        final_ir_path = artifact_base / "code_runtime" / "ir" / "final" / "final_ir.json"
        if not final_ir_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"final_ir.json not found; run generate-task first (expected at {final_ir_path})",
            )

        output_pptx = base_dir / "paper2ppt_code_editable.pptx"

        def _run_assemble() -> dict:
            return assemble_deck_from_final_ir(
                final_ir_path=final_ir_path,
                output_pptx=output_pptx,
                include_pdf_preview=bool(req.include_pdf_preview),
                progress_callback=progress_callback,
            )

        assembled = await asyncio.to_thread(_run_assemble)
        pptx_abs = assembled.get("pptx_path") or ""
        pdf_abs = assembled.get("pdf_path") or ""
        response: Dict[str, Any] = {
            "success": True,
            "result_path": str(base_dir),
            "ppt_pptx_path": pptx_abs,
            "ppt_pdf_path": pdf_abs,
            "error": "",
        }
        if request is not None:
            response["ppt_pptx_path"] = _to_outputs_url(pptx_abs, request) if pptx_abs else ""
            response["ppt_pdf_path"] = _to_outputs_url(pdf_abs, request) if pdf_abs else ""
        return response

    async def generate_editable_ppt(
        self,
        req: EditablePPTGenerationRequest,
        request: Request | None,
    ) -> Dict[str, Any]:
        slides_response = await self.generate_editable_ppt_slides(req=req, request=request)
        if not req.include_pdf_preview:
            return slides_response
        assemble_req = AssembleEditablePPTRequest(
            result_path=slides_response["result_path"], include_pdf_preview=True,
        )
        assembled = await self.assemble_editable_ppt(req=assemble_req, request=request)
        merged = dict(slides_response)
        merged["ppt_pptx_path"] = assembled.get("ppt_pptx_path") or merged.get("ppt_pptx_path", "")
        merged["ppt_pdf_path"] = assembled.get("ppt_pdf_path") or merged.get("ppt_pdf_path", "")
        return merged
