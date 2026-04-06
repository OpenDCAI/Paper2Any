from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from fastapi_app.services.managed_api_service import resolve_llm_credentials
from fastapi_app.utils import _to_outputs_url
from fastapi_app.workflow_adapters.wa_paper2poster import run_paper2poster_generate_wf_api

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()


class Paper2PosterService:
    """paper2poster 请求编排与文件落盘。"""

    @staticmethod
    def _validate_poster_dimensions(width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="poster dimensions must be positive")
        ratio = width / height
        if ratio > 2.0 or ratio < 1.4:
            raise HTTPException(
                status_code=400,
                detail=f"Poster aspect ratio {ratio:.2f} is out of range. Please use a ratio between 1.4 and 2.0",
            )

    @staticmethod
    def _safe_filename(filename: Optional[str], fallback: str) -> str:
        candidate = Path(filename or fallback).name
        if candidate in {"", ".", ".."}:
            candidate = fallback
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
        return sanitized or fallback

    @staticmethod
    async def _save_upload(
        upload: UploadFile,
        dst: Path,
        *,
        allowed_ext: tuple[str, ...],
    ) -> Path:
        suffix = Path(upload.filename or dst.name).suffix.lower()
        if suffix not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"file type not allowed, expected one of {list(allowed_ext)}",
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(await upload.read())
        return dst

    def _create_run_dir(self, email: Optional[str]) -> Path:
        run_id = f"{int(time.time())}-{uuid4().hex[:8]}"
        owner = (email or "").strip() or "api"
        run_dir = PROJECT_ROOT / "outputs" / owner / "paper2poster" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    async def generate(
        self,
        *,
        paper_file: UploadFile,
        chat_api_url: str,
        api_key: str,
        model: str,
        vision_model: str,
        poster_width: float,
        poster_height: float,
        logo_file: Optional[UploadFile] = None,
        aff_logo_file: Optional[UploadFile] = None,
        url: str = "",
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            chat_api_url,
            api_key,
            scope="paper2poster",
        )
        self._validate_poster_dimensions(poster_width, poster_height)

        run_dir = self._create_run_dir(email)
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        paper_name = self._safe_filename(paper_file.filename, "paper.pdf")
        paper_path = await self._save_upload(
            paper_file,
            input_dir / paper_name,
            allowed_ext=(".pdf",),
        )

        logo_path = ""
        if logo_file is not None:
            logo_name = self._safe_filename(logo_file.filename, "logo.png")
            saved_logo = await self._save_upload(
                logo_file,
                input_dir / logo_name,
                allowed_ext=(".png", ".jpg", ".jpeg"),
            )
            logo_path = str(saved_logo)

        aff_logo_path = ""
        if aff_logo_file is not None:
            aff_logo_name = self._safe_filename(aff_logo_file.filename, "aff_logo.png")
            saved_aff_logo = await self._save_upload(
                aff_logo_file,
                input_dir / aff_logo_name,
                allowed_ext=(".png", ".jpg", ".jpeg"),
            )
            aff_logo_path = str(saved_aff_logo)

        log.info("[paper2poster] run_dir=%s paper=%s", run_dir, paper_path)
        result = await run_paper2poster_generate_wf_api(
            result_path=run_dir,
            paper_file=str(paper_path),
            chat_api_url=resolved_chat_api_url,
            api_key=resolved_api_key,
            model=model,
            vision_model=vision_model,
            poster_width=poster_width,
            poster_height=poster_height,
            logo_path=logo_path,
            aff_logo_path=aff_logo_path,
            url=url,
            email=email or "",
        )

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("message") or "Failed to generate poster")

        pptx_path = (result.get("output_pptx_path") or "").strip()
        if not pptx_path:
            raise HTTPException(status_code=500, detail="Poster workflow finished without a PPTX output")

        png_path = (result.get("output_png_path") or "").strip()
        return {
            "success": True,
            "pptx_url": _to_outputs_url(pptx_path),
            "png_url": _to_outputs_url(png_path) if png_path else None,
            "message": "Poster generated successfully",
        }
