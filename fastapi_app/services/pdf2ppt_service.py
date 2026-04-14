from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz
from fastapi import File, UploadFile, HTTPException
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest
from fastapi_app.interprocess_lock import AsyncInterProcessSemaphore
from fastapi_app.services.managed_api_service import (
    resolve_image_generation_credentials,
    resolve_llm_credentials,
    resolve_model_name,
)
from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

VISUAL_WORKFLOW_LIMITER = AsyncInterProcessSemaphore("sam3_visual_workflows", limit=1)

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()

class PDF2PPTService:
    def __init__(self):
        pass

    def _count_pdf_pages(self, pdf_path: Path) -> int:
        try:
            with fitz.open(pdf_path) as document:
                return max(1, len(document))
        except Exception as exc:
            log.warning(f"[pdf2ppt] failed to count PDF pages for {pdf_path}: {exc}")
            return 1

    def _create_run_dir(self, email: Optional[str], task_type: str) -> Path:
        """
        为一次 pdf2ppt 请求创建独立目录：
            outputs/{email or 'default'}/{task_type}/{timestamp}/input/
        """
        ts = int(datetime.utcnow().timestamp())
        code = email or "default"
        run_dir = BASE_OUTPUT_DIR / code / task_type / str(ts)

        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        return run_dir

    async def generate_ppt(
        self,
        pdf_file: UploadFile,
        chat_api_url: Optional[str],
        api_key: Optional[str],
        email: Optional[str],
        use_ai_edit: bool,
        model: str,
        gen_fig_model: str,
        language: str,
        style: str,
        page_count: int,
    ) -> tuple[Path, int]:
        """
        执行 pdf2ppt 的业务逻辑，返回生成的 PPTX 文件路径
        """
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            chat_api_url,
            api_key,
            scope="pdf2ppt",
        )
        resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
            chat_api_url,
            api_key,
            scope="pdf2ppt",
        )
        model = resolve_model_name(
            model,
            managed_default=settings.PDF2PPT_DEFAULT_MODEL,
        )
        gen_fig_model = resolve_model_name(
            gen_fig_model,
            managed_default=settings.PDF2PPT_DEFAULT_IMAGE_MODEL,
        )
        # 0.5 如果启用 AI 增强，必须校验 API 配置
        if use_ai_edit:
            if not resolved_chat_api_url or not resolved_api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="When use_ai_edit is True, chat_api_url and api_key are required"
                )

        # 1. 基础参数校验
        if pdf_file is None:
            raise HTTPException(status_code=400, detail="pdf_file is required")

        # 2. 为本次请求创建独立目录
        run_dir = self._create_run_dir(email, "pdf2ppt")
        input_dir = run_dir / "input"

        original_name = pdf_file.filename or "uploaded.pdf"
        ext = Path(original_name).suffix or ".pdf"
        input_path = input_dir / f"input{ext}"

        content_bytes = await pdf_file.read()
        input_path.write_bytes(content_bytes)
        abs_pdf_path = input_path.resolve()
        actual_page_count = self._count_pdf_pages(abs_pdf_path)

        log.info(f"[pdf2ppt] received file saved to {abs_pdf_path}")

        # 3. 构造适配层请求
        wf_req = Paper2PPTRequest(
            input_type="PDF",
            input_content=str(abs_pdf_path),
            chat_api_url=resolved_chat_api_url or "",
            chat_api_key=resolved_api_key or "",
            api_key=resolved_api_key or "",
            image_api_url=resolved_image_api_url or "",
            image_api_key=resolved_image_api_key or "",
            model=model,
            gen_fig_model=gen_fig_model,
            language=language,
            style=style,
            page_count=page_count,
            email=email or "",
            use_ai_edit=use_ai_edit,
        )

        # 4. 调用 workflow（受信号量保护）
        async with VISUAL_WORKFLOW_LIMITER.hold():
            wf_resp = await run_pdf2ppt_wf_api(wf_req, result_path=run_dir)

        # 5. 获取生成的 PPT 路径
        ppt_path = Path(wf_resp.ppt_pptx_path or "")
        if not ppt_path.is_absolute():
            ppt_path = (PROJECT_ROOT / ppt_path).resolve()

        if not ppt_path.exists() or not ppt_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"generated PPT file not found or not a file: {ppt_path}",
            )

        log.info(f"[pdf2ppt] returning PPT file: {ppt_path}")
        
        return ppt_path, actual_page_count
