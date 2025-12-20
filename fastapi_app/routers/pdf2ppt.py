from __future__ import annotations

import asyncio
from datetime import datetime
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse

from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger
from fastapi_app.utils import validate_invite_code  # 邀请码校验，与其它接口保持一致
from fastapi_app.schemas import Paper2PPTRequest
from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api

log = get_logger(__name__)

router = APIRouter()

# 控制重任务并发度，防止 GPU / 内存压力过大
task_semaphore = asyncio.Semaphore(1)

BASE_OUTPUT_DIR = Path("outputs")
PROJECT_ROOT = get_project_root()


def create_run_dir(task_type: str) -> Path:
    """
    为一次 pdf2ppt 请求创建独立目录：
        outputs/{task_type}/{timestamp}_{short_uuid}/input/
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    rid = uuid.uuid4().hex[:6]
    run_dir = BASE_OUTPUT_DIR / task_type / f"{ts}_{rid}"

    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    return run_dir


@router.post("/pdf2ppt/generate")
async def generate_pdf2ppt(
    request: Request,
    pdf_file: UploadFile = File(...),
    invite_code: Optional[str] = Form(None),
):
    """
    pdf2ppt 接口：

    - 前端通过 multipart/form-data 传入：
        - pdf_file: 待转换的 PDF 文件
        - invite_code: （可选）邀请码，用于权限控制/统计
    - 路由层负责：
        - 校验邀请码
        - 保存 pdf_file 到本地 outputs/pdf2ppt/.../input/input.pdf
        - 调用 run_pdf2ppt_wf_api（封装了 pdf2ppt_with_sam workflow）
        - 返回生成的 PPTX 文件（二进制下载）
    """
    # 0. 邀请码校验（如不需要，可去掉本行及参数）
    validate_invite_code(invite_code)

    # 1. 基础参数校验
    if pdf_file is None:
        raise HTTPException(status_code=400, detail="pdf_file is required")

    # 2. 为本次请求创建独立目录
    run_dir = create_run_dir("pdf2ppt")
    input_dir = run_dir / "input"

    original_name = pdf_file.filename or "uploaded.pdf"
    ext = Path(original_name).suffix or ".pdf"
    input_path = input_dir / f"input{ext}"

    content_bytes = await pdf_file.read()
    input_path.write_bytes(content_bytes)
    abs_pdf_path = input_path.resolve()

    log.info(f"[pdf2ppt] received file saved to {abs_pdf_path}")

    # 3. 构造适配层请求
    wf_req = Paper2PPTRequest(
        input_type="PDF",
        input_content=str(abs_pdf_path),
    )

    # 4. 调用 workflow（受信号量保护）
    async with task_semaphore:
        wf_resp = await run_pdf2ppt_wf_api(wf_req)

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

    return FileResponse(
        path=str(ppt_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=ppt_path.name,
    )
