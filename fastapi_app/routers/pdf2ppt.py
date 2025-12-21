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
from fastapi_app.utils import validate_invite_code
from fastapi_app.schemas import Paper2PPTRequest
from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api

log = get_logger(__name__)

router = APIRouter()

# 控制重任务并发度，防止 GPU / 内存压力过大
task_semaphore = asyncio.Semaphore(1)

BASE_OUTPUT_DIR = Path("outputs")
PROJECT_ROOT = get_project_root()


def create_run_dir(invite_code: str, task_type: str) -> Path:
    """
    为一次 pdf2ppt 请求创建独立目录：
        outputs/{invite_code}/{task_type}/{timestamp}/input/
    """
    ts = int(datetime.utcnow().timestamp())
    code = invite_code or "default"
    run_dir = BASE_OUTPUT_DIR / code / task_type / str(ts)

    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    return run_dir


@router.post("/pdf2ppt/generate")
async def generate_pdf2ppt(
    request: Request,
    pdf_file: UploadFile = File(...),
    # API 配置 - 必填
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    invite_code: str = Form(...),
    # 可选配置
    model: str = Form("gpt-4o"),
    gen_fig_model: str = Form("gemini-2.5-flash-image"),
    language: str = Form("zh"),
    style: str = Form("现代简约风格"),
    page_count: int = Form(8),
):
    """
    pdf2ppt 接口：一键将 PDF 转换为 PPT

    - 前端通过 multipart/form-data 传入：
        - pdf_file: 待转换的 PDF 文件
        - chat_api_url: LLM API URL
        - api_key: LLM API Key
        - invite_code: 邀请码
        - model: 语言模型（可选）
        - gen_fig_model: 图像生成模型（可选）
        - language: 语言（可选）
        - style: 风格描述（可选）
        - page_count: 生成页数（可选）
    - 返回：生成的 PPTX 文件（二进制下载）
    """
    # 0. 邀请码校验
    validate_invite_code(invite_code)

    # 1. 基础参数校验
    if pdf_file is None:
        raise HTTPException(status_code=400, detail="pdf_file is required")

    # 2. 为本次请求创建独立目录
    run_dir = create_run_dir(invite_code, "pdf2ppt")
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
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=model,
        gen_fig_model=gen_fig_model,
        language=language,
        style=style,
        page_count=page_count,
        invite_code=invite_code,
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
