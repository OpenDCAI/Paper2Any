from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from fastapi.responses import FileResponse

from dataflow_agent.logger import get_logger
from fastapi_app.services.pdf2ppt_service import PDF2PPTService

log = get_logger(__name__)

router = APIRouter()

def get_service() -> PDF2PPTService:
    return PDF2PPTService()

@router.post("/pdf2ppt/generate")
async def generate_pdf2ppt(
    request: Request,
    pdf_file: UploadFile = File(...),
    # API 配置 - 如果 use_ai_edit=True 则必填
    chat_api_url: str = Form(None),
    api_key: str = Form(None),
    invite_code: Optional[str] = Form(None),
    # 可选配置
    use_ai_edit: bool = Form(False),
    model: str = Form("gpt-4o"),
    gen_fig_model: str = Form("gemini-2.5-flash-image"),
    language: str = Form("zh"),
    style: str = Form("现代简约风格"),
    page_count: int = Form(8),
    service: PDF2PPTService = Depends(get_service),
):
    """
    pdf2ppt 接口：一键将 PDF 转换为 PPT

    - 前端通过 multipart/form-data 传入：
        - pdf_file: 待转换的 PDF 文件
        - invite_code: 邀请码
        - use_ai_edit: 是否启用 AI 增强（默认 False）
        - chat_api_url: LLM API URL（开启 AI 增强时必填）
        - api_key: LLM API Key（开启 AI 增强时必填）
        - model: 语言模型（可选）
        - gen_fig_model: 图像生成模型（可选）
        - language: 语言（可选）
        - style: 风格描述（可选）
        - page_count: 生成页数（可选）
    - 返回：生成的 PPTX 文件（二进制下载）
    """
    # 0. 邀请码校验 (Already commented out in original code, keeping it that way or user might want to uncomment)
    # validate_invite_code(invite_code)

    ppt_path = await service.generate_ppt(
        pdf_file=pdf_file,
        chat_api_url=chat_api_url,
        api_key=api_key,
        invite_code=invite_code,
        use_ai_edit=use_ai_edit,
        model=model,
        gen_fig_model=gen_fig_model,
        language=language,
        style=style,
        page_count=page_count,
    )

    return FileResponse(
        path=str(ppt_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=ppt_path.name,
    )
