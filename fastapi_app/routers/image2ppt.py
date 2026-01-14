from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from fastapi.responses import FileResponse

from dataflow_agent.logger import get_logger
from fastapi_app.services.image2ppt_service import Image2PPTService
from fastapi_app.middleware.billing_decorator import with_hybrid_billing

log = get_logger(__name__)

router = APIRouter()

def get_service() -> Image2PPTService:
    return Image2PPTService()

@router.post("/image2ppt/generate")
@with_hybrid_billing("image2ppt", "image2ppt")
async def generate_image2ppt(
    request: Request,
    image_file: UploadFile = File(...),
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
    page_count: int = Form(1), # 图片通常对应单页，但如果切割可能多页，暂设默认1
    service: Image2PPTService = Depends(get_service),
):
    """
    image2ppt 接口：将 Image 转换为 PPT (Editable)

    - 前端通过 multipart/form-data 传入：
        - image_file: 待转换的图片文件
        - invite_code: 邀请码
        - use_ai_edit: 是否启用 AI 增强（默认 False）
        - chat_api_url: LLM API URL（开启 AI 增强时必填）
        - api_key: LLM API Key（开启 AI 增强时必填）
    - 返回：生成的 PPTX 文件（二进制下载）
    
    计费方式：
    - 不使用 AI：基础价格
    - 使用 AI：基础价格 + (AI单页价格 × page_count)
    """

    ppt_path = await service.generate_ppt(
        image_file=image_file,
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
