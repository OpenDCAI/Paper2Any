from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from fastapi_app.config.settings import settings
from fastapi_app.services.managed_api_service import resolve_model_name

router = APIRouter(prefix="/image2drawio", tags=["image2drawio"])


class Image2DrawioResponse(BaseModel):
    success: bool
    xml_content: str = ""
    file_path: str = ""
    error: Optional[str] = None


@router.post("/generate", response_model=Image2DrawioResponse)
async def generate_image2drawio(
    image_file: UploadFile = File(...),
    chat_api_url: str = Form(settings.PAPER2DRAWIO_OCR_API_URL),
    api_key: str = Form(settings.PAPER2DRAWIO_OCR_API_KEY),
    model: str = Form("gpt-4o"),
    gen_fig_model: str = Form("gemini-3-pro-image-preview"),
    vlm_model: str = Form("qwen-vl-ocr-2025-11-20"),
    language: str = Form("zh"),
    email: Optional[str] = Form(None),
):
    from fastapi_app.services.image2drawio_service import Image2DrawioService

    service = Image2DrawioService()
    try:
        result = await service.generate_drawio(
            image_file=image_file,
            chat_api_url=chat_api_url,
            api_key=api_key,
            email=email,
            model=resolve_model_name(
                model,
                managed_default=settings.IMAGE2DRAWIO_DEFAULT_MODEL,
                fallback_default="gpt-4o",
            ),
            gen_fig_model=resolve_model_name(
                gen_fig_model,
                managed_default=settings.IMAGE2DRAWIO_DEFAULT_IMAGE_MODEL,
                fallback_default="gemini-3-pro-image-preview",
            ),
            vlm_model=resolve_model_name(
                vlm_model,
                managed_default=settings.IMAGE2DRAWIO_VLM_MODEL,
                fallback_default="qwen-vl-ocr-2025-11-20",
            ),
            language=language,
        )
        return Image2DrawioResponse(**result)
    except Exception as e:
        return Image2DrawioResponse(success=False, error=str(e))
