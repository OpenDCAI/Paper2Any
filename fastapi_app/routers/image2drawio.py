from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from fastapi_app.services.image2drawio_service import Image2DrawioService

router = APIRouter(prefix="/image2drawio", tags=["image2drawio"])


class Image2DrawioResponse(BaseModel):
    success: bool
    xml_content: str = ""
    file_path: str = ""
    error: Optional[str] = None


@router.post("/generate", response_model=Image2DrawioResponse)
async def generate_image2drawio(
    image_file: UploadFile = File(...),
    chat_api_url: str = Form(""),
    api_key: str = Form(""),
    model: str = Form("gpt-4o"),
    gen_fig_model: str = Form("gemini-3-pro-image-preview"),
    vlm_model: str = Form("qwen-vl-ocr-2025-11-20"),
    language: str = Form("zh"),
    email: Optional[str] = Form(None),
):
    service = Image2DrawioService()
    try:
        result = await service.generate_drawio(
            image_file=image_file,
            chat_api_url=chat_api_url,
            api_key=api_key,
            email=email,
            model=model,
            gen_fig_model=gen_fig_model,
            vlm_model=vlm_model,
            language=language,
        )
        return Image2DrawioResponse(**result)
    except Exception as e:
        return Image2DrawioResponse(success=False, error=str(e))
