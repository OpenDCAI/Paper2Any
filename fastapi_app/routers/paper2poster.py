from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from fastapi_app.config import settings
from fastapi_app.services.managed_api_service import resolve_model_name

router = APIRouter()


def get_service() -> Paper2PosterService:
    from fastapi_app.services.paper2poster_service import Paper2PosterService

    return Paper2PosterService()


@router.post("/paper2poster/generate")
async def generate_paper2poster(
    paper_file: UploadFile = File(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    model: str = Form("gpt-4o-2024-08-06"),
    vision_model: str = Form("gpt-4o-2024-08-06"),
    poster_width: float = Form(54.0),
    poster_height: float = Form(36.0),
    logo_file: Optional[UploadFile] = File(None),
    aff_logo_file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    email: Optional[str] = Form(None),
    service: Paper2PosterService = Depends(get_service),
):
    """Generate a paper poster and return browser-accessible output URLs."""
    return await service.generate(
        paper_file=paper_file,
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=resolve_model_name(
            model,
            managed_default=settings.PAPER2POSTER_DEFAULT_MODEL,
            fallback_default="gpt-4o",
        ),
        vision_model=resolve_model_name(
            vision_model,
            managed_default=settings.PAPER2POSTER_VISION_MODEL,
            fallback_default="gpt-4o",
        ),
        poster_width=poster_width,
        poster_height=poster_height,
        logo_file=logo_file,
        aff_logo_file=aff_logo_file,
        url=url,
        email=email,
    )
