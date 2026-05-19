from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request

from fastapi_app.dependencies import AuthUser, get_optional_user
from fastapi_app.schemas import ImagePlaygroundRequest, ImagePlaygroundResponse
from fastapi_app.services.image_playground_service import ImagePlaygroundService

router = APIRouter(prefix="/image-playground", tags=["image_playground"])


def get_service() -> ImagePlaygroundService:
    return ImagePlaygroundService()


@router.post("/generate", response_model=ImagePlaygroundResponse)
async def generate_image_playground(
    body: ImagePlaygroundRequest,
    request: Request,
    user: Optional[AuthUser] = Depends(get_optional_user),
    service: ImagePlaygroundService = Depends(get_service),
) -> ImagePlaygroundResponse:
    resolved_user = user or getattr(request.state, "auth_user", None)
    result = await service.generate(
        prompt=body.prompt,
        model=body.model,
        chat_api_url=body.chat_api_url,
        api_key=body.api_key,
        template_key=body.template_key,
        domain_key=body.domain_key,
        aspect_ratio=body.aspect_ratio,
        resolution=body.resolution,
        size=body.size,
        quality=body.quality,
        batch_count=body.batch_count,
        request=request,
        user=resolved_user,
    )
    return ImagePlaygroundResponse(**result)
