from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Form, Request

from fastapi_app.config import settings
from fastapi_app.schemas import AssembleEditablePPTRequest, EditablePPTGenerationRequest, ErrorResponse, PatchSlideRequest
from fastapi_app.services.editable_ppt_service import EditablePPTService
from fastapi_app.services.managed_api_service import resolve_model_name
from fastapi_app.services.paper2ppt_code_task_service import Paper2PPTCodeTaskService

router = APIRouter(tags=["paper2ppt"])


def get_editable_ppt_service() -> EditablePPTService:
    return EditablePPTService()


def get_code_task_service() -> Paper2PPTCodeTaskService:
    return Paper2PPTCodeTaskService()


@router.post(
    "/paper2ppt/code/generate",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_code_generate(
    request: Request,
    result_path: str = Form(...),
    pagecontent: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    model: str = Form(settings.PAPER2PPT_CONTENT_MODEL),
    language: str = Form("zh"),
    style: str = Form(""),
    include_pdf_preview: bool = Form(True),
    service: EditablePPTService = Depends(get_editable_ppt_service),
):
    req = EditablePPTGenerationRequest(
        result_path=result_path,
        pagecontent=pagecontent,
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
        email=email,
        model=resolve_model_name(
            model,
            managed_default=settings.PAPER2PPT_CONTENT_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
        ),
        language=language,
        style=style,
        include_pdf_preview=include_pdf_preview,
    )
    return await service.generate_editable_ppt(req=req, request=request)


@router.post(
    "/paper2ppt/code/generate-task",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_code_generate_task(
    request: Request,
    result_path: str = Form(...),
    pagecontent: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    model: str = Form(settings.PAPER2PPT_CONTENT_MODEL),
    language: str = Form("zh"),
    style: str = Form(""),
    include_pdf_preview: bool = Form(False),
    task_service: Paper2PPTCodeTaskService = Depends(get_code_task_service),
):
    req = EditablePPTGenerationRequest(
        result_path=result_path, pagecontent=pagecontent,
        chat_api_url=chat_api_url, api_key=api_key,
        credential_scope=credential_scope, email=email,
        model=resolve_model_name(
            model, managed_default=settings.PAPER2PPT_CONTENT_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
        ),
        language=language, style=style, include_pdf_preview=include_pdf_preview,
    )
    return await task_service.submit_generate_slides_task(req=req, request=request)


@router.post(
    "/paper2ppt/code/assemble-task",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_code_assemble_task(
    request: Request,
    result_path: str = Form(...),
    include_pdf_preview: bool = Form(True),
    task_service: Paper2PPTCodeTaskService = Depends(get_code_task_service),
):
    req = AssembleEditablePPTRequest(
        result_path=result_path, include_pdf_preview=include_pdf_preview,
    )
    return await task_service.submit_assemble_task(req=req, request=request)


@router.get(
    "/paper2ppt/code/tasks/{task_id}",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_code_get_task(
    task_id: str,
    request: Request,
    task_service: Paper2PPTCodeTaskService = Depends(get_code_task_service),
):
    return task_service.get_task(task_id=task_id, request=request)


@router.post(
    "/paper2ppt/code/patch-slide-task",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_code_patch_slide_task(
    request: Request,
    result_path: str = Form(...),
    slide_index: int = Form(...),
    feedback: str = Form(...),
    feedback_type: str = Form("auto"),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    model: str = Form(settings.PAPER2PPT_CONTENT_MODEL),
    image_model: str = Form(settings.PAPER2PPT_IMAGE_GEN_MODEL),
    task_service: Paper2PPTCodeTaskService = Depends(get_code_task_service),
):
    req = PatchSlideRequest(
        result_path=result_path,
        slide_index=slide_index,
        feedback=feedback,
        feedback_type=feedback_type,
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
        model=resolve_model_name(
            model,
            managed_default=settings.PAPER2PPT_CONTENT_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
        ),
        image_model=image_model,
    )
    return await task_service.submit_patch_slide_task(req=req, request=request)
