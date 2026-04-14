from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from fastapi_app.config import settings
from fastapi_app.schemas import Paper2FigureResponse
from fastapi_app.services.managed_api_service import resolve_model_name


router = APIRouter(prefix="/paper2figure", tags=["paper2figure"])


def get_service() -> "Paper2AnyService":
    from fastapi_app.services.paper2any_service import Paper2AnyService

    return Paper2AnyService()


@router.get("/history")
async def list_paper2figure_history_files(
    request: Request,
    email: str,
    service: "Paper2AnyService" = Depends(get_service),
):
    """
    根据邮箱，列出该用户目录中的所有历史输出文件（pptx/png/svg）。
    """
    return await service.list_history_files(email, request)


@router.post("/generate")
async def generate_paper2figure(
    img_gen_model_name: Optional[str] = Form(None),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    input_type: str = Form(...),
    email: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    file_kind: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    graph_type: str = Form("model_arch"),
    language: str = Form("zh"),
    figure_complex: str = Form("easy"),
    style: str = Form("cartoon"),
    service: "Paper2AnyService" = Depends(get_service),
):
    """
    Paper2Figure 文件下载接口。
    """
    ppt_path = await service.generate_paper2figure(
        img_gen_model_name=resolve_model_name(
            img_gen_model_name,
            managed_default=settings.PAPER2FIGURE_IMAGE_MODEL,
            fallback_default=settings.PAPER2FIGURE_DEFAULT_IMAGE_MODEL,
        ),
        chat_api_url=chat_api_url,
        api_key=api_key,
        input_type=input_type,
        email=email,
        file=file,
        file_kind=file_kind,
        text=text,
        graph_type=graph_type,
        language=language,
        figure_complex=figure_complex,
        style=style,
    )

    return FileResponse(
        path=str(ppt_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=ppt_path.name,
    )


@router.post("/generate-json", response_model=Paper2FigureResponse)
async def generate_paper2figure_json(
    request: Request,
    img_gen_model_name: Optional[str] = Form(None),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    input_type: str = Form(...),
    email: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    file_kind: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    graph_type: str = Form("model_arch"),
    language: str = Form("zh"),
    style: str = Form("cartoon"),
    figure_complex: str = Form("easy"),
    resolution: str = Form("2K"),
    edit_prompt: Optional[str] = Form(None),
    tech_route_palette: str = Form(""),
    tech_route_template: str = Form(""),
    reference_image: Optional[UploadFile] = File(None),
    tech_route_edit_prompt: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    service: "Paper2AnyService" = Depends(get_service),
):
    """
    Paper2Figure JSON 接口。
    """
    resp_data = await service.generate_paper2figure_json(
        request=request,
        img_gen_model_name=resolve_model_name(
            img_gen_model_name,
            managed_default=settings.PAPER2FIGURE_IMAGE_MODEL,
            fallback_default=settings.PAPER2FIGURE_DEFAULT_IMAGE_MODEL,
        ),
        chat_api_url=chat_api_url,
        api_key=api_key,
        input_type=input_type,
        email=email,
        file=file,
        file_kind=file_kind,
        text=text,
        graph_type=graph_type,
        language=language,
        style=style,
        figure_complex=figure_complex,
        resolution=resolution,
        edit_prompt=edit_prompt,
        tech_route_palette=tech_route_palette,
        tech_route_template=tech_route_template,
        reference_image=reference_image,
        tech_route_edit_prompt=tech_route_edit_prompt,
        output_format=output_format,
    )
    return Paper2FigureResponse(**resp_data)
