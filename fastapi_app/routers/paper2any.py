from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, Body
from fastapi.responses import FileResponse
from fastapi_app.schemas import Paper2FigureResponse, VerifyLlmRequest, VerifyLlmResponse
from fastapi_app.services.paper2any_service import Paper2AnyService
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


def get_service() -> Paper2AnyService:
    return Paper2AnyService()


@router.post("/system/verify-llm", response_model=VerifyLlmResponse)
async def verify_llm_connection(
    req: VerifyLlmRequest = Body(...),
    service: Paper2AnyService = Depends(get_service),
):
    """
    Verify LLM connection by sending a simple 'Hi' message from the backend.
    """
    return await service.verify_llm_connection(req)


@router.get("/paper2figure/history")
async def list_paper2figure_history_files(
    request: Request,
    invite_code: str,
    service: Paper2AnyService = Depends(get_service),
):
    """
    根据邀请码，列出该邀请码下面二级子目录中的所有历史输出文件（pptx/png/svg）
    """
    return await service.list_history_files(invite_code, request)


@router.post("/paper2figure/generate")
async def generate_paper2figure(
    img_gen_model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    input_type: str = Form(...),
    invite_code: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    file_kind: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    graph_type: str = Form("model_arch"),   # 'model_arch' | 'tech_route' | 'exp_data'
    language: str = Form("zh"),
    figure_complex: str = Form("easy"),
    style: str = Form("cartoon"),
    service: Paper2AnyService = Depends(get_service),
):
    """
    Paper2Graph 接口（带邀请码校验 + workflow 调用）
    """
    ppt_path = await service.generate_paper2figure(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        input_type=input_type,
        invite_code=invite_code,
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


@router.post("/paper2figure/generate-json", response_model=Paper2FigureResponse)
async def generate_paper2figure_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    input_type: str = Form(...),  # 'file' | 'text' | 'image'
    invite_code: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    file_kind: Optional[str] = Form(None),  # 'pdf' | 'image'
    text: Optional[str] = Form(None),
    graph_type: str = Form("model_arch"),   # 'model_arch' | 'tech_route' | 'exp_data'
    language: str = Form("zh"),
    style: str = Form("cartoon"),
    figure_complex: str = Form("easy"),
    edit_prompt: Optional[str] = Form(None),
    service: Paper2AnyService = Depends(get_service),
):
    """
    Paper2Graph JSON 接口
    """
    resp_data = await service.generate_paper2figure_json(
        request=request,
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        input_type=input_type,
        invite_code=invite_code,
        file=file,
        file_kind=file_kind,
        text=text,
        graph_type=graph_type,
        language=language,
        style=style,
        figure_complex=figure_complex,
        edit_prompt=edit_prompt,
    )
    
    return Paper2FigureResponse(**resp_data)
