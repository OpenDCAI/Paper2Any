"""
Paper2Drawio API 路由
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dataflow_agent.logger import get_logger
from fastapi_app.config.settings import settings

log = get_logger(__name__)
router = APIRouter(prefix="/paper2drawio", tags=["paper2drawio"])


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    """对话编辑请求"""
    current_xml: str
    message: str
    chat_history: List[Dict[str, str]] = []
    chat_api_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o"


class ExportRequest(BaseModel):
    """导出请求"""
    xml_content: str
    format: str = "drawio"  # "png" | "svg" | "drawio"
    filename: str = "diagram"


class GenerateResponse(BaseModel):
    """生成响应"""
    success: bool
    xml_content: str = ""
    file_path: str = ""
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """对话响应"""
    success: bool
    xml_content: str = ""
    message: str = ""
    error: Optional[str] = None


# ==================== Endpoints ====================

@router.post("/generate", response_model=GenerateResponse)
async def generate_diagram(
    request: Request,
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    model: Optional[str] = Form(None),
    enable_vlm_validation: Optional[bool] = Form(None),
    vlm_model: Optional[str] = Form(None),
    vlm_validation_max_retries: int = Form(3),
    input_type: str = Form("TEXT"),
    diagram_type: str = Form("auto"),
    diagram_style: str = Form("default"),
    language: str = Form("en"),
    email: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None),
):
    """
    生成 draw.io 图表
    """
    from fastapi_app.services.paper2drawio_service import Paper2DrawioService
    service = Paper2DrawioService()

    return await service.generate_diagram(
        request=request,
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=model or settings.PAPER2DRAWIO_DEFAULT_MODEL,
        enable_vlm_validation=(
            enable_vlm_validation
            if enable_vlm_validation is not None
            else settings.PAPER2DRAWIO_ENABLE_VLM_VALIDATION
        ),
        vlm_model=vlm_model or settings.PAPER2DRAWIO_VLM_MODEL,
        vlm_validation_max_retries=vlm_validation_max_retries,
        input_type=input_type,
        diagram_type=diagram_type,
        diagram_style=diagram_style,
        language=language,
        email=email,
        file=file,
        text_content=text_content,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_edit_diagram(
    request: Request,
    body: ChatRequest,
):
    """
    对话式编辑图表
    """
    from fastapi_app.services.paper2drawio_service import Paper2DrawioService
    service = Paper2DrawioService()

    return await service.chat_edit(
        request=request,
        current_xml=body.current_xml,
        message=body.message,
        chat_history=body.chat_history,
        chat_api_url=body.chat_api_url,
        api_key=body.api_key,
        model=body.model,
    )


@router.post("/export")
async def export_diagram(
    request: Request,
    body: ExportRequest,
):
    """
    导出图表为指定格式
    """
    from fastapi_app.services.paper2drawio_service import Paper2DrawioService
    service = Paper2DrawioService()

    return await service.export_diagram(
        request=request,
        xml_content=body.xml_content,
        format=body.format,
        filename=body.filename,
    )
