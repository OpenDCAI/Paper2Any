from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from fastapi_app.schemas import (
    ErrorResponse,
    FullPipelineRequest,
    OutlineRefineRequest,
    PageContentRequest,
    PPTGenerationRequest,
)
from fastapi_app.services.paper2ppt_service import Paper2PPTService
from dataflow_agent.utils.version_manager import ImageVersionManager
from fastapi_app.utils import _to_outputs_url

# 注意：prefix 由 main.py 统一加 "/api/paper2ppt"
router = APIRouter(tags=["paper2ppt"])


def get_service() -> Paper2PPTService:
    return Paper2PPTService()


@router.post(
    "/paper2ppt/page-content",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_pagecontent_json(
    request: Request,
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    email: Optional[str] = Form(None),
    # 输入相关：支持 text/pdf/pptx/topic
    input_type: str = Form(...),  # 'text' | 'pdf' | 'pptx' | 'topic'
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    # 可选控制参数（对 pagecontent 也可能有用）
    model: str = Form("gpt-5.1"),
    language: str = Form("zh"),
    style: str = Form(""),
    reference_img: Optional[UploadFile] = File(None),
    gen_fig_model: str = Form(...),
    page_count: int = Form(...),
    use_long_paper: str = Form("false"),
    service: Paper2PPTService = Depends(get_service),
):
    """
    只跑 paper2page_content，返回 pagecontent + result_path。
    """

    req = PageContentRequest(
        chat_api_url=chat_api_url,
        api_key=api_key,
        email=email,
        input_type=input_type,
        text=text,
        model=model,
        language=language,
        style=style,
        gen_fig_model=gen_fig_model,
        page_count=page_count,
        use_long_paper=use_long_paper,
    )

    data = await service.get_page_content(
        req=req,
        file=file,
        reference_img=reference_img,
        request=request,
    )
    return data


@router.post(
    "/paper2ppt/generate",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_ppt_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    email: Optional[str] = Form(None),
    # 控制参数
    style: str = Form(""),
    reference_img: Optional[UploadFile] = File(None),
    aspect_ratio: str = Form("16:9"),
    language: str = Form("en"),
    model: str = Form("gpt-5.1"),
    # 关键：是否进入编辑，是否已经有了 nano 结果，现在要进入页面逐个页面编辑
    get_down: str = Form("false"),  # 字符串形式，需要手动转换
    # 关键： 是否编辑完毕，也就是是否需要重新生成完整的 PPT
    all_edited_down: str = Form("false"),  # 字符串形式，需要手动转换
    # 复用上一次的输出目录（建议必传）
    result_path: str = Form(...),
    # 生成/编辑都需要 pagecontent（生成必传；编辑建议也传，便于回显）
    pagecontent: Optional[str] = Form(None),
    # 编辑参数（get_down=true 时必传）
    page_id: Optional[int] = Form(None),
    # 页面编辑提示词（get_down=true 时必传）
    edit_prompt: Optional[str] = Form(None),
    service: Paper2PPTService = Depends(get_service),
):
    """
    只跑 paper2ppt：
    - get_down=false：生成模式（需要 pagecontent）
    - get_down=true：编辑模式（需要 page_id(0-based) + edit_prompt，pagecontent 可选）
    """

    req = PPTGenerationRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        email=email,
        style=style,
        aspect_ratio=aspect_ratio,
        language=language,
        model=model,
        get_down=get_down,
        all_edited_down=all_edited_down,
        result_path=result_path,
        pagecontent=pagecontent,
        page_id=page_id,
        edit_prompt=edit_prompt,
    )

    data = await service.generate_ppt(
        req=req,
        reference_img=reference_img,
        request=request,
    )
    return data


@router.post(
    "/paper2ppt/outline-refine",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_outline_refine(
    request: Request,
    outline_feedback: str = Form(...),
    pagecontent: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    email: Optional[str] = Form(None),
    model: str = Form("gpt-5.1"),
    language: str = Form("zh"),
    result_path: Optional[str] = Form(None),
    service: Paper2PPTService = Depends(get_service),
):
    """Refine existing outline based on feedback, without re-parsing input."""
    req = OutlineRefineRequest(
        chat_api_url=chat_api_url,
        api_key=api_key,
        email=email,
        model=model,
        language=language,
        result_path=result_path,
        outline_feedback=outline_feedback,
        pagecontent=pagecontent,
    )
    data = await service.refine_outline(req=req, request=request)
    return data


@router.get(
    "/paper2ppt/version-history/{encoded_path}/{page_id}",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_version_history(
    encoded_path: str,
    page_id: int,
    request: Request,
    service: Paper2PPTService = Depends(get_service),
):
    """
    获取指定页面的版本历史。

    Args:
        encoded_path: Base64编码的result_path
        page_id: 页面索引（0-based）

    Returns:
        包含版本列表的字典
    """
    try:
        # 解码 result_path
        decoded_path = base64.b64decode(encoded_path).decode('utf-8')
        img_dir = Path(decoded_path) / "ppt_pages"

        if not img_dir.exists():
            raise HTTPException(status_code=404, detail="图片目录不存在")

        # 获取版本历史
        history = ImageVersionManager.get_version_history(img_dir, page_id)

        # 将文件路径转换为 URL
        for item in history:
            # 使用 _to_outputs_url 转换路径为完整的 HTTP URL
            item["imageUrl"] = _to_outputs_url(item["image_path"], request)

        return {"success": True, "versions": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取版本历史失败: {str(e)}")


@router.post(
    "/paper2ppt/revert-version",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def revert_to_version(
    request: Request,
    result_path: str = Form(...),
    page_id: int = Form(...),
    target_version: int = Form(...),
    service: Paper2PPTService = Depends(get_service),
):
    """
    将页面恢复到指定版本。

    Args:
        result_path: 结果路径
        page_id: 页面索引（0-based）
        target_version: 目标版本号

    Returns:
        包含当前图片URL和恢复版本号的字典
    """
    try:
        img_dir = Path(result_path) / "ppt_pages"

        if not img_dir.exists():
            raise HTTPException(status_code=404, detail="图片目录不存在")

        # 恢复到指定版本
        reverted_path = ImageVersionManager.revert_to_version(
            img_dir, page_id, target_version
        )

        if not reverted_path:
            raise HTTPException(status_code=404, detail="指定版本不存在")

        # 将绝对路径转换为浏览器可访问的 URL
        image_url = _to_outputs_url(reverted_path, request)

        return {
            "success": True,
            "currentImageUrl": image_url,
            "revertedToVersion": target_version
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复版本失败: {str(e)}")
