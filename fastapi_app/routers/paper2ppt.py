from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from fastapi_app.schemas import (
    ErrorResponse,
    FrontendPPTExportRequest,
    FrontendPPTGenerationRequest,
    FrontendPPTReviewRequest,
    FullPipelineRequest,
    OutlineRefineRequest,
    PageContentRequest,
    PPTGenerationRequest,
)
from dataflow_agent.utils.version_manager import ImageVersionManager
from fastapi_app.services.billing_service import BillingService
from fastapi_app.utils import _to_outputs_url, resolve_outputs_path

# 注意：prefix 由 main.py 统一加 "/api/paper2ppt"
router = APIRouter(tags=["paper2ppt"])
_SYNC_SUBMISSION_WINDOW_SECONDS = 20


def get_service() -> Paper2PPTService:
    from fastapi_app.services.paper2ppt_service import Paper2PPTService

    return Paper2PPTService()


def get_task_service() -> Paper2PPTTaskService:
    from fastapi_app.services.paper2ppt_task_service import Paper2PPTTaskService

    return Paper2PPTTaskService()


def get_frontend_service() -> Paper2PPTFrontendService:
    from fastapi_app.services.paper2ppt_frontend_service import Paper2PPTFrontendService

    return Paper2PPTFrontendService()


def _is_truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _pagecontent_count(raw: Any) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _parse_page_index_list(raw: Any, *, max_count: int | None = None) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    max_index = max_count - 1 if max_count is not None and max_count > 0 else None
    indices: set[int] = set()
    for item in payload:
        try:
            value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        if max_index is not None and value > max_index:
            continue
        indices.add(value)
    return sorted(indices)


def _build_submission_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_authenticated_event_key(
    request: Request,
    workflow_type: str,
    submission_key: str,
) -> Optional[str]:
    user = getattr(request.state, "auth_user", None)
    if not user:
        return None
    user_id = str(getattr(user, "id", "") or "").strip()
    if not user_id:
        return None
    time_bucket = int(time.time() // _SYNC_SUBMISSION_WINDOW_SECONDS)
    return f"workflow_{workflow_type}_{user_id}_{time_bucket}_{submission_key}"


def _consume_workflow_before_execute(
    request: Request,
    *,
    workflow_type: str,
    amount: int,
    submission_payload: Dict[str, Any],
) -> None:
    submission_key = _build_submission_key(submission_payload)
    event_key = _build_authenticated_event_key(request, workflow_type, submission_key)
    BillingService().consume_workflow(
        workflow_type=workflow_type,
        amount=max(1, int(amount)),
        user=getattr(request.state, "auth_user", None),
        guest_id=getattr(request.state, "guest_id", None),
        event_key=event_key,
    )


def _consume_paper2ppt_generate_charge(request: Request, req: PPTGenerationRequest) -> None:
    if _is_truthy(req.all_edited_down):
        return

    get_down = _is_truthy(req.get_down)
    regenerate_from_outline = _is_truthy(req.regenerate_from_outline)
    if get_down:
        if req.page_id is None:
            return
        if not regenerate_from_outline and not str(req.edit_prompt or "").strip():
            return
        amount = 1
    else:
        total_pages = _pagecontent_count(req.pagecontent)
        skip_count = len(_parse_page_index_list(req.skip_pages, max_count=total_pages))
        amount = max(0, total_pages - skip_count)
        if amount <= 0:
            return

    payload = {
        "path": "/api/v1/paper2ppt/generate",
        "result_path": str(req.result_path or "").strip(),
        "pagecontent": str(req.pagecontent or "").strip(),
        "get_down": get_down,
        "all_edited_down": _is_truthy(req.all_edited_down),
        "page_id": req.page_id,
        "edit_prompt": str(req.edit_prompt or "").strip(),
        "regenerate_from_outline": regenerate_from_outline,
        "style": str(req.style or "").strip(),
        "model": str(req.model or "").strip(),
        "language": str(req.language or "").strip(),
        "aspect_ratio": str(req.aspect_ratio or "").strip(),
        "img_gen_model_name": str(req.img_gen_model_name or "").strip(),
        "image_resolution": str(req.image_resolution or "").strip(),
        "skip_pages": _parse_page_index_list(req.skip_pages),
    }
    _consume_workflow_before_execute(
        request,
        workflow_type="paper2ppt",
        amount=amount,
        submission_payload=payload,
    )


def _consume_paper2ppt_frontend_charge(request: Request, req: FrontendPPTGenerationRequest) -> None:
    pagecontent_count = _pagecontent_count(req.pagecontent)
    if pagecontent_count <= 0:
        return

    if req.page_id is not None:
        amount = 1
    else:
        skip_count = len(_parse_page_index_list(req.skip_slides, max_count=pagecontent_count))
        pages_to_generate = max(0, pagecontent_count - skip_count)
        per_page = 2 if bool(req.include_images) else 1
        amount = pages_to_generate * per_page
        if amount <= 0:
            return

    payload = {
        "path": "/api/v1/paper2ppt/frontend/generate",
        "result_path": str(req.result_path or "").strip(),
        "pagecontent": str(req.pagecontent or "").strip(),
        "page_id": req.page_id,
        "edit_prompt": str(req.edit_prompt or "").strip(),
        "current_slide": str(req.current_slide or "").strip(),
        "style": str(req.style or "").strip(),
        "model": str(req.model or "").strip(),
        "language": str(req.language or "").strip(),
        "include_images": bool(req.include_images),
        "image_style": str(req.image_style or "").strip(),
        "image_model": str(req.image_model or "").strip(),
        "skip_slides": _parse_page_index_list(req.skip_slides),
    }
    _consume_workflow_before_execute(
        request,
        workflow_type="paper2ppt",
        amount=amount,
        submission_payload=payload,
    )


@router.post(
    "/paper2ppt/page-content",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_pagecontent_json(
    request: Request,
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
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
    # 当 input_type=pdf 时，按“幻灯片图片”模式解析
    pdf_as_slides: str = Form("false"),
    # PPT/PDF 转图片时的渲染 DPI（None 表示默认）
    render_dpi: Optional[int] = Form(None),
    service: Paper2PPTService = Depends(get_service),
):
    """
    只跑 paper2page_content，返回 pagecontent + result_path。
    """

    req = PageContentRequest(
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
        email=email,
        input_type=input_type,
        text=text,
        model=model,
        language=language,
        style=style,
        gen_fig_model=gen_fig_model,
        page_count=page_count,
        use_long_paper=use_long_paper,
        pdf_as_slides=pdf_as_slides,
        render_dpi=render_dpi,
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
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    # 控制参数
    style: str = Form(""),
    reference_img: Optional[UploadFile] = File(None),
    aspect_ratio: str = Form("16:9"),
    language: str = Form("en"),
    model: str = Form("gpt-5.1"),
    image_resolution: Optional[str] = Form(None),
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
    regenerate_from_outline: str = Form("false"),
    skip_pages: Optional[str] = Form(None),
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
        credential_scope=credential_scope,
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
        regenerate_from_outline=regenerate_from_outline,
        image_resolution=image_resolution,
        skip_pages=skip_pages,
    )

    _consume_paper2ppt_generate_charge(request, req)

    data = await service.generate_ppt(
        req=req,
        reference_img=reference_img,
        request=request,
    )
    return data


@router.post(
    "/paper2ppt/generate-task",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_generate_task(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    style: str = Form(""),
    reference_img: Optional[UploadFile] = File(None),
    aspect_ratio: str = Form("16:9"),
    language: str = Form("en"),
    model: str = Form("gpt-5.1"),
    image_resolution: Optional[str] = Form(None),
    get_down: str = Form("false"),
    all_edited_down: str = Form("false"),
    result_path: str = Form(...),
    pagecontent: Optional[str] = Form(None),
    page_id: Optional[int] = Form(None),
    edit_prompt: Optional[str] = Form(None),
    regenerate_from_outline: str = Form("false"),
    skip_pages: Optional[str] = Form(None),
    task_service: Paper2PPTTaskService = Depends(get_task_service),
):
    req = PPTGenerationRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
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
        regenerate_from_outline=regenerate_from_outline,
        image_resolution=image_resolution,
        skip_pages=skip_pages,
    )
    return await task_service.submit_generate_task(req=req, reference_img=reference_img, request=request)


@router.get(
    "/paper2ppt/tasks/{task_id}",
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_get_task(
    task_id: str,
    request: Request,
    task_service: Paper2PPTTaskService = Depends(get_task_service),
):
    return task_service.get_task(task_id=task_id, request=request)


@router.post(
    "/paper2ppt/outline-refine",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_outline_refine(
    request: Request,
    outline_feedback: str = Form(...),
    pagecontent: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
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
        credential_scope=credential_scope,
        email=email,
        model=model,
        language=language,
        result_path=result_path,
        outline_feedback=outline_feedback,
        pagecontent=pagecontent,
    )
    data = await service.refine_outline(req=req, request=request)
    return data


@router.post(
    "/paper2ppt/frontend/generate",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_frontend_generate(
    request: Request,
    result_path: str = Form(...),
    pagecontent: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    model: str = Form("gpt-5.1"),
    language: str = Form("zh"),
    style: str = Form(""),
    include_images: bool = Form(False),
    image_style: str = Form("academic_illustration"),
    image_model: Optional[str] = Form(None),
    page_id: Optional[int] = Form(None),
    edit_prompt: Optional[str] = Form(None),
    current_slide: Optional[str] = Form(None),
    skip_slides: Optional[str] = Form(None),
    service: Paper2PPTFrontendService = Depends(get_frontend_service),
):
    req = FrontendPPTGenerationRequest(
        result_path=result_path,
        pagecontent=pagecontent,
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
        email=email,
        model=model,
        language=language,
        style=style,
        include_images=include_images,
        image_style=image_style,
        image_model=image_model,
        page_id=page_id,
        edit_prompt=edit_prompt,
        current_slide=current_slide,
        skip_slides=skip_slides,
    )
    _consume_paper2ppt_frontend_charge(request, req)
    return await service.generate_slides(req=req, request=request)


@router.post(
    "/paper2ppt/frontend/export",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_frontend_export(
    request: Request,
    result_path: str = Form(...),
    slides: str = Form(...),
    screenshots: list[UploadFile] = File(...),
    service: Paper2PPTFrontendService = Depends(get_frontend_service),
):
    req = FrontendPPTExportRequest(result_path=result_path, slides=slides)
    return await service.export_slides(req=req, screenshots=screenshots, request=request)


@router.post(
    "/paper2ppt/frontend/review",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_frontend_review(
    result_path: str = Form(...),
    slide: str = Form(...),
    screenshot: UploadFile = File(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    credential_scope: Optional[str] = Form(None),
    language: str = Form("zh"),
    layout_issues: Optional[str] = Form(None),
    service: Paper2PPTFrontendService = Depends(get_frontend_service),
):
    req = FrontendPPTReviewRequest(
        result_path=result_path,
        slide=slide,
        chat_api_url=chat_api_url,
        api_key=api_key,
        credential_scope=credential_scope,
        language=language,
        layout_issues=layout_issues,
    )
    return await service.review_slide(req=req, screenshot=screenshot)


@router.post(
    "/paper2ppt/frontend/upload-asset",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def paper2ppt_frontend_upload_asset(
    request: Request,
    result_path: str = Form(...),
    asset_key: str = Form(...),
    file: UploadFile = File(...),
    service: Paper2PPTFrontendService = Depends(get_frontend_service),
):
    return await service.upload_asset(
        result_path=result_path,
        asset_key=asset_key,
        upload=file,
        request=request,
    )


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
        img_dir = resolve_outputs_path(decoded_path, must_exist=True, allow_dirs=True) / "ppt_pages"

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
        img_dir = resolve_outputs_path(result_path, must_exist=True, allow_dirs=True) / "ppt_pages"

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
