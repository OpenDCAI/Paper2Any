from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from fastapi_app.schemas import (
    ErrorResponse,
    FullPipelineRequest,
    PageContentRequest,
    PPTGenerationRequest,
)
from fastapi_app.services.paper2ppt_service import Paper2PPTService
from fastapi_app.utils import validate_invite_code
from fastapi_app.middleware.billing_decorator import with_billing, with_dynamic_billing
from fastapi_app.utils.billing_utils import (
    extract_page_count_from_param,
    extract_page_count_from_pagecontent,
    extract_page_count_from_result,
)

# 注意：prefix 由 main.py 统一加 "/api/paper2ppt"
router = APIRouter(tags=["paper2ppt"])

# 从环境变量获取默认的 LLM 配置
DEFAULT_CHAT_API_URL = os.getenv("DF_API_URL", "https://api.apiyi.com/v1")
DEFAULT_API_KEY = os.getenv("DF_API_KEY", "")


def get_service() -> Paper2PPTService:
    return Paper2PPTService()


@router.post(
    "/pagecontent_json",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
@with_dynamic_billing(
    "paper2ppt",
    "pagecontent",
    lambda kwargs, result: extract_page_count_from_param(kwargs)
)
async def paper2ppt_pagecontent_json(
    request: Request,
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    invite_code: Optional[str] = Form(None),
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
    
    计费方式：单页价格 × page_count
    """
    # validate_invite_code(invite_code)

    # 注入默认值
    final_chat_api_url = chat_api_url or DEFAULT_CHAT_API_URL
    final_api_key = api_key or DEFAULT_API_KEY

    req = PageContentRequest(
        chat_api_url=final_chat_api_url,
        api_key=final_api_key,
        invite_code=invite_code,
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
    "/ppt_json",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
@with_dynamic_billing(
    "paper2ppt",
    "ppt",
    lambda kwargs, result: extract_page_count_from_pagecontent(kwargs)
)
async def paper2ppt_ppt_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    invite_code: Optional[str] = Form(None),
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
    
    计费方式：
    - 生成模式：单页价格 × pagecontent 页数
    - 编辑模式：单页价格 × 1（编辑单页）
    """
    # validate_invite_code(invite_code)

    # 注入默认值
    final_chat_api_url = chat_api_url or DEFAULT_CHAT_API_URL
    final_api_key = api_key or DEFAULT_API_KEY

    req = PPTGenerationRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=final_chat_api_url,
        api_key=final_api_key,
        invite_code=invite_code,
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
    "/full_json",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
@with_dynamic_billing(
    "paper2ppt",
    "full",
    lambda kwargs, result: extract_page_count_from_result(result)
)
async def paper2ppt_full_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    invite_code: Optional[str] = Form(None),
    # 输入：支持 text/pdf/pptx
    input_type: str = Form(...),  # 'text' | 'pdf' | 'pptx'
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    # 其他控制参数
    language: str = Form("zh"),
    aspect_ratio: str = Form("16:9"),
    style: str = Form(""),
    model: str = Form("gpt-5.1"),
    use_long_paper: str = Form("false"),
    service: Paper2PPTService = Depends(get_service),
):
    """
    Full pipeline：
    - paper2page_content -> paper2ppt
    - get_down 固定为 False（首次生成）
    
    计费方式：单页价格 × 返回结果中的页数
    """
    # validate_invite_code(invite_code)

    # 注入默认值
    final_chat_api_url = chat_api_url or DEFAULT_CHAT_API_URL
    final_api_key = api_key or DEFAULT_API_KEY

    req = FullPipelineRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=final_chat_api_url,
        api_key=final_api_key,
        invite_code=invite_code,
        input_type=input_type,
        file=None,  # UploadFile 不放进 Pydantic，作为单独参数传给 service
        text=text,
        language=language,
        aspect_ratio=aspect_ratio,
        style=style,
        model=model,
        use_long_paper=use_long_paper,
    )

    data = await service.run_full_pipeline(
        req=req,
        file=file,
        request=request,
    )
    return data
