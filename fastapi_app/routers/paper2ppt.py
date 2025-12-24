from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse
from fastapi_app.utils import _from_outputs_url, _to_outputs_url, validate_invite_code
from fastapi_app.workflow_adapters.wa_paper2ppt import (
    run_paper2page_content_wf_api,
    run_paper2ppt_full_pipeline,
    run_paper2ppt_wf_api,
)
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

router = APIRouter()

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = Path("outputs")


def _create_run_dir_for_paper2ppt(invite_code: Optional[str]) -> Path:
    """
    为一次 paper2ppt 请求创建 input 目录，结构：
        outputs/{invite_code or 'default'}/paper2ppt_input/
    """
    code = invite_code or "default"
    run_dir = PROJECT_ROOT / BASE_OUTPUT_DIR / code / "paper2ppt_input"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_upload_to_input_dir(input_dir: Path, file: UploadFile, file_kind: str) -> Path:
    original_name = file.filename or "uploaded"
    ext = Path(original_name).suffix or ""

    if file_kind == "pdf":
        ext = ".pdf"
    elif file_kind == "pptx":
        ext = ".pptx"
    else:
        raise HTTPException(status_code=400, detail="file_kind must be 'pdf' or 'pptx'")

    input_path = (input_dir / f"input{ext}").resolve()
    # 注意：UploadFile.read() 是 async
    return input_path


def _collect_output_files_as_urls(result_path: str, request: Request) -> list[str]:
    if not result_path:
        return []

    root = Path(result_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if not root.exists():
        return []

    urls: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".pptx", ".png"}:
            urls.append(_to_outputs_url(str(p), request))
    return urls


def _parse_pagecontent_json(pagecontent_json: str) -> list[dict]:
    try:
        obj = json.loads(pagecontent_json)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid pagecontent json: {e}") from e

    if not isinstance(obj, list):
        raise HTTPException(status_code=400, detail="pagecontent must be a JSON list")

    # 只允许 list[dict]（workflow 里也能处理直接图片路径，但这里按你的要求走结构化 pagecontent）
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            raise HTTPException(status_code=400, detail=f"pagecontent[{i}] must be an object(dict)")
    return obj


@router.post("/pagecontent_json", response_model=Paper2PPTResponse)
async def paper2ppt_pagecontent_json(
    request: Request,
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
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
):
    """
    只跑 paper2page_content，返回 pagecontent + result_path。

    - 必传：chat_api_url, api_key, input_type
    - input_type='pdf'：需要 file
    - input_type='pptx'：需要 file
    - input_type='text'：需要 text
    - 可选：reference_img（风格参考图）
    """
    # validate_invite_code(invite_code)

    norm_input_type = input_type.lower().strip()

    run_dir = _create_run_dir_for_paper2ppt(invite_code)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    wf_input_type: str
    wf_input_content: str

    reference_img_path: Optional[Path] = None
    if reference_img is not None:
        ref_ext = Path(reference_img.filename or "").suffix or ".png"
        reference_img_path = (input_dir / f"reference{ref_ext}").resolve()
        reference_img_path.write_bytes(await reference_img.read())

    if norm_input_type == "pdf":
        if file is None:
            raise HTTPException(status_code=400, detail="file is required when input_type is 'pdf'")

        input_path = (input_dir / "input.pdf").resolve()
        input_path.write_bytes(await file.read())

        wf_input_type = "PDF"
        wf_input_content = str(input_path)
    elif norm_input_type in ("ppt", "pptx"):
        if file is None:
            raise HTTPException(status_code=400, detail="file is required when input_type is 'pptx'")

        input_path = (input_dir / "input.pptx").resolve()
        input_path.write_bytes(await file.read())

        wf_input_type = "PPT"
        wf_input_content = str(input_path)
    elif norm_input_type == "text":
        if not text:
            raise HTTPException(status_code=400, detail="text is required when input_type is 'text'")

        (input_dir / "input.txt").resolve().write_text(text, encoding="utf-8")
        wf_input_type = "TEXT"
        wf_input_content = text
    elif norm_input_type == "topic":
        if not text:
            raise HTTPException(status_code=400, detail="text (topic) is required when input_type is 'topic'")
            
        (input_dir / "input_topic.txt").resolve().write_text(text, encoding="utf-8")
        wf_input_type = "TOPIC"
        wf_input_content = text
    else:
        raise HTTPException(status_code=400, detail="invalid input_type, must be one of: text, pdf, pptx, topic")

    p2ppt_req = Paper2PPTRequest(
        language=language,
        chat_api_url=chat_api_url,
        chat_api_key=api_key,
        api_key=api_key,
        model=model,
        # pagecontent 阶段不会用到 gen_fig_model，但保持字段完整
        gen_fig_model="",
        input_type=wf_input_type,
        input_content=wf_input_content,
        style=style,
        reference_img=str(reference_img_path) if reference_img_path is not None else "",
        invite_code=invite_code or "",
        page_count=page_count,
    )

    resp = await run_paper2page_content_wf_api(p2ppt_req)

    # 等待图片保存完成（PPT 转图片可能需要一点时间）
    import asyncio
    await asyncio.sleep(10)

    # 补齐 all_output_files（便于前端调试/查看 MinerU 输出等）
    resp.all_output_files = _collect_output_files_as_urls(resp.result_path, request)
    return resp


@router.post("/ppt_json", response_model=Paper2PPTResponse)
async def paper2ppt_ppt_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    invite_code: Optional[str] = Form(None),
    # 控制参数
    style: str = Form(...),
    aspect_ratio: str = Form("16:9"),
    language: str = Form("en"),
    model: str = Form("gpt-5.1"),
    # 关键：是否进入编辑，是否已经有了nano结果，现在要进入页面逐个页面编辑
    get_down: str = Form("false"),  # 字符串形式，需要手动转换
    # 关键： 是否编辑完毕，也就是是否需要重新生成完整的 PPT
    all_edited_down: str = Form("false"),  # 字符串形式，需要手动转换
    # 复用上一次的输出目录（建议必传）
    result_path: str = Form(...),
    # 生成/编辑都需要 pagecontent（生成必传；编辑建议也传，便于回显）
    pagecontent: Optional[str] = Form(None),
    # 编辑参数（get_down=true 时必传）
    page_id: Optional[int] = Form(None),
    # 页面2的编辑提示词（get_down=true 时必传）
    edit_prompt: Optional[str] = Form(None),
):
    """
    只跑 paper2ppt：
    - get_down=false：生成模式（需要 pagecontent）
    - get_down=true：编辑模式（需要 page_id(0-based) + edit_prompt，pagecontent 可选）
    """
    # validate_invite_code(invite_code)

    # 转换字符串形式的布尔值
    get_down_bool = get_down.lower() in ("true", "1", "yes")
    all_edited_down_bool = all_edited_down.lower() in ("true", "1", "yes")
    
    log.info(
        f"[ppt_json] Request params: get_down='{get_down}' -> {get_down_bool}, "
        f"all_edited_down='{all_edited_down}' -> {all_edited_down_bool}, "
        f"result_path={result_path}, page_id={page_id}, "
        f"pagecontent_length={len(pagecontent) if pagecontent else 0}"
    )

    # 仅在有值时解析 pagecontent；编辑模式下允许不传。
    if pagecontent is not None:
        pc = _parse_pagecontent_json(pagecontent)
        # 尝试将前端传回的 URL 转换回本地绝对路径
        for item in pc:
            # 常见包含路径的字段
            for key in ["ppt_img_path", "asset_ref"]:
                if key in item and item[key]:
                    item[key] = _from_outputs_url(item[key])
            # 如果有 generated_img_path
            if "generated_img_path" in item and item["generated_img_path"]:
                item["generated_img_path"] = _from_outputs_url(item["generated_img_path"])
    else:
        pc = []

    if get_down_bool:
        # 编辑模式：只要求 page_id + edit_prompt，pagecontent 可为空（便于纯根据 result_path + page_id 编辑）
        if page_id is None:
            raise HTTPException(status_code=400, detail="page_id is required when get_down=true")
        if not (edit_prompt or "").strip():
            raise HTTPException(status_code=400, detail="edit_prompt is required when get_down=true")
    else:
        # 生成模式：pagecontent 必须有
        if not pc:
            raise HTTPException(status_code=400, detail="pagecontent is required when get_down=false")

    p2ppt_req = Paper2PPTRequest(
        language=language,
        chat_api_url=chat_api_url,
        chat_api_key=api_key,
        api_key=api_key,
        model=model,
        gen_fig_model=img_gen_model_name,
        # input_type/input_content 在 ppt-only 场景下不强制，但保持可用
        input_type="PDF",
        input_content="",
        aspect_ratio=aspect_ratio,
        style=style,
        invite_code=invite_code or "",
        all_edited_down=all_edited_down_bool,
    )

    log.info(f"[ppt_json] Calling run_paper2ppt_wf_api with get_down={get_down_bool}")

    resp = await run_paper2ppt_wf_api(
        p2ppt_req,
        pagecontent=pc,
        result_path=result_path,
        get_down=get_down_bool,
        edit_page_num=page_id,
        edit_page_prompt=edit_prompt,
    )

    log.info(
        f"[ppt_json] Workflow completed: success={resp.success}, "
        f"result_path={resp.result_path}, "
        f"all_output_files_count={len(resp.all_output_files) if resp.all_output_files else 0}"
    )

    # 路由层转 URL
    if resp.ppt_pdf_path:
        resp.ppt_pdf_path = _to_outputs_url(resp.ppt_pdf_path, request)
    if resp.ppt_pptx_path:
        resp.ppt_pptx_path = _to_outputs_url(resp.ppt_pptx_path, request)

    resp.all_output_files = _collect_output_files_as_urls(resp.result_path, request)
    return resp


@router.post("/full_json", response_model=Paper2PPTResponse)
async def paper2ppt_full_json(
    request: Request,
    img_gen_model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    invite_code: Optional[str] = Form(None),
    # 输入：支持 text/pdf/pptx
    input_type: str = Form(...),  # 'text' | 'pdf' | 'pptx'
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    # 其他控制参数
    language: str = Form("zh"),
    aspect_ratio: str = Form("16:9"),
    style: str = Form(...),
    model: str = Form("gpt-5.1"),
):
    """
    Full pipeline：
    - paper2page_content -> paper2ppt
    - get_down 固定为 False（首次生成）
    """
    # validate_invite_code(invite_code)

    norm_input_type = input_type.lower().strip()

    run_dir = _create_run_dir_for_paper2ppt(invite_code)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    wf_input_type: str
    wf_input_content: str

    if norm_input_type == "pdf":
        if file is None:
            raise HTTPException(status_code=400, detail="file is required when input_type is 'pdf'")

        input_path = (input_dir / "input.pdf").resolve()
        input_path.write_bytes(await file.read())

        wf_input_type = "PDF"
        wf_input_content = str(input_path)
    elif norm_input_type in ("ppt", "pptx"):
        if file is None:
            raise HTTPException(status_code=400, detail="file is required when input_type is 'pptx'")

        input_path = (input_dir / "input.pptx").resolve()
        input_path.write_bytes(await file.read())

        wf_input_type = "PPT"
        wf_input_content = str(input_path)
    elif norm_input_type == "text":
        if not text:
            raise HTTPException(status_code=400, detail="text is required when input_type is 'text'")

        (input_dir / "input.txt").resolve().write_text(text, encoding="utf-8")
        wf_input_type = "TEXT"
        wf_input_content = text
    else:
        raise HTTPException(status_code=400, detail="invalid input_type, must be one of: text, pdf, pptx")

    p2ppt_req = Paper2PPTRequest(
        language=language,
        chat_api_url=chat_api_url,
        chat_api_key=api_key,
        api_key=api_key,
        model=model,
        gen_fig_model=img_gen_model_name,
        input_type=wf_input_type,
        input_content=wf_input_content,
        aspect_ratio=aspect_ratio,
        style=style,
        invite_code=invite_code or "",
    )

    resp = await run_paper2ppt_full_pipeline(p2ppt_req)

    if resp.ppt_pdf_path:
        resp.ppt_pdf_path = _to_outputs_url(resp.ppt_pdf_path, request)
    if resp.ppt_pptx_path:
        resp.ppt_pptx_path = _to_outputs_url(resp.ppt_pptx_path, request)

    resp.all_output_files = _collect_output_files_as_urls(resp.result_path, request)
    return resp
