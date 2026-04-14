from __future__ import annotations

"""
paper2ppt 业务 Service 层

用法概览（给 Router / 其它调用方看的）：
------------------------------------------------
1. 获取 pagecontent（对应 /paper2ppt/pagecontent_json）

    from fastapi_app.schemas import PageContentRequest
    from fastapi_app.services.paper2ppt_service import Paper2PPTService

    service = Paper2PPTService()
    req = PageContentRequest(
        chat_api_url=chat_api_url,
        api_key=api_key,
        invite_code=invite_code,
        input_type=input_type,   # "text" | "pdf" | "pptx" | "topic"
        text=text,               # 仅 text/topic 时使用
        model=model,
        language=language,
        style=style,
        gen_fig_model=gen_fig_model,
        page_count=page_count,
        use_long_paper=use_long_paper,  # 字符串 "true"/"false"
    )
    resp_dict = await service.get_page_content(
        req=req,
        file=file_upload,                # Optional[UploadFile]，pdf/pptx 输入
        reference_img=reference_upload,  # Optional[UploadFile]，参考风格图
        request=request,                 # FastAPI Request，用于拼 URL
    )

    返回值：dict（等价于 Paper2PPTResponse.model_dump() + all_output_files）


2. 生成/编辑 PPT（对应 /paper2ppt/ppt_json）

    from fastapi_app.schemas import PPTGenerationRequest

    service = Paper2PPTService()
    req = PPTGenerationRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        invite_code=invite_code,
        style=style,
        aspect_ratio=aspect_ratio,
        language=language,
        model=model,
        get_down=get_down,               # 字符串 "true"/"false"
        all_edited_down=all_edited_down, # 字符串 "true"/"false"
        result_path=result_path,         # 上一次生成的 outputs 子目录
        pagecontent=pagecontent_json,    # str | None，pagecontent 的 JSON 字符串
        page_id=page_id,                 # int | None，编辑页号（get_down=true 时必需）
        edit_prompt=edit_prompt,         # str | None，编辑提示词（get_down=true 时必需）
    )
    resp_dict = await service.generate_ppt(
        req=req,
        reference_img=reference_upload,  # Optional[UploadFile]
        request=request,
    )

    返回值：dict（等价于 Paper2PPTResponse.model_dump() + all_output_files，且所有路径已转 URL）


3. full pipeline：pagecontent + ppt 一次完成（对应 /paper2ppt/full_json）

    from fastapi_app.schemas import FullPipelineRequest

    service = Paper2PPTService()
    req = FullPipelineRequest(
        img_gen_model_name=img_gen_model_name,
        chat_api_url=chat_api_url,
        api_key=api_key,
        invite_code=invite_code,
        input_type=input_type,    # "text" | "pdf" | "pptx"
        text=text,                # text 模式下使用
        language=language,
        aspect_ratio=aspect_ratio,
        style=style,
        model=model,
        use_long_paper=use_long_paper,   # 字符串 "true"/"false"
    )
    resp_dict = await service.run_full_pipeline(
        req=req,
        file=file_upload,         # Optional[UploadFile]，pdf/pptx 上传文件
        request=request,
    )

    返回值：dict（等价于 Paper2PPTResponse.model_dump() + all_output_files）
------------------------------------------------

函数级 docstring 里会详细说明每个参数的含义和使用约定。
"""

from uuid import uuid4

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, UploadFile
import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi_app.config import settings
from fastapi_app.schemas import (
    FullPipelineRequest,
    OutlineRefineRequest,
    PageContentRequest,
    PPTGenerationRequest,
)
from fastapi_app.services.managed_api_service import (
    resolve_image_generation_credentials,
    resolve_llm_credentials,
    resolve_model_name,
)
from fastapi_app.utils import (
    _to_outputs_url,
    get_outputs_root,
    resolve_outputs_path,
)
from fastapi_app.workflow_adapters.wa_paper2ppt import (
    run_paper2page_content_wf_api,
    run_paper2ppt_full_pipeline,
    run_paper2ppt_wf_api,
)
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.utils_common import robust_parse_json

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = get_outputs_root()
_PREVIEW_MAX_SIDE = 1280
_PREVIEW_SMALL_FILE_BYTES = 900 * 1024
_PREVIEW_JPEG_QUALITY = 82
_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_LANCZOS = _PIL_RESAMPLING.LANCZOS
_OUTLINE_PATCH_CHUNK_SIZE = 4
_OUTLINE_PLAN_SOURCE_CHAR_LIMIT = 12000
_OUTLINE_PLAN_PAGE_DIGEST_LIMIT = 22000
_OUTLINE_REWRITE_SOURCE_CHAR_LIMIT = 9000


class Paper2PPTService:
    """paper2ppt 相关的业务编排 Service。

    职责：
    - 负责从 HTTP 层 Request 模型和 UploadFile 中落地输入文件（pdf/pptx/text/topic）；
    - 组织调用 workflow adapter（run_paper2page_content_wf_api / run_paper2ppt_wf_api / run_paper2ppt_full_pipeline）；
    - 做路径到 URL 的转换（使用 fastapi_app.utils._to_outputs_url）；
    - 解析/校验 pagecontent JSON，做 URL ↔ 本地路径转换。

    不做的事情：
    - 不直接调用 dataflow_agent.workflow.run_workflow；
    - 不处理 FastAPI 路由/依赖注入（由 routers/paper2ppt.py 完成）。
    """

    @staticmethod
    def _resolve_credential_scope(raw_scope: Optional[str], default_scope: str = "paper2ppt") -> str:
        scope = (raw_scope or "").strip().lower()
        return scope or default_scope

    @staticmethod
    def _parse_page_index_list(raw: Optional[str], *, max_count: int | None = None) -> list[int]:
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

    # ---------------- 公共接口 ---------------- #

    async def get_page_content(
        self,
        req: PageContentRequest,
        file: UploadFile | None,
        reference_img: UploadFile | None,
        request: Request | None,
    ) -> Dict[str, Any]:
        """只跑 pagecontent（paper2page_content 工作流）。

        用途：
        - 从 PDF / PPTX / TEXT / TOPIC 输入中解析出结构化 pagecontent；
        - 只执行解析阶段，不生成最终 PPT。
        """
        run_dir = self._create_timestamp_run_dir(req.email)
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # 参考图
        reference_img_path = await self._save_reference_image(input_dir, reference_img)

        # 根据 input_type 落地输入
        pdf_as_slides = str(req.pdf_as_slides).lower() in ("true", "1", "yes")
        wf_input_type, wf_input_content = await self._prepare_input_for_pagecontent(
            input_dir=input_dir,
            input_type=req.input_type,
            file=file,
            text=req.text,
            pdf_as_slides=pdf_as_slides,
        )

        # 组装老的 Paper2PPTRequest 以复用现有 workflow adapter
        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        # 转换字符串布尔值
        use_long_paper_bool = str(req.use_long_paper).lower() in ("true", "1", "yes")
        credential_scope = self._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=resolved_chat_api_url,
            credential_scope=credential_scope,
            chat_api_key=resolved_api_key,
            api_key=resolved_api_key,
            model=resolve_model_name(
                req.model,
                managed_default=settings.PAPER2PPT_OUTLINE_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
            ),
            gen_fig_model="",
            input_type=wf_input_type,
            input_content=wf_input_content,
            style=req.style,
            ref_img=str(reference_img_path) if reference_img_path is not None else "",
            email=req.email or "",
            page_count=req.page_count,
            use_long_paper=use_long_paper_bool,
            render_dpi=req.render_dpi,
        )

        resp_model = await run_paper2page_content_wf_api(p2ppt_req, result_path=run_dir)

        resp_dict = resp_model.model_dump()
        resp_dict["pagecontent"] = self._normalize_pagecontent_items(resp_dict.get("pagecontent", []))
        if not resp_dict["pagecontent"]:
            backend_error = str(resp_dict.get("error") or "").strip()
            if backend_error:
                raise HTTPException(status_code=502, detail=backend_error)
            raw_text = (req.text or "").strip()
            if str(req.input_type).lower() == "text" and use_long_paper_bool and req.page_count > 20:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"当前为文本模式，输入内容仅 {len(raw_text)} 个字符，不足以稳定生成 {req.page_count} 页长文大纲。"
                        "请提供更完整的正文，或改用 Topic 模式。"
                    ),
                )
            raise HTTPException(status_code=502, detail="后端未生成有效大纲，请稍后重试")
        if request is not None:
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request, resp_model.result_path
            )
        if request is not None:
            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_model.result_path, request)
        else:
            resp_dict["all_output_files"] = []

        return resp_dict

    async def refine_outline(
        self,
        req: OutlineRefineRequest,
        request: Request | None,
    ) -> Dict[str, Any]:
        """Refine outline based on feedback without re-parsing input."""
        if not req.outline_feedback.strip():
            raise HTTPException(status_code=400, detail="outline_feedback is required")

        pc = self._parse_pagecontent_json(req.pagecontent)
        if not pc:
            raise HTTPException(status_code=400, detail="pagecontent is required")

        from fastapi_app.schemas import Paper2PPTRequest
        credential_scope = self._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=resolved_chat_api_url,
            credential_scope=credential_scope,
            chat_api_key=resolved_api_key,
            api_key=resolved_api_key,
            model=resolve_model_name(
                req.model,
                managed_default=settings.PAPER2PPT_OUTLINE_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
            ),
            gen_fig_model="",
            input_type="TEXT",
            input_content="",
            style="",
            email=req.email or "",
            page_count=len(pc),
        )

        result_root: Path | None = None
        if req.result_path:
            result_root = self.resolve_result_path(req.result_path)

        source_text, mineru_output = self._load_outline_refine_source_context(result_root)
        refined_pagecontent = await self._refine_outline_with_patch_plan(
            req=p2ppt_req,
            pagecontent=pc,
            outline_feedback=req.outline_feedback,
            source_text=source_text,
            mineru_output=mineru_output,
        )
        resp_dict: Dict[str, Any] = {
            "success": True,
            "pagecontent": self._normalize_pagecontent_items(refined_pagecontent),
            "result_path": str(result_root) if result_root is not None else (req.result_path or ""),
        }
        if request is not None:
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request, resp_dict.get("result_path")
            )
        if request is not None and resp_dict.get("result_path"):
            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_dict["result_path"], request)
        else:
            resp_dict["all_output_files"] = []

        return resp_dict

    async def generate_ppt(
        self,
        req: PPTGenerationRequest,
        reference_img: UploadFile | None,
        request: Request | None,
    ) -> Dict[str, Any]:
        """只跑 PPT 生成/编辑（paper2ppt 工作流）。"""
        base_dir = self.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        # 处理参考图：上传新的或复用旧的
        reference_img_path = await self._ensure_reference_image(base_dir, reference_img)

        # 解析 pagecontent JSON（如果有）并把 URL 转成本地路径
        pc: List[Dict[str, Any]] = []
        if req.pagecontent is not None:
            pc = self._parse_pagecontent_json(req.pagecontent)
            for item in pc:
                # 常见包含路径的字段
                for key in ["ppt_img_path", "asset_ref", "generated_img_path"]:
                    if key in item and item[key]:
                        value = str(item[key]).strip()
                        if value.startswith(("http://", "https://", "/outputs/")) or Path(value).is_absolute() or value.startswith("outputs/"):
                            item[key] = str(resolve_outputs_path(value, must_exist=False))

        # 转换字符串布尔值
        get_down_bool = str(req.get_down).lower() in ("true", "1", "yes")
        all_edited_down_bool = str(req.all_edited_down).lower() in ("true", "1", "yes")
        regenerate_from_outline_bool = str(req.regenerate_from_outline).lower() in ("true", "1", "yes")
        credential_scope = self._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        skip_pages = self._parse_page_index_list(req.skip_pages, max_count=len(pc) if pc else None)

        # 校验编辑/生成模式
        if get_down_bool:
            if req.page_id is None:
                raise HTTPException(status_code=400, detail="page_id is required when get_down=true")
            if not regenerate_from_outline_bool and not (req.edit_prompt or "").strip():
                raise HTTPException(status_code=400, detail="edit_prompt is required when get_down=true")
        else:
            if not pc:
                raise HTTPException(status_code=400, detail="pagecontent is required when get_down=false")

        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=resolved_chat_api_url,
            credential_scope=credential_scope,
            chat_api_key=resolved_api_key,
            api_key=resolved_api_key,
            image_api_url=resolved_image_api_url,
            image_api_key=resolved_image_api_key,
            model=resolve_model_name(
                req.model,
                managed_default=settings.PAPER2PPT_CONTENT_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
            ),
            gen_fig_model=resolve_model_name(
                req.img_gen_model_name,
                managed_default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
            ),
            input_type="PDF",
            input_content="",
            aspect_ratio=req.aspect_ratio,
            style=req.style,
            ref_img=str(reference_img_path) if reference_img_path else "",
            email=req.email or "",
            all_edited_down=all_edited_down_bool,
            image_resolution=req.image_resolution or "2K",
        )

        resp_model = await run_paper2ppt_wf_api(
            p2ppt_req,
            pagecontent=pc,
            result_path=str(base_dir),
            get_down=get_down_bool,
            edit_page_num=req.page_id,
            edit_page_prompt=req.edit_prompt,
            regenerate_from_outline=regenerate_from_outline_bool,
            skip_pages=skip_pages,
        )

        resp_dict = resp_model.model_dump()
        return self.normalize_ppt_response(resp_dict, request)

    async def run_full_pipeline(
        self,
        req: FullPipelineRequest,
        file: UploadFile | None,
        request: Request | None,
    ) -> Dict[str, Any]:
        """full pipeline：一次性跑完 pagecontent + ppt。"""
        run_dir = self._create_timestamp_run_dir(req.email)
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        wf_input_type, wf_input_content = await self._prepare_input_for_full(
            input_dir=input_dir,
            input_type=req.input_type,
            file=file,
            text=req.text,
        )
        credential_scope = self._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )

        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=resolved_chat_api_url,
            credential_scope=credential_scope,
            chat_api_key=resolved_api_key,
            api_key=resolved_api_key,
            image_api_url=resolved_image_api_url,
            image_api_key=resolved_image_api_key,
            model=resolve_model_name(
                req.model,
                managed_default=settings.PAPER2PPT_CONTENT_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
            ),
            gen_fig_model=resolve_model_name(
                req.img_gen_model_name,
                managed_default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
                fallback_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
            ),
            input_type=wf_input_type,
            input_content=wf_input_content,
            aspect_ratio=req.aspect_ratio,
            style=req.style,
            email=req.email or "",
            use_long_paper=req.use_long_paper,
        )

        resp_model = await run_paper2ppt_full_pipeline(p2ppt_req)

        resp_dict = resp_model.model_dump()

        if request is not None:
            if resp_dict.get("ppt_pdf_path"):
                resp_dict["ppt_pdf_path"] = _to_outputs_url(resp_dict["ppt_pdf_path"], request)
            if resp_dict.get("ppt_pptx_path"):
                resp_dict["ppt_pptx_path"] = _to_outputs_url(resp_dict["ppt_pptx_path"], request)
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request, resp_model.result_path
            )

            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_model.result_path, request)
        else:
            resp_dict["all_output_files"] = []

        return resp_dict

    # ---------------- 内部工具方法 ---------------- #

    def _create_timestamp_run_dir(self, email: Optional[str]) -> Path:
        """根据邮箱创建本次请求的唯一输出目录。

        目录结构：
            outputs/{email or 'default'}/paper2ppt/<run_id>/

        说明：
        - email 为 None 或空字符串时，使用 "default" 作为子目录名；
        - run_id 使用纳秒时间戳 + 随机后缀，避免并发请求撞目录；
        - 始终在 PROJECT_ROOT / outputs 下创建目录，保证和原始实现兼容。
        """
        import time

        run_id = f"{time.time_ns()}-{uuid4().hex[:8]}"
        # 如果有 email，则使用 email，否则使用 default
        code = email or "default"
        run_dir = BASE_OUTPUT_DIR / code / "paper2ppt" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def resolve_result_path(self, result_path: str) -> Path:
        return resolve_outputs_path(result_path, must_exist=False, allow_dirs=True)

    async def cache_reference_image_for_result(
        self,
        result_path: str,
        reference_img: UploadFile | None,
    ) -> Optional[Path]:
        if reference_img is None:
            return None

        base_dir = self.resolve_result_path(result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")
        return await self._ensure_reference_image(base_dir, reference_img)

    def normalize_ppt_response(
        self,
        resp_dict: Dict[str, Any],
        request: Request | None,
    ) -> Dict[str, Any]:
        normalized = copy.deepcopy(resp_dict)
        result_path = normalized.get("result_path", "")

        if request is not None:
            if normalized.get("ppt_pdf_path"):
                normalized["ppt_pdf_path"] = _to_outputs_url(normalized["ppt_pdf_path"], request)
            if normalized.get("ppt_pptx_path"):
                normalized["ppt_pptx_path"] = _to_outputs_url(normalized["ppt_pptx_path"], request)
            normalized["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                normalized.get("pagecontent", []), request, result_path
            )
            normalized["all_output_files"] = self._collect_output_files_as_urls(result_path, request)
        else:
            normalized.setdefault("all_output_files", [])

        failed_pages = self._collect_failed_pages(normalized.get("pagecontent", []))
        if failed_pages:
            normalized["failed_pages"] = failed_pages
            normalized["failed_page_indices"] = [item["page_idx"] for item in failed_pages]
            normalized["partial_success"] = any(
                isinstance(item, dict) and str(item.get("generated_img_path") or "").strip()
                for item in normalized.get("pagecontent", [])
            )
        else:
            normalized.setdefault("failed_pages", [])
            normalized.setdefault("failed_page_indices", [])
            normalized.setdefault("partial_success", False)

        return normalized

    def _collect_failed_pages(self, pagecontent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failed_pages: List[Dict[str, Any]] = []
        for fallback_idx, item in enumerate(pagecontent or []):
            if not isinstance(item, dict):
                continue

            tracked = any(key in item for key in ("generated_img_path", "page_idx", "mode", "error"))
            if not tracked:
                continue

            generated_img_path = str(item.get("generated_img_path") or "").strip()
            if generated_img_path:
                continue

            page_idx_raw = item.get("page_idx", fallback_idx)
            try:
                page_idx = int(page_idx_raw)
            except (TypeError, ValueError):
                page_idx = fallback_idx

            error_text = str(item.get("error") or "").strip()
            mode_text = str(item.get("mode") or "").strip()
            failed_pages.append(
                {
                    "page_idx": page_idx,
                    "reason": error_text or mode_text or "generated image missing",
                    "mode": mode_text,
                    "error": error_text,
                }
            )

        return failed_pages

    def _convert_pagecontent_paths_to_urls(
        self,
        pagecontent: List[Dict[str, Any]],
        request: Request,
        result_path: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Convert local output paths inside pagecontent to browser-accessible URLs."""
        if not pagecontent:
            return pagecontent

        base_dir: Path | None = None
        if result_path:
            try:
                base_dir = self.resolve_result_path(str(result_path))
            except HTTPException:
                base_dir = None

        keys = {
            "ppt_img_path",
            "generated_img_path",
            "img_path",
            "image_path",
            "path",
            "source_img_path",
            "reference_image_path",
            "asset_ref",
        }

        for item in pagecontent:
            if not isinstance(item, dict):
                continue
            for key in keys:
                value = item.get(key)
                if not value or not isinstance(value, str):
                    continue
                local_path = self._try_resolve_output_file(value)
                if not local_path:
                    continue
                item[key] = _to_outputs_url(local_path, request)
                if base_dir is not None:
                    preview_path = self._ensure_preview_asset(base_dir=base_dir, original_path=local_path)
                    if preview_path:
                        item[f"{key}_preview_path"] = _to_outputs_url(preview_path, request)

        return pagecontent

    def _try_resolve_output_file(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            return str(
                resolve_outputs_path(
                    raw,
                    must_exist=False,
                    allow_files=True,
                    allow_dirs=False,
                )
            )
        except HTTPException:
            return ""

    def _ensure_preview_asset(
        self,
        *,
        base_dir: Path,
        original_path: str,
    ) -> str:
        source_path = Path(str(original_path or "").strip())
        if not source_path.exists() or not source_path.is_file():
            return ""
        if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return str(source_path)

        try:
            source_stat = source_path.stat()
        except OSError:
            return str(source_path)

        try:
            with Image.open(source_path) as original_img:
                original_img = ImageOps.exif_transpose(original_img)
                if max(original_img.size) <= _PREVIEW_MAX_SIDE and source_stat.st_size <= _PREVIEW_SMALL_FILE_BYTES:
                    return str(source_path)

                preview_root = base_dir / "image_previews"
                preview_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1(
                    f"{source_path}|{source_stat.st_size}|{source_stat.st_mtime_ns}".encode("utf-8")
                ).hexdigest()[:16]
                has_alpha = self._image_has_alpha(original_img)
                preview_ext = ".png" if has_alpha else ".jpg"
                preview_path = (preview_root / f"{source_path.stem}_{digest}{preview_ext}").resolve()
                if preview_path.exists():
                    return str(preview_path)

                preview_img = original_img.copy()
                preview_img.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE), _PIL_LANCZOS)
                if has_alpha:
                    preview_img = preview_img.convert("RGBA")
                    preview_img.save(preview_path, format="PNG", optimize=True)
                else:
                    preview_img = preview_img.convert("RGB")
                    preview_img.save(
                        preview_path,
                        format="JPEG",
                        quality=_PREVIEW_JPEG_QUALITY,
                        optimize=True,
                        progressive=True,
                    )
                return str(preview_path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("[Paper2PPTService] Failed to build preview for %s: %s", source_path, exc)
            return str(source_path)

    def _image_has_alpha(self, image: Image.Image) -> bool:
        bands = image.getbands()
        if "A" in bands:
            return True
        if image.mode == "P":
            return "transparency" in image.info
        return False

    async def _save_reference_image(self, input_dir: Path, reference_img: UploadFile | None) -> Optional[Path]:
        if reference_img is None:
            return None
        ref_ext = Path(reference_img.filename or "").suffix or ".png"
        reference_img_path = (input_dir / f"reference{ref_ext}").resolve()
        reference_img_path.write_bytes(await reference_img.read())
        return reference_img_path

    async def _ensure_reference_image(self, base_dir: Path, reference_img: UploadFile | None) -> Optional[Path]:
        """
        如果上传了 reference_img 就保存新文件；否则尝试复用 result_path/input 下的历史 reference.*。
        """
        input_dir = base_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        if reference_img is not None:
            ref_ext = Path(reference_img.filename or "").suffix or ".png"
            reference_img_path = (input_dir / f"ppt_ref_style{ref_ext}").resolve()
            reference_img_path.write_bytes(await reference_img.read())
            log.info(f"[Paper2PPTService] Saved reference_img to {reference_img_path}")
            return reference_img_path

        # 尝试复用历史参考图（优先 ppt_ref_style.*，兼容旧的 reference.*）
        if input_dir.exists():
            for prefix in ["ppt_ref_style", "reference"]:
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    candidate = input_dir / f"{prefix}{ext}"
                    if candidate.exists():
                        log.info(f"[Paper2PPTService] Found cached reference_img at {candidate}")
                        return candidate
        return None

    async def _prepare_input_for_pagecontent(
        self,
        input_dir: Path,
        input_type: str,
        file: UploadFile | None,
        text: Optional[str],
        pdf_as_slides: bool,
    ) -> tuple[str, str]:
        """
        pagecontent-only 场景下的输入准备逻辑。
        直接复用 router 里原有的分支：
        - pdf/pptx/topic/text
        """
        norm_input_type = input_type.lower().strip()

        if norm_input_type == "pdf":
            if file is None:
                raise HTTPException(status_code=400, detail="file is required when input_type is 'pdf'")
            input_path = (input_dir / "input.pdf").resolve()
            input_path.write_bytes(await file.read())
            if pdf_as_slides:
                return "PPT", str(input_path)
            return "PDF", str(input_path)

        if norm_input_type in ("ppt", "pptx"):
            if file is None:
                raise HTTPException(status_code=400, detail="file is required when input_type is 'pptx'")
            input_path = (input_dir / "input.pptx").resolve()
            input_path.write_bytes(await file.read())
            return "PPT", str(input_path)

        if norm_input_type == "text":
            if not text:
                raise HTTPException(status_code=400, detail="text is required when input_type is 'text'")
            (input_dir / "input.txt").resolve().write_text(text, encoding="utf-8")
            return "TEXT", text

        if norm_input_type == "topic":
            if not text:
                raise HTTPException(status_code=400, detail="text (topic) is required when input_type is 'topic'")
            (input_dir / "input_topic.txt").resolve().write_text(text, encoding="utf-8")
            return "TOPIC", text

        raise HTTPException(status_code=400, detail="invalid input_type, must be one of: text, pdf, pptx, topic")

    async def _prepare_input_for_full(
        self,
        input_dir: Path,
        input_type: str,
        file: UploadFile | None,
        text: Optional[str],
    ) -> tuple[str, str]:
        """
        full pipeline 场景的输入准备逻辑。
        复用原 full_json 里的 pdf/pptx/text 处理。
        """
        norm_input_type = input_type.lower().strip()

        if norm_input_type == "pdf":
            if file is None:
                raise HTTPException(status_code=400, detail="file is required when input_type is 'pdf'")
            input_path = (input_dir / "input.pdf").resolve()
            input_path.write_bytes(await file.read())
            return "PDF", str(input_path)

        if norm_input_type in ("ppt", "pptx"):
            if file is None:
                raise HTTPException(status_code=400, detail="file is required when input_type is 'pptx'")
            input_path = (input_dir / "input.pptx").resolve()
            input_path.write_bytes(await file.read())
            return "PPT", str(input_path)

        if norm_input_type == "text":
            if not text:
                raise HTTPException(status_code=400, detail="text is required when input_type is 'text'")
            (input_dir / "input.txt").resolve().write_text(text, encoding="utf-8")
            return "TEXT", text

        raise HTTPException(status_code=400, detail="invalid input_type, must be one of: text, pdf, pptx")

    def _collect_output_files_as_urls(self, result_path: str, request: Request) -> list[str]:
        if not result_path:
            return []

        try:
            root = resolve_outputs_path(result_path, must_exist=True, allow_dirs=True)
        except HTTPException:
            return []

        urls: list[str] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".pdf", ".pptx", ".png"}:
                urls.append(_to_outputs_url(str(p), request))
        return urls

    def _parse_pagecontent_json(self, pagecontent_json: str) -> List[Dict[str, Any]]:
        try:
            obj = json.loads(pagecontent_json)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid pagecontent json: {e}") from e

        if not isinstance(obj, list):
            raise HTTPException(status_code=400, detail="pagecontent must be a JSON list")

        for i, it in enumerate(obj):
            if not isinstance(it, dict):
                raise HTTPException(status_code=400, detail=f"pagecontent[{i}] must be an object(dict)")
        return self._normalize_pagecontent_items(obj)

    def _extract_outline_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return " ".join(value.strip().split())
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            preferred_keys = (
                "text",
                "value",
                "content",
                "summary",
                "title",
                "label",
                "body",
                "description",
                "reason",
                "point",
            )
            for key in preferred_keys:
                text = self._extract_outline_text(value.get(key))
                if text:
                    return text
            for item in value.values():
                text = self._extract_outline_text(item)
                if text:
                    return text
            return ""
        if isinstance(value, list):
            parts = [self._extract_outline_text(item) for item in value]
            return " ".join(part for part in parts if part)
        return " ".join(str(value).strip().split())

    def _normalize_outline_points(self, value: Any) -> List[str]:
        if isinstance(value, list):
            items = [self._extract_outline_text(item) for item in value]
        else:
            items = [self._extract_outline_text(value)]
        return [item for item in items if item]

    def _normalize_pagecontent_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in items:
            item = copy.deepcopy(raw)
            if "title" in item:
                item["title"] = self._extract_outline_text(item.get("title"))
            if "layout_description" in item:
                item["layout_description"] = self._extract_outline_text(item.get("layout_description"))
            if "key_points" in item:
                item["key_points"] = self._normalize_outline_points(item.get("key_points"))
            if "asset_ref" in item:
                asset_ref = item.get("asset_ref")
                if isinstance(asset_ref, list):
                    normalized_refs = self._normalize_outline_points(asset_ref)
                    item["asset_ref"] = normalized_refs[0] if normalized_refs else None
                else:
                    normalized_ref = self._extract_outline_text(asset_ref)
                    item["asset_ref"] = normalized_ref or None
            normalized.append(item)
        return normalized

    def _load_outline_refine_source_context(self, result_root: Path | None) -> tuple[str, str]:
        if result_root is None:
            return "", ""

        input_dir = result_root / "input"
        text_candidates = [
            input_dir / "input.txt",
            input_dir / "input_topic.txt",
        ]
        source_text = ""
        for candidate in text_candidates:
            if candidate.exists():
                try:
                    source_text = candidate.read_text(encoding="utf-8")
                    if source_text.strip():
                        break
                except Exception:
                    continue

        mineru_output = ""
        try:
            markdown_candidates = list(result_root.glob("*/auto/*.md"))
        except Exception:
            markdown_candidates = []
        for candidate in markdown_candidates:
            try:
                mineru_output = candidate.read_text(encoding="utf-8")
                if mineru_output.strip():
                    break
            except Exception:
                continue

        return source_text, mineru_output

    async def _refine_outline_with_patch_plan(
        self,
        *,
        req: Any,
        pagecontent: List[Dict[str, Any]],
        outline_feedback: str,
        source_text: str,
        mineru_output: str,
    ) -> List[Dict[str, Any]]:
        original_pages = self._normalize_pagecontent_items(pagecontent)
        if not original_pages:
            return []

        plan = await self._plan_outline_refine_operations(
            req=req,
            pagecontent=original_pages,
            outline_feedback=outline_feedback,
            source_text=source_text,
            mineru_output=mineru_output,
        )
        working_pages, rewrite_targets = self._apply_outline_patch_plan(original_pages, plan)
        rewritten_pages = await self._rewrite_outline_pages_by_chunks(
            req=req,
            pagecontent=working_pages,
            rewrite_targets=rewrite_targets,
            global_instruction=str(plan.get("global_instruction") or "").strip(),
            outline_feedback=outline_feedback,
            source_text=source_text,
            mineru_output=mineru_output,
        )
        return self._normalize_pagecontent_items(rewritten_pages)

    async def _plan_outline_refine_operations(
        self,
        *,
        req: Any,
        pagecontent: List[Dict[str, Any]],
        outline_feedback: str,
        source_text: str,
        mineru_output: str,
    ) -> Dict[str, Any]:
        prompt_generator = PromptsTemplateGenerator(output_language=req.language)
        outline_digest = self._build_outline_digest(pagecontent, max_chars=_OUTLINE_PLAN_PAGE_DIGEST_LIMIT)
        source_excerpt = self._build_source_excerpt(
            source_text=source_text,
            mineru_output=mineru_output,
            max_chars=_OUTLINE_PLAN_SOURCE_CHAR_LIMIT,
        )
        system_prompt = prompt_generator.render(
            "system_prompt_for_paper2ppt_outline_edit_planner_agent",
            language=req.language,
        )
        task_prompt = prompt_generator.render(
            "task_prompt_for_paper2ppt_outline_edit_planner_agent",
            page_count=len(pagecontent),
            outline_feedback=outline_feedback,
            outline_digest=outline_digest,
            source_excerpt=source_excerpt,
            language=req.language,
        )

        try:
            raw_plan = await self._invoke_json_llm(
                chat_api_url=req.chat_api_url,
                api_key=req.api_key,
                model_name=req.model,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                max_tokens=4096,
            )
        except Exception as exc:
            log.warning("[paper2ppt][outline-refine] planner failed: %s", exc)
            raw_plan = {}

        return self._normalize_outline_patch_plan(
            raw_plan,
            page_count=len(pagecontent),
            outline_feedback=outline_feedback,
        )

    async def _rewrite_outline_pages_by_chunks(
        self,
        *,
        req: Any,
        pagecontent: List[Dict[str, Any]],
        rewrite_targets: set[int],
        global_instruction: str,
        outline_feedback: str,
        source_text: str,
        mineru_output: str,
    ) -> List[Dict[str, Any]]:
        normalized_pages = self._normalize_pagecontent_items(pagecontent)
        if not normalized_pages:
            return []

        if not rewrite_targets:
            return normalized_pages

        prompt_generator = PromptsTemplateGenerator(output_language=req.language)
        source_excerpt = self._build_source_excerpt(
            source_text=source_text,
            mineru_output=mineru_output,
            max_chars=_OUTLINE_REWRITE_SOURCE_CHAR_LIMIT,
        )
        chunks = self._build_rewrite_chunks(sorted(rewrite_targets), len(normalized_pages), _OUTLINE_PATCH_CHUNK_SIZE)
        rewritten_pages = copy.deepcopy(normalized_pages)

        for chunk_start, chunk_end in chunks:
            chunk_pages = copy.deepcopy(normalized_pages[chunk_start : chunk_end + 1])
            page_instruction_lines: List[str] = []
            for offset, page in enumerate(chunk_pages, start=chunk_start + 1):
                specific_instruction = str(page.pop("_patch_instruction", "") or "").strip()
                if specific_instruction:
                    page_instruction_lines.append(f"- Page {offset}: {specific_instruction}")
            previous_title = rewritten_pages[chunk_start - 1]["title"] if chunk_start > 0 else ""
            next_title = rewritten_pages[chunk_end + 1]["title"] if chunk_end + 1 < len(rewritten_pages) else ""

            system_prompt = prompt_generator.render(
                "system_prompt_for_paper2ppt_outline_patch_rewriter_agent",
                language=req.language,
            )
            task_prompt = prompt_generator.render(
                "task_prompt_for_paper2ppt_outline_patch_rewriter_agent",
                page_count=len(chunk_pages),
                chunk_start=chunk_start + 1,
                chunk_end=chunk_end + 1,
                outline_feedback=outline_feedback,
                global_instruction=global_instruction or outline_feedback,
                page_specific_instructions="\n".join(page_instruction_lines) or "- None",
                previous_title=previous_title or "None",
                next_title=next_title or "None",
                source_excerpt=source_excerpt,
                pagecontent=json.dumps(chunk_pages, ensure_ascii=False, indent=2),
                language=req.language,
            )

            try:
                raw_rewrite = await self._invoke_json_llm(
                    chat_api_url=req.chat_api_url,
                    api_key=req.api_key,
                    model_name=req.model,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    max_tokens=4096,
                )
            except Exception as exc:
                log.warning(
                    "[paper2ppt][outline-refine] chunk rewrite failed for %s-%s: %s",
                    chunk_start + 1,
                    chunk_end + 1,
                    exc,
                )
                continue

            if not isinstance(raw_rewrite, list) or len(raw_rewrite) != len(chunk_pages):
                log.warning(
                    "[paper2ppt][outline-refine] chunk rewrite length mismatch for %s-%s, keep original",
                    chunk_start + 1,
                    chunk_end + 1,
                )
                continue

            normalized_chunk = self._normalize_pagecontent_items(
                [item for item in raw_rewrite if isinstance(item, dict)]
            )
            if len(normalized_chunk) != len(chunk_pages):
                log.warning(
                    "[paper2ppt][outline-refine] chunk rewrite invalid payload for %s-%s, keep original",
                    chunk_start + 1,
                    chunk_end + 1,
                )
                continue

            merged_chunk: List[Dict[str, Any]] = []
            for original_page, rewritten_page in zip(chunk_pages, normalized_chunk):
                merged_page = copy.deepcopy(original_page)
                merged_page["title"] = rewritten_page.get("title") or original_page.get("title") or ""
                merged_page["layout_description"] = (
                    rewritten_page.get("layout_description")
                    or original_page.get("layout_description")
                    or ""
                )
                rewritten_points = self._normalize_outline_points(rewritten_page.get("key_points", []))
                merged_page["key_points"] = rewritten_points or original_page.get("key_points") or []
                if rewritten_page.get("asset_ref") is not None:
                    merged_page["asset_ref"] = rewritten_page.get("asset_ref")
                merged_chunk.append(merged_page)

            rewritten_pages[chunk_start : chunk_end + 1] = merged_chunk

        for page in rewritten_pages:
            if isinstance(page, dict):
                page.pop("_origin_page_number", None)
                page.pop("_patch_instruction", None)
        return rewritten_pages

    def _normalize_outline_patch_plan(
        self,
        raw_plan: Any,
        *,
        page_count: int,
        outline_feedback: str,
    ) -> Dict[str, Any]:
        base_plan: Dict[str, Any] = {
            "global_instruction": outline_feedback.strip(),
            "apply_global_rewrite": True,
            "operations": [],
        }
        if not isinstance(raw_plan, dict):
            return base_plan

        global_instruction = self._extract_outline_text(raw_plan.get("global_instruction")) or outline_feedback.strip()
        apply_global_rewrite = bool(raw_plan.get("apply_global_rewrite"))
        operations: List[Dict[str, Any]] = []
        raw_operations = raw_plan.get("operations")
        if isinstance(raw_operations, list):
            for raw_operation in raw_operations:
                if not isinstance(raw_operation, dict):
                    continue
                op_type = str(raw_operation.get("type") or "").strip().lower()
                if op_type not in {"update", "delete", "insert_after", "move"}:
                    continue
                normalized_op: Dict[str, Any] = {"type": op_type}
                if op_type in {"update", "delete", "move"}:
                    page_numbers = self._normalize_plan_page_numbers(raw_operation.get("page_numbers"), page_count)
                    if not page_numbers:
                        continue
                    normalized_op["page_numbers"] = page_numbers
                if op_type == "update":
                    instruction = self._extract_outline_text(raw_operation.get("instruction"))
                    if not instruction:
                        continue
                    normalized_op["instruction"] = instruction
                if op_type == "insert_after":
                    after_page = self._normalize_single_page_number(raw_operation.get("page_number"), page_count)
                    count = self._normalize_insert_count(raw_operation.get("count"))
                    instruction = self._extract_outline_text(raw_operation.get("instruction"))
                    if after_page is None or count <= 0 or not instruction:
                        continue
                    normalized_op["page_number"] = after_page
                    normalized_op["count"] = count
                    normalized_op["instruction"] = instruction
                if op_type == "move":
                    after_page = self._normalize_single_page_number(raw_operation.get("after_page_number"), page_count)
                    if after_page is None:
                        continue
                    normalized_op["after_page_number"] = after_page
                operations.append(normalized_op)

        if operations:
            explicit_restructure = any(op["type"] in {"insert_after", "delete", "move"} for op in operations)
            apply_global_rewrite = bool(raw_plan.get("apply_global_rewrite")) if raw_plan.get("apply_global_rewrite") is not None else not explicit_restructure

        return {
            "global_instruction": global_instruction,
            "apply_global_rewrite": apply_global_rewrite or not operations,
            "operations": operations,
        }

    def _normalize_plan_page_numbers(self, value: Any, page_count: int) -> List[int]:
        if not isinstance(value, list):
            return []
        numbers: set[int] = set()
        for item in value:
            normalized = self._normalize_single_page_number(item, page_count)
            if normalized is not None:
                numbers.add(normalized)
        return sorted(numbers)

    def _normalize_single_page_number(self, value: Any, page_count: int) -> Optional[int]:
        try:
            number = int(str(value).strip())
        except Exception:
            return None
        if 1 <= number <= page_count:
            return number
        return None

    def _normalize_insert_count(self, value: Any) -> int:
        try:
            count = int(str(value).strip())
        except Exception:
            return 0
        return max(0, min(count, 8))

    def _apply_outline_patch_plan(
        self,
        pagecontent: List[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], set[int]]:
        working_pages: List[Dict[str, Any]] = []
        for idx, page in enumerate(self._normalize_pagecontent_items(pagecontent), start=1):
            page_copy = copy.deepcopy(page)
            page_copy["_origin_page_number"] = idx
            working_pages.append(page_copy)

        rewrite_targets: set[int] = set()
        for operation in plan.get("operations", []):
            op_type = operation.get("type")
            if op_type == "delete":
                delete_numbers = set(operation.get("page_numbers", []))
                working_pages = [
                    page for page in working_pages
                    if page.get("_origin_page_number") not in delete_numbers
                ]
                continue

            if op_type == "move":
                move_numbers = operation.get("page_numbers", [])
                after_page_number = operation.get("after_page_number")
                moving_pages = [
                    page for page in working_pages
                    if page.get("_origin_page_number") in move_numbers
                ]
                if not moving_pages:
                    continue
                working_pages = [
                    page for page in working_pages
                    if page.get("_origin_page_number") not in move_numbers
                ]
                insert_index = self._find_insert_index_after_origin(working_pages, after_page_number)
                working_pages[insert_index:insert_index] = moving_pages
                continue

            if op_type == "insert_after":
                insert_after = operation.get("page_number")
                insert_index = self._find_insert_index_after_origin(working_pages, insert_after)
                for offset in range(operation.get("count", 0)):
                    placeholder = {
                        "title": "",
                        "layout_description": "",
                        "key_points": [],
                        "asset_ref": None,
                        "_origin_page_number": None,
                        "_patch_instruction": operation.get("instruction", ""),
                    }
                    working_pages.insert(insert_index + offset, placeholder)
                    rewrite_targets.add(insert_index + offset)
                continue

            if op_type == "update":
                update_numbers = set(operation.get("page_numbers", []))
                for idx, page in enumerate(working_pages):
                    if page.get("_origin_page_number") in update_numbers:
                        existing_instruction = str(page.get("_patch_instruction") or "").strip()
                        new_instruction = str(operation.get("instruction") or "").strip()
                        if existing_instruction and new_instruction:
                            page["_patch_instruction"] = f"{existing_instruction}\n{new_instruction}"
                        elif new_instruction:
                            page["_patch_instruction"] = new_instruction
                        rewrite_targets.add(idx)

        if plan.get("apply_global_rewrite"):
            rewrite_targets.update(range(len(working_pages)))

        return working_pages, rewrite_targets

    def _find_insert_index_after_origin(self, pages: List[Dict[str, Any]], after_page_number: Optional[int]) -> int:
        if after_page_number is None:
            return len(pages)
        for idx, page in enumerate(pages):
            if page.get("_origin_page_number") == after_page_number:
                return idx + 1
        return len(pages)

    def _build_rewrite_chunks(self, target_indices: List[int], total_pages: int, max_chunk_size: int) -> List[tuple[int, int]]:
        if not target_indices or total_pages <= 0:
            return []

        chunks: List[tuple[int, int]] = []
        group_start = target_indices[0]
        previous = target_indices[0]
        for index in target_indices[1:]:
            group_size = previous - group_start + 1
            if index != previous + 1 or group_size >= max_chunk_size:
                chunks.append((group_start, previous))
                group_start = index
            previous = index
        chunks.append((group_start, previous))
        return chunks

    def _build_outline_digest(self, pagecontent: List[Dict[str, Any]], *, max_chars: int) -> str:
        parts: List[str] = []
        current_len = 0
        for idx, page in enumerate(pagecontent, start=1):
            bullet_preview = "; ".join((page.get("key_points") or [])[:3])
            line = (
                f"Page {idx}: title={page.get('title') or ''} | "
                f"layout={page.get('layout_description') or ''} | "
                f"bullets={bullet_preview}"
            ).strip()
            if not line:
                continue
            if current_len + len(line) > max_chars:
                break
            parts.append(line)
            current_len += len(line) + 1
        return "\n".join(parts)

    def _build_source_excerpt(
        self,
        *,
        source_text: str,
        mineru_output: str,
        max_chars: int,
    ) -> str:
        combined = (source_text or "").strip()
        if mineru_output and mineru_output.strip():
            combined = f"{combined}\n\n{mineru_output.strip()}".strip()
        if len(combined) <= max_chars:
            return combined
        return combined[:max_chars]

    async def _invoke_json_llm(
        self,
        *,
        chat_api_url: str,
        api_key: str,
        model_name: str,
        system_prompt: str,
        task_prompt: str,
        max_tokens: int = 4096,
    ) -> Any:
        base_url = str(chat_api_url or "").rstrip("/")
        if not base_url:
            raise ValueError("chat_api_url is required for outline refine")
        endpoint = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

        proxy = None
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            value = str(os.getenv(key) or "").strip()
            if value.startswith(("http://", "https://")):
                proxy = value
                break

        client_kwargs: Dict[str, Any] = {
            "timeout": httpx.Timeout(300.0),
            "trust_env": False,
        }
        if proxy:
            client_kwargs["proxy"] = proxy

        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        raw_text = self._extract_outline_text(
            (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
        )
        if not raw_text.strip():
            raise ValueError("LLM returned empty content for outline refine")
        return robust_parse_json(raw_text, strip_double_braces=True)
