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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, UploadFile

from fastapi_app.schemas import (
    FullPipelineRequest,
    OutlineRefineRequest,
    PageContentRequest,
    PPTGenerationRequest,
)
from fastapi_app.utils import _from_outputs_url, _to_outputs_url
from fastapi_app.workflow_adapters.wa_paper2ppt import (
    run_paper2page_content_wf_api,
    run_paper2page_content_refine_wf_api,
    run_paper2ppt_full_pipeline,
    run_paper2ppt_wf_api,
)
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()


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
        wf_input_type, wf_input_content = await self._prepare_input_for_pagecontent(
            input_dir=input_dir,
            input_type=req.input_type,
            file=file,
            text=req.text,
        )

        # 组装老的 Paper2PPTRequest 以复用现有 workflow adapter
        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        # 转换字符串布尔值
        use_long_paper_bool = str(req.use_long_paper).lower() in ("true", "1", "yes")

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=req.chat_api_url,
            chat_api_key=req.api_key,
            api_key=req.api_key,
            model=req.model,
            gen_fig_model="",
            input_type=wf_input_type,
            input_content=wf_input_content,
            style=req.style,
            reference_img=str(reference_img_path) if reference_img_path is not None else "",
            email=req.email or "",
            page_count=req.page_count,
            use_long_paper=use_long_paper_bool,
        )

        resp_model = await run_paper2page_content_wf_api(p2ppt_req, result_path=run_dir)

        resp_dict = resp_model.model_dump()
        if request is not None:
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request
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

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=req.chat_api_url,
            chat_api_key=req.api_key,
            api_key=req.api_key,
            model=req.model,
            gen_fig_model="",
            input_type="TEXT",
            input_content="",
            style="",
            email=req.email or "",
            page_count=len(pc),
        )

        result_root: Path | None = None
        if req.result_path:
            base_dir = Path(req.result_path)
            if not base_dir.is_absolute():
                base_dir = PROJECT_ROOT / base_dir
            result_root = base_dir.resolve()

        resp_model = await run_paper2page_content_refine_wf_api(
            p2ppt_req,
            pagecontent=pc,
            outline_feedback=req.outline_feedback,
            result_path=result_root,
        )

        resp_dict = resp_model.model_dump()
        if request is not None:
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request
            )
        if request is not None:
            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_model.result_path, request)
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
        # 处理 result_path
        base_dir = Path(req.result_path)
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
        base_dir = base_dir.resolve()

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
                        item[key] = _from_outputs_url(item[key])

        # 转换字符串布尔值
        get_down_bool = str(req.get_down).lower() in ("true", "1", "yes")
        all_edited_down_bool = str(req.all_edited_down).lower() in ("true", "1", "yes")

        # 校验编辑/生成模式
        if get_down_bool:
            if req.page_id is None:
                raise HTTPException(status_code=400, detail="page_id is required when get_down=true")
            if not (req.edit_prompt or "").strip():
                raise HTTPException(status_code=400, detail="edit_prompt is required when get_down=true")
        else:
            if not pc:
                raise HTTPException(status_code=400, detail="pagecontent is required when get_down=false")

        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=req.chat_api_url,
            chat_api_key=req.api_key,
            api_key=req.api_key,
            model=req.model,
            gen_fig_model=req.img_gen_model_name,
            input_type="PDF",
            input_content="",
            aspect_ratio=req.aspect_ratio,
            style=req.style,
            ref_img=str(reference_img_path) if reference_img_path else "",
            email=req.email or "",
            all_edited_down=all_edited_down_bool,
        )

        resp_model = await run_paper2ppt_wf_api(
            p2ppt_req,
            pagecontent=pc,
            result_path=str(base_dir),
            get_down=get_down_bool,
            edit_page_num=req.page_id,
            edit_page_prompt=req.edit_prompt,
        )

        resp_dict = resp_model.model_dump()

        # 路径 -> URL
        if request is not None:
            if resp_dict.get("ppt_pdf_path"):
                resp_dict["ppt_pdf_path"] = _to_outputs_url(resp_dict["ppt_pdf_path"], request)
            if resp_dict.get("ppt_pptx_path"):
                resp_dict["ppt_pptx_path"] = _to_outputs_url(resp_dict["ppt_pptx_path"], request)
            resp_dict["pagecontent"] = self._convert_pagecontent_paths_to_urls(
                resp_dict.get("pagecontent", []), request
            )

            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_model.result_path, request)
        else:
            resp_dict["all_output_files"] = []

        return resp_dict

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

        from fastapi_app.schemas import Paper2PPTRequest  # 局部导入避免循环

        p2ppt_req = Paper2PPTRequest(
            language=req.language,
            chat_api_url=req.chat_api_url,
            chat_api_key=req.api_key,
            api_key=req.api_key,
            model=req.model,
            gen_fig_model=req.img_gen_model_name,
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
                resp_dict.get("pagecontent", []), request
            )

            resp_dict["all_output_files"] = self._collect_output_files_as_urls(resp_model.result_path, request)
        else:
            resp_dict["all_output_files"] = []

        return resp_dict

    # ---------------- 内部工具方法 ---------------- #

    def _create_timestamp_run_dir(self, email: Optional[str]) -> Path:
        """根据当前时间戳和邮箱创建本次请求的根输出目录。

        目录结构：
            outputs/{email or 'default'}/paper2ppt/<timestamp>/

        说明：
        - email 为 None 或空字符串时，使用 "default" 作为子目录名；
        - 始终在 PROJECT_ROOT / outputs 下创建目录，保证和原始实现兼容。
        """
        import time

        ts = int(time.time())
        # 如果有 email，则使用 email，否则使用 default
        code = email or "default"
        run_dir = PROJECT_ROOT / BASE_OUTPUT_DIR / code / "paper2ppt" / str(ts)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _convert_pagecontent_paths_to_urls(
        self,
        pagecontent: List[Dict[str, Any]],
        request: Request,
    ) -> List[Dict[str, Any]]:
        """Convert local output paths inside pagecontent to browser-accessible URLs."""
        if not pagecontent:
            return pagecontent

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
                # Skip already-normalized URLs
                if value.startswith("http") or value.startswith("/outputs/"):
                    continue
                # Only convert when path is absolute or explicitly points into outputs
                if os.path.isabs(value) or "/outputs/" in value or value.startswith("outputs/"):
                    item[key] = _to_outputs_url(value, request)

        return pagecontent

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

        # 尝试复用历史 reference.*
        if input_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = input_dir / f"reference{ext}"
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

    def _parse_pagecontent_json(self, pagecontent_json: str) -> List[Dict[str, Any]]:
        try:
            import json

            obj = json.loads(pagecontent_json)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid pagecontent json: {e}") from e

        if not isinstance(obj, list):
            raise HTTPException(status_code=400, detail="pagecontent must be a JSON list")

        for i, it in enumerate(obj):
            if not isinstance(it, dict):
                raise HTTPException(status_code=400, detail=f"pagecontent[{i}] must be an object(dict)")
        return obj
