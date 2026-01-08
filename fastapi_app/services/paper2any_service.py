from __future__ import annotations

import asyncio
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from fastapi import HTTPException, UploadFile, Request
from fastapi_app.routers.paper2video import paper2video_endpoint, FeaturePaper2VideoRequest, FeaturePaper2VideoResponse
from fastapi_app.schemas import Paper2FigureRequest, VerifyLlmRequest, VerifyLlmResponse
from fastapi_app.workflow_adapters import run_paper2figure_wf_api
from fastapi_app.utils import _to_outputs_url
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = Path("outputs")

# 全局信号量：控制重任务并发度（排队机制）
# 保持在 Service 层或模块级别，因为它是全局共享的资源控制
task_semaphore = asyncio.Semaphore(1)


class Paper2AnyService:
    """
    Paper2Any 业务 Service 层
    
    职责：
    - 处理 paper2figure (Paper2Graph) 相关逻辑
    - 处理 paper2beamer 相关逻辑
    - 处理 LLM 验证逻辑
    - 文件输入落地与目录管理
    """

    async def verify_llm_connection(self, req: VerifyLlmRequest) -> VerifyLlmResponse:
        """
        Verify LLM connection by sending a simple 'Hi' message.
        """
        api_url = req.api_url.rstrip("/")
        if api_url.endswith("/chat/completions"):
            target_url = api_url
        else:
            target_url = f"{api_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {req.api_key}",
        }
        
        payload = {
            "model": req.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(target_url, json=payload, headers=headers)
                
                if resp.status_code != 200:
                    error_msg = f"API Error {resp.status_code}: {resp.text[:200]}"
                    return VerifyLlmResponse(success=False, error=error_msg)
                
                return VerifyLlmResponse(success=True)
                
        except Exception as e:
            log.error(f"LLM Verification failed: {e}")
            return VerifyLlmResponse(success=False, error=str(e))

    async def list_history_files(self, invite_code: str, request: Request) -> Dict[str, Any]:
        """
        列出历史文件
        """
        base_dir = PROJECT_ROOT / "outputs" / invite_code

        if not base_dir.exists():
            return {
                "success": True,
                "files": [],
            }

        file_urls: list[str] = []

        # 第一层：invite_code/level1
        for level1 in base_dir.iterdir():
            if not level1.is_dir():
                continue

            # 第二层：invite_code/level1/level2
            for level2 in level1.iterdir():
                if level2.is_file():
                    # 若 level2 直接是文件，也纳入
                    if level2.suffix.lower() in {".pptx", ".png", ".svg"}:
                        file_urls.append(_to_outputs_url(str(level2), request))
                elif level2.is_dir():
                    # 只取该目录里的直接文件，不再往下递归
                    for p in level2.iterdir():
                        if p.is_file() and p.suffix.lower() in {".pptx", ".png", ".svg"}:
                            file_urls.append(_to_outputs_url(str(p), request))

        # 排序：按路径字符串倒序，粗略实现“新文件在前”
        file_urls.sort(reverse=True)

        return {
            "success": True,
            "files": file_urls,
        }

    async def generate_paper2figure(
        self,
        img_gen_model_name: str,
        chat_api_url: str,
        api_key: str,
        input_type: str,
        invite_code: Optional[str],
        file: Optional[UploadFile],
        file_kind: Optional[str],
        text: Optional[str],
        graph_type: str,
        language: str,
        figure_complex: str,
        style: str,
    ) -> Path:
        """
        执行 paper2figure 生成，返回生成的 PPTX 文件绝对路径。
        """
        # 1. 基础参数校验
        self._validate_input(input_type, file, file_kind, text)

        # 2. 确定 task_type 和 complexity
        if graph_type == "model_arch":
            task_type = "paper2fig"
            final_figure_complex = figure_complex or "easy"
        elif graph_type == "tech_route":
            task_type = "paper2tec"
            final_figure_complex = "easy"
        elif graph_type == "exp_data":
            task_type = "paper2exp"
            final_figure_complex = "easy"
        else:
            raise HTTPException(status_code=400, detail="invalid graph_type")

        # 3. 创建目录并保存输入
        run_dir = self._create_run_dir(task_type)
        input_dir = run_dir / "input"
        
        real_input_type, real_input_content = await self._save_and_prepare_input(
            input_dir, input_type, file, file_kind, text
        )

        # 4. 构造 Request
        p2f_req = Paper2FigureRequest(
            language=language,
            chat_api_url=chat_api_url,
            chat_api_key=api_key,
            api_key=api_key,
            model="gpt-4o",
            gen_fig_model=img_gen_model_name,
            input_type=real_input_type,
            input_content=real_input_content,
            aspect_ratio="16:9",
            graph_type=graph_type,
            style=style,
            figure_complex=final_figure_complex,
            invite_code=invite_code or "",
        )

        # 5. 执行 workflow
        async with task_semaphore:
            p2f_resp = await run_paper2figure_wf_api(p2f_req)

        # 6. 处理返回路径
        raw_path = Path(p2f_resp.ppt_filename)
        if not raw_path.is_absolute():
            ppt_path = PROJECT_ROOT / raw_path
        else:
            ppt_path = raw_path

        if not ppt_path.exists() or not ppt_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"generated ppt file not found or not a file: {ppt_path}",
            )

        return ppt_path

    async def generate_paper2figure_json(
        self,
        request: Request,
        img_gen_model_name: str,
        chat_api_url: str,
        api_key: str,
        input_type: str,
        invite_code: Optional[str],
        file: Optional[UploadFile],
        file_kind: Optional[str],
        text: Optional[str],
        graph_type: str,
        language: str,
        style: str,
    ) -> Dict[str, Any]:
        """
        执行 paper2figure 生成，返回 JSON 响应数据（包含 URL）。
        """
        # 1. 基础参数校验
        self._validate_input(input_type, file, file_kind, text)

        # 2. 确定 task_type
        if graph_type == "model_arch":
            task_type = "paper2fig"
        elif graph_type == "tech_route":
            task_type = "paper2tec"
        elif graph_type == "exp_data":
            task_type = "paper2exp"
        else:
            raise HTTPException(status_code=400, detail="invalid graph_type")

        # 3. 创建目录并保存输入
        run_dir = self._create_run_dir(task_type)
        input_dir = run_dir / "input"
        
        real_input_type, real_input_content = await self._save_and_prepare_input(
            input_dir, input_type, file, file_kind, text
        )

        # 4. 构造 Request
        p2f_req = Paper2FigureRequest(
            language=language,
            chat_api_url=chat_api_url,
            chat_api_key=api_key,
            api_key=api_key,
            model="gpt-4o",
            gen_fig_model=img_gen_model_name,
            input_type=real_input_type,
            input_content=real_input_content,
            aspect_ratio="16:9",
            graph_type=graph_type,
            style=style,
            invite_code=invite_code or "",
        )

        # 5. 执行 workflow
        async with task_semaphore:
            p2f_resp = await run_paper2figure_wf_api(p2f_req)

        # 6. 构造 URL 响应
        safe_ppt = _to_outputs_url(p2f_resp.ppt_filename, request)
        safe_svg = _to_outputs_url(p2f_resp.svg_filename, request) if p2f_resp.svg_filename else ""
        safe_png = _to_outputs_url(p2f_resp.svg_image_filename, request) if p2f_resp.svg_image_filename else ""

        safe_all_files: list[str] = []
        for abs_path in getattr(p2f_resp, "all_output_files", []) or []:
            if abs_path:
                safe_all_files.append(_to_outputs_url(abs_path, request))

        return {
            "success": p2f_resp.success,
            "ppt_filename": safe_ppt,
            "svg_filename": safe_svg,
            "svg_image_filename": safe_png,
            "all_output_files": safe_all_files,
        }

    async def generate_paper2beamer(
        self,
        model_name: str,
        chat_api_url: str,
        api_key: str,
        input_type: str,
        invite_code: Optional[str],
        file: Optional[UploadFile],
        file_kind: Optional[str],
        language: str,
    ) -> Path:
        """
        执行 paper2beamer 生成，返回生成的 PDF 文件绝对路径。
        """
        if input_type != "file":
            raise HTTPException(status_code=400, detail="paper2beamer currently only supports input_type='file'")

        if file is None:
            raise HTTPException(status_code=400, detail="file is required for paper2beamer")

        if file_kind not in ("pdf", None):
            raise HTTPException(status_code=400, detail="file_kind must be 'pdf' for paper2beamer")

        # 2. 创建目录
        run_dir = self._create_run_dir("paper2beamer")
        input_dir = run_dir / "input"
        
        # 3. 保存输入 PDF
        original_name = file.filename or "uploaded.pdf"
        ext = Path(original_name).suffix or ".pdf"
        input_path = input_dir / f"input{ext}"
        content_bytes = await file.read()
        input_path.write_bytes(content_bytes)
        abs_input_path = input_path.resolve()

        # 4. 执行 workflow
        async with task_semaphore:
            req = FeaturePaper2VideoRequest(
                model=model_name,
                chat_api_url=chat_api_url,
                api_key=api_key,
                pdf_path=str(abs_input_path),
                img_path="",
                language=language,
            )
            # 注意：paper2video_endpoint 实际上是在 router 中定义的，但看起来它像是一个 helper function
            # 如果它是一个 router endpoint 函数，直接调用可能不合适，但在 python 中它只是个 async 函数。
            # 检查 paper2video.py 的实现，如果它包含 HTTP 逻辑（如 Form, File 依赖），直接调用会有问题。
            # 这里原代码直接调用了 paper2video_endpoint(req)，这意味着 paper2video_endpoint 接受 Pydantic model
            # 而不是 FastAPI 依赖。我们需要确认这一点。
            # 根据原 routers/paper2any.py: 
            # from fastapi_app.routers.paper2video import paper2video_endpoint, ...
            # resp: FeaturePaper2VideoResponse = await paper2video_endpoint(req)
            # 假设 paper2video_endpoint 是设计为可重用的 service-like 函数。
            resp: FeaturePaper2VideoResponse = await paper2video_endpoint(req)
            
            if not resp.success:
                raise HTTPException(status_code=500, detail="Paper to PPT generation failed.")
            output_path = Path(resp.ppt_path)

        return output_path

    # ---------------- 内部工具方法 ---------------- #

    def _create_run_dir(self, task_type: str) -> Path:
        """
        为一次请求创建独立目录：
            outputs/{task_type}/{timestamp}_{short_uuid}/
        """
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        rid = uuid.uuid4().hex[:6]
        run_dir = BASE_OUTPUT_DIR / task_type / f"{ts}_{rid}"

        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(parents=True, exist_ok=True)

        return run_dir

    def _validate_input(
        self,
        input_type: str,
        file: Optional[UploadFile],
        file_kind: Optional[str],
        text: Optional[str],
    ):
        if input_type in ("file", "image"):
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="file is required when input_type is 'file' or 'image'",
                )
            if file_kind not in ("pdf", "image"):
                raise HTTPException(
                    status_code=400,
                    detail="file_kind must be 'pdf' or 'image'",
                )
        elif input_type == "text":
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="text is required when input_type is 'text'",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="invalid input_type, must be one of: file, text, image",
            )

    async def _save_and_prepare_input(
        self,
        input_dir: Path,
        input_type: str,
        file: Optional[UploadFile],
        file_kind: Optional[str],
        text: Optional[str],
    ) -> tuple[str, str]:
        """
        保存文件并返回 (real_input_type, real_input_content)
        """
        if input_type in ("file", "image"):
            if file is None: # Should be caught by validate_input, but type checker might complain
                raise HTTPException(status_code=400, detail="File missing")
                
            original_name = file.filename or "uploaded"
            ext = Path(original_name).suffix or ""
            input_path = input_dir / f"input{ext}"
            content_bytes = await file.read()
            input_path.write_bytes(content_bytes)
            
            if file_kind == "pdf":
                return "PDF", str(input_path)
            else:
                return "FIGURE", str(input_path)
        elif input_type == "text":
            input_path = input_dir / "input.txt"
            input_path.write_text(text or "", encoding="utf-8")
            return "TEXT", text or ""
        else:
            raise HTTPException(status_code=400, detail="unsupported input_type")
