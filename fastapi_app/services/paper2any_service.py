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
BASE_OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()

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

    async def list_history_files(self, email: str, request: Request) -> Dict[str, Any]:
        """
        列出历史文件
        """
        if not email:
             return {
                "success": True,
                "files": [],
            }
        base_dir = PROJECT_ROOT / "outputs" / email

        if not base_dir.exists():
            return {
                "success": True,
                "files": [],
            }

        files_data: list[dict] = []

        # 递归扫描所有文件
        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            
            # 排除 input 目录中的文件
            # 检查路径各部分是否包含 "input"
            if "input" in p.parts:
                continue

            # 只保留特定类型的文件
            suffix = p.suffix.lower()
            filename = p.name
            
            if suffix in {".pptx", ".pdf", ".png", ".svg"}:
                # 筛选逻辑优化：
                # 1. .pptx 都要显示
                # 2. paper2ppt 开头的文件都要显示 (包含 paper2ppt.pdf, paper2ppt_*.pptx)
                # 3. fig_ 开头的图片(png, svg)都要显示
                should_show = False
                if suffix == ".pptx":
                    should_show = True
                elif filename.startswith("paper2ppt"):
                    should_show = True
                elif filename.startswith("fig_") and suffix in {".png", ".svg"}:
                    should_show = True
                
                if should_show:
                    stat = p.stat()
                    url = _to_outputs_url(str(p), request)
                    
                    # 推断 workflow_type: outputs/email/task_type/...
                    try:
                        rel = p.relative_to(base_dir)
                        wf_type = rel.parts[0] if len(rel.parts) > 0 else "unknown"
                        file_id = str(rel)  # 使用相对路径作为唯一ID
                    except Exception:
                        wf_type = "unknown"
                        file_id = str(p.name) + "_" + str(stat.st_mtime)

                    files_data.append({
                        "id": file_id,
                        "file_name": p.name,
                        "file_size": stat.st_size,
                        "workflow_type": wf_type,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "download_url": url
                    })

        # 排序：按修改时间倒序
        files_data.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "success": True,
            "files": files_data,
        }

    async def generate_paper2figure(
        self,
        img_gen_model_name: str,
        chat_api_url: str,
        api_key: str,
        input_type: str,
        email: Optional[str],
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
        run_dir = self._create_run_dir(task_type, email)
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
            email=email or "",
        )

        # 5. 执行 workflow
        async with task_semaphore:
            p2f_resp = await run_paper2figure_wf_api(p2f_req, result_path=run_dir)

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
        email: Optional[str],
        file: Optional[UploadFile],
        file_kind: Optional[str],
        text: Optional[str],
        graph_type: str,
        language: str,
        style: str,
        figure_complex: str = "easy",
        edit_prompt: Optional[str] = None,
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
        run_dir = self._create_run_dir(task_type, email)
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
            figure_complex=figure_complex,
            email=email or "",
            edit_prompt=edit_prompt or "",
        )

        # 5. 执行 workflow
        async with task_semaphore:
            p2f_resp = await run_paper2figure_wf_api(p2f_req, result_path=run_dir)

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
        email: Optional[str],
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
        run_dir = self._create_run_dir("paper2beamer", email)
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
            resp: FeaturePaper2VideoResponse = await paper2video_endpoint(req)
            
            if not resp.success:
                raise HTTPException(status_code=500, detail="Paper to PPT generation failed.")
            output_path = Path(resp.ppt_path)

        return output_path

    # ---------------- 内部工具方法 ---------------- #

    def _create_run_dir(self, task_type: str, email: Optional[str] = None) -> Path:
        """
        为一次请求创建独立目录：
        - 登录用户 (有 email): outputs/{email}/{task_type}/{timestamp}/
        - 匿名用户 (无 email): outputs/{task_type}/{timestamp}_{short_uuid}/
        """
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        
        if email:
            # 登录用户：邮箱/任务/时间戳
            run_dir = BASE_OUTPUT_DIR / email / task_type / ts
        else:
            # 匿名用户：保持旧逻辑
            rid = uuid.uuid4().hex[:6]
            run_dir = BASE_OUTPUT_DIR / task_type / f"{ts}_{rid}"

        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(parents=True, exist_ok=True)

        return run_dir.resolve()

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
        elif input_type == "FIGURE":
            # FIGURE 模式下，可能是重新生成（传入URL/path在text里），也可能是上传图片（但这种情况通常用 input_type=image）
            # 这里宽松校验，允许 text 不为空
            if not text and not file:
                raise HTTPException(
                    status_code=400,
                    detail="text (image path) or file is required when input_type is 'FIGURE'",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="invalid input_type, must be one of: file, text, image, FIGURE",
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
        elif input_type == "FIGURE":
            # 如果是 FIGURE 模式
            if file:
                # 依然支持上传文件
                original_name = file.filename or "uploaded.png"
                ext = Path(original_name).suffix or ".png"
                input_path = input_dir / f"input{ext}"
                content_bytes = await file.read()
                input_path.write_bytes(content_bytes)
                return "FIGURE", str(input_path)
            else:
                # 认为是传入了图片路径或 URL (在 text 字段)
                # 简单起见，直接透传 text 作为 input_content
                # 后续 workflow 需自行处理该路径/URL
                return "FIGURE", text or ""
        else:
            raise HTTPException(status_code=400, detail="unsupported input_type")
