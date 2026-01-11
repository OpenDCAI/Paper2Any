from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import File, UploadFile, HTTPException
from fastapi_app.schemas import Paper2PPTRequest
from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# 全局信号量控制并发
task_semaphore = asyncio.Semaphore(1)

BASE_OUTPUT_DIR = Path("outputs")
PROJECT_ROOT = get_project_root()

class Image2PPTService:
    def __init__(self):
        pass

    def _create_run_dir(self, invite_code: Optional[str], task_type: str) -> Path:
        """
        为一次 image2ppt 请求创建独立目录：
            outputs/{invite_code}/{task_type}/{timestamp}/input/
        """
        ts = int(datetime.utcnow().timestamp())
        code = invite_code or "default"
        run_dir = BASE_OUTPUT_DIR / code / task_type / str(ts)

        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        return run_dir

    async def generate_ppt(
        self,
        image_file: UploadFile,
        chat_api_url: Optional[str],
        api_key: Optional[str],
        invite_code: Optional[str],
        use_ai_edit: bool,
        model: str,
        gen_fig_model: str,
        language: str,
        style: str,
        page_count: int,
    ) -> Path:
        """
        执行 image2ppt 的业务逻辑，返回生成的 PPTX 文件路径
        """
        # 0.5 如果启用 AI 增强，必须校验 API 配置
        if use_ai_edit:
            if not chat_api_url or not api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="When use_ai_edit is True, chat_api_url and api_key are required"
                )

        # 1. 基础参数校验
        if image_file is None:
            raise HTTPException(status_code=400, detail="image_file is required")

        # 2. 为本次请求创建独立目录
        run_dir = self._create_run_dir(invite_code, "image2ppt")
        input_dir = run_dir / "input"

        original_name = image_file.filename or "uploaded.png"
        ext = Path(original_name).suffix or ".png"
        input_path = input_dir / f"input{ext}"

        content_bytes = await image_file.read()
        input_path.write_bytes(content_bytes)
        abs_img_path = input_path.resolve()

        log.info(f"[image2ppt] received file saved to {abs_img_path}")

        # 3. 构造适配层请求
        # 注意：这里复用 run_pdf2ppt_wf_api，因为底层 wf_pdf2ppt_optimized 支持 input_type="FIGURE"
        wf_req = Paper2PPTRequest(
            input_type="FIGURE",
            input_content=str(abs_img_path),
            chat_api_url=chat_api_url or "",
            api_key=api_key or "",
            model=model,
            gen_fig_model=gen_fig_model,
            language=language,
            style=style,
            page_count=page_count,
            invite_code=invite_code or "",
            use_ai_edit=use_ai_edit,
        )

        # 4. 调用 workflow（受信号量保护）
        async with task_semaphore:
            wf_resp = await run_pdf2ppt_wf_api(wf_req)

        # 5. 获取生成的 PPT 路径
        ppt_path = Path(wf_resp.ppt_pptx_path or "")
        if not ppt_path.is_absolute():
            ppt_path = (PROJECT_ROOT / ppt_path).resolve()

        if not ppt_path.exists() or not ppt_path.is_file():
            # 尝试查找同一目录下的pptx文件
            if ppt_path.parent.exists():
                pptx_files = list(ppt_path.parent.glob("*.pptx"))
                if pptx_files:
                    ppt_path = pptx_files[0]
                    log.info(f"[image2ppt] found alternative PPT file: {ppt_path}")
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"generated PPT file not found: {ppt_path}",
                    )
            else:
                 raise HTTPException(
                    status_code=500,
                    detail=f"generated PPT file not found or not a file: {ppt_path}",
                )

        log.info(f"[image2ppt] returning PPT file: {ppt_path}")
        
        return ppt_path
