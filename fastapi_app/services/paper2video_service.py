# -*- coding: utf-8 -*-
"""
paper2video 业务 Service 层。

职责：
- 第一步（generate-subtitle）：创建任务目录、落盘 PDF/头像/语音，调用 adapter 跑“生成字幕/脚本”工作流，
  将返回的 script_pages 中图片路径转为前端可访问 URL，并返回 result_path + script_pages + state_snapshot。
- 第二步（generate-video）：校验 result_path、解析 script_pages JSON，可选传入 state_snapshot，
  调用 adapter 跑“生成视频”工作流（有 state_snapshot 时复用第一步 state），将返回的视频路径转为 video_url。

不直接调用 dataflow_agent.workflow.run_workflow，由 workflow_adapters.wa_paper2video 完成。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, UploadFile

from fastapi_app.schemas import GenerateSubtitleResponse, GenerateVideoResponse
from fastapi_app.utils import _to_outputs_url
from fastapi_app.workflow_adapters.wa_paper2video import (
    run_paper2video_generate_subtitle_wf_api,
    run_paper2video_generate_video_wf_api,
)
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.toolkits.p2vtool.p2v_tool import liveportrait_face_detect, pptx_to_pdf

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
BASE_OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()


class Paper2VideoService:
    """paper2video 两步流程的业务编排。"""

    @staticmethod
    def _normalize_talking_model(talking_model: str) -> str:
        normalized = (talking_model or "").strip().lower()
        if normalized not in {"", "liveportrait"}:
            log.info(
                "[Paper2VideoService] force talking_model=%s -> liveportrait",
                talking_model,
            )
        return "liveportrait"

    @staticmethod
    def _resolve_video_path(base_dir: Path, video_path: str) -> str:
        candidate = (video_path or "").strip()
        if candidate:
            path = Path(candidate)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            else:
                path = path.resolve()
            if path.is_file():
                return str(path)

        for fallback in (base_dir / "video.mp4", base_dir / "2_merge.mp4", base_dir / "1_merge.mp4"):
            if fallback.is_file():
                return str(fallback)

        mp4_files = [
            p for p in base_dir.rglob("*.mp4")
            if p.is_file() and "talking_video" not in p.parts and "merge" not in p.parts
        ]
        if mp4_files:
            mp4_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(mp4_files[0])

        return ""

    def _create_timestamp_run_dir(self, email: Optional[str]) -> Path:
        """
        根据当前时间戳和邮箱创建本次请求的根输出目录。
        目录结构：outputs/{email or 'default'}/paper2video/<timestamp>/
        """
        import time
        ts = int(time.time())
        code = email or "default"
        run_dir = PROJECT_ROOT / BASE_OUTPUT_DIR / code / "paper2video" / str(ts)
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("[Paper2VideoService] created run_dir: %s", run_dir)
        return run_dir

    async def _save_upload(
        self,
        save_path: Path,
        upload: UploadFile,
        allowed_ext: Optional[List[str]] = None,
    ) -> None:
        """将 UploadFile 写入 save_path；可选校验后缀。"""
        if allowed_ext:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in allowed_ext:
                raise HTTPException(
                    status_code=400,
                    detail=f"file type not allowed, expected one of {allowed_ext}",
                )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        content = await upload.read()
        save_path.write_bytes(content)
        log.info("[Paper2VideoService] saved upload to %s", save_path)

    async def run_generate_subtitle(
        self,
        *,
        email: Optional[str] = None,
        api_key: str = "",
        chat_api_url: str = "",
        model: str = "gpt-4o",
        tts_model: str = "cosyvoice-v3-flash",
        tts_voice_name: str = "",
        language: str = "en",
        talking_model: str = "liveportrait",
        file: Optional[UploadFile] = None,
        avatar: Optional[UploadFile] = None,
        avatar_preset: Optional[str] = None,
        voice: Optional[UploadFile] = None,
        voice_preset: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> Dict[str, Any]:
        """
        第一步：落盘 PDF（必填）、可选头像/语音，调用工作流生成字幕/脚本。
        头像可为上传文件(avatar)或系统预设(avatar_preset)，预设从 frontend-workflow/public/paper2video/avatar/{id}.png 复制。
        返回 dict：success, result_path, script_pages, state_snapshot, all_output_files（可选 URL 列表）。
        """
        if not file:
            log.warning("[Paper2VideoService] run_generate_subtitle: missing file")
            raise HTTPException(status_code=400, detail="file is required (PDF or PPTX)")

        run_dir = self._create_timestamp_run_dir(email)
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # 落盘：支持 PDF 或 PPTX
        ext = (Path(file.filename or "").suffix or ".pdf").lower()
        if ext not in [".pdf", ".pptx"]:
            raise HTTPException(
                status_code=400,
                detail="file must be PDF or PPTX",
            )
        input_path = (input_dir / f"input{ext}").resolve()
        await self._save_upload(input_path, file, allowed_ext=[".pdf", ".pptx"])
        log.info("[Paper2VideoService] saved file to %s", input_path)

        # 若为 PPTX，先转为 PDF，后续流程统一用 PDF
        if ext == ".pptx":
            try:
                pdf_path_str = await asyncio.to_thread(
                    pptx_to_pdf,
                    input_path,
                    input_dir,
                )
                pdf_path = Path(pdf_path_str)
            except Exception as e:
                log.exception("[Paper2VideoService] PPTX to PDF conversion failed: %s", e)
                msg = str(e).strip() or "PPTX 转 PDF 失败"
                if "LibreOffice not found" in str(e) or "not found" in str(e).lower():
                    msg = "未检测到 LibreOffice，无法转换 PPTX。请安装 LibreOffice（如 apt install libreoffice）或上传 PDF。"
                raise HTTPException(status_code=500, detail=msg) from e
        else:
            pdf_path = input_path
        log.info("[Paper2VideoService] using PDF for workflow: %s", pdf_path)
        talking_model = self._normalize_talking_model(talking_model)

        # 可选：数字人头像（上传文件优先；否则使用系统预设 avatar_preset）
        # 如果都没有，则说明没有选择数字人，该字段为 None
        avatar_path: Optional[Path] = None
        if avatar:
            ext = Path(avatar.filename or "").suffix.lower() or ".png"
            if ext not in [".jpg", ".jpeg", ".png"]:
                raise HTTPException(status_code=400, detail="avatar must be jpg/png")
            avatar_path = (input_dir / f"avatar{ext}").resolve()
            await self._save_upload(avatar_path, avatar)
            log.info("[Paper2VideoService] saved avatar to %s", avatar_path)
        elif avatar_preset and avatar_preset.strip():
            preset_id = avatar_preset.strip()
            if not re.match(r"^[a-zA-Z0-9_-]+$", preset_id):
                raise HTTPException(status_code=400, detail="avatar_preset must be alphanumeric (e.g. avatar1)")
            preset_dir = (PROJECT_ROOT / "frontend-workflow" / "public" / "paper2video" / "avatar").resolve()
            for ext in (".png", ".jpg", ".jpeg"):
                src = preset_dir / f"{preset_id}{ext}"
                if src.is_file():
                    avatar_path = (input_dir / f"avatar{ext}").resolve()
                    shutil.copy2(src, avatar_path)
                    log.info("[Paper2VideoService] copied preset avatar %s to %s", src, avatar_path)
                    break
            if avatar_path is None:
                log.warning("[Paper2VideoService] preset avatar not found: %s in %s", preset_id, preset_dir)
                raise HTTPException(status_code=400, detail=f"avatar_preset '{preset_id}' not found in public/paper2video/avatar")

        # 用户上传的数字人头像且使用云数字人(LivePortrait)时，调用 LivePortrait 图像检测；系统预设头像不检测；Key 仅从环境变量 LIVEPORTRAIT_KEY 读取
        if avatar_path is not None and avatar is not None and (talking_model or "").strip().lower() == "liveportrait":
            detect_key = (os.environ.get("LIVEPORTRAIT_KEY", "") or "").strip()
            if detect_key:
                try:
                    passed, detect_message = liveportrait_face_detect(detect_key, avatar_path)
                    if not passed:
                        msg = (detect_message or "图像不符合数字人规范").strip()
                        if not msg:
                            msg = "图像检测未通过"
                        log.warning("[Paper2VideoService] LivePortrait face detect failed: %s", msg)
                        return {
                            "success": False,
                            "message": f"数字人图像检测未通过：{msg}",
                            "result_path": "",
                            "script_pages": [],
                            "state_snapshot": None,
                        }
                except Exception as e:
                    log.exception("[Paper2VideoService] LivePortrait face detect error: %s", e)
                    return {
                        "success": False,
                        "message": f"数字人图像检测服务异常：{str(e)}",
                        "result_path": "",
                        "script_pages": [],
                        "state_snapshot": None,
                    }
            else:
                log.warning("[Paper2VideoService] no API key for LivePortrait detect, skip validation")

        voice_path: Optional[Path] = None
        if voice is not None or (voice_preset and voice_preset.strip()):
            log.info("[Paper2VideoService] ignore local voice input, paper2video now uses CosyVoice API only")

        # 调用 adapter：生成字幕/脚本（工作流内部会读 run_dir/input，写 script_pages 等）
        resp = await run_paper2video_generate_subtitle_wf_api(
            result_path=run_dir,
            paper_pdf_path=str(pdf_path),
            ref_img_path=str(avatar_path) if avatar_path else "",
            ref_audio_path=str(voice_path) if voice_path else "",
            ref_text="",
            chat_api_url=chat_api_url,
            api_key=api_key,
            model=model,
            tts_model=tts_model,
            tts_voice_name=tts_voice_name or "",
            language=language,
            email=email or "",
            talking_model=talking_model,
        )
        log.info("[Paper2VideoService] run_generate_subtitle adapter returned success=%s", resp.get("success"))
        if not resp.get("success", False):
            return {
                "success": False,
                "message": resp.get("message") or "脚本生成失败",
                "result_path": "",
                "script_pages": [],
                "state_snapshot": None,
                "all_output_files": [],
            }

        result_path = resp.get("result_path", "")
        script_pages_raw = resp.get("script_pages") or []

        # 将 script_pages 中的本地路径转为前端可访问 URL
        script_pages: List[Dict[str, Any]] = []
        for item in script_pages_raw:
            page_num = item.get("page_num", 0)
            image_path = item.get("image_url") or item.get("image_path") or ""
            script_text = item.get("script_text") or item.get("scriptText") or ""
            if image_path and request is not None:
                image_url = _to_outputs_url(image_path, request)
            else:
                image_url = image_path if image_path and (image_path.startswith("http") or image_path.startswith("/")) else ""
            script_pages.append({
                "page_num": page_num,
                "image_url": image_url,
                "script_text": script_text,
            })
        log.info("[Paper2VideoService] script_pages count=%s", len(script_pages))

        # 可选：收集本次任务产出文件 URL，便于前端预加载
        all_output_files: List[str] = []
        if request is not None and result_path:
            root = Path(result_path)
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            if root.exists():
                for p in root.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".wav", ".mp4"}:
                        all_output_files.append(_to_outputs_url(str(p), request))
        log.info("[Paper2VideoService] all_output_files count=%s", len(all_output_files))

        state_snapshot = resp.get("state_snapshot")
        return {
            "success": True,
            "result_path": result_path,
            "script_pages": script_pages,
            "state_snapshot": state_snapshot,
            "all_output_files": all_output_files,
        }

    async def run_generate_video(
        self,
        *,
        result_path: str = "",
        script_pages_json: str = "",
        state_snapshot_json: Optional[str] = None,
        email: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> Dict[str, Any]:
        """
        第二步：根据 result_path 与用户编辑后的 script_pages（JSON 字符串），可选传入第一步返回的 state_snapshot（JSON），调用工作流生成最终视频。
        返回 dict：success, video_url（优先）, video_path。
        """
        if not result_path or not result_path.strip():
            log.warning("[Paper2VideoService] run_generate_video: missing result_path")
            raise HTTPException(status_code=400, detail="result_path is required")

        base_dir = Path(result_path)
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
        base_dir = base_dir.resolve()
        if not base_dir.exists():
            log.warning("[Paper2VideoService] run_generate_video: result_path not exists: %s", base_dir)
            raise HTTPException(status_code=400, detail=f"result_path not exists: {result_path}")

        # 解析 script_pages JSON
        try:
            script_pages = json.loads(script_pages_json or "[]")
        except json.JSONDecodeError as e:
            log.warning("[Paper2VideoService] run_generate_video: invalid script_pages json: %s", e)
            raise HTTPException(status_code=400, detail="invalid script_pages json") from e
        if not isinstance(script_pages, list):
            raise HTTPException(status_code=400, detail="script_pages must be a JSON array")

        log.info("[Paper2VideoService] run_generate_video: script_pages count=%s", len(script_pages))

        state_snapshot: Optional[Dict[str, Any]] = None
        if state_snapshot_json and state_snapshot_json.strip():
            try:
                state_snapshot = json.loads(state_snapshot_json)
            except json.JSONDecodeError:
                log.warning("[Paper2VideoService] run_generate_video: invalid state_snapshot json, ignoring")
        resp = await run_paper2video_generate_video_wf_api(
            result_path=result_path,
            script_pages=script_pages,
            state_snapshot=state_snapshot,
        )
        log.info("[Paper2VideoService] run_generate_video adapter returned success=%s", resp.get("success"))
        if not resp.get("success", False):
            return {
                "success": False,
                "message": resp.get("message") or "视频生成失败",
                "video_url": "",
                "video_path": "",
            }

        video_path = self._resolve_video_path(base_dir, resp.get("video_path") or "")
        if not video_path:
            return {
                "success": False,
                "message": "视频生成完成，但后端未找到最终视频文件",
                "video_url": "",
                "video_path": "",
            }
        video_url = ""
        if video_path and request is not None:
            # 若是本地路径，转为前端可访问 URL
            if not video_path.startswith("http"):
                video_url = _to_outputs_url(video_path, request)
            else:
                video_url = video_path
        elif video_path:
            video_url = video_path

        return {
            "success": True,
            "video_url": video_url,
            "video_path": video_path,
        }
