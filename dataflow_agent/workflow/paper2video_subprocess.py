# dataflow_agent/workflow/paper2video_subprocess.py
"""
paper2video 子进程调用：在 p2v 环境中执行工作流，主应用（online 环境）通过 subprocess 调用。
子进程的 stdout/stderr 统一转发到主应用日志。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState

# 项目根：本文件在 dataflow_agent/workflow/ 下
_workflow_dir = Path(__file__).resolve().parent
_project_root = _workflow_dir.parent.parent

# 与 wa_paper2video 一致的 snapshot 字段
_P2V_SNAPSHOT_KEYS = [
    "result_path", "ppt_path", "slide_timesteps_path", "slide_img_dir",
    "subtitle_and_cursor", "subtitle_and_cursor_path",
    "speech_save_dir", "cursor_save_path", "talking_video_save_dir",
]


def _p2v_state_to_snapshot(state: Paper2VideoState | dict) -> dict:
    """将 state 序列化为可传给 worker 的 snapshot dict（不含 script_pages）。"""
    if isinstance(state, dict):
        req = state.get("request")
        snapshot = {
            "request": asdict(req) if req is not None and hasattr(req, "__dataclass_fields__") else (req if isinstance(req, dict) else {}),
        }
        for key in _P2V_SNAPSHOT_KEYS:
            snapshot[key] = state.get(key)
    else:
        snapshot = {"request": asdict(state.request)}
        for key in _P2V_SNAPSHOT_KEYS:
            snapshot[key] = getattr(state, key, None)
    return snapshot


def _p2v_build_worker_input(state, mode: str) -> dict:
    """根据 state 和 mode 构建 worker 的 input JSON 内容。"""
    if mode == "generate_subtitle":
        req = state.request if hasattr(state, "request") else state.get("request")
        return {
            "result_path": getattr(state, "result_path", "") or state.get("result_path", ""),
            "paper_pdf_path": getattr(req, "paper_pdf_path", "") or req.get("paper_pdf_path", ""),
            "ref_img_path": getattr(req, "ref_img_path", "") or req.get("ref_img_path", ""),
            "ref_audio_path": getattr(req, "ref_audio_path", "") or req.get("ref_audio_path", ""),
            "ref_text": getattr(req, "ref_text", "") or req.get("ref_text", ""),
            "chat_api_url": getattr(req, "chat_api_url", "") or req.get("chat_api_url", ""),
            "api_key": getattr(req, "api_key", "") or req.get("api_key", ""),
            "model": getattr(req, "model", "gpt-4o") or req.get("model", "gpt-4o"),
            "tts_model": getattr(req, "tts_model", "") or req.get("tts_model", ""),
            "tts_voice_name": getattr(req, "tts_voice_name", "") or req.get("tts_voice_name", ""),
            "language": getattr(req, "language", "en") or req.get("language", "en"),
        }
    else:
        return {
            "result_path": getattr(state, "result_path", "") or state.get("result_path", ""),
            "script_pages": getattr(state, "script_pages", []) or state.get("script_pages", []),
            "state_snapshot": _p2v_state_to_snapshot(state),
        }


async def run_paper2video_via_subprocess(name: str, state) -> dict:
    """
    在 p2v 子进程中执行 paper2video 工作流，子进程的 stdout/stderr 统一转发到主应用日志。
    环境变量：PAPER2VIDEO_PYTHON 可指定 worker 的 Python；未配置时优先复用当前解释器。
    """
    log = get_logger(__name__)
    worker_script = _project_root / "script" / "paper2video_worker.py"
    if not worker_script.exists():
        log.error("[paper2video] worker script not found: %s", worker_script)
        raise FileNotFoundError(f"paper2video worker script not found: {worker_script}")

    script_stage = getattr(state.request, "script_stage", True) if hasattr(state, "request") else state.get("request", {}).get("script_stage", True)
    mode = "generate_subtitle" if script_stage else "generate_video"
    in_dict = _p2v_build_worker_input(state, mode)

    configured_python = (os.getenv("PAPER2VIDEO_PYTHON", "") or "").strip()
    python_candidates = []
    if configured_python:
        python_candidates.append(configured_python)
    python_candidates.extend([
        sys.executable,
        "python3",
        "python",
    ])

    python_bin = ""
    for candidate in python_candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                python_bin = candidate
                break
            continue
        resolved = shutil.which(candidate)
        if resolved:
            python_bin = resolved
            break

    if not python_bin:
        raise FileNotFoundError(
            "No usable Python found for paper2video worker. "
            "Set PAPER2VIDEO_PYTHON to a valid interpreter path."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f_in:
        json.dump(in_dict, f_in, ensure_ascii=False, indent=2)
        in_path = f_in.name
    out_fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(out_fd)

    try:
        cmd = [python_bin, str(worker_script), "--mode", mode, "--input-json", in_path, "--output-json", out_path]
        log.info("[paper2video] running worker with python=%s: %s", python_bin, " ".join(cmd))

        # limit 调大，避免 worker 输出超长单行（如 MoviePy/FFmpeg 进度）时触发 asyncio 的 "Separator is not found, and chunk exceed the limit"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=2 * 1024 * 1024,  # 2MB，默认约 64KB
        )

        async def _forward_stream(stream, level="info"):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                getattr(log, level, log.info)("[p2v-worker] %s", text)

        # 子进程的 stdout 与 stderr 均按 INFO 转发（Paddle 等库常把进度/警告写到 stderr，并非 error）
        await asyncio.gather(
            _forward_stream(proc.stdout, "info"),
            _forward_stream(proc.stderr, "info"),
        )
        await proc.wait()

        try:
            with open(out_path, "r", encoding="utf-8") as f:
                out_data = json.load(f)
        except Exception:
            out_data = {}

        if proc.returncode != 0:
            err = out_data.get("error") or "unknown error"
            log.error("[paper2video] worker exited with code %s: %s", proc.returncode, err)
            raise RuntimeError(f"paper2video worker exited with code {proc.returncode}: {err}")

        if not out_data.get("success"):
            err = out_data.get("error", "unknown error")
            log.error("[paper2video] worker returned success=False: %s", err)
            raise RuntimeError(f"paper2video worker failed: {err}")

        if mode == "generate_subtitle":
            snapshot = out_data.get("state_snapshot", {})
            return {**snapshot, "script_pages": out_data.get("script_pages", []), "result_path": out_data.get("result_path", "")}
        else:
            return {"video_path": out_data.get("video_path", "")}
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass
        try:
            os.unlink(out_path)
        except OSError:
            pass
