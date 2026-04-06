#!/usr/bin/env python3
"""
Paper2Video 子进程 Worker：在 p2v 环境中执行 paper2video 工作流。

由主进程 (workflow/__init__.py) 通过 subprocess 调用，输入/输出通过 JSON 文件传递。
子进程的日志打到 stdout/stderr，由主进程统一采集并写入主应用日志。

用法:
  python script/paper2video_worker.py --mode generate_subtitle --input-json /path/to/in.json --output-json /path/to/out.json
  python script/paper2video_worker.py --mode generate_video   --input-json /path/to/in.json --output-json /path/to/out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import asdict

from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
from dataflow_agent.workflow import get_workflow
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

_STATE_SNAPSHOT_KEYS = [
    "result_path",
    "ppt_path",
    "slide_timesteps_path",
    "slide_img_dir",
    "subtitle_and_cursor",
    "subtitle_and_cursor_path",
    "speech_save_dir",
    "cursor_save_path",
    "talking_video_save_dir",
]


def _state_to_snapshot(state: Paper2VideoState | dict) -> dict:
    if isinstance(state, dict):
        req = state.get("request")
        snapshot = {
            "request": asdict(req) if req is not None and hasattr(req, "__dataclass_fields__") else (req if isinstance(req, dict) else {}),
        }
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = state.get(key)
    else:
        snapshot = {"request": asdict(state.request)}
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = getattr(state, key, None)
    return snapshot


def _request_field_names():
    names = set()
    for cls in Paper2VideoRequest.__mro__:
        if hasattr(cls, "__dataclass_fields__"):
            names.update(getattr(cls, "__dataclass_fields__"))
    return names


def _state_from_snapshot(snapshot: dict, script_pages: list) -> Paper2VideoState:
    req_dict = snapshot.get("request") or {}
    req_dict = dict(req_dict)
    req_dict["script_stage"] = False
    allowed = _request_field_names()
    request = Paper2VideoRequest(**{k: v for k, v in req_dict.items() if k in allowed})
    state = Paper2VideoState(request=request, messages=[])
    for key in _STATE_SNAPSHOT_KEYS:
        if key in snapshot:
            setattr(state, key, snapshot[key])
    state.script_pages = script_pages
    return state


def _run_generate_subtitle(in_data: dict) -> dict:
    result_root = in_data.get("result_path", "")
    req = Paper2VideoRequest(
        language=in_data.get("language", "en"),
        chat_api_url=in_data.get("chat_api_url", ""),
        api_key=in_data.get("api_key", ""),
        chat_api_key=in_data.get("chat_api_key", ""),
        model=in_data.get("model", "gpt-4o"),
        paper_pdf_path=in_data.get("paper_pdf_path", ""),
        ref_audio_path=in_data.get("ref_audio_path", ""),
        ref_text=in_data.get("ref_text", ""),
        ref_img_path=in_data.get("ref_img_path", ""),
        tts_model=in_data.get("tts_model", ""),
        tts_voice_name=in_data.get("tts_voice_name", ""),
        script_stage=True,
    )
    state = Paper2VideoState(request=req, messages=[])
    setattr(state, "result_path", result_root)

    async def _run():
        factory = get_workflow("paper2video")
        graph = factory().build()
        return await graph.ainvoke(state)

    log.info("[p2v-worker] generate_subtitle: running workflow (script_stage=True)")
    final_state = asyncio.run(_run())
    script_pages = getattr(final_state, "script_pages", None) or (final_state.get("script_pages") if isinstance(final_state, dict) else [])
    if not isinstance(script_pages, list):
        script_pages = []
    result_path_str = getattr(final_state, "result_path", None) or (final_state.get("result_path") if isinstance(final_state, dict) else None) or result_root
    snapshot = _state_to_snapshot(final_state)
    log.info("[p2v-worker] generate_subtitle done, script_pages count=%s", len(script_pages))
    return {
        "success": True,
        "result_path": result_path_str,
        "script_pages": script_pages,
        "state_snapshot": snapshot,
    }


def _run_generate_video(in_data: dict) -> dict:
    result_path = in_data.get("result_path", "")
    script_pages = in_data.get("script_pages", [])
    state_snapshot = in_data.get("state_snapshot")

    if state_snapshot:
        state = _state_from_snapshot(state_snapshot, script_pages)
        setattr(state, "result_path", result_path)
        setattr(state, "script_pages", script_pages)
    else:
        req = Paper2VideoRequest(language="en", chat_api_url="", api_key="", chat_api_key="", script_stage=False)
        state = Paper2VideoState(request=req, messages=[])
        setattr(state, "result_path", result_path)
        setattr(state, "script_pages", script_pages)

    async def _run():
        factory = get_workflow("paper2video")
        graph = factory().build()
        return await graph.ainvoke(state)

    log.info("[p2v-worker] generate_video: running workflow (script_stage=False)")
    final_state = asyncio.run(_run())
    video_path = getattr(final_state, "video_path", None)
    if isinstance(final_state, dict):
        video_path = video_path or final_state.get("video_path")
    video_path = video_path or ""
    log.info("[p2v-worker] generate_video done, video_path=%s", video_path)
    return {
        "success": True,
        "video_path": str(video_path) if video_path else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Paper2Video worker (run in p2v env)")
    parser.add_argument("--mode", required=True, choices=["generate_subtitle", "generate_video"])
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_json)
    if not in_path.exists():
        out_path.write_text(json.dumps({"success": False, "error": f"Input file not found: {in_path}"}, ensure_ascii=False, indent=2))
        return 1

    try:
        in_data = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.exception("[p2v-worker] failed to load input JSON")
        out_path.write_text(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2))
        return 1

    try:
        if args.mode == "generate_subtitle":
            result = _run_generate_subtitle(in_data)
        elif args.mode == "generate_video":
            result = _run_generate_video(in_data)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        log.exception("[p2v-worker] workflow failed")
        result = {"success": False, "error": str(e)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
