# -*- coding: utf-8 -*-
"""
paper2video 工作流封装（两步流程）。

- run_paper2video_generate_subtitle_wf_api：只跑“生成字幕/脚本”阶段，
  输入 result_path、PDF 路径、可选头像/语音路径及配置，
  输出 result_path + script_pages + state_snapshot（供第二步复用除 script_pages 外的 state）。
- run_paper2video_generate_video_wf_api：只跑“生成视频”阶段，
  输入 result_path、用户编辑后的 script_pages、可选的 state_snapshot（第一步返回），
  若有 state_snapshot 则复用第一步的 state（仅 script_pages 以用户输入为准），否则仅用 result_path + script_pages 构造 state。

工作流名称约定（需在 dataflow_agent.workflow 中注册）：
- paper2video（script_stage=True）：对应第一步 generate_subtitle
- paper2video（script_stage=False）：对应第二步 generate_video

若工作流尚未实现，当前会返回占位数据并打 log，便于联调前端。
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from fastapi_app.schemas import FeaturePaper2VideoRequest, FeaturePaper2VideoResponse

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
from dataflow_agent.utils import get_project_root
import os

log = get_logger(__name__)

# 第一步完成后需要带到第二步的 state 字段（不含 script_pages，由用户在第二步传入）
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


def _find_generated_video(base_dir: Path) -> str:
    """在工作流没有正确回填 video_path 时，尽量从输出目录回捞最终视频。"""
    preferred = [
        base_dir / "video.mp4",
        base_dir / "2_merge.mp4",
        base_dir / "1_merge.mp4",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return str(candidate)

    mp4_files = [
        p for p in base_dir.rglob("*.mp4")
        if p.is_file() and "talking_video" not in p.parts and "merge" not in p.parts
    ]
    if mp4_files:
        mp4_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(mp4_files[0])

    return ""


def _state_to_snapshot(state: Paper2VideoState | dict) -> dict:
    """将第一步的 state 序列化为可 JSON 的 dict，不含 script_pages。LangGraph 返回 dict，故需兼容。"""
    if isinstance(state, dict):
        req = state.get("request")
        snapshot = {"request": asdict(req) if req is not None and hasattr(req, "__dataclass_fields__") else (req if isinstance(req, dict) else {})}
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = state.get(key)
    else:
        req = state.request
        snapshot = {"request": asdict(req)}
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = getattr(state, key, None)
    return snapshot


def _state_from_snapshot(snapshot: dict, script_pages: List[dict]) -> Paper2VideoState:
    """从 snapshot 恢复 state，并设置用户传入的 script_pages；request.script_stage 设为 False。"""
    req_dict = snapshot.get("request") or {}
    req_dict = dict(req_dict)
    req_dict["script_stage"] = False
    request = Paper2VideoRequest(**req_dict)
    state = Paper2VideoState(request=request, messages=[])
    for key in _STATE_SNAPSHOT_KEYS:
        if key in snapshot:
            setattr(state, key, snapshot[key])
    state.script_pages = script_pages
    return state


def _ensure_result_path(result_path: Optional[Path] = None, email: str = "") -> Path:
    """若 result_path 为空，则按 email 与时间戳创建 outputs/.../paper2video/<ts>/。"""
    import time
    project_root = get_project_root()
    ts = int(time.time())
    code = email or "default"
    base_dir = (project_root / "outputs" / code / "paper2video" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return result_path if result_path is not None else base_dir


async def run_paper2video_generate_subtitle_wf_api(
    result_path: Path,
    paper_pdf_path: str,
    ref_img_path: str = "",
    ref_audio_path: str = "",
    ref_text: str = "",
    chat_api_url: str = "",
    api_key: str = "",
    model: str = "gpt-4o",
    tts_model: str = "cosyvoice-v3-flash",
    tts_voice_name: str = "",
    language: str = "en",
    email: str = "",
    talking_model: str = "liveportrait",
) -> dict[str, Any]:
    """
    执行“生成字幕/脚本”工作流：从 PDF（及可选头像/语音）解析出每页图片与语音脚本。
    返回 script_pages 与 state_snapshot，供第二步 generate_video 复用除 script_pages 外的 state。

    参数：
    - result_path: 本次任务输出根目录（由 Service 创建并传入）
    - paper_pdf_path: PDF 绝对路径
    - ref_img_path: 数字人头像路径，空表示未上传
    - ref_audio_path: 参考语音路径，空表示未上传
    - ref_text: 参考语音对应文本，空时工作流内部可选用 Whisper 等转录
    - chat_api_url / api_key / model: 脚本生成用 LLM 配置
    - tts_model: 语音模型名（如 cosyvoice-v3-flash）
    - language: zh / en
    - email: 用户标识，用于日志与可选路径

    返回：
    - success: bool
    - result_path: str，与传入的 result_path 一致
    - script_pages: list[dict]，每项含 page_num, image_url 或 image_path, script_text
    - state_snapshot: dict，第一步 state 的序列化（不含 script_pages），第二步请求时原样回传以复用 state
    """
    result_root = Path(result_path).resolve() if result_path else _ensure_result_path(None, email)
    log.info("[wa_paper2video] run_paper2video_generate_subtitle_wf_api: result_path=%s, paper_pdf_path=%s", result_root, paper_pdf_path)

    normalized_talking_model = "liveportrait"
    if (talking_model or "").strip().lower() not in {"", "liveportrait"}:
        log.info(
            "[wa_paper2video] force talking_model=%s -> %s",
            talking_model,
            normalized_talking_model,
        )

    req = Paper2VideoRequest(
        language=language,
        chat_api_url=chat_api_url or "",
        api_key=api_key or "",
        chat_api_key=api_key or "",
        model=model,
        paper_pdf_path=paper_pdf_path,
        ref_audio_path=ref_audio_path or "",
        ref_text=ref_text or "",
        ref_img_path=ref_img_path or "",
        tts_model=tts_model or "",
        tts_voice_name=(tts_voice_name or "").strip(),
        script_stage=True,
        talking_model=normalized_talking_model,
    )
    state = Paper2VideoState(request=req, messages=[])
    setattr(state, "result_path", str(result_root))

    workflow_name = "paper2video"
    try:
        from dataflow_agent.workflow import run_workflow
        log.info("[wa_paper2video] running workflow %s, the state is %s", workflow_name, state)
        final_state: Paper2VideoState= await run_workflow(workflow_name, state)
    except Exception as e:
        log.warning("[wa_paper2video] workflow %s not found or failed: %s; returning placeholder script_pages", workflow_name, e)
        return {
            "success": True,
            "result_path": str(result_root),
            "script_pages": [],
            "state_snapshot": _state_to_snapshot(state),
        }

    script_pages = getattr(final_state, "script_pages", None) or (final_state.get("script_pages") if isinstance(final_state, dict) else [])
    if not isinstance(script_pages, list):
        script_pages = []
        log.warning("[wa_paper2video] script_pages is not a list, returning empty list")
    result_path_str = getattr(final_state, "result_path", None) or (final_state.get("result_path") if isinstance(final_state, dict) else None) or str(result_root)
    snapshot = _state_to_snapshot(final_state)
    log.info("[wa_paper2video] generate_subtitle done, script_pages count=%s", len(script_pages))
    return {
        "success": True,
        "result_path": result_path_str,
        "script_pages": script_pages,
        "state_snapshot": snapshot,
    }


async def run_paper2video_generate_video_wf_api(
    result_path: str,
    script_pages: List[dict],
    state_snapshot: Optional[dict] = None,
) -> dict[str, Any]:
    """
    执行“生成视频”工作流：根据 result_path 与用户编辑后的 script_pages 合成最终视频。
    若传入第一步返回的 state_snapshot，则复用第一步的 state（仅 script_pages 以用户输入为准）。

    参数：
    - result_path: 第一步返回的任务输出根目录（绝对或相对项目根）
    - script_pages: 列表，每项含 page_num, script_text（及可选 image_url/image_path）
    - state_snapshot: 第一步 generate_subtitle 返回的 state_snapshot，可选；有则复用除 script_pages 外的 state
    - email: 用户标识，用于日志

    返回：
    - success: bool
    - video_path: str，后端本地路径（Service 层会据此转 video_url）
    """
    base_dir = Path(result_path).expanduser().resolve()
    project_root = get_project_root()
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()
    log.info("[wa_paper2video] run_paper2video_generate_video_wf_api: result_path=%s, script_pages count=%s, has_snapshot=%s", base_dir, len(script_pages), state_snapshot is not None)

    if state_snapshot:
        state = _state_from_snapshot(state_snapshot, script_pages)
        # 确保 result_path 以本次请求为准
        setattr(state, "result_path", str(base_dir))
        setattr(state, "script_pages", script_pages)
    else:
        req = Paper2VideoRequest(language="en", chat_api_url="", api_key="", chat_api_key="", script_stage=False)
        state = Paper2VideoState(request=req, messages=[])
        setattr(state, "result_path", str(base_dir))
        setattr(state, "script_pages", script_pages)
        log.warning("[wa_paper2video] no state_snapshot, script_pages injected (count=%s)", len(script_pages))
    log.info("[wa_paper2video] state.result_path=%s, script_pages injected (count=%s)", getattr(state, "result_path", ""), len(script_pages))

    workflow_name = "paper2video"
    try:
        from dataflow_agent.workflow import run_workflow
        final_state = await run_workflow(workflow_name, state)
        log.info("[wa_paper2video] workflow %s finished", workflow_name)
    except Exception as e:
        log.exception("[wa_paper2video] workflow %s failed during generate_video: %s", workflow_name, e)
        return {
            "success": False,
            "message": f"视频生成失败：{str(e)}",
            "video_path": "",
        }

    video_path = getattr(final_state, "video_path", None)
    if isinstance(final_state, dict):
        video_path = video_path or final_state.get("video_path")
    video_path = video_path or _find_generated_video(base_dir)
    if not video_path:
        log.warning("[wa_paper2video] generate_video finished but no video_path was produced under %s", base_dir)
        return {
            "success": False,
            "message": "视频生成流程已结束，但未找到最终视频文件",
            "video_path": "",
        }
    log.info("[wa_paper2video] generate_video done, video_path=%s", video_path)
    return {
        "success": True,
        "video_path": str(video_path) if video_path else "",
    }




# ------------------- paper2video 工作流封装 -------------------


async def run_paper_to_video_api(
    req: "FeaturePaper2VideoRequest",
) -> "FeaturePaper2VideoResponse":
    """
    旧版：单次请求跑完整 paper2video 工作流（与 gradio 行为对齐）。
    新前端请使用 generate-subtitle + generate-video 两步接口。
    """
    from fastapi_app.schemas import FeaturePaper2VideoRequest, FeaturePaper2VideoResponse

    log.info("[wa_paper2video] run_paper_to_video_api: req=%s", req)

    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    else:
        req.api_key = os.getenv("DF_API_KEY", "sk-dummy")

    from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
    paper_req = Paper2VideoRequest(
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model,
        paper_pdf_path=req.pdf_path,
        user_imgs_path=req.img_path,
        language=req.language or "en",
    )
    state = Paper2VideoState(request=paper_req, messages=[])

    try:
        from dataflow_agent.workflow.wf_paper2video import create_paper2video_graph
        graph = create_paper2video_graph().build()
        final_state = await graph.ainvoke(state)
    except Exception as e:
        log.warning("[wa_paper2video] legacy run_paper_to_video_api workflow failed: %s", e)
        return FeaturePaper2VideoResponse(success=False, ppt_path="")

    try:
        ppt_path = final_state.get("ppt_path") if isinstance(final_state, dict) else getattr(final_state, "ppt_path", "")
        if isinstance(ppt_path, list):
            ppt_path = ppt_path[0] if ppt_path else ""
        ppt_path = str(ppt_path) if ppt_path else ""
    except Exception:
        ppt_path = ""
    return FeaturePaper2VideoResponse(success=True, ppt_path=ppt_path or "")
