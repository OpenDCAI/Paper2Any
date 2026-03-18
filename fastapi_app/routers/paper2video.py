# -*- coding: utf-8 -*-
"""
paper2video 路由：两步流程
- POST /paper2video/generate-subtitle：上传论文 + 可选头像/语音，返回 result_path + script_pages + state_snapshot
- POST /paper2video/generate-video：根据 result_path + 用户编辑后的 script_pages（可选 state_snapshot）生成视频，返回 video_url / video_path
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from fastapi_app.schemas import (
    ErrorResponse,
    GenerateSubtitleResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    FeaturePaper2VideoRequest,
    FeaturePaper2VideoResponse,
)
from fastapi_app.services.paper2video_service import Paper2VideoService
from fastapi_app.workflow_adapters import run_paper_to_video_api

from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# 与 main 中 prefix 配合：prefix="/api/v1" 时，完整路径为 /api/v1/paper2video/...
router = APIRouter(tags=["paper2video"])


def get_service() -> Paper2VideoService:
    """依赖注入：获取 Paper2VideoService 单例。"""
    return Paper2VideoService()


# ===================== 两步流程接口 =====================


@router.post(
    "/paper2video/generate-subtitle",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="上传论文并解析得到语音脚本（字幕）",
    description="接收 PDF、可选数字人头像、可选语音文件及配置，执行解析/字幕生成，返回 result_path、script_pages 与 state_snapshot（供第二步复用 state）。",
)
async def paper2video_generate_subtitle(
    request: Request,
    email: Optional[str] = Form(None),
    api_key: str = Form(...),
    chat_api_url: str = Form(...),
    model: str = Form("gpt-4o"),
    tts_model: str = Form("cosyvoice-v3-flash"),
    tts_voice_name: Optional[str] = Form(None),
    language: str = Form("en"),
    talking_model: str = Form("liveportrait"),
    file: Optional[UploadFile] = File(None),
    avatar: Optional[UploadFile] = File(None),
    avatar_preset: Optional[str] = Form(None),
    voice: Optional[UploadFile] = File(None),
    voice_preset: Optional[str] = Form(None),
    service: Paper2VideoService = Depends(get_service),
) -> Dict[str, Any]:
    """
    第一步：上传论文（必填 PDF）、可选头像/语音，后端落盘并调用工作流生成字幕/脚本。
    头像可为上传(avatar)或系统预设(avatar_preset)；语音可为上传(voice)或系统预设(voice_preset，对应 public/paper2video/sys_audio/{id}.wav)。
    返回 result_path、script_pages 与 state_snapshot。
    """
    log.info("[paper2video] generate-subtitle: request received")
    if not file:
        log.warning("[paper2video] generate-subtitle: missing file")
        raise HTTPException(status_code=400, detail="file is required (PDF)")
    effective_talking_model = "liveportrait"
    if (talking_model or "").strip().lower() not in {"", "liveportrait"}:
        log.info(
            "[paper2video] generate-subtitle: override talking_model=%s -> %s",
            talking_model,
            effective_talking_model,
        )
    data = await service.run_generate_subtitle(
        email=email,
        api_key=api_key,
        chat_api_url=chat_api_url,
        model=model,
        tts_model=tts_model,
        tts_voice_name=tts_voice_name or "",
        language=language,
        talking_model=effective_talking_model,
        file=file,
        avatar=avatar,
        avatar_preset=avatar_preset,
        voice=voice,
        voice_preset=voice_preset,
        request=request,
    )
    log.info("[paper2video] generate-subtitle: success, result_path=%s", data.get("result_path"))
    return data

@router.post(
    "/paper2video/generate-video",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="根据脚本生成最终视频",
    description="接收 result_path、用户编辑后的 script_pages，可选第一步返回的 state_snapshot，执行视频合成，返回 video_url 或 video_path。",
)
async def paper2video_generate_video(
    request: Request,
    result_path: str = Form(...),
    script_pages: str = Form(...),
    state_snapshot: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    service: Paper2VideoService = Depends(get_service),
) -> Dict[str, Any]:
    """
    第二步：根据第一步返回的 result_path、用户可能编辑过的 script_pages（JSON 字符串），
    可选传入第一步返回的 state_snapshot（JSON 字符串）以复用第一步 state，
    调用工作流生成最终视频，返回可访问的 video_url 或 video_path。
    """
    log.info("[paper2video] generate-video: request received, result_path=%s", result_path)
    data = await service.run_generate_video(
        result_path=result_path,
        script_pages_json=script_pages,
        state_snapshot_json=state_snapshot,
        email=email,
        request=request,
    )
    log.info("[paper2video] generate-video: success, video_url=%s", data.get("video_url") or data.get("video_path"))
    return data


# ===================== 兼容旧版单接口（可选保留） =====================


@router.post(
    "/paper2video",
    response_model=FeaturePaper2VideoResponse,
    summary="[兼容] 将 Paper 一次性转成汇报 video",
    description="旧版单接口，与 gradio 行为对齐；新前端请使用 generate-subtitle + generate-video 两步。",
)
async def paper2video_endpoint(body: FeaturePaper2VideoRequest) -> FeaturePaper2VideoResponse:
    """旧版：单次请求跑完整 paper2video 工作流。"""
    log.info("[paper2video] legacy /paper2video endpoint called")
    return await run_paper_to_video_api(body)
