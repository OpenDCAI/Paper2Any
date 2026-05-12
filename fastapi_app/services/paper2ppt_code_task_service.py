from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import HTTPException, Request

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from fastapi_app.schemas import AssembleEditablePPTRequest, EditablePPTGenerationRequest, PatchSlideRequest
from fastapi_app.services.editable_ppt_service import EditablePPTService
from fastapi_app.utils import _to_outputs_url

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
TASK_ROOT = (PROJECT_ROOT / "outputs" / ".tasks" / "paper2ppt_code").resolve()
_ACTIVE_TASKS: set[asyncio.Task[Any]] = set()
_SUBMISSION_WINDOW_SECONDS = 20
_TASK_SUBMISSION_LOCK = Lock()
_TASK_HEARTBEAT_INTERVAL_SECONDS = 15
_TASK_STALE_TIMEOUT_SECONDS = 180

_STAGE_ORDER = ["planning", "final_ir", "slide_rendering", "assembling"]
_STAGE_TOTAL = 4

_STAGE_INDEX: dict[str, int] = {stage: idx for idx, stage in enumerate(_STAGE_ORDER)}


def _initial_progress() -> Dict[str, Any]:
    return {
        "stage": "planning",
        "stageIndex": 0,
        "stageTotal": _STAGE_TOTAL,
        "slideDone": 0,
        "slideTotal": 0,
        "message": "Starting…",
        "plannedIrPath": "",
        "materialsManifestPath": "",
        "materialResolutionPath": "",
        "finalIrPath": "",
        "slideArtifacts": [],
    }


class Paper2PPTCodeTaskService:
    """File-backed async task service for paper2ppt code mode."""

    def __init__(self, service: EditablePPTService | None = None) -> None:
        self.service = service or EditablePPTService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_generate_slides_task(
        self,
        req: EditablePPTGenerationRequest,
        request: Request | None = None,
    ) -> Dict[str, Any]:
        submission_key = self._resolve_generate_submission_key(req, request)

        with _TASK_SUBMISSION_LOCK:
            if submission_key:
                existing_record = self._find_recent_submission(submission_key)
                if existing_record is not None:
                    return self._serialize_record(existing_record, request)

            task_id = uuid.uuid4().hex
            payload = req.model_dump()

            record = {
                "task_id": task_id,
                "task_type": "generate",
                "status": "queued",
                "message": "Slide generation queued",
                "error": None,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "request": payload,
                "progress": _initial_progress(),
                "result": None,
            }
            self._write_record(task_id, record)
            if submission_key:
                self._write_submission(submission_key, task_id)

        task = asyncio.create_task(self._run_generate_task(task_id))
        _ACTIVE_TASKS.add(task)
        task.add_done_callback(_ACTIVE_TASKS.discard)

        return self._serialize_record(record, request)

    async def submit_assemble_task(
        self,
        req: AssembleEditablePPTRequest,
        request: Request | None = None,
    ) -> Dict[str, Any]:
        submission_key = self._resolve_assemble_submission_key(req, request)

        with _TASK_SUBMISSION_LOCK:
            if submission_key:
                existing_record = self._find_recent_submission(submission_key)
                if existing_record is not None:
                    return self._serialize_record(existing_record, request)

            task_id = uuid.uuid4().hex
            payload = req.model_dump()

            record = {
                "task_id": task_id,
                "task_type": "assemble",
                "status": "queued",
                "message": "Assembly queued",
                "error": None,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "request": payload,
                "progress": {
                    "stage": "assembling",
                    "stageIndex": _STAGE_INDEX["assembling"],
                    "stageTotal": _STAGE_TOTAL,
                    "slideDone": 0,
                    "slideTotal": 0,
                    "message": "Assembly queued",
                    "slideArtifacts": [],
                },
                "result": None,
            }
            self._write_record(task_id, record)
            if submission_key:
                self._write_submission(submission_key, task_id)

        task = asyncio.create_task(self._run_assemble_task(task_id))
        _ACTIVE_TASKS.add(task)
        task.add_done_callback(_ACTIVE_TASKS.discard)

        return self._serialize_record(record, request)

    def get_task(self, task_id: str, request: Request | None = None) -> Dict[str, Any]:
        record = self._refresh_record_state(task_id)
        return self._serialize_record(record, request)

    async def submit_patch_slide_task(
        self,
        req: PatchSlideRequest,
        request: Request | None = None,
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        payload = req.model_dump()
        record = {
            "task_id": task_id,
            "task_type": "patch_slide",
            "status": "queued",
            "message": "Patch queued",
            "error": None,
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "request": payload,
            "progress": {
                "stage": "patch_analyzing",
                "slideIndex": req.slide_index,
                "message": "Patch queued",
                "slideArtifact": None,
            },
            "result": None,
        }
        self._write_record(task_id, record)

        task = asyncio.create_task(self._run_patch_slide_task(task_id))
        _ACTIVE_TASKS.add(task)
        task.add_done_callback(_ACTIVE_TASKS.discard)

        return self._serialize_record(record, request)

    # ------------------------------------------------------------------
    # Background task runners
    # ------------------------------------------------------------------

    async def _run_generate_task(self, task_id: str) -> None:
        record = self._read_record(task_id)
        payload = record.get("request") or {}
        req = EditablePPTGenerationRequest(**payload)
        heartbeat_task = asyncio.create_task(self._heartbeat_task(task_id))

        def on_progress(event: str, params: dict) -> None:
            try:
                current = self._read_record(task_id)
                progress: Dict[str, Any] = dict(current.get("progress") or _initial_progress())

                if event == "planning_done":
                    progress["stage"] = "final_ir"
                    progress["stageIndex"] = _STAGE_INDEX["final_ir"]
                    progress["message"] = "Planning done, building final IR…"
                    progress["plannedIrPath"] = str(params.get("planned_ir_path") or "")
                    progress["materialsManifestPath"] = str(params.get("materials_manifest_path") or "")
                    progress["materialResolutionPath"] = str(params.get("material_resolution_path") or "")

                elif event == "final_ir_done":
                    progress["stage"] = "slide_rendering"
                    progress["stageIndex"] = _STAGE_INDEX["slide_rendering"]
                    progress["message"] = "Final IR ready, rendering slides…"
                    progress["finalIrPath"] = str(params.get("final_ir_path") or "")

                elif event == "slide_rendered":
                    # runner emits: index (0-based), total, slide_id, title, pptx_path, preview_png_path
                    slide_total = int(params.get("total") or 0)
                    slide_index = int(params.get("index") or 0)
                    slide_done = slide_index + 1
                    progress["stage"] = "slide_rendering"
                    progress["stageIndex"] = _STAGE_INDEX["slide_rendering"]
                    progress["slideTotal"] = slide_total
                    progress["slideDone"] = slide_done
                    progress["message"] = f"Rendered slide {slide_done}/{slide_total}"
                    artifact = {
                        "index": slide_index,
                        "slide_id": params.get("slide_id", ""),
                        "title": params.get("title", ""),
                        "pptx_path": params.get("pptx_path", ""),
                        "preview_png_path": params.get("preview_png_path", ""),
                        "status": "rendered",
                    }
                    existing: list = list(progress.get("slideArtifacts") or [])
                    existing.append(artifact)
                    progress["slideArtifacts"] = existing

                elif event == "rendering_done":
                    # runner emits full slide_artifacts list (with updated preview paths)
                    raw_artifacts = params.get("slide_artifacts")
                    if isinstance(raw_artifacts, list):
                        progress["slideArtifacts"] = raw_artifacts
                    progress["message"] = "All slides rendered"

                current["progress"] = progress
                self._write_record(task_id, current)
            except Exception:  # noqa: BLE001
                log.warning("[paper2ppt-code-task] progress update failed for task %s", task_id)

        try:
            self._update_record(
                task_id,
                status="running",
                message="Generating slides…",
                error=None,
            )

            result = await self.service.generate_editable_ppt_slides(
                req=req,
                request=None,
                # Call directly from worker thread — on_progress only does file I/O (thread-safe).
                # Using call_soon_threadsafe would defer writes until the event loop polls,
                # which causes progress updates to batch instead of flushing per-slide.
                progress_callback=on_progress,
            )

            # Mark progress as done
            final_record = self._read_record(task_id)
            progress = dict(final_record.get("progress") or _initial_progress())
            progress["stage"] = "done"
            progress["message"] = "Generation complete"
            final_record["progress"] = progress
            final_record["status"] = "done"
            final_record["message"] = "Task completed"
            final_record["error"] = None
            final_record["result"] = result
            self._write_record(task_id, final_record)

        except HTTPException as exc:
            message = str(exc.detail)
            log.warning("[paper2ppt-code-task] task %s failed: %s", task_id, message)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result=None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or exc.__class__.__name__
            log.exception("[paper2ppt-code-task] task %s failed", task_id)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result={
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    async def _run_assemble_task(self, task_id: str) -> None:
        record = self._read_record(task_id)
        payload = record.get("request") or {}
        req = AssembleEditablePPTRequest(**payload)
        heartbeat_task = asyncio.create_task(self._heartbeat_task(task_id))

        try:
            self._update_record(
                task_id,
                status="running",
                message="Assembling PPTX…",
                error=None,
            )

            result = await self.service.assemble_editable_ppt(
                req=req,
                request=None,
            )

            self._update_record(
                task_id,
                status="done",
                message="Assembly completed",
                error=None,
                result=result,
            )
        except HTTPException as exc:
            message = str(exc.detail)
            log.warning("[paper2ppt-code-task] assemble task %s failed: %s", task_id, message)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result=None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or exc.__class__.__name__
            log.exception("[paper2ppt-code-task] assemble task %s failed", task_id)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result={
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    async def _heartbeat_task(self, task_id: str) -> None:
        while True:
            await asyncio.sleep(_TASK_HEARTBEAT_INTERVAL_SECONDS)
            try:
                record = self._read_record(task_id)
                record["updated_at"] = self._now_iso()
                self._write_record(task_id, record)
            except HTTPException:
                return
            except Exception:
                log.warning("[paper2ppt-code-task] heartbeat update failed for task %s", task_id)
                return

    async def _run_patch_slide_task(self, task_id: str) -> None:
        record = self._read_record(task_id)
        payload = record.get("request") or {}
        req = PatchSlideRequest(**payload)
        heartbeat_task = asyncio.create_task(self._heartbeat_task(task_id))

        def on_progress(event: str, params: dict) -> None:
            try:
                current = self._read_record(task_id)
                progress: Dict[str, Any] = dict(current.get("progress") or {})
                progress["message"] = {
                    "patch_analyzing": "分析修改意见…",
                    "patch_ir_done": "SlideIR 已更新，准备生成代码…",
                    "patch_codegen": "重新生成幻灯片代码…",
                    "patch_rendering": "重新渲染幻灯片…",
                    "patch_done": "修改完成",
                }.get(event, event)
                progress["stage"] = event
                if event == "patch_done":
                    artifact = {
                        "index": int(params.get("index") or req.slide_index),
                        "slide_id": params.get("slide_id", ""),
                        "title": params.get("title", ""),
                        "pptx_path": params.get("pptx_path", ""),
                        "preview_png_path": params.get("preview_png_path", ""),
                        "status": "rendered",
                    }
                    progress["slideArtifact"] = artifact
                current["progress"] = progress
                self._write_record(task_id, current)
            except Exception:
                pass

        try:
            self._update_record(task_id, status="running", message="Patching slide…", error=None)

            from fastapi_app.services.managed_api_service import resolve_image_generation_credentials, resolve_llm_credentials
            from fastapi_app.services.paper2ppt_service import Paper2PPTService
            from vendor.presentagent_runtime.runner import repatch_single_slide

            credential_scope = Paper2PPTService().resolve_credential_scope(req.credential_scope)
            resolved_api_url, resolved_api_key = resolve_llm_credentials(
                req.chat_api_url, req.api_key, scope=credential_scope,
            )
            resolved_image_url, resolved_image_key = resolve_image_generation_credentials(
                req.image_api_url, req.image_api_key, scope=credential_scope,
            )
            result_path = Paper2PPTService().resolve_result_path(req.result_path)

            sa = await asyncio.to_thread(
                repatch_single_slide,
                str(result_path),
                req.slide_index,
                req.feedback,
                req.feedback_type,
                api_url=resolved_api_url or "",
                api_key=resolved_api_key or "",
                model=req.model or "",
                image_api_url=resolved_image_url or "",
                image_api_key=resolved_image_key or "",
                image_model=req.image_model or "",
                progress_callback=on_progress,
            )

            final_record = self._read_record(task_id)
            progress = dict(final_record.get("progress") or {})
            progress["stage"] = "done"
            progress["message"] = "修改完成"
            artifact_dict = sa.model_dump()
            progress["slideArtifact"] = artifact_dict
            final_record["progress"] = progress
            final_record["status"] = "done"
            final_record["message"] = "Patch completed"
            final_record["error"] = None
            final_record["result"] = {
                "success": True,
                "slide_index": req.slide_index,
                "slide_artifact": artifact_dict,
            }
            self._write_record(task_id, final_record)

        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            log.exception("[paper2ppt-code-task] patch task %s failed", task_id)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result={"traceback": traceback.format_exc(limit=20)},
            )
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize_record(
        self,
        record: Dict[str, Any],
        request: Request | None = None,
    ) -> Dict[str, Any]:
        status = str(record.get("status") or "queued")
        result = record.get("result") if status == "done" else None

        progress = record.get("progress")
        if progress and isinstance(progress, dict):
            progress = self._url_ify_progress(progress, request)

        if result and isinstance(result, dict):
            result = self._url_ify_result(result, request)

        return {
            "success": True,
            "task_id": record["task_id"],
            "task_type": record.get("task_type", "generate"),
            "status": status,
            "message": record.get("message", ""),
            "error": record.get("error"),
            "progress": progress,
            "result": result,
        }

    def _url_ify_progress(self, progress: Dict[str, Any], request: Request | None) -> Dict[str, Any]:
        result = dict(progress)
        raw_artifacts = result.get("slideArtifacts") or []
        if isinstance(raw_artifacts, list) and raw_artifacts:
            url_artifacts = []
            for sa in raw_artifacts:
                if not isinstance(sa, dict):
                    url_artifacts.append(sa)
                    continue
                pptx_p = str(sa.get("pptx_path") or "")
                preview_p = str(sa.get("preview_png_path") or "")
                url_artifacts.append({
                    **sa,
                    "pptx_path": _to_outputs_url(pptx_p, request) if pptx_p else "",
                    "preview_png_path": _to_outputs_url(preview_p, request) if preview_p else "",
                })
            result["slideArtifacts"] = url_artifacts
        return result

    def _url_ify_result(self, result: Dict[str, Any], request: Request | None) -> Dict[str, Any]:
        out = dict(result)
        # URL-convert per-slide artifacts stored as absolute paths (list form)
        raw_artifacts = out.get("slide_artifacts") or []
        if isinstance(raw_artifacts, list) and raw_artifacts:
            url_artifacts = []
            for sa in raw_artifacts:
                if not isinstance(sa, dict):
                    url_artifacts.append(sa)
                    continue
                pptx_p = str(sa.get("pptx_path") or "")
                preview_p = str(sa.get("preview_png_path") or "")
                url_artifacts.append({
                    **sa,
                    "pptx_path": _to_outputs_url(pptx_p, request) if pptx_p else "",
                    "preview_png_path": _to_outputs_url(preview_p, request) if preview_p else "",
                })
            out["slide_artifacts"] = url_artifacts
        # URL-convert single slide_artifact (patch task result)
        single = out.get("slide_artifact")
        if isinstance(single, dict):
            pptx_p = str(single.get("pptx_path") or "")
            preview_p = str(single.get("preview_png_path") or "")
            out["slide_artifact"] = {
                **single,
                "pptx_path": _to_outputs_url(pptx_p, request) if pptx_p else "",
                "preview_png_path": _to_outputs_url(preview_p, request) if preview_p else "",
            }
        # URL-convert top-level path fields
        for field in ("ppt_pptx_path", "ppt_pdf_path", "ir_path", "render_log_path",
                      "planned_ir_path", "final_ir_path", "materials_manifest_path",
                      "material_resolution_path"):
            val = str(out.get(field) or "")
            if val and not val.startswith(("/outputs/", "http://", "https://")):
                out[field] = _to_outputs_url(val, request)
        return out

    # ------------------------------------------------------------------
    # Submission dedup
    # ------------------------------------------------------------------

    def _resolve_generate_submission_key(
        self,
        req: EditablePPTGenerationRequest,
        request: Request | None,
    ) -> str | None:
        request_state = getattr(request, "state", None)
        existing = getattr(request_state, "workflow_submission_key", None)
        if existing:
            return str(existing).strip() or None

        payload = {
            "path": "/api/v1/paper2ppt/code/generate-task",
            "result_path": str(req.result_path or "").strip(),
            "pagecontent": str(req.pagecontent or "").strip(),
            "model": str(req.model or "").strip(),
            "language": str(req.language or "").strip(),
            "style": str(req.style or "").strip(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        submission_key = hashlib.sha256(encoded).hexdigest()
        if request_state is not None:
            request_state.workflow_submission_key = submission_key
        return submission_key

    def _resolve_assemble_submission_key(
        self,
        req: AssembleEditablePPTRequest,
        request: Request | None,
    ) -> str | None:
        request_state = getattr(request, "state", None)
        existing = getattr(request_state, "workflow_submission_key", None)
        if existing:
            return str(existing).strip() or None

        payload = {
            "path": "/api/v1/paper2ppt/code/assemble-task",
            "result_path": str(req.result_path or "").strip(),
            "include_pdf_preview": bool(req.include_pdf_preview),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        submission_key = hashlib.sha256(encoded).hexdigest()
        if request_state is not None:
            request_state.workflow_submission_key = submission_key
        return submission_key

    def _find_recent_submission(self, submission_key: str) -> Dict[str, Any] | None:
        submission_file = self._submission_file(submission_key)
        if not submission_file.exists():
            return None
        try:
            payload = json.loads(submission_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

        created_at = float(payload.get("created_at") or 0)
        if time.time() - created_at > _SUBMISSION_WINDOW_SECONDS:
            return None

        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            return None

        try:
            record = self._refresh_record_state(task_id)
        except HTTPException:
            return None
        if str(record.get("status") or "").lower() == "failed":
            return None
        return record

    def _write_submission(self, submission_key: str, task_id: str) -> None:
        submission_file = self._submission_file(submission_key)
        submission_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "created_at": time.time(),
        }
        tmp_file = submission_file.with_suffix(".tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_file.replace(submission_file)

    # ------------------------------------------------------------------
    # Stale detection
    # ------------------------------------------------------------------

    def _refresh_record_state(self, task_id: str) -> Dict[str, Any]:
        record = self._read_record(task_id)
        status = str(record.get("status") or "").strip().lower()

        if status not in {"queued", "running"}:
            return record

        age_seconds = self._record_age_seconds(record)
        if age_seconds is None or age_seconds < _TASK_STALE_TIMEOUT_SECONDS:
            return record

        message = "任务已中断：后端 worker 可能已重启，请重新生成"
        return self._update_record(
            task_id,
            status="failed",
            message=message,
            error=message,
        )

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return TASK_ROOT / task_id

    def _task_file(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "task.json"

    def _submission_file(self, submission_key: str) -> Path:
        return TASK_ROOT / ".submissions" / f"{submission_key}.json"

    def _read_record(self, task_id: str) -> Dict[str, Any]:
        task_file = self._task_file(task_id)
        if not task_file.exists():
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        try:
            return json.loads(task_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"failed to load task: {task_id}") from exc

    def _write_record(self, task_id: str, record: Dict[str, Any]) -> None:
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        record["updated_at"] = self._now_iso()

        task_file = self._task_file(task_id)
        tmp_file = task_file.with_suffix(".tmp")
        tmp_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_file.replace(task_file)

    def _update_record(self, task_id: str, **updates: Any) -> Dict[str, Any]:
        record = self._read_record(task_id)
        record.update(updates)
        self._write_record(task_id, record)
        return record

    def _record_age_seconds(self, record: Dict[str, Any]) -> float | None:
        updated_at = str(record.get("updated_at") or "").strip()
        if not updated_at:
            return None
        try:
            updated = datetime.fromisoformat(updated_at)
        except ValueError:
            return None
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
