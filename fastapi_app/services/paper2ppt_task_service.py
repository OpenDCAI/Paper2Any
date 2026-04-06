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

from fastapi import HTTPException, Request, UploadFile

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from fastapi_app.schemas import PPTGenerationRequest
from fastapi_app.services.billing_service import BillingService
from fastapi_app.services.paper2ppt_service import Paper2PPTService

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
TASK_ROOT = (PROJECT_ROOT / "outputs" / ".tasks" / "paper2ppt").resolve()
_ACTIVE_TASKS: set[asyncio.Task[Any]] = set()
_SUBMISSION_WINDOW_SECONDS = 20
_TASK_SUBMISSION_LOCK = Lock()
_TASK_HEARTBEAT_INTERVAL_SECONDS = 15
_TASK_STALE_TIMEOUT_SECONDS = 90


def _is_truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(raw: Any) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _pagecontent_count(raw: Any) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _parse_page_index_list(raw: Any, *, max_count: int | None = None) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    max_index = max_count - 1 if max_count is not None and max_count > 0 else None
    indices: set[int] = set()
    for item in payload:
        try:
            value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        if max_index is not None and value > max_index:
            continue
        indices.add(value)
    return sorted(indices)


class Paper2PPTTaskService:
    """File-backed async tasks for long-running paper2ppt generation."""

    def __init__(self, service: Paper2PPTService | None = None) -> None:
        self.service = service or Paper2PPTService()

    async def submit_generate_task(
        self,
        req: PPTGenerationRequest,
        reference_img: UploadFile | None,
        request: Request | None = None,
    ) -> Dict[str, Any]:
        submission_key = self._resolve_submission_key(req, reference_img, request)

        base_dir = self.service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        if reference_img is not None:
            await self.service.cache_reference_image_for_result(str(base_dir), reference_img)

        with _TASK_SUBMISSION_LOCK:
            if submission_key:
                existing_record = self._find_recent_submission(submission_key)
                if existing_record is not None:
                    return self._serialize_record(existing_record, request)

            self._consume_submission_charge(req=req, request=request, submission_key=submission_key)

            task_id = uuid.uuid4().hex
            payload = req.model_dump()
            payload["result_path"] = str(base_dir)

            record = {
                "task_id": task_id,
                "task_type": self._task_type(req),
                "status": "queued",
                "message": self._queued_message(req),
                "error": None,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "request": payload,
                "result": None,
            }
            self._write_record(task_id, record)
            if submission_key:
                self._write_submission(submission_key, task_id)

        task = asyncio.create_task(self._run_generate_task(task_id))
        _ACTIVE_TASKS.add(task)
        task.add_done_callback(_ACTIVE_TASKS.discard)

        return self._serialize_record(record, request)

    def get_task(self, task_id: str, request: Request | None = None) -> Dict[str, Any]:
        record = self._refresh_record_state(task_id)
        return self._serialize_record(record, request)

    async def _run_generate_task(self, task_id: str) -> None:
        record = self._read_record(task_id)
        payload = record.get("request") or {}
        req = PPTGenerationRequest(**payload)
        heartbeat_task = asyncio.create_task(self._heartbeat_task(task_id, req))

        try:
            self._update_record(
                task_id,
                status="running",
                message=self._running_message(req),
                error=None,
            )

            result = await self.service.generate_ppt(
                req=req,
                reference_img=None,
                request=None,
            )

            self._update_record(
                task_id,
                status="done",
                message="Task completed",
                error=None,
                result=result,
            )
        except HTTPException as exc:
            message = str(exc.detail)
            log.warning("[paper2ppt-task] task %s failed: %s", task_id, message)
            self._update_record(
                task_id,
                status="failed",
                message=message,
                error=message,
                result=None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or exc.__class__.__name__
            log.exception("[paper2ppt-task] task %s failed", task_id)
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

    async def _heartbeat_task(self, task_id: str, req: PPTGenerationRequest) -> None:
        while True:
            await asyncio.sleep(_TASK_HEARTBEAT_INTERVAL_SECONDS)
            try:
                self._update_record(
                    task_id,
                    status="running",
                    message=self._running_message(req),
                    error=None,
                )
            except HTTPException:
                return
            except Exception:
                log.warning("[paper2ppt-task] heartbeat update failed for task %s", task_id)
                return

    def _serialize_record(
        self,
        record: Dict[str, Any],
        request: Request | None = None,
    ) -> Dict[str, Any]:
        result = record.get("result")
        if record.get("status") == "done" and self._result_needs_recovery(result):
            recovered = self._build_recovered_result(record)
            if self._result_has_any_outputs(recovered):
                result = recovered
        normalized_result = None
        if record.get("status") == "done" and isinstance(result, dict):
            normalized_result = self.service.normalize_ppt_response(result, request)

        return {
            "success": True,
            "task_id": record["task_id"],
            "task_type": record.get("task_type", "generate"),
            "status": record.get("status", "queued"),
            "message": record.get("message", ""),
            "error": record.get("error"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "result": normalized_result,
        }

    def _resolve_submission_key(
        self,
        req: PPTGenerationRequest,
        reference_img: UploadFile | None,
        request: Request | None,
    ) -> str | None:
        request_state = getattr(request, "state", None)
        existing = getattr(request_state, "workflow_submission_key", None)
        if existing:
            return str(existing).strip() or None

        reference_img_name = getattr(reference_img, "filename", "") or ""
        payload = {
            "path": "/api/v1/paper2ppt/generate-task",
            "result_path": str(req.result_path or "").strip(),
            "pagecontent": str(req.pagecontent or "").strip(),
            "get_down": _is_truthy(req.get_down),
            "all_edited_down": _is_truthy(req.all_edited_down),
            "page_id": _coerce_int(req.page_id),
            "edit_prompt": str(req.edit_prompt or "").strip(),
            "regenerate_from_outline": _is_truthy(req.regenerate_from_outline),
            "style": str(req.style or "").strip(),
            "model": str(req.model or "").strip(),
            "language": str(req.language or "").strip(),
            "aspect_ratio": str(req.aspect_ratio or "").strip(),
            "img_gen_model_name": str(req.img_gen_model_name or "").strip(),
            "reference_img_name": reference_img_name,
            "skip_pages": _parse_page_index_list(
                req.skip_pages,
                max_count=_pagecontent_count(req.pagecontent),
            ),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        submission_key = hashlib.sha256(encoded).hexdigest()
        if request_state is not None:
            request_state.workflow_submission_key = submission_key
        return submission_key

    def _consume_submission_charge(
        self,
        *,
        req: PPTGenerationRequest,
        request: Request | None,
        submission_key: str | None,
    ) -> None:
        if request is None or _is_truthy(req.all_edited_down):
            return

        if _is_truthy(req.get_down):
            if _coerce_int(req.page_id) is None:
                return
            if not _is_truthy(req.regenerate_from_outline) and not str(req.edit_prompt or "").strip():
                return
            amount = 1
        else:
            total_pages = _pagecontent_count(req.pagecontent)
            skip_count = len(_parse_page_index_list(req.skip_pages, max_count=total_pages))
            amount = max(0, total_pages - skip_count)
            if amount <= 0:
                return
        user = getattr(request.state, "auth_user", None)
        guest_id = getattr(request.state, "guest_id", None)

        event_key = None
        if user and submission_key:
            time_bucket = int(time.time() // _SUBMISSION_WINDOW_SECONDS)
            event_key = f"workflow_paper2ppt_{user.id}_{time_bucket}_{submission_key}"

        BillingService().consume_workflow(
            workflow_type="paper2ppt",
            amount=amount,
            user=user,
            guest_id=guest_id,
            event_key=event_key,
        )

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

    def _refresh_record_state(self, task_id: str) -> Dict[str, Any]:
        record = self._read_record(task_id)
        status = str(record.get("status") or "").strip().lower()

        if status == "done" and self._result_needs_recovery(record.get("result")):
            recovered = self._build_recovered_result(record)
            if self._result_has_any_outputs(recovered):
                return self._update_record(task_id, result=recovered)
            return record

        if status not in {"queued", "running"}:
            return record

        age_seconds = self._record_age_seconds(record)
        if age_seconds is None or age_seconds < _TASK_STALE_TIMEOUT_SECONDS:
            return record

        recovered = self._build_recovered_result(record)
        task_type = self._task_type_from_payload(record.get("request") or {})
        actual_outputs, expected_outputs, is_complete = self._summarize_recovered_outputs(
            task_type=task_type,
            record=record,
            recovered=recovered,
        )

        if is_complete:
            return self._update_record(
                task_id,
                status="done",
                message="Task completed",
                error=None,
                result=recovered,
            )

        if actual_outputs > 0 and expected_outputs > 0:
            message = (
                f"任务已中断：仅生成了 {actual_outputs}/{expected_outputs} 页，"
                "后端 worker 可能已重启，请重新生成"
            )
        else:
            message = "任务已中断：后端 worker 可能已重启，请重新生成"

        return self._update_record(
            task_id,
            status="failed",
            message=message,
            error=message,
            result=recovered if self._result_has_any_outputs(recovered) else None,
        )

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

    def _result_needs_recovery(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return True
        if str(result.get("ppt_pdf_path") or "").strip():
            return False
        if str(result.get("ppt_pptx_path") or "").strip():
            return False
        if result.get("all_output_files"):
            return False
        return not any(
            isinstance(item, dict) and str(item.get("generated_img_path") or "").strip()
            for item in (result.get("pagecontent") or [])
        )

    def _build_recovered_result(self, record: Dict[str, Any]) -> Dict[str, Any]:
        payload = record.get("request") or {}
        result_path_raw = str(payload.get("result_path") or "").strip()
        if not result_path_raw:
            return {
                "success": True,
                "ppt_pdf_path": "",
                "ppt_pptx_path": "",
                "pagecontent": [],
                "result_path": "",
                "all_output_files": [],
            }

        result_root = self.service.resolve_result_path(result_path_raw)
        pdf_path = result_root / "paper2ppt.pdf"
        pptx_path = result_root / "paper2ppt_editable.pptx"

        return {
            "success": True,
            "ppt_pdf_path": str(pdf_path.resolve()) if pdf_path.exists() else "",
            "ppt_pptx_path": str(pptx_path.resolve()) if pptx_path.exists() else "",
            "pagecontent": self._recover_pagecontent(payload, result_root),
            "result_path": str(result_root),
            "all_output_files": self._collect_result_files(result_root),
        }

    def _recover_pagecontent(self, payload: Dict[str, Any], result_root: Path) -> list[Dict[str, Any]]:
        raw_pagecontent = payload.get("pagecontent")
        try:
            parsed = json.loads(str(raw_pagecontent or "").strip()) if raw_pagecontent else []
        except json.JSONDecodeError:
            parsed = []

        if not isinstance(parsed, list):
            parsed = []

        img_dir = result_root / "ppt_pages"
        recovered: list[Dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            page_item = dict(item) if isinstance(item, dict) else {"raw_content": str(item)}
            page_img = img_dir / f"page_{idx:03d}.png"
            page_item["page_idx"] = idx
            page_item["generated_img_path"] = str(page_img.resolve()) if page_img.exists() else ""
            recovered.append(page_item)
        return recovered

    def _collect_result_files(self, result_root: Path) -> list[str]:
        if not result_root.exists():
            return []
        return [
            str(path.resolve())
            for path in sorted(result_root.rglob("*"))
            if path.is_file()
        ]

    def _result_has_any_outputs(self, result: Dict[str, Any]) -> bool:
        if str(result.get("ppt_pdf_path") or "").strip():
            return True
        if str(result.get("ppt_pptx_path") or "").strip():
            return True
        if result.get("all_output_files"):
            return True
        return any(
            isinstance(item, dict) and str(item.get("generated_img_path") or "").strip()
            for item in (result.get("pagecontent") or [])
        )

    def _task_type_from_payload(self, payload: Dict[str, Any]) -> str:
        if _is_truthy(payload.get("get_down")):
            return "edit"
        if _is_truthy(payload.get("all_edited_down")):
            return "finalize"
        return "generate"

    def _summarize_recovered_outputs(
        self,
        *,
        task_type: str,
        record: Dict[str, Any],
        recovered: Dict[str, Any],
    ) -> tuple[int, int, bool]:
        payload = record.get("request") or {}
        pagecontent = recovered.get("pagecontent") or []

        if task_type == "finalize":
            actual = int(bool(recovered.get("ppt_pdf_path") or recovered.get("ppt_pptx_path")))
            return actual, 1, actual >= 1

        if task_type == "edit":
            page_id = _coerce_int(payload.get("page_id"))
            if page_id is None:
                return 0, 1, False
            actual = 0
            for item in pagecontent:
                if not isinstance(item, dict):
                    continue
                if _coerce_int(item.get("page_idx")) == page_id and str(item.get("generated_img_path") or "").strip():
                    actual = 1
                    break
            return actual, 1, actual == 1

        expected = _pagecontent_count(payload.get("pagecontent"))
        actual = sum(
            1
            for item in pagecontent
            if isinstance(item, dict) and str(item.get("generated_img_path") or "").strip()
        )
        if expected <= 0:
            return actual, expected, actual > 0
        return actual, expected, actual >= expected

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

    def _task_type(self, req: PPTGenerationRequest) -> str:
        if str(req.get_down).lower() in ("true", "1", "yes"):
            return "edit"
        if str(req.all_edited_down).lower() in ("true", "1", "yes"):
            return "finalize"
        return "generate"

    def _queued_message(self, req: PPTGenerationRequest) -> str:
        task_type = self._task_type(req)
        if task_type == "finalize":
            return "Final export queued"
        if task_type == "edit":
            return "Slide regeneration queued"
        return "Batch page generation queued"

    def _running_message(self, req: PPTGenerationRequest) -> str:
        task_type = self._task_type(req)
        if task_type == "finalize":
            return "Generating final PPT/PDF"
        if task_type == "edit":
            return "Regenerating slide"
        return "Generating slide pages"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
