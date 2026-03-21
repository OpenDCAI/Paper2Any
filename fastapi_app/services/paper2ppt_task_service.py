from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException, Request, UploadFile

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from fastapi_app.schemas import PPTGenerationRequest
from fastapi_app.services.paper2ppt_service import Paper2PPTService

log = get_logger(__name__)

PROJECT_ROOT = get_project_root()
TASK_ROOT = (PROJECT_ROOT / "outputs" / ".tasks" / "paper2ppt").resolve()
_ACTIVE_TASKS: set[asyncio.Task[Any]] = set()


class Paper2PPTTaskService:
    """File-backed async tasks for long-running paper2ppt generation."""

    def __init__(self, service: Paper2PPTService | None = None) -> None:
        self.service = service or Paper2PPTService()

    async def submit_generate_task(
        self,
        req: PPTGenerationRequest,
        reference_img: UploadFile | None,
    ) -> Dict[str, Any]:
        base_dir = self.service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        if reference_img is not None:
            await self.service.cache_reference_image_for_result(str(base_dir), reference_img)

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

        task = asyncio.create_task(self._run_generate_task(task_id))
        _ACTIVE_TASKS.add(task)
        task.add_done_callback(_ACTIVE_TASKS.discard)

        return self._serialize_record(record)

    def get_task(self, task_id: str, request: Request | None = None) -> Dict[str, Any]:
        record = self._read_record(task_id)
        return self._serialize_record(record, request)

    async def _run_generate_task(self, task_id: str) -> None:
        record = self._read_record(task_id)
        payload = record.get("request") or {}
        req = PPTGenerationRequest(**payload)

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

    def _serialize_record(
        self,
        record: Dict[str, Any],
        request: Request | None = None,
    ) -> Dict[str, Any]:
        result = record.get("result")
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

    def _task_dir(self, task_id: str) -> Path:
        return TASK_ROOT / task_id

    def _task_file(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "task.json"

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
