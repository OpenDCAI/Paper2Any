from __future__ import annotations

import time
from pathlib import Path

from dataflow_agent.workflow import run_workflow
from dataflow_agent.state import KBReportState, KBReportRequest
from dataflow_agent.utils import get_project_root
from fastapi_app.services.managed_api_service import resolve_llm_credentials
from fastapi_app.utils import _to_outputs_url
from fastapi_app.schemas import KBReportRequest as KBReportRequestModel, KBReportResponse


def _ensure_result_path(email: str | None) -> Path:
    project_root = get_project_root()
    ts = int(time.time())
    code = email or "default"
    base_dir = (project_root / "outputs" / "kb_outputs" / code / f"{ts}_report").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


async def run_kb_report_wf_api(req: KBReportRequestModel) -> KBReportResponse:
    resolved_api_url, resolved_api_key = resolve_llm_credentials(
        req.api_url,
        req.api_key,
        scope="kb",
    )
    if req.notebook_id and req.email:
        from fastapi_app.routers.kb import _generated_dir
        result_root = _generated_dir(req.email, req.notebook_id, "report", req.user_id or "default")
    else:
        result_root = _ensure_result_path(req.email)

    kb_req = KBReportRequest(
        file_paths=req.file_paths or [],
        report_style=req.report_style,
        length=req.length,
        email=req.email or "",
        user_id=req.user_id or "",
        chat_api_url=resolved_api_url,
        api_key=resolved_api_key,
        chat_api_key=resolved_api_key,
        model=req.model,
        language=req.language,
    )

    state = KBReportState(request=kb_req, result_path=str(result_root))
    result_state = await run_workflow("kb_report", state)

    report_markdown = ""
    result_path = ""

    if isinstance(result_state, dict):
        report_markdown = result_state.get("report_markdown", "")
        result_path = result_state.get("result_path", "") or str(result_root)
    else:
        report_markdown = getattr(result_state, "report_markdown", "")
        result_path = getattr(result_state, "result_path", "") or str(result_root)

    report_file = Path(result_path) / "report.md"
    report_url = _to_outputs_url(str(report_file)) if report_file.exists() else ""

    return KBReportResponse(
        success=True,
        report_markdown=report_markdown or "",
        report_path=report_url,
        output_file_id=f"kb_report_{int(time.time())}",
    )
