from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from dataflow_agent.workflow import run_workflow
from dataflow_agent.state import KBDeepResearchState, KBDeepResearchRequest
from dataflow_agent.utils import get_project_root
from fastapi_app.config.settings import settings
from fastapi_app.services.managed_api_service import is_free_billing_mode, resolve_llm_credentials
from fastapi_app.utils import _to_outputs_url
from fastapi_app.schemas import DeepResearchRequest, DeepResearchResponse


def _ensure_result_path(email: str | None) -> Path:
    project_root = get_project_root()
    ts = int(time.time())
    code = email or "default"
    base_dir = (project_root / "outputs" / "kb_outputs" / code / f"{ts}_deepresearch").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


async def run_kb_deepresearch_wf_api(req: DeepResearchRequest) -> DeepResearchResponse:
    resolved_api_url, resolved_api_key = resolve_llm_credentials(
        req.api_url,
        req.api_key,
        scope="kb_deepresearch",
    )
    resolved_search_api_key = req.search_api_key
    resolved_google_cse_id = req.google_cse_id
    if is_free_billing_mode():
        resolved_search_api_key = resolved_search_api_key or settings.DEFAULT_SEARCH_API_KEY
        resolved_google_cse_id = resolved_google_cse_id or settings.DEFAULT_GOOGLE_CSE_ID
    if req.notebook_id and req.email:
        from fastapi_app.routers.kb import _generated_dir
        result_root = _generated_dir(req.email, req.notebook_id, "deepresearch", req.user_id or "default")
    else:
        result_root = _ensure_result_path(req.email)

    kb_req = KBDeepResearchRequest(
        mode=req.mode,
        topic=req.topic or "",
        file_paths=req.file_paths or [],
        search_provider=req.search_provider,
        search_api_key=resolved_search_api_key,
        search_engine=req.search_engine,
        search_num=req.search_num,
        google_cse_id=resolved_google_cse_id,
        brave_summarizer=req.brave_summarizer,
        search_depth=req.search_depth,
        max_queries=req.max_queries,
        top_k_per_query=req.top_k_per_query,
        fetch_top_n=req.fetch_top_n,
        max_page_chars=req.max_page_chars,
        enable_agentic=req.enable_agentic,
        email=req.email or "",
        user_id=req.user_id or "",
        chat_api_url=resolved_api_url,
        api_key=resolved_api_key,
        chat_api_key=resolved_api_key,
        model=req.model,
        language=req.language,
    )

    state = KBDeepResearchState(request=kb_req, result_path=str(result_root))
    result_state = await run_workflow("kb_deepresearch", state)

    report_markdown = ""
    search_results: list[dict] = []
    sources: list[dict] = []
    summaries: list[dict] = []
    result_path = ""

    if isinstance(result_state, dict):
        report_markdown = result_state.get("report_markdown", "")
        search_results = result_state.get("search_results", []) or []
        sources = result_state.get("sources", []) or []
        summaries = result_state.get("summaries", []) or []
        result_path = result_state.get("result_path", "") or str(result_root)
    else:
        report_markdown = getattr(result_state, "report_markdown", "")
        search_results = getattr(result_state, "search_results", []) or []
        sources = getattr(result_state, "sources", []) or []
        summaries = getattr(result_state, "summaries", []) or []
        result_path = getattr(result_state, "result_path", "") or str(result_root)

    report_file = Path(result_path) / "report.md"
    report_url = _to_outputs_url(str(report_file)) if report_file.exists() else ""

    return DeepResearchResponse(
        success=True,
        report_markdown=report_markdown or "",
        report_path=report_url,
        search_results=search_results or [],
        sources=sources or [],
        summaries=summaries or [],
        output_file_id=f"kb_deepresearch_{int(time.time())}",
    )
