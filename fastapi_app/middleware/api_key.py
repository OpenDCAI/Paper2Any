"""
API key, quota, and rate-limit middleware for workflow endpoints.

The frontend still sends X-API-Key for browser-to-backend traffic, but this
middleware also enforces:
1. per-IP write-rate limiting
2. server-side quota / billing checks before expensive workflow execution
"""

from __future__ import annotations

import os
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import Lock
from typing import Any, Optional

from fastapi import HTTPException, Header, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from fastapi_app.config.pricing import get_workflow_cost
from fastapi_app.config.settings import settings
from fastapi_app.dependencies.auth import AuthUser, get_optional_user
from fastapi_app.services.billing_service import BillingService

# Internal API key for frontend-backend communication.
# Read from environment so deployment can rotate it without code changes.
API_KEY = (os.getenv("BACKEND_API_KEY", "") or "").strip()

# Paths that don't require API key
EXCLUDED_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/v1/files/stream",
}

# Path prefixes that don't require API key
EXCLUDED_PREFIXES = (
    "/outputs/",  # Static files
)

_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@dataclass(frozen=True)
class WorkflowGuard:
    workflow_type: str
    consume_before_execute: bool = True


@dataclass(frozen=True)
class WorkflowChargeDecision:
    workflow_type: str
    amount: Optional[int] = None
    event_key: Optional[str] = None


@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int
    bucket: str


_DEFAULT_WRITE_RULE = RateLimitRule(limit=120, window_seconds=60, bucket="write:global")
_RATE_LIMIT_RULES: dict[str, RateLimitRule] = {
    "/api/v1/system/verify-llm": RateLimitRule(limit=20, window_seconds=60, bucket="verify-llm"),
    "/api/v1/paper2figure/generate": RateLimitRule(limit=12, window_seconds=300, bucket="paper2figure"),
    "/api/v1/paper2figure/generate-json": RateLimitRule(limit=12, window_seconds=300, bucket="paper2figure-json"),
    "/api/v1/pdf2ppt/generate": RateLimitRule(limit=8, window_seconds=300, bucket="pdf2ppt"),
    "/api/v1/image2ppt/generate": RateLimitRule(limit=12, window_seconds=300, bucket="image2ppt"),
    "/api/v1/image2drawio/generate": RateLimitRule(limit=12, window_seconds=300, bucket="image2drawio"),
    "/api/v1/mindmap/generate": RateLimitRule(limit=16, window_seconds=300, bucket="mindmap-generate"),
    "/api/v1/paper2drawio/generate": RateLimitRule(limit=10, window_seconds=300, bucket="paper2drawio-generate"),
    "/api/v1/paper2drawio/chat": RateLimitRule(limit=24, window_seconds=300, bucket="paper2drawio-chat"),
    "/api/v1/image-playground/generate": RateLimitRule(limit=12, window_seconds=300, bucket="image-playground-generate"),
    "/api/v1/paper2poster/generate": RateLimitRule(limit=8, window_seconds=300, bucket="paper2poster"),
    "/api/v1/paper2video/generate-subtitle": RateLimitRule(limit=6, window_seconds=300, bucket="paper2video-subtitle"),
    "/api/v1/paper2video/generate-video": RateLimitRule(limit=4, window_seconds=600, bucket="paper2video-video"),
    "/api/v1/paper2citation/author/detail": RateLimitRule(limit=20, window_seconds=300, bucket="paper2citation-author"),
    "/api/v1/paper2citation/paper/detail": RateLimitRule(limit=20, window_seconds=300, bucket="paper2citation-paper"),
    "/api/v1/paper2citation/paper/context": RateLimitRule(limit=30, window_seconds=300, bucket="paper2citation-context"),
    "/api/v1/paper2rebuttal/parse-review": RateLimitRule(limit=10, window_seconds=300, bucket="paper2rebuttal-parse"),
    "/api/v1/paper2rebuttal/start": RateLimitRule(limit=4, window_seconds=600, bucket="paper2rebuttal-start"),
    "/api/v1/paper2rebuttal/revise": RateLimitRule(limit=20, window_seconds=600, bucket="paper2rebuttal-revise"),
    "/api/v1/paper2rebuttal/generate-final": RateLimitRule(limit=8, window_seconds=600, bucket="paper2rebuttal-final"),
    "/api/v1/paper2ppt/page-content": RateLimitRule(limit=12, window_seconds=300, bucket="paper2ppt-page-content"),
    "/api/v1/paper2ppt/generate": RateLimitRule(limit=12, window_seconds=300, bucket="paper2ppt-generate"),
    "/api/v1/paper2ppt/generate-task": RateLimitRule(limit=12, window_seconds=300, bucket="paper2ppt-generate-task"),
    "/api/v1/paper2ppt/outline-refine": RateLimitRule(limit=20, window_seconds=300, bucket="paper2ppt-outline-refine"),
    "/api/v1/paper2ppt/frontend/generate": RateLimitRule(limit=16, window_seconds=300, bucket="paper2ppt-frontend-generate"),
    "/api/v1/paper2ppt/frontend/review": RateLimitRule(limit=24, window_seconds=300, bucket="paper2ppt-frontend-review"),
    "/api/v1/kb/chat": RateLimitRule(limit=30, window_seconds=300, bucket="kb-chat"),
    "/api/v1/kb/search": RateLimitRule(limit=40, window_seconds=300, bucket="kb-search"),
    "/api/v1/kb/generate-ppt": RateLimitRule(limit=12, window_seconds=300, bucket="kb-ppt"),
    "/api/v1/kb/generate-podcast": RateLimitRule(limit=8, window_seconds=300, bucket="kb-podcast"),
    "/api/v1/kb/generate-mindmap": RateLimitRule(limit=16, window_seconds=300, bucket="kb-mindmap"),
    "/api/v1/kb/deep-research": RateLimitRule(limit=6, window_seconds=600, bucket="kb-deep-research"),
    "/api/v1/kb/generate-report": RateLimitRule(limit=10, window_seconds=300, bucket="kb-report"),
}

_WORKFLOW_GUARDS: dict[str, WorkflowGuard] = {
    "/api/v1/paper2figure/generate": WorkflowGuard("paper2figure"),
    "/api/v1/paper2figure/generate-json": WorkflowGuard("paper2figure"),
    "/api/v1/pdf2ppt/generate": WorkflowGuard("pdf2ppt"),
    "/api/v1/image2ppt/generate": WorkflowGuard("image2ppt"),
    "/api/v1/image2drawio/generate": WorkflowGuard("image2drawio"),
    "/api/v1/mindmap/generate": WorkflowGuard("kb_mindmap", consume_before_execute=False),
    "/api/v1/paper2drawio/generate": WorkflowGuard("paper2drawio"),
    "/api/v1/paper2drawio/chat": WorkflowGuard("paper2drawio"),
    "/api/v1/image-playground/generate": WorkflowGuard("image_playground", consume_before_execute=False),
    "/api/v1/paper2poster/generate": WorkflowGuard("paper2poster"),
    "/api/v1/paper2video/generate-subtitle": WorkflowGuard("paper2video", consume_before_execute=False),
    "/api/v1/paper2video/generate-video": WorkflowGuard("paper2video"),
    "/api/v1/paper2video": WorkflowGuard("paper2video"),
    "/api/v1/paper2citation/author/detail": WorkflowGuard("paper2citation"),
    "/api/v1/paper2citation/paper/detail": WorkflowGuard("paper2citation"),
    "/api/v1/paper2citation/paper/context": WorkflowGuard("paper2citation"),
    "/api/v1/paper2rebuttal/parse-review": WorkflowGuard("paper2rebuttal"),
    "/api/v1/paper2rebuttal/start": WorkflowGuard("paper2rebuttal"),
    "/api/v1/paper2rebuttal/revise": WorkflowGuard("paper2rebuttal"),
    "/api/v1/paper2rebuttal/generate-final": WorkflowGuard("paper2rebuttal"),
    "/api/v1/paper2ppt/generate": WorkflowGuard("paper2ppt"),
    "/api/v1/paper2ppt/generate-task": WorkflowGuard("paper2ppt"),
    "/api/v1/paper2ppt/frontend/generate": WorkflowGuard("paper2ppt"),
    "/api/v1/kb/chat": WorkflowGuard("kb_chat"),
    "/api/v1/kb/search": WorkflowGuard("kb_search"),
    "/api/v1/kb/generate-ppt": WorkflowGuard("kb_ppt"),
    "/api/v1/kb/generate-podcast": WorkflowGuard("kb_podcast"),
    "/api/v1/kb/generate-mindmap": WorkflowGuard("kb_mindmap"),
    "/api/v1/kb/deep-research": WorkflowGuard("kb_deepresearch"),
    "/api/v1/kb/generate-report": WorkflowGuard("kb_report"),
}

_RATE_LIMIT_STORAGE: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = Lock()


def _should_check_api_key(path: str) -> bool:
    return path.startswith("/api/") or path.startswith("/paper2video/")


def _blocked_public_output_prefixes() -> set[str]:
    raw = (settings.SECURITY_BLOCKED_PUBLIC_OUTPUT_PREFIXES or "").strip()
    if not raw:
        return set()
    return {item.strip().strip("/") for item in raw.split(",") if item.strip()}


def _is_blocked_public_output_path(path: str) -> bool:
    if not path.startswith("/outputs/"):
        return False

    relative = path[len("/outputs/") :]
    parts = [part for part in PurePosixPath(relative).parts if part and part != "/"]
    if not parts:
        return False

    if parts[0] in _blocked_public_output_prefixes():
        return True

    return any(part.startswith(".") or part == "__pycache__" for part in parts)


def _extract_client_ip(request: Request) -> str:
    if settings.SECURITY_TRUST_PROXY_HEADERS:
        for header_name in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
            value = (request.headers.get(header_name) or "").strip()
            if not value:
                continue
            if header_name == "X-Forwarded-For":
                first_hop = value.split(",", 1)[0].strip()
                if first_hop:
                    return first_hop
                continue
            return value
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _build_rate_limit_key(ip: str, bucket: str) -> str:
    return f"{ip}:{bucket}"


def _is_rate_limited(ip: str, rule: RateLimitRule) -> bool:
    now = time.monotonic()
    cutoff = now - float(rule.window_seconds)
    key = _build_rate_limit_key(ip, rule.bucket)
    with _RATE_LIMIT_LOCK:
        hits = _RATE_LIMIT_STORAGE[key]
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= rule.limit:
            return True
        hits.append(now)
    return False


def _resolve_requested_amount(request: Request, workflow_type: str) -> Optional[int]:
    raw = (request.headers.get("X-Workflow-Amount") or "").strip()
    if not raw:
        return None
    try:
        amount = int(raw)
    except ValueError:
        return None
    default_amount = max(1, int(get_workflow_cost(workflow_type, default=1)))
    return max(default_amount, min(amount, 10_000))


def _resolve_guest_id(
    request: Request,
    user: Optional[AuthUser],
    client_ip: str,
) -> str:
    return f"ip:{client_ip}"


def _is_truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(raw: Any) -> Optional[int]:
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

async def _resolve_workflow_charge_decision(
    request: Request,
    path: str,
    workflow_guard: WorkflowGuard,
    user: Optional[AuthUser],
    guest_id: str,
) -> Optional[WorkflowChargeDecision]:
    if path == "/api/v1/paper2video/generate-subtitle":
        return WorkflowChargeDecision(workflow_type=workflow_guard.workflow_type, amount=0)

    if path == "/api/v1/mindmap/generate":
        return WorkflowChargeDecision(workflow_type=workflow_guard.workflow_type, amount=0)

    if path == "/api/v1/image-playground/generate":
        amount = _resolve_requested_amount(request, workflow_guard.workflow_type)
        inferred_amount = max(1, int(get_workflow_cost(workflow_guard.workflow_type, default=1)))
        try:
            payload = json.loads((await request.body() or b"{}").decode("utf-8"))
            batch_count = int(payload.get("batch_count") or 1)
            if batch_count > 1:
                inferred_amount *= batch_count
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            pass
        resolved_amount = max(inferred_amount, int(amount or inferred_amount))
        return WorkflowChargeDecision(workflow_type=workflow_guard.workflow_type, amount=resolved_amount)

    if not path.startswith("/api/v1/paper2ppt/"):
        amount = _resolve_requested_amount(request, workflow_guard.workflow_type)
        return WorkflowChargeDecision(workflow_type=workflow_guard.workflow_type, amount=amount)

    # These routes use form payloads that must remain available to FastAPI's
    # downstream parsing. Route/task services now handle billing and dedupe
    # after parameter binding succeeds.
    if path in {
        "/api/v1/paper2ppt/generate",
        "/api/v1/paper2ppt/generate-task",
        "/api/v1/paper2ppt/frontend/generate",
    }:
        return None

    form = await request.form()

    if path == "/api/v1/paper2ppt/frontend/generate":
        page_id = _coerce_int(form.get("page_id"))
        amount = _resolve_requested_amount(request, workflow_guard.workflow_type)
        if amount is None:
            if page_id is not None:
                amount = 1
            else:
                per_page = 2 if _is_truthy(form.get("include_images")) else 1
                amount = max(1, _pagecontent_count(form.get("pagecontent")) * per_page)
        return WorkflowChargeDecision(
            workflow_type=workflow_guard.workflow_type,
            amount=max(1, int(amount or 1)),
        )

    amount = _resolve_requested_amount(request, workflow_guard.workflow_type)
    return WorkflowChargeDecision(workflow_type=workflow_guard.workflow_type, amount=amount)


async def _resolve_user(request: Request) -> Optional[AuthUser]:
    return await get_optional_user(request.headers.get("Authorization"))


def _billing_error_response(exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that verifies API key for /api/* routes and enforces
    server-side workflow protection for high-cost endpoints.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # CORS preflight requests do not include custom auth headers.
        # Let CORSMiddleware handle them.
        if request.method == "OPTIONS":
            return await call_next(request)

        if _is_blocked_public_output_path(path):
            return Response(status_code=404)

        # Skip excluded paths
        if path in EXCLUDED_PATHS:
            return await call_next(request)

        # Skip excluded prefixes
        if path.startswith(EXCLUDED_PREFIXES):
            return await call_next(request)

        client_ip = _extract_client_ip(request)
        request.state.client_ip = client_ip

        if (
            settings.SECURITY_RATE_LIMIT_ENABLED
            and request.method in _WRITE_METHODS
            and _should_check_api_key(path)
        ):
            if _is_rate_limited(client_ip, _DEFAULT_WRITE_RULE):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many write requests from this IP. Please retry later."},
                )
            specific_rule = _RATE_LIMIT_RULES.get(path)
            if specific_rule and _is_rate_limited(client_ip, specific_rule):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded for this endpoint. Please retry later."},
                )

        # Only check API key for /api/* and /paper2video/* routes
        if _should_check_api_key(path):
            if not API_KEY:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "BACKEND_API_KEY is not configured"},
                )
            api_key = request.headers.get("X-API-Key")
            # EventSource cannot set custom headers, allow query param for rebuttal SSE.
            if not api_key and request.method == "GET" and "/paper2rebuttal/progress/" in path:
                api_key = request.query_params.get("x_api_key") or request.query_params.get("X-API-Key")

            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key required"},
                )

            if api_key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"},
                )

        user = await _resolve_user(request) if _should_check_api_key(path) else None
        guest_id = _resolve_guest_id(request, user, client_ip)
        request.state.auth_user = user
        request.state.guest_id = guest_id

        workflow_guard = _WORKFLOW_GUARDS.get(path)
        if workflow_guard and request.method in _WRITE_METHODS:
            billing_service = BillingService()
            decision = await _resolve_workflow_charge_decision(
                request,
                path,
                workflow_guard,
                user,
                guest_id,
            )
            if decision is None:
                return await call_next(request)

            try:
                if workflow_guard.consume_before_execute:
                    billing_service.consume_workflow(
                        workflow_type=decision.workflow_type,
                        amount=decision.amount,
                        user=user,
                        guest_id=guest_id,
                        event_key=decision.event_key,
                    )
                else:
                    quota = billing_service.get_quota(user=user, guest_id=guest_id)
                    required = decision.amount if decision.amount is not None else max(
                        1, int(get_workflow_cost(decision.workflow_type, default=1))
                    )
                    if required > 0 and int(quota.get("remaining", 0) or 0) < required:
                        raise HTTPException(status_code=402, detail="Insufficient points")
            except HTTPException as exc:
                return _billing_error_response(exc)

        return await call_next(request)


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> None:
    """
    Verify the API key in request header (for use as Depends).

    Raises:
        HTTPException 401 if key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_API_KEY is not configured",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
