from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from fastapi import HTTPException

from dataflow_agent.logger import get_logger
from fastapi_app.config.pricing import get_pricing_config, get_redeem_code_files, get_workflow_cost
from fastapi_app.config.settings import settings
from fastapi_app.dependencies import AuthUser
from fastapi_app.dependencies.auth import is_auth_configured
from fastapi_app.services.managed_api_service import get_runtime_billing_config, is_free_billing_mode
from fastapi_app.services.supabase_admin_service import extract_response_data, get_supabase_admin_client

log = get_logger(__name__)


@dataclass
class _QuotaCacheEntry:
    value: Dict[str, Any]
    expires_at: float
    stale_until: float


_quota_cache: Dict[str, _QuotaCacheEntry] = {}
_quota_cache_lock = threading.Lock()


class BillingService:
    def __init__(self) -> None:
        pass

    def _pricing(self) -> dict:
        return get_pricing_config()

    def _billing_config(self) -> dict:
        return self._pricing().get("billing", {})

    def _auth_enabled(self) -> bool:
        return is_auth_configured()

    def _quota_cache_ttl_seconds(self) -> int:
        return max(0, int(settings.BILLING_QUOTA_CACHE_TTL_SECONDS))

    def _quota_stale_ttl_seconds(self) -> int:
        return max(self._quota_cache_ttl_seconds(), int(settings.BILLING_QUOTA_STALE_TTL_SECONDS))

    def _quota_cache_key(self, user_id: str) -> str:
        return str(user_id)

    def _get_cached_quota(self, user_id: str, *, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        key = self._quota_cache_key(user_id)
        with _quota_cache_lock:
            entry = _quota_cache.get(key)
            if entry is None:
                return None
            if entry.stale_until <= now:
                _quota_cache.pop(key, None)
                return None
            if not allow_stale and entry.expires_at <= now:
                return None
            return dict(entry.value)

    def _set_cached_quota(self, user_id: str, quota: Dict[str, Any]) -> None:
        ttl_seconds = self._quota_cache_ttl_seconds()
        stale_ttl_seconds = self._quota_stale_ttl_seconds()
        now = time.monotonic()
        key = self._quota_cache_key(user_id)
        with _quota_cache_lock:
            _quota_cache[key] = _QuotaCacheEntry(
                value=dict(quota),
                expires_at=now + ttl_seconds,
                stale_until=now + stale_ttl_seconds,
            )

    def _invalidate_cached_quota(self, *user_ids: Optional[str]) -> None:
        keys = [self._quota_cache_key(user_id) for user_id in user_ids if user_id]
        if not keys:
            return
        with _quota_cache_lock:
            for key in keys:
                _quota_cache.pop(key, None)

    def _billing_backend_unavailable(self) -> HTTPException:
        return HTTPException(
            status_code=503,
            detail="Billing service temporarily unavailable. Please retry shortly.",
            headers={"Retry-After": "3"},
        )

    def _execute_query(self, query: Any) -> Any:
        try:
            return query.execute()
        except httpx.TimeoutException as exc:
            log.warning("Supabase billing request timed out: %s", exc)
            raise self._billing_backend_unavailable() from exc
        except httpx.HTTPError as exc:
            log.warning("Supabase billing request failed: %s", exc)
            raise self._billing_backend_unavailable() from exc

    def _unlimited_quota(self) -> Dict[str, Any]:
        unlimited = 9_999_999
        return {
            "used": 0,
            "limit": unlimited,
            "remaining": unlimited,
            "is_authenticated": False,
            "billing_mode": get_runtime_billing_config()["billing_mode"],
        }

    def _supabase(self):
        client = get_supabase_admin_client()
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="Supabase service-role configuration is required for account and billing APIs",
            )
        return client

    def _select_many(
        self,
        table: str,
        columns: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        ascending: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = self._supabase().table(table).select(columns)
        for key, value in (filters or {}).items():
            query = query.eq(key, value)
        if order_by:
            query = query.order(order_by, desc=not ascending)
        if limit is not None:
            query = query.limit(limit)
        data = extract_response_data(self._execute_query(query)) or []
        return data if isinstance(data, list) else []

    def _select_first(self, table: str, columns: str, **filters: Any) -> Optional[Dict[str, Any]]:
        rows = self._select_many(table, columns, filters=filters, limit=1)
        return rows[0] if rows else None

    def _insert_row(self, table: str, payload: Dict[str, Any]) -> Any:
        response = self._execute_query(self._supabase().table(table).insert(payload))
        return response

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def _resolve_optional_path(self, value: str | None) -> Optional[Path]:
        raw = (value or "").strip()
        if not raw:
            return None

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (self._project_root() / path).resolve()
        return path

    def _points_redeem_catalog_paths(self) -> List[tuple[int, Path]]:
        candidates = tuple(sorted(get_redeem_code_files().items(), key=lambda item: item[0]))

        resolved: List[tuple[int, Path]] = []
        for points, raw_path in candidates:
            path = self._resolve_optional_path(raw_path)
            if path is not None:
                resolved.append((points, path))
        return resolved

    def _load_points_redeem_catalog(self) -> Dict[str, int]:
        catalog: Dict[str, int] = {}
        duplicates: List[str] = []

        for points, path in self._points_redeem_catalog_paths():
            if not path.is_file():
                continue

            for raw_line in path.read_text(encoding="utf-8").splitlines():
                code = raw_line.strip().upper()
                if not code or code.startswith("#"):
                    continue
                if code in catalog:
                    duplicates.append(code)
                    continue
                catalog[code] = points

        if duplicates:
            raise HTTPException(
                status_code=500,
                detail="Duplicate redeem code detected in configured code files",
            )
        return catalog

    def _normalize_redeem_code(self, redeem_code: str) -> str:
        return (redeem_code or "").strip().upper()

    def _redeem_event_key(self, normalized_code: str) -> str:
        digest = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()
        return f"redeem_code_{digest}"

    def _is_unique_violation(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "duplicate key" in message or "unique" in message or "23505" in message

    def _generate_invite_code(self) -> str:
        for _ in range(12):
            candidate = uuid4().hex[:8].upper()
            if not self._select_first("profiles", "user_id", invite_code=candidate):
                return candidate
        raise HTTPException(status_code=500, detail="Failed to allocate invite code")

    def _ensure_profile(self, user: AuthUser) -> None:
        existing = self._select_first("profiles", "user_id, invite_code", user_id=user.id)
        if existing:
            return
        self._insert_row(
            "profiles",
            {
                "user_id": user.id,
                "invite_code": self._generate_invite_code(),
            },
        )

    def _append_ledger(
        self,
        *,
        user_id: str,
        points: int,
        reason: str,
        event_key: Optional[str] = None,
    ) -> None:
        payload = {
            "user_id": user_id,
            "points": int(points),
            "reason": reason,
        }
        if event_key:
            payload["event_key"] = event_key
        self._insert_row("points_ledger", payload)
        self._invalidate_cached_quota(user_id)

    def _ensure_signup_bonus(self, user: AuthUser) -> None:
        signup_bonus_points = int(self._billing_config().get("signup_bonus_points", 0))
        if signup_bonus_points <= 0:
            return
        event_key = f"signup_bonus_{user.id}"
        existing = self._select_first("points_ledger", "id", event_key=event_key)
        if existing:
            return
        self._append_ledger(
            user_id=user.id,
            points=signup_bonus_points,
            reason="signup_bonus",
            event_key=event_key,
        )

    def _bootstrap_user(self, user: AuthUser) -> None:
        self._ensure_profile(user)
        self._ensure_signup_bonus(user)

    def _get_balance(self, user_id: str) -> int:
        balance_row = self._select_first("points_balance", "user_id, balance", user_id=user_id)
        if not balance_row:
            return 0
        try:
            return int(balance_row.get("balance", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _grant_daily_points_if_needed(self, user: AuthUser) -> None:
        billing = self._billing_config()
        daily_points = int(billing.get("daily_grant_points", 5))
        balance_cap = int(billing.get("daily_grant_balance_cap", 15))
        if daily_points <= 0 or balance_cap <= 0:
            return

        current_balance = self._get_balance(user.id)
        if current_balance >= balance_cap:
            return

        grant_points = min(daily_points, balance_cap - current_balance)
        if grant_points <= 0:
            return

        today = date.today().isoformat()
        event_key = f"daily_grant_{today}_{user.id}"
        existing = self._select_first("points_ledger", "id", event_key=event_key)
        if existing:
            return

        self._append_ledger(
            user_id=user.id,
            points=grant_points,
            reason="daily_grant",
            event_key=event_key,
        )

    def _quota_from_user(self, user: AuthUser) -> Dict[str, Any]:
        self._bootstrap_user(user)

        if not is_free_billing_mode():
            unlimited = 9_999_999
            return {
                "used": 0,
                "limit": unlimited,
                "remaining": unlimited,
                "is_authenticated": True,
                "billing_mode": "paid",
                "user_id": user.id,
            }

        self._grant_daily_points_if_needed(user)
        balance = self._get_balance(user.id)
        return {
            "used": 0,
            "limit": balance,
            "remaining": balance,
            "is_authenticated": True,
            "billing_mode": "free",
            "user_id": user.id,
        }

    def get_runtime_config(self) -> Dict[str, Any]:
        return get_runtime_billing_config()

    def get_quota(self, *, user: Optional[AuthUser], guest_id: Optional[str]) -> Dict[str, Any]:
        if not self._auth_enabled():
            return self._unlimited_quota()
        if user and not getattr(user, "is_anonymous", False):
            cached_quota = self._get_cached_quota(user.id)
            if cached_quota is not None:
                return cached_quota

            try:
                quota = self._quota_from_user(user)
            except HTTPException as exc:
                if exc.status_code == 503:
                    stale_quota = self._get_cached_quota(user.id, allow_stale=True)
                    if stale_quota is not None:
                        stale_quota["stale"] = True
                        return stale_quota
                raise

            self._set_cached_quota(user.id, quota)
            return quota
        raise HTTPException(status_code=401, detail="Authentication required")

    def consume_workflow(
        self,
        *,
        workflow_type: str,
        user: Optional[AuthUser],
        guest_id: Optional[str],
        amount: Optional[int] = None,
        event_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        requested_amount = max(1, int(amount or get_workflow_cost(workflow_type, default=1)))
        if not self._auth_enabled():
            return {
                "success": True,
                "workflow_type": workflow_type,
                "amount": 0,
                "remaining": None,
                "billing_mode": get_runtime_billing_config()["billing_mode"],
            }

        if user and not getattr(user, "is_anonymous", False):
            if not is_free_billing_mode():
                return {
                    "success": True,
                    "workflow_type": workflow_type,
                    "amount": 0,
                    "remaining": None,
                    "billing_mode": "paid",
                }

            self._bootstrap_user(user)
            self._grant_daily_points_if_needed(user)
            if event_key and self._select_first("points_ledger", "id", event_key=event_key):
                remaining = self._get_balance(user.id)
                self._set_cached_quota(
                    user.id,
                    {
                        "used": 0,
                        "limit": remaining,
                        "remaining": remaining,
                        "is_authenticated": True,
                        "billing_mode": "free",
                        "user_id": user.id,
                    },
                )
                return {
                    "success": True,
                    "workflow_type": workflow_type,
                    "amount": 0,
                    "remaining": remaining,
                    "billing_mode": "free",
                    "deduplicated": True,
                }
            balance = self._get_balance(user.id)
            if balance < requested_amount:
                raise HTTPException(status_code=402, detail="Insufficient points")
            self._append_ledger(
                user_id=user.id,
                points=-requested_amount,
                reason=f"workflow_{workflow_type}",
                event_key=event_key or f"workflow_{workflow_type}_{user.id}_{uuid4().hex}",
            )
            remaining = balance - requested_amount
            self._set_cached_quota(
                user.id,
                {
                    "used": 0,
                    "limit": remaining,
                    "remaining": remaining,
                    "is_authenticated": True,
                    "billing_mode": "free",
                    "user_id": user.id,
                },
            )
            return {
                "success": True,
                "workflow_type": workflow_type,
                "amount": requested_amount,
                "remaining": remaining,
                "billing_mode": "free",
            }

        raise HTTPException(status_code=401, detail="Authentication required")

    def get_account_profile(self, user: AuthUser) -> Dict[str, Any]:
        if getattr(user, "is_anonymous", False):
            raise HTTPException(status_code=400, detail="Anonymous users do not have an account profile")

        self._bootstrap_user(user)
        if is_free_billing_mode():
            self._grant_daily_points_if_needed(user)

        profile = self._select_first("profiles", "user_id, invite_code, created_at, updated_at", user_id=user.id) or {}
        referrals = self._select_many(
            "referrals",
            "id, inviter_user_id, invitee_user_id, invite_code, created_at",
            filters={"inviter_user_id": user.id},
            order_by="created_at",
            limit=20,
        )
        ledger = self._select_many(
            "points_ledger",
            "id, points, reason, event_key, created_at",
            filters={"user_id": user.id},
            order_by="created_at",
            limit=50,
        )

        return {
            "billing_mode": get_runtime_billing_config()["billing_mode"],
            "profile": profile,
            "points": {"balance": self._get_balance(user.id)},
            "referrals": referrals,
            "points_ledger": ledger,
            "pricing": self._pricing(),
        }

    def redeem_points_code(self, *, user: AuthUser, redeem_code: str) -> Dict[str, Any]:
        if getattr(user, "is_anonymous", False):
            raise HTTPException(status_code=400, detail="Anonymous users cannot redeem points codes")

        normalized_code = self._normalize_redeem_code(redeem_code)
        if not normalized_code:
            raise HTTPException(status_code=400, detail="redeem_code is required")

        self._bootstrap_user(user)

        event_key = self._redeem_event_key(normalized_code)
        existing = self._select_first("points_ledger", "id", event_key=event_key)
        if existing:
            raise HTTPException(status_code=409, detail="Redeem code already claimed")

        catalog = self._load_points_redeem_catalog()
        if not catalog:
            raise HTTPException(status_code=503, detail="Redeem code catalog is not configured")

        points = catalog.get(normalized_code)
        if points is None:
            raise HTTPException(status_code=404, detail="Invalid redeem code")

        try:
            self._append_ledger(
                user_id=user.id,
                points=points,
                reason=f"redeem_code_{points}",
                event_key=event_key,
            )
        except Exception as exc:
            if self._is_unique_violation(exc):
                raise HTTPException(status_code=409, detail="Redeem code already claimed") from exc
            raise

        remaining = self._get_balance(user.id)
        self._invalidate_cached_quota(user.id)
        return {
            "success": True,
            "points_added": points,
            "remaining": remaining,
            "billing_mode": get_runtime_billing_config()["billing_mode"],
        }

    def claim_invite_code(self, *, user: AuthUser, invite_code: str) -> Dict[str, Any]:
        if getattr(user, "is_anonymous", False):
            raise HTTPException(status_code=400, detail="Anonymous users cannot claim invite codes")

        normalized_code = (invite_code or "").strip().upper()
        if not normalized_code:
            raise HTTPException(status_code=400, detail="invite_code is required")

        self._bootstrap_user(user)

        existing_referral = self._select_first("referrals", "id", invitee_user_id=user.id)
        if existing_referral:
            raise HTTPException(status_code=409, detail="Invite code already claimed")

        inviter_profile = self._select_first("profiles", "user_id, invite_code", invite_code=normalized_code)
        if not inviter_profile:
            raise HTTPException(status_code=404, detail="Invalid invite code")

        inviter_user_id = str(inviter_profile.get("user_id", "") or "")
        if inviter_user_id == user.id:
            raise HTTPException(status_code=400, detail="Cannot claim your own invite code")

        self._insert_row(
            "referrals",
            {
                "inviter_user_id": inviter_user_id,
                "invitee_user_id": user.id,
                "invite_code": normalized_code,
            },
        )

        inviter_points = int(self._billing_config().get("referral_inviter_points", 5))
        invitee_points = int(self._billing_config().get("referral_invitee_points", 0))

        if inviter_points > 0:
            self._append_ledger(
                user_id=inviter_user_id,
                points=inviter_points,
                reason="referral_inviter",
                event_key=f"referral_inviter_{inviter_user_id}_{user.id}",
            )
        if invitee_points > 0:
            self._append_ledger(
                user_id=user.id,
                points=invitee_points,
                reason="referral_invitee",
                event_key=f"referral_invitee_{user.id}_{normalized_code}",
            )
        self._invalidate_cached_quota(inviter_user_id, user.id)

        return {
            "success": True,
            "invite_code": normalized_code,
            "inviter_user_id": inviter_user_id,
            "inviter_reward_points": inviter_points,
            "invitee_reward_points": invitee_points,
            "remaining": self._get_balance(user.id),
        }
