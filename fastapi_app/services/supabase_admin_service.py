from __future__ import annotations

from typing import Any, Optional

import httpx
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from fastapi_app.config.settings import settings

_supabase_admin_client: Optional[Client] = None


def get_supabase_admin_client() -> Optional[Client]:
    global _supabase_admin_client

    if _supabase_admin_client is None:
        supabase_url = (settings.SUPABASE_URL or "").strip()
        service_role_key = (settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
        if not supabase_url or not service_role_key:
            return None
        timeout_seconds = max(1.0, float(settings.SUPABASE_POSTGREST_TIMEOUT_SECONDS))
        options = SyncClientOptions(
            postgrest_client_timeout=httpx.Timeout(timeout_seconds),
        )
        _supabase_admin_client = create_client(supabase_url, service_role_key, options=options)

    return _supabase_admin_client


def extract_response_data(response: Any) -> Any:
    return getattr(response, "data", None)
