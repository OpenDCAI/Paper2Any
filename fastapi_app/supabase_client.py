"""
Supabase client factory module.

Provides two client types:
- get_supabase(): Uses anon key, respects Row Level Security (RLS)
- get_supabase_admin(): Uses service role key, bypasses RLS

Usage:
    from fastapi_app.supabase_client import get_supabase, get_supabase_admin

    # For user-scoped operations (respects RLS)
    client = get_supabase()
    result = client.table("usage_records").select("*").execute()

    # For admin operations (bypasses RLS)
    admin = get_supabase_admin()
    result = admin.table("usage_records").select("*").execute()
"""

from typing import Optional

from supabase import create_client, Client

from fastapi_app.config import get_settings

# Module-level singletons for client reuse
_client: Optional[Client] = None
_admin_client: Optional[Client] = None


def get_supabase() -> Client:
    """
    Get Supabase client using anon key (respects RLS).

    This client should be used for operations where Row Level Security
    policies should be enforced. The anon key is safe to expose to clients.

    Returns:
        Supabase client instance

    Raises:
        ValueError: If Supabase is not configured
    """
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.is_supabase_configured():
            raise ValueError(
                "Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY."
            )
        _client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _client


def get_supabase_admin() -> Client:
    """
    Get Supabase admin client using service role key (bypasses RLS).

    This client should ONLY be used in server-side code for admin operations
    that need to bypass Row Level Security. Never expose this to clients.

    Use cases:
    - Recording usage for rate limiting
    - Querying across all users for admin dashboards
    - Cleanup operations

    Returns:
        Supabase admin client instance

    Raises:
        ValueError: If Supabase admin is not configured
    """
    global _admin_client
    if _admin_client is None:
        settings = get_settings()
        if not settings.is_supabase_admin_configured():
            raise ValueError(
                "Supabase admin not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )
        _admin_client = create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    return _admin_client


def reset_clients() -> None:
    """
    Reset client singletons. Useful for testing or config reload.
    """
    global _client, _admin_client
    _client = None
    _admin_client = None
