"""
Rate limiting service for workflow API calls.

Tracks daily usage per user and enforces call limits.

Usage:
    from fastapi_app.services.rate_limiter import rate_limiter, Quota

    # Check quota before processing
    if await rate_limiter.is_limited(user_id):
        raise HTTPException(429, "Daily limit exceeded")

    # Record usage after successful call
    await rate_limiter.record_usage(user_id, "paper2ppt")

    # Get quota for display
    quota = await rate_limiter.check_quota(user_id)
    print(f"Used {quota.used}/{quota.limit} calls today")
"""

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel

from fastapi_app.config import get_settings
from fastapi_app.supabase_client import get_supabase_admin


class Quota(BaseModel):
    """User's daily quota information."""

    used: int
    limit: int
    remaining: int = 0

    def model_post_init(self, __context) -> None:
        """Calculate remaining after initialization."""
        self.remaining = max(0, self.limit - self.used)


class RateLimitService:
    """
    Service for tracking and enforcing rate limits.

    Uses Supabase usage_records table to count daily API calls.
    Limits are configured via DAILY_WORKFLOW_LIMIT environment variable.
    """

    def __init__(self):
        self.settings = get_settings()

    async def get_today_count(self, user_id: str) -> int:
        """
        Count workflow calls made today by user.

        Uses UTC timezone for consistent daily boundaries.

        Args:
            user_id: The user's UUID from Supabase Auth

        Returns:
            Number of calls made today
        """
        supabase = get_supabase_admin()
        today = date.today()

        # Build UTC timestamp range for today
        start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)

        result = (
            supabase.table("usage_records")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .gte("called_at", start.isoformat())
            .lte("called_at", end.isoformat())
            .execute()
        )

        return result.count or 0

    async def check_quota(self, user_id: str) -> Quota:
        """
        Get current quota status for user.

        Args:
            user_id: The user's UUID

        Returns:
            Quota with used, limit, and remaining fields
        """
        used = await self.get_today_count(user_id)
        return Quota(used=used, limit=self.settings.daily_workflow_limit)

    async def is_limited(self, user_id: str) -> bool:
        """
        Check if user has exceeded their daily limit.

        Args:
            user_id: The user's UUID

        Returns:
            True if user is rate limited, False otherwise
        """
        quota = await self.check_quota(user_id)
        return quota.used >= quota.limit

    async def record_usage(
        self, user_id: str, workflow_type: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Record a workflow call for rate limiting.

        Should be called AFTER successful workflow completion.

        Args:
            user_id: The user's UUID
            workflow_type: Type of workflow (paper2ppt, paper2figure, etc.)
            metadata: Optional additional data to store
        """
        supabase = get_supabase_admin()
        supabase.table("usage_records").insert(
            {
                "user_id": user_id,
                "workflow_type": workflow_type,
            }
        ).execute()


# Module-level singleton
rate_limiter = RateLimitService()
