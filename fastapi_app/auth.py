"""
JWT Authentication middleware for FastAPI.

Provides Supabase JWT token verification and user context extraction.

Usage:
    from fastapi_app.auth import get_current_user, CurrentUser, get_optional_user

    # Required auth - returns 401 if not authenticated
    @router.get("/protected")
    async def protected(user: CurrentUser = Depends(get_current_user)):
        return {"user_id": user.user_id}

    # Optional auth - returns None if not authenticated
    @router.get("/public")
    async def public(user: CurrentUser | None = Depends(get_optional_user)):
        if user:
            return {"message": f"Hello {user.user_id}"}
        return {"message": "Hello anonymous"}
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel

from fastapi_app.config import get_settings

# HTTPBearer with auto_error=False allows us to handle missing auth ourselves
security = HTTPBearer(auto_error=False)


class CurrentUser(BaseModel):
    """Authenticated user context extracted from JWT token."""

    user_id: str
    email: Optional[str] = None
    role: Optional[str] = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    """
    Verify JWT token and return authenticated user.

    Raises HTTPException 401 if:
    - No Authorization header provided
    - Token is malformed
    - Token signature is invalid
    - Token is expired
    - Token is missing 'sub' claim

    Returns:
        CurrentUser with user_id, email, and role from token

    Raises:
        HTTPException: 401 Unauthorized if authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        settings = get_settings()
        if not settings.supabase_jwt_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT secret not configured",
            )

        # Decode and verify the JWT
        payload = jwt.decode(
            credentials.credentials,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )

        # Extract user_id from 'sub' claim (Supabase standard)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
            )

        return CurrentUser(
            user_id=user_id,
            email=payload.get("email"),
            role=payload.get("role"),
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[CurrentUser]:
    """
    Optionally verify JWT token if provided.

    Unlike get_current_user, this returns None instead of raising 401
    when no token is provided. Useful for endpoints that have different
    behavior for authenticated vs anonymous users.

    Returns:
        CurrentUser if valid token provided, None otherwise
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def require_auth_and_quota(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """
    Verify authentication and check rate limit quota.

    This dependency chains:
    1. JWT authentication (via get_current_user)
    2. Rate limit check (via rate_limiter.is_limited)

    Use this for workflow endpoints that should be rate-limited.

    Returns:
        CurrentUser if authenticated and within quota

    Raises:
        HTTPException: 401 if not authenticated
        HTTPException: 429 if daily quota exceeded
    """
    # Import here to avoid circular dependency
    from fastapi_app.services.rate_limiter import rate_limiter

    if await rate_limiter.is_limited(user.user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily limit reached (10 calls/day). Try again tomorrow.",
        )

    return user
