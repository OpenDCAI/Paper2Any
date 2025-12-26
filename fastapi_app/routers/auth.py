"""
Authentication API router.

Provides endpoints for user authentication status and info.
"""

from fastapi import APIRouter, Depends

from fastapi_app.auth import get_current_user, CurrentUser

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)) -> dict:
    """
    Get current authenticated user information.

    Returns:
        User ID and email from the authenticated JWT token

    Raises:
        401 Unauthorized if not authenticated
    """
    return {
        "user_id": user.user_id,
        "email": user.email,
        "role": user.role,
    }
