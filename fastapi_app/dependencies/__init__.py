"""FastAPI dependencies."""
from .auth import get_current_user, get_current_user_or_system, get_optional_user, is_auth_configured, AuthUser

__all__ = [
    "get_current_user",
    "get_current_user_or_system",
    "get_optional_user",
    "is_auth_configured",
    "AuthUser",
]
