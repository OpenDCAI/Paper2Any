"""
JWT Authentication dependency for FastAPI.

Validates Supabase JWT tokens and extracts user information.
"""
import os
from typing import Optional
from fastapi import Header, HTTPException
from supabase import create_client, Client


# Supabase client singleton
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """Get or create Supabase client. Returns None if not configured."""
    global _supabase_client
    
    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            return None
        
        _supabase_client = create_client(supabase_url, supabase_key)
    
    return _supabase_client


class AuthUser:
    """Authenticated user information."""
    
    def __init__(self, user_id: str, email: Optional[str], phone: Optional[str]):
        self.id = user_id
        self.email = email
        self.phone = phone
    
    @property
    def identifier(self) -> str:
        """Get user identifier (email or user_id)."""
        return self.email or self.id


async def get_current_user(authorization: Optional[str] = Header(None)) -> AuthUser:
    """
    Validate JWT token and extract user information.
    
    Args:
        authorization: Authorization header with Bearer token
        
    Returns:
        AuthUser object with user information
        
    Raises:
        HTTPException: If token is invalid or missing or Supabase not configured
    """
    # Check if Supabase is configured
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Authentication service not configured"
        )
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'"
        )
    
    token = authorization.split(" ", 1)[1]
    
    try:
        # Verify token and get user
        response = supabase.auth.get_user(token)
        
        if not response or not response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        user = response.user
        
        return AuthUser(
            user_id=user.id,
            email=user.email,
            phone=user.phone
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token validation failed: {str(e)}"
        )


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[AuthUser]:
    """
    Optional authentication - returns None if no token provided or Supabase not configured.
    
    Useful for endpoints that work both with and without authentication.
    """
    # If Supabase is not configured, skip authentication
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    if not authorization:
        return None
    
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None
