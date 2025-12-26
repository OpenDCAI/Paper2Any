"""
Centralized configuration management using pydantic-settings.

Loads settings from environment variables with .env file support.
Use get_settings() to access configuration throughout the application.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase Configuration
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    # Rate Limiting
    daily_workflow_limit: int = 10
    daily_anonymous_limit: int = 3  # Lower limit for anonymous users

    # Existing DataFlow-Agent settings
    df_api_key: Optional[str] = None
    df_api_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not in schema
    )

    def is_supabase_configured(self) -> bool:
        """Check if Supabase credentials are configured."""
        return bool(self.supabase_url and self.supabase_anon_key)

    def is_supabase_admin_configured(self) -> bool:
        """Check if Supabase admin (service role) is configured."""
        return bool(self.supabase_url and self.supabase_service_role_key)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache for singleton pattern - settings are loaded once
    and reused across the application lifetime.
    """
    return Settings()
