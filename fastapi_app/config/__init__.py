"""
Configuration module for Paper2Any FastAPI application.

Provides centralized configuration management for:
- Model names (LLM, Image, VLM, etc.)
- Workflow-specific model defaults
- Role-specific model assignments
- API URLs and keys

Usage:
    from fastapi_app.config import settings

    # Access configuration
    model_name = settings.PAPER2PPT_DEFAULT_MODEL
"""

from .settings import settings, AppSettings

__all__ = ['settings', 'AppSettings']
