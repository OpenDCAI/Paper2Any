"""
Application Settings - Three-tier Configuration System

This module provides a centralized configuration system with three layers:
1. Base Models: Fundamental model name definitions
2. Workflow-level: Default models for each workflow
3. Role-level: Fine-grained model assignments for specific roles

All settings can be overridden via environment variables in .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class AppSettings(BaseSettings):
    """
    Application configuration using three-tier architecture:
    Base Models + Workflow-level + Role-level

    Environment variables can override any setting by using the same name.
    Example: export PAPER2PPT_DEFAULT_MODEL=gpt-4o
    """

    # ============================================
    # Layer 1: Base Model Definitions
    # ============================================
    # Define all available model constants
    MODEL_GPT_4O: str = "gpt-4o"
    MODEL_GPT_5_1: str = "gpt-5.1"
    MODEL_CLAUDE_HAIKU: str = "claude-haiku-4-5-20251001"
    MODEL_GEMINI_PRO_IMAGE: str = "gemini-3-pro-image-preview"
    MODEL_GEMINI_FLASH_IMAGE: str = "gemini-2.5-flash-image"
    MODEL_GEMINI_FLASH: str = "gemini-2.5-flash"
    MODEL_QWEN_VL_OCR: str = "qwen-vl-ocr-2025-11-20"

    # API Configuration
    DEFAULT_LLM_API_URL: str = "http://123.129.219.111:3000/v1/"

    # ============================================
    # Layer 2: Workflow-level Default Models
    # ============================================
    # Paper2PPT Workflow
    PAPER2PPT_DEFAULT_MODEL: str = "gpt-5.1"
    PAPER2PPT_DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview"

    # PDF2PPT Workflow
    PDF2PPT_DEFAULT_MODEL: str = "gpt-4o"
    PDF2PPT_DEFAULT_IMAGE_MODEL: str = "gemini-2.5-flash-image"

    # Paper2Figure Workflow
    PAPER2FIGURE_DEFAULT_MODEL: str = "gpt-4o"
    PAPER2FIGURE_DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview"

    # Paper2Video Workflow
    PAPER2VIDEO_DEFAULT_MODEL: str = "gpt-4o"

    # Knowledge Base
    KB_EMBEDDING_MODEL: str = "gemini-2.5-flash"
    KB_CHAT_MODEL: str = "gpt-4o"

    # ============================================
    # Layer 3: Role-level Model Configuration
    # ============================================
    # Paper2PPT role-specific models
    PAPER2PPT_OUTLINE_MODEL: str = "gpt-5.1"           # Outline generation
    PAPER2PPT_CONTENT_MODEL: str = "gpt-5.1"           # Content generation
    PAPER2PPT_IMAGE_GEN_MODEL: str = "gemini-3-pro-image-preview"  # Image generation
    PAPER2PPT_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"  # VLM vision understanding
    PAPER2PPT_CHART_MODEL: str = "gpt-4o"              # Chart generation
    PAPER2PPT_DESC_MODEL: str = "gpt-5.1"              # Figure description
    PAPER2PPT_TECHNICAL_MODEL: str = "claude-haiku-4-5-20251001"  # Technical details

    # Paper2Figure role-specific models
    PAPER2FIGURE_TEXT_MODEL: str = "gpt-4o"
    PAPER2FIGURE_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
    PAPER2FIGURE_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"
    PAPER2FIGURE_CHART_MODEL: str = "gpt-4o"
    PAPER2FIGURE_DESC_MODEL: str = "gpt-5.1"
    PAPER2FIGURE_TECHNICAL_MODEL: str = "claude-haiku-4-5-20251001"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global configuration instance
settings = AppSettings()
