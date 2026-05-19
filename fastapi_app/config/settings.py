"""
Application Settings - Three-tier Configuration System

This module provides a centralized configuration system with three layers:
1. Base Models: Fundamental model name definitions
2. Workflow-level: Default models for each workflow
3. Role-level: Fine-grained model assignments for specific roles

All settings can be overridden via environment variables in .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from typing import Optional

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
    PAPER2ANY_CONFIG_MODE: str = "advanced"
    SIMPLE_TEXT_API_URL: str = ""
    SIMPLE_TEXT_API_KEY: str = ""
    SIMPLE_IMAGE_API_URL: str = ""
    SIMPLE_IMAGE_API_KEY: str = ""
    SIMPLE_OCR_API_URL: str = ""
    SIMPLE_OCR_API_KEY: str = ""
    SIMPLE_TEXT_MODEL: str = "gpt-4o"
    SIMPLE_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
    SIMPLE_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"
    SIMPLE_EMBEDDING_MODEL: str = "text-embedding-3-small"

    DEFAULT_LLM_API_URL: str = "http://123.129.219.111:3000/v1/"
    DF_API_URL: str = "http://123.129.219.111:3000/v1"
    DF_API_KEY: str = ""
    DF_IMAGE_API_URL: str = ""
    DF_IMAGE_API_KEY: str = ""
    PAPER2ANY_MANAGED_API_URL: str = ""
    PAPER2ANY_MANAGED_API_KEY: str = ""
    PAPER2ANY_MANAGED_IMAGE_API_URL: str = ""
    PAPER2ANY_MANAGED_IMAGE_API_KEY: str = ""
    PAPER2PPT_MANAGED_API_URL: str = ""
    PAPER2PPT_MANAGED_API_KEY: str = ""
    PAPER2PPT_MANAGED_IMAGE_API_URL: str = ""
    PAPER2PPT_MANAGED_IMAGE_API_KEY: str = ""
    PPT2POLISH_MANAGED_API_URL: str = ""
    PPT2POLISH_MANAGED_API_KEY: str = ""
    PPT2POLISH_MANAGED_IMAGE_API_URL: str = ""
    PPT2POLISH_MANAGED_IMAGE_API_KEY: str = ""
    PDF2PPT_MANAGED_API_URL: str = ""
    PDF2PPT_MANAGED_API_KEY: str = ""
    PDF2PPT_MANAGED_IMAGE_API_URL: str = ""
    PDF2PPT_MANAGED_IMAGE_API_KEY: str = ""
    IMAGE2PPT_MANAGED_API_URL: str = ""
    IMAGE2PPT_MANAGED_API_KEY: str = ""
    IMAGE2PPT_MANAGED_IMAGE_API_URL: str = ""
    IMAGE2PPT_MANAGED_IMAGE_API_KEY: str = ""
    PAPER2DRAWIO_MANAGED_API_URL: str = ""
    PAPER2DRAWIO_MANAGED_API_KEY: str = ""
    PAPER2DRAWIO_MANAGED_IMAGE_API_URL: str = ""
    PAPER2DRAWIO_MANAGED_IMAGE_API_KEY: str = ""
    PAPER2POSTER_MANAGED_API_URL: str = ""
    PAPER2POSTER_MANAGED_API_KEY: str = ""
    PAPER2POSTER_MANAGED_IMAGE_API_URL: str = ""
    PAPER2POSTER_MANAGED_IMAGE_API_KEY: str = ""
    PAPER2VIDEO_MANAGED_API_URL: str = ""
    PAPER2VIDEO_MANAGED_API_KEY: str = ""
    PAPER2VIDEO_MANAGED_IMAGE_API_URL: str = ""
    PAPER2VIDEO_MANAGED_IMAGE_API_KEY: str = ""
    KB_MANAGED_API_URL: str = ""
    KB_MANAGED_API_KEY: str = ""
    KB_MANAGED_IMAGE_API_URL: str = ""
    KB_MANAGED_IMAGE_API_KEY: str = ""
    KB_DEEPRESEARCH_MANAGED_API_URL: str = ""
    KB_DEEPRESEARCH_MANAGED_API_KEY: str = ""
    KB_DEEPRESEARCH_MANAGED_IMAGE_API_URL: str = ""
    KB_DEEPRESEARCH_MANAGED_IMAGE_API_KEY: str = ""
    PAPER2REBUTTAL_MANAGED_API_URL: str = ""
    PAPER2REBUTTAL_MANAGED_API_KEY: str = ""
    PAPER2REBUTTAL_MANAGED_IMAGE_API_URL: str = ""
    PAPER2REBUTTAL_MANAGED_IMAGE_API_KEY: str = ""
    IMAGE_PLAYGROUND_MANAGED_API_URL: str = ""
    IMAGE_PLAYGROUND_MANAGED_API_KEY: str = ""
    IMAGE_PLAYGROUND_MANAGED_IMAGE_API_URL: str = ""
    IMAGE_PLAYGROUND_MANAGED_IMAGE_API_KEY: str = ""
    LLM_VERIFY_TIMEOUT_SECONDS: int = 25
    LLM_VERIFY_MAX_TOKENS: int = 32
    APP_BILLING_MODE: str = "paid"
    BILLING_PRICING_CONFIG_PATH: str = str(_project_root() / "fastapi_app" / "config" / "pricing.yaml")
    GUEST_USAGE_DB_PATH: str = str(_project_root() / "outputs" / "system" / "guest_quota.sqlite3")
    SECURITY_RATE_LIMIT_ENABLED: bool = True
    SECURITY_TRUST_PROXY_HEADERS: bool = False
    FILE_ACCESS_URL_TTL_SECONDS: int = 900
    FILE_ACCESS_TOKEN_SECRET: str = ""
    SECURITY_BLOCKED_PUBLIC_OUTPUT_PREFIXES: str = "kb_data,kb_outputs,kb_exports,system"
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    CORS_ALLOW_ORIGIN_REGEX: str = ""
    POINTS_PURCHASE_URL: str = ""
    POINTS_REDEEM_CODE_FILE_10: str = str(_project_root() / "data" / "redeem_codes" / "points_10.txt")
    POINTS_REDEEM_CODE_FILE_50: str = str(_project_root() / "data" / "redeem_codes" / "points_50.txt")
    POINTS_REDEEM_CODE_FILE_100: str = str(_project_root() / "data" / "redeem_codes" / "points_100.txt")
    DEFAULT_SEARCH_API_KEY: str = ""
    DEFAULT_GOOGLE_CSE_ID: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_POSTGREST_TIMEOUT_SECONDS: float = 8.0
    BILLING_QUOTA_CACHE_TTL_SECONDS: int = 10
    BILLING_QUOTA_STALE_TTL_SECONDS: int = 300

    # RMBG-2.0 background removal model path
    RMBG_MODEL_PATH: str = str(_project_root() / "models" / "RMBG-2.0")

    # Paper2Drawio SAM3 + OCR service configuration
    # 阿里云相关真实 key 统一放在 fastapi_app/.env，本文件不再写死密钥。
    PAPER2DRAWIO_SAM3_CHECKPOINT_PATH: str = str(_project_root() / "models" / "sam3" / "sam3.pt")
    PAPER2DRAWIO_SAM3_BPE_PATH: str = str(_project_root() / "models" / "sam3" / "bpe_simple_vocab_16e6.txt.gz")
    PAPER2DRAWIO_OCR_API_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    PAPER2DRAWIO_OCR_API_KEY: str = ""
    PAPER2DRAWIO_SEGMENT_HINT_API_URL: str = ""
    PAPER2DRAWIO_SEGMENT_HINT_API_KEY: str = ""
    PAPER2DRAWIO_SEGMENT_HINT_VLM_MODEL: str = "gpt-4o"
    PAPER2DRAWIO_SEGMENT_HINT_TIMEOUT: int = 120
    PAPER2PPT_SEGMENT_HINT_API_URL: str = ""
    PAPER2PPT_SEGMENT_HINT_API_KEY: str = ""
    PAPER2PPT_SEGMENT_HINT_VLM_MODEL: str = "gpt-4o"
    PAPER2PPT_SEGMENT_HINT_TIMEOUT: int = 120
    MINERU_API_BASE_URL: str = "https://mineru.net/api/v4"
    MINERU_API_KEY: str = ""
    MINERU_API_MODEL_VERSION: str = "vlm"
    MINERU_API_POLL_INTERVAL_SECONDS: int = 5
    MINERU_API_TIMEOUT_SECONDS: int = 900

    # Paper2Citation optional websearch-LLM enrichment
    PAPER2CITATION_WEBSEARCH_ENABLED: bool = False
    PAPER2CITATION_WEBSEARCH_API_URL: str = ""
    PAPER2CITATION_WEBSEARCH_API_KEY: str = ""
    PAPER2CITATION_WEBSEARCH_MODEL: str = ""
    PAPER2CITATION_WEBSEARCH_TOOL: str = "web_search_preview"
    PAPER2CITATION_WEBSEARCH_TIMEOUT_SECONDS: int = 45
    PAPER2CITATION_WEBSEARCH_DISABLE_SECONDS: int = 600
    PAPER2CITATION_WEBSEARCH_MAX_AUTHORS: int = 24
    PAPER2CITATION_WEBSEARCH_MAX_OUTPUT_TOKENS: int = 1200

    # ============================================
    # Layer 2: Workflow-level Default Models
    # ============================================
    # Paper2PPT Workflow
    PAPER2PPT_DEFAULT_MODEL: str = "gpt-5.1"
    PAPER2PPT_DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview"

    # PDF2PPT Workflow
    PDF2PPT_DEFAULT_MODEL: str = "gpt-4o"
    PDF2PPT_DEFAULT_IMAGE_MODEL: str = "gemini-2.5-flash-image"

    # Image2PPT Workflow
    IMAGE2PPT_DEFAULT_MODEL: str = "gpt-4o"
    IMAGE2PPT_DEFAULT_IMAGE_MODEL: str = "gemini-2.5-flash-image"

    # Paper2Figure Workflow
    PAPER2FIGURE_DEFAULT_MODEL: str = "gpt-4o"
    PAPER2FIGURE_DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview"

    # Paper2Video Workflow
    PAPER2VIDEO_DEFAULT_MODEL: str = "gpt-4o"
    PAPER2VIDEO_TTS_MODEL: str = "cosyvoice-v3-flash"
    PAPER2VIDEO_TALKING_MODEL: str = "liveportrait"

    # Paper2Drawio Workflow
    PAPER2DRAWIO_DEFAULT_MODEL: str = "gpt-5.4"
    PAPER2DRAWIO_VLM_MODEL: str = "gpt-4o"
    PAPER2DRAWIO_ENABLE_VLM_VALIDATION: bool = False

    # Image2Drawio Workflow
    IMAGE2DRAWIO_DEFAULT_MODEL: str = "gpt-4o"
    IMAGE2DRAWIO_DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
    IMAGE2DRAWIO_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"

    # Knowledge Base
    KB_EMBEDDING_MODEL: str = "gemini-2.5-flash"
    KB_CHAT_MODEL: str = "gpt-4o"

    # MindMap / Poster / Rebuttal
    MINDMAP_DEFAULT_MODEL: str = "gpt-4o"
    PAPER2POSTER_DEFAULT_MODEL: str = "gpt-4o"
    PAPER2POSTER_VISION_MODEL: str = "gpt-4o"
    PAPER2REBUTTAL_DEFAULT_MODEL: str = "gpt-4o"
    IMAGE_PLAYGROUND_DEFAULT_IMAGE_MODEL: str = "gemini-3.1-flash-image-preview"

    # ============================================
    # Layer 3: Role-level Model Configuration
    # ============================================
    # Paper2PPT role-specific models
    PAPER2PPT_OUTLINE_MODEL: str = "gpt-5.1"           # Outline generation
    PAPER2PPT_CONTENT_MODEL: str = "gpt-5.1"           # Content generation
    PAPER2PPT_IMAGE_GEN_MODEL: str = "gemini-3-pro-image-preview"  # Image generation
    PAPER2PPT_MASK_EDIT_IMAGE_MODEL: str = ""
    PAPER2PPT_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"  # VLM vision understanding
    PAPER2PPT_VLM_TIMEOUT_SECONDS: int = 45
    PAPER2PPT_CHART_MODEL: str = "gpt-4o"              # Chart generation
    PAPER2PPT_DESC_MODEL: str = "gpt-5.1"              # Figure description
    PAPER2PPT_TECHNICAL_MODEL: str = "claude-haiku-4-5-20251001"  # Technical details

    # Paper2Figure role-specific models
    PAPER2FIGURE_TEXT_MODEL: str = "gpt-4o"
    PAPER2FIGURE_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
    PAPER2FIGURE_VLM_MODEL: str = "qwen-vl-ocr-2025-11-20"
    PAPER2FIGURE_CHART_MODEL: str = "gpt-4o"
    PAPER2FIGURE_DESC_MODEL: str = "gpt-5.1"
    PAPER2FIGURE_REF_IMG_DESC_MODEL: str = "gpt-4o"
    PAPER2FIGURE_TECHNICAL_MODEL: str = "gpt-5.4"

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _apply_simple_mode(settings_obj: AppSettings) -> AppSettings:
    mode = str(getattr(settings_obj, "PAPER2ANY_CONFIG_MODE", "") or "").strip().lower()
    if mode != "simple":
        return settings_obj

    text_api_url = _first_non_empty(
        settings_obj.SIMPLE_TEXT_API_URL,
        settings_obj.DF_API_URL,
        settings_obj.DEFAULT_LLM_API_URL,
    )
    text_api_key = _first_non_empty(
        settings_obj.SIMPLE_TEXT_API_KEY,
        settings_obj.DF_API_KEY,
    )
    image_api_url = _first_non_empty(
        settings_obj.SIMPLE_IMAGE_API_URL,
        settings_obj.DF_IMAGE_API_URL,
        text_api_url,
    )
    image_api_key = _first_non_empty(
        settings_obj.SIMPLE_IMAGE_API_KEY,
        settings_obj.DF_IMAGE_API_KEY,
        text_api_key,
    )
    ocr_api_url = _first_non_empty(
        settings_obj.SIMPLE_OCR_API_URL,
        settings_obj.PAPER2DRAWIO_OCR_API_URL,
        text_api_url,
    )
    ocr_api_key = _first_non_empty(
        settings_obj.SIMPLE_OCR_API_KEY,
        settings_obj.PAPER2DRAWIO_OCR_API_KEY,
        text_api_key,
    )

    text_model = _first_non_empty(settings_obj.SIMPLE_TEXT_MODEL, "gpt-4o")
    image_model = _first_non_empty(settings_obj.SIMPLE_IMAGE_MODEL, "gemini-3-pro-image-preview")
    vlm_model = _first_non_empty(settings_obj.SIMPLE_VLM_MODEL, "qwen-vl-ocr-2025-11-20")
    embedding_model = _first_non_empty(settings_obj.SIMPLE_EMBEDDING_MODEL, "text-embedding-3-small")

    settings_obj.DEFAULT_LLM_API_URL = text_api_url or settings_obj.DEFAULT_LLM_API_URL
    settings_obj.DF_API_URL = text_api_url or settings_obj.DF_API_URL
    settings_obj.DF_API_KEY = text_api_key or settings_obj.DF_API_KEY
    settings_obj.DF_IMAGE_API_URL = image_api_url or settings_obj.DF_IMAGE_API_URL
    settings_obj.DF_IMAGE_API_KEY = image_api_key or settings_obj.DF_IMAGE_API_KEY
    settings_obj.PAPER2DRAWIO_OCR_API_URL = ocr_api_url or settings_obj.PAPER2DRAWIO_OCR_API_URL
    settings_obj.PAPER2DRAWIO_OCR_API_KEY = ocr_api_key or settings_obj.PAPER2DRAWIO_OCR_API_KEY

    for scope in (
        "PAPER2ANY",
        "PAPER2PPT",
        "PPT2POLISH",
        "PDF2PPT",
        "IMAGE2PPT",
        "PAPER2DRAWIO",
        "PAPER2POSTER",
        "PAPER2VIDEO",
        "KB",
        "KB_DEEPRESEARCH",
        "PAPER2REBUTTAL",
    ):
        setattr(settings_obj, f"{scope}_MANAGED_API_URL", text_api_url)
        setattr(settings_obj, f"{scope}_MANAGED_API_KEY", text_api_key)
        setattr(settings_obj, f"{scope}_MANAGED_IMAGE_API_URL", image_api_url)
        setattr(settings_obj, f"{scope}_MANAGED_IMAGE_API_KEY", image_api_key)

    settings_obj.PAPER2DRAWIO_SEGMENT_HINT_API_URL = text_api_url
    settings_obj.PAPER2DRAWIO_SEGMENT_HINT_API_KEY = text_api_key
    settings_obj.PAPER2PPT_SEGMENT_HINT_API_URL = text_api_url
    settings_obj.PAPER2PPT_SEGMENT_HINT_API_KEY = text_api_key

    settings_obj.PAPER2PPT_DEFAULT_MODEL = text_model
    settings_obj.PAPER2PPT_DEFAULT_IMAGE_MODEL = image_model
    settings_obj.PDF2PPT_DEFAULT_MODEL = text_model
    settings_obj.PDF2PPT_DEFAULT_IMAGE_MODEL = image_model
    settings_obj.IMAGE2PPT_DEFAULT_MODEL = text_model
    settings_obj.IMAGE2PPT_DEFAULT_IMAGE_MODEL = image_model
    settings_obj.PAPER2FIGURE_DEFAULT_MODEL = text_model
    settings_obj.PAPER2FIGURE_DEFAULT_IMAGE_MODEL = image_model
    settings_obj.PAPER2VIDEO_DEFAULT_MODEL = text_model
    settings_obj.PAPER2VIDEO_TTS_MODEL = settings_obj.PAPER2VIDEO_TTS_MODEL or "cosyvoice-v3-flash"
    settings_obj.PAPER2VIDEO_TALKING_MODEL = settings_obj.PAPER2VIDEO_TALKING_MODEL or "liveportrait"
    settings_obj.PAPER2DRAWIO_DEFAULT_MODEL = text_model
    settings_obj.PAPER2DRAWIO_VLM_MODEL = vlm_model
    settings_obj.IMAGE2DRAWIO_DEFAULT_MODEL = text_model
    settings_obj.IMAGE2DRAWIO_DEFAULT_IMAGE_MODEL = image_model
    settings_obj.IMAGE2DRAWIO_VLM_MODEL = vlm_model
    settings_obj.KB_CHAT_MODEL = text_model
    settings_obj.KB_EMBEDDING_MODEL = embedding_model
    settings_obj.MINDMAP_DEFAULT_MODEL = text_model
    settings_obj.PAPER2POSTER_DEFAULT_MODEL = text_model
    settings_obj.PAPER2POSTER_VISION_MODEL = text_model
    settings_obj.PAPER2REBUTTAL_DEFAULT_MODEL = text_model

    settings_obj.PAPER2PPT_OUTLINE_MODEL = text_model
    settings_obj.PAPER2PPT_CONTENT_MODEL = text_model
    settings_obj.PAPER2PPT_IMAGE_GEN_MODEL = image_model
    settings_obj.PAPER2PPT_MASK_EDIT_IMAGE_MODEL = image_model
    settings_obj.PAPER2PPT_VLM_MODEL = vlm_model
    settings_obj.PAPER2PPT_CHART_MODEL = text_model
    settings_obj.PAPER2PPT_DESC_MODEL = text_model
    settings_obj.PAPER2PPT_TECHNICAL_MODEL = text_model

    settings_obj.PAPER2FIGURE_TEXT_MODEL = text_model
    settings_obj.PAPER2FIGURE_IMAGE_MODEL = image_model
    settings_obj.PAPER2FIGURE_VLM_MODEL = vlm_model
    settings_obj.PAPER2FIGURE_CHART_MODEL = text_model
    settings_obj.PAPER2FIGURE_DESC_MODEL = text_model
    settings_obj.PAPER2FIGURE_REF_IMG_DESC_MODEL = text_model
    settings_obj.PAPER2FIGURE_TECHNICAL_MODEL = text_model

    settings_obj.PAPER2CITATION_WEBSEARCH_MODEL = settings_obj.PAPER2CITATION_WEBSEARCH_MODEL or text_model
    return settings_obj


# Global configuration instance
settings = _apply_simple_mode(AppSettings())
