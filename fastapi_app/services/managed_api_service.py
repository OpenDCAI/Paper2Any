from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from fastapi_app.config.pricing import get_points_purchase_url, get_pricing_config, get_redeem_code_files
from fastapi_app.config.settings import settings

_SCOPE_TO_SETTING_NAMES: dict[str, tuple[str, str]] = {
    "paper2any": ("PAPER2ANY_MANAGED_API_URL", "PAPER2ANY_MANAGED_API_KEY"),
    "paper2ppt": ("PAPER2PPT_MANAGED_API_URL", "PAPER2PPT_MANAGED_API_KEY"),
    "ppt2polish": ("PPT2POLISH_MANAGED_API_URL", "PPT2POLISH_MANAGED_API_KEY"),
    "pdf2ppt": ("PDF2PPT_MANAGED_API_URL", "PDF2PPT_MANAGED_API_KEY"),
    "image2ppt": ("IMAGE2PPT_MANAGED_API_URL", "IMAGE2PPT_MANAGED_API_KEY"),
    "paper2drawio": ("PAPER2DRAWIO_MANAGED_API_URL", "PAPER2DRAWIO_MANAGED_API_KEY"),
    "paper2poster": ("PAPER2POSTER_MANAGED_API_URL", "PAPER2POSTER_MANAGED_API_KEY"),
    "paper2video": ("PAPER2VIDEO_MANAGED_API_URL", "PAPER2VIDEO_MANAGED_API_KEY"),
    "kb": ("KB_MANAGED_API_URL", "KB_MANAGED_API_KEY"),
    "kb_deepresearch": ("KB_DEEPRESEARCH_MANAGED_API_URL", "KB_DEEPRESEARCH_MANAGED_API_KEY"),
    "paper2rebuttal": ("PAPER2REBUTTAL_MANAGED_API_URL", "PAPER2REBUTTAL_MANAGED_API_KEY"),
}
_SCOPE_TO_IMAGE_SETTING_NAMES: dict[str, tuple[str, str]] = {
    scope: (
        api_url_name.replace("_MANAGED_API_URL", "_MANAGED_IMAGE_API_URL"),
        api_key_name.replace("_MANAGED_API_KEY", "_MANAGED_IMAGE_API_KEY"),
    )
    for scope, (api_url_name, api_key_name) in _SCOPE_TO_SETTING_NAMES.items()
}


def _normalize_scope(scope: str | None) -> str:
    return (scope or "").strip().lower()


def _normalize_api_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def _normalize_api_key(value: str | None) -> str:
    return (value or "").strip()


def _normalize_model_name(value: str | None) -> str:
    return (value or "").strip()


def _get_default_managed_llm_credentials() -> tuple[str, str]:
    return (
        _normalize_api_url(settings.DF_API_URL or settings.DEFAULT_LLM_API_URL),
        _normalize_api_key(settings.DF_API_KEY),
    )


def _get_default_managed_image_credentials() -> tuple[str, str]:
    return (
        _normalize_api_url(settings.DF_IMAGE_API_URL),
        _normalize_api_key(settings.DF_IMAGE_API_KEY),
    )


def _get_scoped_managed_credentials(
    scope: str | None,
    *,
    image: bool,
) -> tuple[str, str, tuple[str, str] | None]:
    normalized_scope = _normalize_scope(scope)
    if not normalized_scope:
        return "", "", None

    mapping = _SCOPE_TO_IMAGE_SETTING_NAMES if image else _SCOPE_TO_SETTING_NAMES
    setting_names = mapping.get(normalized_scope)
    if setting_names is None:
        return "", "", None

    api_url_name, api_key_name = setting_names
    api_url = _normalize_api_url(getattr(settings, api_url_name, ""))
    api_key = _normalize_api_key(getattr(settings, api_key_name, ""))
    return api_url, api_key, setting_names


def _get_scoped_managed_llm_credentials(scope: str | None) -> tuple[str, str, tuple[str, str] | None]:
    return _get_scoped_managed_credentials(scope, image=False)


def _get_scoped_managed_image_credentials(scope: str | None) -> tuple[str, str, tuple[str, str] | None]:
    return _get_scoped_managed_credentials(scope, image=True)


def _pick_first_complete_credentials(*candidates: tuple[str, str]) -> tuple[str, str]:
    for api_url, api_key in candidates:
        if api_url and api_key:
            return api_url, api_key
    return "", ""


def _get_any_configured_managed_llm_credentials() -> tuple[str, str]:
    for scope in _SCOPE_TO_SETTING_NAMES:
        api_url, api_key, _ = _get_scoped_managed_llm_credentials(scope)
        if api_url and api_key:
            return api_url, api_key
    return _get_default_managed_llm_credentials()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_optional_path(value: str | None) -> Path | None:
    raw = (value or "").strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_project_root() / path).resolve()
    return path


def has_points_redeem_catalog() -> bool:
    for path_value in get_redeem_code_files().values():
        path = _resolve_optional_path(path_value)
        if path and path.is_file():
            return True
    return False


def get_billing_mode() -> str:
    mode = (settings.APP_BILLING_MODE or "paid").strip().lower()
    if mode not in {"paid", "free"}:
        return "paid"
    return mode


def is_free_billing_mode() -> bool:
    return get_billing_mode() == "free"


def is_user_api_config_required() -> bool:
    return not is_free_billing_mode()


def get_managed_llm_credentials(*, required: bool = True, scope: str | None = None) -> tuple[str, str]:
    scoped_api_url, scoped_api_key, setting_names = _get_scoped_managed_llm_credentials(scope)
    default_api_url, default_api_key = _get_default_managed_llm_credentials()

    api_url = scoped_api_url or default_api_url
    api_key = scoped_api_key or default_api_key

    if required and (not api_url or not api_key):
        normalized_scope = _normalize_scope(scope)
        scope_hint = ""
        if setting_names is not None:
            api_url_name, api_key_name = setting_names
            scope_hint = (
                f" for scope '{normalized_scope}'"
                f" ({api_url_name} / {api_key_name}, fallback DF_API_URL / DF_API_KEY)"
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "Managed billing mode is enabled, but backend managed API credentials are not configured"
                f"{scope_hint}"
            ),
        )

    return api_url, api_key


def get_managed_image_credentials(*, required: bool = True, scope: str | None = None) -> tuple[str, str]:
    scoped_image_api_url, scoped_image_api_key, image_setting_names = _get_scoped_managed_image_credentials(scope)
    default_image_api_url, default_image_api_key = _get_default_managed_image_credentials()
    scoped_text_api_url, scoped_text_api_key, text_setting_names = _get_scoped_managed_llm_credentials(scope)
    default_text_api_url, default_text_api_key = _get_default_managed_llm_credentials()

    api_url, api_key = _pick_first_complete_credentials(
        (scoped_image_api_url, scoped_image_api_key),
        (default_image_api_url, default_image_api_key),
        (scoped_text_api_url, scoped_text_api_key),
        (default_text_api_url, default_text_api_key),
    )

    if required and (not api_url or not api_key):
        normalized_scope = _normalize_scope(scope)
        scope_hint = ""
        if image_setting_names is not None or text_setting_names is not None:
            hints: list[str] = []
            if image_setting_names is not None:
                hints.extend(image_setting_names)
            if text_setting_names is not None:
                hints.extend(text_setting_names)
            scope_hint = (
                f" for scope '{normalized_scope}'"
                f" ({', '.join(hints)}, fallback DF_IMAGE_API_URL / DF_IMAGE_API_KEY, then DF_API_URL / DF_API_KEY)"
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "Managed billing mode is enabled, but backend managed image-generation credentials are not configured"
                f"{scope_hint}"
            ),
        )

    return api_url, api_key


def resolve_llm_credentials(
    chat_api_url: str | None,
    api_key: str | None,
    *,
    scope: str | None = None,
) -> tuple[str, str]:
    if is_free_billing_mode():
        return get_managed_llm_credentials(required=True, scope=scope)
    return _normalize_api_url(chat_api_url), _normalize_api_key(api_key)


def resolve_image_generation_credentials(
    chat_api_url: str | None,
    api_key: str | None,
    *,
    scope: str | None = None,
) -> tuple[str, str]:
    if is_free_billing_mode():
        return get_managed_image_credentials(required=True, scope=scope)
    return _normalize_api_url(chat_api_url), _normalize_api_key(api_key)


def resolve_model_name(
    requested_model: str | None,
    *,
    managed_default: str | None,
    fallback_default: str | None = None,
) -> str:
    """
    Resolve a workflow model name under the current billing mode.

    In free/managed mode we intentionally ignore any client-provided model and
    always use the backend-managed default from .env. In paid mode we still
    respect the client model and only fall back when it is empty.
    """
    managed_value = _normalize_model_name(managed_default)
    fallback_value = _normalize_model_name(fallback_default)
    if is_free_billing_mode():
        return managed_value or fallback_value
    requested_value = _normalize_model_name(requested_model)
    return requested_value or managed_value or fallback_value


def get_runtime_billing_config() -> dict:
    pricing = get_pricing_config()
    managed_api_url, managed_api_key = _get_any_configured_managed_llm_credentials()
    billing = pricing.get("billing", {})
    return {
        "billing_mode": get_billing_mode(),
        "user_api_config_required": is_user_api_config_required(),
        "model_selection_locked": is_free_billing_mode(),
        "managed_api_enabled": bool(managed_api_url and managed_api_key),
        "managed_api_url": managed_api_url,
        "server_side_billing_enforced": True,
        "workflow_costs": pricing.get("workflows", {}),
        "guest_daily_limit": int(billing.get("guest_daily_limit", 0)),
        "signup_bonus_points": int(billing.get("signup_bonus_points", 0)),
        "daily_grant_points": int(billing.get("daily_grant_points", 5)),
        "daily_grant_balance_cap": int(billing.get("daily_grant_balance_cap", 15)),
        "referral_inviter_points": int(billing.get("referral_inviter_points", 5)),
        "referral_invitee_points": int(billing.get("referral_invitee_points", 0)),
        "points_purchase_url": get_points_purchase_url(),
        "points_redeem_enabled": has_points_redeem_catalog(),
    }
