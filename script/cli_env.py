from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "fastapi_app" / ".env"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _load_env_manually(env_file: Path) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_project_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    if not ENV_FILE.is_file():
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=False)
        return

    _load_env_manually(ENV_FILE)


def resolve_cli_text_credentials(api_url: str | None, api_key: str | None) -> tuple[str, str]:
    return (
        api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1"),
        api_key or os.getenv("DF_API_KEY", ""),
    )


def resolve_cli_image_credentials(
    image_api_url: str | None,
    image_api_key: str | None,
    *,
    fallback_url: str | None = None,
    fallback_key: str | None = None,
) -> tuple[str, str]:
    return (
        image_api_url or os.getenv("DF_IMAGE_API_URL", "") or fallback_url or os.getenv("DF_API_URL", "https://api.openai.com/v1"),
        image_api_key or os.getenv("DF_IMAGE_API_KEY", "") or fallback_key or os.getenv("DF_API_KEY", ""),
    )


def find_output_artifacts(base_dir: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(base_dir.rglob(pattern)))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in found:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.resolve())
    return unique
