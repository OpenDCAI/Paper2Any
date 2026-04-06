from __future__ import annotations

import os
from pathlib import Path
from typing import Set
from urllib.parse import urlparse

from fastapi import HTTPException, Request

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


def get_outputs_root() -> Path:
    return (get_project_root() / "outputs").resolve()


def get_outputs_subdir(subdir: str | Path | None = None) -> Path:
    outputs_root = get_outputs_root()
    if subdir in (None, "", "."):
        return outputs_root
    return (outputs_root / Path(subdir)).resolve()


def ensure_outputs_subpath(
    path: str | Path,
    *,
    subdir: str | Path | None = None,
    must_exist: bool = False,
    allow_files: bool = True,
    allow_dirs: bool = True,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (get_project_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    allowed_root = get_outputs_subdir(subdir)
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid output path") from exc

    if must_exist and not candidate.exists():
        raise HTTPException(status_code=404, detail="Output path does not exist")

    if candidate.exists():
        if candidate.is_file() and not allow_files:
            raise HTTPException(status_code=400, detail="Expected directory path inside outputs")
        if candidate.is_dir() and not allow_dirs:
            raise HTTPException(status_code=400, detail="Expected file path inside outputs")

    return candidate


def resolve_outputs_path(
    path_or_url: str | Path,
    *,
    subdir: str | Path | None = None,
    must_exist: bool = False,
    allow_files: bool = True,
    allow_dirs: bool = True,
) -> Path:
    raw = str(path_or_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Output path is required")

    if raw.startswith(("http://", "https://")) and "/outputs/" not in raw:
        raise HTTPException(status_code=400, detail="Only /outputs resources are allowed")

    normalized = _from_outputs_url(raw)
    return ensure_outputs_subpath(
        normalized,
        subdir=subdir,
        must_exist=must_exist,
        allow_files=allow_files,
        allow_dirs=allow_dirs,
    )


def _is_local_host(host: str | None) -> bool:
    raw = (host or "").strip()
    if not raw:
        return True
    hostname = raw.split(",", 1)[0].strip().split(":", 1)[0].strip("[]").lower()
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _resolve_public_base_url(request: Request) -> str:
    xf_proto = (request.headers.get("x-forwarded-proto") or "").split(",", 1)[0].strip()
    xf_host = (request.headers.get("x-forwarded-host") or "").split(",", 1)[0].strip()
    if xf_proto and xf_host and not _is_local_host(xf_host):
        return f"{xf_proto}://{xf_host}"

    host = (request.headers.get("host") or "").split(",", 1)[0].strip()
    if host and not _is_local_host(host):
        scheme = xf_proto or request.url.scheme
        return f"{scheme}://{host}"

    return ""


def _to_outputs_url(abs_path: str, request: Request | None = None) -> str:
    """
    将绝对路径转换为浏览器可访问的完整 URL。
    默认认为所有输出文件都位于项目根目录下的 outputs/ 目录中。
    """
    project_root = get_project_root()
    outputs_root = get_outputs_root()

    log.info(f"[DEBUG] project_root: {project_root}")
    log.info(f"[DEBUG] outputs_root: {outputs_root}")
    log.info(f"[DEBUG] abs_path: {abs_path}")

    try:
        p = ensure_outputs_subpath(abs_path)
    except HTTPException as e:
        log.error(f"[ERROR] Cannot convert path outside outputs to URL: {abs_path} ({e.detail})")
        return ""

    try:
        rel = p.relative_to(outputs_root)

        # 构造 URL（优先使用公开可访问的 Host/Proto，否则降级为相对路径）
        if request is not None:
            base_url = _resolve_public_base_url(request)
            if base_url:
                url = f"{base_url}/outputs/{rel.as_posix()}"
            else:
                # 没有透传头时，用相对路径避免 http/https 混合内容问题
                url = f"/outputs/{rel.as_posix()}"
        else:
            url = f"/outputs/{rel.as_posix()}"

        log.warning(f"[DEBUG] generated URL: {url}")
        return url
    except ValueError as e:
        log.error(f"[ERROR] Path conversion failed: {e}")
        if "/outputs/" in abs_path:
            idx = abs_path.index("/outputs/")
            fallback_url = abs_path[idx:]
            log.warning(f"[WARN] Using fallback URL: {fallback_url}")
            return fallback_url
        log.error(f"[ERROR] Cannot convert path to URL: {abs_path}")
        return abs_path


def _from_outputs_url(url_or_path: str) -> str:
    """
    尝试将前端传来的 URL (包含 /outputs/) 转换回本地绝对路径。
    如果不是 URL 或者转换失败，则返回原值。
    """
    if not url_or_path or not isinstance(url_or_path, str):
        return url_or_path

    # 如果已经是绝对路径且存在，直接返回
    if os.path.isabs(url_or_path) and os.path.exists(url_or_path):
        return url_or_path

    # 简单判断是否是 http URL
    if not url_or_path.startswith("http") and not url_or_path.startswith("/outputs/"):
        return url_or_path

    # 查找 /outputs/ 的位置
    if "/outputs/" not in url_or_path:
        return url_or_path

    try:
        # 统一去掉 query / fragment，兼容 /outputs/foo.png?t=123 这类前端缓存参数
        parsed = urlparse(url_or_path)
        path_str = parsed.path or url_or_path

        if "/outputs/" in path_str:
            idx = path_str.index("/outputs/")
            # outputs/xxx/yyy
            rel_path = path_str[idx + len("/outputs/") :]
            # 去除可能的开头的 / (虽然 relative_to 不需要，但拼接待会儿用)
            rel_path = rel_path.lstrip("/")

            project_root = get_project_root()
            outputs_root = project_root / "outputs"
            abs_path = (outputs_root / rel_path).resolve()

            log.info(f"[DEBUG] Converted URL {url_or_path} to path {abs_path}")
            return str(abs_path)

    except Exception as e:
        log.warning(f"[WARN] Failed to convert URL to path: {e}")

    return url_or_path
