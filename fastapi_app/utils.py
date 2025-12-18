from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from fastapi import HTTPException, Request

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

# 简单的邀请码校验：从本地文本文件加载白名单
INVITE_CODES_FILE = Path(os.getenv("INVITE_CODES_FILE", f"{get_project_root()}/invite_codes.txt"))


def _to_outputs_url(abs_path: str, request: Request | None = None) -> str:
    """
    将绝对路径转换为浏览器可访问的完整 URL。
    默认认为所有输出文件都位于项目根目录下的 outputs/ 目录中。
    """
    project_root = get_project_root()
    outputs_root = project_root / "outputs"

    log.info(f"[DEBUG] project_root: {project_root}")
    log.info(f"[DEBUG] outputs_root: {outputs_root}")
    log.info(f"[DEBUG] abs_path: {abs_path}")

    p = Path(abs_path)
    try:
        rel = p.relative_to(outputs_root)

        # 构造完整 URL（包含协议、域名和端口）
        if request is not None:
            base_url = str(request.base_url).rstrip("/")
            url = f"{base_url}/outputs/{rel.as_posix()}"
        else:
            # 降级：使用相对路径
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


def load_invite_codes() -> Set[str]:
    """
    从 invite_codes.txt 中加载邀请码列表。

    文件格式：每行一个邀请码，忽略空行和以 # 开头的注释行。
    """
    codes: Set[str] = set()
    if not INVITE_CODES_FILE.exists():
        return codes
    for line in INVITE_CODES_FILE.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if not code or code.startswith("#"):
            continue
        codes.add(code)
    return codes


VALID_INVITE_CODES = load_invite_codes()


def validate_invite_code(code: str | None) -> None:
    """
    校验邀请码是否有效。无效则抛出 403。
    """
    if not code:
        raise HTTPException(status_code=403, detail="invite_code is required")
    if code not in VALID_INVITE_CODES:
        raise HTTPException(status_code=403, detail="Invalid invite_code")
