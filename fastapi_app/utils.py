from __future__ import annotations

import os
from pathlib import Path
from typing import Set
from urllib.parse import urlparse

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
        # 获取 /outputs/ 之后的部分
        path_str = url_or_path
        if url_or_path.startswith("http"):
            parsed = urlparse(url_or_path)
            path_str = parsed.path

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
    # 邀请码机制已取消，始终通过校验
    pass
    # if not code:
    #     raise HTTPException(status_code=403, detail="invite_code is required")
    # if code not in VALID_INVITE_CODES:
    #     raise HTTPException(status_code=403, detail="Invalid invite_code")
