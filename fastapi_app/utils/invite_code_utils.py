"""
邀请码校验工具函数
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from fastapi import HTTPException

from dataflow_agent.utils import get_project_root

# 简单的邀请码校验：从本地文本文件加载白名单
INVITE_CODES_FILE = Path(os.getenv("INVITE_CODES_FILE", f"{get_project_root()}/invite_codes.txt"))


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
