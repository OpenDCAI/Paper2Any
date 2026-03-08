"""Utilities package for dataflow_agent.

保持向后兼容，同时避免在 import 阶段就强制拉起 `utils_common` 的重依赖。
常用的 `get_project_root` 直接在这里提供；其余符号按需惰性转发。
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def __getattr__(name: str):
    if name == "ImageVersionManager":
        from dataflow_agent.utils.version_manager import ImageVersionManager
        return ImageVersionManager

    from dataflow_agent import utils_common as _utils_common
    if hasattr(_utils_common, name):
        return getattr(_utils_common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_project_root", "ImageVersionManager"]
