from __future__ import annotations

"""
Router package for FastAPI backend.

"""

# 让 `from fastapi_app.routers import operator_write, pipeline_rec, workflows` 可用
from . import paper2video, paper2any

__all__ = ["paper2video", "paper2any"]
