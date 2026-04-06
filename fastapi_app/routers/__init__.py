from __future__ import annotations

"""
Router package for FastAPI backend.

"""

from . import (
    account,
    paper2citation,
    paper2video,
    paper2any,
    paper2ppt,
    pdf2ppt,
    image2ppt,
    kb,
    kb_embedding,
    files,
    image2drawio,
    mindmap,
    paper2drawio,
    paper2rebuttal,
)

__all__ = [
    "account",
    "paper2citation",
    "paper2video",
    "paper2any",
    "paper2ppt",
    "pdf2ppt",
    "image2ppt",
    "kb",
    "kb_embedding",
    "files",
    "image2drawio",
    "mindmap",
    "paper2drawio",
    "paper2rebuttal",
]
