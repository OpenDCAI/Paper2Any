from __future__ import annotations

import json
from typing import Any, List


_BBOX_LIST_KEYS = ("bbox_result", "result", "results", "data", "items", "boxes")


def normalize_ocr_max_tokens(api_url: str, model: str, max_tokens: int) -> int:
    """Clamp OCR token budgets to provider-supported ranges."""
    normalized = max(1, int(max_tokens))
    api_url_l = (api_url or "").lower()
    model_l = (model or "").lower()

    # DashScope's qwen-vl-ocr endpoint rejects values above 8192.
    if "dashscope.aliyuncs.com" in api_url_l and "qwen-vl-ocr" in model_l:
        return min(normalized, 8192)

    return normalized


def extract_bbox_items(payload: Any) -> List[Any]:
    """Normalize OCR/VLM parser output into the bbox-item list expected by workflows."""
    if isinstance(payload, str):
        reparsed = _parse_bbox_json(payload)
        if reparsed is None:
            return []
        return extract_bbox_items(reparsed)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in _BBOX_LIST_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate

    raw = payload.get("raw")
    if isinstance(raw, str):
        return extract_bbox_items(raw)

    return []


def _parse_bbox_json(payload: str) -> Any | None:
    text = (payload or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    if text.lower().startswith("json\n"):
        text = text[5:].strip()

    try:
        return json.loads(text)
    except Exception:
        return None
