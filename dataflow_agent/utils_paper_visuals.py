from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_IMAGE_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_FIGURE_HINT_RE = re.compile(
    r"(figure|fig\.?|image|diagram|architecture|pipeline|framework|overview|method|"
    r"图|图片|示意|架构|框架|流程|方法|模型)",
    re.IGNORECASE,
)
_SKIP_SLIDE_RE = re.compile(r"(cover|title\s*slide|thank|acknowledg|封面|标题页|致谢|谢谢)", re.IGNORECASE)


def _clean_text(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit]


def _normalize_ref(raw: Any) -> str:
    text = str(raw or "").strip().strip('"').strip("'")
    if not text:
        return ""
    text = text.split("#", 1)[0].split("?", 1)[0].strip()
    if text.startswith("./"):
        text = text[2:]
    return text


def _resolve_markdown_image_path(mineru_root: Path, ref: str) -> Path | None:
    ref = _normalize_ref(ref)
    if not ref:
        return None
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    search_paths = [
        mineru_root / candidate,
        mineru_root / "images" / candidate.name,
    ]
    for path in search_paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _nearby_caption(lines: Sequence[str], line_index: int) -> str:
    candidates: list[str] = []
    for idx in range(line_index + 1, min(len(lines), line_index + 5)):
        text = _clean_text(lines[idx], 220)
        if text:
            candidates.append(text)
        if text and re.search(r"(figure|fig\.?|图\s*\d+|table|表\s*\d+)", text, re.IGNORECASE):
            return text
    for idx in range(max(0, line_index - 3), line_index):
        text = _clean_text(lines[idx], 220)
        if text:
            candidates.append(text)
    return candidates[0] if candidates else ""


def build_markdown_image_catalog(markdown_text: str, mineru_root: str | Path, *, limit: int = 80) -> List[Dict[str, str]]:
    root = Path(str(mineru_root or "")).expanduser()
    if not markdown_text or not root.exists():
        return []

    lines = markdown_text.splitlines()
    catalog: list[dict[str, str]] = []
    seen: set[str] = set()

    for line_index, line in enumerate(lines):
        matches = [(m.group(2), m.group(1)) for m in _MD_IMAGE_RE.finditer(line)]
        matches.extend((m.group(1), "") for m in _HTML_IMAGE_RE.finditer(line))
        for raw_ref, alt in matches:
            ref = _normalize_ref(raw_ref)
            path = _resolve_markdown_image_path(root, ref)
            if not path:
                continue
            display_ref = ref if not Path(ref).is_absolute() else path.name
            if display_ref in seen:
                continue
            seen.add(display_ref)
            caption = _nearby_caption(lines, line_index)
            nearby = " ".join(
                _clean_text(lines[idx], 160)
                for idx in range(max(0, line_index - 2), min(len(lines), line_index + 4))
                if _clean_text(lines[idx], 160)
            )
            catalog.append(
                {
                    "ref": display_ref,
                    "path": str(path),
                    "caption": _clean_text(caption or alt, 240),
                    "nearby_text": _clean_text(nearby, 420),
                }
            )
            if len(catalog) >= limit:
                return catalog
    return catalog


def format_image_catalog_for_prompt(catalog: Sequence[Dict[str, Any]], *, limit: int = 40) -> str:
    rows: list[str] = []
    for idx, item in enumerate(catalog[:limit], start=1):
        ref = _normalize_ref(item.get("ref") or item.get("src") or item.get("path"))
        if not ref:
            continue
        caption = _clean_text(item.get("caption") or item.get("nearby_text"), 180)
        rows.append(f"{idx}. ref={ref}" + (f" | caption={caption}" if caption else ""))
    return "\n".join(rows) if rows else "No extracted paper images are available."


def _catalog_lookup(catalog: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in catalog:
        ref = _normalize_ref(item.get("ref") or item.get("src") or item.get("path"))
        if not ref:
            continue
        lookup[ref] = dict(item)
        lookup[Path(ref).name] = dict(item)
    return lookup


def _extract_page_refs(page: Dict[str, Any]) -> List[str]:
    refs: list[str] = []

    def push(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key in ("asset_ref", "assetRef", "src", "path", "storage_path", "storagePath", "ref"):
                if key in value:
                    push(value.get(key))
                    return
            return
        if isinstance(value, list):
            for item in value:
                push(item)
            return
        for part in re.split(r"[,\n]+", str(value)):
            ref = _normalize_ref(part)
            if ref and ref.lower() not in {"null", "none", "n/a"} and ref not in refs:
                refs.append(ref)

    for key in ("asset_ref", "assetRef", "asset", "assets", "visual_assets", "visualAssets"):
        push(page.get(key))
    return refs


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_\-]{4,}", text.lower()) if token}


def _best_catalog_match(page: Dict[str, Any], catalog: Sequence[Dict[str, Any]], used_refs: set[str]) -> str:
    page_text = " ".join(
        [
            _clean_text(page.get("title"), 200),
            _clean_text(page.get("layout_description"), 260),
            " ".join(_clean_text(item, 120) for item in page.get("key_points") or []),
        ]
    )
    if _SKIP_SLIDE_RE.search(page_text):
        return ""

    page_tokens = _token_set(page_text)
    best_ref = ""
    best_score = 0
    for item in catalog:
        ref = _normalize_ref(item.get("ref") or item.get("src") or item.get("path"))
        if not ref or ref in used_refs:
            continue
        item_text = " ".join([str(item.get("caption") or ""), str(item.get("nearby_text") or ""), ref])
        score = len(page_tokens.intersection(_token_set(item_text)))
        if _FIGURE_HINT_RE.search(page_text) and score > 0:
            score += 1
        if score > best_score:
            best_score = score
            best_ref = ref
    return best_ref if best_score >= 2 else ""


def enrich_pagecontent_with_visual_assets(
    pagecontent: Sequence[Dict[str, Any]],
    catalog: Sequence[Dict[str, Any]],
    *,
    max_assets_per_page: int = 1,
) -> List[Dict[str, Any]]:
    if not pagecontent:
        return []
    if not catalog:
        return [dict(page) for page in pagecontent]

    lookup = _catalog_lookup(catalog)
    used_refs: set[str] = set()
    enriched: list[dict[str, Any]] = []

    for page in pagecontent:
        item = dict(page)
        selected: list[dict[str, Any]] = []
        for ref in _extract_page_refs(item):
            match = lookup.get(ref) or lookup.get(Path(ref).name)
            if not match:
                continue
            normalized_ref = _normalize_ref(match.get("ref") or ref)
            if normalized_ref in used_refs:
                continue
            selected.append({**match, "ref": normalized_ref})
            used_refs.add(normalized_ref)
            if len(selected) >= max_assets_per_page:
                break

        if not selected:
            matched_ref = _best_catalog_match(item, catalog, used_refs)
            if matched_ref:
                match = lookup.get(matched_ref) or lookup.get(Path(matched_ref).name)
                if match:
                    selected.append({**match, "ref": matched_ref})
                    used_refs.add(matched_ref)

        if selected:
            visuals = []
            for idx, visual in enumerate(selected):
                key = "main_visual" if idx == 0 else f"visual_{idx + 1}"
                ref = _normalize_ref(visual.get("ref"))
                visuals.append(
                    {
                        "key": key,
                        "label": visual.get("caption") or ("Paper Figure" if idx == 0 else f"Paper Figure {idx + 1}"),
                        "src": ref,
                        "alt": visual.get("caption") or item.get("title") or key,
                        "source_type": "paper_asset",
                    }
                )
            item["asset_ref"] = visuals[0]["src"]
            item["visual_assets"] = visuals

        enriched.append(item)

    return enriched
