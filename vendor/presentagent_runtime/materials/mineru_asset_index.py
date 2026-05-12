from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_REF_RE = re.compile(r"(?:\b(fig(?:ure)?|table)|([图表]))\s*\.?\s*([0-9]+[a-z]?)", re.IGNORECASE)
_CAPTION_START_RE = re.compile(
    r"^\s*(?:\b(fig(?:ure)?|table)|([图表]))\s*\.?\s*([0-9]+[a-z]?)\s*[:：]",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "figure",
    "fig",
    "table",
}


def find_mineru_asset_matches(
    asset_ref: str,
    *,
    asset_base_dir: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Resolve a caption-like asset_ref against MinerU output assets.

    MinerU names extracted images by content hash, but keeps captions in
    content_list JSON and full.md. This function builds a small caption index
    and returns image/table assets as paths relative to asset_base_dir.
    """

    ref_text = str(asset_ref or "").strip()
    if not ref_text:
        return []

    base_dir = Path(asset_base_dir or ".").expanduser()
    candidates = _build_mineru_asset_index(base_dir)
    if not candidates:
        return []

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        score = _score_candidate(ref_text, candidate)
        if score <= 0:
            continue
        scored.append({**candidate, "match_score": round(score, 4)})

    if not scored:
        return []

    scored.sort(
        key=lambda item: (
            float(item.get("match_score") or 0.0),
            -len(str(item.get("relative_path") or "")),
        ),
        reverse=True,
    )
    best_score = float(scored[0].get("match_score") or 0.0)
    threshold = 2.5 if _extract_ref_key(ref_text) else 0.62
    if best_score < threshold:
        return []
    return scored[: max(1, int(limit))]


def _build_mineru_asset_index(base_dir: Path) -> list[dict[str, Any]]:
    base_dir = base_dir.expanduser()
    auto_dirs = _discover_auto_dirs(base_dir)
    if not auto_dirs:
        return []

    by_relative_path: dict[str, dict[str, Any]] = {}
    for auto_dir in auto_dirs:
        _index_json_assets(by_relative_path, base_dir=base_dir, auto_dir=auto_dir)
        _index_markdown_assets(by_relative_path, base_dir=base_dir, auto_dir=auto_dir)

    assets: list[dict[str, Any]] = []
    for asset in by_relative_path.values():
        captions = [caption for caption in asset.pop("_captions", []) if caption]
        caption = " ".join(dict.fromkeys(captions)).strip()
        asset["caption"] = caption
        asset["normalized_caption"] = _normalize_text(caption)
        asset["ref_keys"] = sorted(_extract_ref_keys(caption))
        assets.append(asset)
    return assets


def _discover_auto_dirs(base_dir: Path) -> list[Path]:
    roots = _auto_dirs_under(base_dir)
    if not roots:
        parent = base_dir.parent
        if parent.name == "auto" and parent.is_dir():
            roots.append(parent)
        parent_auto = parent / "auto"
        if parent_auto.is_dir():
            roots.append(parent_auto)
    if not roots:
        roots = _auto_dirs_under(base_dir.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def _auto_dirs_under(root: Path) -> list[Path]:
    roots: list[Path] = []
    if not root:
        return roots
    if root.name == "auto" and root.is_dir():
        roots.append(root)
    auto_dir = root / "auto"
    if auto_dir.is_dir():
        roots.append(auto_dir)
    if root.is_dir():
        try:
            roots.extend(child / "auto" for child in root.iterdir() if (child / "auto").is_dir())
        except OSError:
            pass
    return roots


def _index_json_assets(
    by_relative_path: dict[str, dict[str, Any]],
    *,
    base_dir: Path,
    auto_dir: Path,
) -> None:
    json_paths = list(auto_dir.glob("*_content_list.json"))
    content_v2 = auto_dir / "content_list_v2.json"
    if content_v2.is_file():
        json_paths.insert(0, content_v2)

    seen: set[Path] = set()
    for json_path in json_paths:
        if json_path in seen or not json_path.is_file():
            continue
        seen.add(json_path)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in _walk_objects(payload):
            kind = str(item.get("type") or "").strip().lower()
            if kind not in {"image", "table"}:
                continue
            raw_path = _extract_asset_path(item)
            if not raw_path:
                continue
            caption = _extract_caption(item, kind=kind)
            _add_asset(
                by_relative_path,
                base_dir=base_dir,
                auto_dir=auto_dir,
                raw_path=raw_path,
                kind=kind,
                caption=caption,
                source=str(json_path),
                page_idx=item.get("page_idx"),
            )


def _index_markdown_assets(
    by_relative_path: dict[str, dict[str, Any]],
    *,
    base_dir: Path,
    auto_dir: Path,
) -> None:
    markdown_path = auto_dir / "full.md"
    if not markdown_path.is_file():
        return
    try:
        lines = markdown_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    pending: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        image_match = _MARKDOWN_IMAGE_RE.search(line)
        if image_match:
            pending.append({"path": image_match.group(1).strip(), "labels": []})
            continue

        if not line:
            continue

        if _CAPTION_START_RE.match(line):
            for item in pending:
                caption = " ".join([*item.get("labels", []), line]).strip()
                _add_asset(
                    by_relative_path,
                    base_dir=base_dir,
                    auto_dir=auto_dir,
                    raw_path=str(item.get("path") or ""),
                    kind="image",
                    caption=caption,
                    source=str(markdown_path),
                )
            pending.clear()
            continue

        if not pending:
            continue
        if line.startswith("#") or len(line) > 220 or line.startswith("<table"):
            pending.clear()
            continue
        pending[-1].setdefault("labels", []).append(line)


def _walk_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _extract_asset_path(item: dict[str, Any]) -> str:
    for key in ("img_path", "image_path", "table_img_path", "path"):
        value = str(item.get(key) or "").strip()
        if value and Path(value).suffix.lower() in _IMAGE_EXTENSIONS:
            return value

    content = item.get("content")
    if isinstance(content, dict):
        for key in ("image_source", "table_source"):
            source = content.get(key)
            if isinstance(source, dict):
                value = str(source.get("path") or "").strip()
                if value and Path(value).suffix.lower() in _IMAGE_EXTENSIONS:
                    return value
    return ""


def _extract_caption(item: dict[str, Any], *, kind: str) -> str:
    if kind == "table":
        parts = [
            _text_from_value(item.get("table_caption")),
            _text_from_value(item.get("table_footnote")),
        ]
    else:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        parts = [
            _text_from_value(item.get("image_caption")),
            _text_from_value(item.get("image_footnote")),
            _text_from_value(content.get("image_caption") if isinstance(content, dict) else ""),
            _text_from_value(content.get("image_footnote") if isinstance(content, dict) else ""),
        ]
    return " ".join(part for part in parts if part).strip()


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text_from_value(item) for item in value if _text_from_value(item)).strip()
    if isinstance(value, dict):
        if "content" in value:
            return _text_from_value(value.get("content"))
        return " ".join(_text_from_value(item) for item in value.values() if _text_from_value(item)).strip()
    return str(value).strip()


def _add_asset(
    by_relative_path: dict[str, dict[str, Any]],
    *,
    base_dir: Path,
    auto_dir: Path,
    raw_path: str,
    kind: str,
    caption: str,
    source: str,
    page_idx: Any = None,
) -> None:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return

    path = Path(raw_path).expanduser()
    absolute_path = path if path.is_absolute() else auto_dir / path
    try:
        resolved_absolute = absolute_path.resolve()
    except OSError:
        resolved_absolute = absolute_path
    if not resolved_absolute.exists():
        return

    relative_path = _relative_path(resolved_absolute, base_dir)
    existing = by_relative_path.get(relative_path)
    if existing is None:
        existing = {
            "relative_path": relative_path,
            "path": relative_path,
            "absolute_path": str(resolved_absolute),
            "raw_mineru_path": raw_path,
            "document_object_type": kind,
            "_captions": [],
            "source_files": [],
        }
        by_relative_path[relative_path] = existing

    if caption:
        existing.setdefault("_captions", []).append(caption.strip())
    if source and source not in existing.setdefault("source_files", []):
        existing["source_files"].append(source)
    if page_idx is not None and "page_idx" not in existing:
        existing["page_idx"] = page_idx
    if kind == "table":
        existing["document_object_type"] = "table"


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _score_candidate(asset_ref: str, candidate: dict[str, Any]) -> float:
    normalized_ref = _normalize_text(asset_ref)
    normalized_caption = str(candidate.get("normalized_caption") or "")
    if not normalized_ref or not normalized_caption:
        return 0.0

    ref_key = _extract_ref_key(asset_ref)
    candidate_keys = set(candidate.get("ref_keys") or [])
    score = 0.0
    if ref_key:
        if ref_key in candidate_keys:
            score += 4.0
        elif candidate_keys:
            return 0.0

    if normalized_ref in normalized_caption:
        score += 1.5

    ref_tokens = _tokens(normalized_ref)
    caption_tokens = _tokens(normalized_caption)
    if ref_tokens and caption_tokens:
        overlap = ref_tokens & caption_tokens
        score += len(overlap) / max(1, len(ref_tokens))
        score += 0.25 * (len(overlap) / max(1, len(caption_tokens)))
    return score


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("：", ":")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) > 2 and token not in _STOPWORDS}


def _extract_ref_key(value: str) -> str:
    keys = _extract_ref_keys(value)
    return sorted(keys)[0] if keys else ""


def _extract_ref_keys(value: str) -> set[str]:
    keys: set[str] = set()
    for match in _REF_RE.finditer(str(value or "")):
        english_kind = (match.group(1) or "").lower()
        chinese_kind = match.group(2) or ""
        normalized_kind = "table" if english_kind == "table" or chinese_kind == "表" else "figure"
        keys.add(f"{normalized_kind}:{match.group(3).lower()}")
    return keys
