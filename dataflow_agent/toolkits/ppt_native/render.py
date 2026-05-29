from __future__ import annotations

import json
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape as xml_escape

from dataflow_agent.logger import get_logger

from .svg_to_pptx.pptx_builder import create_pptx_with_native_svg

log = get_logger(__name__)

SLIDE_W = 1280
SLIDE_H = 720


@dataclass(frozen=True)
class NativeDeckSpec:
    background: str = "#F8FAFC"
    surface: str = "#FFFFFF"
    primary: str = "#2563EB"
    primary_dark: str = "#1E3A8A"
    text: str = "#111827"
    muted: str = "#64748B"
    border: str = "#D7DEE8"
    accent: str = "#F59E0B"
    title_font: str = "Arial"
    body_font: str = "Arial"

    def to_dict(self) -> dict[str, str]:
        return {
            "background": self.background,
            "surface": self.surface,
            "primary": self.primary,
            "primary_dark": self.primary_dark,
            "text": self.text,
            "muted": self.muted,
            "border": self.border,
            "accent": self.accent,
            "title_font": self.title_font,
            "body_font": self.body_font,
        }


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _escape(value: Any) -> str:
    return xml_escape(_clean_text(value), {'"': "&quot;"})


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [_clean_text(item) for item in parsed if _clean_text(item)]
        except Exception:
            pass
        return [
            _clean_text(part)
            for part in re.split(r"[\n;；]+", stripped)
            if _clean_text(part)
        ]
    return []


def _first_text(item: dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = _clean_text(item.get(key))
        if value:
            return value
    return default


def _extract_points(item: dict[str, Any]) -> list[str]:
    for key in ("key_points", "points", "bullets", "content_points", "main_points"):
        points = _coerce_list(item.get(key))
        if points:
            return points
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("key_points", "points", "bullets", "summary"):
            points = _coerce_list(content.get(key))
            if points:
                return points
    summary = _first_text(item, ("summary", "description", "abstract"), "")
    if summary:
        return [summary]
    return []


def _resolve_asset_path(item: dict[str, Any], result_root: Path) -> Path | None:
    candidates: list[Any] = []
    for key in ("asset_ref", "image_path", "img_path", "figure_path", "generated_img_path"):
        candidates.append(item.get(key))
    for visual in item.get("visual_assets") or []:
        if isinstance(visual, dict):
            candidates.extend([visual.get("storage_path"), visual.get("src"), visual.get("path")])

    for raw in candidates:
        text = _clean_text(raw)
        if not text or text.lower().startswith("table"):
            continue
        if text.startswith(("http://", "https://", "/outputs/")):
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = result_root / path
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _wrapped_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for raw_line in text.splitlines() or [text]:
        wrapped = textwrap.wrap(raw_line.strip(), width=max_chars) or [raw_line.strip()]
        lines.extend(wrapped)
        if len(lines) >= max_lines:
            break
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines and len(lines) == max_lines and len(text) > sum(len(line) for line in lines):
        lines[-1] = lines[-1].rstrip(" .，。") + "..."
    return lines


def _text_block(
    lines: Sequence[str],
    *,
    x: int,
    y: int,
    font_size: int,
    fill: str,
    font_family: str,
    line_gap: int,
    weight: str | None = None,
) -> str:
    parts: list[str] = []
    weight_attr = f' font-weight="{weight}"' if weight else ""
    for index, line in enumerate(lines):
        if not line:
            continue
        parts.append(
            f'<text x="{x}" y="{y + index * line_gap}" '
            f'font-family="{font_family}" font-size="{font_size}" '
            f'fill="{fill}"{weight_attr}>{_escape(line)}</text>'
        )
    return "\n".join(parts)


def _render_slide_svg(
    item: dict[str, Any],
    *,
    index: int,
    total: int,
    result_root: Path,
    spec: NativeDeckSpec,
) -> str:
    title = _first_text(item, ("title", "page_title", "section"), f"Slide {index + 1}")
    subtitle = _first_text(item, ("subtitle", "layout_description", "topic"), "")
    points = _extract_points(item)
    asset_path = _resolve_asset_path(item, result_root)

    has_image = asset_path is not None
    content_w = 700 if has_image else 1000
    title_lines = _wrapped_lines(title, 34 if has_image else 44, 2)
    subtitle_lines = _wrapped_lines(subtitle, 72 if has_image else 96, 2)

    body_parts: list[str] = []
    y = 245
    for point_idx, point in enumerate(points[:6]):
        wrapped = _wrapped_lines(point, 54 if has_image else 78, 2)
        bullet_y = y - 7
        body_parts.append(
            f'<circle cx="86" cy="{bullet_y}" r="5" fill="{spec.primary}"/>'
        )
        body_parts.append(
            _text_block(
                wrapped,
                x=104,
                y=y,
                font_size=25,
                fill=spec.text,
                font_family=spec.body_font,
                line_gap=33,
            )
        )
        y += max(54, len(wrapped) * 33 + 18)
        if y > 605:
            break

    if not body_parts and subtitle_lines:
        body_parts.append(
            _text_block(
                subtitle_lines,
                x=82,
                y=255,
                font_size=27,
                fill=spec.text,
                font_family=spec.body_font,
                line_gap=38,
            )
        )

    image_xml = ""
    if has_image and asset_path is not None:
        href = xml_escape(str(asset_path), {'"': "&quot;"})
        image_xml = f"""
<g id="p{index + 1:02d}-visual">
  <rect x="835" y="188" width="365" height="322" rx="18" fill="{spec.surface}" stroke="{spec.border}" stroke-width="2"/>
  <image href="{href}" x="855" y="208" width="325" height="282" preserveAspectRatio="xMidYMid meet"/>
  <text x="835" y="545" font-family="{spec.body_font}" font-size="17" fill="{spec.muted}">Source visual</text>
</g>"""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SLIDE_W}" height="{SLIDE_H}" viewBox="0 0 {SLIDE_W} {SLIDE_H}">
<rect x="0" y="0" width="{SLIDE_W}" height="{SLIDE_H}" fill="{spec.background}"/>
<rect x="0" y="0" width="{SLIDE_W}" height="16" fill="{spec.primary}"/>
<circle cx="1118" cy="84" r="54" fill="{spec.primary}" opacity="0.10"/>
<circle cx="1188" cy="118" r="28" fill="{spec.accent}" opacity="0.18"/>
<g id="p{index + 1:02d}-title">
  <rect x="72" y="68" width="{content_w}" height="4" fill="{spec.accent}"/>
  {_text_block(title_lines, x=72, y=130, font_size=42, fill=spec.primary_dark, font_family=spec.title_font, line_gap=50, weight="700")}
  {_text_block(subtitle_lines, x=74, y=190, font_size=20, fill=spec.muted, font_family=spec.body_font, line_gap=27)}
</g>
<g id="p{index + 1:02d}-body">
  {"".join(body_parts)}
</g>
{image_xml}
<g id="p{index + 1:02d}-footer">
  <line x1="72" y1="650" x2="1208" y2="650" stroke="{spec.border}" stroke-width="1"/>
  <text x="72" y="682" font-family="{spec.body_font}" font-size="18" fill="{spec.muted}">Paper2PPT Native</text>
  <text x="1168" y="682" font-family="{spec.body_font}" font-size="18" fill="{spec.muted}">{index + 1}/{total}</text>
</g>
</svg>
'''


def _validate_svg(svg_path: Path) -> None:
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG XML: {svg_path}: {exc}") from exc

    view_box = root.get("viewBox") or ""
    if view_box.strip() != f"0 0 {SLIDE_W} {SLIDE_H}":
        raise ValueError(f"unexpected SVG viewBox for {svg_path}: {view_box!r}")


def render_pagecontent_to_svg_deck(
    pagecontent: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    result_root: Path | str | None = None,
    style: str = "",
    language: str = "zh",
) -> list[Path]:
    """Render current Paper2Any pagecontent into PowerPoint-safe SVG files.

    This is the conservative first native renderer. It intentionally uses a
    narrow SVG subset that PPT Master's DrawingML converter can translate into
    editable PowerPoint shapes. The LLM-based SVG executor can replace this
    function later without changing the workflow/export boundary.
    """
    del style, language  # Reserved for the LLM-backed renderer.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(result_root or out.parent)
    spec = NativeDeckSpec()

    (out.parent / "spec_lock.native.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    svg_files: list[Path] = []
    total = len(pagecontent)
    for index, raw_item in enumerate(pagecontent):
        item = raw_item if isinstance(raw_item, dict) else {"title": str(raw_item)}
        svg = _render_slide_svg(
            item,
            index=index,
            total=total,
            result_root=root,
            spec=spec,
        )
        svg_path = out / f"page_{index + 1:03d}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        _validate_svg(svg_path)
        svg_files.append(svg_path)

    return svg_files


def export_svg_deck_to_pptx(
    svg_files: Sequence[Path | str],
    output_path: Path | str,
    *,
    verbose: bool = False,
) -> Path:
    """Export SVG files to a native editable PPTX using PPT Master's converter."""
    paths = [Path(path) for path in svg_files]
    if not paths:
        raise ValueError("svg_files is empty")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = create_pptx_with_native_svg(
        svg_files=paths,
        output_path=out,
        canvas_format="ppt169",
        verbose=verbose,
        transition=None,
        auto_advance=None,
        use_compat_mode=False,
        notes={},
        enable_notes=False,
        use_native_shapes=True,
        animation=None,
    )
    if not ok or not out.exists():
        raise RuntimeError(f"native PPTX export failed: {out}")
    return out
