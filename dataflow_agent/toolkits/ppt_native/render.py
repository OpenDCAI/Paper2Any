from __future__ import annotations

import json
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape as xml_escape

import httpx

from dataflow_agent.logger import get_logger

from .svg_quality_checker import SVGQualityChecker
from .svg_to_pptx.pptx_builder import create_pptx_with_native_svg

log = get_logger(__name__)

SLIDE_W = 1280
SLIDE_H = 720
_PPT_FORMAT = "ppt169"


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
    success: str = "#16A34A"
    warning: str = "#DC2626"
    title_font: str = "Arial, Microsoft YaHei, sans-serif"
    body_font: str = "Arial, Microsoft YaHei, sans-serif"
    code_font: str = "Consolas, Courier New, monospace"
    body_size: int = 22
    title_size: int = 36
    subtitle_size: int = 26
    annotation_size: int = 15

    def to_dict(self) -> dict[str, str | int]:
        return {
            "background": self.background,
            "surface": self.surface,
            "primary": self.primary,
            "primary_dark": self.primary_dark,
            "text": self.text,
            "muted": self.muted,
            "border": self.border,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "title_font": self.title_font,
            "body_font": self.body_font,
            "code_font": self.code_font,
            "body_size": self.body_size,
            "title_size": self.title_size,
            "subtitle_size": self.subtitle_size,
            "annotation_size": self.annotation_size,
        }


@dataclass(frozen=True)
class NativeLLMConfig:
    chat_api_url: str = ""
    api_key: str = ""
    model: str = ""
    style: str = ""
    language: str = "zh"
    temperature: float = 0.28
    max_tokens: int = 5200
    timeout_seconds: float = 180.0

    @property
    def enabled(self) -> bool:
        url = (self.chat_api_url or "").strip().lower()
        model = (self.model or "").strip()
        return bool(model and url.startswith(("http://", "https://")))


@dataclass
class NativeRenderResult:
    svg_files: list[Path]
    page_reports: list[dict[str, Any]] = field(default_factory=list)
    design_spec_path: Path | None = None
    spec_lock_path: Path | None = None


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
    for key in ("asset_ref", "image_path", "img_path", "figure_path", "generated_img_path", "ppt_img_path"):
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


def _collect_image_locks(pagecontent: Sequence[dict[str, Any]], result_root: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    for index, raw_item in enumerate(pagecontent):
        item = raw_item if isinstance(raw_item, dict) else {"title": str(raw_item)}
        asset_path = _resolve_asset_path(item, result_root)
        if asset_path is not None:
            images[f"p{index + 1:02d}_visual"] = f"{asset_path} | no-crop"
    return images


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
    for point in points[:6]:
        wrapped = _wrapped_lines(point, 54 if has_image else 78, 2)
        bullet_y = y - 7
        body_parts.append(f'<circle cx="86" cy="{bullet_y}" r="5" fill="{spec.primary}"/>')
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
        image_xml = f'''
<g id="p{index + 1:02d}-visual">
  <rect x="835" y="188" width="365" height="322" rx="18" fill="{spec.surface}" stroke="{spec.border}" stroke-width="2"/>
  <image href="{href}" x="855" y="208" width="325" height="282" preserveAspectRatio="xMidYMid meet"/>
  <text x="835" y="545" font-family="{spec.body_font}" font-size="17" fill="{spec.muted}">Source visual</text>
</g>'''

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


def _validate_svg_text(svg_text: str, *, context: str) -> None:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG XML in {context}: {exc}") from exc

    view_box = root.get("viewBox") or ""
    if view_box.strip() != f"0 0 {SLIDE_W} {SLIDE_H}":
        raise ValueError(f"unexpected SVG viewBox in {context}: {view_box!r}")


def _validate_svg(svg_path: Path) -> None:
    _validate_svg_text(svg_path.read_text(encoding="utf-8"), context=str(svg_path))


def _slug_title(value: str, fallback: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", _clean_text(value)).strip("._")
    text = re.sub(r"_+", "_", text)
    return (text or fallback)[:36]


def _page_key(index: int) -> str:
    return f"P{index + 1:02d}"


def _as_item(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {"title": str(raw)}


def _guess_rhythm(item: dict[str, Any], index: int, total: int) -> str:
    title = _first_text(item, ("title", "page_title", "section"), "")
    points = _extract_points(item)
    text = f"{title} {_first_text(item, ('subtitle', 'summary', 'description'), '')}".lower()
    if index == 0 or index == total - 1:
        return "anchor"
    if len(points) <= 2 or any(word in text for word in ("结论", "summary", "conclusion", "chapter", "章节", "致谢")):
        return "breathing"
    return "dense"


def _page_outline_line(item: dict[str, Any], index: int) -> str:
    title = _first_text(item, ("title", "page_title", "section"), f"Slide {index + 1}")
    subtitle = _first_text(item, ("subtitle", "layout_description", "topic", "summary"), "")
    points = _extract_points(item)[:5]
    point_text = "; ".join(points)
    parts = [f"- {_page_key(index)}: {title}"]
    if subtitle:
        parts.append(f"  - brief: {subtitle}")
    if point_text:
        parts.append(f"  - points: {point_text}")
    return "\n".join(parts)


def _write_design_spec(
    result_root: Path,
    pagecontent: Sequence[dict[str, Any]],
    *,
    spec: NativeDeckSpec,
    style: str,
    language: str,
    images: dict[str, str],
) -> Path:
    del language
    created = datetime.now().strftime("%Y-%m-%d")
    outline = "\n".join(_page_outline_line(_as_item(raw), index) for index, raw in enumerate(pagecontent))
    image_rows = "\n".join(
        f"| {name} | `{href.split(' | ')[0]}` | no-crop |" for name, href in images.items()
    ) or "| None |  |  |"
    rhythm_rows = "\n".join(
        f"| {_page_key(index)} | {_guess_rhythm(_as_item(raw), index, len(pagecontent))} |"
        for index, raw in enumerate(pagecontent)
    )
    text = f'''# Paper2PPT Native - Design Spec

> Human-readable design narrative. Machine-readable execution contract: `spec_lock.md`. Executor re-reads `spec_lock.md` before every SVG page; on divergence, `spec_lock.md` wins.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | Paper2PPT Native |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | {len(pagecontent)} |
| **Design Style** | {style or 'General Consulting'} |
| **Target Audience** | Academic / technical presentation audience |
| **Use Case** | Editable PowerPoint generated from paper2ppt pagecontent |
| **Created Date** | {created} |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 72px, top 64px, bottom 56px |
| **Content Area** | 1136x600 |

---

## III. Visual Theme

### Theme Style

- **Style**: {style or 'General Consulting'}
- **Theme**: Light theme
- **Tone**: professional, clear, research-oriented

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `{spec.background}` | Page background |
| **Secondary bg** | `{spec.surface}` | Cards / panels |
| **Primary** | `{spec.primary}` | Structure, rules, key marks |
| **Accent** | `{spec.accent}` | Emphasis |
| **Secondary accent** | `{spec.primary_dark}` | Titles / deep emphasis |
| **Body text** | `{spec.text}` | Main text |
| **Secondary text** | `{spec.muted}` | Captions / footer |
| **Border/divider** | `{spec.border}` | Lines / containers |
| **Success** | `{spec.success}` | Positive indicators |
| **Warning** | `{spec.warning}` | Risk indicators |

---

## IV. Typography System

### Font Plan

**Typography direction**: PPT-safe CJK/Latin sans-serif, optimized for editable PowerPoint output.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | Microsoft YaHei | Arial | sans-serif |
| **Body** | Microsoft YaHei | Arial | sans-serif |
| **Emphasis** | Microsoft YaHei | Arial | sans-serif |
| **Code** | - | Consolas, Courier New | monospace |

**Per-role font stacks**:

- Title: `{spec.title_font}`
- Body: `{spec.body_font}`
- Emphasis: same as Body
- Code: `{spec.code_font}`

### Font Size Hierarchy

**Baseline**: Body font size = {spec.body_size}px.

| Purpose | Ratio to body | Current Project | Weight |
| ------- | ------------- | --------------- | ------ |
| Cover title | 2.5-5x | 60-88px | Bold |
| Page title | 1.5-2x | {spec.title_size}px | Bold |
| Subtitle | 1.2-1.5x | {spec.subtitle_size}px | SemiBold |
| Body content | 1x | {spec.body_size}px | Regular |
| Annotation / caption | 0.7-0.85x | {spec.annotation_size}px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: 64-170px, page title and chapter cue.
- **Content area**: 180-620px, free layout based on page rhythm.
- **Footer area**: 650-690px, divider and page number.

### Spacing Specification

| Element | Current Project |
| ------- | --------------- |
| Safe margin from canvas edge | 72px |
| Content block gap | 28-40px |
| Icon-text gap | 12px |
| Card gap | 24px |
| Card padding | 24px |
| Card border radius | 8-16px |

---

## VI. Icon Usage Specification

- **Built-in icon library**: none for this phase unless explicitly added to `spec_lock.md`.
- **Usage method**: Executor must not use icon placeholders because the current lock has an empty inventory.

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| None |  |  |

---

## VII. Visualization Reference List

No chart template inheritance is locked in this phase. Executor may draw simple native SVG diagrams from pagecontent and must include `chart-plot-area` comments when it draws coordinate charts.

---

## VIII. Image Resource List

| Resource | Path | Constraint |
| -------- | ---- | ---------- |
{image_rows}

---

## IX. Content Outline

{outline}

### Page Rhythm

| Page | Rhythm |
| ---- | ------ |
{rhythm_rows}

---

## X. Speaker Notes Plan

Speaker notes are not generated in this native rendering phase.

---

## XI. Technical Constraints

- SVG files must use `viewBox="0 0 1280 720"`.
- Use inline attributes only; no `<style>`, no `class`, no `<foreignObject>`, no `textPath`, no `@font-face`, no animation, no script, no iframe, no `<symbol>` plus `<use>`.
- Use HEX colors from `spec_lock.md`; use opacity attributes on individual primitives, never `rgba()` or `<g opacity>`.
- Fonts and sizes must stay anchored to `spec_lock.md`.
- Run `SVGQualityChecker` on `svg_output/` before export.
'''
    path = result_root / "design_spec.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_spec_lock(
    result_root: Path,
    pagecontent: Sequence[dict[str, Any]],
    *,
    spec: NativeDeckSpec,
    images: dict[str, str],
) -> Path:
    image_section = ""
    if images:
        lines = "\n".join(f"- {name}: {href}" for name, href in images.items())
        image_section = f"\n## images\n{lines}\n"
    rhythm = "\n".join(
        f"- {_page_key(index)}: {_guess_rhythm(_as_item(raw), index, len(pagecontent))}"
        for index, raw in enumerate(pagecontent)
    )
    text = f'''# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## colors
- bg: {spec.background}
- surface: {spec.surface}
- primary: {spec.primary}
- primary_dark: {spec.primary_dark}
- accent: {spec.accent}
- text: {spec.text}
- text_secondary: {spec.muted}
- border: {spec.border}
- success: {spec.success}
- warning: {spec.warning}

## typography
- font_family: {spec.body_font}
- title_family: {spec.title_font}
- body_family: {spec.body_font}
- code_family: {spec.code_font}
- body: {spec.body_size}
- title: {spec.title_size}
- subtitle: {spec.subtitle_size}
- annotation: {spec.annotation_size}

## icons
- library: chunk-filled
- inventory:
{image_section}
## page_rhythm
{rhythm}

## page_layouts

## page_charts

## forbidden
- Mixing icon libraries
- rgba()
- `<style>`, `class`, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<script>`, `<iframe>`, `<symbol>`+`<use>`
- `<g opacity>` (set opacity on each child element individually)
- HTML named entities in text (`&nbsp;`, `&mdash;`, `&copy;`, `&ndash;`, `&reg;`, `&hellip;`, `&bull;`)
'''
    path = result_root / "spec_lock.md"
    path.write_text(text, encoding="utf-8")
    return path


def _extract_svg(raw_text: str) -> str:
    text = (raw_text or "").strip()
    fence = re.search(r"```(?:svg|xml)?\s*(<svg[\s\S]*?</svg>)\s*```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    match = re.search(r"<svg[\s\S]*?</svg>", text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    raise ValueError("LLM response did not contain a complete <svg>...</svg> document")


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text"), dict) and isinstance(item["text"].get("value"), str):
                    parts.append(item["text"]["value"])
        return "\n".join(parts)
    return str(content)


async def _call_chat_completion_text(
    *,
    llm: NativeLLMConfig,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    api_url = llm.chat_api_url.rstrip("/")
    target_url = api_url if api_url.endswith("/chat/completions") else f"{api_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if llm.api_key:
        headers["Authorization"] = f"Bearer {llm.api_key}"
    payload = {
        "model": llm.model,
        "messages": messages,
        "temperature": llm.temperature if temperature is None else temperature,
        "max_tokens": llm.max_tokens if max_tokens is None else max_tokens,
    }
    timeout = httpx.Timeout(timeout=llm.timeout_seconds, connect=min(20.0, llm.timeout_seconds))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post(target_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"LLM SVG generation failed ({response.status_code}): {response.text[:400]}")
    data = response.json()
    return _extract_message_content(data)


def _page_context(pagecontent: Sequence[dict[str, Any]], index: int) -> dict[str, Any]:
    def summary_at(idx: int) -> dict[str, Any] | None:
        if idx < 0 or idx >= len(pagecontent):
            return None
        item = _as_item(pagecontent[idx])
        return {
            "page": _page_key(idx),
            "title": _first_text(item, ("title", "page_title", "section"), f"Slide {idx + 1}"),
            "points": _extract_points(item)[:3],
        }

    current = _as_item(pagecontent[index])
    return {
        "current": current,
        "previous": summary_at(index - 1),
        "next": summary_at(index + 1),
        "page_key": _page_key(index),
        "page_index": index + 1,
        "page_count": len(pagecontent),
    }


def _executor_messages(
    *,
    design_spec: str,
    spec_lock: str,
    page_ctx: dict[str, Any],
    language: str,
    repair_errors: Sequence[str] | None = None,
    broken_svg: str | None = None,
) -> list[dict[str, Any]]:
    repair_block = ""
    if repair_errors:
        repair_block = (
            "\nRepair the SVG below. Keep the same page intent and return a full corrected SVG only.\n"
            f"Checker errors:\n{json.dumps(list(repair_errors), ensure_ascii=False, indent=2)}\n"
            f"Broken SVG:\n{broken_svg or ''}\n"
        )
    system = '''You are PPT Master Executor. Generate one PowerPoint-safe SVG page.
Hard rules:
- Read and obey spec_lock.md for every page; use only declared colors, font families, font-size ramp, images, and page rhythm.
- Return raw complete SVG only. No markdown, no explanation.
- Canvas must be <svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">.
- Use inline SVG attributes only. Do not use <style>, class, <foreignObject>, textPath, @font-face, animations, scripts, iframes, masks, rgba(), <g opacity>, <image opacity>, or <symbol>+<use>.
- Use HEX colors from spec_lock; transparency must be fill-opacity/stroke-opacity/opacity on individual primitive elements.
- Use semantic top-level <g id="..."> groups.
- Keep text editable with <text>/<tspan>; escape XML reserved chars as &amp; &lt; &gt; &quot; &apos;. Do not use HTML named entities.
- Do not use icon placeholders unless spec_lock icons.inventory lists icon names; current empty inventory means no icons.
- If drawing a coordinate chart, include a chart-plot-area comment inside <g id="chartArea">.
'''
    user = f'''Language: {language}

DESIGN_SPEC.md:
{design_spec}

SPEC_LOCK.md (authoritative):
{spec_lock}

Page context JSON:
{json.dumps(page_ctx, ensure_ascii=False, indent=2)}
{repair_block}
Generate this page as editable SVG primitives suitable for conversion to DrawingML.
'''
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_quality_check(svg_path: Path) -> dict[str, Any]:
    checker = SVGQualityChecker()
    result = checker.check_file(str(svg_path), _PPT_FORMAT)
    return {
        "passed": bool(result.get("passed")),
        "errors": list(result.get("errors") or []),
        "warnings": list(result.get("warnings") or []),
        "info": dict(result.get("info") or {}),
    }


async def _render_llm_svg_page(
    *,
    pagecontent: Sequence[dict[str, Any]],
    index: int,
    svg_path: Path,
    result_root: Path,
    design_spec_path: Path,
    spec_lock_path: Path,
    llm: NativeLLMConfig,
) -> tuple[str, dict[str, Any]]:
    del result_root
    design_spec_text = design_spec_path.read_text(encoding="utf-8")
    spec_lock_text = spec_lock_path.read_text(encoding="utf-8")
    page_ctx = _page_context(pagecontent, index)
    raw = await _call_chat_completion_text(
        llm=llm,
        messages=_executor_messages(
            design_spec=design_spec_text,
            spec_lock=spec_lock_text,
            page_ctx=page_ctx,
            language=llm.language,
        ),
    )
    svg = _extract_svg(raw)
    _validate_svg_text(svg, context=f"LLM page {_page_key(index)}")
    svg_path.write_text(svg, encoding="utf-8")
    check = _run_quality_check(svg_path)
    if check["passed"]:
        return "llm_svg", check

    spec_lock_text = spec_lock_path.read_text(encoding="utf-8")
    repair_raw = await _call_chat_completion_text(
        llm=llm,
        messages=_executor_messages(
            design_spec=design_spec_text,
            spec_lock=spec_lock_text,
            page_ctx=page_ctx,
            language=llm.language,
            repair_errors=check["errors"],
            broken_svg=svg,
        ),
        temperature=0.12,
        max_tokens=llm.max_tokens,
    )
    repaired_svg = _extract_svg(repair_raw)
    _validate_svg_text(repaired_svg, context=f"LLM repaired page {_page_key(index)}")
    svg_path.write_text(repaired_svg, encoding="utf-8")
    repaired_check = _run_quality_check(svg_path)
    if repaired_check["passed"]:
        return "llm_svg_repaired", repaired_check
    raise ValueError(f"SVG quality check failed after repair: {repaired_check['errors']}")


def _write_fallback_svg(
    *,
    raw_item: Any,
    index: int,
    total: int,
    svg_path: Path,
    result_root: Path,
    spec: NativeDeckSpec,
) -> dict[str, Any]:
    item = _as_item(raw_item)
    svg = _render_slide_svg(item, index=index, total=total, result_root=result_root, spec=spec)
    _validate_svg_text(svg, context=f"fallback page {_page_key(index)}")
    svg_path.write_text(svg, encoding="utf-8")
    return _run_quality_check(svg_path)


async def render_pagecontent_to_svg_deck(
    pagecontent: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    result_root: Path | str | None = None,
    style: str = "",
    language: str = "zh",
    chat_api_url: str = "",
    api_key: str = "",
    model: str = "",
) -> NativeRenderResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(result_root or out.parent)
    root.mkdir(parents=True, exist_ok=True)
    spec = NativeDeckSpec()
    llm = NativeLLMConfig(
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=model,
        style=style,
        language=language or "zh",
    )

    normalized_pagecontent: list[dict[str, Any]] = [_as_item(raw) for raw in pagecontent]
    images = _collect_image_locks(normalized_pagecontent, root)
    design_spec_path = _write_design_spec(
        root,
        normalized_pagecontent,
        spec=spec,
        style=style,
        language=language,
        images=images,
    )
    spec_lock_path = _write_spec_lock(root, normalized_pagecontent, spec=spec, images=images)
    (root / "spec_lock.native.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    svg_files: list[Path] = []
    page_reports: list[dict[str, Any]] = []
    total = len(normalized_pagecontent)
    for index, item in enumerate(normalized_pagecontent):
        title = _first_text(item, ("title", "page_title", "section"), f"slide_{index + 1}")
        svg_path = out / f"{index + 1:02d}_{_slug_title(title, f'slide_{index + 1}')}.svg"
        report: dict[str, Any] = {
            "page_idx": index,
            "page_key": _page_key(index),
            "svg_path": str(svg_path),
            "executor": "llm_svg" if llm.enabled else "fallback_no_llm_config",
        }
        try:
            if not llm.enabled:
                raise RuntimeError("LLM config missing; using conservative native fallback")
            mode, check = await _render_llm_svg_page(
                pagecontent=normalized_pagecontent,
                index=index,
                svg_path=svg_path,
                result_root=root,
                design_spec_path=design_spec_path,
                spec_lock_path=spec_lock_path,
                llm=llm,
            )
            report.update({"mode": mode, "quality": check})
        except Exception as exc:  # noqa: BLE001
            log.warning("[paper2ppt_native] fallback SVG for %s: %s", _page_key(index), exc)
            fallback_check = _write_fallback_svg(
                raw_item=item,
                index=index,
                total=total,
                svg_path=svg_path,
                result_root=root,
                spec=spec,
            )
            if not fallback_check["passed"]:
                raise ValueError(f"fallback SVG failed quality check for {_page_key(index)}: {fallback_check['errors']}") from exc
            report.update(
                {
                    "mode": "fallback_conservative_svg",
                    "fallback_reason": str(exc),
                    "quality": fallback_check,
                }
            )
        svg_files.append(svg_path)
        page_reports.append(report)

    (root / "native_render_manifest.json").write_text(
        json.dumps(
            {
                "design_spec": str(design_spec_path),
                "spec_lock": str(spec_lock_path),
                "llm_enabled": llm.enabled,
                "pages": page_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return NativeRenderResult(
        svg_files=svg_files,
        page_reports=page_reports,
        design_spec_path=design_spec_path,
        spec_lock_path=spec_lock_path,
    )


def render_pagecontent_to_svg_deck_conservative(
    pagecontent: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    result_root: Path | str | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(result_root or out.parent)
    spec = NativeDeckSpec()
    svg_files: list[Path] = []
    total = len(pagecontent)
    for index, raw_item in enumerate(pagecontent):
        item = _as_item(raw_item)
        title = _first_text(item, ("title", "page_title", "section"), f"slide_{index + 1}")
        svg_path = out / f"{index + 1:02d}_{_slug_title(title, f'slide_{index + 1}')}.svg"
        _write_fallback_svg(
            raw_item=item,
            index=index,
            total=total,
            svg_path=svg_path,
            result_root=root,
            spec=spec,
        )
        _validate_svg(svg_path)
        svg_files.append(svg_path)
    return svg_files


def export_svg_deck_to_pptx(
    svg_files: Sequence[Path | str],
    output_path: Path | str,
    *,
    verbose: bool = False,
) -> Path:
    paths = [Path(path) for path in svg_files]
    if not paths:
        raise ValueError("svg_files is empty")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = create_pptx_with_native_svg(
        svg_files=paths,
        output_path=out,
        canvas_format=_PPT_FORMAT,
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
