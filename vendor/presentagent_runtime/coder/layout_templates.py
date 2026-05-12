"""Layout template library for deterministic slide rendering.

The coder's job on any slide reduces to: pick a template, then call one
`render_from_template(...)`. Templates own the geometry, font sizing, image
fitting, and visual hierarchy — the LLM does not touch python-pptx directly.

Why templates instead of a generic scaffold:
- Content volume is matched to slots that hold it without blanks or overflow.
- Image slots go through `fit_image_rect` so figures scale with the layout.
- Every shape the template draws is tagged with `_template_shape = True`, so
  any polish step that adds extra decorative boxes can be filtered out before
  save — no more shapes overlapping body text.
"""

from __future__ import annotations

from typing import Any, Callable

from pptx.util import Inches

from .pptx_library import (
    _deck_language,
    _hex_to_rgb,
    _presenter_fallback,
    add_blank_slide,
    add_panel,
    add_picture,
    add_subtitle_box,
    add_textbox,
    add_title_box,
    fit_image_rect,
    normalize_runtime_theme,
    set_background_color,
)

# Slide canvas (16:9 @ 13.33 x 7.5 inches).
_SLIDE_W = 13.33
_SLIDE_H = 7.5
_TITLE_ROW = (0.6, 0.45, 12.13, 0.95)  # shared title band across content templates
_TITLE_UNDERLINE_COLOR = "#E5E7EB"


def _tag(shape) -> None:
    """Mark a shape as template-owned so polish steps can keep it."""

    try:
        shape._element.set("data-template-shape", "1")
    except Exception:
        pass


def _resolve_asset_path(slide_ir: dict[str, Any], materials: dict[str, Any]) -> str:
    assets = slide_ir.get("assets") or {}
    path = (
        slide_ir.get("selected_asset_path")
        or assets.get("selected_asset_path")
        or ((materials.get("current_slide") or {}).get("selected_asset_path") if materials else "")
    )
    if path:
        return str(path).strip()
    for candidate in slide_ir.get("asset_paths") or []:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _points(slide_ir: dict[str, Any]) -> list[str]:
    content = slide_ir.get("content") or {}
    raw = content.get("points") or slide_ir.get("points") or []
    return [str(item).strip() for item in raw if str(item).strip()]


_BODY_SLOTS = ("body", "supporting_body", "metrics", "callout")


def _block_texts_by_slot(slide_ir: dict[str, Any]) -> dict[str, list[str]]:
    """Group body-like block texts by their slot_id / slot.

    Returns a dict keyed by the four body slots. Each value is the ordered list
    of text snippets that belong to that slot (both `text`/`content` and each
    entry in `items`). Title/image blocks are ignored.
    """

    result: dict[str, list[str]] = {slot: [] for slot in _BODY_SLOTS}
    for block in slide_ir.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind in {"headline", "image"}:
            continue
        slot = str(
            block.get("slot_id") or block.get("slot") or block.get("role") or ""
        ).strip().lower()
        if slot not in result:
            continue
        text = str(block.get("text") or block.get("content") or "").strip()
        if text:
            result[slot].append(text)
        for item in block.get("items") or []:
            item_text = str(item or "").strip()
            if item_text:
                result[slot].append(item_text)
    return result


def _body_texts(slide_ir: dict[str, Any]) -> list[str]:
    """Flattened body texts: explicit points first; else every body-block text."""

    direct = _points(slide_ir)
    if direct:
        return direct
    by_slot = _block_texts_by_slot(slide_ir)
    flat: list[str] = []
    for slot in _BODY_SLOTS:
        flat.extend(by_slot.get(slot, []))
    return flat


def _title(slide_ir: dict[str, Any]) -> str:
    return str(slide_ir.get("title") or (slide_ir.get("content") or {}).get("title") or "").strip()


def _subtitle(slide_ir: dict[str, Any]) -> str:
    return str(slide_ir.get("subtitle") or (slide_ir.get("content") or {}).get("subtitle") or "").strip()


def _core_message(slide_ir: dict[str, Any]) -> str:
    return str(slide_ir.get("core_message") or (slide_ir.get("content") or {}).get("core_message") or "").strip()


def _draw_title_band(slide, title: str, theme: dict[str, Any]) -> None:
    if not title:
        return
    left, top, width, height = _TITLE_ROW
    box = add_title_box(
        slide,
        title,
        left=left,
        top=top,
        width=width,
        height=height,
        color=theme.get("primary_color", "#134E8E"),
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
        font_size=28,
        align="left",
    )
    _tag(box)
    underline = add_panel(
        slide,
        left=left,
        top=top + height + 0.05,
        width=0.6,
        height=0.06,
        fill_color=theme.get("accent_color", "#C00707"),
        line_color=theme.get("accent_color", "#C00707"),
    )
    _tag(underline)


def _draw_card(
    slide,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    fill: str,
    outline: str | None = None,
):
    card = add_panel(
        slide,
        left=left,
        top=top,
        width=width,
        height=height,
        fill_color=fill,
        line_color=outline or fill,
    )
    _tag(card)
    return card


def _draw_image_in_slot(
    slide,
    image_path: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    theme: dict[str, Any],
) -> None:
    """Place image in slot using aspect-fit; fill slot with soft bg if aspect mismatches."""

    if not image_path:
        placeholder = add_panel(
            slide,
            left=left,
            top=top,
            width=width,
            height=height,
            fill_color=theme.get("panel_color", "#F3F4F6"),
            line_color=theme.get("panel_color", "#F3F4F6"),
        )
        _tag(placeholder)
        return
    fit_left, fit_top, fit_w, fit_h = fit_image_rect(image_path, left, top, width, height, mode="contain")
    slot_aspect = width / max(height, 0.01)
    img_aspect = fit_w / max(fit_h, 0.01)
    if abs(slot_aspect - img_aspect) / max(slot_aspect, 0.01) > 0.15:
        backdrop = add_panel(
            slide,
            left=left,
            top=top,
            width=width,
            height=height,
            fill_color=theme.get("panel_color", "#F3F4F6"),
            line_color=theme.get("panel_color", "#F3F4F6"),
        )
        _tag(backdrop)
    picture = add_picture(slide, image_path, fit_left, fit_top, fit_w, fit_h)
    if picture is not None:
        _tag(picture)


# ---------- Templates ----------


def _render_title_cover(slide, deck_ir, slide_ir, materials, theme, lang):
    title = _title(slide_ir) or str(deck_ir.get("title") or "").strip() or ("演示" if lang == "zh" else "Presentation")
    t = add_title_box(
        slide, title, left=0.6, top=2.6, width=12.13, height=1.3,
        color=theme.get("primary_color", "#134E8E"),
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
        font_size=42, align="center",
    )
    _tag(t)
    subtitle = _subtitle(slide_ir)
    presenter_line = _presenter_fallback(deck_ir, slide_ir, lang)
    if not subtitle:
        prefix = "汇报人：" if lang == "zh" else "Presenter: "
        has_prefix = any(tok in presenter_line for tok in ("汇报人", "Presenter", "presenter"))
        subtitle = presenter_line if has_prefix else f"{prefix}{presenter_line}"
    s = add_subtitle_box(
        slide, subtitle, left=0.6, top=4.15, width=12.13, height=0.7,
        color=theme.get("text_color", "#4B5563"),
        font_name=theme.get("font_family", "Aptos"),
        font_size=22, align="center",
    )
    _tag(s)
    accent = add_panel(
        slide, left=6.11, top=4.95, width=1.1, height=0.07,
        fill_color=theme.get("accent_color", "#C00707"),
        line_color=theme.get("accent_color", "#C00707"),
    )
    _tag(accent)


def _render_closing_thanks(slide, deck_ir, slide_ir, materials, theme, lang):
    from .pptx_library import _CLOSING_TITLE_TOKENS
    title_raw = _title(slide_ir)
    if not title_raw or not any(tok in title_raw.lower() for tok in _CLOSING_TITLE_TOKENS):
        title_text = "致谢" if lang == "zh" else "Thank You"
    else:
        title_text = title_raw
    subtitle_text = _subtitle(slide_ir) or _core_message(slide_ir)
    if not subtitle_text:
        points = _points(slide_ir)
        if points:
            subtitle_text = points[0]
    if not subtitle_text or subtitle_text == title_text:
        subtitle_text = "感谢聆听" if lang == "zh" else "Thanks for listening"
    t = add_title_box(
        slide, title_text, left=0.6, top=2.4, width=12.13, height=1.4,
        color=theme.get("primary_color", "#134E8E"),
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
        font_size=52, align="center",
    )
    _tag(t)
    s = add_subtitle_box(
        slide, subtitle_text, left=0.6, top=4.2, width=12.13, height=0.9,
        color=theme.get("text_color", "#4B5563"),
        font_name=theme.get("font_family", "Aptos"),
        font_size=26, align="center",
    )
    _tag(s)


def _render_section_divider(slide, deck_ir, slide_ir, materials, theme, lang):
    title = _title(slide_ir) or ("章节" if lang == "zh" else "Section")
    subtitle = _subtitle(slide_ir) or _core_message(slide_ir)
    t = add_title_box(
        slide, title, left=0.6, top=3.0, width=12.13, height=1.5,
        color=theme.get("primary_color", "#134E8E"),
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
        font_size=44, align="center",
    )
    _tag(t)
    divider = add_panel(
        slide, left=6.11, top=4.6, width=1.1, height=0.07,
        fill_color=theme.get("accent_color", "#C00707"),
        line_color=theme.get("accent_color", "#C00707"),
    )
    _tag(divider)
    if subtitle:
        s = add_subtitle_box(
            slide, subtitle, left=1.2, top=4.9, width=10.93, height=0.8,
            color=theme.get("text_color", "#4B5563"),
            font_name=theme.get("font_family", "Aptos"),
            font_size=20, align="center",
        )
        _tag(s)


def _render_single_card_center(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    core = _core_message(slide_ir)
    points = _body_texts(slide_ir)
    if not core and points:
        core = points[0]
        points = points[1:]
    card_left, card_top, card_w, card_h = 2.17, 2.2, 9.0, 3.8
    _draw_card(
        slide,
        left=card_left, top=card_top, width=card_w, height=card_h,
        fill=theme.get("panel_color", "#F3F4F6"),
        outline=theme.get("accent_color", "#C00707"),
    )
    if core:
        core_box = add_textbox(
            slide, core,
            left=card_left + 0.4, top=card_top + 0.5,
            width=card_w - 0.8, height=1.4,
            font_size=28, color=theme.get("primary_color", "#134E8E"),
            bold=True, align="center",
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
        )
        _tag(core_box)
    secondary = "\n".join(f"• {p}" for p in points[:2])
    if secondary:
        sec_box = add_textbox(
            slide, secondary,
            left=card_left + 0.6, top=card_top + 2.0,
            width=card_w - 1.2, height=card_h - 2.2,
            font_size=18, color=theme.get("text_color", "#374151"),
            align="left", font_name=theme.get("font_family", "Aptos"),
            fit=True, min_font_size=14,
        )
        _tag(sec_box)


def _render_quote_center(slide, deck_ir, slide_ir, materials, theme, lang):
    title = _title(slide_ir)
    if title:
        t = add_subtitle_box(
            slide, title, left=0.6, top=0.6, width=12.13, height=0.6,
            color=theme.get("text_color", "#6B7280"),
            font_name=theme.get("font_family", "Aptos"),
            font_size=16, align="center",
        )
        _tag(t)
    quote = _core_message(slide_ir) or (_body_texts(slide_ir) or [""])[0]
    if not quote:
        quote = "—"
    left_quote = add_textbox(
        slide, "“",
        left=1.2, top=2.0, width=1.0, height=1.6,
        font_size=110, color=theme.get("accent_color", "#C00707"),
        bold=True, align="left",
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
    )
    _tag(left_quote)
    q_box = add_textbox(
        slide, quote,
        left=1.5, top=2.6, width=10.33, height=2.8,
        font_size=32, color=theme.get("primary_color", "#134E8E"),
        bold=True, align="center",
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
        fit=True, min_font_size=22,
    )
    _tag(q_box)
    attribution = _subtitle(slide_ir)
    if attribution:
        a = add_subtitle_box(
            slide, f"— {attribution}",
            left=1.5, top=5.6, width=10.33, height=0.5,
            color=theme.get("text_color", "#4B5563"),
            font_name=theme.get("font_family", "Aptos"),
            font_size=18, align="center",
        )
        _tag(a)


def _render_three_column(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    points = _body_texts(slide_ir)[:4]
    if not points:
        points = [_core_message(slide_ir) or ("要点" if lang == "zh" else "Point")]
    cols = len(points) if len(points) in (2, 3, 4) else 3
    points = points[:cols]
    gap = 0.3
    total_w = _SLIDE_W - 1.2
    col_w = (total_w - gap * (cols - 1)) / cols
    top, height = 1.9, 4.9
    for index, text in enumerate(points):
        left = 0.6 + index * (col_w + gap)
        _draw_card(
            slide,
            left=left, top=top, width=col_w, height=height,
            fill=theme.get("panel_color", "#F3F4F6"),
            outline=theme.get("panel_color", "#F3F4F6"),
        )
        badge = add_textbox(
            slide, str(index + 1),
            left=left + 0.3, top=top + 0.3, width=0.9, height=0.9,
            font_size=42, color=theme.get("accent_color", "#C00707"),
            bold=True, align="left",
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
        )
        _tag(badge)
        body = add_textbox(
            slide, text,
            left=left + 0.35, top=top + 1.3,
            width=col_w - 0.7, height=height - 1.6,
            font_size=18, color=theme.get("text_color", "#1F2937"),
            align="left", font_name=theme.get("font_family", "Aptos"),
            fit=True, min_font_size=13,
        )
        _tag(body)


def _render_metric_grid(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    points = _body_texts(slide_ir)[:4]
    if not points:
        points = [_core_message(slide_ir) or "—"]
    count = len(points)
    gap = 0.3
    total_w = _SLIDE_W - 1.2
    card_w = (total_w - gap * (count - 1)) / count
    card_top, card_h = 2.4, 3.8
    for index, raw in enumerate(points):
        left = 0.6 + index * (card_w + gap)
        _draw_card(
            slide,
            left=left, top=card_top, width=card_w, height=card_h,
            fill="#FFFFFF",
            outline=theme.get("panel_color", "#E5E7EB"),
        )
        value, sep, label = raw.partition("：") if "：" in raw else raw.partition(":")
        if not sep:
            value, label = raw, ""
        big = add_textbox(
            slide, value.strip() or raw,
            left=left + 0.2, top=card_top + 0.5,
            width=card_w - 0.4, height=1.8,
            font_size=48, color=theme.get("accent_color", "#C00707"),
            bold=True, align="center",
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
            fit=True, min_font_size=28,
        )
        _tag(big)
        lab_box = add_textbox(
            slide, label.strip(),
            left=left + 0.2, top=card_top + 2.5,
            width=card_w - 0.4, height=1.0,
            font_size=16, color=theme.get("text_color", "#4B5563"),
            align="center", font_name=theme.get("font_family", "Aptos"),
            fit=True, min_font_size=12,
        )
        _tag(lab_box)


def _render_two_column_text_visual(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    points = _body_texts(slide_ir)[:6]
    if not points and _core_message(slide_ir):
        points = [_core_message(slide_ir)]
    bullets_text = "\n".join(f"• {p}" for p in points) or "—"
    body = add_textbox(
        slide, bullets_text,
        left=0.6, top=1.9, width=5.87, height=5.0,
        font_size=20, color=theme.get("text_color", "#1F2937"),
        align="left", font_name=theme.get("font_family", "Aptos"),
        fit=True, min_font_size=14,
    )
    _tag(body)
    asset = _resolve_asset_path(slide_ir, materials)
    _draw_image_in_slot(
        slide, asset,
        left=7.13, top=1.9, width=5.6, height=5.0, theme=theme,
    )


def _render_image_hero(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    asset = _resolve_asset_path(slide_ir, materials)
    _draw_image_in_slot(
        slide, asset,
        left=1.33, top=1.9, width=10.67, height=4.6, theme=theme,
    )
    caption = _core_message(slide_ir) or (_body_texts(slide_ir) or [""])[0]
    if caption:
        c = add_subtitle_box(
            slide, caption,
            left=1.33, top=6.65, width=10.67, height=0.6,
            color=theme.get("text_color", "#4B5563"),
            font_name=theme.get("font_family", "Aptos"),
            font_size=18, align="center",
        )
        _tag(c)


def _render_comparison(slide, deck_ir, slide_ir, materials, theme, lang):
    _draw_title_band(slide, _title(slide_ir), theme)
    points = _body_texts(slide_ir)
    if len(points) < 2:
        core = _core_message(slide_ir)
        points = points + ([core] if core else []) + ["—"] * (2 - len(points))
    mid = max(1, len(points) // 2)
    left_items, right_items = points[:mid], points[mid:]
    col_top, col_h = 1.9, 5.0

    for col_idx, (items, left) in enumerate(((left_items, 0.6), (right_items, 6.87))):
        _draw_card(
            slide,
            left=left, top=col_top, width=5.87, height=col_h,
            fill=theme.get("panel_color", "#F3F4F6"),
            outline=theme.get("panel_color", "#F3F4F6"),
        )
        header_text = ("A" if col_idx == 0 else "B")
        header = add_textbox(
            slide, header_text,
            left=left + 0.35, top=col_top + 0.25, width=1.0, height=0.8,
            font_size=36, color=theme.get("accent_color", "#C00707"),
            bold=True, align="left",
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
        )
        _tag(header)
        bullets = "\n".join(f"• {p}" for p in items) or "—"
        body = add_textbox(
            slide, bullets,
            left=left + 0.4, top=col_top + 1.1,
            width=5.07, height=col_h - 1.3,
            font_size=18, color=theme.get("text_color", "#1F2937"),
            align="left", font_name=theme.get("font_family", "Aptos"),
            fit=True, min_font_size=13,
        )
        _tag(body)
    vs = add_textbox(
        slide, "vs",
        left=6.39, top=4.0, width=0.55, height=0.6,
        font_size=18, color=theme.get("accent_color", "#C00707"),
        bold=True, align="center",
        font_name=f"{theme.get('font_family', 'Aptos')} Display",
    )
    _tag(vs)


def _render_quadrant(slide, deck_ir, slide_ir, materials, theme, lang):
    """4-cell grid. Reads one text per slot from blocks: body / supporting_body / metrics / callout."""

    _draw_title_band(slide, _title(slide_ir), theme)
    by_slot = _block_texts_by_slot(slide_ir)
    # Collect one representative text per slot; fall back to flat body texts
    # when a slot is empty so we always fill all 4 cells when content allows.
    fallback = list(_body_texts(slide_ir))
    cells: list[str] = []
    for slot in _BODY_SLOTS:
        texts = by_slot.get(slot) or []
        if texts:
            cells.append(texts[0])
            # drop from fallback to avoid double-using the same snippet below
            for used in texts:
                if used in fallback:
                    fallback.remove(used)
        else:
            cells.append("")
    for index in range(4):
        if not cells[index] and fallback:
            cells[index] = fallback.pop(0)
    if not any(cells):
        cells[0] = _core_message(slide_ir) or ("要点" if lang == "zh" else "Point")

    gap_x, gap_y = 0.3, 0.25
    total_w = _SLIDE_W - 1.2
    total_h = 5.0
    cell_w = (total_w - gap_x) / 2
    cell_h = (total_h - gap_y) / 2
    top = 1.9
    labels = ("A", "B", "C", "D")
    positions = [
        (0.6, top),
        (0.6 + cell_w + gap_x, top),
        (0.6, top + cell_h + gap_y),
        (0.6 + cell_w + gap_x, top + cell_h + gap_y),
    ]
    for index, (left, cell_top) in enumerate(positions):
        text = cells[index]
        if not text:
            continue
        _draw_card(
            slide,
            left=left, top=cell_top, width=cell_w, height=cell_h,
            fill=theme.get("panel_color", "#F3F4F6"),
            outline=theme.get("panel_color", "#F3F4F6"),
        )
        badge = add_textbox(
            slide, labels[index],
            left=left + 0.3, top=cell_top + 0.2, width=0.9, height=0.8,
            font_size=34, color=theme.get("accent_color", "#C00707"),
            bold=True, align="left",
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
        )
        _tag(badge)
        body = add_textbox(
            slide, text,
            left=left + 0.35, top=cell_top + 1.05,
            width=cell_w - 0.7, height=cell_h - 1.25,
            font_size=18, color=theme.get("text_color", "#1F2937"),
            align="left", font_name=theme.get("font_family", "Aptos"),
            fit=True, min_font_size=13,
        )
        _tag(body)


TEMPLATES: dict[str, Callable[..., None]] = {
    "title_cover": _render_title_cover,
    "closing_thanks": _render_closing_thanks,
    "section_divider": _render_section_divider,
    "single_card_center": _render_single_card_center,
    "quote_center": _render_quote_center,
    "three_column": _render_three_column,
    "quadrant": _render_quadrant,
    "metric_grid": _render_metric_grid,
    "two_column_text_visual": _render_two_column_text_visual,
    "image_hero": _render_image_hero,
    "comparison": _render_comparison,
}


def pick_template(slide_ir: dict[str, Any], materials: dict[str, Any] | None = None) -> str:
    """Content-aware template selection. Returns a key in TEMPLATES."""

    layout_name = str(
        (slide_ir.get("layout") or {}).get("name")
        or slide_ir.get("layout_type")
        or ""
    ).lower().strip()
    slide_type = str(slide_ir.get("type") or "").lower().strip()
    try:
        page_num = int(slide_ir.get("page_num") or 0)
    except (TypeError, ValueError):
        page_num = 0
    title_lc = _title(slide_ir).lower()
    points = _body_texts(slide_ir)
    has_visual = bool(_resolve_asset_path(slide_ir, materials or {}))

    if slide_type == "title" or (page_num == 1 and layout_name in {"title_only", "hero", ""}):
        return "title_cover"
    from .pptx_library import _CLOSING_TITLE_TOKENS
    if slide_type == "closing" or layout_name == "closing" or any(tok in title_lc for tok in _CLOSING_TITLE_TOKENS):
        return "closing_thanks"
    if layout_name == "quote_callout":
        return "quote_center"
    if layout_name == "metric_focus":
        return "metric_grid"
    if layout_name == "comparison":
        return "comparison"
    if layout_name == "quadrant" and not has_visual:
        return "quadrant"
    if has_visual and len(points) <= 1:
        return "image_hero"
    if has_visual:
        return "two_column_text_visual"
    has_core = bool(_core_message(slide_ir))
    if layout_name == "section_divider" or slide_type == "section":
        if not has_core and not points:
            return "section_divider"
    if len(points) == 0 and not has_core:
        return "section_divider"
    if len(points) >= 4:
        return "quadrant"
    if len(points) <= 2:
        return "single_card_center"
    return "three_column"


def render_from_template(
    prs,
    deck_ir: dict[str, Any],
    slide_ir: dict[str, Any],
    materials: dict[str, Any] | None = None,
    *,
    template_name: str | None = None,
):
    """Create a slide using a template. Non-template shapes are removed before return."""

    from .pptx_library import _runtime_slide_ir

    slide_ir = _runtime_slide_ir(slide_ir)
    materials = materials or {}
    theme = normalize_runtime_theme(deck_ir.get("theme", {}))
    lang = _deck_language(deck_ir)
    name = template_name or pick_template(slide_ir, materials)
    renderer = TEMPLATES.get(name) or TEMPLATES["section_divider"]

    slide = add_blank_slide(prs)
    set_background_color(slide, theme.get("background_color", "#F7F4EE"))
    renderer(slide, deck_ir, slide_ir, materials, theme, lang)
    _prune_nontemplate_shapes(slide)
    return slide


def _prune_nontemplate_shapes(slide) -> None:
    """Remove any shape not tagged by the template (protects against polish add-ons)."""

    spTree = slide.shapes._spTree
    for shape in list(slide.shapes):
        try:
            tag = shape._element.get("data-template-shape")
        except Exception:
            tag = None
        if tag == "1":
            continue
        try:
            spTree.remove(shape._element)
        except Exception:
            pass


__all__ = [
    "TEMPLATES",
    "pick_template",
    "render_from_template",
]


