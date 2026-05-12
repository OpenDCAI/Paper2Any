"""Higher-level python-pptx helpers for runtime LLM slide generation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


def create_presentation(width: float = 13.33, height: float = 7.5) -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(width)
    prs.slide_height = Inches(height)
    return prs


def add_blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def set_background_color(slide, color: str) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = _hex_to_rgb(color)


def add_textbox(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 20,
    color: str = "#1F2937",
    bold: bool = False,
    align: str = "left",
    fill_color: str | None = None,
    font_name: str = "Aptos",
    margin: float = 0.06,
    fit: bool = False,
    min_font_size: int = 8,
):
    if fit:
        font_size = fit_font_size(text, width, height, font_size, min_font_size=min_font_size, margin=margin)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    if fill_color:
        box.fill.solid()
        box.fill.fore_color.rgb = _hex_to_rgb(fill_color)
    else:
        box.fill.background()

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = _map_align(align)
    font = p.font
    font.name = font_name
    font.size = Pt(font_size)
    font.bold = bold
    font.color.rgb = _hex_to_rgb(color)
    return box


def add_title_box(slide, text: str, left: float = 0.6, top: float = 0.35, width: float = 12.1, height: float = 0.9,
                  font_size: int = 28, color: str = "#134E8E", font_name: str = "Aptos Display",
                  align: str = "left"):
    return add_textbox(
        slide,
        text,
        left=left,
        top=top,
        width=width,
        height=height,
        font_size=font_size,
        color=color,
        bold=True,
        align=align,
        font_name=font_name,
    )


def add_subtitle_box(slide, text: str, left: float = 0.6, top: float = 1.1, width: float = 8.5, height: float = 0.55,
                     font_size: int = 14, color: str = "#4B5563", font_name: str = "Aptos",
                     align: str = "left"):
    return add_textbox(
        slide,
        text,
        left=left,
        top=top,
        width=width,
        height=height,
        font_size=font_size,
        color=color,
        bold=False,
        align=align,
        font_name=font_name,
    )


def add_bullet_list(
    slide,
    items: Sequence[str],
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 18,
    color: str = "#1F2937",
    bullet_char: str = "•",
    font_name: str = "Aptos",
    fit: bool = False,
    min_font_size: int = 8,
):
    if fit:
        text = "\n".join(f"{bullet_char} {item}" for item in items)
        font_size = fit_font_size(text, width, height, font_size, min_font_size=min_font_size, margin=0.06)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for index, item in enumerate(items):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = f"{bullet_char} {item}"
        p.alignment = PP_ALIGN.LEFT
        font = p.font
        font.name = font_name
        font.size = Pt(font_size)
        font.color.rgb = _hex_to_rgb(color)
    return box


def fit_font_size(
    text: str,
    width: float,
    height: float,
    preferred_font_size: int,
    *,
    min_font_size: int = 8,
    margin: float = 0.06,
) -> int:
    """Return a conservative font size for text inside an inch-based box."""

    text = str(text or "")
    usable_width = max(width - 2 * margin, 0.2)
    usable_height = max(height - 2 * margin, 0.16)
    preferred = max(int(preferred_font_size or min_font_size), min_font_size)
    for size in range(preferred, min_font_size - 1, -1):
        lines = _estimate_wrapped_line_count(text, usable_width, size)
        line_height = size * 1.18 / 72.0
        if lines * line_height <= usable_height:
            return size
    return min_font_size


def _estimate_wrapped_line_count(text: str, usable_width: float, font_size: int) -> int:
    if not text:
        return 1
    # A practical mixed CJK/Latin estimate. python-pptx does not expose real
    # layout measurement, so qwen_lib uses a conservative deterministic bound.
    chars_per_line = max(int(usable_width * 6.0 * 12.0 / max(font_size, 1)), 4)
    lines = 0
    for raw_line in str(text).splitlines() or [""]:
        line = raw_line.strip()
        weighted = 0.0
        for char in line:
            weighted += 1.0 if ord(char) > 127 else 0.56
        lines += max(1, math.ceil(weighted / chars_per_line))
    return max(lines, 1)


def _normalize_asset_path(image_path: Any) -> str:
    if isinstance(image_path, dict):
        path = image_path.get("path")
        if isinstance(path, str) and path:
            return path
        raise ValueError(f"image asset dict missing path: {image_path}")
    if isinstance(image_path, str) and image_path:
        return image_path
    raise ValueError(f"invalid image path: {image_path}")


def add_picture(slide, image_path: str | dict[str, Any], left: float, top: float, width: float, height: float):
    normalized_path = _normalize_asset_path(image_path)
    if not os.path.exists(normalized_path):
        # Gracefully skip missing assets instead of crashing the slide
        return None
    return slide.shapes.add_picture(normalized_path, Inches(left), Inches(top), Inches(width), Inches(height))


def image_aspect_ratio(image_path: str | dict[str, Any]) -> float | None:
    normalized_path = _normalize_asset_path(image_path)
    with Image.open(normalized_path) as image:
        image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        return None
    return float(image_width) / float(image_height)


def fit_image_rect(
    image_path: str | dict[str, Any],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    mode: str = "contain",
) -> tuple[float, float, float, float]:
    try:
        native_aspect = image_aspect_ratio(image_path)
    except Exception:
        native_aspect = None
    if not native_aspect or width <= 0 or height <= 0:
        return left, top, width, height
    box_aspect = width / height
    if mode == "cover":
        if native_aspect > box_aspect:
            render_height = height
            render_width = height * native_aspect
        else:
            render_width = width
            render_height = width / native_aspect
    else:
        if native_aspect > box_aspect:
            render_width = width
            render_height = width / native_aspect
        else:
            render_height = height
            render_width = height * native_aspect
    return (
        left + (width - render_width) / 2,
        top + (height - render_height) / 2,
        render_width,
        render_height,
    )


def safe_add_picture(
    slide,
    image_path: str | dict[str, Any],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    fit: str = "contain",
):
    try:
        if fit in {"contain", "cover"}:
            left, top, width, height = fit_image_rect(image_path, left, top, width, height, mode=fit)
        return add_picture(slide, image_path, left, top, width, height)
    except Exception:
        return None


def add_icon(slide, image_path: str, left: float, top: float, size: float = 0.35):
    return add_picture(slide, image_path, left, top, size, size)


def add_shape(
    slide,
    shape_type: str,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: str | None = "#FFFFFF",
    line_color: str | None = "#D1D5DB",
    line_width: float = 1.0,
):
    shape_enum = getattr(MSO_AUTO_SHAPE_TYPE, shape_type.upper())
    shape = slide.shapes.add_shape(shape_enum, Inches(left), Inches(top), Inches(width), Inches(height))
    if fill_color is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = _hex_to_rgb(fill_color)
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = _hex_to_rgb(line_color)
        shape.line.width = line_width if hasattr(line_width, "emu") else Pt(line_width)
    return shape


def add_panel(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: str = "#FFFFFF",
    line_color: str = "#D1D5DB",
    line_width: float = 1.0,
    radius_shape: str = "ROUNDED_RECTANGLE",
):
    return add_shape(
        slide,
        radius_shape,
        left,
        top,
        width,
        height,
        fill_color=fill_color,
        line_color=line_color,
        line_width=line_width,
    )


def add_connector(slide, connector_type: str, x1: float, y1: float, x2: float, y2: float, color: str = "#9CA3AF"):
    connector_enum = getattr(MSO_CONNECTOR, connector_type.upper())
    line = slide.shapes.add_connector(
        connector_enum,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = _hex_to_rgb(color)
    return line


def add_banner(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: str = "#134E8E",
    text_color: str = "#FFFFFF",
    font_name: str = "Aptos",
):
    add_shape(slide, "RECTANGLE", left, top, width, height, fill_color=fill_color, line_color=fill_color)
    return add_textbox(
        slide,
        text,
        left=left + 0.08,
        top=top + 0.02,
        width=width - 0.16,
        height=height - 0.04,
        font_size=20,
        color=text_color,
        bold=True,
        font_name=font_name,
    )


def add_metric_card(slide, label: str, value: str, left: float, top: float, width: float, height: float,
                    accent_color: str = "#C00707", background_color: str = "#FFF7ED",
                    text_color: str = "#1F2937"):
    add_shape(slide, "ROUNDED_RECTANGLE", left, top, width, height, fill_color=background_color, line_color=accent_color, line_width=1.5)
    add_textbox(slide, label, left + 0.18, top + 0.14, width - 0.36, 0.35, font_size=12, color=text_color)
    add_textbox(slide, value, left + 0.18, top + 0.5, width - 0.36, 0.55, font_size=26, color=accent_color, bold=True)


def add_process_flow(slide, steps: Sequence[str], left: float, top: float, width: float, height: float,
                     accent_color: str = "#134E8E", fill_color: str = "#EFF6FF", text_color: str = "#1F2937"):
    if not steps:
        return
    if len(steps) >= 4 or height >= 2.2:
        return add_numbered_process_list(
            slide,
            steps,
            left,
            top,
            width,
            height,
            accent_color=accent_color,
            text_color=text_color,
        )
    step_width = width / max(len(steps), 1)
    box_width = max(step_width - 0.2, 1.0)
    center_y = top + height / 2
    for index, step in enumerate(steps):
        box_left = left + index * step_width
        add_shape(slide, "ROUNDED_RECTANGLE", box_left, top + 0.15, box_width, height - 0.3, fill_color=fill_color, line_color=accent_color)
        add_textbox(slide, step, box_left + 0.12, top + 0.28, box_width - 0.24, height - 0.56, font_size=14, color=text_color, bold=True)
        if index < len(steps) - 1:
            add_connector(slide, "STRAIGHT", box_left + box_width, center_y, box_left + step_width - 0.05, center_y, color=accent_color)


def add_numbered_process_list(
    slide,
    steps: Sequence[str],
    left: float,
    top: float,
    width: float,
    height: float,
    accent_color: str = "#134E8E",
    text_color: str = "#1F2937",
):
    if not steps:
        return []
    shapes = []
    count = len(steps)
    row_gap = min(0.08, max(height * 0.025, 0.03))
    row_height = max((height - row_gap * max(count - 1, 0)) / count, 0.28)
    marker_size = min(0.34, max(row_height * 0.58, 0.22))
    line_x = left + marker_size / 2
    first_center_y = top + marker_size / 2
    last_center_y = top + (count - 1) * (row_height + row_gap) + marker_size / 2
    if count > 1:
        shapes.append(add_connector(slide, "STRAIGHT", line_x, first_center_y, line_x, last_center_y, color=accent_color))
    font_size = fit_font_size("\n".join(str(step) for step in steps), max(width - marker_size - 0.28, 0.2), height, 15, min_font_size=9)
    for index, step in enumerate(steps):
        row_top = top + index * (row_height + row_gap)
        marker = add_shape(
            slide,
            "OVAL",
            left,
            row_top,
            marker_size,
            marker_size,
            fill_color=accent_color,
            line_color=accent_color,
            line_width=0.8,
        )
        shapes.append(marker)
        shapes.append(
            add_textbox(
                slide,
                str(index + 1),
                left,
                row_top + 0.01,
                marker_size,
                marker_size,
                font_size=max(font_size - 2, 8),
                color="#FFFFFF",
                bold=True,
                align="center",
                margin=0.01,
            )
        )
        shapes.append(
            add_textbox(
                slide,
                str(step),
                left + marker_size + 0.18,
                row_top - 0.02,
                max(width - marker_size - 0.18, 0.2),
                max(row_height + 0.04, 0.24),
                font_size=font_size,
                color=text_color,
                bold=True,
                margin=0.02,
                fit=True,
                min_font_size=9,
            )
        )
    return shapes


def add_comparison_columns(slide, headers: Sequence[str], columns: Sequence[Sequence[str]], left: float, top: float,
                           width: float, height: float, header_fill: str = "#E5E7EB", body_fill: str = "#FFFFFF"):
    count = max(len(headers), 1)
    column_width = width / count
    for idx in range(count):
        col_left = left + idx * column_width
        add_shape(slide, "RECTANGLE", col_left, top, column_width - 0.08, 0.5, fill_color=header_fill, line_color="#D1D5DB")
        add_textbox(slide, headers[idx], col_left + 0.1, top + 0.06, column_width - 0.28, 0.34, font_size=14, bold=True)
        add_shape(slide, "RECTANGLE", col_left, top + 0.52, column_width - 0.08, height - 0.52, fill_color=body_fill, line_color="#E5E7EB")
        add_bullet_list(slide, list(columns[idx]), col_left + 0.12, top + 0.66, column_width - 0.32, height - 0.82, font_size=13)


def add_table(slide, rows: Sequence[Sequence[str]], left: float, top: float, width: float, height: float):
    row_count = max(len(rows), 1)
    col_count = max(max((len(row) for row in rows), default=1), 1)
    table = slide.shapes.add_table(row_count, col_count, Inches(left), Inches(top), Inches(width), Inches(height)).table
    for r_idx, row in enumerate(rows):
        for c_idx in range(col_count):
            value = row[c_idx] if c_idx < len(row) else ""
            cell = table.cell(r_idx, c_idx)
            cell.text = str(value)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(8 if row_count >= 5 or col_count >= 3 else 9)
                paragraph.font.name = "Noto Sans CJK SC"
    return table


def add_bar_chart(
    slide,
    categories: Sequence[str],
    series_name: str,
    values: Sequence[float],
    left: float,
    top: float,
    width: float,
    height: float,
):
    chart_data = CategoryChartData()
    chart_data.categories = list(categories)
    chart_data.add_series(series_name, list(values))
    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
        chart_data,
    )
    return graphic_frame.chart


def add_takeaway_block(
    slide,
    text: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    add_panel(
        slide,
        left,
        top,
        width,
        height,
        fill_color=theme.get("takeaway_fill", "#EFF6FF"),
        line_color=theme.get("primary_color", "#134E8E"),
        line_width=1.3,
    )
    return add_textbox(
        slide,
        text,
        left + 0.18,
        top + 0.12,
        max(width - 0.36, 0.2),
        max(height - 0.24, 0.25),
        font_size=16,
        color=theme.get("primary_color", "#134E8E"),
        bold=True,
        font_name=theme.get("font_family", "Aptos"),
    )


def add_highlight_block(
    slide,
    text: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    accent = theme.get("accent_color", "#FFB33F")
    add_panel(
        slide,
        left,
        top,
        width,
        height,
        fill_color=theme.get("highlight_fill", "#FFF7ED"),
        line_color=accent,
        line_width=1.2,
    )
    return add_textbox(
        slide,
        text,
        left + 0.2,
        top + 0.14,
        max(width - 0.4, 0.2),
        max(height - 0.28, 0.25),
        font_size=15,
        color=theme.get("text_color", "#1F2937"),
        bold=True,
        font_name=theme.get("font_family", "Aptos"),
    )


def add_metric_pair_block(
    slide,
    metrics: Sequence[dict[str, Any]],
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    items = list(metrics or [])[:2]
    if not items:
        return add_highlight_block(slide, "No metrics available", slot_rect, theme)
    card_gap = 0.18
    card_width = max((width - card_gap * (len(items) - 1)) / len(items), 0.8)
    shapes = []
    for index, metric in enumerate(items):
        label = str(metric.get("label") or f"Metric {index + 1}")
        value = str(metric.get("value") or metric.get("content") or "")
        add_metric_card(
            slide,
            label,
            value,
            left + index * (card_width + card_gap),
            top,
            card_width,
            height,
            accent_color=theme.get("accent_color", "#C00707"),
            background_color=theme.get("metric_fill", "#FFF7ED"),
            text_color=theme.get("text_color", "#1F2937"),
        )
        shapes.append(metric)
    return shapes


def add_visual_with_caption_block(
    slide,
    image_path: str | dict[str, Any] | None,
    caption: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    caption_height = min(0.45, max(height * 0.18, 0.28))
    visual_height = max(height - caption_height - 0.1, 0.3)
    image_shape = None
    if image_path:
        image_shape = safe_add_picture(slide, image_path, left, top, width, visual_height)
    if image_shape is None:
        image_shape = safe_placeholder_panel(
            slide,
            (left, top, width, visual_height),
            label=theme.get("visual_placeholder", "visual unavailable"),
            theme=theme,
        )
    caption_shape = add_textbox(
        slide,
        caption,
        left,
        top + visual_height + 0.1,
        width,
        caption_height,
        font_size=int(theme.get("caption_font_size", 11) or 11),
        color=theme.get("text_color", "#4B5563"),
        font_name=theme.get("font_family", "Aptos"),
    )
    return {"visual": image_shape, "caption": caption_shape}


def add_evidence_footer_block(
    slide,
    evidence_items: Sequence[str],
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    text = " | ".join(str(item) for item in evidence_items if str(item).strip()) or "Evidence available"
    add_panel(
        slide,
        left,
        top,
        width,
        height,
        fill_color=theme.get("footer_fill", "#F9FAFB"),
        line_color=theme.get("footer_line", "#E5E7EB"),
        line_width=0.8,
        radius_shape="RECTANGLE",
    )
    return add_textbox(
        slide,
        text,
        left + 0.12,
        top + 0.04,
        max(width - 0.24, 0.2),
        max(height - 0.08, 0.2),
        font_size=int(theme.get("footer_font_size", 9) or 9),
        color=theme.get("text_color", "#4B5563"),
        font_name=theme.get("font_family", "Aptos"),
    )


def compose_chart_with_takeaway(
    slide,
    categories: Sequence[str],
    series_name: str,
    values: Sequence[float],
    takeaway: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    takeaway_height = min(0.85, max(height * 0.22, 0.55))
    chart_height = max(height - takeaway_height - 0.18, 0.5)
    try:
        chart = add_bar_chart(slide, categories, series_name, values, left, top, width, chart_height)
    except Exception:
        chart = safe_placeholder_panel(slide, (left, top, width, chart_height), label="chart unavailable", theme=theme)
    takeaway_shape = add_takeaway_block(
        slide,
        takeaway,
        (left, top + chart_height + 0.18, width, takeaway_height),
        theme,
    )
    return {"chart": chart, "takeaway": takeaway_shape}


def compose_visual_with_observations(
    slide,
    image_path: str | dict[str, Any] | None,
    observations: Sequence[str],
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
    caption: str = "",
):
    theme = theme or {}
    left, top, width, height = slot_rect
    gap = 0.22
    visual_width = max(width * 0.58, 1.0)
    note_width = max(width - visual_width - gap, 1.0)
    visual = add_visual_with_caption_block(
        slide,
        image_path,
        caption,
        (left, top, visual_width, height),
        theme,
    )
    notes = add_panel(
        slide,
        left + visual_width + gap,
        top,
        note_width,
        height,
        fill_color=theme.get("observation_fill", "#FFFFFF"),
        line_color=theme.get("primary_color", "#134E8E"),
    )
    add_bullet_list(
        slide,
        [str(item) for item in observations],
        left + visual_width + gap + 0.15,
        top + 0.18,
        max(note_width - 0.3, 0.2),
        max(height - 0.36, 0.25),
        font_size=13,
        color=theme.get("text_color", "#1F2937"),
        font_name=theme.get("font_family", "Aptos"),
    )
    return {"visual": visual, "observations": notes}


def compose_metrics_with_summary(
    slide,
    metrics: Sequence[dict[str, Any]],
    summary: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    metric_height = min(1.45, max(height * 0.38, 0.9))
    metrics_shape = add_metric_pair_block(slide, metrics, (left, top, width, metric_height), theme)
    summary_shape = add_takeaway_block(
        slide,
        summary,
        (left, top + metric_height + 0.18, width, max(height - metric_height - 0.18, 0.5)),
        theme,
    )
    return {"metrics": metrics_shape, "summary": summary_shape}


def apply_two_column_layout(slide, title: str, body_items: Sequence[str], image_path: str | None = None):
    add_title_box(slide, title)
    add_bullet_list(slide, body_items, 0.7, 1.5, 5.6, 4.9)
    if image_path:
        add_picture(slide, image_path, 7.0, 1.45, 5.2, 4.4)


def apply_three_column_layout(slide, title: str, columns: Sequence[Sequence[str]], headers: Sequence[str]):
    add_title_box(slide, title)
    add_comparison_columns(slide, headers, columns, 0.7, 1.5, 11.9, 4.9)


DEFAULT_LAYOUT_LIBRARY = {
    "title_only": [
        {"slot_id": "title", "x_ratio": 0.13, "y_ratio": 0.32, "w_ratio": 0.74, "h_ratio": 0.18},
        {"slot_id": "subtitle", "x_ratio": 0.58, "y_ratio": 0.78, "w_ratio": 0.34, "h_ratio": 0.09},
        {"slot_id": "footer", "x_ratio": 0.58, "y_ratio": 0.88, "w_ratio": 0.34, "h_ratio": 0.05},
    ],
    "cover": [
        {"slot_id": "title", "x_ratio": 0.09, "y_ratio": 0.25, "w_ratio": 0.72, "h_ratio": 0.16},
        {"slot_id": "subtitle", "x_ratio": 0.09, "y_ratio": 0.44, "w_ratio": 0.62, "h_ratio": 0.1},
        {"slot_id": "body", "x_ratio": 0.09, "y_ratio": 0.58, "w_ratio": 0.58, "h_ratio": 0.16},
    ],
    "bullets": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.08, "y_ratio": 0.24, "w_ratio": 0.72, "h_ratio": 0.5},
        {"slot_id": "callout", "x_ratio": 0.08, "y_ratio": 0.79, "w_ratio": 0.72, "h_ratio": 0.1},
    ],
    "hero": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.07, "w_ratio": 0.5, "h_ratio": 0.14},
        {"slot_id": "subtitle", "x_ratio": 0.06, "y_ratio": 0.19, "w_ratio": 0.42, "h_ratio": 0.09},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.31, "w_ratio": 0.4, "h_ratio": 0.45},
        {"slot_id": "image", "x_ratio": 0.52, "y_ratio": 0.12, "w_ratio": 0.42, "h_ratio": 0.68},
    ],
    "section_divider": [
        {"slot_id": "title", "x_ratio": 0.1, "y_ratio": 0.24, "w_ratio": 0.8, "h_ratio": 0.18},
        {"slot_id": "body", "x_ratio": 0.2, "y_ratio": 0.46, "w_ratio": 0.6, "h_ratio": 0.16},
    ],
    "two_column": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.23, "w_ratio": 0.42, "h_ratio": 0.62},
        {"slot_id": "image", "x_ratio": 0.54, "y_ratio": 0.23, "w_ratio": 0.38, "h_ratio": 0.56},
        {"slot_id": "callout", "x_ratio": 0.54, "y_ratio": 0.82, "w_ratio": 0.38, "h_ratio": 0.1},
    ],
    "three_column": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.25, "w_ratio": 0.27, "h_ratio": 0.5},
        {"slot_id": "supporting_body", "x_ratio": 0.365, "y_ratio": 0.25, "w_ratio": 0.27, "h_ratio": 0.5},
        {"slot_id": "callout", "x_ratio": 0.67, "y_ratio": 0.25, "w_ratio": 0.27, "h_ratio": 0.5},
    ],
    "comparison": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.08, "y_ratio": 0.24, "w_ratio": 0.84, "h_ratio": 0.58},
        {"slot_id": "callout", "x_ratio": 0.08, "y_ratio": 0.84, "w_ratio": 0.84, "h_ratio": 0.08},
    ],
    "metric_focus": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "metrics", "x_ratio": 0.06, "y_ratio": 0.25, "w_ratio": 0.88, "h_ratio": 0.28},
        {"slot_id": "body", "x_ratio": 0.1, "y_ratio": 0.58, "w_ratio": 0.8, "h_ratio": 0.22},
    ],
    "timeline": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.08, "y_ratio": 0.28, "w_ratio": 0.84, "h_ratio": 0.4},
        {"slot_id": "callout", "x_ratio": 0.1, "y_ratio": 0.75, "w_ratio": 0.8, "h_ratio": 0.12},
    ],
    "process_flow": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.08, "y_ratio": 0.24, "w_ratio": 0.84, "h_ratio": 0.42},
        {"slot_id": "supporting_body", "x_ratio": 0.1, "y_ratio": 0.7, "w_ratio": 0.8, "h_ratio": 0.16},
    ],
    "quadrant": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.08, "y_ratio": 0.24, "w_ratio": 0.39, "h_ratio": 0.27},
        {"slot_id": "supporting_body", "x_ratio": 0.53, "y_ratio": 0.24, "w_ratio": 0.39, "h_ratio": 0.27},
        {"slot_id": "metrics", "x_ratio": 0.08, "y_ratio": 0.56, "w_ratio": 0.39, "h_ratio": 0.27},
        {"slot_id": "callout", "x_ratio": 0.53, "y_ratio": 0.56, "w_ratio": 0.39, "h_ratio": 0.27},
    ],
    "image_focus": [
        {"slot_id": "image", "x_ratio": 0.0, "y_ratio": 0.0, "w_ratio": 1.0, "h_ratio": 1.0},
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.68, "w_ratio": 0.5, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.81, "w_ratio": 0.45, "h_ratio": 0.1},
    ],
    "full_bleed_image": [
        {"slot_id": "image", "x_ratio": 0.0, "y_ratio": 0.0, "w_ratio": 1.0, "h_ratio": 1.0},
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.68, "w_ratio": 0.5, "h_ratio": 0.12},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.81, "w_ratio": 0.45, "h_ratio": 0.1},
    ],
    "quote_callout": [
        {"slot_id": "title", "x_ratio": 0.08, "y_ratio": 0.1, "w_ratio": 0.8, "h_ratio": 0.12},
        {"slot_id": "callout", "x_ratio": 0.12, "y_ratio": 0.28, "w_ratio": 0.76, "h_ratio": 0.34},
        {"slot_id": "body", "x_ratio": 0.16, "y_ratio": 0.68, "w_ratio": 0.68, "h_ratio": 0.16},
        {"slot_id": "image", "x_ratio": 0.58, "y_ratio": 0.28, "w_ratio": 0.34, "h_ratio": 0.34},
    ],
    "table_focus": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "image", "x_ratio": 0.08, "y_ratio": 0.22, "w_ratio": 0.84, "h_ratio": 0.5},
        {"slot_id": "body", "x_ratio": 0.12, "y_ratio": 0.76, "w_ratio": 0.76, "h_ratio": 0.12},
    ],
    "chart_focus": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.06, "w_ratio": 0.88, "h_ratio": 0.12},
        {"slot_id": "image", "x_ratio": 0.1, "y_ratio": 0.22, "w_ratio": 0.8, "h_ratio": 0.5},
        {"slot_id": "body", "x_ratio": 0.12, "y_ratio": 0.76, "w_ratio": 0.76, "h_ratio": 0.12},
    ],
    "closing": [
        {"slot_id": "title", "x_ratio": 0.1, "y_ratio": 0.22, "w_ratio": 0.8, "h_ratio": 0.14},
        {"slot_id": "body", "x_ratio": 0.18, "y_ratio": 0.42, "w_ratio": 0.64, "h_ratio": 0.18},
        {"slot_id": "callout", "x_ratio": 0.28, "y_ratio": 0.72, "w_ratio": 0.44, "h_ratio": 0.1},
    ],
}


def resolve_layout_slots(
    slide_ir: dict[str, Any],
    slide_width: float = 13.33,
    slide_height: float = 7.5,
) -> dict[str, tuple[float, float, float, float]]:
    slide_ir = _runtime_slide_ir(slide_ir)
    layout = slide_ir.get("layout", {})
    layout_name = layout.get("name") or slide_ir.get("layout_type") or "two_column"
    raw_slots = layout.get("slots") or DEFAULT_LAYOUT_LIBRARY.get(layout_name, DEFAULT_LAYOUT_LIBRARY["two_column"])
    slots: dict[str, tuple[float, float, float, float]] = {}
    for slot in raw_slots:
        slot_id = slot.get("slot_id", "body")
        slots[slot_id] = (
            slide_width * float(slot.get("x_ratio", 0.0)),
            slide_height * float(slot.get("y_ratio", 0.0)),
            slide_width * float(slot.get("w_ratio", 1.0)),
            slide_height * float(slot.get("h_ratio", 1.0)),
        )
    return slots


def normalize_runtime_theme(theme: dict[str, Any] | None) -> dict[str, Any]:
    theme = dict(theme or {})
    primary = str(theme.get("primary") or theme.get("primary_color") or "#0F172A")
    secondary = str(theme.get("secondary") or theme.get("secondary_color") or "#475569")
    accent = str(theme.get("accent") or theme.get("accent_color") or "#2563EB")
    background = str(theme.get("background") or theme.get("background_color") or "#FFFFFF")
    title_font = str(theme.get("title_font") or theme.get("title_font_family") or theme.get("font_family") or "Aptos")
    body_font = str(theme.get("body_font") or theme.get("font_family") or "Aptos")
    normalized = {
        **theme,
        "primary_color": primary,
        "secondary_color": secondary,
        "accent_color": accent,
        "background_color": background,
        "text_color": str(theme.get("text_color") or secondary),
        "font_family": body_font,
        "title_font_family": title_font,
        "fit_text": theme.get("fit_text", True),
        "min_body_font_size": theme.get("min_body_font_size", 9),
        "min_bullet_font_size": theme.get("min_bullet_font_size", 9),
    }
    return normalized


def normalize_runtime_block(block: dict[str, Any]) -> dict[str, Any]:
    block = dict(block or {})
    kind = str(block.get("kind") or "").strip()
    role = str(block.get("role") or "").strip().lower()
    if not kind:
        if role == "title":
            kind = "headline"
        elif role in {"bullets", "bullet", "points"}:
            kind = "bullet_list"
        elif role in {"image", "visual"}:
            kind = "summary"
        else:
            kind = "summary" if block.get("text") or block.get("content") else "bullet_list"
    content = str(block.get("content") or block.get("text") or "").strip()
    items = [str(item).strip() for item in (block.get("items") or []) if str(item).strip()]
    slot_id = str(block.get("slot_id") or block.get("slot") or "").strip()
    if role == "title" and not slot_id:
        slot_id = "title"
    if role in {"bullets", "bullet", "points"} and not slot_id:
        slot_id = "body"
    return {**block, "kind": kind, "content": content, "items": items, "slot_id": slot_id}


def normalize_runtime_visual(visual: dict[str, Any] | None) -> dict[str, Any] | None:
    if visual is None:
        return None
    visual = dict(visual or {})
    role = str(visual.get("role") or "").strip().lower()
    slot_id = str(visual.get("slot_id") or visual.get("slot") or "").strip()
    if not slot_id:
        if role in {"hero", "primary", "image"}:
            slot_id = "image"
        else:
            slot_id = "image"
    selected_asset_path = (
        visual.get("selected_asset_path")
        or visual.get("asset_path")
        or visual.get("path")
        or visual.get("absolute_path")
        or visual.get("relative_path")
    )
    return {
        **visual,
        "slot_id": slot_id,
        "selected_asset_id": visual.get("selected_asset_id") or visual.get("asset_id"),
        "selected_asset_path": selected_asset_path,
        "asset_type": visual.get("asset_type") or visual.get("type"),
        "visual_id": visual.get("visual_id") or visual.get("id"),
    }


def build_runtime_body_blocks(slide_ir: dict[str, Any]) -> list[dict[str, Any]]:
    slide_ir = _runtime_slide_ir(slide_ir)
    blocks = [normalize_runtime_block(block) for block in (slide_ir.get("blocks") or [])]
    visible_blocks = [
        block
        for block in blocks
        if block.get("kind") != "headline"
        and (
            block.get("content")
            or block.get("items")
            or (
                block.get("kind") == "image"
                and (block.get("path") or block.get("asset_path") or block.get("asset_id"))
            )
        )
    ]
    if visible_blocks:
        return visible_blocks
    points = [str(item).strip() for item in (slide_ir.get("points") or []) if str(item).strip()]
    if points:
        return [{"kind": "bullet_list", "slot_id": "body", "items": points}]
    core_message = str(slide_ir.get("core_message") or "").strip()
    if core_message:
        return [{"kind": "summary", "slot_id": "body", "content": core_message}]
    return []


def normalize_runtime_slide_ir(slide_ir: dict[str, Any]) -> dict[str, Any]:
    return _runtime_slide_ir(slide_ir)


def resolve_asset_path(materials: dict[str, Any], slide_ir: dict[str, Any], visual: dict[str, Any] | None = None) -> str | None:
    slide_ir = _runtime_slide_ir(slide_ir)
    candidates = _asset_candidates(materials, slide_ir, visual)
    seen: set[str] = set()
    for candidate in candidates:
        for path in _candidate_asset_paths(candidate, slide_ir=slide_ir):
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                return path
    return None


def _asset_candidates(
    materials: dict[str, Any],
    slide_ir: dict[str, Any],
    visual: dict[str, Any] | None,
) -> list[Any]:
    slide_ir = _runtime_slide_ir(slide_ir)
    visual = normalize_runtime_visual(visual)
    current_slide = materials.get("current_slide") or {}
    assets = slide_ir.get("assets") or {}
    candidates: list[Any] = []
    visuals = [visual] if visual else []
    visuals.extend(candidate_visual for candidate_visual in (slide_ir.get("visuals") or []) if candidate_visual is not visual)
    for candidate_visual in visuals:
        if not candidate_visual:
            continue
        selected_candidate = candidate_visual.get("selected_candidate") or {}
        candidates.extend(
            [
                candidate_visual.get("selected_asset_path"),
                candidate_visual.get("asset_path"),
                selected_candidate,
                selected_candidate.get("path"),
                selected_candidate.get("absolute_path"),
                selected_candidate.get("relative_path"),
            ]
        )
        asset_id = candidate_visual.get("selected_asset_id") or candidate_visual.get("asset_id") or candidate_visual.get("use_existing_asset_id")
        if asset_id:
            asset = (materials.get("asset_index") or {}).get(asset_id)
            if asset:
                candidates.append(asset)
        slot_id = str(candidate_visual.get("slot_id") or candidate_visual.get("slot") or "").strip()
        if slot_id:
            for block in slide_ir.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                block_slot = str(block.get("slot_id") or block.get("slot") or "").strip()
                if block_slot != slot_id:
                    continue
                if block.get("kind") != "image" and not _is_image_slot(block_slot):
                    continue
                candidates.extend([block.get("path"), block.get("asset_path")])
                block_asset_id = block.get("asset_id") or block.get("selected_asset_id")
                if block_asset_id:
                    asset = (materials.get("asset_index") or {}).get(block_asset_id)
                    if asset:
                        candidates.append(asset)
                break
    candidates.extend(
        [
            slide_ir.get("selected_asset_path"),
            assets.get("selected_asset_path"),
            current_slide.get("selected_asset_path"),
            current_slide.get("selected_asset"),
        ]
    )
    selected_asset_id = slide_ir.get("selected_asset_id") or assets.get("selected_asset_id") or current_slide.get("selected_asset_id")
    if selected_asset_id:
        asset = (materials.get("asset_index") or {}).get(selected_asset_id)
        if asset:
            candidates.append(asset)
    for block in slide_ir.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("kind") != "image" and not _is_image_slot(block.get("slot")) and not _is_image_slot(block.get("slot_id")):
            continue
        candidates.extend([block.get("path"), block.get("asset_path")])
        asset_id = block.get("asset_id") or block.get("selected_asset_id")
        if asset_id:
            asset = (materials.get("asset_index") or {}).get(asset_id)
            if asset:
                candidates.append(asset)
    return candidates


def _candidate_asset_paths(candidate: Any, *, slide_ir: dict[str, Any]) -> list[str]:
    slide_ir = _runtime_slide_ir(slide_ir)
    raw_paths: list[str] = []
    if isinstance(candidate, dict):
        raw_paths.extend(
            str(candidate.get(key) or "").strip()
            for key in ("absolute_path", "path", "relative_path")
        )
    elif isinstance(candidate, str):
        raw_paths.append(candidate.strip())

    asset_base_dir = str(slide_ir.get("asset_base_dir") or "").strip()
    paths: list[str] = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        paths.append(raw_path)
        if asset_base_dir:
            path_obj = Path(raw_path).expanduser()
            if not path_obj.is_absolute():
                paths.append(str(Path(asset_base_dir).expanduser() / path_obj))
    return paths


def safe_resolve_asset_path(
    materials: dict[str, Any],
    slide_ir: dict[str, Any],
    visual: dict[str, Any] | None = None,
) -> str | None:
    try:
        return resolve_asset_path(materials, slide_ir, visual)
    except Exception:
        return None


def safe_placeholder_panel(
    slide,
    slot_rect: tuple[float, float, float, float],
    label: str = "visual unavailable",
    theme: dict[str, Any] | None = None,
):
    theme = theme or {}
    left, top, width, height = slot_rect
    add_panel(
        slide,
        left,
        top,
        width,
        height,
        fill_color=theme.get("placeholder_fill", "#F9FAFB"),
        line_color=theme.get("placeholder_line", theme.get("primary_color", "#CBD5E1")),
    )
    return add_textbox(
        slide,
        label,
        left + 0.15,
        top + max(height / 2 - 0.2, 0.05),
        max(width - 0.3, 0.2),
        min(0.45, max(height - 0.1, 0.25)),
        font_size=14,
        color=theme.get("placeholder_text", theme.get("text_color", "#475569")),
        bold=True,
        align="center",
    )


def render_block_in_slot(
    slide,
    block: dict[str, Any],
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any],
    font_name: str | None = None,
):
    block = normalize_runtime_block(block)
    left, top, width, height = slot_rect
    font_name = font_name or theme.get("font_family", "Aptos")
    kind = block.get("kind", "bullet_list")
    text_color = theme.get("text_color", "#1F2937")
    primary = theme.get("primary_color", "#134E8E")
    accent = theme.get("accent_color", "#FFB33F")
    secondary = theme.get("secondary_color", "#C00707")
    fit_text = bool(theme.get("fit_text"))
    min_body_font_size = int(theme.get("min_body_font_size", 8) or 8)
    min_bullet_font_size = int(theme.get("min_bullet_font_size", min_body_font_size) or min_body_font_size)

    if kind == "headline":
        title_font_name = theme.get("title_font_family") or f"{font_name} Display"
        return add_textbox(slide, block.get("content", ""), left, top, width, height, font_size=26, color=primary, bold=True, font_name=title_font_name, fit=fit_text, min_font_size=max(min_body_font_size, 16))
    if kind == "summary":
        return add_textbox(slide, block.get("content", ""), left, top, width, height, font_size=18, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_body_font_size)
    if kind == "bullet_list":
        return add_bullet_list(slide, block.get("items", []), left, top, width, height, font_size=16, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_bullet_font_size)
    if kind == "metric_strip":
        items = block.get("items", [])
        if not items:
            return add_textbox(slide, block.get("content", ""), left, top, width, height, font_size=16, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_body_font_size)
        card_width = max((width - 0.2 * (len(items) - 1)) / max(len(items), 1), 1.4)
        for idx, item in enumerate(items):
            label, _, value = item.partition(":")
            add_metric_card(
                slide,
                label.strip() or f"Metric {idx + 1}",
                (value.strip() or label.strip() or item).strip(),
                left + idx * (card_width + 0.2),
                top,
                card_width,
                height,
                accent_color=secondary,
                background_color="#FFF7ED",
                text_color=text_color,
            )
        return None
    if kind == "process":
        return add_process_flow(slide, block.get("items", []), left, top, width, height, accent_color=primary, text_color=text_color)
    if kind == "comparison":
        headers = [item.split(":", 1)[0].strip() for item in block.get("items", []) if ":" in item]
        columns = [[item.split(":", 1)[1].strip()] for item in block.get("items", []) if ":" in item]
        if len(headers) >= 2:
            return add_comparison_columns(slide, headers, columns, left, top, width, height)
        return add_bullet_list(slide, block.get("items", []), left, top, width, height, font_size=15, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_bullet_font_size)
    if kind == "quote":
        add_panel(slide, left, top, width, height, fill_color="#FFF7ED", line_color=accent)
        return add_textbox(slide, block.get("content", "") or " ".join(block.get("items", [])), left + 0.18, top + 0.12, width - 0.36, height - 0.24, font_size=18, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_body_font_size)
    if kind == "callout":
        line_x = left + 0.02
        line_top = top + min(0.08, max(height * 0.12, 0.02))
        line_bottom = top + height - min(0.08, max(height * 0.12, 0.02))
        if line_bottom > line_top:
            add_connector(slide, "STRAIGHT", line_x, line_top, line_x, line_bottom, color=accent)
        return add_textbox(slide, block.get("content", "") or " ".join(block.get("items", [])), left + 0.12, top + 0.02, max(width - 0.14, 0.2), max(height - 0.04, 0.2), font_size=16, color=text_color, bold=True, font_name=font_name, fit=fit_text, min_font_size=min_body_font_size)
    if kind == "image":
        return None
    return add_textbox(slide, block.get("content", "") or " ".join(block.get("items", [])), left, top, width, height, font_size=16, color=text_color, font_name=font_name, fit=fit_text, min_font_size=min_body_font_size)


def render_visual_in_slot(
    slide,
    slide_ir: dict[str, Any],
    materials: dict[str, Any],
    visual: dict[str, Any] | None,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any],
):
    visual = normalize_runtime_visual(visual)
    left, top, width, height = _inset_rect(slot_rect, pad=0.04)
    if visual is None:
        return safe_placeholder_panel(slide, (left, top, width, height), label="visual", theme=theme)
    asset_path = safe_resolve_asset_path(materials, slide_ir, visual)
    if asset_path:
        picture = safe_add_picture(slide, asset_path, left, top, width, height)
        if picture is not None:
            return picture
    placeholder = visual.get("intent") or "visual"
    return safe_placeholder_panel(slide, (left, top, width, height), label=placeholder, theme=theme)


def render_title_body_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    return _render_scaffold_with_layout(prs, deck_ir, slide_ir, materials, layout_name=_requested_or_fallback_layout(slide_ir, "section_divider"))


def render_title_body_visual_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    return _render_scaffold_with_layout(prs, deck_ir, slide_ir, materials, layout_name=_requested_or_fallback_layout(slide_ir, "two_column"))


def render_comparison_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    return _render_scaffold_with_layout(prs, deck_ir, slide_ir, materials, layout_name=_requested_or_fallback_layout(slide_ir, "comparison"))


def render_metric_focus_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    return _render_scaffold_with_layout(prs, deck_ir, slide_ir, materials, layout_name=_requested_or_fallback_layout(slide_ir, "metric_focus"))


def render_chart_focus_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    return _render_scaffold_with_layout(prs, deck_ir, slide_ir, materials, layout_name=_requested_or_fallback_layout(slide_ir, "chart_focus"))


def _requested_or_fallback_layout(slide_ir: dict[str, Any], fallback: str) -> str:
    slide_ir = _runtime_slide_ir(slide_ir)
    return str((slide_ir.get("layout") or {}).get("name") or slide_ir.get("layout_type") or fallback)


def render_slide_scaffold(prs, deck_ir: dict[str, Any], slide_ir: dict[str, Any], materials: dict[str, Any]):
    slide_ir = _runtime_slide_ir(slide_ir)
    from .layout_templates import render_from_template
    return render_from_template(prs, deck_ir, slide_ir, materials)


_CLOSING_TITLE_TOKENS = ("致谢", "谢谢", "thanks", "thank you", "closing")


def _deck_language(deck_ir: dict[str, Any]) -> str:
    raw = deck_ir.get("language") or (deck_ir.get("metadata") or {}).get("language") or "zh"
    lang = str(raw).strip().lower()
    return "en" if lang.startswith("en") else "zh"


def _presenter_fallback(deck_ir: dict[str, Any], slide_ir: dict[str, Any], lang: str) -> str:
    metadata = deck_ir.get("metadata") or {}
    candidates = [
        slide_ir.get("subtitle"),
        (slide_ir.get("content") or {}).get("subtitle"),
        metadata.get("presenter"),
        deck_ir.get("presenter"),
        deck_ir.get("author"),
        metadata.get("author"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return "汇报人：____" if lang == "zh" else "Presenter: ____"



def append_takeaway_block(
    slide,
    text: str,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
):
    return add_takeaway_block(slide, text, slot_rect, theme)


def emphasize_takeaway_block(shape, theme: dict[str, Any] | None = None):
    theme = theme or {}
    try:
        shape.line.color.rgb = _hex_to_rgb(theme.get("accent_color", "#C00707"))
        shape.line.width = Pt(2.0)
    except Exception:
        pass
    try:
        text_frame = shape.text_frame
        for paragraph in text_frame.paragraphs:
            paragraph.font.bold = True
            paragraph.font.color.rgb = _hex_to_rgb(theme.get("accent_color", "#C00707"))
    except Exception:
        pass
    return shape


def replace_visual_block(
    slide,
    image_path: str | dict[str, Any] | None,
    slot_rect: tuple[float, float, float, float],
    theme: dict[str, Any] | None = None,
    caption: str = "",
):
    return add_visual_with_caption_block(slide, image_path, caption, slot_rect, theme)


def tighten_text_spacing(shape, level: str = "compact"):
    spacing_map = {
        "compact": 0.92,
        "tight": 0.84,
        "relaxed": 1.08,
    }
    scale = spacing_map.get(level, 0.92)
    try:
        text_frame = shape.text_frame
        for paragraph in text_frame.paragraphs:
            if paragraph.font.size:
                paragraph.font.size = Pt(max(int(paragraph.font.size.pt * scale), 8))
            paragraph.space_after = Pt(0)
            paragraph.space_before = Pt(0)
    except Exception:
        pass
    return shape


def rebalance_visual_text_ratio(text_shape, visual_shape, ratio: str = "balanced"):
    result = {"text": text_shape, "visual": visual_shape, "ratio": ratio}
    text_width_map = {
        "visual_heavy": 0.88,
        "text_heavy": 1.12,
        "balanced": 1.0,
    }
    line_width_map = {
        "visual_heavy": 1.6,
        "text_heavy": 1.0,
        "balanced": 1.3,
    }
    try:
        text_frame = text_shape.text_frame
        for paragraph in text_frame.paragraphs:
            if paragraph.font.size:
                paragraph.font.size = Pt(max(int(paragraph.font.size.pt * text_width_map.get(ratio, 1.0)), 8))
    except Exception:
        pass
    try:
        visual_shape.line.width = Pt(line_width_map.get(ratio, 1.3))
    except Exception:
        pass
    return result


def _render_scaffold_with_layout(
    prs,
    deck_ir: dict[str, Any],
    slide_ir: dict[str, Any],
    materials: dict[str, Any],
    layout_name: str | None = None,
):
    slide_ir = _runtime_slide_ir(slide_ir)
    if layout_name:
        layout = dict(slide_ir.get("layout", {}))
        if not layout.get("name"):
            layout["name"] = layout_name
        slide_ir = {**slide_ir, "layout": layout}
    slide = add_blank_slide(prs)
    theme = normalize_runtime_theme(deck_ir.get("theme", {}))
    set_background_color(slide, theme.get("background_color", "#F7F4EE"))
    slots = resolve_layout_slots(slide_ir)
    runtime_layout = dict(slide_ir.get("layout") or {})
    effective_layout_name = str(runtime_layout.get("name") or slide_ir.get("layout_type") or layout_name or "").strip()
    title_align = str(runtime_layout.get("title_align") or ("center" if effective_layout_name == "title_only" else "left"))
    subtitle_align = str(runtime_layout.get("subtitle_align") or ("right" if effective_layout_name == "title_only" else "left"))

    visuals = [normalize_runtime_visual(visual) for visual in (slide_ir.get("visuals") or [])]
    has_explicit_image_block = any(
        isinstance(block, dict)
        and (block.get("kind") == "image" or _is_image_slot(block.get("slot_id")) or _is_image_slot(block.get("slot")))
        for block in slide_ir.get("blocks") or []
    )
    if (
        not visuals
        and not has_explicit_image_block
        and (slide_ir.get("selected_asset_path") or (materials.get("current_slide") or {}).get("selected_asset_path"))
    ):
        visual_slot = "image"
        visuals = [{"slot_id": visual_slot, "intent": "selected asset"}]
    visual_first_slots = {
        slot_id
        for slot_id in slots
        if _is_image_slot(slot_id)
    } if effective_layout_name in {"image_focus", "full_bleed_image"} else set()
    rendered_visual_indexes: set[int] = set()

    def render_visuals_for_slots(slot_ids: set[str]) -> None:
        for index, visual in enumerate(visuals):
            if visual is None or index in rendered_visual_indexes:
                continue
            slot_id = visual.get("slot_id")
            if slot_ids and slot_id not in slot_ids:
                continue
            slot_rect = slots.get(slot_id) or slots.get("image") or slots.get("supporting_visual") or slots.get("hero_visual")
            if slot_rect:
                render_visual_in_slot(slide, slide_ir, materials, visual, slot_rect, theme)
                rendered_visual_indexes.add(index)

    def mark_matching_visual_rendered(rendered_visual: dict[str, Any]) -> None:
        rendered_visual = normalize_runtime_visual(rendered_visual) or {}
        rendered_slot = str(rendered_visual.get("slot_id") or rendered_visual.get("slot") or "").strip()
        rendered_asset_id = str(rendered_visual.get("selected_asset_id") or rendered_visual.get("asset_id") or "").strip()
        rendered_path = str(rendered_visual.get("selected_asset_path") or rendered_visual.get("path") or "").strip()
        for index, visual in enumerate(visuals):
            if visual is None or index in rendered_visual_indexes:
                continue
            visual_slot = str(visual.get("slot_id") or visual.get("slot") or "").strip()
            if rendered_slot and visual_slot and rendered_slot != visual_slot:
                continue
            visual_asset_id = str(visual.get("selected_asset_id") or visual.get("asset_id") or "").strip()
            if rendered_asset_id and visual_asset_id and rendered_asset_id != visual_asset_id:
                continue
            visual_path = str(visual.get("selected_asset_path") or visual.get("path") or "").strip()
            if rendered_path and visual_path and rendered_path != visual_path:
                continue
            rendered_visual_indexes.add(index)

    if visual_first_slots:
        render_visuals_for_slots(visual_first_slots)

    title_rect = slots.get("title")
    if title_rect and slide_ir.get("title"):
        add_title_box(
            slide,
            slide_ir["title"],
            left=title_rect[0],
            top=title_rect[1],
            width=title_rect[2],
            height=title_rect[3],
            color=theme.get("primary_color", "#134E8E"),
            font_name=f"{theme.get('font_family', 'Aptos')} Display",
            font_size=34 if effective_layout_name == "title_only" else 28,
            align=title_align,
        )
    subtitle_rect = slots.get("subtitle")
    if subtitle_rect and slide_ir.get("subtitle"):
        add_subtitle_box(
            slide,
            slide_ir["subtitle"],
            left=subtitle_rect[0],
            top=subtitle_rect[1],
            width=subtitle_rect[2],
            height=subtitle_rect[3],
            color=theme.get("text_color", "#1F2937"),
            font_name=theme.get("font_family", "Aptos"),
            align=subtitle_align,
        )

    body_fallback = slots.get("body", (0.9, 1.8, 5.6, 4.8))
    supporting_body = slots.get("supporting_body", body_fallback)
    metrics_rect = slots.get("metrics", body_fallback)
    callout_rect = slots.get("callout", supporting_body)
    block_slot_defaults = {
        "headline": title_rect or body_fallback,
        "summary": body_fallback,
        "bullet_list": body_fallback,
        "metric_strip": metrics_rect,
        "process": body_fallback,
        "comparison": body_fallback,
        "quote": callout_rect,
        "callout": callout_rect,
        "image": slots.get("image") or slots.get("supporting_visual") or slots.get("hero_visual") or body_fallback,
    }

    blocks_by_slot: dict[str, list[dict[str, Any]]] = {}
    for block in build_runtime_body_blocks(slide_ir):
        if (
            block.get("kind") == "headline"
            and title_rect
            and (block.get("slot_id") in {"title", "", None} or block.get("content", "").strip() == slide_ir.get("title", "").strip())
        ):
            continue
        if (
            block.get("kind") == "summary"
            and subtitle_rect
            and block.get("slot_id") == "subtitle"
        ):
            continue
        if block.get("kind") == "image":
            slot_id = block.get("slot_id") or "image"
            slot_rect = slots.get(slot_id) or slots.get("image") or block_slot_defaults["image"]
            block_visual = {
                "slot_id": slot_id,
                "caption": block.get("caption"),
                "intent": block.get("description") or block.get("caption"),
                "selected_asset_path": block.get("path"),
                "selected_asset_id": block.get("asset_id"),
                "asset_type": "image",
            }
            render_visual_in_slot(
                slide,
                slide_ir,
                materials,
                block_visual,
                slot_rect,
                theme,
            )
            mark_matching_visual_rendered(block_visual)
            continue
        slot_id = block.get("slot_id", "")
        blocks_by_slot.setdefault(slot_id, []).append(block)

    for slot_id, grouped_blocks in blocks_by_slot.items():
        base_rect = slots.get(slot_id, block_slot_defaults.get(grouped_blocks[0].get("kind", "bullet_list"), body_fallback))
        sub_rects = _stack_slot_rects(base_rect, len(grouped_blocks))
        for block, slot_rect in zip(grouped_blocks, sub_rects):
            render_block_in_slot(slide, block, slot_rect, theme, font_name=theme.get("font_family", "Aptos"))

    render_visuals_for_slots(set())

    return slide


def _runtime_slide_ir(slide_ir: dict[str, Any]) -> dict[str, Any]:
    slide_ir = dict(slide_ir or {})
    content = dict(slide_ir.get("content") or {})
    layout = dict(slide_ir.get("layout") or {})
    assets = dict(slide_ir.get("assets") or {})
    notes = dict(slide_ir.get("notes") or {})
    blocks = [normalize_runtime_block(block) for block in (slide_ir.get("blocks") or []) if isinstance(block, dict)]
    title = str(content.get("title") or slide_ir.get("title") or _first_runtime_block_text(blocks, "title") or "").strip()
    subtitle = str(content.get("subtitle") or slide_ir.get("subtitle") or _first_runtime_block_text(blocks, "subtitle") or "").strip()
    points = [
        str(item).strip()
        for item in (content.get("points") or slide_ir.get("points") or [])
        if str(item).strip()
    ]
    if not points:
        points = _first_runtime_block_items(blocks, "body")
    core_message = str(
        content.get("core_message")
        or slide_ir.get("core_message")
        or _first_runtime_block_text(blocks, "callout")
        or _first_runtime_block_text(blocks, "body")
        or (points[0] if points else "")
        or ""
    ).strip()
    selected_asset_path = str(assets.get("selected_asset_path") or slide_ir.get("selected_asset_path") or "").strip()
    selected_asset_id = str(assets.get("selected_asset_id") or slide_ir.get("selected_asset_id") or "").strip()
    if not selected_asset_path:
        for block in blocks:
            if block.get("kind") == "image" or _is_image_slot(block.get("slot")) or _is_image_slot(block.get("slot_id")):
                selected_asset_path = str(block.get("path") or block.get("asset_path") or "").strip()
                selected_asset_id = str(block.get("asset_id") or block.get("selected_asset_id") or selected_asset_id).strip()
                break
    asset_paths = [
        str(path).strip()
        for path in (assets.get("paths") or slide_ir.get("asset_paths") or [])
        if str(path).strip()
    ]
    if not asset_paths and selected_asset_path:
        asset_paths = [selected_asset_path]
    return {
        **slide_ir,
        "layout": layout,
        "layout_type": layout.get("name") or slide_ir.get("layout_type") or "section_divider",
        "title": title,
        "subtitle": subtitle,
        "core_message": core_message,
        "points": points,
        "blocks": blocks,
        "selected_asset_id": selected_asset_id,
        "selected_asset_path": selected_asset_path,
        "asset_paths": asset_paths,
        "source_evidence": slide_ir.get("evidence") or slide_ir.get("source_evidence") or [],
        "design_notes": notes.get("design") or slide_ir.get("design_notes") or [],
        "speaker_notes": notes.get("speaker") or slide_ir.get("speaker_notes") or [],
    }


def _is_image_slot(slot_id: Any) -> bool:
    value = str(slot_id or "").strip()
    return value == "image" or (value.startswith("image_") and value[6:].isdigit())


def _first_runtime_block_text(blocks: list[dict[str, Any]], role_or_slot: str) -> str:
    for block in blocks:
        role = str(block.get("role") or "").strip()
        slot_id = str(block.get("slot_id") or block.get("slot") or "").strip()
        kind = str(block.get("kind") or "").strip()
        if role_or_slot not in {role, slot_id, kind}:
            continue
        text = str(block.get("content") or block.get("text") or "").strip()
        if text:
            return text
    return ""


def _first_runtime_block_items(blocks: list[dict[str, Any]], role_or_slot: str) -> list[str]:
    for block in blocks:
        role = str(block.get("role") or "").strip()
        slot_id = str(block.get("slot_id") or block.get("slot") or "").strip()
        if role_or_slot not in {role, slot_id}:
            continue
        items = [str(item).strip() for item in (block.get("items") or []) if str(item).strip()]
        if items:
            return items
    return []


def _stack_slot_rects(slot_rect: tuple[float, float, float, float], count: int) -> list[tuple[float, float, float, float]]:
    if count <= 1:
        return [slot_rect]
    left, top, width, height = slot_rect
    gap = 0.14
    total_gap = gap * (count - 1)
    unit_height = max((height - total_gap) / count, 0.45)
    rects = []
    current_top = top
    for _ in range(count):
        rects.append((left, current_top, width, unit_height))
        current_top += unit_height + gap
    return rects


def _inset_rect(slot_rect: tuple[float, float, float, float], pad: float = 0.02) -> tuple[float, float, float, float]:
    left, top, width, height = slot_rect
    horizontal_pad = min(pad, width / 8)
    vertical_pad = min(pad, height / 8)
    return (
        left + horizontal_pad,
        top + vertical_pad,
        max(width - 2 * horizontal_pad, 0.2),
        max(height - 2 * vertical_pad, 0.2),
    )


def _hex_to_rgb(color: str) -> RGBColor:
    value = color.lstrip("#")
    if len(value) == 8:
        value = value[:6]
    if len(value) != 6:
        raise ValueError(f"Expected 6-char hex color, got: {color}")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _map_align(align: str):
    mapping = {
        "left": PP_ALIGN.LEFT,
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
        "justify": PP_ALIGN.JUSTIFY,
    }
    return mapping.get(align, PP_ALIGN.LEFT)
