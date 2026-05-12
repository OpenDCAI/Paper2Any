from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

from ..planner.ir_models import DeckIR, SlideIR
from ..contracts import SlideArtifact
from .pptx_library import (
    add_blank_slide as library_add_blank_slide,
    add_title_box as library_add_title_box,
    create_presentation as library_create_presentation,
    render_slide_scaffold,
    set_background_color as library_set_background_color,
)

SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5


def render_recipe(deck_ir: DeckIR) -> dict[str, object]:
    return {
        "deck_title": deck_ir.title,
        "storyline": deck_ir.storyline.model_dump() if getattr(deck_ir, "storyline", None) else {},
        "theme": deck_ir.theme.model_dump(),
        "slide_count": len(deck_ir.slides),
        "slides": [
            {
                "slide_id": slide.slide_id,
                "page_num": slide.page_num,
                "layout_type": slide.layout_type,
                "layout": dict(slide.layout),
                "title": slide.title,
                "core_message": slide.core_message,
                "blocks": list(slide.blocks),
                "points": list(slide.points),
                "visuals": [visual.model_dump() for visual in slide.visuals],
                "selected_asset_path": slide.selected_asset_path,
                "asset_paths": list(slide.asset_paths),
            }
            for slide in deck_ir.slides
        ],
    }


def render_pptx(deck_ir: DeckIR, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    presentation = library_create_presentation(SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES)
    _apply_theme_defaults(presentation, deck_ir)

    deck_payload = deck_ir.model_dump()
    for slide_ir in deck_ir.slides:
        render_slide_scaffold(
            presentation,
            deck_payload,
            slide_ir.model_dump(),
            _build_library_materials(deck_ir, slide_ir),
        )

    if not deck_ir.slides:
        slide = library_add_blank_slide(presentation)
        library_set_background_color(slide, deck_ir.theme.background)
        library_add_title_box(slide, deck_ir.title or "Editable Deck", top=1.2, height=0.8, font_size=26)

    presentation.save(output_path)


def render_pptx_per_slide(deck_ir: DeckIR, slides_dir: Path) -> list[SlideArtifact]:
    """Render each slide into its own standalone pptx file."""
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[SlideArtifact] = []
    if not deck_ir.slides:
        return artifacts

    deck_payload_base = deck_ir.model_dump()
    for idx, slide_ir in enumerate(deck_ir.slides):
        single_path = (slides_dir / f"slide_{idx:03d}.pptx").resolve()
        presentation = library_create_presentation(SLIDE_WIDTH_INCHES, SLIDE_HEIGHT_INCHES)
        _apply_theme_defaults(presentation, deck_ir)
        render_slide_scaffold(
            presentation,
            deck_payload_base,
            slide_ir.model_dump(),
            _build_library_materials(deck_ir, slide_ir),
        )
        presentation.save(single_path)
        artifacts.append(
            SlideArtifact(
                index=idx,
                slide_id=slide_ir.slide_id,
                title=slide_ir.title or "",
                pptx_path=str(single_path),
            )
        )
    return artifacts


def _build_library_materials(deck_ir: DeckIR, slide_ir: SlideIR) -> dict[str, object]:
    asset_index = {
        str(asset_id): dict(asset)
        for asset_id, asset in (deck_ir.source_asset_index or {}).items()
        if isinstance(asset, dict)
    }
    assets = list(asset_index.values())
    selected_asset_id = str(slide_ir.selected_asset_id or "").strip()
    selected_asset_path = str(slide_ir.selected_asset_path or "").strip()
    selected_asset = dict(asset_index.get(selected_asset_id, {})) if selected_asset_id else {}
    if not selected_asset and selected_asset_path:
        for asset in assets:
            if selected_asset_path in {
                str(asset.get("path") or "").strip(),
                str(asset.get("relative_path") or "").strip(),
                str(asset.get("absolute_path") or "").strip(),
            }:
                selected_asset = dict(asset)
                break
    return {
        "theme": deck_ir.theme.model_dump(),
        "assets": assets,
        "asset_index": asset_index,
        "asset_catalog": assets,
        "asset_request_contexts": [],
        "resolution": {"requests": []},
        "current_slide": {
            "slide_id": slide_ir.slide_id,
            "selected_asset_id": selected_asset_id or str(selected_asset.get("asset_id") or ""),
            "selected_asset_path": selected_asset_path or str(selected_asset.get("path") or ""),
            "selected_asset": selected_asset,
            "resolution": {},
        },
    }


def _apply_theme_defaults(presentation: Presentation, deck_ir: DeckIR) -> None:
    properties = presentation.core_properties
    properties.title = deck_ir.title or "Editable Deck"
    properties.author = "presentagent_runtime"
    properties.subject = deck_ir.style or "editable ppt runtime"
    properties.language = deck_ir.language


def _render_slide(slide, slide_ir: SlideIR, deck_ir: DeckIR) -> None:
    layout_type = slide_ir.layout_type
    if layout_type == "cover":
        _render_cover(slide, slide_ir, deck_ir)
        return
    if layout_type == "bullets":
        _render_bullets(slide, slide_ir, deck_ir)
        return
    if layout_type == "two_column":
        _render_two_column(slide, slide_ir, deck_ir)
        return
    if layout_type == "image_focus":
        _render_image_focus(slide, slide_ir, deck_ir)
        return
    if layout_type == "full_bleed_image":
        _render_full_bleed_image(slide, slide_ir)
        return
    _render_bullets(slide, slide_ir, deck_ir)


def _render_cover(slide, slide_ir: SlideIR, deck_ir: DeckIR) -> None:
    _add_title(slide, slide_ir.title or deck_ir.title or "Editable Deck", deck_ir, top=1.35, height=1.0, font_size=26)
    subtitle = slide_ir.core_message or "\n".join(slide_ir.points[:2])
    if subtitle:
        text_box = slide.shapes.add_textbox(Inches(1.25), Inches(2.65), Inches(10.5), Inches(1.8))
        frame = text_box.text_frame
        frame.word_wrap = True
        frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        paragraph = frame.paragraphs[0]
        paragraph.text = subtitle
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.runs[0]
        run.font.size = Pt(18)
        run.font.name = deck_ir.theme.body_font
        run.font.color.rgb = _rgb(deck_ir.theme.secondary)


def _render_bullets(slide, slide_ir: SlideIR, deck_ir: DeckIR) -> None:
    _add_title(slide, slide_ir.title or deck_ir.title or "Slide", deck_ir)
    body_lines = slide_ir.points or ([slide_ir.core_message] if slide_ir.core_message else [])
    _add_bullet_box(slide, body_lines, deck_ir, left=1.2, top=1.75, width=10.6, height=4.9)


def _render_two_column(slide, slide_ir: SlideIR, deck_ir: DeckIR) -> None:
    _add_title(slide, slide_ir.title or deck_ir.title or "Slide", deck_ir)
    left_lines = slide_ir.points or ([slide_ir.core_message] if slide_ir.core_message else [])
    right_lines: list[str] = []
    asset_candidates = [slide_ir.selected_asset_path] if slide_ir.selected_asset_path else list(slide_ir.asset_paths)
    if asset_candidates:
        right_lines.extend(f"Asset: {Path(asset).name}" for asset in asset_candidates[:2] if asset)

    _add_panel(slide, left=0.8, top=1.6, width=5.8, height=4.9, fill_color="#F8FAFC")
    _add_panel(slide, left=6.75, top=1.6, width=5.8, height=4.9, fill_color="#F8FAFC")
    _add_bullet_box(slide, left_lines, deck_ir, left=1.0, top=1.9, width=5.2, height=4.2)
    _add_bullet_box(slide, right_lines, deck_ir, left=6.95, top=1.9, width=5.2, height=4.2)


def _render_image_focus(slide, slide_ir: SlideIR, deck_ir: DeckIR) -> None:
    _add_title(slide, slide_ir.title or deck_ir.title or "Slide", deck_ir)
    asset_path = _first_existing_asset(slide_ir)
    if asset_path:
        slide.shapes.add_picture(str(asset_path), Inches(0.9), Inches(1.6), width=Inches(7.3), height=Inches(4.9))
    else:
        _add_panel(slide, left=0.9, top=1.6, width=7.3, height=4.9, fill_color="#E2E8F0", line_color="#CBD5E1")

    notes = slide_ir.points or ([slide_ir.core_message] if slide_ir.core_message else [])
    _add_panel(slide, left=8.45, top=1.6, width=3.95, height=4.9, fill_color="#F8FAFC")
    _add_bullet_box(slide, notes, deck_ir, left=8.7, top=1.95, width=3.45, height=4.2)


def _render_full_bleed_image(slide, slide_ir: SlideIR) -> None:
    asset_path = _first_existing_asset(slide_ir)
    if asset_path:
        slide.shapes.add_picture(
            str(asset_path),
            0,
            0,
            width=Inches(SLIDE_WIDTH_INCHES),
            height=Inches(SLIDE_HEIGHT_INCHES),
        )
        return
    _add_panel(
        slide,
        left=0.0,
        top=0.0,
        width=SLIDE_WIDTH_INCHES,
        height=SLIDE_HEIGHT_INCHES,
        fill_color="#E5E7EB",
        line_color="#CBD5E1",
    )


def _add_title(slide, text: str, deck_ir: DeckIR, *, top: float = 0.55, height: float = 0.75, font_size: int = 22) -> None:
    text_box = slide.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.8), Inches(height))
    frame = text_box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.runs[0]
    run.font.bold = True
    run.font.size = Pt(font_size)
    run.font.name = deck_ir.theme.title_font
    run.font.color.rgb = _rgb(deck_ir.theme.primary)


def _add_bullet_box(slide, lines: list[str], deck_ir: DeckIR, *, left: float, top: float, width: float, height: float) -> None:
    text_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = text_box.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    frame.clear()

    normalized_lines = [line for line in lines if str(line).strip()]
    if not normalized_lines:
        normalized_lines = [" "]

    for index, line in enumerate(normalized_lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = str(line)
        paragraph.level = 0
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.runs[0]
        run.font.size = Pt(17)
        run.font.name = deck_ir.theme.body_font
        run.font.color.rgb = _rgb(deck_ir.theme.secondary)
        if normalized_lines != [" "]:
            paragraph.bullet = True


def _add_panel(
    slide,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: str,
    line_color: str | None = None,
) -> None:
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(fill_color)
    if line_color:
        shape.line.color.rgb = _rgb(line_color)
    else:
        shape.line.fill.background()


def _paint_background(slide, color_hex: str) -> None:
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = _rgb(color_hex)


def _first_existing_asset(slide_ir: SlideIR) -> Path | None:
    asset_candidates = []
    if slide_ir.selected_asset_path:
        asset_candidates.append(slide_ir.selected_asset_path)
    asset_candidates.extend(slide_ir.asset_paths)
    for asset in asset_candidates:
        raw_asset = str(asset).strip()
        if not raw_asset:
            continue
        path = Path(raw_asset).expanduser()
        if not path.is_absolute() and slide_ir.asset_base_dir:
            path = Path(slide_ir.asset_base_dir).expanduser() / path
        if path.exists():
            return path.resolve()
    return None


def _rgb(value: str) -> RGBColor:
    normalized = value.strip().lstrip("#")
    if len(normalized) != 6:
        normalized = "000000"
    return RGBColor.from_string(normalized.upper())
