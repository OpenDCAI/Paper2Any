from __future__ import annotations

from typing import Any

from .ir_models import IRMetadata, SlideBrief, SlideBriefDeck, Storyline


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_points(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    text = _normalize_text(value)
    return [text] if text else []


def _slide_type(item: dict[str, Any], *, page_num: int, points: list[str]) -> str:
    if item.get("ppt_img_path") or item.get("asset_paths"):
        return "direct_image"
    title = _normalize_text(item.get("title")).lower()
    layout_description = _normalize_text(item.get("layout_description")).lower()
    if page_num == 1 and len(points) <= 2:
        return "title"
    if "致谢" in title or "thank" in title or "closing" in layout_description:
        return "closing"
    return "content"


def _build_slide_brief(item: dict[str, Any], page_num: int) -> SlideBrief:
    title = _normalize_text(item.get("title")) or f"Slide {page_num}"
    summary = _normalize_text(item.get("summary"))
    points = _normalize_points(item.get("key_points") or item.get("bullets"))
    core_message = points[0] if points else summary or title
    slide_type = _slide_type(item, page_num=page_num, points=points)

    return SlideBrief(
        brief_id=f"brief-{page_num:03d}",
        slide_id=f"slide-{page_num:03d}",
        type=slide_type,
        section_id=f"section-{page_num:03d}",
        section_title=title,
        title=title,
        core_message=core_message,
        objective=summary or title,
        content_points=points,
        source_chunk_ids=[f"pagecontent-{page_num:03d}"],
        source_headings=[title],
        source_excerpt=summary,
    )


def pagecontent_to_slide_briefs(
    pagecontent: list[dict[str, Any]],
    *,
    language: str,
    style: str,
) -> dict[str, Any]:
    """Normalize pagecontent into the slide_briefs artifact consumed by planning."""

    briefs = [_build_slide_brief(item, index) for index, item in enumerate(pagecontent or [], start=1)]
    title_hint = briefs[0].title if briefs else "Editable Deck"
    subtitle_hint = ""
    storyline = Storyline(
        summary=" -> ".join(brief.title for brief in briefs[:5]),
        sections=[brief.title for brief in briefs],
    )
    brief_deck = SlideBriefDeck(
        metadata=IRMetadata(
            schema_name="presentagent.slide_briefs",
            language=language,
            style=style,
        ),
        title_hint=title_hint,
        subtitle_hint=subtitle_hint,
        storyline_hint=storyline,
        slide_briefs=briefs,
        planner_notes=["constructed from pagecontent"],
    )
    return brief_deck.model_dump()
