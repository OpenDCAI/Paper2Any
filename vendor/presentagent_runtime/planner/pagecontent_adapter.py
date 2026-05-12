from __future__ import annotations

from pathlib import Path
from typing import Any

from .ir_models import (
    ALLOWED_LAYOUTS,
    ALLOWED_VISUAL_TYPES,
    DeckIR,
    DeckTheme,
    IRMetadata,
    LEGACY_LAYOUT_ALIASES,
    MaterialRequest,
    SlideIR,
    SlideLayoutType,
    Storyline,
    VisualBinding,
)


PAGECONTENT_LAYOUT_SLOTS: dict[str, list[dict[str, float | str]]] = {
    "hero": [
        {"slot_id": "title", "x_ratio": 0.06, "y_ratio": 0.07, "w_ratio": 0.5, "h_ratio": 0.14},
        {"slot_id": "subtitle", "x_ratio": 0.06, "y_ratio": 0.19, "w_ratio": 0.42, "h_ratio": 0.09},
        {"slot_id": "body", "x_ratio": 0.06, "y_ratio": 0.31, "w_ratio": 0.4, "h_ratio": 0.45},
        {"slot_id": "image", "x_ratio": 0.52, "y_ratio": 0.12, "w_ratio": 0.42, "h_ratio": 0.68},
    ],
    "title_only": [
        {"slot_id": "title", "x_ratio": 0.13, "y_ratio": 0.32, "w_ratio": 0.74, "h_ratio": 0.18},
        {"slot_id": "subtitle", "x_ratio": 0.58, "y_ratio": 0.78, "w_ratio": 0.34, "h_ratio": 0.09},
        {"slot_id": "footer", "x_ratio": 0.58, "y_ratio": 0.88, "w_ratio": 0.34, "h_ratio": 0.05},
    ],
    "section_divider": [
        {"slot_id": "title", "x_ratio": 0.1, "y_ratio": 0.24, "w_ratio": 0.8, "h_ratio": 0.18},
        {"slot_id": "body", "x_ratio": 0.2, "y_ratio": 0.46, "w_ratio": 0.6, "h_ratio": 0.16},
        {"slot_id": "callout", "x_ratio": 0.2, "y_ratio": 0.66, "w_ratio": 0.6, "h_ratio": 0.1},
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


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_points(raw_points: Any) -> list[str]:
    if not raw_points:
        return []
    if not isinstance(raw_points, list):
        return [_normalize_text(raw_points)] if _normalize_text(raw_points) else []

    points: list[str] = []
    for point in raw_points:
        normalized = _normalize_text(point)
        if normalized:
            points.append(normalized)
    return points


def _normalize_summary(item: dict[str, Any]) -> str:
    return _normalize_text(item.get("summary"))


def _normalize_caption(item: dict[str, Any]) -> str:
    for key in (
        "caption",
        "image_caption",
        "figure_caption",
        "table_caption",
        "asset_caption",
    ):
        caption = _normalize_text(item.get(key))
        if caption:
            return caption
    return ""


def _normalize_bullets(item: dict[str, Any]) -> list[str]:
    raw_points = item["bullets"] if "bullets" in item else item.get("key_points")
    return _normalize_points(raw_points)


def _normalize_asset_paths(raw_assets: Any) -> list[str]:
    if not raw_assets:
        return []
    if isinstance(raw_assets, list):
        return [path for path in (_normalize_text(raw_asset) for raw_asset in raw_assets) if path]

    asset_path = _normalize_text(raw_assets)
    return [asset_path] if asset_path else []


def _normalize_direct_image_asset_paths(item: dict[str, Any]) -> list[str]:
    if "asset_paths" in item:
        return _normalize_asset_paths(item.get("asset_paths"))
    return _normalize_asset_paths(item.get("ppt_img_path"))


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _layout_text(*parts: Any) -> str:
    return " ".join(_normalize_text(part).lower() for part in parts if _normalize_text(part))


def _canonical_layout(layout_type: str) -> SlideLayoutType:
    normalized = _normalize_text(layout_type).lower().replace("-", "_").replace(" ", "_")
    if normalized in ALLOWED_LAYOUTS:
        return normalized  # type: ignore[return-value]
    if normalized in LEGACY_LAYOUT_ALIASES:
        return LEGACY_LAYOUT_ALIASES[normalized]  # type: ignore[return-value]
    return "section_divider"


def _is_known_layout(layout_type: str) -> bool:
    normalized = _normalize_text(layout_type).lower().replace("-", "_").replace(" ", "_")
    return normalized in ALLOWED_LAYOUTS or normalized in LEGACY_LAYOUT_ALIASES


def _layout_from_text_hint(text: str) -> SlideLayoutType | None:
    normalized = _normalize_text(text).lower().replace("-", "_").replace(" ", "_")
    for layout_name in ALLOWED_LAYOUTS:
        if layout_name in normalized:
            return layout_name  # type: ignore[return-value]
    return None


def _select_outline_layout(
    *,
    page_num: int,
    title: str,
    summary: str,
    bullets: list[str],
    asset_paths: list[str],
    explicit_layout: str = "",
    layout_hint: str = "",
) -> SlideLayoutType:
    if _is_known_layout(explicit_layout):
        return _canonical_layout(explicit_layout)

    description = _layout_text(title, summary, layout_hint, " ".join(bullets))
    hinted_layout = _layout_from_text_hint(description)
    if hinted_layout:
        return hinted_layout
    has_asset = bool(asset_paths)

    if _contains_any(description, ("closing", "thank", "thanks", "致谢", "总结页", "结束页")):
        return "closing"
    if page_num == 1 and title and not has_asset:
        return "title_only"
    if _contains_any(description, ("cover", "封面", "首页", "title only", "title_only", "标题页")):
        return "title_only"
    if _contains_any(description, ("section", "chapter", "章节", "分节", "过渡页")):
        return "section_divider"
    if _contains_any(description, ("quadrant", "四象限", "2x2", "2×2")):
        return "quadrant"
    if _contains_any(description, ("three column", "three-column", "三列", "三栏")):
        return "three_column"
    if _contains_any(description, ("timeline", "时间线", "里程碑", "roadmap", "路线图")):
        return "timeline"
    if _contains_any(description, ("process", "flow", "pipeline", "workflow", "流程", "步骤", "链路")):
        return "process_flow"
    if _contains_any(description, ("metric", "kpi", "指标", "数字", "数据卡", "性能")):
        return "metric_focus"
    if _contains_any(description, ("quote", "引用", "引言", "金句")):
        return "quote_callout"
    if _contains_any(description, ("compare", "comparison", "versus", " vs ", "对比", "比较", "优缺点")):
        return "comparison"
    if has_asset and _contains_any(description, ("table", "表格", "表 ", "table ")):
        return "table_focus"
    if has_asset and _contains_any(description, ("chart", "plot", "柱状", "折线", "曲线", "图表")):
        return "chart_focus"
    if has_asset and _contains_any(description, ("hero", "主视觉", "大图")):
        return "hero"
    if has_asset and (
        _contains_any(description, ("two column", "two-column", "left", "right", "左图右文", "右图左文", "左右"))
        or bullets
    ):
        return "two_column"
    if has_asset:
        return "image_focus"
    if len(bullets) == 3:
        return "three_column"
    if len(bullets) >= 4:
        return "quadrant"
    return "section_divider"


def _infer_visual_type(layout_type: str, summary: str, asset_paths: list[str]) -> str:
    text = _layout_text(layout_type, summary, " ".join(asset_paths))
    if _contains_any(text, ("table", "表格", "表 ")):
        return "table"
    if _contains_any(text, ("chart", "plot", "柱状", "折线", "曲线", "图表")):
        return "chart"
    if _contains_any(text, ("diagram", "architecture", "流程图", "架构", "结构图", "示意图")):
        return "diagram"
    if _contains_any(text, ("photo", "照片", "图片")):
        return "photo"
    if _contains_any(text, ("interface", "ui", "界面", "截图")):
        return "interface"
    return "figure" if asset_paths else "text_only"


def _visual_slot_for_layout(layout_type: str) -> str:
    return "image"


def _preferred_material_slot(layout_type: str) -> str:
    return _visual_slot_for_layout(layout_type)


def _layout_payload(layout_type: str, *, variant: str, design_hint: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": layout_type,
        "variant": variant,
        "slots": PAGECONTENT_LAYOUT_SLOTS.get(layout_type, PAGECONTENT_LAYOUT_SLOTS["section_divider"]),
        "design_hint": design_hint,
    }
    if layout_type == "title_only":
        payload.update(
            {
                "title_align": "center",
                "subtitle_align": "right",
                "title_position": "center",
                "subtitle_position": "bottom_right",
            }
        )
    return payload


def _title_only_subtitle(item: dict[str, Any], summary: str, bullets: list[str]) -> str:
    explicit_subtitle = _normalize_text(item.get("subtitle") or item.get("sub_title"))
    if explicit_subtitle:
        return explicit_subtitle
    if bullets:
        return " / ".join(bullets[:2])
    return ""


def _block(
    block_id: str,
    *,
    role: str,
    kind: str,
    slot_id: str,
    text: str = "",
    items: list[str] | None = None,
    caption: str = "",
    path: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "block_id": block_id,
        "role": role,
        "kind": kind,
        "slot_id": slot_id,
    }
    if text:
        payload["text"] = text
        payload["content"] = text
    if items:
        payload["items"] = list(items)
    if caption:
        payload["caption"] = caption
    if path:
        payload["path"] = path
    return payload


def _image_block(slide_id: str, *, caption: str, path: str = "") -> dict[str, Any]:
    return _block(
        f"{slide_id}-image",
        role="image",
        kind="image",
        slot_id="image",
        caption=caption,
        path=path,
    )


def _build_outline_blocks(
    *,
    slide_id: str,
    title: str,
    subtitle: str,
    summary: str,
    bullets: list[str],
    layout_type: str,
    asset_paths: list[str] | None = None,
    image_caption: str = "",
) -> list[dict[str, Any]]:
    asset_paths = asset_paths or []
    blocks: list[dict[str, Any]] = []
    if title:
        blocks.append(_block(f"{slide_id}-title", role="title", kind="headline", slot_id="title", text=title))

    body_text = summary if summary and summary != subtitle else ""
    if layout_type == "title_only":
        if subtitle:
            blocks.append(_block(f"{slide_id}-subtitle", role="subtitle", kind="summary", slot_id="subtitle", text=subtitle))
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if layout_type in {"three_column", "quadrant"} and bullets:
        slots = ["body", "supporting_body", "callout"] if layout_type == "three_column" else ["body", "supporting_body", "metrics", "callout"]
        for index, item in enumerate(bullets[: len(slots)]):
            blocks.append(
                _block(
                    f"{slide_id}-block-{index + 1:02d}",
                    role=slots[index],
                    kind="summary",
                    slot_id=slots[index],
                    text=item,
                )
            )
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if layout_type in {"process_flow", "timeline"} and bullets:
        blocks.append(_block(f"{slide_id}-process", role="body", kind="process", slot_id="body", items=bullets))
        if body_text:
            blocks.append(_block(f"{slide_id}-supporting", role="supporting_body", kind="summary", slot_id="supporting_body", text=body_text))
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if layout_type == "metric_focus":
        if bullets:
            blocks.append(_block(f"{slide_id}-metrics", role="metrics", kind="metric_strip", slot_id="metrics", items=bullets[:4]))
        if body_text:
            blocks.append(_block(f"{slide_id}-body", role="body", kind="summary", slot_id="body", text=body_text))
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if layout_type == "comparison" and bullets:
        blocks.append(_block(f"{slide_id}-comparison", role="body", kind="comparison", slot_id="body", items=bullets))
        if body_text:
            blocks.append(_block(f"{slide_id}-callout", role="callout", kind="callout", slot_id="callout", text=body_text))
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if layout_type == "quote_callout":
        quote_text = bullets[0] if bullets else body_text
        if quote_text:
            blocks.append(_block(f"{slide_id}-quote", role="callout", kind="quote", slot_id="callout", text=quote_text))
        remaining = bullets[1:] if bullets else []
        if remaining:
            blocks.append(_block(f"{slide_id}-body", role="body", kind="bullet_list", slot_id="body", items=remaining))
        if asset_paths:
            blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
        return blocks

    if bullets:
        blocks.append(_block(f"{slide_id}-body", role="body", kind="bullet_list", slot_id="body", items=bullets))
    elif body_text:
        blocks.append(_block(f"{slide_id}-body", role="body", kind="summary", slot_id="body", text=body_text))

    if layout_type in {"two_column", "table_focus", "chart_focus"} and body_text and bullets:
        blocks.append(_block(f"{slide_id}-callout", role="callout", kind="callout", slot_id="callout", text=body_text))
    if asset_paths:
        blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
    return blocks



def is_direct_image_slide(item: dict[str, Any]) -> bool:
    return bool(_normalize_direct_image_asset_paths(item))


def _normalize_asset_base_dir(asset_base_dir: str) -> str:
    raw = _normalize_text(asset_base_dir)
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve())


def normalize_outline_slide(item: dict[str, Any], page_num: int, *, asset_base_dir: str = "") -> SlideIR:
    title = _normalize_text(item.get("title"))
    summary = _normalize_summary(item)
    layout_description = _normalize_text(item.get("layout_description"))
    image_caption = _normalize_caption(item)
    bullets = _normalize_bullets(item)
    asset_paths = (
        _normalize_asset_paths(item.get("asset_paths"))
        if "asset_paths" in item
        else _normalize_asset_paths(item.get("asset_ref"))
    )
    explicit_layout = _normalize_text(item.get("layout") or item.get("layout_name") or item.get("layout_type"))
    layout_hint = " ".join(part for part in (explicit_layout, layout_description) if part)

    slide_id = "slide-{0:03d}".format(page_num)
    layout_type = _select_outline_layout(
        page_num=page_num,
        title=title,
        summary=summary,
        bullets=bullets,
        asset_paths=asset_paths,
        explicit_layout=explicit_layout,
        layout_hint=layout_hint,
    )
    visual_hint = _layout_text(summary, layout_description, image_caption)
    visual_type = _infer_visual_type(layout_type, visual_hint, asset_paths)
    visual_slot = _visual_slot_for_layout(layout_type)
    subtitle = _title_only_subtitle(item, summary, bullets) if layout_type == "title_only" else _normalize_text(item.get("subtitle"))
    return SlideIR(
        slide_id=slide_id,
        page_num=page_num,
        brief_id=f"brief-{page_num:03d}",
        type="title" if layout_type == "title_only" and page_num == 1 else ("closing" if layout_type == "closing" else "content"),
        section_id=f"section-{page_num:03d}",
        section_title=title or f"Slide {page_num}",
        title=title,
        subtitle=subtitle,
        core_message=bullets[0] if bullets else summary or title,
        objective=summary or title,
        layout_type=layout_type,
        layout=_layout_payload(layout_type, variant="pagecontent_outline", design_hint=layout_description),
        blocks=_build_outline_blocks(
            slide_id=slide_id,
            title=title,
            subtitle=subtitle,
            summary=summary,
            bullets=bullets,
            layout_type=layout_type,
            asset_paths=asset_paths,
            image_caption=image_caption,
        ),
        points=list(bullets),
        visuals=[
            VisualBinding(
                visual_id=f"{slide_id}-visual-1",
                role="primary",
                asset_type=visual_type if visual_type in ALLOWED_VISUAL_TYPES else "figure",
                slot_id=visual_slot,
                target_area=visual_slot,
                caption=image_caption,
                intent=f"{visual_type} in {visual_slot}",
            )
        ]
        if asset_paths
        else [],
        asset_paths=asset_paths,
        asset_base_dir=_normalize_asset_base_dir(asset_base_dir),
        source_chunk_ids=[f"pagecontent-{page_num:03d}"],
        source_evidence=[{"kind": "pagecontent", "page_num": page_num, "title": title}],
        design_notes=[note for note in (layout_description, summary) if note],
        speaker_notes=list(bullets[:2]),
    )


def normalize_image_slide(item: dict[str, Any], page_num: int, *, asset_base_dir: str = "") -> SlideIR:
    slide_id = "slide-{0:03d}".format(page_num)
    asset_paths = _normalize_direct_image_asset_paths(item)
    summary = _normalize_summary(item)
    layout_description = _normalize_text(item.get("layout_description"))
    image_caption = _normalize_caption(item)
    title = _normalize_text(item.get("title"))
    blocks: list[dict[str, Any]] = []
    if title:
        blocks.append(_block(f"{slide_id}-title", role="title", kind="headline", slot_id="title", text=title))
    elif summary:
        blocks.append(_block(f"{slide_id}-image-note", role="body", kind="summary", slot_id="body", text=summary))
    if asset_paths:
        blocks.append(_image_block(slide_id, caption=image_caption, path=asset_paths[0]))
    return SlideIR(
        slide_id=slide_id,
        page_num=page_num,
        brief_id=f"brief-{page_num:03d}",
        type="direct_image",
        section_id=f"section-{page_num:03d}",
        section_title=title or f"Slide {page_num}",
        title=title,
        subtitle="",
        core_message=summary or title,
        objective="show the primary image asset",
        layout_type="image_focus",
        layout=_layout_payload("image_focus", variant="pagecontent_direct_image", design_hint=layout_description),
        blocks=blocks,
        points=[],
        visuals=[
            VisualBinding(
                visual_id=f"{slide_id}-visual-1",
                role="hero",
                asset_type=_infer_visual_type("image_focus", _layout_text(summary, layout_description, image_caption), asset_paths),
                slot_id="image",
                target_area="image",
                caption=image_caption,
                intent="full slide image",
            )
        ]
        if asset_paths
        else [],
        asset_paths=asset_paths,
        asset_base_dir=_normalize_asset_base_dir(asset_base_dir),
        source_chunk_ids=[f"pagecontent-{page_num:03d}"],
        source_evidence=[{"kind": "pagecontent", "page_num": page_num, "title": title}],
        design_notes=[note for note in (layout_description, summary) if note],
    )


_THEME_PRESETS: dict[str, dict[str, str]] = {
    "clean": {
        "primary": "#0F172A",
        "secondary": "#475569",
        "accent": "#2563EB",
        "background": "#FFFFFF",
    },
    "warm": {
        "primary": "#7C2D12",
        "secondary": "#9A3412",
        "accent": "#EA580C",
        "background": "#FFF7ED",
    },
    "dark": {
        "primary": "#E5E7EB",
        "secondary": "#CBD5E1",
        "accent": "#38BDF8",
        "background": "#0F172A",
    },
}

_PROMPT_THEME_KEYWORDS: list[tuple[tuple[str, ...], dict[str, str]]] = [
    (
        ("暖白", "象牙白", "赤陶", "赭红", "ivory", "terracotta"),
        {"primary": "#3D2B1F", "secondary": "#6B5B4F", "accent": "#B85C38", "background": "#F4EFE6"},
    ),
    (
        ("午夜蓝", "深海军蓝", "冰灰", "电蓝", "midnight", "navy", "ice gray"),
        {"primary": "#E2E8F0", "secondary": "#94A3B8", "accent": "#60A5FA", "background": "#0F172A"},
    ),
    (
        ("纸感", "米白", "墨黑", "酒红", "parchment", "burgundy"),
        {"primary": "#1C1917", "secondary": "#44403C", "accent": "#7F1D1D", "background": "#F8F2E7"},
    ),
    (
        ("森林绿", "橄榄", "沙金", "奶油白", "forest green", "olive", "sand gold"),
        {"primary": "#1F3D2B", "secondary": "#4D6B3A", "accent": "#D4B483", "background": "#FDFBF5"},
    ),
    (
        ("黑白灰", "亮橙", "极简", "monochrome", "bright orange"),
        {"primary": "#111827", "secondary": "#6B7280", "accent": "#F97316", "background": "#FFFFFF"},
    ),
    (
        ("深紫红", "雾粉", "银灰", "plum", "mist pink"),
        {"primary": "#3B0D2E", "secondary": "#9D6381", "accent": "#C9C9D6", "background": "#FDF8FC"},
    ),
]

_HEX_COLOR_RE = __import__("re").compile(r"#[0-9A-Fa-f]{6}")


def _extract_theme_from_prompt(style_prompt: str) -> dict[str, str] | None:
    lower = style_prompt.lower()
    for keywords, palette in _PROMPT_THEME_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return palette
    hex_colors = _HEX_COLOR_RE.findall(style_prompt)
    if len(hex_colors) >= 3:
        return {
            "background": hex_colors[0],
            "primary": hex_colors[1],
            "accent": hex_colors[2],
            "secondary": hex_colors[1],
        }
    return None


def build_default_theme(style: str, language: str) -> DeckTheme:
    normalized_style = _normalize_text(style).lower()
    normalized_language = _normalize_text(language).lower()

    theme = DeckTheme()
    if normalized_language.startswith("zh"):
        theme.title_font = "PingFang SC"
        theme.body_font = "PingFang SC"
    else:
        theme.title_font = "Aptos"
        theme.body_font = "Aptos"

    if normalized_style in _THEME_PRESETS:
        palette = _THEME_PRESETS[normalized_style]
    else:
        palette = _extract_theme_from_prompt(style) or _THEME_PRESETS["clean"]

    theme.primary = palette["primary"]
    theme.secondary = palette["secondary"]
    theme.accent = palette["accent"]
    theme.background = palette["background"]

    if style and normalized_style not in _THEME_PRESETS:
        theme.style_guardrails = [
            style,
            "Keep one main conclusion per slide.",
            "Separate visual and text regions clearly.",
        ]

    return theme


def pagecontent_to_deck_ir(
    pagecontent: list[dict[str, Any]],
    *,
    language: str = "zh",
    style: str = "",
    asset_base_dir: str = "",
) -> DeckIR:
    slides: list[SlideIR] = []
    material_requests: list[MaterialRequest] = []
    for index, item in enumerate(pagecontent or [], start=1):
        slide = (
            normalize_image_slide(item, index, asset_base_dir=asset_base_dir)
            if is_direct_image_slide(item)
            else normalize_outline_slide(item, index, asset_base_dir=asset_base_dir)
        )
        if slide.asset_paths:
            material_requests.append(
                MaterialRequest(
                    request_id=f"material-{index:03d}",
                    target_slide_id=slide.slide_id,
                    asset_type="image",
                    purpose=slide.core_message or slide.title,
                    preferred_layout_slot=_preferred_material_slot(slide.layout_type),
                    acquisition_plan={"source_options": ["pagecontent_assets"]},
                )
            )
        slides.append(slide)

    title = slides[0].title if slides and slides[0].title else "Editable Deck"
    storyline_sections = [slide.title for slide in slides if slide.title]

    return DeckIR(
        metadata=IRMetadata(
            deck_id="deck-pagecontent",
            stage="planned",
            language=language,
            style=style,
        ),
        title=title,
        subtitle="",
        language=language,
        style=style,
        theme=build_default_theme(style=style, language=language),
        storyline=Storyline(
            summary=" -> ".join(storyline_sections[:5]),
            sections=storyline_sections,
        ),
        material_requests=material_requests,
        planner_notes=["generated from pagecontent adapter"],
        slide_manifest=[
            {
                "slide_id": slide.slide_id,
                "page_num": slide.page_num,
                "layout_type": slide.layout_type,
                "title": slide.title,
            }
            for slide in slides
        ],
        source_asset_index={slide.slide_id: list(slide.asset_paths) for slide in slides if slide.asset_paths},
        slides=slides,
    )
