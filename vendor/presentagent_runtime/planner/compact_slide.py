from __future__ import annotations

from typing import Any

from .ir_models import SlideIR, VisualBinding


def build_compact_slide_payload(
    slide: SlideIR | dict[str, Any],
    *,
    source_assets: list[dict[str, Any]] | None = None,
    max_blocks: int = 10,
    max_points: int = 8,
    max_assets: int = 5,
    max_evidence: int = 3,
) -> dict[str, Any]:
    """Return the concise slide contract used by artifacts and codegen."""

    source_assets = source_assets or []
    slide_id = _slide_get(slide, "slide_id", "")
    page_num = _slide_get(slide, "page_num", 0)
    layout = _compact_layout(slide)
    blocks = _compact_blocks(
        _slide_get(slide, "blocks", []) or [],
        slide_id=slide_id,
        title=_slide_get(slide, "title", ""),
        subtitle=_slide_get(slide, "subtitle", ""),
        summary=_slide_get(slide, "core_message", ""),
        points=_string_list(_slide_get(slide, "points", []), limit=max_points),
        max_blocks=max_blocks,
    )
    selected_assets = _selected_assets_for_slide(slide, source_assets, limit=max_assets)
    selected_asset = selected_assets[0] if selected_assets else {}
    asset_paths = _asset_paths_for_slide(slide, source_assets, selected_assets=selected_assets, limit=max_assets)
    blocks = _ensure_compact_image_blocks(
        blocks,
        slide_id=slide_id,
        selected_assets=selected_assets,
        slide=slide,
    )
    layout = _ensure_layout_slot_for_blocks(layout, blocks)

    payload = {
        "slide_id": slide_id,
        "page_num": page_num,
        "type": _slide_get(slide, "type", "content"),
        "layout": layout,
        "blocks": blocks,
        "evidence": _truncate_jsonish(_slide_get(slide, "source_evidence", []) or [], max_items=max_evidence, max_text=220),
    }
    return _drop_empty(payload)


def expand_compact_slide_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand the compact contract into the flat shape expected by helpers."""

    payload = dict(payload or {})
    layout = dict(payload.get("layout") or {})
    blocks = list(payload.get("blocks") or [])
    title = _first_block_text(blocks, "title")
    subtitle = _first_block_text(blocks, "subtitle")
    summary = _first_block_text(blocks, "summary") or _first_block_text(blocks, "body")
    points = _first_block_items(blocks, "body")
    image_blocks = [block for block in blocks if block.get("kind") == "image" or _is_image_slot(block.get("slot"))]
    image_block = image_blocks[0] if image_blocks else {}
    selected_asset_path = str(image_block.get("path") or payload.get("selected_asset_path") or "").strip()
    selected_asset_id = str(image_block.get("asset_id") or payload.get("selected_asset_id") or "").strip()
    asset_paths = [
        str(block.get("path") or "").strip()
        for block in image_blocks
        if str(block.get("path") or "").strip()
    ]
    expanded = {
        **payload,
        "layout": layout,
        "layout_type": layout.get("name") or payload.get("layout_type") or "section_divider",
        "title": title or payload.get("title") or "",
        "subtitle": subtitle or payload.get("subtitle") or "",
        "core_message": summary or (points[0] if points else "") or payload.get("core_message") or "",
        "points": points,
        "blocks": blocks,
        "visuals": payload.get("visuals") or _visuals_from_image_blocks(payload, image_blocks),
        "selected_asset_id": selected_asset_id,
        "selected_asset_path": selected_asset_path,
        "asset_paths": list(dict.fromkeys(asset_paths or ([selected_asset_path] if selected_asset_path else []))),
        "source_evidence": payload.get("evidence") or [],
        "design_notes": [],
        "speaker_notes": [],
    }
    return expanded


def compact_slide_for_codegen(
    slide: SlideIR | dict[str, Any],
    *,
    source_assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact payload while tolerating already-compact inputs."""

    if isinstance(slide, dict) and "content" in slide and "layout" in slide:
        return build_compact_slide_payload(expand_compact_slide_payload(slide), source_assets=source_assets)
    return build_compact_slide_payload(slide, source_assets=source_assets)


def _compact_layout(slide: SlideIR | dict[str, Any]) -> dict[str, Any]:
    raw_layout = dict(_slide_get(slide, "layout", {}) or {})
    layout_name = str(raw_layout.get("name") or _slide_get(slide, "layout_type", "") or "section_divider").strip()
    payload = {
        "name": layout_name,
        "slots": list(raw_layout.get("slots") or []),
        "title_align": raw_layout.get("title_align") or "",
        "subtitle_align": raw_layout.get("subtitle_align") or "",
        "title_position": raw_layout.get("title_position") or "",
        "subtitle_position": raw_layout.get("subtitle_position") or "",
        "visual_intent": raw_layout.get("visual_intent") or "",
    }
    return _drop_empty(payload)


def _compact_blocks(
    raw_blocks: list[dict[str, Any]],
    *,
    slide_id: str,
    title: str,
    subtitle: str,
    summary: str,
    points: list[str],
    max_blocks: int,
) -> list[dict[str, Any]]:
    blocks = [block for block in raw_blocks if isinstance(block, dict)]
    if not blocks:
        blocks = _fallback_blocks(slide_id=slide_id, title=title, subtitle=subtitle, summary=summary, points=points)

    blocks = [
        block
        for block in blocks
        if not _is_image_block(block)
        or str(block.get("path") or block.get("asset_path") or block.get("asset_id") or block.get("selected_asset_id") or "").strip()
    ]

    compact_blocks: list[dict[str, Any]] = []
    for index, block in enumerate(blocks[:max_blocks], start=1):
        text = str(block.get("content") or block.get("text") or "").strip()
        items = _string_list(block.get("items") or [], limit=10)
        compact = _drop_empty(
            {
                "id": str(block.get("block_id") or block.get("id") or f"{slide_id}-block-{index:02d}"),
                "kind": str(block.get("kind") or "summary"),
                "role": str(block.get("role") or block.get("slot_id") or "body"),
                "slot": str(block.get("slot_id") or block.get("slot") or block.get("role") or "body"),
                "text": text,
                "items": items,
                "caption": str(block.get("caption") or "").strip(),
                "path": str(block.get("path") or block.get("asset_path") or "").strip(),
                "asset_id": str(block.get("asset_id") or block.get("selected_asset_id") or "").strip(),
                "description": str(block.get("description") or "").strip(),
            }
        )
        if _is_image_block(compact):
            compact.pop("description", None)
        if compact:
            compact_blocks.append(compact)
    return compact_blocks


def _ensure_compact_image_blocks(
    blocks: list[dict[str, Any]],
    *,
    slide_id: str,
    selected_assets: list[dict[str, Any]],
    slide: SlideIR | dict[str, Any],
) -> list[dict[str, Any]]:
    selected_assets = selected_assets or _assets_from_existing_image_blocks(blocks)
    updated: list[dict[str, Any]] = []
    for block in blocks:
        if not _is_image_block(block):
            updated.append(block)
            continue
        block_path = str(block.get("path") or "").strip()
        block_asset_id = str(block.get("asset_id") or "").strip()
        if not block_path and not block_asset_id:
            continue
    if not selected_assets:
        return updated

    for index, asset in enumerate(selected_assets, start=1):
        path = str(asset.get("path") or asset.get("relative_path") or "").strip()
        caption = str(asset.get("caption") or "").strip()
        asset_id = str(asset.get("asset_id") or (_slide_get(slide, "selected_asset_id", "") if index == 1 else "") or "").strip()
        payload = _drop_empty(
            {
                "id": _image_block_id(slide_id, index),
                "kind": "image",
                "role": "image",
                "slot": _image_slot_id(index),
                "caption": caption,
                "path": path,
                "asset_id": asset_id,
            }
        )
        if payload.get("path") or payload.get("asset_id"):
            updated.append(payload)
    return _dedupe_image_blocks(updated)


def _is_image_block(block: dict[str, Any]) -> bool:
    return block.get("kind") == "image" or _is_image_slot(block.get("slot")) or block.get("role") == "image"


def _is_image_slot(slot_id: Any) -> bool:
    value = str(slot_id or "").strip()
    return value == "image" or (value.startswith("image_") and value[6:].isdigit())


def _image_slot_id(index: int) -> str:
    return "image" if index <= 1 else f"image_{index:02d}"


def _image_block_id(slide_id: str, index: int) -> str:
    return f"{slide_id}-image" if index <= 1 else f"{slide_id}-image-{index:02d}"


def _assets_from_existing_image_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for block in blocks:
        if not _is_image_block(block):
            continue
        path = str(block.get("path") or "").strip()
        asset_id = str(block.get("asset_id") or "").strip()
        if path or asset_id:
            assets.append(
                _drop_empty(
                    {
                        "asset_id": asset_id,
                        "path": path,
                        "caption": str(block.get("caption") or "").strip(),
                    }
                )
            )
    return assets


def _dedupe_image_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        if _is_image_block(block):
            key = str(block.get("asset_id") or block.get("path") or block.get("id") or "").strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
        deduped.append(block)
    return deduped


def _ensure_layout_slot_for_blocks(layout: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    slots = list(layout.get("slots") or [])
    slot_ids = {str(slot.get("slot_id") or "") for slot in slots if isinstance(slot, dict)}
    wanted_slots = {str(block.get("slot") or "") for block in blocks if str(block.get("slot") or "")}
    missing_image_slots = [slot_id for slot_id in wanted_slots if _is_image_slot(slot_id) and slot_id not in slot_ids]
    if missing_image_slots:
        base_slot = next(
            (dict(slot) for slot in slots if isinstance(slot, dict) and str(slot.get("slot_id") or "") == "image"),
            {"slot_id": "image", "x_ratio": 0.54, "y_ratio": 0.23, "w_ratio": 0.38, "h_ratio": 0.56},
        )
        image_count = max(1, len([block for block in blocks if _is_image_block(block)]))
        generated = _split_image_slots(base_slot, image_count)
        slots = [
            slot
            for slot in slots
            if not (isinstance(slot, dict) and _is_image_slot(slot.get("slot_id")))
        ]
        slots.extend(generated)
    return {**layout, "slots": slots} if slots else layout


def _split_image_slots(base_slot: dict[str, Any], image_count: int) -> list[dict[str, float | str]]:
    if image_count <= 1:
        return [{**base_slot, "slot_id": "image"}]
    x = float(base_slot.get("x_ratio", 0.54))
    y = float(base_slot.get("y_ratio", 0.23))
    w = float(base_slot.get("w_ratio", 0.38))
    h = float(base_slot.get("h_ratio", 0.56))
    gap = min(0.025, max(h * 0.06, 0.012))
    each_h = max((h - gap * (image_count - 1)) / image_count, 0.08)
    return [
        {
            "slot_id": _image_slot_id(index),
            "x_ratio": x,
            "y_ratio": y + (index - 1) * (each_h + gap),
            "w_ratio": w,
            "h_ratio": each_h,
        }
        for index in range(1, image_count + 1)
    ]


def _fallback_blocks(
    *,
    slide_id: str,
    title: str,
    subtitle: str,
    summary: str,
    points: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if title:
        blocks.append({"id": f"{slide_id}-title", "kind": "headline", "role": "title", "slot": "title", "text": title})
    if subtitle:
        blocks.append({"id": f"{slide_id}-subtitle", "kind": "summary", "role": "subtitle", "slot": "subtitle", "text": subtitle})
    if points:
        blocks.append({"id": f"{slide_id}-body", "kind": "bullet_list", "role": "body", "slot": "body", "items": points})
    return blocks


def _compact_visuals(
    slide: SlideIR | dict[str, Any],
    *,
    selected_asset: dict[str, Any],
    max_items: int,
) -> list[dict[str, Any]]:
    visuals = _slide_get(slide, "visuals", []) or []
    compact_visuals: list[dict[str, Any]] = []
    for visual in visuals[:max_items]:
        visual_dict = _model_dump(visual)
        selected_candidate = dict(visual_dict.get("selected_candidate") or {})
        compact_visuals.append(
            _drop_empty(
                {
                    "id": visual_dict.get("visual_id") or visual_dict.get("id"),
                    "type": visual_dict.get("asset_type") or visual_dict.get("type"),
                    "role": visual_dict.get("role"),
                    "slot": visual_dict.get("slot_id") or visual_dict.get("target_area"),
                    "caption": visual_dict.get("caption"),
                    "intent": visual_dict.get("intent"),
                    "asset_id": visual_dict.get("selected_asset_id") or selected_candidate.get("asset_id"),
                    "asset_path": visual_dict.get("selected_asset_path") or selected_candidate.get("path"),
                }
            )
        )
    if not compact_visuals and selected_asset:
        slot = _visual_slot_for_layout(str((_slide_get(slide, "layout", {}) or {}).get("name") or _slide_get(slide, "layout_type", "")))
        compact_visuals.append(
            _drop_empty(
                {
                    "id": f"{_slide_get(slide, 'slide_id', '')}-visual-1",
                    "type": selected_asset.get("asset_kind") or "figure",
                    "role": "primary",
                    "slot": slot,
                    "caption": selected_asset.get("caption") or selected_asset.get("description"),
                    "intent": (_slide_get(slide, "layout", {}) or {}).get("visual_intent"),
                    "asset_id": selected_asset.get("asset_id"),
                    "asset_path": selected_asset.get("path"),
                }
            )
        )
    return compact_visuals


def _compact_assets(
    source_assets: list[dict[str, Any]],
    *,
    selected_asset: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    if selected_asset:
        assets.append(selected_asset)
    assets.extend(source_assets)

    compact_assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset in assets:
        key = str(asset.get("asset_id") or asset.get("path") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        compact_assets.append(
            _drop_empty(
                {
                    "asset_id": asset.get("asset_id"),
                    "kind": asset.get("asset_kind") or asset.get("kind"),
                    "path": asset.get("path"),
                    "absolute_path": asset.get("absolute_path"),
                    "caption": asset.get("caption"),
                    "description": asset.get("description"),
                    "exists": asset.get("asset_exists"),
                }
            )
        )
        if len(compact_assets) >= limit:
            break
    return compact_assets


def _selected_assets_for_slide(
    slide: SlideIR | dict[str, Any],
    source_assets: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected_asset_id = str(_slide_get(slide, "selected_asset_id", "") or "").strip()
    selected_asset_path = str(_slide_get(slide, "selected_asset_path", "") or "").strip()
    if source_assets:
        ordered: list[dict[str, Any]] = []
        for asset in source_assets:
            asset_id = str(asset.get("asset_id") or "").strip()
            paths = {str(asset.get("path") or "").strip(), str(asset.get("relative_path") or "").strip()}
            if (selected_asset_id and asset_id == selected_asset_id) or (selected_asset_path and selected_asset_path in paths):
                ordered.append(dict(asset))
        ordered.extend(
            dict(asset)
            for asset in source_assets
            if not any(
                str(asset.get("asset_id") or asset.get("path") or "").strip()
                == str(item.get("asset_id") or item.get("path") or "").strip()
                for item in ordered
            )
        )
        return ordered[:limit]

    block_assets = _assets_from_existing_image_blocks(_slide_get(slide, "blocks", []) or [])
    if block_assets:
        return block_assets[:limit]

    asset_paths = _string_list(_slide_get(slide, "asset_paths", []) or [])
    if selected_asset_path and selected_asset_path not in asset_paths:
        asset_paths.insert(0, selected_asset_path)
    return [
        _drop_empty(
            {
                "asset_id": selected_asset_id if index == 1 else "",
                "path": path,
                "relative_path": path,
            }
        )
        for index, path in enumerate(list(dict.fromkeys(asset_paths)), start=1)
    ][:limit]


def _selected_asset_for_slide(slide: SlideIR | dict[str, Any], source_assets: list[dict[str, Any]]) -> dict[str, Any]:
    selected_assets = _selected_assets_for_slide(slide, source_assets, limit=1)
    if selected_assets:
        return selected_assets[0]
    selected_asset_id = str(_slide_get(slide, "selected_asset_id", "") or "").strip()
    selected_asset_path = str(_slide_get(slide, "selected_asset_path", "") or "").strip()
    if selected_asset_id:
        for asset in source_assets:
            if str(asset.get("asset_id") or "").strip() == selected_asset_id:
                return dict(asset)
    if selected_asset_path:
        for asset in source_assets:
            paths = {str(asset.get("path") or "").strip(), str(asset.get("relative_path") or "").strip()}
            if selected_asset_path in paths:
                return dict(asset)
        return {"asset_id": selected_asset_id, "path": selected_asset_path}
    return dict(source_assets[0]) if source_assets else {}


def _asset_paths_for_slide(
    slide: SlideIR | dict[str, Any],
    source_assets: list[dict[str, Any]],
    *,
    selected_assets: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    paths: list[str] = []
    for asset in selected_assets:
        path = str(asset.get("path") or asset.get("relative_path") or "").strip()
        if path:
            paths.append(path)
    for asset in source_assets:
        path = str(asset.get("path") or "").strip()
        if path:
            paths.append(path)
    if paths:
        return list(dict.fromkeys(path for path in paths if path))[:limit]
    paths.extend(_string_list(_slide_get(slide, "asset_paths", []) or []))
    return list(dict.fromkeys(path for path in paths if path))[:limit]


def _visual_slot_for_layout(layout_name: str) -> str:
    return "image"


def _first_block_text(blocks: list[dict[str, Any]], role_or_slot: str) -> str:
    for block in blocks:
        role = str(block.get("role") or "").strip()
        slot = str(block.get("slot") or block.get("slot_id") or "").strip()
        kind = str(block.get("kind") or "").strip()
        if role_or_slot not in {role, slot, kind}:
            continue
        text = str(block.get("text") or block.get("content") or "").strip()
        if text:
            return text
    return ""


def _first_block_items(blocks: list[dict[str, Any]], role_or_slot: str) -> list[str]:
    for block in blocks:
        role = str(block.get("role") or "").strip()
        slot = str(block.get("slot") or block.get("slot_id") or "").strip()
        if role_or_slot not in {role, slot}:
            continue
        items = _string_list(block.get("items") or [])
        if items:
            return items
    return []


def _visuals_from_image_blocks(payload: dict[str, Any], image_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visuals: list[dict[str, Any]] = []
    for index, image_block in enumerate(image_blocks, start=1):
        path = str(image_block.get("path") or "").strip()
        asset_id = str(image_block.get("asset_id") or "").strip()
        if not image_block or (not path and not asset_id):
            continue
        visuals.append(
            _drop_empty(
                {
                    "visual_id": image_block.get("id") or _image_block_id(str(payload.get("slide_id") or ""), index),
                    "role": "primary" if index == 1 else "supporting",
                    "asset_type": "image",
                    "slot_id": image_block.get("slot") or _image_slot_id(index),
                    "target_area": image_block.get("slot") or _image_slot_id(index),
                    "caption": image_block.get("caption"),
                    "intent": image_block.get("description"),
                    "selected_asset_id": image_block.get("asset_id"),
                    "selected_asset_path": path,
                }
            )
        )
    return visuals


def _visuals_from_image_block(payload: dict[str, Any], image_block: dict[str, Any]) -> list[dict[str, Any]]:
    path = str(image_block.get("path") or "").strip()
    asset_id = str(image_block.get("asset_id") or "").strip()
    if not image_block or (not path and not asset_id):
        return []
    return [
        _drop_empty(
            {
                "visual_id": image_block.get("id") or f"{payload.get('slide_id', '')}-image",
                "role": "image",
                "asset_type": "image",
                "slot_id": "image",
                "target_area": "image",
                "caption": image_block.get("caption"),
                "intent": image_block.get("description"),
                "selected_asset_id": image_block.get("asset_id"),
                "selected_asset_path": path,
            }
        )
    ]


def _slide_get(slide: SlideIR | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(slide, dict):
        return slide.get(key, default)
    return getattr(slide, key, default)


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, VisualBinding):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, [], {})
    }


def _string_list(values: Any, *, limit: int | None = None) -> list[str]:
    if not values:
        return []
    if isinstance(values, list):
        items = [str(value).strip() for value in values if str(value or "").strip()]
    else:
        item = str(values).strip()
        items = [item] if item else []
    return items[:limit] if limit is not None else items


def _unique_string_list(values: Any, *, limit: int) -> list[str]:
    return list(dict.fromkeys(_string_list(values)))[:limit]


def _truncate_jsonish(value: Any, *, max_items: int = 6, max_text: int = 240) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            compact[str(key)] = _truncate_jsonish(item, max_items=max_items, max_text=max_text)
        return compact
    if isinstance(value, list):
        return [_truncate_jsonish(item, max_items=max_items, max_text=max_text) for item in value[:max_items]]
    if isinstance(value, str):
        return value[:max_text]
    return value
