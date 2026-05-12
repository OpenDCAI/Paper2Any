from __future__ import annotations

from pathlib import Path
from typing import Any

from ..planner.ir_models import DeckIR, VisualBinding
from .mineru_asset_index import find_mineru_asset_matches


def _normalize_asset_path(raw_path: str, *, asset_base_dir: str) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""

    path = Path(value).expanduser()
    if not path.is_absolute() and asset_base_dir:
        path = Path(asset_base_dir).expanduser() / path
    if path.exists():
        return str(path.resolve())
    return str(path)


def _relative_asset_path(resolved_path: str, *, asset_base_dir: str) -> str:
    value = str(resolved_path or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    base_dir = Path(asset_base_dir or "").expanduser()
    if not str(base_dir):
        return value
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return value


def _build_image_layout_description(slide) -> str:
    layout_name = str((slide.layout or {}).get("name") or slide.layout_type or "").strip()
    layout_design_hint = str((slide.layout or {}).get("design_hint") or "").strip()
    material_request_purpose = str((slide.layout or {}).get("material_request_purpose") or "").strip()
    visual_intents = [
        str(visual.intent or "").strip()
        for visual in slide.visuals
        if str(visual.intent or "").strip()
    ]
    parts = []
    if layout_name:
        parts.append(f"layout={layout_name}")
    if layout_design_hint:
        parts.append(f"layout_description={layout_design_hint}")
    if material_request_purpose:
        parts.append(f"material_purpose={material_request_purpose}")
    if visual_intents:
        parts.append("visual_intent=" + " | ".join(visual_intents))
    text_anchor = slide.core_message or (slide.points[0] if slide.points else "") or slide.title
    if text_anchor:
        parts.append(f"text_anchor={text_anchor}")
    return "; ".join(parts)


def _material_candidates_for_asset_ref(
    raw_path: str,
    *,
    asset_base_dir: str,
) -> list[dict[str, Any]]:
    resolved_path = _normalize_asset_path(raw_path, asset_base_dir=asset_base_dir)
    resolved_path_obj = Path(resolved_path)
    if resolved_path_obj.exists():
        return [
            {
                "path": str(resolved_path_obj.resolve()),
                "relative_path": _relative_asset_path(str(resolved_path_obj.resolve()), asset_base_dir=asset_base_dir),
                "asset_ref": raw_path,
                "extension": resolved_path_obj.suffix.lower(),
                "document_object_type": "image",
                "match_source": "path",
                "asset_exists": True,
            }
        ]

    mineru_matches = find_mineru_asset_matches(raw_path, asset_base_dir=asset_base_dir)
    if mineru_matches:
        candidates: list[dict[str, Any]] = []
        for match in mineru_matches:
            relative_path = str(match.get("relative_path") or match.get("path") or "").strip()
            absolute_path = str(match.get("absolute_path") or "").strip()
            if not absolute_path:
                absolute_path = _normalize_asset_path(relative_path, asset_base_dir=asset_base_dir)
            extension = Path(absolute_path or relative_path).suffix.lower()
            candidates.append(
                {
                    "path": relative_path,
                    "absolute_path": absolute_path,
                    "relative_path": relative_path,
                    "asset_ref": raw_path,
                    "extension": extension,
                    "document_object_type": str(match.get("document_object_type") or "image"),
                    "caption": str(match.get("caption") or ""),
                    "match_score": match.get("match_score", 0.0),
                    "match_source": "mineru_caption",
                    "asset_exists": bool(Path(absolute_path).exists()),
                    "raw_mineru_path": str(match.get("raw_mineru_path") or ""),
                    "page_idx": match.get("page_idx", None),
                }
            )
        return candidates

    return [
        {
            "path": resolved_path,
            "relative_path": raw_path,
            "asset_ref": raw_path,
            "extension": Path(resolved_path).suffix.lower(),
            "document_object_type": "image",
            "match_source": "unresolved",
            "asset_exists": False,
        }
    ]


def collect_materials(deck_ir: DeckIR) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    asset_catalog: list[dict[str, Any]] = []
    asset_request_contexts: list[dict[str, Any]] = []
    asset_index: dict[str, dict[str, Any]] = {}

    for slide in deck_ir.slides:
        asset_sequence = 1
        for raw_path in slide.asset_paths:
            for candidate in _material_candidates_for_asset_ref(raw_path, asset_base_dir=slide.asset_base_dir):
                selected_path = str(candidate.get("path") or "").strip()
                relative_path = str(candidate.get("relative_path") or raw_path).strip()
                absolute_path = str(candidate.get("absolute_path") or "").strip()
                if not absolute_path:
                    absolute_path = _normalize_asset_path(selected_path, asset_base_dir=slide.asset_base_dir)
                asset_id = f"{slide.slide_id}-asset-{asset_sequence:02d}"
                asset_sequence += 1
                extension = str(candidate.get("extension") or Path(absolute_path or selected_path).suffix.lower())
                description_parts = [_build_image_layout_description(slide)]
                caption = str(candidate.get("caption") or "").strip()
                if caption:
                    description_parts.append(f"source_caption={caption}")
                match_source = str(candidate.get("match_source") or "").strip()
                if match_source:
                    description_parts.append(f"match_source={match_source}")
                asset = {
                    "asset_id": asset_id,
                    "path": selected_path,
                    "absolute_path": absolute_path,
                    "relative_path": relative_path,
                    "asset_ref": raw_path,
                    "category": "pagecontent",
                    "asset_kind": "image",
                    "document_object_type": str(candidate.get("document_object_type") or "image"),
                    "extension": extension,
                    "description": "; ".join(part for part in description_parts if part),
                    "request_context": slide.title,
                    "target_slide_id": slide.slide_id,
                    "match_source": match_source,
                    "asset_exists": bool(candidate.get("asset_exists", False)),
                }
                if caption:
                    asset["caption"] = caption
                if "match_score" in candidate:
                    asset["match_score"] = candidate.get("match_score")
                if candidate.get("raw_mineru_path"):
                    asset["raw_mineru_path"] = candidate.get("raw_mineru_path")
                if candidate.get("page_idx") is not None:
                    asset["page_idx"] = candidate.get("page_idx")
                assets.append(asset)
                asset_catalog.append(asset)
                asset_index[asset_id] = asset
                asset_request_contexts.append(
                    {
                        "request_id": f"material-{slide.page_num:03d}",
                        "target_slide_id": slide.slide_id,
                        "asset_id": asset_id,
                        "title": slide.title,
                        "core_message": slide.core_message,
                    }
                )

    material_manifest = {
        "document_dir": "",
        "markdown_path": "",
        "markdown_preview": "",
        "images": [asset["path"] for asset in assets if asset.get("asset_exists", True)],
        "assets": assets,
        "asset_index": asset_index,
        "assets_by_category": {"pagecontent": assets},
        "asset_request_contexts": asset_request_contexts,
        "asset_catalog": asset_catalog,
    }
    deck_ir.source_asset_index = asset_index
    return material_manifest


def _resolve_request_id(slide) -> str:
    return f"material-{slide.page_num:03d}"


def _build_resolved_source_asset_index(slides) -> dict[str, dict[str, Any]]:
    source_asset_index: dict[str, dict[str, Any]] = {}
    for slide in slides:
        for visual in slide.visuals:
            candidate = dict(visual.selected_candidate or {})
            selected_asset_id = str(visual.selected_asset_id or candidate.get("asset_id") or "").strip()
            selected_asset_path = str(visual.selected_asset_path or candidate.get("path") or "").strip()
            if not selected_asset_id:
                continue
            candidate["asset_id"] = selected_asset_id
            candidate["path"] = selected_asset_path or str(candidate.get("path") or "")
            candidate["relative_path"] = str(candidate.get("relative_path") or candidate["path"])
            candidate["target_slide_id"] = slide.slide_id
            source_asset_index[selected_asset_id] = candidate

        for block in slide.blocks:
            if not isinstance(block, dict) or block.get("kind") != "image":
                continue
            selected_asset_id = str(block.get("asset_id") or block.get("selected_asset_id") or "").strip()
            selected_asset_path = str(block.get("path") or block.get("asset_path") or "").strip()
            if not selected_asset_id or selected_asset_id in source_asset_index:
                continue
            source_asset_index[selected_asset_id] = {
                "asset_id": selected_asset_id,
                "path": selected_asset_path,
                "relative_path": selected_asset_path,
                "target_slide_id": slide.slide_id,
                "asset_kind": "image",
            }
    return source_asset_index


def resolve_materials(
    deck_ir: DeckIR,
    material_manifest: dict[str, Any],
    *,
    descriptor=None,
) -> tuple[DeckIR, dict[str, Any]]:
    manifest_assets = material_manifest.get("assets") or []
    assets_by_slide: dict[str, list[dict[str, Any]]] = {}
    for asset in manifest_assets:
        slide_id = str(asset.get("target_slide_id") or "").strip()
        if not slide_id:
            continue
        assets_by_slide.setdefault(slide_id, []).append(asset)

    requests: list[dict[str, Any]] = []
    resolved_slides = []
    resolved_count = 0

    for slide in deck_ir.slides:
        raw_candidate_pool = assets_by_slide.get(slide.slide_id, [])
        request_payload = {
            "request_id": _resolve_request_id(slide),
            "target_slide_id": slide.slide_id,
            "title": slide.title,
            "purpose": slide.core_message or slide.objective,
            "caption": "",
        }
        descriptor_error = ""
        preferred_asset_id = str(slide.selected_asset_id or "").strip()
        if not preferred_asset_id:
            for request in deck_ir.material_requests:
                if request.target_slide_id != slide.slide_id:
                    continue
                preferred_asset_id = str(
                    request.acquisition_plan.get("selected_asset_id") or ""
                ).strip()
                if preferred_asset_id:
                    break

        if preferred_asset_id and raw_candidate_pool:
            preferred_candidates = [
                candidate
                for candidate in raw_candidate_pool
                if str(candidate.get("asset_id") or "").strip() == preferred_asset_id
            ]
            candidate_pool = preferred_candidates + [
                candidate
                for candidate in raw_candidate_pool
                if str(candidate.get("asset_id") or "").strip() != preferred_asset_id
            ]
        elif descriptor is not None and raw_candidate_pool:
            try:
                candidate_pool = descriptor.score_candidates(request_payload, raw_candidate_pool)
            except Exception as exc:  # noqa: BLE001
                descriptor_error = str(exc)
                candidate_pool = list(raw_candidate_pool)
        else:
            candidate_pool = list(raw_candidate_pool)
        selected_assets = [
            candidate
            for candidate in candidate_pool
            if candidate.get("asset_exists", True)
        ]
        selected_asset = selected_assets[0] if selected_assets else None
        selected_asset_id = str(selected_asset.get("asset_id") or "") if selected_asset else ""
        selected_asset_path = str(selected_asset.get("path") or "") if selected_asset else ""
        selected_relative_path = (
            str(selected_asset.get("relative_path") or selected_asset_path)
            if selected_asset
            else ""
        )
        updated_asset_paths = _asset_paths_from_assets(selected_assets) or list(slide.asset_paths)
        updated_blocks = _update_image_blocks(
            list(slide.blocks),
            slide_id=slide.slide_id,
            selected_assets=selected_assets,
            description="",
        )

        updated_visuals = _update_visuals(
            list(slide.visuals),
            slide_id=slide.slide_id,
            selected_assets=selected_assets,
            candidate_pool=candidate_pool,
            intent=str((slide.layout or {}).get("visual_intent") or ""),
        )

        updated_slide = slide.model_copy(
            update={
                "selected_asset_id": selected_asset_id,
                "selected_asset_path": selected_asset_path,
                "asset_paths": updated_asset_paths,
                "blocks": updated_blocks,
                "visuals": updated_visuals,
            }
        )
        resolved_slides.append(updated_slide)

        request_id = _resolve_request_id(slide)
        resolution_status = "resolved" if selected_asset_path else "unresolved"
        if resolution_status == "resolved":
            resolved_count += 1
        requests.append(
            {
                "request_id": request_id,
                "target_slide_id": slide.slide_id,
                "resolution_status": resolution_status,
                "resolved_candidate": selected_asset or {},
                "resolved_candidates": selected_assets,
                "candidate_pool": candidate_pool,
                "attempt_log": [
                    {
                        "step": "pagecontent_asset_scan",
                        "matched": bool(selected_asset),
                        "preferred_asset_id": preferred_asset_id,
                        "descriptor_used": bool(
                            descriptor is not None
                            and not descriptor_error
                            and not preferred_asset_id
                        ),
                        "descriptor_error": descriptor_error,
                    }
                ],
                "matched_from": "pagecontent_assets" if selected_asset else "",
            }
        )

    resolved_deck = deck_ir.model_copy(
        update={
            "metadata": deck_ir.metadata.model_copy(update={"stage": "final"}),
            "slides": resolved_slides,
            "source_asset_index": _build_resolved_source_asset_index(resolved_slides),
        }
    )
    resolution_payload = {
        "summary": {
            "request_count": len(requests),
            "resolved_count": resolved_count,
            "unresolved_count": max(0, len(requests) - resolved_count),
        },
        "requests": requests,
    }
    return resolved_deck, resolution_payload


def _update_image_blocks(
    blocks: list[dict[str, Any]],
    *,
    slide_id: str,
    selected_assets: list[dict[str, Any]],
    description: str = "",
) -> list[dict[str, Any]]:
    if not selected_assets:
        return blocks

    image_blocks: list[dict[str, Any]] = []
    for index, asset in enumerate(selected_assets, start=1):
        path = str(asset.get("relative_path") or asset.get("path") or "").strip()
        asset_id = str(asset.get("asset_id") or "").strip()
        if not path and not asset_id:
            continue
        image_payload = {
            "block_id": _image_block_id(slide_id, index),
            "role": "image",
            "kind": "image",
            "slot_id": _image_slot_id(index),
        }
        if path:
            image_payload["path"] = path
        if asset_id:
            image_payload["asset_id"] = asset_id
        caption = str(asset.get("caption") or "").strip()
        if caption:
            image_payload["caption"] = caption
        if description:
            image_payload["description"] = description
        image_blocks.append(image_payload)

    if not image_blocks:
        return blocks

    updated = [
        block
        for block in blocks
        if not (block.get("kind") == "image" or block.get("role") == "image" or _is_image_slot(block.get("slot_id")))
    ]
    updated.extend(image_blocks)
    return updated


def _update_visuals(
    visuals: list[VisualBinding],
    *,
    slide_id: str,
    selected_assets: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    intent: str,
) -> list[VisualBinding]:
    if not selected_assets:
        return [
            visual.model_copy(update={"candidate_pool": candidate_pool})
            for visual in visuals
        ]

    updated: list[VisualBinding] = []
    for index, asset in enumerate(selected_assets, start=1):
        visual = visuals[index - 1] if index - 1 < len(visuals) else None
        slot_id = _image_slot_id(index)
        payload = {
            "visual_id": getattr(visual, "visual_id", "") or f"{slide_id}-visual-{index:02d}",
            "role": "primary" if index == 1 else "supporting",
            "asset_type": str(asset.get("asset_kind") or "image"),
            "slot_id": slot_id,
            "target_area": slot_id,
            "selected_asset_id": str(asset.get("asset_id") or ""),
            "selected_asset_path": str(asset.get("path") or ""),
            "selected_candidate": dict(asset),
            "candidate_pool": candidate_pool,
            "caption": str(asset.get("caption") or ""),
            "intent": getattr(visual, "intent", "") or intent,
        }
        updated.append(visual.model_copy(update=payload) if visual else VisualBinding(**payload))
    return updated


def _asset_paths_from_assets(assets: list[dict[str, Any]]) -> list[str]:
    paths = [
        str(asset.get("relative_path") or asset.get("path") or "").strip()
        for asset in assets
        if str(asset.get("relative_path") or asset.get("path") or "").strip()
    ]
    return list(dict.fromkeys(paths))


def _image_slot_id(index: int) -> str:
    return "image" if index <= 1 else f"image_{index:02d}"


def _image_block_id(slide_id: str, index: int) -> str:
    return f"{slide_id}-image" if index <= 1 else f"{slide_id}-image-{index:02d}"


def _is_image_slot(slot_id: Any) -> bool:
    value = str(slot_id or "").strip()
    return value == "image" or (value.startswith("image_") and value[6:].isdigit())
