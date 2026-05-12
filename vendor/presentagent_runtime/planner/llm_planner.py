from __future__ import annotations

import json
from typing import Any

from .ir_models import (
    ALLOWED_BLOCK_KINDS,
    ALLOWED_LAYOUTS,
    ALLOWED_SLOT_ROLES,
    ALLOWED_VISUAL_TYPES,
    LEGACY_LAYOUT_ALIASES,
    DeckIR,
    MaterialRequest,
    SlideIR,
    VisualBinding,
)
from .layout_planner import plan_layout


def parse_json_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text or "{}")


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_points(values: Any) -> list[str]:
    if isinstance(values, list):
        return [normalize_text(value) for value in values if normalize_text(value)]
    text = normalize_text(values)
    return [text] if text else []


def normalize_layout_name(value: Any, fallback: Any) -> str:
    layout_name = normalize_text(value or fallback).lower().replace("-", "_").replace(" ", "_")
    layout_name = LEGACY_LAYOUT_ALIASES.get(layout_name, layout_name)
    if layout_name in ALLOWED_LAYOUTS:
        return layout_name
    fallback_name = normalize_text(fallback).lower().replace("-", "_").replace(" ", "_")
    fallback_name = LEGACY_LAYOUT_ALIASES.get(fallback_name, fallback_name)
    return fallback_name if fallback_name in ALLOWED_LAYOUTS else "section_divider"


_VISUAL_DOMINATED_LAYOUTS = {"image_focus", "chart_focus", "table_focus", "hero"}
_TEXT_ONLY_LAYOUTS = {"title_only", "section_divider", "closing", "quote_callout"}


def auto_repick_layout(
    layout_name: str,
    *,
    points_count: int,
    has_visual: bool,
    visuals_count: int,
    slide_type: str = "content",
) -> str:
    """Repick a layout when the one the LLM picked doesn't match content volume.

    Content volume wins: if the LLM asks for a visual-dominated layout but no
    visual is available, we fall back to a text layout (and vice versa); if
    points count mismatches the layout's slot budget, we pick a better-sized
    layout. Returns the (possibly replaced) layout name.
    """
    if slide_type == "title" and layout_name not in {"title_only", "hero"}:
        return "title_only"
    if slide_type == "closing" and layout_name != "closing":
        return "closing"

    if layout_name in _VISUAL_DOMINATED_LAYOUTS and not has_visual:
        return "two_column" if points_count >= 3 else "section_divider"

    if layout_name == "two_column" and not has_visual:
        if points_count >= 4:
            return "quadrant" if points_count >= 4 else "three_column"
        if points_count >= 2:
            return "three_column"
        return "section_divider"

    if has_visual and layout_name in _TEXT_ONLY_LAYOUTS and slide_type not in {"title", "closing"}:
        return "image_focus" if points_count <= 1 else "two_column"

    if layout_name == "quadrant" and points_count < 3:
        return "three_column" if points_count >= 2 else "section_divider"
    if layout_name == "three_column" and points_count < 2:
        return "section_divider" if not has_visual else "image_focus"
    if layout_name == "two_column" and points_count > 6 and not has_visual:
        return "quadrant" if points_count <= 8 else "section_divider"
    if layout_name == "metric_focus" and points_count > 4:
        return "two_column" if has_visual else "quadrant"
    if layout_name in {"timeline", "process_flow"} and points_count < 2:
        return "section_divider"

    return layout_name


def build_deck_context(deck_ir: DeckIR, *, materials: dict[str, Any] | None = None) -> dict[str, Any]:
    materials = materials or {}
    return {
        "metadata": deck_ir.metadata.model_dump(),
        "title": deck_ir.title,
        "subtitle": deck_ir.subtitle,
        "storyline": deck_ir.storyline.model_dump(),
        "theme": deck_ir.theme.model_dump(),
        "slide_manifest": list(deck_ir.slide_manifest),
        "planner_notes": list(deck_ir.planner_notes),
        "source_asset_index": dict(deck_ir.source_asset_index),
        "asset_catalog": materials.get("asset_catalog", []),
        "asset_request_contexts": materials.get("asset_request_contexts", []),
    }


class PagecontentDeckStagePlanner:
    """Plan deck-scoped context from normalized pagecontent briefs."""

    def __init__(self, client) -> None:
        self.client = client

    def plan_deck_stage(
        self,
        *,
        slide_briefs: dict[str, Any],
        base_deck_ir: DeckIR,
        materials: dict[str, Any],
    ) -> DeckIR:
        prompt = self._build_prompt(slide_briefs=slide_briefs, base_deck_ir=base_deck_ir, materials=materials)
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format="json",
        )
        payload = parse_json_response(raw)
        return base_deck_ir.model_copy(
            update={
                "title": normalize_text(payload.get("deck_title") or base_deck_ir.title),
                "subtitle": normalize_text(payload.get("deck_subtitle") or base_deck_ir.subtitle),
                "storyline": base_deck_ir.storyline.model_copy(
                    update={
                        "summary": normalize_text(
                            payload.get("storyline_summary") or base_deck_ir.storyline.summary
                        ),
                        "sections": [
                            normalize_text(section)
                            for section in payload.get("storyline_sections", [])
                            if normalize_text(section)
                        ]
                        or list(base_deck_ir.storyline.sections),
                    }
                ),
                "planner_notes": [
                    normalize_text(note)
                    for note in payload.get("planner_notes", [])
                    if normalize_text(note)
                ]
                or list(base_deck_ir.planner_notes),
            }
        )

    def _build_prompt(
        self,
        *,
        slide_briefs: dict[str, Any],
        base_deck_ir: DeckIR,
        materials: dict[str, Any],
    ) -> str:
        summary = {
            "title_hint": slide_briefs.get("title_hint", ""),
            "subtitle_hint": slide_briefs.get("subtitle_hint", ""),
            "storyline_hint": slide_briefs.get("storyline_hint", {}),
            "slide_briefs": slide_briefs.get("slide_briefs", []),
            "current_deck": {
                "title": base_deck_ir.title,
                "subtitle": base_deck_ir.subtitle,
                "storyline": base_deck_ir.storyline.model_dump(),
                "slide_manifest": list(base_deck_ir.slide_manifest),
                "theme": base_deck_ir.theme.model_dump(),
            },
            "asset_request_contexts": materials.get("asset_request_contexts", []),
        }
        return (
            "You are the deck-level planner for an editable PPT runtime. "
            "Use normalized pagecontent slide_briefs to set only deck-level context. "
            "Do not generate per-slide layout details here.\n\n"
            "Return ONE valid JSON object with keys:\n"
            "- deck_title: string\n"
            "- deck_subtitle: string\n"
            "- storyline_summary: string describing the full presentation flow\n"
            "- storyline_sections: array of section/slide titles in presentation order\n"
            "- planner_notes: array of short operational notes for slide planning\n\n"
            "Rules:\n"
            "- Preserve the pagecontent order and facts.\n"
            "- Keep natural-language fields in the same language as the input.\n"
            "- Do not invent new claims, experiments, numbers, or figures.\n"
            "- planner_notes should guide per-slide planning, especially visual/text balance.\n\n"
            f"Input summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
        )


class PagecontentSlidePlanner:
    """Plan concrete SlideIR fields while grounding each slide on DeckIR."""

    def __init__(self, client) -> None:
        self.client = client

    def plan_slides(
        self,
        *,
        slide_briefs: dict[str, Any],
        deck_ir: DeckIR,
        materials: dict[str, Any],
    ) -> DeckIR:
        brief_map = self._build_slide_brief_map(slide_briefs)
        planned_slides: list[SlideIR] = []
        planned_payloads: dict[str, dict[str, Any]] = {}
        previous_planned_summaries: list[dict[str, str]] = []

        for slide in deck_ir.slides:
            brief = brief_map.get(slide.slide_id) or brief_map.get(slide.brief_id) or {}
            payload = self.plan_slide(
                deck_ir=deck_ir,
                slide=slide,
                slide_brief=brief,
                materials=materials,
                previous_planned_slides=previous_planned_summaries,
            )
            planned_payloads[slide.slide_id] = payload
            merged_slide = self._merge_slide(base_slide=slide, planned_slide=payload, deck_ir=deck_ir)
            planned_slides.append(merged_slide)
            previous_planned_summaries.append(self._slide_summary(merged_slide))

        return deck_ir.model_copy(
            update={
                "slides": planned_slides,
                "material_requests": self._merge_material_requests(
                    list(deck_ir.material_requests),
                    planned_slides,
                    planned_payloads,
                ),
            }
        )

    def plan_slide(
        self,
        *,
        deck_ir: DeckIR,
        slide: SlideIR,
        slide_brief: dict[str, Any],
        materials: dict[str, Any],
        previous_planned_slides: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            deck_ir=deck_ir,
            slide=slide,
            slide_brief=slide_brief,
            materials=materials,
            previous_planned_slides=previous_planned_slides or [],
        )
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format="json",
        )
        return parse_json_response(raw)

    def _build_prompt(
        self,
        *,
        deck_ir: DeckIR,
        slide: SlideIR,
        slide_brief: dict[str, Any],
        materials: dict[str, Any],
        previous_planned_slides: list[dict[str, str]],
    ) -> str:
        deck_context = build_deck_context(deck_ir, materials=materials)
        slide_assets = self._source_assets_for_slide(
            deck_context.get("source_asset_index", {}),
            slide_id=slide.slide_id,
        )
        payload = {
            "deck_ir_context": deck_context,
            "current_slide_ir": slide.model_dump(),
            "current_slide_brief": slide_brief,
            "available_assets_for_this_slide": slide_assets,
            "previous_planned_slides": previous_planned_slides,
        }
        return (
            "You are the slide-level planner for an editable PPT runtime.\n"
            "Generate exactly one planned slide patch for the current slide.\n\n"
            "Critical rule: the slide must stay aligned with deck_ir_context. "
            "Use the deck title, subtitle, storyline, theme, slide_manifest, planner_notes, and source_asset_index as constraints. "
            "Do not change the slide topic or invent facts outside current_slide_brief/current_slide_ir.\n"
            "Prefer layouts and wording that are consistent with previous_planned_slides when possible.\n\n"
            "Return ONE valid JSON object only with these keys:\n"
            "- slide_id: copy current_slide_ir.slide_id\n"
            "- title: concise title aligned with the deck and brief\n"
            "- subtitle: visible slide subtitle only; empty string for layout instructions or repeated title\n"
            "- core_message: one sentence, supported by current_slide_brief\n"
            "- objective: one sentence explaining this slide's role in the deck storyline\n"
            f"- layout_name: exactly one of {', '.join(ALLOWED_LAYOUTS)}\n"
            "- points: array of 0-5 short strings using current_slide_brief.content_points/key points\n"
            "- visual_intent: string. If a visual is needed, state image slot/position, text position, and the point it supports. Empty string only for text-only slides.\n"
            "- material_request_title: string naming the needed visual; prefer asset_ref/available asset title when present\n"
            "- material_request_purpose: string used later as source_asset_index.description. It must describe visual role, text-visual separation, and overlap avoidance.\n"
            "- selected_asset_id: primary available_assets_for_this_slide.asset_id when relevant; otherwise empty string\n"
            "- selected_asset_ids: optional array of all available asset_ids that should be rendered on this slide, in visual order\n\n"
            "Layout selection (match content volume to capacity; prefer the smallest fit):\n"
            "  Decision tree:\n"
            "  1. points<=1 and 1 visual -> image_focus | chart_focus | table_focus (visual dominates, caption <= 2 lines).\n"
            "  2. points<=1 and no visual, opening/section slide -> title_only | section_divider.\n"
            "  3. points<=1 and no visual, single quote/thesis -> quote_callout.\n"
            "  4. points 2-3 parallel short items, no ordering -> three_column (each column <= 4 short lines).\n"
            "  5. points 2-3 with ordering/time -> timeline | process_flow.\n"
            "  6. points 2-3 contrast/versus -> comparison.\n"
            "  7. points 3-6 with 1 visual side-by-side -> two_column (left text <= 6 short lines, right visual 4:3).\n"
            "  8. points 4-6 parallel, no visual -> quadrant (each cell <= 3 short lines).\n"
            "  9. points emphasise numbers/KPIs -> metric_focus (<= 4 metrics).\n"
            "  10. final page -> closing.\n\n"
            "  Slot capacity cheat-sheet (exceeding these means pick a different layout, do not overflow):\n"
            "  - two_column: text area ~42% width x 60% height, 5-6 short lines max; visual area ~38% width x 55% height, aspect ~4:3.\n"
            "  - three_column: each column ~27% width x 50% height, <= 4 short lines each.\n"
            "  - quadrant: each cell ~40% width x 27% height, <= 3 short lines each.\n"
            "  - metric_focus: <= 4 metric cards, each a single number + <= 8-char label.\n"
            "  - image_focus / chart_focus / table_focus: visual fills 70%+ of body; text <= 2 lines.\n"
            "  - timeline / process_flow: 3-6 steps max; each step caption <= 10 chars.\n"
            "  - quote_callout: single quote <= 2 lines, attribution optional.\n"
            "  - title_only / section_divider / closing: no body points, only title/subtitle.\n\n"
            "  Anti-patterns to avoid:\n"
            "  - Do NOT pick two_column when there is no visual; use section_divider or three_column instead.\n"
            "  - Do NOT pick quadrant with fewer than 3 points.\n"
            "  - Do NOT pick metric_focus when points are prose, not numbers.\n"
            "  - Do NOT exceed the per-slot line budgets above; trim points or change layout.\n\n"
            f"Allowed block kinds for current_slide_ir.blocks: {', '.join(ALLOWED_BLOCK_KINDS)}.\n"
            f"Allowed slot roles: {', '.join(ALLOWED_SLOT_ROLES)}.\n"
            f"Allowed visual types: {', '.join(ALLOWED_VISUAL_TYPES)}.\n\n"
            "If the slide has available visual assets, plan every relevant asset as block kind `image`; use slot `image` for the primary asset and `image_02`, `image_03` for additional assets. "
            "For slides without a visual asset, keep material_request_title, material_request_purpose, and selected_asset_id empty. "
            "Do not put layout instructions into subtitle.\n\n"
            "Output must be operational, not decorative: describe where content goes and why. "
            "Keep language consistent with the input.\n\n"
            f"Planning input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _merge_slide(
        self,
        *,
        base_slide: SlideIR,
        planned_slide: dict[str, Any],
        deck_ir: DeckIR,
    ) -> SlideIR:
        planned_points = normalize_points(planned_slide.get("points") or base_slide.points)
        planned_title = normalize_text(planned_slide.get("title") or base_slide.title)
        planned_subtitle = normalize_text(planned_slide.get("subtitle") or base_slide.subtitle)
        planned_core_message = normalize_text(planned_slide.get("core_message") or base_slide.core_message)
        planned_objective = normalize_text(planned_slide.get("objective") or base_slide.objective)
        layout_name = normalize_layout_name(
            planned_slide.get("layout_name"),
            base_slide.layout.get("name") or base_slide.layout_type,
        )
        base_layout_name = normalize_layout_name(base_slide.layout.get("name") or base_slide.layout_type, base_slide.layout_type)
        visual_intent = normalize_text(planned_slide.get("visual_intent"))
        source_assets = self._source_assets_for_slide(deck_ir.source_asset_index, slide_id=base_slide.slide_id)
        selected_assets = self._selected_assets_for_slide(
            planned_slide,
            base_slide=base_slide,
            source_assets=source_assets,
        )
        selected_asset = selected_assets[0] if selected_assets else {}
        selected_asset_id = normalize_text(selected_asset.get("asset_id") or base_slide.selected_asset_id)
        selected_asset_path = normalize_text(selected_asset.get("path") or base_slide.selected_asset_path)
        material_request_purpose = normalize_text(planned_slide.get("material_request_purpose"))
        visual_slot = self._preferred_visual_slot(layout_name)
        has_visual_asset = bool(selected_assets or selected_asset_id or selected_asset_path or base_slide.asset_paths)

        repicked = auto_repick_layout(
            layout_name,
            points_count=len(planned_points),
            has_visual=has_visual_asset,
            visuals_count=len(selected_assets),
            slide_type=base_slide.type,
        )
        if repicked != layout_name:
            layout_name = repicked
            visual_slot = self._preferred_visual_slot(layout_name)

        visuals = list(base_slide.visuals)
        if selected_assets:
            visuals = [
                VisualBinding(
                    visual_id=f"{base_slide.slide_id}-visual-{index:02d}",
                    role="primary" if index == 1 else "supporting",
                    asset_type=str(asset.get("asset_kind") or "image"),
                    intent=visual_intent,
                    slot_id=self._image_slot_id(index),
                    target_area=self._image_slot_id(index),
                    selected_asset_id=normalize_text(asset.get("asset_id")),
                    selected_asset_path=normalize_text(asset.get("path")),
                    selected_candidate=dict(asset),
                    candidate_pool=source_assets,
                    caption=normalize_text(asset.get("caption")),
                )
                for index, asset in enumerate(selected_assets, start=1)
            ]
        elif visual_intent and has_visual_asset and not visuals:
            visuals = [
                VisualBinding(
                    visual_id=f"{base_slide.slide_id}-visual-01",
                    intent=visual_intent,
                    slot_id=visual_slot,
                    target_area=visual_slot,
                    selected_asset_id=selected_asset_id,
                    selected_asset_path=selected_asset_path,
                    selected_candidate=selected_asset,
                )
            ]
        elif visual_intent or selected_asset_id:
            visuals = [
                visual.model_copy(
                    update={
                        "intent": visual_intent or visual.intent,
                        "slot_id": visual.slot_id or visual_slot,
                        "target_area": visual.target_area or visual_slot,
                        "selected_asset_id": selected_asset_id or visual.selected_asset_id,
                        "selected_asset_path": selected_asset_path or visual.selected_asset_path,
                        "selected_candidate": selected_asset or visual.selected_candidate,
                    }
                )
                for visual in visuals
            ]

        layout = dict(base_slide.layout)
        if layout_name != base_layout_name:
            layout.pop("slots", None)
        layout.update(
            {
                "name": layout_name,
                "planner": "slide_llm",
                "visual_intent": visual_intent,
                "material_request_purpose": material_request_purpose,
            }
        )
        if layout_name == "title_only":
            layout.update(
                {
                    "title_align": "center",
                    "subtitle_align": "right",
                    "title_position": "center",
                    "subtitle_position": "bottom_right",
                }
            )
        blocks = self._normalize_or_build_blocks(
            planned_slide.get("blocks"),
            slide_id=base_slide.slide_id,
            title=planned_title,
            subtitle=planned_subtitle,
            summary=planned_core_message,
            points=planned_points,
            layout_name=layout_name,
        )
        if has_visual_asset:
            blocks = self._ensure_image_blocks(
                blocks,
                slide_id=base_slide.slide_id,
                selected_assets=selected_assets
                or [
                    {
                        "asset_id": selected_asset_id,
                        "path": selected_asset_path or (base_slide.asset_paths[0] if base_slide.asset_paths else ""),
                        "caption": normalize_text(selected_asset.get("caption")),
                    }
                ],
                description="",
            )

        image_count = max(len(selected_assets), 1) if has_visual_asset else 0
        layout = plan_layout(
            layout,
            layout_name=layout_name,
            points_count=len(planned_points),
            image_count=image_count,
        )

        return base_slide.model_copy(
            update={
                "title": planned_title,
                "subtitle": planned_subtitle,
                "core_message": planned_core_message,
                "objective": planned_objective,
                "points": planned_points,
                "type": self._resolve_slide_type(base_slide, layout_name),
                "blocks": blocks,
                "layout": layout,
                "layout_type": layout_name,
                "visuals": visuals,
                "asset_paths": self._asset_paths_from_assets(selected_assets) or list(base_slide.asset_paths),
                "selected_asset_id": selected_asset_id,
                "selected_asset_path": selected_asset_path,
            }
        )

    def _merge_material_requests(
        self,
        material_requests: list[MaterialRequest],
        slides: list[SlideIR],
        slide_payloads: dict[str, dict[str, Any]],
    ) -> list[MaterialRequest]:
        updated_requests = list(material_requests)
        for slide in slides:
            planned_slide = slide_payloads.get(slide.slide_id, {})
            material_title = normalize_text(planned_slide.get("material_request_title"))
            material_purpose = normalize_text(planned_slide.get("material_request_purpose"))
            has_visual_asset = bool(slide.asset_paths or slide.visuals or slide.selected_asset_id or slide.selected_asset_path)
            asks_for_visual = bool(material_title or (material_purpose and not self._is_no_visual_request(material_purpose)))
            needs_material_request = has_visual_asset or asks_for_visual
            existing_request_index = next(
                (
                    index
                    for index, request in enumerate(updated_requests)
                    if request.target_slide_id == slide.slide_id
                ),
                None,
            )
            if needs_material_request and existing_request_index is not None:
                existing_request = updated_requests[existing_request_index]
                acquisition_plan = dict(existing_request.acquisition_plan)
                if slide.selected_asset_id:
                    acquisition_plan["selected_asset_id"] = slide.selected_asset_id
                selected_asset_ids = self._selected_asset_ids_from_slide(slide)
                if selected_asset_ids:
                    acquisition_plan["selected_asset_ids"] = selected_asset_ids
                updated_requests[existing_request_index] = existing_request.model_copy(
                    update={
                        "purpose": material_purpose or existing_request.purpose,
                        "preferred_layout_slot": self._preferred_visual_slot(slide.layout_type),
                        "acquisition_plan": acquisition_plan,
                    }
                )
            elif needs_material_request:
                updated_requests.append(
                    MaterialRequest(
                        request_id=f"material-{slide.page_num:03d}",
                        target_slide_id=slide.slide_id,
                        asset_type="image",
                        purpose=material_purpose or slide.core_message or slide.objective,
                        preferred_layout_slot=self._preferred_visual_slot(slide.layout_type),
                        acquisition_plan={
                            "source_options": ["pagecontent_assets"],
                            "selected_asset_id": slide.selected_asset_id,
                            "selected_asset_ids": self._selected_asset_ids_from_slide(slide),
                        },
                    )
                )
        return updated_requests

    @staticmethod
    def _build_slide_brief_map(slide_briefs: dict[str, Any]) -> dict[str, dict[str, Any]]:
        brief_map: dict[str, dict[str, Any]] = {}
        for brief in slide_briefs.get("slide_briefs", []):
            if not isinstance(brief, dict):
                continue
            slide_id = normalize_text(brief.get("slide_id"))
            brief_id = normalize_text(brief.get("brief_id"))
            if slide_id:
                brief_map[slide_id] = brief
            if brief_id:
                brief_map[brief_id] = brief
        return brief_map

    @staticmethod
    def _slide_summary(slide: SlideIR) -> dict[str, str]:
        return {
            "slide_id": slide.slide_id,
            "title": slide.title,
            "layout_type": slide.layout_type,
            "core_message": slide.core_message,
        }

    @staticmethod
    def _source_assets_for_slide(source_asset_index: Any, *, slide_id: str) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        if not isinstance(source_asset_index, dict):
            return assets

        for key, value in source_asset_index.items():
            if isinstance(value, dict):
                if str(value.get("target_slide_id") or "") == slide_id:
                    assets.append(value)
                continue

            if key != slide_id:
                continue
            raw_paths = value if isinstance(value, list) else [value]
            for index, raw_path in enumerate(raw_paths, start=1):
                path = normalize_text(raw_path)
                if path:
                    assets.append(
                        {
                            "asset_id": f"{slide_id}-asset-{index:02d}",
                            "path": path,
                            "relative_path": path,
                            "target_slide_id": slide_id,
                        }
                    )
        return assets

    @staticmethod
    def _is_no_visual_request(text: str) -> bool:
        normalized = normalize_text(text).lower()
        return any(
            token in normalized
            for token in (
                "无图",
                "无需视觉",
                "无需图",
                "不需要视觉",
                "不放图",
                "no visual",
                "no image",
                "text-only",
                "text only",
            )
        )

    @staticmethod
    def _normalize_or_build_blocks(
        raw_blocks: Any,
        *,
        slide_id: str,
        title: str,
        subtitle: str,
        summary: str,
        points: list[str],
        layout_name: str,
    ) -> list[dict[str, Any]]:
        if isinstance(raw_blocks, list):
            blocks = [
                PagecontentSlidePlanner._normalize_block(block, slide_id=slide_id, index=index)
                for index, block in enumerate(raw_blocks, start=1)
                if isinstance(block, dict)
            ]
            visible_blocks = [
                block
                for block in blocks
                if (
                    block.get("content")
                    or block.get("text")
                    or block.get("items")
                    or block.get("kind") == "headline"
                    or (block.get("kind") == "image" and (block.get("path") or block.get("asset_id")))
                )
            ]
            if visible_blocks:
                return visible_blocks
        return PagecontentSlidePlanner._build_blocks_for_layout(
            slide_id=slide_id,
            title=title,
            subtitle=subtitle,
            summary=summary,
            points=points,
            layout_name=layout_name,
        )

    @staticmethod
    def _normalize_block(block: dict[str, Any], *, slide_id: str, index: int) -> dict[str, Any]:
        role = normalize_text(block.get("role") or block.get("slot_id") or "body")
        kind = normalize_text(block.get("kind") or "summary")
        slot_id = normalize_text(block.get("slot_id") or block.get("slot") or role or "body")
        content = normalize_text(block.get("content") or block.get("text"))
        items = normalize_points(block.get("items"))
        normalized = {
            "block_id": normalize_text(block.get("block_id") or f"{slide_id}-block-{index:02d}"),
            "role": role,
            "kind": kind,
            "slot_id": slot_id,
        }
        if content:
            normalized["content"] = content
            normalized["text"] = content
        if items:
            normalized["items"] = items
        caption = normalize_text(block.get("caption"))
        if caption:
            normalized["caption"] = caption
        path = normalize_text(block.get("path") or block.get("asset_path"))
        if path:
            normalized["path"] = path
        asset_id = normalize_text(block.get("asset_id") or block.get("selected_asset_id"))
        if asset_id:
            normalized["asset_id"] = asset_id
        return normalized

    @staticmethod
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
        description: str = "",
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
        if description:
            payload["description"] = description
        return payload

    @staticmethod
    def _ensure_image_blocks(
        blocks: list[dict[str, Any]],
        *,
        slide_id: str,
        selected_assets: list[dict[str, Any]],
        description: str = "",
    ) -> list[dict[str, Any]]:
        image_blocks: list[dict[str, Any]] = []
        for index, asset in enumerate(selected_assets, start=1):
            path = normalize_text(asset.get("path") or asset.get("relative_path"))
            asset_id = normalize_text(asset.get("asset_id") or asset.get("selected_asset_id"))
            if not path and not asset_id:
                continue
            image_block = PagecontentSlidePlanner._block(
                PagecontentSlidePlanner._image_block_id(slide_id, index),
                role="image",
                kind="image",
                slot_id=PagecontentSlidePlanner._image_slot_id(index),
                caption=normalize_text(asset.get("caption")),
                path=path,
                description=description,
            )
            if asset_id:
                image_block["asset_id"] = asset_id
            image_blocks.append(image_block)

        if not image_blocks:
            return blocks

        updated = [
            block
            for block in blocks
            if not (block.get("kind") == "image" or block.get("slot_id") == "image" or block.get("role") == "image")
        ]
        updated.extend(image_blocks)
        return updated

    @staticmethod
    def _ensure_image_slot(layout: dict[str, Any], layout_name: str) -> dict[str, Any]:
        return PagecontentSlidePlanner._ensure_image_slots(layout, layout_name, image_count=1)

    @staticmethod
    def _ensure_image_slots(layout: dict[str, Any], layout_name: str, *, image_count: int) -> dict[str, Any]:
        if image_count <= 0:
            return layout
        slots = [dict(slot) for slot in (layout.get("slots") or []) if isinstance(slot, dict)]
        existing_by_id = {str(slot.get("slot_id") or ""): slot for slot in slots}
        base_slot = existing_by_id.get("image") or PagecontentSlidePlanner._default_image_slot(layout_name)
        generated_slots = PagecontentSlidePlanner._split_image_slots(base_slot, image_count)
        kept_slots = [
            slot
            for slot in slots
            if not PagecontentSlidePlanner._is_generated_image_slot(str(slot.get("slot_id") or ""))
        ]
        slot_ids = {str(slot.get("slot_id") or "") for slot in kept_slots}
        for slot in generated_slots:
            slot_id = str(slot.get("slot_id") or "")
            if slot_id in slot_ids:
                for index, existing_slot in enumerate(kept_slots):
                    if str(existing_slot.get("slot_id") or "") == slot_id:
                        kept_slots[index] = slot
                        break
            else:
                kept_slots.append(slot)
                slot_ids.add(slot_id)
        return {**layout, "slots": kept_slots}

    @staticmethod
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
                "slot_id": PagecontentSlidePlanner._image_slot_id(index),
                "x_ratio": x,
                "y_ratio": y + (index - 1) * (each_h + gap),
                "w_ratio": w,
                "h_ratio": each_h,
            }
            for index in range(1, image_count + 1)
        ]

    @staticmethod
    def _is_generated_image_slot(slot_id: str) -> bool:
        return slot_id == "image" or (slot_id.startswith("image_") and slot_id[6:].isdigit())

    @staticmethod
    def _image_slot_id(index: int) -> str:
        return "image" if index <= 1 else f"image_{index:02d}"

    @staticmethod
    def _image_block_id(slide_id: str, index: int) -> str:
        return f"{slide_id}-image" if index <= 1 else f"{slide_id}-image-{index:02d}"

    @staticmethod
    def _selected_assets_for_slide(
        planned_slide: dict[str, Any],
        *,
        base_slide: SlideIR,
        source_assets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        requested_ids = normalize_points(planned_slide.get("selected_asset_ids"))
        primary_id = normalize_text(planned_slide.get("selected_asset_id") or base_slide.selected_asset_id)
        if primary_id and primary_id not in requested_ids:
            requested_ids.insert(0, primary_id)

        if requested_ids:
            indexed = {normalize_text(asset.get("asset_id")): dict(asset) for asset in source_assets}
            selected = [indexed[asset_id] for asset_id in requested_ids if asset_id in indexed]
            if selected:
                return selected

        selected_path = normalize_text(base_slide.selected_asset_path)
        if selected_path:
            for asset in source_assets:
                paths = {normalize_text(asset.get("path")), normalize_text(asset.get("relative_path"))}
                if selected_path in paths:
                    return [dict(asset)]

        if source_assets:
            return [dict(asset) for asset in source_assets]

        return [
            {
                "asset_id": f"{base_slide.slide_id}-asset-{index:02d}",
                "path": path,
                "relative_path": path,
                "target_slide_id": base_slide.slide_id,
            }
            for index, path in enumerate(base_slide.asset_paths, start=1)
            if normalize_text(path)
        ]

    @staticmethod
    def _asset_paths_from_assets(assets: list[dict[str, Any]]) -> list[str]:
        paths = [
            normalize_text(asset.get("relative_path") or asset.get("path"))
            for asset in assets
            if normalize_text(asset.get("relative_path") or asset.get("path"))
        ]
        return list(dict.fromkeys(paths))

    @staticmethod
    def _selected_asset_ids_from_slide(slide: SlideIR) -> list[str]:
        ids: list[str] = []
        for visual in slide.visuals:
            asset_id = normalize_text(visual.selected_asset_id)
            if asset_id:
                ids.append(asset_id)
        for block in slide.blocks:
            asset_id = normalize_text(block.get("asset_id") or block.get("selected_asset_id"))
            if asset_id:
                ids.append(asset_id)
        if slide.selected_asset_id:
            ids.insert(0, slide.selected_asset_id)
        return list(dict.fromkeys(ids))

    @staticmethod
    def _default_image_slot(layout_name: str) -> dict[str, float | str]:
        if layout_name in {"hero", "image_focus"}:
            return {"slot_id": "image", "x_ratio": 0.0, "y_ratio": 0.0, "w_ratio": 1.0, "h_ratio": 1.0}
        if layout_name in {"table_focus", "chart_focus"}:
            return {"slot_id": "image", "x_ratio": 0.08, "y_ratio": 0.22, "w_ratio": 0.84, "h_ratio": 0.5}
        if layout_name == "quote_callout":
            return {"slot_id": "image", "x_ratio": 0.58, "y_ratio": 0.28, "w_ratio": 0.34, "h_ratio": 0.34}
        return {"slot_id": "image", "x_ratio": 0.54, "y_ratio": 0.23, "w_ratio": 0.38, "h_ratio": 0.56}

    @staticmethod
    def _build_blocks_for_layout(
        *,
        slide_id: str,
        title: str,
        subtitle: str,
        summary: str,
        points: list[str],
        layout_name: str,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if title:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-title", role="title", kind="headline", slot_id="title", text=title))

        body_text = summary if summary and summary != subtitle else ""
        if layout_name == "title_only":
            if subtitle:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-subtitle", role="subtitle", kind="summary", slot_id="subtitle", text=subtitle))
            return blocks

        if layout_name in {"three_column", "quadrant"} and points:
            slots = ["body", "supporting_body", "callout"] if layout_name == "three_column" else ["body", "supporting_body", "metrics", "callout"]
            for index, item in enumerate(points[: len(slots)]):
                blocks.append(
                    PagecontentSlidePlanner._block(
                        f"{slide_id}-block-{index + 1:02d}",
                        role=slots[index],
                        kind="summary",
                        slot_id=slots[index],
                        text=item,
                    )
                )
            return blocks

        if layout_name in {"process_flow", "timeline"} and points:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-process", role="body", kind="process", slot_id="body", items=points))
            if body_text:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-supporting", role="supporting_body", kind="summary", slot_id="supporting_body", text=body_text))
            return blocks

        if layout_name == "metric_focus":
            if points:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-metrics", role="metrics", kind="metric_strip", slot_id="metrics", items=points[:4]))
            if body_text:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-body", role="body", kind="summary", slot_id="body", text=body_text))
            return blocks

        if layout_name == "comparison" and points:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-comparison", role="body", kind="comparison", slot_id="body", items=points))
            if body_text:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-callout", role="callout", kind="callout", slot_id="callout", text=body_text))
            return blocks

        if layout_name == "quote_callout":
            quote_text = points[0] if points else body_text
            if quote_text:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-quote", role="callout", kind="quote", slot_id="callout", text=quote_text))
            remaining = points[1:] if points else []
            if remaining:
                blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-body", role="body", kind="bullet_list", slot_id="body", items=remaining))
            return blocks

        if points:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-body", role="body", kind="bullet_list", slot_id="body", items=points))
        elif body_text:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-body", role="body", kind="summary", slot_id="body", text=body_text))

        if layout_name in {"two_column", "table_focus", "chart_focus"} and body_text and points:
            blocks.append(PagecontentSlidePlanner._block(f"{slide_id}-callout", role="callout", kind="callout", slot_id="callout", text=body_text))
        return blocks

    @staticmethod
    def _resolve_slide_type(slide: SlideIR, layout_name: str) -> str:
        title = normalize_text(slide.title).lower()
        if layout_name == "title_only":
            if slide.page_num == 1:
                return "title"
            if "致谢" in title or "thank" in title or "closing" in title:
                return "closing"
        if layout_name == "closing":
            return "closing"
        if layout_name == "section_divider":
            return "section" if slide.type == "section" else slide.type
        if layout_name == "image_focus" and slide.asset_paths and slide.type == "direct_image":
            return "direct_image"
        return slide.type

    @staticmethod
    def _preferred_visual_slot(layout_name: str) -> str:
        return "image"


class PagecontentDeckPlanner:
    """Compatibility orchestrator for the two-stage pagecontent planner.

    Stage 1 plans deck-level context. Stage 2 plans each SlideIR while reading
    that deck context, so per-slide planning stays aligned with the overall deck.
    """

    def __init__(self, client) -> None:
        self.deck_planner = PagecontentDeckStagePlanner(client)
        self.slide_planner = PagecontentSlidePlanner(client)

    def plan_deck(
        self,
        *,
        slide_briefs: dict[str, Any],
        base_deck_ir: DeckIR,
        materials: dict[str, Any],
    ) -> DeckIR:
    
        planned_deck_ir = self.deck_planner.plan_deck_stage(
            slide_briefs=slide_briefs,
            base_deck_ir=base_deck_ir,
            materials=materials,
        )
        return self.slide_planner.plan_slides(
            slide_briefs=slide_briefs,
            deck_ir=planned_deck_ir,
            materials=materials,
        )


__all__ = [
    "ALLOWED_LAYOUTS",
    "PagecontentDeckPlanner",
    "PagecontentDeckStagePlanner",
    "PagecontentSlidePlanner",
    "build_deck_context",
    "normalize_layout_name",
    "normalize_points",
    "normalize_text",
    "parse_json_response",
]
