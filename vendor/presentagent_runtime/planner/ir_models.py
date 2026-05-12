from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

IR_SCHEMA_VERSION = "0.2.0"

ALLOWED_LAYOUTS = [
    "hero",
    "title_only",
    "section_divider",
    "two_column",
    "three_column",
    "comparison",
    "metric_focus",
    "timeline",
    "process_flow",
    "quadrant",
    "image_focus",
    "quote_callout",
    "table_focus",
    "chart_focus",
    "closing",
]

LEGACY_LAYOUT_ALIASES = {
    "cover": "title_only",
    "bullets": "section_divider",
    "full_bleed_image": "image_focus",
}

RUNTIME_LAYOUTS = ALLOWED_LAYOUTS + list(LEGACY_LAYOUT_ALIASES)

ALLOWED_BLOCK_KINDS = [
    "headline",
    "summary",
    "bullet_list",
    "metric_strip",
    "process",
    "comparison",
    "quote",
    "callout",
    "image",
]

ALLOWED_SLOT_ROLES = [
    "title",
    "subtitle",
    "body",
    "supporting_body",
    "hero_visual",
    "supporting_visual",
    "metrics",
    "callout",
    "image",
    "footer",
]

ALLOWED_VISUAL_TYPES = [
    "figure",
    "diagram",
    "chart",
    "icon_cluster",
    "photo",
    "interface",
    "table",
    "text_only",
]

SlideLayoutType = Literal[
    "hero",
    "title_only",
    "section_divider",
    "two_column",
    "three_column",
    "comparison",
    "metric_focus",
    "timeline",
    "process_flow",
    "quadrant",
    "image_focus",
    "quote_callout",
    "table_focus",
    "chart_focus",
    "closing",
    "cover",
    "bullets",
    "full_bleed_image",
]

SlideKind = Literal["title", "section", "content", "closing", "direct_image"]


class RuntimeIRModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeckTheme(RuntimeIRModel):
    name: str = "runtime_clean"
    primary: str = "#1F2937"
    secondary: str = "#4B5563"
    accent: str = "#2563EB"
    background: str = "#FFFFFF"
    title_font: str = "Arial"
    body_font: str = "Arial"
    density: str = "balanced"
    style_guardrails: list[str] = Field(
        default_factory=lambda: [
            "Keep one main conclusion per slide.",
            "Separate visual and text regions clearly.",
            "Avoid overlapping text with pagecontent assets.",
        ]
    )


class IRMetadata(RuntimeIRModel):
    schema_name: str = "paper2ppt_code_runtime"
    schema_version: str = IR_SCHEMA_VERSION
    deck_id: str = ""
    stage: str = "planned"
    language: str = "zh"
    style: str = ""


class Storyline(RuntimeIRModel):
    summary: str = ""
    sections: list[str] = Field(default_factory=list)


class SlideBrief(RuntimeIRModel):
    brief_id: str
    slide_id: str = ""
    type: SlideKind = "content"
    section_id: str
    section_title: str = ""
    title: str
    core_message: str
    objective: str = ""
    content_points: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_headings: list[str] = Field(default_factory=list)
    source_excerpt: str = ""


class SlideBriefDeck(RuntimeIRModel):
    metadata: IRMetadata = Field(
        default_factory=lambda: IRMetadata(schema_name="presentagent.slide_briefs")
    )
    title_hint: str = ""
    subtitle_hint: str = ""
    storyline_hint: Storyline = Field(default_factory=Storyline)
    slide_briefs: list[SlideBrief] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)


class MaterialRequest(RuntimeIRModel):
    request_id: str
    target_slide_id: str
    asset_type: str = "image"
    purpose: str = ""
    preferred_layout_slot: str = ""
    acquisition_plan: dict[str, Any] = Field(default_factory=dict)


class VisualBinding(RuntimeIRModel):
    visual_id: str
    role: str = "primary"
    asset_type: str = "image"
    slot_id: str = ""
    target_area: str = ""
    selected_asset_id: str = ""
    selected_asset_path: str = ""
    selected_candidate: dict[str, Any] = Field(default_factory=dict)
    candidate_pool: list[dict[str, Any]] = Field(default_factory=list)
    caption: str = ""
    intent: str = ""


class SlideIR(RuntimeIRModel):
    metadata: IRMetadata = Field(
        default_factory=lambda: IRMetadata(schema_name="presentagent.slide_ir")
    )
    deck_id: str = ""
    slide_id: str
    page_num: int
    brief_id: str = ""
    type: SlideKind = "content"
    section_id: str = ""
    section_title: str = ""
    title: str = ""
    subtitle: str = ""
    core_message: str = ""
    objective: str = ""
    layout_type: SlideLayoutType = "section_divider"
    layout: dict[str, Any] = Field(default_factory=dict)
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    points: list[str] = Field(default_factory=list)
    visuals: list[VisualBinding] = Field(default_factory=list)
    asset_paths: list[str] = Field(default_factory=list)
    asset_base_dir: str = ""
    selected_asset_id: str = ""
    selected_asset_path: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)
    speaker_notes: list[str] = Field(default_factory=list)


class DeckIR(RuntimeIRModel):
    metadata: IRMetadata = Field(
        default_factory=lambda: IRMetadata(schema_name="presentagent.deck_ir")
    )
    title: str = ""
    subtitle: str = ""
    language: str = "zh"
    style: str = ""
    theme: DeckTheme = Field(default_factory=DeckTheme)
    storyline: Storyline = Field(default_factory=Storyline)
    material_requests: list[MaterialRequest] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)
    slide_manifest: list[dict[str, Any]] = Field(default_factory=list)
    source_asset_index: dict[str, Any] = Field(default_factory=dict)
    slides: list[SlideIR] = Field(default_factory=list)
