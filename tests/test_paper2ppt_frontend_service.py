from __future__ import annotations

import sys
from pathlib import Path

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.services.paper2ppt_frontend_service import (
    Paper2PPTFrontendService,
    _CANVAS_SCHEMA_VERSION,
    _format_exception_for_log,
)


def _build_theme(service: Paper2PPTFrontendService) -> dict:
    return service._build_fallback_theme(language="zh", style="")


def _collect_canvas_refs(node: dict) -> set[str]:
    refs: set[str] = set()
    props = node.get("props") if isinstance(node.get("props"), dict) else {}
    for key, value in props.items():
        if key.endswith("_ref") or key.endswith("Ref") or key == "ref":
            refs.add(str(value))
    for child in node.get("children") or []:
        if isinstance(child, dict):
            refs.update(_collect_canvas_refs(child))
    return refs


def test_build_messages_requests_canvas_schema():
    service = Paper2PPTFrontendService()
    messages = service._build_messages(
        outline_item={
            "title": "Method Overview",
            "layout_description": "Explain the core pipeline.",
            "key_points": ["Step A", "Step B"],
        },
        slide_index=0,
        slide_count=6,
        language="zh",
        style="",
        edit_prompt=None,
        current_slide=None,
        theme=_build_theme(service),
        deck_identity=service._build_deck_identity_summary(_build_theme(service)),
        reference_slides=[],
        visual_assets=[
            {
                "key": "main_visual",
                "label": "Main Visual",
                "source_type": "paper_asset",
                "alt": "Pipeline diagram",
            }
        ],
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert '"root"' in system_prompt
    assert '"content"' in system_prompt
    assert '"visual_spec"' in system_prompt
    assert "Do not output blocks" in system_prompt
    assert "html_template" in system_prompt
    assert "layout_description is layout intent only" in system_prompt
    assert "Canvas root/content/visual_spec JSON" in user_prompt
    assert '"visual_assets"' in user_prompt
    assert "root tree fully covers all meaningful visible content" in user_prompt


def test_normalize_slide_payload_accepts_schema_blocks():
    service = Paper2PPTFrontendService()
    theme = _build_theme(service)
    visual_assets = [
        {
            "key": "main_visual",
            "label": "Main Visual",
            "src": "",
            "alt": "Main visual",
            "source_type": "paper_asset",
            "storage_path": "",
        }
    ]

    normalized = service._normalize_slide_payload(
        payload={
            "title": "Method Overview",
            "template_key": "split_media",
            "layout_mode": "fluid",
            "blocks": [
                {
                    "id": "title",
                    "type": "text",
                    "role": "title",
                    "content": "Method Overview",
                    "layout": {"zone": "header", "span": 12},
                },
                {
                    "id": "summary",
                    "type": "text",
                    "role": "summary",
                    "content": "A concise explanation of the end-to-end workflow.",
                    "layout": {"zone": "main", "span": 6},
                },
                {
                    "id": "key_points",
                    "type": "list",
                    "role": "key_points",
                    "items": ["Parse outline", "Render blocks"],
                    "layout": {"zone": "main", "span": 6},
                },
                {
                    "id": "figure",
                    "type": "image",
                    "role": "main_visual",
                    "asset_key": "main_visual",
                    "layout": {"zone": "aside", "span": 6},
                },
            ],
            "generation_note": "Schema output",
        },
        outline_item={
            "title": "Method Overview",
            "layout_description": "Explain the pipeline.",
            "key_points": ["Parse outline", "Render blocks"],
        },
        slide_index=0,
        slide_count=6,
        theme=theme,
        visual_assets=visual_assets,
    )

    assert normalized["schema_version"] == _CANVAS_SCHEMA_VERSION
    assert normalized["render_engine"] == "canvas"
    assert normalized["layout_mode"] == "fluid"
    assert normalized["template_key"] == "split_media"
    assert normalized["generation_note"] == "Schema output"
    assert normalized["html_template"]
    assert normalized["css_code"]
    assert normalized["root"]
    assert normalized["content"]
    assert normalized["visual_spec"]

    blocks = normalized["blocks"]
    assert len(blocks) == 4
    assert any(block["type"] == "image" and block["asset_key"] == "main_visual" for block in blocks)
    assert any(block["role"] == "title" and block["content"] == "Method Overview" for block in blocks)

    fields = {field["key"]: field for field in normalized["editable_fields"]}
    assert fields["title"]["value"] == "Method Overview"
    assert fields["summary"]["value"] == "A concise explanation of the end-to-end workflow."
    assert fields["key_points"]["items"] == ["Parse outline", "Render blocks"]


def test_cover_layout_description_does_not_become_visible_content():
    service = Paper2PPTFrontendService()
    theme = _build_theme(service)
    layout_description = (
        "整页仅保留标题与汇报人信息，采用居中排版；标题置于页面中上部，汇报人置于标题下方，"
        "留出大面积空白，形成学术汇报封面效果；不放置其他说明、图表或装饰性正文。"
    )

    normalized = service._normalize_slide_payload(
        payload={
            "title": "CASCADE：通过自主开发与进化实现累积式智能体技能创造\n汇报人：XXX",
            "content": {
                "title": "CASCADE：通过自主开发与进化实现累积式智能体技能创造\n汇报人：XXX",
                "summary": layout_description,
            },
            "root": {
                "type": "container",
                "id": "root",
                "style": {"direction": "column"},
                "children": [
                    {"type": "component", "id": "title", "component": "heading", "props": {"text_ref": "title"}},
                    {"type": "component", "id": "summary", "component": "text", "props": {"text_ref": "summary"}},
                ],
            },
            "editable_fields": [
                {"key": "summary", "label": "Summary", "type": "textarea", "value": layout_description},
            ],
            "visual_spec": {"node_styles": {"summary": {"font_size": 23}}},
        },
        outline_item={
            "title": "CASCADE：通过自主开发与进化实现累积式智能体技能创造\n汇报人：XXX",
            "layout_description": layout_description,
            "key_points": [],
        },
        slide_index=0,
        slide_count=6,
        theme=theme,
        visual_assets=[],
    )

    assert normalized["template_key"] == "title_cover"
    assert normalized["layout_family"] == "title_cover"
    assert set(normalized["content"]) == {"eyebrow", "title", "presenter", "assets"}
    assert layout_description not in str(normalized["content"])
    assert layout_description not in str(normalized["editable_fields"])
    assert "summary" not in _collect_canvas_refs(normalized["root"])


def test_clean_canvas_visual_number_handles_clamped_int_bounds():
    service = Paper2PPTFrontendService()

    assert service._clean_canvas_visual_number(-5, min_value=0, max_value=12) == 0
    assert service._clean_canvas_visual_number(99, min_value=0, max_value=12) == 12
    assert service._clean_canvas_visual_number(10.5, min_value=0, max_value=12) == 10.5


def test_format_exception_for_log_includes_http_exception_detail():
    exc = HTTPException(status_code=502, detail="frontend slide generation failed")

    assert _format_exception_for_log(exc) == (
        "HTTPException status_code=502 detail=frontend slide generation failed"
    )


def test_normalize_template_key_aligns_with_supported_frontend_templates():
    service = Paper2PPTFrontendService()

    assert service._normalize_template_key(
        "cover",
        blocks=[],
        visual_assets=[],
    ) == "title_cover"

    assert service._normalize_template_key(
        "unknown_custom_layout",
        blocks=[
            {"type": "stat"},
            {"type": "stat"},
            {"type": "text"},
        ],
        visual_assets=[],
    ) == "metrics_dashboard"
