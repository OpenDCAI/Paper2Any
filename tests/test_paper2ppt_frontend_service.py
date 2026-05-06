from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.services.paper2ppt_frontend_service import (
    Paper2PPTFrontendService,
    _SLIDE_SCHEMA_VERSION,
)


def _build_theme(service: Paper2PPTFrontendService) -> dict:
    return service._build_fallback_theme(language="zh", style="")


def test_build_messages_requests_schema_blocks():
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

    assert '"blocks"' in system_prompt
    assert '"template_key"' in system_prompt
    assert "Do not output HTML, CSS" in system_prompt
    assert "html_template" not in system_prompt
    assert "title_cover" in system_prompt
    assert "dual_list" in system_prompt
    assert "fluid layout engine" in user_prompt
    assert '"visual_assets"' in user_prompt
    assert "Choose template_key from this fixed frontend-supported set" in user_prompt


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

    assert normalized["schema_version"] == _SLIDE_SCHEMA_VERSION
    assert normalized["layout_mode"] == "fluid"
    assert normalized["template_key"] == "split_media"
    assert normalized["generation_note"] == "Schema output"
    assert normalized["html_template"]
    assert normalized["css_code"]

    blocks = normalized["blocks"]
    assert len(blocks) == 4
    assert any(block["type"] == "image" and block["asset_key"] == "main_visual" for block in blocks)
    assert any(block["role"] == "title" and block["content"] == "Method Overview" for block in blocks)

    fields = {field["key"]: field for field in normalized["editable_fields"]}
    assert fields["title"]["value"] == "Method Overview"
    assert fields["summary"]["value"] == "A concise explanation of the end-to-end workflow."
    assert fields["key_points"]["items"] == ["Parse outline", "Render blocks"]


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
