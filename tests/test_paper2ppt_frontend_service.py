from __future__ import annotations

import sys
import asyncio
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

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


def _run(coro):
    return asyncio.run(coro)


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


def test_prepare_visual_assets_prefers_paper_asset_in_hybrid_mode(tmp_path: Path):
    service = Paper2PPTFrontendService()
    image_dir = tmp_path / "input" / "auto" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "figure_1.png"
    Image.new("RGB", (64, 36), color=(40, 90, 140)).save(image_path)

    assets = _run(
        service._prepare_visual_assets(
            base_dir=tmp_path,
            outline_item={"title": "Method", "asset_ref": "images/figure_1.png"},
            slide_index=0,
            include_images=True,
            image_mode="hybrid",
            image_style="academic_illustration",
            image_model=None,
            image_api_url="",
            image_api_key="",
            chat_api_url="",
            api_key="",
            model="",
            theme=_build_theme(service),
            current_slide=None,
        )
    )

    assert len(assets) == 1
    assert assets[0]["source_type"] == "paper_asset"
    assert assets[0]["storage_path"] == str(image_path.resolve())
    assert assets[0]["key"] == "main_visual"


def test_prepare_visual_assets_generated_mode_skips_paper_asset(tmp_path: Path, monkeypatch):
    service = Paper2PPTFrontendService()
    image_dir = tmp_path / "input" / "auto" / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (64, 36), color=(40, 90, 140)).save(image_dir / "figure_1.png")

    async def fake_generate_visual_asset(**kwargs):
        return {
            "key": "main_visual",
            "label": "Generated",
            "src": "/tmp/generated.png",
            "alt": "generated",
            "source_type": "generated",
            "storage_path": "/tmp/generated.png",
        }

    monkeypatch.setattr(service, "_generate_visual_asset", fake_generate_visual_asset)

    assets = _run(
        service._prepare_visual_assets(
            base_dir=tmp_path,
            outline_item={"title": "Method", "asset_ref": "images/figure_1.png"},
            slide_index=0,
            include_images=True,
            image_mode="generated",
            image_style="academic_illustration",
            image_model=None,
            image_api_url="http://image-api.test/v1",
            image_api_key="key",
            chat_api_url="",
            api_key="",
            model="",
            theme=_build_theme(service),
            current_slide=None,
        )
    )

    assert len(assets) == 1
    assert assets[0]["source_type"] == "generated"
    assert assets[0]["label"] == "Generated"


def test_prepare_visual_assets_hybrid_falls_back_to_generated_placeholder_without_key(tmp_path: Path):
    service = Paper2PPTFrontendService()

    assets = _run(
        service._prepare_visual_assets(
            base_dir=tmp_path,
            outline_item={"title": "No figure slide", "key_points": ["Explain the idea"]},
            slide_index=2,
            include_images=True,
            image_mode="hybrid",
            image_style="academic_illustration",
            image_model=None,
            image_api_url="",
            image_api_key="",
            chat_api_url="",
            api_key="",
            model="",
            theme=_build_theme(service),
            current_slide=None,
        )
    )

    assert len(assets) == 1
    assert assets[0]["source_type"] == "generated"
    assert assets[0]["storage_path"] == ""
    assert "Create one supporting image" in assets[0]["prompt"]


def test_prepare_visual_assets_none_mode_returns_no_assets(tmp_path: Path):
    service = Paper2PPTFrontendService()

    assets = _run(
        service._prepare_visual_assets(
            base_dir=tmp_path,
            outline_item={"title": "Text only", "asset_ref": "images/figure_1.png"},
            slide_index=0,
            include_images=False,
            image_mode="none",
            image_style="academic_illustration",
            image_model=None,
            image_api_url="",
            image_api_key="",
            chat_api_url="",
            api_key="",
            model="",
            theme=_build_theme(service),
            current_slide=None,
        )
    )

    assert assets == []
