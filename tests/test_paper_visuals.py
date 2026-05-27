from __future__ import annotations

from pathlib import Path

from PIL import Image

from dataflow_agent.utils_paper_visuals import (
    build_markdown_image_catalog,
    enrich_pagecontent_with_visual_assets,
)


def test_build_markdown_image_catalog_extracts_existing_mineru_images(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "pipeline.png"
    Image.new("RGB", (32, 18), color=(20, 80, 120)).save(image_path)

    markdown = """
## Method

![](images/pipeline.png)

Figure 2: Overall method pipeline.
"""

    catalog = build_markdown_image_catalog(markdown, tmp_path)

    assert catalog == [
        {
            "ref": "images/pipeline.png",
            "path": str(image_path.resolve()),
            "caption": "Figure 2: Overall method pipeline.",
            "nearby_text": "## Method ![](images/pipeline.png) Figure 2: Overall method pipeline.",
        }
    ]


def test_enrich_pagecontent_with_visual_assets_uses_real_catalog_ref():
    pagecontent = [
        {
            "title": "Method Overview",
            "layout_description": "Use a right-side figure for the method pipeline.",
            "key_points": ["Pipeline contains encoder and reranker"],
            "asset_ref": None,
        }
    ]
    catalog = [
        {
            "ref": "images/pipeline.png",
            "path": "/tmp/pipeline.png",
            "caption": "Figure 2: Method pipeline",
            "nearby_text": "encoder reranker method pipeline",
        }
    ]

    enriched = enrich_pagecontent_with_visual_assets(pagecontent, catalog)

    assert enriched[0]["asset_ref"] == "images/pipeline.png"
    assert enriched[0]["visual_assets"][0]["key"] == "main_visual"
    assert enriched[0]["visual_assets"][0]["src"] == "images/pipeline.png"
    assert enriched[0]["visual_assets"][0]["source_type"] == "paper_asset"
