from __future__ import annotations

import asyncio
import json
from pathlib import Path

from PIL import Image

from fastapi_app.services.paper2ppt_service import Paper2PPTService


def _make_source_slide(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 240), (20, 40, 80)).save(path)


def test_build_mask_from_spec_creates_binary_mask(tmp_path: Path) -> None:
    service = Paper2PPTService()
    source = tmp_path / "ppt_pages" / "page_000.png"
    mask = tmp_path / "edit_masks" / "page_000_mask.png"
    _make_source_slide(source)

    service._build_mask_from_spec(
        source_image_path=str(source),
        output_path=str(mask),
        mask_spec=json.dumps(
            {
                "shape": "rect",
                "x": 0.25,
                "y": 0.25,
                "width": 0.5,
                "height": 0.4,
            }
        ),
    )

    assert mask.exists()
    img = Image.open(mask).convert("L")
    assert img.size == (400, 240)
    assert img.getpixel((5, 5)) == 0
    assert img.getpixel((200, 120)) == 255


def test_prepare_edit_mask_uses_generated_img_path(tmp_path: Path) -> None:
    service = Paper2PPTService()
    source = tmp_path / "ppt_pages" / "page_000.png"
    _make_source_slide(source)

    result = asyncio.run(
        service._prepare_edit_mask(
            base_dir=tmp_path,
            pagecontent=[{"generated_img_path": str(source)}],
            page_id=0,
            get_down=True,
            mask_upload=None,
            mask_spec=json.dumps(
                {
                    "shape": "circle",
                    "x": 0.4,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.5,
                }
            ),
        )
    )

    assert result is not None
    mask_path = Path(result)
    assert mask_path.exists()
    assert mask_path.parent.name == "edit_masks"
