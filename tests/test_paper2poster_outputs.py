from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.services.paper2poster_service import Paper2PosterService
from script.paper2poster_worker import _build_worker_result


def test_worker_result_requires_existing_pptx_file(tmp_path: Path) -> None:
    missing_pptx = tmp_path / "poster.pptx"

    result = _build_worker_result(str(missing_pptx), "", [])

    assert result["success"] is False
    assert result["output_pptx_path"] == ""
    assert "Poster PPTX output missing on disk" in result["message"]


def test_worker_result_drops_missing_png_file(tmp_path: Path) -> None:
    pptx_path = tmp_path / "poster.pptx"
    pptx_path.write_bytes(b"pptx")
    missing_png = tmp_path / "poster.png"

    result = _build_worker_result(str(pptx_path), str(missing_png), [])

    assert result["success"] is True
    assert result["output_pptx_path"] == str(pptx_path)
    assert result["output_png_path"] == ""
    assert any("Poster PNG output missing on disk" in err for err in result["errors"])


def test_service_output_resolver_only_accepts_existing_files(tmp_path: Path) -> None:
    pptx_path = tmp_path / "poster.pptx"
    pptx_path.write_bytes(b"pptx")

    assert Paper2PosterService._resolve_existing_output_file(str(pptx_path)) == pptx_path
    assert Paper2PosterService._resolve_existing_output_file(str(tmp_path / "missing.png")) is None
