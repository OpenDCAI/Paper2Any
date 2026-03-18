from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTERTOOL_ROOT = PROJECT_ROOT / "dataflow_agent" / "toolkits" / "postertool"
if str(POSTERTOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTERTOOL_ROOT))


def _install_import_stubs() -> None:
    langgraph_utils_module = types.ModuleType("utils.langgraph_utils")

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self) -> None:
            pass

        def step(self, _message: str):
            raise RuntimeError("LangGraphAgent should not be used in this unit test")

    def _load_prompt(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    langgraph_utils_module.LangGraphAgent = _DummyAgent
    langgraph_utils_module.extract_json = lambda content: content
    langgraph_utils_module.load_prompt = _load_prompt
    sys.modules.setdefault("utils.langgraph_utils", langgraph_utils_module)


_install_import_stubs()

from src.agents import parser as parser_module


def _create_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_parser_loads_prompts_without_relying_on_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    parser = parser_module.Parser()

    assert parser.enhanced_abt_prompt
    assert parser.visual_classification_prompt
    assert parser.title_authors_prompt
    assert parser.section_extraction_prompt


def test_parser_falls_back_to_pymupdf_when_marker_missing(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, "Poster fallback parser test.\nResults and method summary.")

    parser = parser_module.Parser()
    content_dir = tmp_path / "content"
    assets_dir = tmp_path / "assets"
    content_dir.mkdir()
    assets_dir.mkdir()
    monkeypatch.setattr(
        parser,
        "_extract_raw_text_with_mineru",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mineru unavailable in test")),
    )

    raw_text, raw_result = parser._extract_raw_text(str(pdf_path), content_dir)
    figures, tables = parser._extract_assets(raw_result, "sample", assets_dir)

    assert raw_result is None
    assert "Poster fallback parser test." in raw_text
    assert (content_dir / "raw.md").read_text(encoding="utf-8") == raw_text
    assert figures == {}
    assert tables == {}
    assert json.loads((assets_dir / "figures.json").read_text(encoding="utf-8")) == {}
    assert json.loads((assets_dir / "tables.json").read_text(encoding="utf-8")) == {}
    assert json.loads((assets_dir / "fig_tab_caption_mapping.json").read_text(encoding="utf-8")) == {}


def test_parser_extracts_assets_via_mineru_blocks(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, "Poster MinerU parser test.\nFigure 1 overview.\nTable 1 results.")

    async def _fake_batch_extract(_image_paths, port):
        assert port == 8010
        return [[
            {"type": "title", "bbox": [0.1, 0.05, 0.8, 0.12], "text": "GovBench Poster"},
            {"type": "text", "bbox": [0.1, 0.15, 0.8, 0.2], "text": "Figure 1. System overview"},
            {"type": "figure", "bbox": [0.1, 0.22, 0.5, 0.48]},
            {"type": "text", "bbox": [0.1, 0.5, 0.8, 0.55], "text": "Table 1. Main results"},
            {"type": "table", "bbox": [0.1, 0.58, 0.6, 0.85]},
        ]]

    monkeypatch.setattr(parser_module, "run_aio_batch_two_step_extract", _fake_batch_extract)

    parser = parser_module.Parser()
    content_dir = tmp_path / "content"
    assets_dir = tmp_path / "assets"
    content_dir.mkdir()
    assets_dir.mkdir()

    raw_text, raw_result = parser._extract_raw_text(str(pdf_path), content_dir)
    figures, tables = parser._extract_assets(raw_result, "sample", assets_dir)

    assert raw_result is not None
    assert "GovBench Poster" in raw_text
    assert len(figures) == 1
    assert len(tables) == 1
    assert Path(figures["1"]["path"]).exists()
    assert Path(tables["1"]["path"]).exists()
    assert figures["1"]["caption"] == "Figure 1. System overview"
    assert tables["1"]["caption"] == "Table 1. Main results"
