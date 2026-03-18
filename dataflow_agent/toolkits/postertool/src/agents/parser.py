"""
pdf text and asset extraction
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
from jinja2 import Template
from PIL import Image

from dataflow_agent.toolkits.multimodaltool.mineru_tool import (
    _extract_block_text,
    _normalize_mineru_blocks,
    crop_mineru_blocks_with_meta,
    run_aio_batch_two_step_extract,
)
from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import (
    log_agent_error,
    log_agent_info,
    log_agent_success,
    log_agent_warning,
)

POSTERTOOL_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = POSTERTOOL_ROOT / "config" / "prompts"


class Parser:
    def __init__(self):
        self.name = "parser"
        config_data = load_config()
        pdf_processing = config_data.get("pdf_processing", {})
        self.render_dpi = int(os.getenv("POSTER_MINERU_RENDER_DPI", pdf_processing.get("render_dpi", 220)))
        self.mineru_port = int(os.getenv("MINERU_PORT", "8010"))

        self.clean_pattern = re.compile(r"<!--[\s\S]*?-->")
        self.enhanced_abt_prompt = self._load_prompt_template("narrative_abt_extraction.txt")
        self.visual_classification_prompt = self._load_prompt_template("classify_visuals.txt")
        self.title_authors_prompt = self._load_prompt_template("extract_title_authors.txt")
        self.section_extraction_prompt = self._load_prompt_template("extract_structured_sections.txt")

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting foundation building")

        try:
            output_dir = Path(state["output_dir"])
            content_dir = output_dir / "content"
            assets_dir = output_dir / "assets"
            content_dir.mkdir(parents=True, exist_ok=True)
            assets_dir.mkdir(parents=True, exist_ok=True)

            raw_text, raw_result = self._extract_raw_text(state["pdf_path"], content_dir)
            figures, tables = self._extract_assets(raw_result, state["poster_name"], assets_dir)

            title, authors = self._extract_title_authors(raw_text, state["text_model"])

            narrative_content, inp_tok, out_tok = self._generate_narrative_content(
                raw_text,
                state["text_model"],
            )
            state["tokens"].add_text(inp_tok, out_tok)

            classified_visuals, inp_tok2, out_tok2 = self._classify_visual_assets(
                figures,
                tables,
                raw_text,
                state["text_model"],
            )
            state["tokens"].add_text(inp_tok2, out_tok2)

            narrative_content["meta"] = {
                "poster_title": title,
                "authors": authors,
            }

            structured_sections = self._extract_structured_sections(raw_text, state["text_model"])

            self._save_content(narrative_content, "narrative_content.json", content_dir)
            self._save_content(classified_visuals, "classified_visuals.json", content_dir)
            self._save_content(structured_sections, "structured_sections.json", content_dir)
            self._save_raw_text(raw_text, content_dir)

            state["raw_text"] = raw_text
            state["structured_sections"] = structured_sections
            state["narrative_content"] = narrative_content
            state["classified_visuals"] = classified_visuals
            state["images"] = figures
            state["tables"] = tables
            state["current_agent"] = self.name

            log_agent_success(
                self.name,
                f"extracted raw text, {len(figures)} images, and {len(tables)} tables",
            )
            log_agent_success(self.name, f"extracted title: {title}")
            log_agent_success(self.name, "generated enhanced abt narrative")
            log_agent_success(
                self.name,
                "classified visuals: "
                f"key={classified_visuals.get('key_visual', 'none')}, "
                f"problem_ill={len(classified_visuals.get('problem_illustration', []))}, "
                f"method_wf={len(classified_visuals.get('method_workflow', []))}, "
                f"main_res={len(classified_visuals.get('main_results', []))}, "
                f"comp_res={len(classified_visuals.get('comparative_results', []))}, "
                f"support={len(classified_visuals.get('supporting', []))}",
            )

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(str(e))

        return state

    def _load_prompt_template(self, filename: str) -> str:
        return load_prompt(str(PROMPTS_DIR / filename))

    def _extract_raw_text(self, pdf_path: str, content_dir: Path) -> Tuple[str, Dict[str, Any] | None]:
        log_agent_info(
            self.name,
            f"converting pdf to raw text via MinerU HTTP (port={self.mineru_port})",
        )
        try:
            return self._extract_raw_text_with_mineru(pdf_path, content_dir)
        except Exception as exc:
            log_agent_warning(
                self.name,
                f"MinerU extraction failed, falling back to PyMuPDF text extraction: {exc}",
            )
            return self._extract_raw_text_with_pymupdf(pdf_path, content_dir)

    def _extract_raw_text_with_mineru(
        self,
        pdf_path: str,
        content_dir: Path,
    ) -> Tuple[str, Dict[str, Any]]:
        pages_dir = content_dir / "mineru_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        page_image_paths = self._render_pdf_pages(pdf_path, pages_dir)
        if not page_image_paths:
            raise ValueError("failed to render any pdf pages for MinerU extraction")

        raw_page_results = asyncio.run(
            run_aio_batch_two_step_extract(page_image_paths, port=self.mineru_port)
        )
        if isinstance(raw_page_results, dict):
            raw_page_results = [raw_page_results]
        if not isinstance(raw_page_results, list):
            raise RuntimeError(f"unexpected MinerU result type: {type(raw_page_results)}")

        pages: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        for page_index, image_path in enumerate(page_image_paths, start=1):
            raw_page = raw_page_results[page_index - 1] if (page_index - 1) < len(raw_page_results) else []
            blocks = self._sort_blocks_for_reading(_normalize_mineru_blocks(raw_page))
            pages.append(
                {
                    "page_index": page_index,
                    "image_path": image_path,
                    "blocks": blocks,
                }
            )

            page_text = self._blocks_to_markdown(page_index, blocks)
            if page_text:
                text_parts.append(page_text)

        text = self.clean_pattern.sub("", "\n\n".join(text_parts)).strip()
        if not text:
            raise ValueError("MinerU returned no usable text")

        (content_dir / "raw.md").write_text(text, encoding="utf-8")
        log_agent_info(
            self.name,
            f"extracted {len(text)} chars via MinerU across {len(pages)} pages",
        )
        return text, {"pages": pages}

    def _render_pdf_pages(self, pdf_path: str, pages_dir: Path) -> List[str]:
        zoom = max(self.render_dpi / 72.0, 1.0)
        matrix = fitz.Matrix(zoom, zoom)
        page_paths: List[str] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                out_path = pages_dir / f"page_{page_index:04d}.png"
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(out_path))
                page_paths.append(str(out_path.resolve()))

        return page_paths

    def _blocks_to_markdown(self, page_index: int, blocks: List[Dict[str, Any]]) -> str:
        lines: List[str] = [f"## Page {page_index}"]
        for block in blocks:
            block_type = str(block.get("type") or "").lower()
            text = _extract_block_text(block)
            if not text:
                continue

            if block_type in {"title", "heading", "section_title"}:
                lines.append(f"# {text}")
            else:
                lines.append(text)

        return "\n\n".join(line.rstrip() for line in lines if line.strip())

    def _extract_raw_text_with_pymupdf(self, pdf_path: str, content_dir: Path) -> Tuple[str, None]:
        log_agent_warning(
            self.name,
            "using PyMuPDF fallback parser because MinerU is unavailable",
        )

        with fitz.open(pdf_path) as document:
            pages = []
            for page in document:
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append(page_text)

        text = self.clean_pattern.sub("", "\n\n".join(pages)).strip()
        if not text:
            raise ValueError("failed to extract any text from pdf")

        (content_dir / "raw.md").write_text(text, encoding="utf-8")
        log_agent_info(self.name, f"extracted {len(text)} chars via fallback parser")
        return text, None

    def _generate_narrative_content(self, text: str, config) -> Tuple[Dict, int, int]:
        log_agent_info(self.name, "generating abt narrative")
        agent = LangGraphAgent("expert poster design consultant", config)

        for attempt in range(3):
            try:
                prompt = Template(self.enhanced_abt_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)

                narrative = extract_json(response.content)

                if "and" in narrative and "but" in narrative and "therefore" in narrative:
                    return narrative, response.input_tokens, response.output_tokens

            except Exception as e:
                log_agent_warning(self.name, f"attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise

        raise ValueError("failed to generate enhanced narrative after 3 attempts")

    def _save_content(self, content: Dict, filename: str, content_dir: Path):
        with open(content_dir / filename, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    def _save_raw_text(self, raw_text: str, content_dir: Path):
        with open(content_dir / "raw.md", "w", encoding="utf-8") as f:
            f.write(raw_text)

    def _extract_assets(self, result, name: str, assets_dir: Path) -> Tuple[Dict, Dict]:
        log_agent_info(self.name, "extracting assets")

        if not result or not result.get("pages"):
            return self._write_asset_metadata({}, {}, {}, assets_dir)

        figures: Dict[str, Dict[str, Any]] = {}
        tables: Dict[str, Dict[str, Any]] = {}
        caption_map: Dict[str, Dict[str, Any]] = {}
        figure_count = 0
        table_count = 0

        for page in result["pages"]:
            blocks = page["blocks"]
            page_image_path = page["image_path"]
            page_index = page["page_index"]

            for block_index, block in enumerate(blocks):
                asset_kind = self._block_asset_kind(block)
                if asset_kind is None:
                    continue

                cropped = crop_mineru_blocks_with_meta(
                    page_image_path,
                    [block],
                    target_type=None,
                    output_dir=str(assets_dir),
                    prefix=f"{asset_kind}-p{page_index:03d}-b{block_index:04d}-",
                )
                if not cropped:
                    continue

                image_meta = cropped[0]
                png_path = image_meta["png_path"]
                caption = self._infer_asset_caption(blocks, block_index, asset_kind)
                with Image.open(png_path) as image:
                    width, height = image.size

                item = {
                    "caption": caption,
                    "path": png_path,
                    "width": width,
                    "height": height,
                    "aspect": width / height if height > 0 else 1,
                }

                if asset_kind == "table":
                    table_count += 1
                    tables[str(table_count)] = item
                    key = f"table_{table_count}"
                else:
                    figure_count += 1
                    figures[str(figure_count)] = item
                    key = f"figure_{figure_count}"

                caption_map[key] = {
                    "page": page_index,
                    "block_type": block.get("type"),
                    "bbox": block.get("bbox"),
                    "captions": [caption],
                }

        return self._write_asset_metadata(figures, tables, caption_map, assets_dir)

    def _block_asset_kind(self, block: Dict[str, Any]) -> str | None:
        block_type = str(block.get("type") or "").lower().strip()
        if block_type in {"table"}:
            return "table"
        if block_type in {"figure", "image", "img", "picture"}:
            return "figure"
        return None

    def _infer_asset_caption(
        self,
        blocks: List[Dict[str, Any]],
        block_index: int,
        asset_kind: str,
    ) -> str:
        keywords = ("table",) if asset_kind == "table" else ("figure", "fig.", "image")
        candidates: List[Tuple[int, str]] = []

        for distance in (1, 2, 3, -1, -2, -3):
            idx = block_index + distance
            if idx < 0 or idx >= len(blocks):
                continue

            text = _extract_block_text(blocks[idx])
            if not text:
                continue

            lowered = text.lower()
            score = abs(distance)
            if any(keyword in lowered for keyword in keywords):
                score -= 10
            candidates.append((score, text.strip()))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1][:300]

        label = "Table" if asset_kind == "table" else "Figure"
        return f"{label} {block_index + 1}"

    def _write_asset_metadata(
        self,
        figures: Dict,
        tables: Dict,
        caption_map: Dict,
        assets_dir: Path,
    ) -> Tuple[Dict, Dict]:
        with open(assets_dir / "figures.json", "w", encoding="utf-8") as f:
            json.dump(figures, f, indent=2)
        with open(assets_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(tables, f, indent=2)
        with open(assets_dir / "fig_tab_caption_mapping.json", "w", encoding="utf-8") as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)

        return figures, tables

    def _sort_blocks_for_reading(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _key(block: Dict[str, Any]) -> tuple[float, float]:
            bbox = block.get("bbox") or [1e9, 1e9, 1e9, 1e9]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return (1e9, 1e9)
            try:
                return (float(bbox[1]), float(bbox[0]))
            except Exception:
                return (1e9, 1e9)

        return sorted(blocks, key=_key)

    def _extract_title_authors(self, text: str, config) -> Tuple[str, str]:
        """extract title and authors via llm api"""
        log_agent_info(self.name, "extracting title and authors with llm")
        agent = LangGraphAgent("expert academic paper parser", config)

        for attempt in range(3):
            try:
                prompt = Template(self.title_authors_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)

                result = extract_json(response.content)

                if "title" in result and "authors" in result:
                    title = result["title"].strip()
                    authors = result["authors"].strip()

                    if title and authors:
                        return title, authors

            except Exception as e:
                log_agent_warning(self.name, f"title/authors extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return "Untitled", "Authors not found"

        return "Untitled", "Authors not found"

    def _classify_visual_assets(self, figures: Dict, tables: Dict, raw_text: str, config) -> Tuple[Dict, int, int]:
        all_visuals = []
        for fig_id, fig_data in figures.items():
            all_visuals.append(
                {
                    "id": f"figure_{fig_id}",
                    "type": "figure",
                    "caption": fig_data.get("caption", ""),
                    "aspect_ratio": fig_data.get("aspect", 1.0),
                }
            )

        for tab_id, tab_data in tables.items():
            all_visuals.append(
                {
                    "id": f"table_{tab_id}",
                    "type": "table",
                    "caption": tab_data.get("caption", ""),
                    "aspect_ratio": tab_data.get("aspect", 1.0),
                }
            )

        if not all_visuals:
            return {
                "key_visual": None,
                "problem_illustration": [],
                "method_workflow": [],
                "main_results": [],
                "comparative_results": [],
                "supporting": [],
            }, 0, 0

        log_agent_info(self.name, f"classifying {len(all_visuals)} visual assets")
        agent = LangGraphAgent("expert poster designer", config)

        for attempt in range(3):
            try:
                prompt = Template(self.visual_classification_prompt).render(
                    visuals_list=json.dumps(all_visuals, indent=2)
                )

                agent.reset()
                response = agent.step(prompt)
                classification = extract_json(response.content)

                required_keys = [
                    "key_visual",
                    "problem_illustration",
                    "method_workflow",
                    "main_results",
                    "comparative_results",
                    "supporting",
                ]
                if all(key in classification for key in required_keys):
                    return classification, response.input_tokens, response.output_tokens

            except Exception as e:
                log_agent_warning(self.name, f"visual classification attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return self._fallback_visual_classification(all_visuals), 0, 0

        return self._fallback_visual_classification(all_visuals), 0, 0

    def _fallback_visual_classification(self, visuals):
        classification = {
            "key_visual": None,
            "problem_illustration": [],
            "method_workflow": [],
            "main_results": [],
            "comparative_results": [],
            "supporting": [],
        }

        for visual in visuals:
            caption = visual.get("caption", "").lower()
            if "comparison" in caption:
                classification["comparative_results"].append(visual["id"])
            elif "result" in caption or "performance" in caption:
                classification["main_results"].append(visual["id"])
            elif "method" in caption or "architecture" in caption or "framework" in caption:
                classification["method_workflow"].append(visual["id"])
            elif "problem" in caption or "motivation" in caption:
                classification["problem_illustration"].append(visual["id"])
            else:
                classification["supporting"].append(visual["id"])

        if classification["main_results"]:
            classification["key_visual"] = classification["main_results"][0]
        elif classification["method_workflow"]:
            classification["key_visual"] = classification["method_workflow"][0]
        elif classification["comparative_results"]:
            classification["key_visual"] = classification["comparative_results"][0]

        return classification

    def _extract_structured_sections(self, raw_text: str, config) -> Dict:
        """extract structured sections from raw paper text"""

        log_agent_info(self.name, "extracting structured sections from paper")
        agent = LangGraphAgent("expert paper section extractor", config)

        for attempt in range(3):
            try:
                prompt = Template(self.section_extraction_prompt).render(raw_text=raw_text)
                agent.reset()
                response = agent.step(prompt)

                structured_sections = extract_json(response.content)

                if self._validate_structured_sections(structured_sections):
                    log_agent_success(
                        self.name,
                        f"extracted {len(structured_sections.get('paper_sections', []))} structured sections",
                    )
                    return structured_sections
                log_agent_warning(self.name, f"attempt {attempt + 1}: invalid structured sections")

            except Exception as e:
                log_agent_warning(self.name, f"section extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise ValueError("failed to extract structured sections after multiple attempts")

        return {
            "paper_sections": [],
            "paper_structure": {
                "total_sections": 0,
                "foundation_sections": 0,
                "method_sections": 0,
                "evaluation_sections": 0,
                "conclusion_sections": 0,
            },
        }

    def _validate_structured_sections(self, structured_sections: Dict) -> bool:
        """validate structured sections format"""
        if "paper_sections" not in structured_sections:
            log_agent_warning(self.name, "validation error: missing 'paper_sections'")
            return False

        sections = structured_sections["paper_sections"]
        if not isinstance(sections, list) or len(sections) < 3:
            log_agent_warning(self.name, f"validation error: need at least 3 sections, got {len(sections)}")
            return False

        for i, section in enumerate(sections):
            required_fields = ["section_name", "section_type", "content"]
            for field in required_fields:
                if field not in section:
                    log_agent_warning(self.name, f"validation error: section {i} missing '{field}'")
                    return False

        return True


def parser_node(state: PosterState) -> PosterState:
    return Parser()(state)
