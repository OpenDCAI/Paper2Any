from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from dataflow_agent.workflow import run_workflow
import dataflow_agent.workflow.wf_paper2figure_research_image as wf_mod


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nK7cAAAAASUVORK5CYII="
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


@pytest.mark.asyncio
async def test_paper2fig_research_image_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    decisions = iter(["research_more", "local_edit", "accept"])

    async def fake_build_search_plan(state, requirement):
        return {
            "intent": "draw a method figure",
            "search_queries": ["multimodal diagram generation"],
            "keywords": ["multimodal", "diagram"],
            "visual_focus": ["pipeline", "modules"],
        }

    async def fake_search_arxiv_papers(query: str, max_results: int):
        return [
            {
                "arxiv_id": "2501.00001",
                "title": "First Relevant Paper",
                "abstract": "First abstract",
                "authors": ["A"],
                "published": "2025-01-01",
                "pdf_url": "https://example.com/1.pdf",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
            {
                "arxiv_id": "2501.00002",
                "title": "Second Relevant Paper",
                "abstract": "Second abstract",
                "authors": ["B"],
                "published": "2025-01-02",
                "pdf_url": "https://example.com/2.pdf",
                "abs_url": "https://arxiv.org/abs/2501.00002",
            },
        ][:max_results]

    async def fake_download_pdf_file(paper, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(PDF_BYTES)
        return save_path

    async def fake_parse_pdf_with_mineru(pdf_path: Path, output_dir: Path, port: int):
        auto_dir = output_dir / pdf_path.stem / "auto"
        images_dir = auto_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        fig_path = images_dir / "page_0001_blk_0001.png"
        fig_path.write_bytes(PNG_BYTES)
        md_path = auto_dir / f"{pdf_path.stem}.md"
        md_path.write_text("# Fake Paper\n\nMethod and figure description.", encoding="utf-8")
        return {
            "markdown_text": md_path.read_text(encoding="utf-8"),
            "auto_dir": str(auto_dir),
            "markdown_path": str(md_path),
            "figure_paths": [str(fig_path)],
        }

    async def fake_summarize_paper_content(state, requirement, paper, markdown_text):
        return {
            "paper_id": paper["paper_id"],
            "idea": f"idea for {paper['paper_id']}",
            "method": f"method for {paper['paper_id']}",
            "visual_takeaways": ["encoder-decoder layout", "cross-modal fusion"],
            "relevance": "high",
        }

    async def fake_describe_figure_image(state, requirement, figure_id, image_path, paper_title):
        return {
            "figure_id": figure_id,
            "summary": f"summary for {figure_id}",
            "layout": "left-to-right pipeline",
            "key_elements": ["input", "encoder", "decoder"],
            "style_notes": ["flat color", "rounded boxes"],
            "reference_value": "useful for layout",
            "reuse_for": ["layout", "content"],
            "image_path": image_path,
        }

    async def fake_analyze_reference_bundle(state, requirement, paper_contexts):
        figure_ids = [
            fig["figure_id"]
            for paper in paper_contexts
            for fig in paper.get("figure_analyses", [])
        ]
        return {
            "reference_summary": f"using {len(paper_contexts)} papers",
            "paper_rankings": [
                {"paper_id": paper["paper_id"], "score": 90 - idx, "reason": "relevant"}
                for idx, paper in enumerate(paper_contexts)
            ],
            "figure_rankings": [
                {"figure_id": fig_id, "score": 88 - idx, "reason": "good layout", "reuse_for": ["layout"]}
                for idx, fig_id in enumerate(figure_ids)
            ],
            "recommended_reference_images": figure_ids[:2],
            "gaps": [],
        }

    async def fake_compose_generation_plan(state, requirement, paper_contexts, reference_analysis, previous_feedback, round_index):
        return {
            "final_prompt": f"round {round_index} generate scientific figure",
            "edit_prompt": f"round {round_index} refine local details",
            "negative_prompt": "",
            "reference_image_ids": reference_analysis.get("recommended_reference_images", [])[:1],
            "prompt_strategy": "reuse layout and module names",
        }

    async def fake_generate_or_edit_figure(state, prompt, save_path: Path, reference_images, previous_image_path, use_edit):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(PNG_BYTES)
        return str(save_path)

    async def fake_critic_generated_figure(state, requirement, image_path, prompt_payload, reference_analysis):
        decision = next(decisions)
        return {
            "decision": decision,
            "score": 92 if decision == "accept" else 65,
            "summary": f"critic says {decision}",
            "major_issues": ["need more references"] if decision == "research_more" else [],
            "minor_issues": ["tune alignment"] if decision == "local_edit" else [],
            "edit_prompt": "tighten spacing and labels" if decision == "local_edit" else "",
            "search_focus": ["more multimodal pipeline figures"] if decision == "research_more" else [],
        }

    monkeypatch.setattr(wf_mod, "_build_search_plan", fake_build_search_plan)
    monkeypatch.setattr(wf_mod, "_search_arxiv_papers", fake_search_arxiv_papers)
    monkeypatch.setattr(wf_mod, "_download_pdf_file", fake_download_pdf_file)
    monkeypatch.setattr(wf_mod, "_parse_pdf_with_mineru", fake_parse_pdf_with_mineru)
    monkeypatch.setattr(wf_mod, "_summarize_paper_content", fake_summarize_paper_content)
    monkeypatch.setattr(wf_mod, "_describe_figure_image", fake_describe_figure_image)
    monkeypatch.setattr(wf_mod, "_analyze_reference_bundle", fake_analyze_reference_bundle)
    monkeypatch.setattr(wf_mod, "_compose_generation_plan", fake_compose_generation_plan)
    monkeypatch.setattr(wf_mod, "_generate_or_edit_figure", fake_generate_or_edit_figure)
    monkeypatch.setattr(wf_mod, "_critic_generated_figure", fake_critic_generated_figure)

    output_dir = tmp_path / "workflow_output"
    request = Paper2FigureRequest(
        chat_api_url="http://127.0.0.1:3000/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        target="draw a scientific figure",
        input_type="TEXT",
        input_content="Please draw a multimodal research pipeline figure with training and inference stages.",
        gen_fig_model="gemini-3.1-flash-image-preview",
        vlm_model="gpt-4o-mini",
    )
    request.image_api_url = "https://api.apiyi.com/v1"
    request.image_api_key = "image-key"
    request.initial_reference_papers = 1
    request.max_reference_papers = 2
    request.max_rounds = 3
    request.max_figures_per_paper = 1
    request.max_reference_images = 2
    request.search_top_k = 2

    state = Paper2FigureState(request=request, result_path=str(output_dir), messages=[])

    final_state = await run_workflow("paper2fig_research_image", state)

    def pick(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    assert final_state is not None
    assert pick(final_state, "fig_draft_path")
    assert Path(pick(final_state, "fig_draft_path")).exists()
    assert Path(pick(final_state, "result_path")) == output_dir.resolve()

    result = pick(final_state, "agent_results")["paper2fig_research_image"]["results"]
    assert result["accepted"] is True
    assert len(result["rounds"]) == 3
    assert result["active_papers"] == ["2501.00001", "2501.00002"]

    assert (output_dir / "input" / "request.json").exists()
    assert (output_dir / "search" / "query_plan.json").exists()
    assert (output_dir / "analysis" / "reference_analysis.json").exists()
    assert (output_dir / "analysis" / "reference_analysis_round_01.json").exists()
    assert (output_dir / "papers").exists()
    assert any((output_dir / "papers").rglob("figure_001.json"))
    assert (output_dir / "generation" / "round_03" / "generated.png").exists()
    assert (output_dir / "final" / "result.json").exists()
