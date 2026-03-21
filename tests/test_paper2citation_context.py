from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.toolkits.citationtool.context_locator import (
    _extract_blocks_from_html,
    _locate_contexts,
    _match_reference_entry,
)


def test_match_reference_entry_prefers_doi() -> None:
    reference_blocks = [
        {"kind": "text", "text": "[11] Someone Else. Another paper. 2023."},
        {
            "kind": "text",
            "text": "[12] Bin Cui, Wentao Zhang. SQLGovernor: An LLM-powered SQL Toolkit for Real World Application. 2025. doi:10.48550/arxiv.2509.08575",
        },
    ]

    result = _match_reference_entry(
        reference_blocks,
        target_doi="10.48550/arxiv.2509.08575",
        target_title="SQLGovernor: An LLM-powered SQL Toolkit for Real World Application",
        target_authors=["Bin Cui", "Wentao Zhang"],
        target_year=2025,
    )

    assert result["matched"] is True
    assert result["matched_by"] == "doi"
    assert result["marker"] == "[12]"


def test_locate_contexts_finds_numeric_inline_citation() -> None:
    body_blocks = [
        {"kind": "heading", "text": "Related Work"},
        {
            "kind": "text",
            "text": "Recent SQL agent systems build on prior toolkits [12]. We follow their execution framing in our prototype.",
        },
        {"kind": "heading", "text": "Experiments"},
        {
            "kind": "text",
            "text": "We compare against a stronger baseline in the final evaluation.",
        },
    ]
    reference_entry = {
        "matched": True,
        "marker": "[12]",
        "matched_by": "doi",
        "reference_text": "[12] Bin Cui, Wentao Zhang. SQLGovernor...",
        "score": 12,
    }

    contexts = _locate_contexts(
        body_blocks,
        reference_entry=reference_entry,
        target_authors=["Bin Cui", "Wentao Zhang"],
        target_year=2025,
    )

    assert len(contexts) == 1
    assert contexts[0]["section"] == "Related Work"
    assert "[12]" in contexts[0]["sentence"]


def test_extract_blocks_from_html_preserves_headings_and_paragraphs() -> None:
    html = """
    <html><body>
      <h2>References</h2>
      <p>[1] Example reference</p>
      <h2>Discussion</h2>
      <p>This paragraph cites prior work.</p>
    </body></html>
    """

    blocks = _extract_blocks_from_html(html)
    texts = [item["text"] for item in blocks]

    assert "References" in texts
    assert "[1] Example reference" in texts
    assert "Discussion" in texts
