from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.schemas import OutlineRefineRequest
from fastapi_app.services.paper2ppt_service import Paper2PPTService


def _build_pagecontent(count: int) -> list[dict]:
    return [
        {
            "title": f"Slide {idx}",
            "layout_description": f"Layout {idx}",
            "key_points": [f"Point {idx}.1", f"Point {idx}.2"],
            "asset_ref": None,
        }
        for idx in range(1, count + 1)
    ]


def test_refine_outline_preserves_page_count_when_chunk_rewrite_returns_short(monkeypatch) -> None:
    service = Paper2PPTService()
    pagecontent = _build_pagecontent(6)

    calls: list[str] = []

    async def fake_invoke_json_llm(**kwargs):
        task_prompt = kwargs["task_prompt"]
        if "结构化编辑计划" in task_prompt:
            calls.append("planner")
            return {
                "global_instruction": "Polish the full deck",
                "apply_global_rewrite": True,
                "operations": [],
            }

        calls.append("rewriter")
        if "第 1 页到第 4 页" in task_prompt:
            # Invalid short response: should fallback to original pages for this chunk.
            return [
                {
                    "title": "Broken rewrite",
                    "layout_description": "Broken layout",
                    "key_points": ["Broken"],
                    "asset_ref": None,
                }
            ]

        return [
            {
                "title": "Slide 5 rewritten",
                "layout_description": "Layout 5 rewritten",
                "key_points": ["Updated 5"],
                "asset_ref": None,
            },
            {
                "title": "Slide 6 rewritten",
                "layout_description": "Layout 6 rewritten",
                "key_points": ["Updated 6"],
                "asset_ref": None,
            },
        ]

    monkeypatch.setattr(service, "_invoke_json_llm", fake_invoke_json_llm)

    result = asyncio.run(
        service.refine_outline(
            OutlineRefineRequest(
                outline_feedback="Polish the entire outline and make it more concise.",
                pagecontent=json.dumps(pagecontent, ensure_ascii=False),
                model="gpt-4o",
                language="en",
            ),
            request=None,
        )
    )

    refined = result["pagecontent"]
    assert len(refined) == 6
    assert refined[0]["title"] == "Slide 1"
    assert refined[3]["title"] == "Slide 4"
    assert refined[4]["title"] == "Slide 5 rewritten"
    assert refined[5]["title"] == "Slide 6 rewritten"
    assert calls[0] == "planner"
    assert calls.count("rewriter") == 2


def test_refine_outline_applies_structured_patch_plan(monkeypatch) -> None:
    service = Paper2PPTService()
    pagecontent = _build_pagecontent(4)

    async def fake_invoke_json_llm(**kwargs):
        task_prompt = kwargs["task_prompt"]
        if "结构化编辑计划" in task_prompt:
            return {
                "global_instruction": "Tighten flow and add one experiment page.",
                "apply_global_rewrite": False,
                "operations": [
                    {
                        "type": "update",
                        "page_numbers": [2],
                        "instruction": "Rewrite page 2 to focus on method contribution.",
                    },
                    {
                        "type": "insert_after",
                        "page_number": 2,
                        "count": 1,
                        "instruction": "Add a new experiment design page after page 2.",
                    },
                    {
                        "type": "delete",
                        "page_numbers": [4],
                    },
                ],
            }

        return [
            {
                "title": "Method Contribution",
                "layout_description": "Updated method-focused layout",
                "key_points": ["Clarify the core contribution", "Tighten method bullets"],
                "asset_ref": None,
            },
            {
                "title": "Experiment Design",
                "layout_description": "New experiment planning page",
                "key_points": ["Dataset split", "Evaluation metrics"],
                "asset_ref": None,
            },
        ]

    monkeypatch.setattr(service, "_invoke_json_llm", fake_invoke_json_llm)

    result = asyncio.run(
        service.refine_outline(
            OutlineRefineRequest(
                outline_feedback="Rewrite page 2, add one experiment page after it, and remove the old ending page.",
                pagecontent=json.dumps(pagecontent, ensure_ascii=False),
                model="gpt-4o",
                language="en",
            ),
            request=None,
        )
    )

    refined = result["pagecontent"]
    assert len(refined) == 4
    assert [page["title"] for page in refined] == [
        "Slide 1",
        "Method Contribution",
        "Experiment Design",
        "Slide 3",
    ]
    assert refined[1]["key_points"] == ["Clarify the core contribution", "Tighten method bullets"]
    assert refined[2]["key_points"] == ["Dataset split", "Evaluation metrics"]
