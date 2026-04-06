from __future__ import annotations

import asyncio

import pytest

from dataflow_agent.agentroles.paper2any_agents.p2v_subtitle_and_cursor_agent import (
    parse_subtitle_and_cursor_result,
)
from dataflow_agent.workflow.wf_paper2video import (
    _build_fallback_slide_script,
    _clip_slide_script,
)


def test_parse_subtitle_and_cursor_result_accepts_raw_json() -> None:
    result = {
        "raw": '{"subtitle_and_cursor":"This slide explains the method and results."}'
    }
    assert parse_subtitle_and_cursor_result(result) == "This slide explains the method and results."


def test_parse_subtitle_and_cursor_result_rejects_empty_or_error() -> None:
    assert parse_subtitle_and_cursor_result({"subtitle_and_cursor": "   "}) is None
    assert parse_subtitle_and_cursor_result({"error": "502"}) is None


def test_clip_slide_script_limits_and_deduplicates_english() -> None:
    raw_text = "Introduction\nIntroduction\nMethod overview with key modules and outputs.\nResults summary."
    clipped = _clip_slide_script(raw_text, "en")
    assert clipped
    assert clipped.count("Introduction") == 1
    assert len(clipped.split()) <= 45


def test_build_fallback_slide_script_uses_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dataflow_agent.toolkits.multimodaltool.ocr_config.get_ocr_api_credentials",
        lambda: ("https://example.com/v1", "test-key"),
    )

    async def _fake_call_ocr_async(**kwargs):
        return "Overview of the proposed model.\nEncoder and decoder blocks.\nFinal prediction results."

    monkeypatch.setattr(
        "dataflow_agent.toolkits.multimodaltool.req_ocr.call_ocr_async",
        _fake_call_ocr_async,
    )

    script = asyncio.run(_build_fallback_slide_script("/tmp/fake.png", "en", 1))
    assert "Overview of the proposed model" in script


def test_build_fallback_slide_script_uses_placeholder_when_ocr_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dataflow_agent.toolkits.multimodaltool.ocr_config.get_ocr_api_credentials",
        lambda: ("https://example.com/v1", "test-key"),
    )

    async def _raise_call_ocr_async(**kwargs):
        raise RuntimeError("bad gateway")

    monkeypatch.setattr(
        "dataflow_agent.toolkits.multimodaltool.req_ocr.call_ocr_async",
        _raise_call_ocr_async,
    )

    script = asyncio.run(_build_fallback_slide_script("/tmp/fake.png", "zh", 2))
    assert "第 2 页" in script
