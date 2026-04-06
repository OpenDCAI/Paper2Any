from __future__ import annotations

from dataflow_agent.agentroles.paper2any_agents.p2v_refine_subtitle_and_cursor_agent import (
    parse_subtitle_and_cursor_result,
)


def test_parse_refine_subtitle_and_cursor_result_accepts_primary_key() -> None:
    result = {
        "refine_subtitle_and_cursor": "Sentence A | title\nSentence B | chart"
    }
    assert parse_subtitle_and_cursor_result(result) == "Sentence A | title\nSentence B | chart"


def test_parse_refine_subtitle_and_cursor_result_accepts_legacy_key() -> None:
    result = {
        "subtitle_and_cursor": "Sentence A | title\nSentence B | chart"
    }
    assert parse_subtitle_and_cursor_result(result) == "Sentence A | title\nSentence B | chart"


def test_parse_refine_subtitle_and_cursor_result_accepts_raw_json() -> None:
    result = {
        "raw": '{"refine_subtitle_and_cursor":"Sentence A | title\\nSentence B | chart"}'
    }
    assert parse_subtitle_and_cursor_result(result) == "Sentence A | title\nSentence B | chart"


def test_parse_refine_subtitle_and_cursor_result_rejects_empty_payload() -> None:
    assert parse_subtitle_and_cursor_result({"refine_subtitle_and_cursor": "  "}) is None
