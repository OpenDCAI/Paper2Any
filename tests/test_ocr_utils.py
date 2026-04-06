from dataflow_agent.toolkits.multimodaltool.ocr_utils import (
    extract_bbox_items,
    normalize_ocr_max_tokens,
)


def test_normalize_ocr_max_tokens_clamps_dashscope_qwen_ocr() -> None:
    assert (
        normalize_ocr_max_tokens(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-vl-ocr-2025-11-20",
            16384,
        )
        == 8192
    )


def test_normalize_ocr_max_tokens_keeps_other_providers() -> None:
    assert normalize_ocr_max_tokens("https://api.example.com/v1", "qwen-vl-ocr-2025-11-20", 16384) == 16384


def test_extract_bbox_items_accepts_wrapped_payloads() -> None:
    items = [{"rotate_rect": [1, 2, 3, 4, 5], "text": "hello"}]
    assert extract_bbox_items(items) == items
    assert extract_bbox_items({"result": items}) == items
    assert extract_bbox_items({"raw": '[{"rotate_rect":[1,2,3,4,5],"text":"hello"}]'}) == items


def test_extract_bbox_items_rejects_error_payloads() -> None:
    assert extract_bbox_items({"error": "boom"}) == []
    assert extract_bbox_items("not-json") == []
