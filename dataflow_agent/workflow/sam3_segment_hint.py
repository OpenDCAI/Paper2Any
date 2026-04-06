from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataflow_agent.agentroles import create_vlm_agent
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.ocr_config import get_ocr_api_credentials
from dataflow_agent.utils_common import robust_parse_json

log = get_logger(__name__)

DEFAULT_SEGMENT_HINT_MAX_PROMPTS = 8
DEFAULT_SEGMENT_HINT_MAX_WORDS = 5
DEFAULT_SEGMENT_HINT_MAX_CHARS = 40
DEFAULT_SEGMENT_HINT_MAX_TEXT_LINES = 30
DEFAULT_SEGMENT_HINT_MAX_TEXT_CHARS = 1200
DEFAULT_SEGMENT_HINT_FALLBACK_MODEL = "qwen-vl-ocr-2025-11-20"

DEFAULT_SEGMENT_HINT_ABSTRACT_TOKENS = {
    "input",
    "output",
    "context",
    "intent",
    "description",
    "reference",
    "verification",
    "guideline",
    "guidelines",
    "phase",
    "loop",
    "round",
    "rounds",
}

DEFAULT_SEGMENT_HINT_BLOCKLIST = {
    "text",
    "word",
    "words",
    "letter",
    "letters",
    "label",
    "labels",
    "caption",
    "captions",
    "input",
    "output",
    "context",
    "intent",
    "description",
    "reference",
    "reference set",
    "phase",
    "loop",
    "verification",
    "guideline",
    "guidelines",
    "round",
    "rounds",
    "shape",
    "shapes",
    "box",
    "boxes",
    "rectangle",
    "rounded rectangle",
    "diamond",
    "ellipse",
    "circle",
    "triangle",
    "hexagon",
    "arrow",
    "line",
    "connector",
    "background",
    "panel",
    "container",
    "filled region",
    "image",
    "images",
    "icon",
    "icons",
    "symbol",
    "symbols",
    "pictogram",
    "glyph",
    "picture",
    "logo",
    "chart",
    "plot",
    "graph",
    "diagram",
    "object",
    "illustration",
    "device",
    "character",
    "avatar",
    "mascot",
    "screenshot",
    "cell",
    "molecule",
    "complex",
    "receptor",
    "particle",
    "node",
    "blob",
}


def normalize_prompt(prompt: str) -> str:
    return (prompt or "").strip().lower()


def dedupe_prompts(prompts: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized = normalize_prompt(prompt)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def collect_segment_hint_text_lines(
    text_blocks: Sequence[Dict[str, Any]],
    *,
    max_text_lines: int = DEFAULT_SEGMENT_HINT_MAX_TEXT_LINES,
    max_text_chars: int = DEFAULT_SEGMENT_HINT_MAX_TEXT_CHARS,
) -> List[str]:
    lines: List[str] = []
    total_chars = 0
    for block in text_blocks:
        text = re.sub(r"\s+", " ", (block.get("text") or "").strip())
        if not text or text in lines:
            continue
        next_total = total_chars + len(text)
        if next_total > max_text_chars and lines:
            break
        lines.append(text)
        total_chars = next_total
        if len(lines) >= max_text_lines:
            break
    return lines


def _resolve_segment_hint_ocr_config(
    state: Any,
    *,
    env_prefix: str,
    fallback_model: str = DEFAULT_SEGMENT_HINT_FALLBACK_MODEL,
    fallback_model_env_var: str = "MODEL_QWEN_VL_OCR",
) -> Tuple[str, str, str, str]:
    request = getattr(state, "request", None)
    request_vlm_model = (getattr(request, "vlm_model", "") or "").strip()
    explicit_model = (os.getenv(f"{env_prefix}_VLM_MODEL") or "").strip()
    ocr_api_url, ocr_api_key = get_ocr_api_credentials()
    ocr_model = (os.getenv(fallback_model_env_var) or "").strip()
    for candidate in (explicit_model, request_vlm_model, ocr_model, fallback_model):
        if candidate and "qwen-vl-ocr" in candidate.lower():
            return ocr_api_url, ocr_api_key, candidate, "ocr"
    return ocr_api_url, ocr_api_key, fallback_model, "ocr"


def _resolve_segment_hint_vlm_config(
    state: Any,
    *,
    env_prefix: str,
    fallback_model: str = DEFAULT_SEGMENT_HINT_FALLBACK_MODEL,
    generic_api_url_env_var: str = "DF_API_URL",
    generic_api_key_env_var: str = "DF_API_KEY",
) -> Tuple[str, str, str, str]:
    request = getattr(state, "request", None)
    request_api_url = (getattr(request, "chat_api_url", "") or "").strip()
    request_api_key = (getattr(request, "api_key", "") or "").strip()
    request_vlm_model = (getattr(request, "vlm_model", "") or "").strip()
    request_model = (getattr(request, "model", "") or "").strip()

    explicit_api_url = (os.getenv(f"{env_prefix}_API_URL") or "").strip()
    explicit_api_key = (os.getenv(f"{env_prefix}_API_KEY") or "").strip()
    explicit_model = (os.getenv(f"{env_prefix}_VLM_MODEL") or "").strip()
    generic_api_url = (os.getenv(generic_api_url_env_var) or "").strip()
    generic_api_key = (os.getenv(generic_api_key_env_var) or "").strip()

    def _select_understanding_model() -> str:
        if explicit_model:
            return explicit_model
        if request_vlm_model and "qwen-vl-ocr" not in request_vlm_model.lower():
            return request_vlm_model
        return request_model or request_vlm_model or "gpt-4o"

    if explicit_api_url and explicit_api_key:
        return explicit_api_url, explicit_api_key, _select_understanding_model(), "understanding"

    if request_api_url and request_api_key and "dashscope.aliyuncs.com" not in request_api_url.lower():
        return request_api_url, request_api_key, _select_understanding_model(), "understanding"

    if generic_api_url and generic_api_key:
        return generic_api_url, generic_api_key, _select_understanding_model(), "understanding"

    return _resolve_segment_hint_ocr_config(
        state,
        env_prefix=env_prefix,
        fallback_model=fallback_model,
    )


def _extract_segment_hint_candidates(raw_result: Any) -> List[Any]:
    if isinstance(raw_result, dict):
        for key in ("extra_image_prompts", "image_prompts", "prompts", "items"):
            value = raw_result.get(key)
            if isinstance(value, list):
                return value
        raw_value = raw_result.get("raw")
        if isinstance(raw_value, str):
            try:
                return _extract_segment_hint_candidates(robust_parse_json(raw_value))
            except Exception:
                return [item.strip("- ").strip() for item in raw_value.splitlines() if item.strip()]
        return []

    if isinstance(raw_result, str):
        try:
            return _extract_segment_hint_candidates(robust_parse_json(raw_result))
        except Exception:
            return [item.strip("- ").strip() for item in raw_result.splitlines() if item.strip()]

    if isinstance(raw_result, list):
        return raw_result

    return []


def normalize_segment_hint_prompts(
    raw_result: Any,
    *,
    text_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    blocked_prompts: Optional[Sequence[str]] = None,
    max_prompts: int = DEFAULT_SEGMENT_HINT_MAX_PROMPTS,
    max_words: int = DEFAULT_SEGMENT_HINT_MAX_WORDS,
    max_chars: int = DEFAULT_SEGMENT_HINT_MAX_CHARS,
    abstract_tokens: Optional[set[str]] = None,
) -> List[str]:
    blocked = {
        normalize_prompt(prompt)
        for prompt in (blocked_prompts or DEFAULT_SEGMENT_HINT_BLOCKLIST)
    }
    ocr_label_prompts = {
        normalize_prompt(text)
        for text in collect_segment_hint_text_lines(text_blocks or [])
    }
    abstract_tokens = abstract_tokens or DEFAULT_SEGMENT_HINT_ABSTRACT_TOKENS

    normalized_prompts: List[str] = []
    seen: set[str] = set()
    for candidate in _extract_segment_hint_candidates(raw_result):
        if not isinstance(candidate, str):
            continue
        prompt = normalize_prompt(candidate)
        prompt = prompt.replace("_", " ")
        prompt = re.sub(r"[\"'`“”‘’]", "", prompt)
        prompt = re.sub(r"\s+", " ", prompt).strip(" .,:;!?/-")
        if not prompt:
            continue
        if prompt in blocked or prompt in seen:
            continue
        if prompt in ocr_label_prompts:
            continue
        if len(prompt.split()) >= 2 and any(prompt in label for label in ocr_label_prompts):
            continue
        if set(prompt.split()) & abstract_tokens:
            continue
        if len(prompt) > max_chars:
            continue
        if len(prompt.split()) > max_words:
            continue
        if any(ch in prompt for ch in "{}[]<>|()"):
            continue
        seen.add(prompt)
        normalized_prompts.append(prompt)
        if len(normalized_prompts) >= max_prompts:
            break
    return normalized_prompts


def _resolve_timeout(
    env_name: str,
    *,
    default_env_name: str = "VLM_OCR_TIMEOUT",
    default_timeout: int = 120,
) -> int:
    try:
        return int(os.getenv(env_name, os.getenv(default_env_name, str(default_timeout))))
    except ValueError:
        return default_timeout


async def generate_sam3_segment_hints(
    *,
    state: Any,
    image_path: str,
    text_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    env_prefix: str,
    base_image_prompts: Sequence[str],
    base_recall_prompts: Optional[Sequence[str]] = None,
    extra_blocked_prompts: Optional[Sequence[str]] = None,
    timeout_env_name: Optional[str] = None,
    fallback_model: str = DEFAULT_SEGMENT_HINT_FALLBACK_MODEL,
    log_prefix: str = "[sam3_segment_hint]",
) -> Tuple[List[str], Any, str, str]:
    if not image_path or not os.path.exists(image_path):
        return [], {}, "", ""

    text_blocks = list(text_blocks or [])
    base_recall_prompts = list(base_recall_prompts or [])
    blocked_prompts = list(DEFAULT_SEGMENT_HINT_BLOCKLIST)
    blocked_prompts.extend(base_image_prompts)
    blocked_prompts.extend(base_recall_prompts)
    blocked_prompts.extend(extra_blocked_prompts or [])
    text_lines = collect_segment_hint_text_lines(text_blocks)

    primary_config = _resolve_segment_hint_vlm_config(
        state,
        env_prefix=env_prefix,
        fallback_model=fallback_model,
    )
    attempt_configs = [primary_config]
    ocr_fallback_config = _resolve_segment_hint_ocr_config(
        state,
        env_prefix=env_prefix,
        fallback_model=fallback_model,
    )
    if ocr_fallback_config not in attempt_configs:
        attempt_configs.append(ocr_fallback_config)

    timeout_seconds = _resolve_timeout(timeout_env_name or f"{env_prefix}_TIMEOUT")

    raw_result: Any = {}
    hints: List[str] = []
    model_name = ""
    vlm_mode = ""
    for api_url, api_key, current_model_name, current_vlm_mode in attempt_configs:
        model_name = current_model_name
        vlm_mode = current_vlm_mode
        try:
            temp_state = copy.copy(state)
            if getattr(temp_state, "request", None):
                temp_state.request = copy.copy(state.request)
                temp_state.request.chat_api_url = api_url
                temp_state.request.api_key = api_key
                temp_state.request.chat_api_key = api_key
            temp_state.temp_data = copy.copy(getattr(state, "temp_data", {}) or {})
            temp_state.temp_data["diagram_segment_hint_text_lines_json"] = json.dumps(
                text_lines,
                ensure_ascii=False,
                indent=2,
            )
            temp_state.temp_data["diagram_segment_hint_base_prompts_json"] = json.dumps(
                dedupe_prompts(list(base_image_prompts) + list(base_recall_prompts)),
                ensure_ascii=False,
                indent=2,
            )

            agent = create_vlm_agent(
                name="diagram_segment_hint_agent",
                model_name=model_name,
                chat_api_url=api_url,
                max_tokens=2048,
                temperature=0.0,
                vlm_mode=vlm_mode,
                parser_type="json",
                parser_config={
                    "schema": {
                        "extra_image_prompts": ["string"],
                        "excluded_types": ["string"],
                    },
                    "required_fields": ["extra_image_prompts"],
                },
                additional_params={"input_image": image_path, "timeout": timeout_seconds},
            )
            new_state = await agent.execute(temp_state)
            raw_result = new_state.agent_results.get("diagram_segment_hint_agent", {}).get("results", {})
        except Exception as e:
            log.warning(
                f"{log_prefix} VLM inference failed for model={model_name} mode={vlm_mode}: {e}"
            )
            raw_result = {"error": str(e)}

        hints = normalize_segment_hint_prompts(
            raw_result,
            text_blocks=text_blocks,
            blocked_prompts=blocked_prompts,
        )
        if hints or current_vlm_mode == "ocr":
            break

    log.info(
        f"{log_prefix} model={model_name} mode={vlm_mode} "
        f"hints={json.dumps(hints, ensure_ascii=False)} image={image_path}"
    )
    return hints, raw_result, model_name, vlm_mode
