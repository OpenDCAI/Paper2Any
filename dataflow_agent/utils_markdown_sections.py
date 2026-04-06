from __future__ import annotations

import math
import re
from typing import Any, Dict, List


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def is_probably_english(text: str | Any) -> bool:
    if not text:
        return False
    if not isinstance(text, str):
        text = str(text)
    sample = text[:5000]
    if not sample:
        return False
    ascii_count = sum(1 for ch in sample if ord(ch) < 128)
    return (ascii_count / len(sample)) > 0.8


def estimate_text_tokens(text: str | Any) -> int:
    """
    Conservative heuristic token estimator.

    - English-ish ASCII text: ~ 4 chars / token
    - CJK chars: ~ 1.2 tokens / char
    - Other unicode: ~ 1 token / char
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)

    ascii_count = 0
    cjk_count = 0
    other_count = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_count += 1
        elif _CJK_RE.match(ch):
            cjk_count += 1
        else:
            other_count += 1

    tokens = (ascii_count / 4.0) + (cjk_count * 1.2) + other_count
    return max(1, math.ceil(tokens))


def get_safe_outline_input_budget(model_name: str | None) -> int:
    """
    Return a conservative input-token budget for outline generation.

    This is intentionally lower than the advertised model context to leave
    room for prompts, schemas, and large JSON outputs.
    """
    name = (model_name or "").strip().lower()
    if not name:
        return 24_000

    long_context_markers = (
        "gpt-5",
        "gpt-4.1",
        "claude",
        "gemini",
        "o1",
        "o3",
        "deepseek",
        "qwen",
    )
    if any(marker in name for marker in long_context_markers):
        return 60_000
    return 24_000


def split_plain_text_to_sections(
    text: str,
    fallback_section_tokens: int = 6_000,
) -> List[Dict[str, Any]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        clean = (text or "").strip()
        return [{
            "title": "Document",
            "heading_path": ["Document"],
            "content": clean,
            "order_index": 0,
            "estimated_tokens": estimate_text_tokens(clean),
        }] if clean else []

    sections: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = estimate_text_tokens(para)
        if current_parts and current_tokens + para_tokens > fallback_section_tokens:
            content = "\n\n".join(current_parts).strip()
            idx = len(sections)
            sections.append({
                "title": f"Document Part {idx + 1}",
                "heading_path": [f"Document Part {idx + 1}"],
                "content": content,
                "order_index": idx,
                "estimated_tokens": estimate_text_tokens(content),
            })
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        content = "\n\n".join(current_parts).strip()
        idx = len(sections)
        sections.append({
            "title": f"Document Part {idx + 1}",
            "heading_path": [f"Document Part {idx + 1}"],
            "content": content,
            "order_index": idx,
            "estimated_tokens": estimate_text_tokens(content),
        })

    return sections


def extract_markdown_sections(
    markdown_text: str,
    *,
    max_heading_level: int = 3,
    fallback_section_tokens: int = 6_000,
) -> List[Dict[str, Any]]:
    """
    Parse MinerU markdown into ordered sections.

    Preferred split signal is markdown headings (#/##/###).
    If no usable headings are found, fall back to paragraph-based sections.
    """
    md = (markdown_text or "").strip()
    if not md:
        return []

    lines = md.splitlines(keepends=True)
    sections: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_title = ""
    current_path: List[str] = []
    heading_stack: List[str] = []
    preamble_lines: List[str] = []

    def _flush_current() -> None:
        nonlocal current_lines, current_title, current_path
        content = "".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        sections.append({
            "title": current_title or "Untitled Section",
            "heading_path": list(current_path) if current_path else [current_title or "Untitled Section"],
            "content": content,
            "order_index": len(sections),
            "estimated_tokens": estimate_text_tokens(content),
        })
        current_lines = []

    for line in lines:
        match = _HEADING_RE.match(line.strip("\n"))
        if not match:
            if current_title:
                current_lines.append(line)
            else:
                preamble_lines.append(line)
            continue

        level = len(match.group(1))
        title = match.group(2).strip()
        if level > max_heading_level:
            if current_title:
                current_lines.append(line)
            else:
                preamble_lines.append(line)
            continue

        if current_title:
            _flush_current()

        heading_stack = heading_stack[: max(0, level - 1)]
        heading_stack.append(title)
        current_title = title
        current_path = list(heading_stack)
        current_lines = [line]

    if current_title:
        _flush_current()

    preamble = "".join(preamble_lines).strip()
    if preamble:
        sections.insert(0, {
            "title": "Introduction",
            "heading_path": ["Introduction"],
            "content": preamble,
            "order_index": 0,
            "estimated_tokens": estimate_text_tokens(preamble),
        })

    if not sections:
        return split_plain_text_to_sections(md, fallback_section_tokens=fallback_section_tokens)

    for idx, section in enumerate(sections):
        section["order_index"] = idx
        section["estimated_tokens"] = estimate_text_tokens(section.get("content", ""))

    return sections


def split_large_section(
    section: Dict[str, Any],
    *,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    content = (section.get("content") or "").strip()
    if not content:
        return []

    if estimate_text_tokens(content) <= max_tokens:
        out = dict(section)
        out["estimated_tokens"] = estimate_text_tokens(content)
        return [out]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if len(paragraphs) <= 1:
        chunk_chars = max(3_000, int(max_tokens * (4 if is_probably_english(content) else 1.5)))
        paragraphs = [content[i:i + chunk_chars] for i in range(0, len(content), chunk_chars)]

    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_tokens = 0
    base_title = section.get("title") or "Section"
    heading_path = list(section.get("heading_path") or [base_title])

    for para in paragraphs:
        para_tokens = estimate_text_tokens(para)
        if current_parts and current_tokens + para_tokens > max_tokens:
            idx = len(chunks) + 1
            chunk_content = "\n\n".join(current_parts).strip()
            chunks.append({
                "title": f"{base_title} (Part {idx})",
                "heading_path": heading_path,
                "content": chunk_content,
                "order_index": section.get("order_index", 0),
                "estimated_tokens": estimate_text_tokens(chunk_content),
            })
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        idx = len(chunks) + 1
        chunk_content = "\n\n".join(current_parts).strip()
        chunks.append({
            "title": f"{base_title} (Part {idx})" if len(chunks) else base_title,
            "heading_path": heading_path,
            "content": chunk_content,
            "order_index": section.get("order_index", 0),
            "estimated_tokens": estimate_text_tokens(chunk_content),
        })

    return chunks


def _normalize_sections_for_budget(
    sections: List[Dict[str, Any]],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for section in sections:
        normalized.extend(split_large_section(section, max_tokens=max_tokens))
    for idx, section in enumerate(normalized):
        section["order_index"] = idx
    return normalized


def _rebalance_batches_to_count(
    batches: List[List[Dict[str, Any]]],
    target_count: int,
) -> List[List[Dict[str, Any]]]:
    out = [list(batch) for batch in batches if batch]
    while len(out) < target_count:
        split_idx = max(
            range(len(out)),
            key=lambda idx: (len(out[idx]), sum(s.get("estimated_tokens", 0) for s in out[idx])),
        )
        batch = out[split_idx]
        if len(batch) <= 1:
            break
        mid = len(batch) // 2
        out[split_idx:split_idx + 1] = [batch[:mid], batch[mid:]]
    return out


def proportional_allocate(
    total: int,
    weights: List[int],
    *,
    min_each: int = 0,
) -> List[int]:
    n = len(weights)
    if n == 0:
        return []
    if total <= 0:
        return [0] * n
    if min_each * n > total:
        min_each = 0

    alloc = [min_each] * n
    remaining = total - (min_each * n)
    if remaining <= 0:
        return alloc

    safe_weights = [max(1, int(w or 0)) for w in weights]
    weight_sum = sum(safe_weights)
    raw = [(remaining * w) / weight_sum for w in safe_weights]
    floors = [math.floor(v) for v in raw]
    alloc = [a + f for a, f in zip(alloc, floors)]
    leftovers = remaining - sum(floors)
    order = sorted(
        range(n),
        key=lambda idx: (raw[idx] - floors[idx], safe_weights[idx]),
        reverse=True,
    )
    for idx in order[:leftovers]:
        alloc[idx] += 1
    return alloc


def build_section_batches(
    sections: List[Dict[str, Any]],
    *,
    target_pages: int,
    pages_per_batch: int,
    max_batch_tokens: int,
) -> List[Dict[str, Any]]:
    if not sections:
        return []

    normalized_sections = _normalize_sections_for_budget(sections, max_tokens=max_batch_tokens)
    total_tokens = sum(section.get("estimated_tokens", 0) for section in normalized_sections)
    if total_tokens <= 0:
        return []

    required_by_pages = max(1, math.ceil(max(1, target_pages) / max(1, pages_per_batch)))
    required_by_tokens = max(1, math.ceil(total_tokens / max(1, max_batch_tokens)))
    num_batches = max(required_by_pages, required_by_tokens)
    num_batches = min(num_batches, max(1, target_pages))

    target_tokens_per_batch = total_tokens / max(1, num_batches)
    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_tokens = 0

    for idx, section in enumerate(normalized_sections):
        remaining_sections = len(normalized_sections) - idx
        remaining_batches = num_batches - len(batches) - 1
        if (
            current_batch
            and current_tokens >= target_tokens_per_batch
            and remaining_sections > remaining_batches
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(section)
        current_tokens += section.get("estimated_tokens", 0)

    if current_batch:
        batches.append(current_batch)

    batches = _rebalance_batches_to_count(batches, num_batches)
    batch_tokens = [sum(section.get("estimated_tokens", 0) for section in batch) for batch in batches]

    if len(batches) == 1:
        page_budgets = [max(1, target_pages)]
    else:
        cover_pages = 1 if target_pages >= 1 else 0
        thank_pages = 1 if target_pages >= 2 else 0
        body_pages = max(0, target_pages - cover_pages - thank_pages)
        min_each = 1 if body_pages >= len(batches) else 0
        body_alloc = proportional_allocate(body_pages, batch_tokens, min_each=min_each)
        page_budgets = body_alloc
        page_budgets[0] += cover_pages
        page_budgets[-1] += thank_pages

    batch_dicts: List[Dict[str, Any]] = []
    for idx, (batch_sections, page_budget) in enumerate(zip(batches, page_budgets)):
        content = "\n\n".join((section.get("content") or "").strip() for section in batch_sections if section.get("content"))
        titles = [section.get("title") or "Untitled Section" for section in batch_sections]
        batch_dicts.append({
            "batch_index": idx,
            "total_batches": len(batches),
            "is_first": idx == 0,
            "is_last": idx == len(batches) - 1,
            "pages_to_generate": max(1, page_budget),
            "section_titles": titles,
            "sections": batch_sections,
            "content": content,
            "estimated_tokens": batch_tokens[idx],
        })

    return batch_dicts
