from __future__ import annotations

import json
import os
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional

import httpx

from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.citationtool.citation_utils import (
    CitationDataError,
    extract_doi_from_input,
    is_close_title,
    normalize_name,
)

log = get_logger(__name__)

BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "section",
    "article",
    "blockquote",
    "table",
    "tr",
    "td",
    "br",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
REFERENCE_HEADINGS = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "citations",
    "reference list",
}
ALLOWED_INTENTS = [
    "related_work",
    "background",
    "method_use",
    "baseline_comparison",
    "extension",
    "critique",
]
MAX_CONTEXTS = 6


class _HTMLBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: List[Dict[str, str]] = []
        self._buffer: List[str] = []
        self._current_kind = "text"
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        lower_tag = tag.lower()
        if lower_tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lower_tag in HEADING_TAGS:
            self._flush()
            self._current_kind = "heading"
        elif lower_tag in BLOCK_TAGS:
            self._flush()
            self._current_kind = "text"

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in {"script", "style", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if lower_tag in HEADING_TAGS | BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        raw = unescape(" ".join(self._buffer))
        text = re.sub(r"\s+", " ", raw).strip()
        self._buffer = []
        if not text:
            return
        if self.blocks and self.blocks[-1]["text"] == text:
            return
        self.blocks.append({"kind": self._current_kind, "text": text})
        self._current_kind = "text"


def _extract_blocks_from_html(html: str) -> List[Dict[str, str]]:
    parser = _HTMLBlockParser()
    parser.feed(html)
    parser.close()
    return parser.blocks


def _looks_like_heading(block: Dict[str, str]) -> bool:
    if block.get("kind") == "heading":
        return True
    text = (block.get("text") or "").strip()
    if not text or len(text) > 90:
        return False
    if re.search(r"[.!?;:]$", text):
        return False
    words = text.split()
    if len(words) > 12:
        return False
    return True


def _split_body_and_references(blocks: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ref_index = -1
    for index, block in enumerate(blocks):
        text = (block.get("text") or "").strip().casefold()
        if text in REFERENCE_HEADINGS:
            ref_index = index
            break
    if ref_index < 0:
        return blocks, []
    return blocks[:ref_index], blocks[ref_index + 1 :]


def _normalize_title_tokens(title: str) -> List[str]:
    return [token for token in re.split(r"\W+", normalize_name(title)) if token]


def _title_overlap_score(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if is_close_title(left, right):
        return 8
    left_tokens = set(_normalize_title_tokens(left))
    right_tokens = set(_normalize_title_tokens(right))
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens)
    if overlap >= max(4, int(len(left_tokens) * 0.7)):
        return 6
    if overlap >= max(3, int(len(left_tokens) * 0.45)):
        return 4
    return 0


def _extract_author_surnames(authors: Iterable[str]) -> List[str]:
    surnames: List[str] = []
    for author in authors:
        tokens = [part.strip() for part in re.split(r"[\s,]+", author or "") if part.strip()]
        if not tokens:
            continue
        surname = tokens[-1]
        normalized = normalize_name(surname)
        if normalized and normalized not in {normalize_name(item) for item in surnames}:
            surnames.append(surname)
    return surnames


def _reference_match_score(
    block_text: str,
    *,
    target_doi: str,
    target_title: str,
    target_authors: List[str],
    target_year: Optional[int],
) -> int:
    score = 0
    lower_text = block_text.casefold()
    if target_doi and target_doi in lower_text:
        score += 12
    score += _title_overlap_score(block_text, target_title)
    surnames = _extract_author_surnames(target_authors[:3])
    for surname in surnames:
        if normalize_name(surname) and normalize_name(surname) in normalize_name(block_text):
            score += 2
            break
    if target_year and str(target_year) in block_text:
        score += 1
    return score


def _extract_reference_marker(reference_text: str) -> str:
    text = reference_text.strip()
    patterns = [
        r"^(\[\s*\d+(?:\s*[-,;]\s*\d+)*\])",
        r"^(\(\s*\d+(?:\s*[-,;]\s*\d+)*\))",
        r"^(\d+\.)",
        r"^(\d+\))",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1)
    return ""


def _match_reference_entry(
    reference_blocks: List[Dict[str, str]],
    *,
    target_doi: str,
    target_title: str,
    target_authors: List[str],
    target_year: Optional[int],
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"matched": False, "score": 0, "reference_text": "", "marker": "", "matched_by": ""}
    for block in reference_blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        score = _reference_match_score(
            text,
            target_doi=target_doi,
            target_title=target_title,
            target_authors=target_authors,
            target_year=target_year,
        )
        if score <= 0 or score <= best["score"]:
            continue
        matched_by = "title"
        if target_doi and target_doi in text.casefold():
            matched_by = "doi"
        elif target_year and str(target_year) in text:
            matched_by = "author_year"
        best = {
            "matched": True,
            "score": score,
            "reference_text": text,
            "marker": _extract_reference_marker(text),
            "matched_by": matched_by,
        }
    return best


def _build_inline_patterns(
    *,
    reference_marker: str,
    target_authors: List[str],
    target_year: Optional[int],
) -> List[re.Pattern[str]]:
    patterns: List[re.Pattern[str]] = []
    marker = reference_marker.strip()
    marker_digits = re.search(r"\d+", marker)
    if marker_digits:
        number = marker_digits.group(0)
        patterns.append(
            re.compile(
                rf"\[\s*(?:\d+\s*[-,;]\s*)*{re.escape(number)}(?:\s*[-,;]\s*\d+)*\s*\]",
                re.I,
            )
        )
        patterns.append(
            re.compile(
                rf"\(\s*(?:\d+\s*[-,;]\s*)*{re.escape(number)}(?:\s*[-,;]\s*\d+)*\s*\)",
                re.I,
            )
        )

    surnames = _extract_author_surnames(target_authors[:2])
    year_fragment = str(target_year) if target_year else r"20\d{2}|19\d{2}"
    for surname in surnames:
        patterns.append(
            re.compile(
                rf"\b{re.escape(surname)}(?:\s+et\s+al\.?)?(?:,?\s*\(?{year_fragment}\)?)?",
                re.I,
            )
        )
        patterns.append(
            re.compile(
                rf"\b{re.escape(surname)}(?:\s+and\s+[A-Z][a-z]+)?(?:,?\s*{year_fragment})",
                re.I,
            )
        )
    return patterns


def _split_sentences(paragraph: str) -> List[str]:
    text = re.sub(r"\s+", " ", paragraph).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [part.strip() for part in parts if part.strip()]


def _best_sentence(paragraph: str, patterns: List[re.Pattern[str]]) -> tuple[str, str]:
    sentences = _split_sentences(paragraph)
    for sentence in sentences:
        for pattern in patterns:
            match = pattern.search(sentence)
            if match:
                return sentence, match.group(0)
    return (sentences[0] if sentences else paragraph, "")


def _section_label(blocks: List[Dict[str, str]], index: int) -> str:
    for reverse_index in range(index - 1, -1, -1):
        block = blocks[reverse_index]
        if _looks_like_heading(block):
            return block["text"]
    return ""


def _locate_contexts(
    body_blocks: List[Dict[str, str]],
    *,
    reference_entry: Dict[str, Any],
    target_authors: List[str],
    target_year: Optional[int],
) -> List[Dict[str, Any]]:
    patterns = _build_inline_patterns(
        reference_marker=reference_entry.get("marker", ""),
        target_authors=target_authors,
        target_year=target_year,
    )
    contexts: List[Dict[str, Any]] = []
    seen_sentences: set[str] = set()
    for index, block in enumerate(body_blocks):
        paragraph = (block.get("text") or "").strip()
        if not paragraph or _looks_like_heading(block):
            continue
        if not any(pattern.search(paragraph) for pattern in patterns):
            continue
        sentence, marker = _best_sentence(paragraph, patterns)
        normalized_sentence = normalize_name(sentence)
        if not normalized_sentence or normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        score = 0.55
        if reference_entry.get("marker") and marker and any(char.isdigit() for char in marker):
            score = 0.92
        elif marker:
            score = 0.82
        contexts.append(
            {
                "section": _section_label(body_blocks, index),
                "sentence": sentence,
                "paragraph": paragraph[:1600],
                "marker": marker or reference_entry.get("marker", ""),
                "confidence": round(score, 2),
            }
        )
        if len(contexts) >= MAX_CONTEXTS:
            break
    return contexts


def _heuristic_intents(contexts: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    joined = " ".join(
        f"{item.get('section', '')} {item.get('sentence', '')} {item.get('paragraph', '')}"
        for item in contexts
    ).casefold()
    keyword_map = [
        ("related_work", ["related work", "prior work", "previous work", "existing work"]),
        ("background", ["background", "motivation", "overview"]),
        ("method_use", ["we use", "we adopt", "following", "based on", "build on", "implemented"]),
        ("baseline_comparison", ["compare", "baseline", "outperform", "against", "compared with"]),
        ("extension", ["extend", "extends", "extension", "inspired by"]),
        ("critique", ["limitation", "however", "fails", "weakness", "critic"]),
    ]
    for label, keywords in keyword_map:
        if any(keyword in joined for keyword in keywords):
            labels.append(label)
    return labels or ["related_work"]


def _heuristic_summary(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not contexts:
        return {
            "summary": "",
            "citation_intents": [],
            "best_effort_notice": "No inline citation context was found in the fetched HTML text.",
        }
    sections = [item.get("section", "").strip() for item in contexts if item.get("section")]
    intents = _heuristic_intents(contexts)
    section_text = ", ".join(dict.fromkeys(sections[:3]))
    summary = "The citing paper mentions the target work"
    if section_text:
        summary += f" in sections such as {section_text}"
    summary += f", most likely as {', '.join(intents).replace('_', ' ')}."
    return {
        "summary": summary,
        "citation_intents": intents,
        "best_effort_notice": "Citation context was extracted from publicly accessible HTML text using rule-based reference matching.",
    }


def _citation_llm_config() -> Optional[Dict[str, str]]:
    base_url = (
        os.getenv("PAPER2CITATION_WEBSEARCH_API_URL")
        or os.getenv("DF_API_URL")
        or ""
    ).strip()
    api_key = (
        os.getenv("PAPER2CITATION_WEBSEARCH_API_KEY")
        or os.getenv("DF_API_KEY")
        or ""
    ).strip()
    model = (
        os.getenv("PAPER2CITATION_WEBSEARCH_MODEL")
        or os.getenv("THIRD_PARTY_MODEL")
        or ""
    ).strip()
    if not base_url or not api_key or not model:
        return None
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model}


def _extract_json_block(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


async def _llm_summarize_contexts(
    *,
    target_work: Dict[str, Any],
    citing_work: Dict[str, Any],
    reference_entry: Dict[str, Any],
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    config = _citation_llm_config()
    if not config or not contexts:
        return {}

    prompt = f"""
You are summarizing how a citing paper uses a target paper.
Return valid JSON only.

Allowed citation_intents:
{json.dumps(ALLOWED_INTENTS)}

Target paper:
{json.dumps(target_work, ensure_ascii=False, indent=2)}

Citing paper:
{json.dumps(citing_work, ensure_ascii=False, indent=2)}

Matched reference:
{json.dumps(reference_entry, ensure_ascii=False, indent=2)}

Extracted citation contexts:
{json.dumps(contexts, ensure_ascii=False, indent=2)}

Return JSON with this exact shape:
{{
  "summary": "one short paragraph",
  "citation_intents": ["related_work"],
  "best_effort_notice": "one short sentence"
}}

Rules:
- Summarize only the provided contexts.
- Do not invent contexts that are not present.
- citation_intents must come from the allowed list.
- Prefer precise wording over broad claims.
""".strip()

    payload = {
        "model": config["model"],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
    }

    timeout = float(os.getenv("PAPER2CITATION_WEBSEARCH_TIMEOUT_SECONDS") or "45")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(
                f"{config['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        content = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
        parsed = _extract_json_block(content)
        intents = [
            str(item).strip()
            for item in parsed.get("citation_intents") or []
            if str(item).strip() in ALLOWED_INTENTS
        ]
        summary = str(parsed.get("summary") or "").strip()
        notice = str(parsed.get("best_effort_notice") or "").strip()
        if not summary:
            return {}
        return {
            "summary": summary,
            "citation_intents": intents,
            "best_effort_notice": notice,
        }
    except Exception as exc:
        log.warning(f"[paper2citation] LLM citation-context summary failed: {exc}")
        return {}


def _candidate_html_urls(raw_work: Dict[str, Any]) -> List[str]:
    urls: List[str] = []

    def _push(value: str) -> None:
        url = (value or "").strip()
        if not url or url.lower().endswith(".pdf") or url in urls:
            return
        urls.append(url)

    ids = raw_work.get("ids") or {}
    doi = extract_doi_from_input(ids.get("doi", "") or raw_work.get("doi", ""))
    if doi:
        _push(f"https://doi.org/{doi}")

    for location in [
        raw_work.get("best_oa_location") or {},
        raw_work.get("primary_location") or {},
        *(raw_work.get("locations") or []),
    ]:
        if isinstance(location, dict):
            _push(location.get("landing_page_url", "") or "")

    return urls


async def _fetch_first_html(urls: List[str]) -> tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Paper2Any/1.0; +https://paper2any.ai)"
    }
    last_error = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(25.0), headers=headers, follow_redirects=True) as client:
        for url in urls:
            try:
                response = await client.get(url)
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "").casefold()
                if "html" not in content_type and "xml" not in content_type:
                    last_error = f"non-html content-type for {url}: {content_type}"
                    continue
                return url, response.text
            except Exception as exc:
                last_error = str(exc)
                continue
    raise CitationDataError(f"Could not fetch a publicly readable HTML page for the citing paper: {last_error}")


def _target_metadata(raw_work: Dict[str, Any], simplified_work: Dict[str, Any]) -> Dict[str, Any]:
    authors = []
    for authorship in raw_work.get("authorships") or []:
        author = authorship.get("author") or {}
        display_name = (author.get("display_name") or "").strip()
        if display_name:
            authors.append(display_name)
    return {
        "title": simplified_work.get("title", ""),
        "doi": simplified_work.get("doi", ""),
        "year": simplified_work.get("year"),
        "authors": authors,
    }


async def extract_citation_context_for_work(
    *,
    target_raw_work: Dict[str, Any],
    target_work: Dict[str, Any],
    citing_raw_work: Dict[str, Any],
    citing_work: Dict[str, Any],
) -> Dict[str, Any]:
    urls = _candidate_html_urls(citing_raw_work)
    if not urls:
        raise CitationDataError("No candidate HTML URL was found for the citing paper.")

    source_url, html = await _fetch_first_html(urls)
    blocks = _extract_blocks_from_html(html)
    if not blocks:
        raise CitationDataError("Fetched page did not contain readable HTML text blocks.")

    body_blocks, reference_blocks = _split_body_and_references(blocks)
    meta = _target_metadata(target_raw_work, target_work)
    reference_entry = _match_reference_entry(
        reference_blocks,
        target_doi=(meta.get("doi") or "").casefold(),
        target_title=meta.get("title", ""),
        target_authors=meta.get("authors") or [],
        target_year=meta.get("year"),
    )
    contexts = _locate_contexts(
        body_blocks,
        reference_entry=reference_entry,
        target_authors=meta.get("authors") or [],
        target_year=meta.get("year"),
    )
    summary_bundle = _heuristic_summary(contexts)
    llm_bundle = await _llm_summarize_contexts(
        target_work=target_work,
        citing_work=citing_work,
        reference_entry=reference_entry,
        contexts=contexts,
    )
    if llm_bundle.get("summary"):
        summary_bundle.update({k: v for k, v in llm_bundle.items() if v})

    if reference_entry.get("matched") and not contexts:
        summary_bundle["best_effort_notice"] = (
            "The reference entry was matched, but no inline citation sentence was found in the fetched HTML body. "
            "This usually means the page is truncated, dynamically rendered, or hides the full text."
        )

    return {
        "source_url": source_url,
        "target_reference_match": {
            "matched": bool(reference_entry.get("matched")),
            "matched_by": reference_entry.get("matched_by", ""),
            "marker": reference_entry.get("marker", ""),
            "reference_text": reference_entry.get("reference_text", ""),
            "confidence": round(min(float(reference_entry.get("score", 0)) / 12.0, 1.0), 2) if reference_entry.get("score") else 0.0,
        },
        "contexts": contexts,
        "summary": summary_bundle.get("summary", ""),
        "citation_intents": summary_bundle.get("citation_intents", []),
        "best_effort_notice": summary_bundle.get("best_effort_notice", ""),
    }
