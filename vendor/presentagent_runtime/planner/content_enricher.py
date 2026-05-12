"""Enrich DeckIR slides with actual paper content from full.md.

Parses the MinerU-generated full.md, matches sections to slides by
sequential order + keyword overlap, and fills in real text/images.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ir_models import DeckIR, SlideIR

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*|\n+")
_DESCRIPTIVE_VERBS_ZH = ("介绍", "说明", "概括", "总结", "展示", "指出", "提出", "阐述", "描述", "强调", "比较")
_DESCRIPTIVE_VERBS_EN = ("introduce", "describe", "summarize", "explain", "present", "outline", "discuss", "highlight")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "while", "that",
    "this", "these", "those", "it", "its", "we", "our", "they", "their",
})


@dataclass
class MarkdownSection:
    heading: str
    level: int
    body_lines: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0

    @property
    def body_text(self) -> str:
        return "\n".join(self.body_lines)

    @property
    def text_length(self) -> int:
        return sum(len(line) for line in self.body_lines)


def enrich_deck_ir_from_markdown(deck_ir: DeckIR, markdown_path: Path) -> DeckIR:
    """Enrich each slide in deck_ir with actual content from full.md."""
    if not markdown_path.exists():
        return deck_ir

    sections = _parse_markdown(markdown_path)
    if not sections:
        return deck_ir

    auto_dir = markdown_path.parent.name
    slide_section_map = _match_slides_to_sections(deck_ir.slides, sections)

    updated_slides: list[SlideIR] = []
    for slide in deck_ir.slides:
        matched_sections = slide_section_map.get(slide.slide_id, [])
        if matched_sections:
            slide = _enrich_slide(slide, matched_sections, auto_dir=auto_dir)
        updated_slides.append(slide)

    return deck_ir.model_copy(update={"slides": updated_slides})


def _parse_markdown(markdown_path: Path) -> list[MarkdownSection]:
    try:
        lines = markdown_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    sections: list[MarkdownSection] = []
    current: MarkdownSection | None = None

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        heading_match = _HEADING_RE.match(line)

        if heading_match:
            if current is not None:
                current.end_line = line_num - 1
                sections.append(current)
            current = MarkdownSection(
                heading=heading_match.group(2).strip(),
                level=len(heading_match.group(1)),
                start_line=line_num,
            )
            continue

        if current is None:
            current = MarkdownSection(heading="", level=0, start_line=1)

        image_match = _IMAGE_RE.search(line)
        if image_match:
            current.image_paths.append(image_match.group(1).strip())
            continue

        stripped = line.strip()
        if stripped and not stripped.startswith("<table"):
            current.body_lines.append(stripped)

    if current is not None:
        current.end_line = len(lines)
        sections.append(current)

    return sections


def _match_slides_to_sections(
    slides: list[SlideIR],
    sections: list[MarkdownSection],
) -> dict[str, list[MarkdownSection]]:
    if not slides or not sections:
        return {}

    content_sections = [s for s in sections if s.heading.lower() not in ("", "references", "参考文献")]
    if not content_sections:
        content_sections = sections

    result: dict[str, list[MarkdownSection]] = {}
    num_slides = len(slides)

    first_slide = slides[0]
    last_slide = slides[-1]

    title_abstract = [s for s in content_sections if s.level == 1 or "abstract" in s.heading.lower() or "摘要" in s.heading]
    conclusion_sections = [s for s in content_sections if _is_conclusion_heading(s.heading)]

    first_assigned = title_abstract[:2] if title_abstract else content_sections[:1]
    result[first_slide.slide_id] = first_assigned

    last_title = (last_slide.title or "").lower()
    if "致谢" in last_title or "thank" in last_title or "结论" in last_title or "conclusion" in last_title:
        result[last_slide.slide_id] = conclusion_sections if conclusion_sections else content_sections[-1:]

    assigned_headings = {id(s) for group in result.values() for s in group}
    remaining_sections = [s for s in content_sections if id(s) not in assigned_headings]
    middle_slides = [s for s in slides if s.slide_id not in result]

    if middle_slides and remaining_sections:
        _assign_by_proportion(middle_slides, remaining_sections, result)

    _refine_by_keywords(slides, sections, result)

    return result


def _assign_by_proportion(
    slides: list[SlideIR],
    sections: list[MarkdownSection],
    result: dict[str, list[MarkdownSection]],
) -> None:
    num_slides = len(slides)
    num_sections = len(sections)

    if num_sections <= num_slides:
        for i, slide in enumerate(slides):
            if i < num_sections:
                result[slide.slide_id] = [sections[i]]
            else:
                result[slide.slide_id] = [sections[-1]]
        return

    sections_per_slide = num_sections / num_slides
    for i, slide in enumerate(slides):
        start_idx = int(i * sections_per_slide)
        end_idx = int((i + 1) * sections_per_slide)
        end_idx = max(end_idx, start_idx + 1)
        result[slide.slide_id] = sections[start_idx:end_idx]


def _refine_by_keywords(
    slides: list[SlideIR],
    all_sections: list[MarkdownSection],
    result: dict[str, list[MarkdownSection]],
) -> None:
    for slide in slides:
        assigned = result.get(slide.slide_id, [])
        if not assigned:
            continue

        slide_tokens = _tokenize(slide.title or "")
        if not slide_tokens:
            continue

        best_score = _keyword_overlap(slide_tokens, _tokenize(assigned[0].heading))
        best_section = assigned[0]

        for section in all_sections:
            if id(section) == id(assigned[0]):
                continue
            score = _keyword_overlap(slide_tokens, _tokenize(section.heading))
            if score > best_score + 0.3:
                already_used = any(
                    id(section) in {id(s) for s in group}
                    for group in result.values()
                )
                if not already_used:
                    best_score = score
                    best_section = section

        if id(best_section) != id(assigned[0]) and best_score > 0.4:
            result[slide.slide_id] = [best_section] + assigned


def _enrich_slide(
    slide: SlideIR,
    matched_sections: list[MarkdownSection],
    *,
    auto_dir: str,
) -> SlideIR:
    combined_body = "\n".join(s.body_text for s in matched_sections if s.body_text)
    all_images = []
    for s in matched_sections:
        for img_path in s.image_paths:
            full_path = f"{auto_dir}/{img_path}" if not img_path.startswith(auto_dir) else img_path
            all_images.append(full_path)

    updates: dict[str, Any] = {}

    if all_images and not slide.asset_paths:
        updates["asset_paths"] = all_images[:3]

    new_points = _extract_points(combined_body, slide.points, target_count=max(len(slide.points), 3))
    if new_points:
        updates["points"] = new_points
        updates["blocks"] = _rebuild_blocks(slide, new_points, all_images, auto_dir=auto_dir)

    source_headings = [s.heading for s in matched_sections if s.heading]
    if source_headings:
        evidence = list(slide.source_evidence)
        evidence.append({"kind": "full_md_enrichment", "matched_headings": source_headings})
        updates["source_evidence"] = evidence

    if not combined_body.strip():
        if all_images and "asset_paths" in updates:
            return slide.model_copy(update=updates)
        return slide

    if not updates.get("points") and combined_body:
        summary_sentences = _split_sentences(combined_body)[:2]
        if summary_sentences:
            updates["core_message"] = summary_sentences[0]
            updates["speaker_notes"] = summary_sentences[:3]

    return slide.model_copy(update=updates)


def _extract_points(
    body_text: str,
    original_points: list[str],
    target_count: int,
) -> list[str]:
    if not body_text.strip():
        return []

    if original_points and not _is_descriptive_points(original_points):
        return []

    sentences = _split_sentences(body_text)
    meaningful = [s for s in sentences if len(s) > 15 and not s.startswith("$")]

    if not meaningful:
        return []

    return meaningful[:target_count]


def _is_descriptive_points(points: list[str]) -> bool:
    descriptive_count = 0
    for point in points:
        lower = point.lower()
        if any(verb in lower for verb in _DESCRIPTIVE_VERBS_ZH):
            descriptive_count += 1
        elif any(verb in lower for verb in _DESCRIPTIVE_VERBS_EN):
            descriptive_count += 1
    return descriptive_count >= len(points) * 0.5


def _rebuild_blocks(
    slide: SlideIR,
    new_points: list[str],
    images: list[str],
    *,
    auto_dir: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    if slide.title:
        blocks.append({
            "block_id": f"{slide.slide_id}-title",
            "role": "title",
            "kind": "headline",
            "slot_id": "title",
            "text": slide.title,
            "content": slide.title,
        })

    if new_points:
        blocks.append({
            "block_id": f"{slide.slide_id}-body",
            "role": "body",
            "kind": "bullet_list",
            "slot_id": "body",
            "items": new_points,
        })

    if images:
        img_path = images[0]
        blocks.append({
            "block_id": f"{slide.slide_id}-image",
            "role": "image",
            "kind": "image",
            "slot_id": "image",
            "path": img_path,
        })

    return blocks


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if s and len(s) > 8:
            sentences.append(s)
    return sentences


def _is_conclusion_heading(heading: str) -> bool:
    lower = heading.lower()
    return any(kw in lower for kw in ("conclusion", "结论", "总结", "致谢", "acknowledgment"))


def _tokenize(text: str) -> set[str]:
    lower = text.lower()
    lower = re.sub(r"[^\w\s一-鿿]", " ", lower)
    words = set(lower.split()) - _STOPWORDS
    words = {w for w in words if len(w) > 1}
    chars = set()
    chinese = re.findall(r"[一-鿿]+", lower)
    for segment in chinese:
        for i in range(len(segment) - 1):
            chars.add(segment[i:i+2])
    return words | chars


def _keyword_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(1, len(tokens_a))
