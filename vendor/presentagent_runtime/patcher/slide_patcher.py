"""Single-slide patch logic: text enrichment + image binding/generation."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from ..planner.ir_models import DeckIR, SlideIR, VisualBinding
from ..planner.content_enricher import (
    _parse_markdown,
    _match_slides_to_sections,
    MarkdownSection,
)
from ..planner.llm_planner import (
    PagecontentSlidePlanner,
    build_deck_context,
    parse_json_response,
)
from ..materials.mineru_asset_index import find_mineru_asset_matches

# ---------------------------------------------------------------------------
# Feedback classification
# ---------------------------------------------------------------------------

_IMAGE_KEYWORDS_ZH = ("图", "图片", "图像", "插图", "配图", "视觉", "缺图", "没有图", "加图", "添加图")
_IMAGE_KEYWORDS_EN = ("image", "figure", "photo", "picture", "visual", "illustration", "missing image", "add image")
_TEXT_KEYWORDS_ZH = ("文字", "文本", "内容", "补充", "修改", "更新", "增加", "丰富", "详细", "说明", "描述")
_TEXT_KEYWORDS_EN = ("text", "content", "update", "add more", "enrich", "detail", "description", "explain")


def _classify_feedback(feedback: str) -> str:
    """Return 'image', 'text', or 'both'."""
    lower = feedback.lower()
    has_image = any(kw in lower for kw in _IMAGE_KEYWORDS_ZH + _IMAGE_KEYWORDS_EN)
    has_text = any(kw in lower for kw in _TEXT_KEYWORDS_ZH + _TEXT_KEYWORDS_EN)
    if has_image and has_text:
        return "both"
    if has_image:
        return "image"
    return "text"


# ---------------------------------------------------------------------------
# Text patch (LLM-based SlideIR regeneration)
# ---------------------------------------------------------------------------

def _find_relevant_sections(
    slide: SlideIR,
    all_slides: list[SlideIR],
    sections: list[MarkdownSection],
    feedback: str,
) -> list[MarkdownSection]:
    """Find markdown sections most relevant to this slide + feedback."""
    slide_section_map = _match_slides_to_sections(all_slides, sections)
    matched = slide_section_map.get(slide.slide_id, [])

    feedback_tokens = set(re.findall(r"[\w一-鿿]+", feedback.lower()))
    if feedback_tokens:
        scored: list[tuple[float, MarkdownSection]] = []
        for sec in sections:
            heading_tokens = set(re.findall(r"[\w一-鿿]+", sec.heading.lower()))
            body_tokens = set(re.findall(r"[\w一-鿿]+", sec.body_text.lower()))
            overlap = len(feedback_tokens & (heading_tokens | body_tokens))
            if overlap > 0:
                scored.append((overlap, sec))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            top_feedback_sections = [s for _, s in scored[:2]]
            merged_ids = {id(s) for s in top_feedback_sections}
            extra = [s for s in matched if id(s) not in merged_ids]
            matched = top_feedback_sections + extra[:1]

    return matched


def _build_patch_prompt(
    slide: SlideIR,
    deck_ir: DeckIR,
    source_text: str,
    feedback: str,
) -> str:
    """Build the LLM prompt for patch-based SlideIR regeneration."""
    import json as _json
    from ..planner.llm_planner import ALLOWED_BLOCK_KINDS, ALLOWED_LAYOUTS, ALLOWED_SLOT_ROLES, ALLOWED_VISUAL_TYPES

    deck_context = build_deck_context(deck_ir)
    payload = {
        "deck_ir_context": {
            "metadata": deck_context["metadata"],
            "title": deck_context["title"],
            "theme": deck_context["theme"],
            "planner_notes": deck_context["planner_notes"],
        },
        "current_slide_ir": slide.model_dump(),
        "source_text_from_paper": source_text,
        "user_feedback": feedback,
    }
    return (
        "You are patching a single slide in an editable PPT runtime.\n"
        "The user has requested a modification. Use the source_text_from_paper as the factual basis "
        "and regenerate the slide content to satisfy the user_feedback.\n\n"
        "Critical rules:\n"
        "- Stay aligned with deck_ir_context: preserve the deck theme (colors, fonts, language, style).\n"
        "- Use source_text_from_paper as the primary content source — do not invent facts.\n"
        "- Keep the same slide topic and slide_id as current_slide_ir.\n"
        "- The output language must match the deck metadata language.\n\n"
        "Return ONE valid JSON object with these keys:\n"
        "- slide_id: copy current_slide_ir.slide_id\n"
        "- title: concise title (keep or refine from current)\n"
        "- subtitle: visible subtitle only; empty string if not needed\n"
        "- core_message: one sentence summarizing the slide's key insight\n"
        "- objective: one sentence on this slide's role in the deck\n"
        f"- layout_name: exactly one of {', '.join(ALLOWED_LAYOUTS)}\n"
        "- points: array of 3-5 short strings drawn from source_text_from_paper\n"
        "- visual_intent: describe image placement if a visual is needed; empty string for text-only\n"
        "- material_request_title: name of needed visual; empty if no visual needed\n"
        "- material_request_purpose: describe visual role; empty if no visual needed\n"
        "- selected_asset_id: copy from current_slide_ir if still relevant; otherwise empty string\n\n"
        f"Allowed block kinds: {', '.join(ALLOWED_BLOCK_KINDS)}.\n"
        f"Allowed slot roles: {', '.join(ALLOWED_SLOT_ROLES)}.\n"
        f"Allowed visual types: {', '.join(ALLOWED_VISUAL_TYPES)}.\n\n"
        f"Input:\n{_json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _patch_text(
    slide: SlideIR,
    deck_ir: DeckIR,
    markdown_path: Path,
    feedback: str,
    llm_client: Any = None,
) -> SlideIR:
    """Regenerate slide content via LLM using source text from full.md."""
    if not markdown_path.exists():
        return slide

    sections = _parse_markdown(markdown_path)
    if not sections:
        return slide

    matched = _find_relevant_sections(slide, deck_ir.slides, sections, feedback)
    if not matched:
        return slide

    combined_body = "\n\n".join(s.body_text for s in matched if s.body_text)
    if not combined_body:
        return slide

    # LLM path: regenerate full SlideIR fields
    if llm_client is not None:
        try:
            prompt = _build_patch_prompt(slide, deck_ir, combined_body, feedback)
            raw = llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format="json",
            )
            planned = parse_json_response(raw)
            slide_planner = PagecontentSlidePlanner(llm_client)
            patched = slide_planner._merge_slide(
                base_slide=slide,
                planned_slide=planned,
                deck_ir=deck_ir,
            )
            evidence = list(patched.source_evidence)
            evidence.append({
                "kind": "patch_text_llm",
                "feedback": feedback,
                "matched_headings": [s.heading for s in matched if s.heading],
            })
            return patched.model_copy(update={"source_evidence": evidence})
        except Exception:
            pass  # fall through to no-op if LLM fails

    # No LLM available: return slide unchanged (don't silently corrupt content)
    return slide


# ---------------------------------------------------------------------------
# Image patch
# ---------------------------------------------------------------------------

def _find_existing_image(slide: SlideIR, result_path: Path) -> str:
    """Try to find a matching image from the MinerU asset index."""
    query = slide.core_message or (slide.points[0] if slide.points else "") or slide.title
    if not query:
        return ""

    matches = find_mineru_asset_matches(
        query,
        asset_base_dir=str(result_path),
        limit=3,
    )
    if not matches:
        # Also try with title
        matches = find_mineru_asset_matches(
            slide.title,
            asset_base_dir=str(result_path),
            limit=3,
        )
    if matches:
        rel_path = str(matches[0].get("relative_path") or "")
        if rel_path:
            abs_path = result_path / rel_path
            if abs_path.exists():
                return str(abs_path)
    return ""


def _generate_image(
    slide: SlideIR,
    result_path: Path,
    image_client: Any,
) -> str:
    """Generate an image via dataflow_agent image pipeline and save to code_runtime/materials/generated/."""
    if image_client is None:
        return ""

    api_url = getattr(image_client, "api_base", "") or getattr(image_client, "api_url", "")
    api_key = getattr(image_client, "api_key", "")
    model = getattr(image_client, "model", "")
    if not (api_url and api_key and model):
        return ""

    out_dir = result_path / "code_runtime" / "materials" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"gen_{uuid.uuid4().hex[:12]}.png"

    prompt = _build_image_prompt(slide)
    try:
        import asyncio
        import concurrent.futures
        from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async

        async def _run():
            return await generate_or_edit_and_save_image_async(
                prompt=prompt,
                save_path=str(img_path),
                api_url=api_url,
                api_key=api_key,
                model=model,
                aspect_ratio="16:9",
                resolution="2K",
            )

        try:
            loop = asyncio.get_running_loop()
            # Running inside an async context (e.g. FastAPI) — run in a new thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run())
                future.result(timeout=300)
        except RuntimeError:
            # No running event loop — safe to call asyncio.run directly
            asyncio.run(_run())
    except Exception:
        return ""

    return str(img_path) if img_path.exists() and img_path.stat().st_size > 0 else ""


def _build_image_prompt(slide: SlideIR) -> str:
    title = slide.title or ""
    core = slide.core_message or ""
    points = "; ".join(slide.points[:2]) if slide.points else ""
    parts = [p for p in [title, core, points] if p]
    base = " — ".join(parts) if parts else "academic presentation slide"
    return f"Academic illustration for: {base}. Clean, professional, white background, suitable for a research presentation."


def _patch_image(
    slide: SlideIR,
    result_path: Path,
    feedback: str,
    image_client: Any = None,
) -> SlideIR:
    """Bind an existing or generated image to the slide."""
    slide_has_image = bool(slide.selected_asset_path or slide.asset_paths)

    if slide_has_image:
        # Slide already has an image — try to find a better match from paper assets first,
        # fall back to generation only if nothing found.
        img_path = _find_existing_image(slide, result_path)
        if not img_path and image_client is not None:
            img_path = _generate_image(slide, result_path, image_client)
    else:
        # Slide has no image — generate one directly, fall back to paper assets.
        if image_client is not None:
            img_path = _generate_image(slide, result_path, image_client)
        if not img_path:
            img_path = _find_existing_image(slide, result_path)

    if not img_path:
        return slide

    updates: dict[str, Any] = {}

    # Update asset_paths
    existing_paths = list(slide.asset_paths)
    if img_path not in existing_paths:
        updates["asset_paths"] = [img_path] + existing_paths

    # Update selected_asset_path
    updates["selected_asset_path"] = img_path

    # Update or add a VisualBinding
    existing_visuals = list(slide.visuals)
    visual_id = f"patch-visual-{uuid.uuid4().hex[:8]}"
    new_visual = VisualBinding(
        visual_id=visual_id,
        role="primary",
        asset_type="image",
        slot_id="hero_visual",
        selected_asset_path=img_path,
        intent=feedback or "patch image",
    )
    # Replace existing primary visual if present, else prepend
    replaced = False
    for i, v in enumerate(existing_visuals):
        if v.role == "primary":
            existing_visuals[i] = new_visual
            replaced = True
            break
    if not replaced:
        existing_visuals.insert(0, new_visual)
    updates["visuals"] = existing_visuals

    # Upgrade layout to image_focus if currently text-only
    if slide.layout_type in ("section_divider", "title_only", "closing"):
        updates["layout_type"] = "two_column"

    evidence = list(slide.source_evidence)
    evidence.append({
        "kind": "patch_image",
        "feedback": feedback,
        "image_path": img_path,
        "source": "generated" if "generated" in img_path else "existing",
    })
    updates["source_evidence"] = evidence

    return slide.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def patch_slide_ir(
    slide_ir: SlideIR,
    deck_ir: DeckIR,
    feedback: str,
    feedback_type: str = "auto",
    *,
    result_path: Path,
    llm_client: Any = None,
    image_client: Any = None,
) -> SlideIR:
    """Patch a single SlideIR based on user feedback.

    feedback_type: 'text' | 'image' | 'both' | 'auto'
    When 'auto', the type is inferred from the feedback text.
    llm_client: LLM client for text-based SlideIR regeneration.
    """
    if feedback_type == "auto":
        feedback_type = _classify_feedback(feedback)

    markdown_path = result_path / "auto" / "full.md"
    patched = slide_ir

    if feedback_type in ("text", "both"):
        patched = _patch_text(patched, deck_ir, markdown_path, feedback, llm_client=llm_client)

    if feedback_type in ("image", "both"):
        patched = _patch_image(patched, result_path, feedback, image_client=image_client)

    return patched
