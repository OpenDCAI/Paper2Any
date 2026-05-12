"""Compact skill profiles for runtime library-first code generation."""

from __future__ import annotations


def build_library_generation_skill_prompt(profile: str) -> str:
    normalized = str(profile or "qwen_v1").strip().lower()
    if normalized in {"", "none"}:
        return ""
    if normalized == "qwen_v1":
        return """
Library generation skill profile: qwen_v1
- Read the current slide IR before choosing helpers; the visible slide must be grounded in this one slide only.
- Prefer a single `render_slide_scaffold(prs, deck_ir, slide_ir, materials)` call as the entry point; the library picks an appropriate layout template from the IR (title/closing/section/card/quote/three_column/metric/two_column/image_hero/comparison) and handles title, body, visuals, and tasteful decoration internally.
- Do not add freestanding decorative shapes (extra `add_panel`, `add_shape`, background strips, or banners) after the scaffold call; any shape the template did not create is pruned before save and will not appear in the final deck.
- Render body content from real runtime IR fields only: `blocks`, `points`, `core_message`, `visuals`, `selected_asset_path`.
- If a real asset exists, the scaffold renders it in the intended visual slot with aspect-fit; do not replace it with a conceptual fake visual or reposition it with low-level `slide.shapes.add_picture` calls.
- If no real asset exists, let the scaffold route to a text-centric template; do not draw your own placeholder boxes.
""".strip()
    return ""


def build_library_generation_repair_hint(profile: str, error_message: str) -> str:
    normalized = str(profile or "qwen_v1").strip().lower()
    if normalized != "qwen_v1":
        return ""
    error_lower = str(error_message or "").lower()
    if "rgbcolor" in error_lower or "rgb" in error_lower:
        return """
Qwen generation harness repair hint:
- This is an RGBColor/color assignment failure. Use helper functions or `RGBColor(...)`; do not assign tuples or ints to `.rgb`.
""".strip()
    if "syntaxerror" in error_lower or "was never closed" in error_lower:
        return """
Qwen generation harness repair hint:
- This is a SyntaxError. Return one complete function, simplify the failing block, and close every bracket explicitly.
""".strip()
    return ""
