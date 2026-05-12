from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from ..planner.compact_slide import build_compact_slide_payload, expand_compact_slide_payload
from ..planner.ir_models import DeckIR, SlideIR
from .library_skill_profiles import (
    build_library_generation_repair_hint,
    build_library_generation_skill_prompt,
)
from .pptx_library import create_presentation


class PPTXCodeAgent:
    def __init__(
        self,
        client,
        *,
        max_attempts: int = 2,
        library_generation_skill: str = "qwen_v1",
    ) -> None:
        self.client = client
        self.max_attempts = max(1, int(max_attempts))
        self.library_generation_skill = str(library_generation_skill or "qwen_v1").strip() or "qwen_v1"

    def generate_slide_code(self, deck_ir: dict, slide_ir: dict, materials: dict, *, function_name: str) -> str:
        prompt = self._build_slide_prompt(
            deck_ir=deck_ir,
            slide_ir=slide_ir,
            materials=materials,
            function_name=function_name,
        )
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return self._clean_code_block(raw, function_name=function_name)

    def generate_slide_code_with_feedback(
        self,
        deck_ir: dict,
        slide_ir: dict,
        materials: dict,
        *,
        function_name: str,
        artifact_dir: str | Path | None = None,
    ) -> tuple[str, dict[str, object]]:
        previous_code = ""
        last_error = ""
        validation_dir = self._build_validation_dir(artifact_dir, function_name) if artifact_dir is not None else None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if attempt == 1:
                    candidate_code = self.generate_slide_code(
                        deck_ir,
                        slide_ir,
                        materials,
                        function_name=function_name,
                    )
                else:
                    repair_prompt = self._build_slide_repair_prompt(
                        deck_ir=deck_ir,
                        slide_ir=slide_ir,
                        materials=materials,
                        function_name=function_name,
                        previous_code=previous_code,
                        error_message=last_error,
                    )
                    raw = self.client.chat(
                        [{"role": "user", "content": repair_prompt}],
                        temperature=0.2,
                    )
                    candidate_code = self._clean_code_block(raw, function_name=function_name)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt >= self.max_attempts:
                    break
                continue

            validation = self._validate_slide_code(
                candidate_code,
                function_name=function_name,
                deck_ir=deck_ir,
                slide_ir=slide_ir,
                materials=materials,
                validation_dir=validation_dir,
                attempt=attempt,
            )
            if validation["valid"]:
                metadata = {
                    "validated": True,
                    "attempt_count": attempt,
                    "repair_count": max(0, attempt - 1),
                    "last_error": "",
                    "validation_output_path": validation.get("output_path", ""),
                }
                return candidate_code, metadata

            previous_code = candidate_code
            last_error = str(validation["error"])
            if attempt >= self.max_attempts:
                break

        raise RuntimeError(f"Slide code validation failed for {function_name}: {last_error}")

    def _build_slide_prompt(self, *, deck_ir: dict, slide_ir: dict, materials: dict, function_name: str) -> str:
        slide_payload = self._build_prompt_slide_payload(slide_ir)
        materials_payload = self._build_prompt_materials_payload(materials)
        recommended_scaffold = self._recommended_library_scaffold(slide_ir)
        skill_prompt = build_library_generation_skill_prompt(self.library_generation_skill)
        return f"""You are a top-tier PowerPoint coding agent. Convert the current `SlideIR` into executable python-pptx code for exactly one slide.

Return code only: one complete function definition from `def {function_name}(prs, deck_ir, slide_ir, materials):` through `return slide`.

Role boundary:
- The planner has already produced the deck and this current `SlideIR`.
- The current `SlideIR` is the single source of truth for this slide's content, layout, and image/text design.
- Do not rewrite, summarize, invent, reorder, or replace slide content. Render what is in the current `SlideIR`.
- Use `deck_ir` only for global theme/language/context. Do not inspect other slides to decide this slide.
- Use `materials["current_slide"]` only for the already-selected asset context. Do not choose a different asset.

Hard constraints:
1. Work on only the current slide.
2. The function name must be `{function_name}`.
3. Keep this exact signature: `def {function_name}(prs, deck_ir, slide_ir, materials):`
4. The function must add exactly one slide to `prs`, fully render the page, and return the slide object.
5. `deck_ir`, `slide_ir`, and `materials` are plain dicts. Use `.get(...)` access, never dot notation.
6. Use library-first generation: preserve `slide_ir.layout`, use scaffold/slot helpers first, then body/visual helpers, then small local polish.
7. Use the compact slide payload. `blocks` owns all renderable content including title/body/image; `layout.slots` owns placement; `evidence` owns source trace. Do not invent separate image placement.
8. Use `materials["current_slide"]` for selected asset path/id/metadata when helpful.
9. If you use a scaffold helper, do not render the title/headline slot again; scaffold helpers already render the slide title.
10. If you use a scaffold helper, do not loop over `slide_ir["blocks"]` to render the same body/image/callout blocks again, and do not add panels behind slots already rendered by the scaffold.
11. Do not include markdown, explanations, imports, or code outside the requested function.

Recommended scaffold start:
- `{recommended_scaffold}`

Reliable extraction harness:
- `blocks = slide_ir.get("blocks") or []`
- `points = next((b.get("items") for b in blocks if b.get("slot") == "body"), []) or []`
- `image_block = next((b for b in blocks if b.get("kind") == "image"), {{}})`
- `visuals = slide_ir.get("visuals") or []`
- `slots = resolve_layout_slots(slide_ir)`

{skill_prompt}

Available helper API:
{self._library_reference()}

Deck constraints:
{json.dumps({k: v for k, v in deck_ir.items() if k != "slides"}, ensure_ascii=False, indent=2)[:4500]}

Current SlideIR payload:
{json.dumps(slide_payload, ensure_ascii=False, indent=2)[:6500]}

Current slide materials:
{json.dumps(materials_payload, ensure_ascii=False, indent=2)[:3500]}
"""

    def _build_slide_repair_prompt(
        self,
        *,
        deck_ir: dict,
        slide_ir: dict,
        materials: dict,
        function_name: str,
        previous_code: str,
        error_message: str,
    ) -> str:
        error_lower = str(error_message or "").lower()
        repair_policy = build_library_generation_skill_prompt(self.library_generation_skill)
        repair_hint = build_library_generation_repair_hint(self.library_generation_skill, error_message)
        if repair_hint:
            repair_policy += f"\n\n{repair_hint}"
        if "librarystaticcheckerror" in error_lower or "scaffold/helper" in error_lower or "low-level" in error_lower:
            repair_policy += """

Targeted repair:
- Rewrite around one scaffold helper for the main structure.
- Render visible body content from `blocks`, `points`, `bullets`, `summary`, or `core_message`.
- Keep low-level python-pptx calls only for small polish after helper placement.
"""
        elif "filenotfounderror" in error_lower or "asset" in error_lower or "image" in error_lower:
            repair_policy += """

Targeted repair:
- Preserve the scaffold and use `safe_resolve_asset_path`, `render_visual_in_slot`, `replace_visual_block`, or `safe_placeholder_panel`.
- Do not crash if an asset is missing; keep text content visible.
"""
        elif "nonetype" in error_lower or "is not iterable" in error_lower or "typeerror" in error_lower:
            repair_policy += """

Targeted repair:
- Add `or []`, `or {}`, and `or ""` guards before iterating or reading string values.
- Preserve the current slide structure.
"""
        elif "nameerror" in error_lower:
            repair_policy += """

Targeted repair:
- Do not add imports. Use only helper names from the available API.
- Replace unknown names with available helper calls.
"""

        return f"""Repair the previously generated runtime PowerPoint slide function.

Return code only: one complete function named `{function_name}` with signature `def {function_name}(prs, deck_ir, slide_ir, materials):`.
The function must add one slide and return the slide object. Do not return explanations, markdown, imports, or fragments.

Validation error:
{error_message}

Repair policy:
{repair_policy}

Recommended scaffold start:
- `{self._recommended_library_scaffold(slide_ir)}`

Available helper API:
{self._library_reference()}

Deck constraints:
{json.dumps({k: v for k, v in deck_ir.items() if k != "slides"}, ensure_ascii=False, indent=2)[:3500]}

Current SlideIR payload:
{json.dumps(self._build_prompt_slide_payload(slide_ir), ensure_ascii=False, indent=2)[:6500]}

Current slide materials:
{json.dumps(self._build_prompt_materials_payload(materials), ensure_ascii=False, indent=2)[:3500]}

Previous code:
{previous_code}
"""

    def _validate_slide_code(
        self,
        code: str,
        *,
        function_name: str,
        deck_ir: dict,
        slide_ir: dict,
        materials: dict,
        validation_dir: Path | None,
        attempt: int,
    ) -> dict[str, object]:
        try:
            parsed = ast.parse(code)
        except SyntaxError as exc:
            return {"valid": False, "error": f"SyntaxError: {exc.msg} (line {exc.lineno})"}

        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        target = next((node for node in function_defs if node.name == function_name), None)
        if target is None:
            return {"valid": False, "error": f"Missing function definition for {function_name}"}
        if len(function_defs) != 1 or any(not isinstance(node, (ast.FunctionDef, ast.Expr)) for node in parsed.body):
            return {"valid": False, "error": "Generated code must contain only the requested function definition"}

        arg_names = [arg.arg for arg in target.args.args]
        expected_args = ["prs", "deck_ir", "slide_ir", "materials"]
        if arg_names != expected_args:
            return {
                "valid": False,
                "error": f"Invalid signature for {function_name}: expected {expected_args}, got {arg_names}",
            }

        static_error = self._library_static_check(code, slide_ir=slide_ir)
        if static_error:
            return {"valid": False, "error": f"LibraryStaticCheckError: {static_error}"}

        runtime_validation = self._execute_single_slide_validation(
            code=code,
            function_name=function_name,
            deck_ir=deck_ir,
            slide_ir=slide_ir,
            materials=materials,
            validation_dir=validation_dir,
            attempt=attempt,
        )
        return runtime_validation

    def _execute_single_slide_validation(
        self,
        *,
        code: str,
        function_name: str,
        deck_ir: dict,
        slide_ir: dict,
        materials: dict,
        validation_dir: Path | None,
        attempt: int,
    ) -> dict[str, object]:
        output_path = ""
        script_path = ""
        if validation_dir is not None:
            validation_dir.mkdir(parents=True, exist_ok=True)
            script_path = str(validation_dir / f"attempt_{attempt:02d}.py")
            output_path = str(validation_dir / f"attempt_{attempt:02d}.pptx")
            Path(script_path).write_text(_build_validation_script(code, function_name), encoding="utf-8")

        namespace = _library_exec_namespace()
        try:
            exec(code, namespace)
            slide_fn = namespace.get(function_name)
            if not callable(slide_fn):
                return {"valid": False, "error": f"Generated object {function_name} is not callable", "output_path": output_path}

            prs = create_presentation()
            runtime_slide_ir = namespace.get("normalize_runtime_slide_ir", lambda value: value)(slide_ir)
            slides_before = len(prs.slides)
            slide = slide_fn(prs, deck_ir, runtime_slide_ir, materials)
            slides_after = len(prs.slides)
            if slides_after != slides_before + 1:
                return {"valid": False, "error": "Generated slide function must add exactly one slide", "output_path": output_path}
            if slide is None:
                return {"valid": False, "error": "Generated slide function returned None instead of the slide", "output_path": output_path}

            if output_path:
                prs.save(output_path)
                if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                    return {"valid": False, "error": "Single-slide validation PPTX was not created", "output_path": output_path}
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": f"{exc.__class__.__name__}: {exc}", "output_path": output_path}

        return {"valid": True, "error": "", "output_path": output_path, "script_path": script_path}

    def _build_prompt_slide_payload(self, slide_ir: dict) -> dict[str, Any]:
        if "content" in slide_ir or "assets" in slide_ir or "blocks" in slide_ir:
            return slide_ir
        return build_compact_slide_payload(slide_ir)

    @staticmethod
    def _build_prompt_materials_payload(materials: dict) -> dict[str, Any]:
        current_slide = dict(materials.get("current_slide") or {})
        selected_asset = dict(current_slide.get("selected_asset") or {})
        assets = []
        target_slide_id = str(current_slide.get("slide_id") or "").strip()
        for asset in materials.get("assets") or []:
            if target_slide_id and str(asset.get("target_slide_id") or "").strip() != target_slide_id:
                continue
            assets.append(
                {
                    "asset_id": asset.get("asset_id"),
                    "path": asset.get("path"),
                    "absolute_path": asset.get("absolute_path"),
                    "relative_path": asset.get("relative_path"),
                    "caption": asset.get("caption"),
                    "asset_exists": asset.get("asset_exists"),
                }
            )
            if len(assets) >= 5:
                break
        return {
            "current_slide": {
                "slide_id": current_slide.get("slide_id"),
                "selected_asset_id": current_slide.get("selected_asset_id"),
                "selected_asset_path": current_slide.get("selected_asset_path"),
                "selected_asset": {
                    "asset_id": selected_asset.get("asset_id"),
                    "path": selected_asset.get("path"),
                    "absolute_path": selected_asset.get("absolute_path"),
                    "caption": selected_asset.get("caption"),
                    "asset_exists": selected_asset.get("asset_exists"),
                },
                "selected_assets": current_slide.get("selected_assets") or assets,
                "resolution": current_slide.get("resolution") or {},
            },
            "assets_for_current_slide": assets,
        }

    @staticmethod
    def _recommended_library_scaffold(slide_ir: dict) -> str:
        if "content" in slide_ir or "assets" in slide_ir:
            slide_ir = expand_compact_slide_payload(slide_ir)
        layout_name = str((slide_ir.get("layout") or {}).get("name") or slide_ir.get("layout_type") or "").lower()
        slide_type = str(slide_ir.get("type") or "content").lower()
        has_visual = _slide_has_visual_binding(slide_ir)
        if layout_name == "comparison":
            return "render_comparison_scaffold(prs, deck_ir, slide_ir, materials)"
        if layout_name == "metric_focus":
            return "render_metric_focus_scaffold(prs, deck_ir, slide_ir, materials)"
        if layout_name == "chart_focus":
            return "render_chart_focus_scaffold(prs, deck_ir, slide_ir, materials)"
        if layout_name in {"image_focus", "full_bleed_image", "hero"} or has_visual:
            return "render_title_body_visual_scaffold(prs, deck_ir, slide_ir, materials)"
        if slide_type in {"title", "cover", "closing", "section"}:
            return "render_title_body_scaffold(prs, deck_ir, slide_ir, materials)"
        return "render_slide_scaffold(prs, deck_ir, slide_ir, materials)"

    @staticmethod
    def _library_static_check(code: str, *, slide_ir: dict) -> str | None:
        normalized = code.lower()
        scaffold_names = (
            "render_slide_scaffold",
            "render_title_body_scaffold",
            "render_title_body_visual_scaffold",
            "render_comparison_scaffold",
            "render_metric_focus_scaffold",
            "render_chart_focus_scaffold",
        )
        helper_names = (
            "render_block_in_slot",
            "render_visual_in_slot",
            "add_takeaway_block",
            "add_highlight_block",
            "add_metric_pair_block",
            "add_visual_with_caption_block",
            "compose_chart_with_takeaway",
            "compose_visual_with_observations",
            "compose_metrics_with_summary",
            "append_takeaway_block",
            "replace_visual_block",
            "safe_placeholder_panel",
            "safe_resolve_asset_path",
            "resolve_layout_slots",
        )
        uses_scaffold = any(name in normalized for name in scaffold_names)
        uses_helper = any(name in normalized for name in helper_names)
        low_level_ops = (
            "slide.shapes.add_textbox",
            "slide.shapes.add_shape",
            "slide.shapes.add_picture",
            "slide.shapes.add_table",
            "slide.shapes.add_chart",
        )
        low_level_count = sum(normalized.count(pattern) for pattern in low_level_ops)
        if low_level_count >= 2 and not (uses_scaffold or uses_helper):
            return "library mode should use scaffold/helper first for the main layout before low-level python-pptx calls"
        if re.search(r"\bslide_ir\.(?!get\()[a-zA-Z_]", code) or re.search(r"\bdeck_ir\.(?!get\()[a-zA-Z_]", code):
            return "runtime IR objects are dicts; use `.get(...)` instead of dot notation"
        if re.search(r"\bmaterials\.[a-zA-Z_]", code):
            return "materials is a dict; use `.get(...)` instead of dot notation"
        if ".delete()" in normalized:
            return "library mode may not call shape.delete(); rebuild or overwrite content without unsupported deletion APIs"
        if "deck_ir.get(\"slides\"" in normalized or "deck_ir.get('slides'" in normalized:
            return "coder must render only the current slide_ir and must not inspect deck_ir['slides']"
        if uses_scaffold and (
            "add_title_box(" in normalized
            or (
                "render_block_in_slot" in normalized
                and (
                    "title_block" in normalized
                    or 'slot") == "title"' in normalized
                    or "slot') == 'title'" in normalized
                    or 'slot_id") == "title"' in normalized
                    or "slot_id') == 'title'" in normalized
                    or 'slots.get("title"' in normalized
                    or "slots.get('title'" in normalized
                )
            )
        ):
            return "scaffold helpers already render the title/headline slot; do not render title again after scaffold"
        if uses_scaffold and (
            "render_block_in_slot" in normalized
            or "render_visual_in_slot" in normalized
            or "add_panel(" in normalized
            or "safe_placeholder_panel(" in normalized
            or "for block in blocks" in normalized
            or "for block in slide_ir.get" in normalized
        ):
            return "scaffold helpers already render slide blocks and visuals; do not render blocks, visual slots, or slot panels again after scaffold"
        slide_type = str(slide_ir.get("type") or "content").lower()
        if slide_type not in {"title", "cover", "closing", "section"}:
            content_sources = (
                'slide_ir.get("title"',
                "slide_ir.get('title'",
                'slide_ir.get("content"',
                "slide_ir.get('content'",
                'slide_ir.get("blocks"',
                "slide_ir.get('blocks'",
                'slide_ir.get("points"',
                "slide_ir.get('points'",
                'slide_ir.get("core_message"',
                "slide_ir.get('core_message'",
                "render_slide_scaffold",
                "render_title_body_scaffold",
                "render_title_body_visual_scaffold",
                "render_comparison_scaffold",
                "render_metric_focus_scaffold",
                "render_chart_focus_scaffold",
            )
            if not any(source in normalized for source in content_sources):
                return "content slides must render at least one body content source from `blocks`, `points`, `bullets`, `summary`, or `core_message`"
        if _slide_has_visual_binding(slide_ir) and not (
            "render_visual_in_slot" in normalized
            or "safe_resolve_asset_path" in normalized
            or "render_slide_scaffold" in normalized
            or "render_title_body_visual_scaffold" in normalized
            or "add_visual_with_caption_block" in normalized
            or "replace_visual_block" in normalized
        ):
            return "slides with visual bindings should render the selected visual through library visual helpers or a visual scaffold"
        return None

    @staticmethod
    def _library_reference() -> str:
        return """Helper names are already available in the execution namespace. Do not import anything.
- `slide = render_slide_scaffold(prs, deck_ir, slide_ir, materials)`
- `render_title_body_scaffold(prs, deck_ir, slide_ir, materials)`
- `render_title_body_visual_scaffold(prs, deck_ir, slide_ir, materials)`
- `render_comparison_scaffold(prs, deck_ir, slide_ir, materials)`
- `render_metric_focus_scaffold(prs, deck_ir, slide_ir, materials)`
- `render_chart_focus_scaffold(prs, deck_ir, slide_ir, materials)`
- `slots = resolve_layout_slots(slide_ir)` returns dict slot_id -> `(left, top, width, height)` in inches
- `render_block_in_slot(slide, block, slot_rect, theme, font_name=None)`
- `render_visual_in_slot(slide, slide_ir, materials, visual, slot_rect, theme)`
- `safe_resolve_asset_path(materials, slide_ir, visual=None)`
- `safe_placeholder_panel(slide, slot_rect, label="visual unavailable", theme=None)`
- `add_title_box`, `add_subtitle_box`, `add_textbox`, `add_bullet_list`, `add_panel`, `add_shape`
- `add_takeaway_block`, `add_highlight_block`, `add_metric_pair_block`, `add_visual_with_caption_block`, `replace_visual_block`
- `compose_visual_with_observations`, `compose_metrics_with_summary`, `compose_chart_with_takeaway`
Coordinates are in inches. Prefer scaffold/helper calls for the main layout."""

    def _clean_code_block(self, raw: str, *, function_name: str) -> str:
        text = self._extract_code_payload(raw).strip()
        match = re.search(rf"def\s+{re.escape(function_name)}\s*\(", text)
        if not match:
            raise RuntimeError(f"Generated code did not define {function_name}")
        text = text[match.start():].lstrip()
        lines = text.splitlines()
        collected: list[str] = []
        for index, line in enumerate(lines):
            if index == 0:
                collected.append(line)
                continue
            if line.strip() == "":
                collected.append(line)
                continue
            if not line.startswith((" ", "\t")):
                break
            collected.append(line)
        cleaned = "\n".join(collected).rstrip()
        cleaned = self._trim_trailing_syntax_fragments(cleaned)
        cleaned = re.sub(r"\bRgbColor\(", "RGBColor(", cleaned)
        if not cleaned.startswith(f"def {function_name}"):
            raise RuntimeError(f"Generated code did not start with expected function {function_name}")
        return cleaned + "\n"

    @staticmethod
    def _extract_code_payload(raw: str) -> str:
        text = raw or ""
        match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1)
        open_match = re.search(r"```(?:python)?\s*([\s\S]*)", text)
        if open_match:
            return open_match.group(1)
        return text

    @staticmethod
    def _trim_trailing_syntax_fragments(code: str) -> str:
        lines = code.splitlines()
        while len(lines) > 1:
            candidate = "\n".join(lines).rstrip()
            try:
                compile(candidate, "<generated-slide-code>", "exec")
                return candidate
            except SyntaxError as exc:
                if exc.lineno == len(lines):
                    lines.pop()
                    continue
                return candidate
        return "\n".join(lines).rstrip()

    @staticmethod
    def _build_validation_dir(artifact_dir: str | Path, function_name: str) -> Path:
        return Path(artifact_dir) / "validation" / function_name


def write_code_artifacts(
    run_dir: str | Path,
    deck_ir: DeckIR,
    *,
    code_agent: PPTXCodeAgent | None = None,
    material_manifest: dict[str, Any] | None = None,
    material_resolution: dict[str, Any] | None = None,
    per_slide_callback: "Callable[[int, int, str, str], None] | None" = None,
) -> Path:
    run_path = Path(run_dir)
    generated_dir = run_path / "code" / "generated"
    slides_dir = generated_dir / "slides"
    cache_dir = generated_dir / "cache"
    validation_root = generated_dir
    slides_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    materials_payload = _build_materials_payload(
        deck_ir=deck_ir,
        material_manifest=material_manifest or {},
        material_resolution=material_resolution or {},
    )

    slide_functions: list[tuple[SlideIR, str, str, dict[str, Any]]] = []
    total_slides = len(deck_ir.slides)
    deck_ir_dump = deck_ir.model_dump()
    for slide_idx, slide in enumerate(deck_ir.slides):
        module_name = slide.slide_id.replace("-", "_")
        function_name = module_name
        source_assets = _source_assets_for_slide_materials(slide, materials_payload)
        compact_slide = build_compact_slide_payload(slide, source_assets=source_assets)
        slide_materials = _build_slide_materials_payload(slide, materials_payload)
        if code_agent is not None:
            try:
                content, metadata = code_agent.generate_slide_code_with_feedback(
                    deck_ir_dump,
                    compact_slide,
                    slide_materials,
                    function_name=function_name,
                    artifact_dir=validation_root,
                )
            except Exception as exc:  # noqa: BLE001
                content = _fallback_slide_builder_content(function_name)
                metadata = {
                    "validated": False,
                    "attempt_count": code_agent.max_attempts,
                    "repair_count": max(0, code_agent.max_attempts - 1),
                    "last_error": str(exc),
                    "fallback_used": True,
                    "validation_output_path": "",
                }
        else:
            content = _fallback_slide_builder_content(function_name)
            metadata = {
                "validated": False,
                "attempt_count": 0,
                "repair_count": 0,
                "last_error": "",
                "fallback_used": True,
                "validation_output_path": "",
            }

        slide_path = slides_dir / f"{module_name}.py"
        slide_path.write_text(content, encoding="utf-8")
        slide_functions.append((slide, function_name, content, compact_slide))
        _write_slide_cache_metadata(
            cache_dir,
            module_name=module_name,
            slide=slide,
            function_name=function_name,
            code_path=slide_path,
            code=content,
            metadata=metadata,
        )
        if per_slide_callback is not None:
            try:
                per_slide_callback(slide_idx, total_slides, slide.slide_id, slide.title or "")
            except Exception:  # noqa: BLE001
                pass

    build_deck_path = generated_dir / "build_deck.py"
    build_deck_path.write_text(_build_deck_script(slide_functions), encoding="utf-8")
    return build_deck_path


def execute_build_deck(build_deck_path: str | Path, output_path: str | Path) -> None:
    build_path = Path(build_deck_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    module_dir = build_path.parent
    sys.path.insert(0, str(module_dir))
    try:
        importlib.invalidate_caches()
        module_name = f"paper2ppt_codegen_build_deck_{sha256(str(build_path).encode('utf-8')).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, build_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load build_deck module from {build_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_presentation"):
            raise RuntimeError("Generated build_deck.py does not define build_presentation")
        module.build_presentation(str(output))
    finally:
        if sys.path and sys.path[0] == str(module_dir):
            sys.path.pop(0)


def _write_slide_cache_metadata(
    cache_dir: Path,
    *,
    module_name: str,
    slide: SlideIR,
    function_name: str,
    code_path: Path,
    code: str,
    metadata: dict[str, object],
) -> None:
    payload = {
        "slide_id": slide.slide_id,
        "function_name": function_name,
        "validated": bool(metadata.get("validated", False)),
        "attempt_count": int(metadata.get("attempt_count", 0)),
        "repair_count": int(metadata.get("repair_count", 0)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "last_error": str(metadata.get("last_error", "")),
        "code_path": str(code_path),
        "code_hash": sha256(code.encode("utf-8")).hexdigest(),
        "validation_output_path": str(metadata.get("validation_output_path", "")),
    }
    (cache_dir / f"{module_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_materials_payload(
    *,
    deck_ir: DeckIR,
    material_manifest: dict[str, Any],
    material_resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "theme": deck_ir.theme.model_dump(),
        "manifest": material_manifest,
        "resolution": material_resolution,
        "assets": material_manifest.get("assets", []),
        "asset_index": material_manifest.get("asset_index", {}),
        "asset_catalog": material_manifest.get("asset_catalog", []),
        "asset_request_contexts": material_manifest.get("asset_request_contexts", []),
    }


def _build_slide_materials_payload(
    slide: SlideIR | dict[str, Any],
    materials_payload: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(slide, dict) and ("content" in slide or "assets" in slide):
        slide = expand_compact_slide_payload(slide)
    slide_id = _slide_get(slide, "slide_id", "")
    selected_asset_id = str(_slide_get(slide, "selected_asset_id", "") or "").strip()
    selected_asset_path = str(_slide_get(slide, "selected_asset_path", "") or "").strip()
    if isinstance(slide, dict):
        image_block = _image_block_from_compact_slide(slide)
        selected_asset_id = selected_asset_id or str(image_block.get("asset_id") or "").strip()
        selected_asset_path = selected_asset_path or str(image_block.get("path") or "").strip()
    selected_asset: dict[str, Any] = {}
    if selected_asset_id:
        selected_asset = dict((materials_payload.get("asset_index") or {}).get(selected_asset_id, {}))
    if not selected_asset and selected_asset_path:
        for asset in materials_payload.get("assets", []):
            if str(asset.get("path") or "").strip() == selected_asset_path:
                selected_asset = dict(asset)
                break
    assets_for_slide = _source_assets_for_slide_materials(slide, materials_payload)

    slide_resolution = {}
    for item in (materials_payload.get("resolution") or {}).get("requests", []):
        if str(item.get("target_slide_id") or "").strip() == slide_id:
            slide_resolution = dict(item)
            if not selected_asset:
                selected_asset = dict(item.get("resolved_candidate") or {})
            if not assets_for_slide:
                assets_for_slide = [
                    dict(asset)
                    for asset in item.get("resolved_candidates") or []
                    if isinstance(asset, dict)
                ]
            break

    return {
        **materials_payload,
        "current_slide": {
            "slide_id": slide_id,
            "selected_asset_id": selected_asset_id or str(selected_asset.get("asset_id") or ""),
            "selected_asset_path": selected_asset_path or str(selected_asset.get("path") or ""),
            "selected_asset": selected_asset,
            "selected_assets": assets_for_slide,
            "resolution": slide_resolution,
        },
    }


def _source_assets_for_slide_materials(
    slide: SlideIR | dict[str, Any],
    materials_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(slide, dict) and ("content" in slide or "assets" in slide):
        slide = expand_compact_slide_payload(slide)
    slide_id = str(_slide_get(slide, "slide_id", "") or "").strip()
    selected_asset_id = str(_slide_get(slide, "selected_asset_id", "") or "").strip()
    selected_asset_path = str(_slide_get(slide, "selected_asset_path", "") or "").strip()
    if isinstance(slide, dict):
        image_block = _image_block_from_compact_slide(slide)
        selected_asset_id = selected_asset_id or str(image_block.get("asset_id") or "").strip()
        selected_asset_path = selected_asset_path or str(image_block.get("path") or "").strip()
    assets: list[dict[str, Any]] = []
    for asset in materials_payload.get("assets") or []:
        target_slide_id = str(asset.get("target_slide_id") or "").strip()
        asset_id = str(asset.get("asset_id") or "").strip()
        path = str(asset.get("path") or "").strip()
        if (
            (slide_id and target_slide_id == slide_id)
            or (selected_asset_id and asset_id == selected_asset_id)
            or (selected_asset_path and path == selected_asset_path)
        ):
            assets.append(dict(asset))
    if selected_asset_id and not any(str(asset.get("asset_id") or "").strip() == selected_asset_id for asset in assets):
        indexed_asset = dict((materials_payload.get("asset_index") or {}).get(selected_asset_id, {}))
        if indexed_asset:
            assets.insert(0, indexed_asset)
    return assets


def _slide_get(slide: SlideIR | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(slide, dict):
        return slide.get(key, default)
    return getattr(slide, key, default)


def _slide_has_visual_binding(slide_ir: dict[str, Any]) -> bool:
    if slide_ir.get("visuals") or slide_ir.get("selected_asset_path"):
        return True
    assets = slide_ir.get("assets") or {}
    if assets.get("selected_asset_path") or assets.get("paths") or assets.get("items"):
        return True
    return bool(_image_block_from_compact_slide(slide_ir))


def _image_block_from_compact_slide(slide_ir: dict[str, Any]) -> dict[str, Any]:
    for block in slide_ir.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("kind") == "image" or block.get("slot") == "image" or block.get("role") == "image":
            return block
    return {}


def _truncate_jsonish(value: Any, *, max_items: int = 6, max_text: int = 240) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            compact[str(key)] = _truncate_jsonish(item, max_items=max_items, max_text=max_text)
        return compact
    if isinstance(value, list):
        return [_truncate_jsonish(item, max_items=max_items, max_text=max_text) for item in value[:max_items]]
    if isinstance(value, str):
        return value[:max_text]
    return value


def _fallback_slide_builder_content(function_name: str) -> str:
    return f"""def {function_name}(prs, deck_ir, slide_ir, materials):
    slide = render_slide_scaffold(prs, deck_ir, slide_ir, materials)
    return slide
"""


def _build_deck_script(slide_functions: list[tuple[SlideIR, str, str, dict[str, Any]]]) -> str:
    function_code = "\n\n".join(code.strip() for _, _, code, _ in slide_functions)
    compact_slides = [compact_slide for _, _, _, compact_slide in slide_functions]
    invocations = []
    for index, (slide, function_name, _, _) in enumerate(slide_functions):
        invocations.append(
            f"    slide_ir = compact_slides[{index}]\n"
            f"    slide_ir = normalize_runtime_slide_ir(slide_ir)\n"
            f"    slide_materials = build_slide_materials(slide_ir, materials)\n"
            f"    {function_name}(prs, deck_ir, slide_ir, slide_materials)"
        )
    invocations_code = "\n".join(invocations) if invocations else "    pass"
    return f"""from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation

from vendor.presentagent_runtime.coder.pptx_library import *


{function_code}


COMPACT_SLIDES = {json.dumps(compact_slides, ensure_ascii=False, indent=2)}


def build_slide_materials(slide_ir, materials):
    assets = slide_ir.get("assets") or {{}}
    selected_asset_id = str(slide_ir.get("selected_asset_id") or assets.get("selected_asset_id") or "").strip()
    selected_asset_path = str(slide_ir.get("selected_asset_path") or assets.get("selected_asset_path") or "").strip()
    if not selected_asset_path:
        for block in slide_ir.get("blocks") or []:
            slot = str(block.get("slot") or block.get("slot_id") or "")
            if block.get("kind") == "image" or slot == "image" or (slot.startswith("image_") and slot[6:].isdigit()):
                selected_asset_id = selected_asset_id or str(block.get("asset_id") or "").strip()
                selected_asset_path = str(block.get("path") or "").strip()
                break
    assets_for_slide = []
    slide_id = str(slide_ir.get("slide_id") or "").strip()
    for asset in materials.get("assets") or []:
        target_slide_id = str(asset.get("target_slide_id") or "").strip()
        asset_id = str(asset.get("asset_id") or "").strip()
        path = str(asset.get("path") or "").strip()
        if (
            (slide_id and target_slide_id == slide_id)
            or (selected_asset_id and asset_id == selected_asset_id)
            or (selected_asset_path and path == selected_asset_path)
        ):
            assets_for_slide.append(dict(asset))
    selected_asset = {{}}
    if selected_asset_id:
        selected_asset = dict((materials.get("asset_index") or {{}}).get(selected_asset_id, {{}}))
    if not selected_asset and selected_asset_path:
        for asset in materials.get("assets") or []:
            if str(asset.get("path") or "").strip() == selected_asset_path:
                selected_asset = dict(asset)
                break
    slide_resolution = {{}}
    for item in (materials.get("resolution") or {{}}).get("requests") or []:
        if str(item.get("target_slide_id") or "").strip() == str(slide_ir.get("slide_id") or "").strip():
            slide_resolution = dict(item)
            if not selected_asset:
                selected_asset = dict(item.get("resolved_candidate") or {{}})
            if not assets_for_slide:
                assets_for_slide = [dict(asset) for asset in item.get("resolved_candidates") or [] if isinstance(asset, dict)]
            break
    return {{
        **materials,
        "current_slide": {{
            "slide_id": slide_ir.get("slide_id", ""),
            "selected_asset_id": selected_asset_id or str(selected_asset.get("asset_id") or ""),
            "selected_asset_path": selected_asset_path or str(selected_asset.get("path") or ""),
            "selected_asset": selected_asset,
            "selected_assets": assets_for_slide,
            "resolution": slide_resolution,
        }},
    }}


def build_presentation(output_path):
    base_dir = Path(__file__).resolve().parents[2]
    deck_ir = json.loads((base_dir / "ir" / "final" / "final_ir.json").read_text(encoding="utf-8"))
    material_manifest_path = base_dir / "materials" / "material_manifest.json"
    material_manifest = json.loads(material_manifest_path.read_text(encoding="utf-8")) if material_manifest_path.exists() else {{}}
    material_resolution_path = base_dir / "materials" / "material_resolution.json"
    material_resolution = json.loads(material_resolution_path.read_text(encoding="utf-8")) if material_resolution_path.exists() else {{}}
    materials = {{
        "theme": deck_ir.get("theme", {{}}),
        "manifest": material_manifest,
        "resolution": material_resolution,
        "assets": material_manifest.get("assets", []),
        "asset_index": material_manifest.get("asset_index", {{}}),
        "asset_catalog": material_manifest.get("asset_catalog", []),
        "asset_request_contexts": material_manifest.get("asset_request_contexts", []),
    }}
    compact_slides = list(COMPACT_SLIDES)
    prs = create_presentation()
{invocations_code}
    prs.save(output_path)


if __name__ == "__main__":
    default_output = Path(__file__).resolve().parents[2] / "exports" / "paper2ppt_code_editable.pptx"
    default_output.parent.mkdir(parents=True, exist_ok=True)
    build_presentation(default_output)
"""


def _build_validation_script(code: str, function_name: str) -> str:
    return f"""from __future__ import annotations

from vendor.presentagent_runtime.coder.pptx_library import *


{code.strip()}
"""


def _library_exec_namespace() -> dict[str, Any]:
    from . import pptx_library

    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    for name in dir(pptx_library):
        if name.startswith("__"):
            continue
        namespace[name] = getattr(pptx_library, name)
    return namespace
