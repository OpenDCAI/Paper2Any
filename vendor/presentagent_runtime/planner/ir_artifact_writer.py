from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..renderer.artifact_store import write_json
from .compact_slide import build_compact_slide_payload
from .ir_models import DeckIR, SlideIR


class IRArtifactWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.materials_dir = self.run_dir / "materials"
        self.planned_dir = self.run_dir / "ir" / "planned"
        self.final_dir = self.run_dir / "ir" / "final"
        self.planned_slides_dir = self.planned_dir / "slides"
        self.final_slides_dir = self.final_dir / "slides"

    def load_existing_slide_docs(self, *, stage: str = "planned") -> dict[str, dict[str, Any]]:
        slides_dir = self.run_dir / "ir" / stage / "slides"
        if not slides_dir.exists():
            return {}

        slide_docs: dict[str, dict[str, Any]] = {}
        for slide_path in sorted(slides_dir.glob("slide_*.json")):
            try:
                slide_doc = json.loads(slide_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            slide_id = str(slide_doc.get("slide_id") or slide_path.stem).strip()
            if slide_id:
                slide_docs[slide_id] = slide_doc
        return slide_docs

    def write_materials(
        self,
        *,
        material_manifest: dict[str, Any],
        material_resolution: dict[str, Any],
    ) -> None:
        write_json(self.materials_dir / "material_manifest.json", material_manifest)
        write_json(
            self.materials_dir / "asset_catalog.json",
            {"assets": material_manifest.get("asset_catalog", [])},
        )
        write_json(
            self.materials_dir / "asset_request_contexts.json",
            {"contexts": material_manifest.get("asset_request_contexts", [])},
        )
        write_json(self.materials_dir / "material_resolution.json", material_resolution)

    def write_slide_briefs(self, slide_briefs: dict[str, Any] | list[dict[str, Any]], *, stage: str = "planned") -> None:
        if isinstance(slide_briefs, dict):
            payload = slide_briefs
        else:
            payload = {"slide_briefs": slide_briefs}
        write_json(self.run_dir / "ir" / stage / "slide_briefs.json", payload)

    def write_deck_stage(self, deck_stage: dict[str, Any], *, stage: str = "planned") -> None:
        write_json(self.run_dir / "ir" / stage / "deck_stage.json", deck_stage)

    def write(self, deck_ir: DeckIR, *, stage: str = "planned") -> dict[str, Any]:
        stage_dir = self.run_dir / "ir" / stage
        slides_dir = stage_dir / "slides"
        deck_artifact = self._build_deck_artifact(deck_ir)
        final_payload = deck_ir.model_dump()

        write_json(stage_dir / "deck_ir.json", deck_artifact)
        write_json(stage_dir / "final_ir.json", final_payload)
        write_json(stage_dir / "slide_evidence.json", self._build_slide_evidence(deck_ir.slides))
        self._write_slide_files(
            slides_dir,
            deck_ir.slides,
            deck_ir.metadata.model_dump(),
            deck_ir.title,
            deck_ir.source_asset_index,
        )
        return {
            "deck_path": str(stage_dir / "deck_ir.json"),
            "bundle_path": str(stage_dir / "final_ir.json"),
            "slides_dir": str(slides_dir),
            "evidence_path": str(stage_dir / "slide_evidence.json"),
        }

    def write_planned_ir(
        self,
        *,
        slide_briefs: list[dict[str, Any]],
        deck_stage: dict[str, Any],
        deck_ir: DeckIR,
    ) -> None:
        slide_briefs_metadata = {
            **deck_ir.metadata.model_dump(),
            "schema_name": "presentagent.slide_briefs",
            "stage": "planned",
        }
        self.write_slide_briefs(
            {
                "metadata": slide_briefs_metadata,
                "title_hint": deck_ir.title,
                "subtitle_hint": deck_ir.subtitle,
                "storyline_hint": deck_ir.storyline.model_dump(),
                "slide_briefs": slide_briefs,
                "planner_notes": list(deck_ir.planner_notes),
            },
            stage="planned",
        )
        self.write_deck_stage(deck_stage, stage="planned")
        self.write(deck_ir, stage="planned")

    def write_final_ir(self, deck_ir: DeckIR) -> None:
        self.write(deck_ir, stage="final")

    def _build_deck_artifact(self, deck_ir: DeckIR) -> dict[str, Any]:
        return {
            "metadata": deck_ir.metadata.model_dump(),
            "slide_manifest": list(deck_ir.slide_manifest),
            "source_asset_index": dict(deck_ir.source_asset_index),
            "storyline": deck_ir.storyline.model_dump(),
            "subtitle": deck_ir.subtitle,
            "theme": deck_ir.theme.model_dump(),
        }

    def _write_slide_files(
        self,
        directory: Path,
        slides: list[SlideIR],
        metadata: dict[str, Any],
        deck_title: str,
        source_asset_index: dict[str, Any],
    ) -> None:
        for slide in slides:
            slide_payload = self._build_slide_artifact(
                slide,
                metadata=metadata,
                deck_title=deck_title,
                source_assets=self._source_assets_for_slide(source_asset_index, slide_id=slide.slide_id),
            )
            write_json(directory / f"{slide.slide_id.replace('-', '_')}.json", slide_payload)

    def _build_slide_artifact(
        self,
        slide: SlideIR,
        *,
        metadata: dict[str, Any],
        deck_title: str,
        source_assets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_compact_slide_payload(slide, source_assets=source_assets)

    def _source_assets_for_slide(
        self,
        source_asset_index: dict[str, Any],
        *,
        slide_id: str,
    ) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        if not isinstance(source_asset_index, dict):
            return assets

        for key, value in source_asset_index.items():
            if isinstance(value, dict):
                target_slide_id = str(value.get("target_slide_id") or "").strip()
                if target_slide_id == slide_id:
                    assets.append(dict(value))
                continue

            if str(key or "").strip() != slide_id:
                continue
            raw_paths = value if isinstance(value, list) else [value]
            for index, raw_path in enumerate(raw_paths, start=1):
                path = str(raw_path or "").strip()
                if not path:
                    continue
                assets.append(
                    {
                        "asset_id": f"{slide_id}-asset-{index:02d}",
                        "asset_kind": "image",
                        "path": path,
                        "relative_path": path,
                        "target_slide_id": slide_id,
                    }
                )

        return sorted(assets, key=lambda item: str(item.get("asset_id") or ""))

    def _selected_asset_for_slide(
        self,
        slide: SlideIR,
        source_assets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected_asset_id = str(slide.selected_asset_id or "").strip()
        selected_asset_path = str(slide.selected_asset_path or "").strip()

        if selected_asset_id:
            for asset in source_assets:
                if str(asset.get("asset_id") or "").strip() == selected_asset_id:
                    return dict(asset)

        if selected_asset_path:
            for asset in source_assets:
                path = str(asset.get("path") or "").strip()
                relative_path = str(asset.get("relative_path") or "").strip()
                if selected_asset_path in {path, relative_path}:
                    return dict(asset)

        return dict(source_assets[0]) if source_assets else {}

    def _asset_paths_for_slide(
        self,
        slide: SlideIR,
        source_assets: list[dict[str, Any]],
        *,
        selected_asset: dict[str, Any],
    ) -> list[str]:
        paths: list[str] = []
        if source_assets:
            selected_path = str(selected_asset.get("path") or "").strip()
            if selected_path:
                paths.append(selected_path)
            for asset in source_assets:
                path = str(asset.get("path") or "").strip()
                if path:
                    paths.append(path)
            return list(dict.fromkeys(path for path in paths if path))

        paths.extend(str(path or "").strip() for path in slide.asset_paths)
        return list(dict.fromkeys(path for path in paths if path))

    def _build_visual_artifact(
        self,
        slide: SlideIR,
        selected_asset: dict[str, Any],
        source_assets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected_asset_id = str(selected_asset.get("asset_id") or "").strip()
        selected_asset_path = str(selected_asset.get("path") or "").strip()
        return {
            "visual_id": f"{slide.slide_id}-visual-1",
            "role": "primary",
            "asset_type": str(selected_asset.get("asset_kind") or "image"),
            "slot_id": self._preferred_visual_slot(slide),
            "target_area": self._preferred_visual_slot(slide),
            "selected_asset_id": selected_asset_id,
            "selected_asset_path": selected_asset_path,
            "selected_candidate": selected_asset,
            "candidate_pool": source_assets,
            "caption": str(selected_asset.get("caption") or ""),
            "intent": str((slide.layout or {}).get("visual_intent") or ""),
        }

    def _build_layout_artifact(self, slide: SlideIR) -> dict[str, Any]:
        layout = dict(slide.layout)
        return {
            "name": layout.get("name") or slide.layout_type,
            "variant": layout.get("variant", ""),
            "planner": layout.get("planner", ""),
            "slots": list(layout.get("slots") or []),
            "design_hint": layout.get("design_hint", ""),
            "title_align": layout.get("title_align", ""),
            "subtitle_align": layout.get("subtitle_align", ""),
            "title_position": layout.get("title_position", ""),
            "subtitle_position": layout.get("subtitle_position", ""),
            "visual_intent": layout.get("visual_intent", ""),
            "material_request_purpose": self._material_purpose_for_artifact(slide),
        }

    def _material_purpose_for_artifact(self, slide: SlideIR) -> str:
        purpose = str((slide.layout or {}).get("material_request_purpose") or "").strip()
        return purpose

    def _build_blocks(self, slide: SlideIR) -> list[dict[str, Any]]:
        if slide.blocks:
            return [dict(block) for block in slide.blocks]

        blocks: list[dict[str, Any]] = []
        if slide.title:
            blocks.append(
                {
                    "block_id": f"{slide.slide_id}-title",
                    "role": "title",
                    "kind": "headline",
                    "slot_id": "title",
                    "text": slide.title,
                    "content": slide.title,
                }
            )

        bullet_items = [point for point in slide.points if str(point).strip()]
        if bullet_items:
            blocks.append(
                {
                    "block_id": f"{slide.slide_id}-body",
                    "role": "bullets",
                    "kind": "bullet_list",
                    "slot_id": "body",
                    "items": bullet_items,
                }
            )
        return blocks

    @staticmethod
    def _preferred_visual_slot(slide: SlideIR) -> str:
        layout_name = str((slide.layout or {}).get("name") or slide.layout_type or "").strip()
        if layout_name in {"hero", "image_focus", "full_bleed_image"}:
            return "hero_visual"
        return "supporting_visual"

    def _layout_description(self, slide: SlideIR) -> str:
        return str(slide.core_message or slide.objective or "").strip()

    def _build_slide_evidence(self, slides: list[SlideIR]) -> dict[str, Any]:
        return {
            "slides": [
                {
                    "slide_id": slide.slide_id,
                    "source_chunk_ids": list(slide.source_chunk_ids),
                    "source_evidence": list(slide.source_evidence),
                }
                for slide in slides
            ]
        }
