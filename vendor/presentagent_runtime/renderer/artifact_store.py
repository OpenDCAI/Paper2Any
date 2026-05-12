from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import EditablePPTRunArtifacts


def resolve_artifact_base_dir(base_dir: Path) -> Path:
    """Return the directory that should own code_runtime artifacts.

    Paper parsing writes MinerU/raw artifacts as:
      <result_path>/<pdf_stem>/auto/...

    In that case code_runtime should be a sibling of auto:
      <result_path>/<pdf_stem>/code_runtime/...

    For direct pagecontent/runtime calls without a nested auto folder, keep the
    historical layout:
      <result_path>/code_runtime/...
    """
    base_dir = Path(base_dir).expanduser()
    if (base_dir / "auto").is_dir():
        return base_dir

    auto_roots = sorted(
        child
        for child in base_dir.iterdir()
        if child.is_dir() and (child / "auto").is_dir()
    ) if base_dir.exists() else []
    if len(auto_roots) == 1:
        return auto_roots[0]
    return base_dir


def ensure_run_dirs(base_dir: Path) -> dict[str, Path]:
    artifact_base_dir = resolve_artifact_base_dir(base_dir)
    run_dir = artifact_base_dir / "code_runtime"
    directories = {
        "run_dir": run_dir,
        "materials_dir": run_dir / "materials",
        "ir_dir": run_dir / "ir",
        "planned_ir_dir": run_dir / "ir" / "planned",
        "planned_slides_dir": run_dir / "ir" / "planned" / "slides",
        "final_ir_dir": run_dir / "ir" / "final",
        "final_slides_dir": run_dir / "ir" / "final" / "slides",
        "refined_ir_dir": run_dir / "ir" / "refined",
        "recipes_dir": run_dir / "recipes",
        "code_generated_dir": run_dir / "code" / "generated",
        "code_generated_slides_dir": run_dir / "code" / "generated" / "slides",
        "exports_dir": run_dir / "exports",
        "logs_dir": run_dir / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def build_artifacts(run_dir: str) -> EditablePPTRunArtifacts:
    directories = ensure_run_dirs(Path(run_dir))
    return EditablePPTRunArtifacts(
        run_dir=str(directories["run_dir"]),
        materials_manifest_path=str(directories["materials_dir"] / "material_manifest.json"),
        material_resolution_path=str(directories["materials_dir"] / "material_resolution.json"),
        planned_ir_path=str(directories["planned_ir_dir"] / "final_ir.json"),
        final_ir_path=str(directories["final_ir_dir"] / "final_ir.json"),
        ir_path=str(directories["final_ir_dir"] / "final_ir.json"),
        recipe_path=str(directories["recipes_dir"] / "render_recipe.json"),
        pptx_path=str(directories["exports_dir"] / "paper2ppt_code_editable.pptx"),
        pdf_path="",
        log_path=str(directories["logs_dir"] / "run.log"),
    )
