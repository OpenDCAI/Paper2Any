from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "https://api.openai.com/v1"
JSON_RESULT_PREFIX = "__PAPER2ANY_JSON__"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cli"


@dataclass
class SkillRunResult:
    success: bool
    skill_name: str
    run_dir: str
    primary_output: str
    artifacts: list[str]
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _create_output_dir(skill_name: str, output_dir: str | os.PathLike[str] | None) -> Path:
    if output_dir:
        run_dir = Path(output_dir).expanduser().resolve()
    else:
        run_dir = (CLI_OUTPUT_ROOT / skill_name / str(int(time.time()))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _detect_input_type(input_value: str) -> str:
    path = Path(input_value)
    if not path.exists():
        return "TEXT"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "PDF"
    if suffix in {".ppt", ".pptx"}:
        return "PPTX"
    return "TEXT"


def _existing_path(*candidates: str | os.PathLike[str] | None) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return ""


def _collect_artifacts(run_dir: Path) -> list[str]:
    files = [path.resolve() for path in run_dir.rglob("*") if path.is_file()]
    return [str(path) for path in sorted(files)]


def _finalize_result(
    *,
    skill_name: str,
    run_dir: Path,
    primary_output: str,
    metadata: dict[str, Any],
    error: str | None = None,
) -> SkillRunResult:
    return SkillRunResult(
        success=bool(primary_output) and error is None,
        skill_name=skill_name,
        run_dir=str(run_dir),
        primary_output=primary_output,
        artifacts=_collect_artifacts(run_dir),
        metadata=metadata,
        error=error,
    )


async def run_paper2ppt_job(
    *,
    input_value: str,
    input_type: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-5.1",
    gen_fig_model: str = "gemini-2.5-flash-image-preview",
    language: str = "zh",
    style: str = "",
    page_count: int = 10,
    aspect_ratio: str = "16:9",
    use_long_paper: bool = False,
    output_dir: str | os.PathLike[str] | None = None,
) -> SkillRunResult:
    from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
    from dataflow_agent.workflow import run_workflow

    run_dir = _create_output_dir("paper2ppt", output_dir)
    resolved_input_type = input_type or _detect_input_type(input_value)

    path = Path(input_value)
    if resolved_input_type in {"PDF", "PPTX"}:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_value}")
        if resolved_input_type == "PDF" and path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got {path.suffix}")
        if resolved_input_type == "PPTX" and path.suffix.lower() not in {".ppt", ".pptx"}:
            raise ValueError(f"Expected PPT/PPTX file, got {path.suffix}")
        workflow_input = str(path.resolve())
    else:
        workflow_input = input_value

    effective_api_key = api_key or os.getenv("DF_API_KEY", "")
    if not effective_api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY.")

    request = Paper2FigureRequest(
        chat_api_url=api_url or os.getenv("DF_API_URL", DEFAULT_API_URL),
        api_key=effective_api_key,
        chat_api_key=effective_api_key,
        model=model,
        gen_fig_model=gen_fig_model,
        language=language,
        style=style,
        page_count=page_count,
        input_type=resolved_input_type,
        all_edited_down=True,
    )
    state = Paper2FigureState(
        request=request,
        messages=[],
        agent_results={},
        result_path=str(run_dir),
        aspect_ratio=aspect_ratio,
    )

    if resolved_input_type == "PDF":
        state.paper_file = workflow_input
    else:
        state.paper_file = workflow_input

    outline_workflow = "paper2page_content_for_long_paper" if use_long_paper else "paper2page_content"
    state = await run_workflow(outline_workflow, state)
    state = await run_workflow("paper2ppt_parallel_consistent_style", state)

    primary_output = _existing_path(
        getattr(state, "ppt_pptx_path", None),
        getattr(state, "ppt_path", None),
        next((str(path) for path in run_dir.rglob("*.pptx")), ""),
    )
    return _finalize_result(
        skill_name="paper2ppt",
        run_dir=run_dir,
        primary_output=primary_output,
        metadata={
            "input_type": resolved_input_type,
            "page_count": page_count,
            "language": language,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "use_long_paper": use_long_paper,
            "outline_workflow": outline_workflow,
            "pagecontent_count": len(getattr(state, "pagecontent", []) or []),
        },
        error=None if primary_output else "PPTX output not found",
    )


async def run_pdf2ppt_job(
    *,
    input_path: str,
    use_ai_edit: bool = False,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    gen_fig_model: str = "gemini-2.5-flash-image-preview",
    language: str = "zh",
    style: str = "现代简约风格",
    page_count: int = 8,
    output_dir: str | os.PathLike[str] | None = None,
) -> SkillRunResult:
    from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
    from dataflow_agent.workflow import run_workflow

    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got {source.suffix}")

    effective_api_key = api_key or os.getenv("DF_API_KEY", "")
    if use_ai_edit and not effective_api_key:
        raise ValueError("API key is required when --use-ai-edit is enabled.")

    run_dir = _create_output_dir("pdf2ppt", output_dir)
    request = Paper2FigureRequest(
        chat_api_url=api_url or os.getenv("DF_API_URL", DEFAULT_API_URL),
        api_key=effective_api_key,
        chat_api_key=effective_api_key,
        model=model,
        gen_fig_model=gen_fig_model,
        language=language,
        style=style,
        page_count=page_count,
        use_ai_edit=use_ai_edit,
    )
    state = Paper2FigureState(
        request=request,
        messages=[],
        result_path=str(run_dir),
    )
    state.pdf_file = str(source)
    workflow_name = "pdf2ppt_qwenvl" if use_ai_edit else "pdf2ppt_parallel"
    state = await run_workflow(workflow_name, state)

    primary_output = _existing_path(
        getattr(state, "ppt_path", None),
        next((str(path) for path in run_dir.rglob("*.pptx")), ""),
    )
    return _finalize_result(
        skill_name="pdf2ppt",
        run_dir=run_dir,
        primary_output=primary_output,
        metadata={
            "input_type": "PDF",
            "use_ai_edit": use_ai_edit,
            "page_count": page_count,
            "language": language,
            "style": style,
            "workflow": workflow_name,
        },
        error=None if primary_output else "PPTX output not found",
    )


async def run_image2ppt_job(
    *,
    input_path: str,
    use_ai_edit: bool = False,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    gen_fig_model: str = "gemini-2.5-flash-image-preview",
    language: str = "zh",
    style: str = "现代简约风格",
    page_count: int = 1,
    output_dir: str | os.PathLike[str] | None = None,
) -> SkillRunResult:
    from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
    from dataflow_agent.workflow import run_workflow

    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"Expected image file, got {source.suffix}")

    effective_api_key = api_key or os.getenv("DF_API_KEY", "")
    if use_ai_edit and not effective_api_key:
        raise ValueError("API key is required when --use-ai-edit is enabled.")

    run_dir = _create_output_dir("image2ppt", output_dir)
    request = Paper2FigureRequest(
        chat_api_url=api_url or os.getenv("DF_API_URL", DEFAULT_API_URL),
        api_key=effective_api_key,
        chat_api_key=effective_api_key,
        model=model,
        gen_fig_model=gen_fig_model,
        language=language,
        style=style,
        page_count=page_count,
        use_ai_edit=use_ai_edit,
        input_type="FIGURE",
    )
    state = Paper2FigureState(
        request=request,
        messages=[],
        result_path=str(run_dir),
        input_type="FIGURE",
    )
    state.pdf_file = str(source)
    workflow_name = "pdf2ppt_qwenvl" if use_ai_edit else "pdf2ppt_parallel"
    state = await run_workflow(workflow_name, state)

    primary_output = _existing_path(
        getattr(state, "ppt_path", None),
        next((str(path) for path in run_dir.rglob("*.pptx")), ""),
    )
    return _finalize_result(
        skill_name="image2ppt",
        run_dir=run_dir,
        primary_output=primary_output,
        metadata={
            "input_type": "FIGURE",
            "use_ai_edit": use_ai_edit,
            "page_count": page_count,
            "language": language,
            "style": style,
            "workflow": workflow_name,
        },
        error=None if primary_output else "PPTX output not found",
    )


def _convert_pptx_to_pdf(pptx_path: Path, output_dir: Path) -> Path:
    pdf_path = output_dir / "temp_slides.pdf"
    command = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "LibreOffice not found. Install libreoffice before running ppt-polish."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("PPT to PDF conversion timed out after 5 minutes.") from exc

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "LibreOffice conversion failed")

    generated = output_dir / f"{pptx_path.stem}.pdf"
    if generated.exists():
        generated.rename(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not created: {pdf_path}")
    return pdf_path


def _convert_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[str]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError("pdf2image is required for ppt-polish.") from exc

    images_dir = output_dir / "slide_images"
    images_dir.mkdir(exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=300)
    image_paths: list[str] = []
    for index, image in enumerate(images):
        image_path = images_dir / f"slide_{index:03d}.png"
        image.save(str(image_path), "PNG")
        image_paths.append(str(image_path))
    return image_paths


async def run_ppt2polish_job(
    *,
    input_path: str,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    gen_fig_model: str = "gemini-2.5-flash-image-preview",
    style: str = "现代简约风格",
    ref_img: str = "",
    output_dir: str | os.PathLike[str] | None = None,
) -> SkillRunResult:
    from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
    from dataflow_agent.workflow import run_workflow

    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if source.suffix.lower() not in {".ppt", ".pptx"}:
        raise ValueError(f"Expected PPT/PPTX file, got {source.suffix}")

    effective_api_key = api_key or os.getenv("DF_API_KEY", "")
    if not effective_api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY.")

    ref_img_path = ""
    if ref_img:
        ref_candidate = Path(ref_img).expanduser().resolve()
        if not ref_candidate.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_img}")
        ref_img_path = str(ref_candidate)

    run_dir = _create_output_dir("ppt2polish", output_dir)
    pdf_path = _convert_pptx_to_pdf(source, run_dir)
    image_paths = _convert_pdf_to_images(pdf_path, run_dir)

    request = Paper2FigureRequest(
        chat_api_url=api_url or os.getenv("DF_API_URL", DEFAULT_API_URL),
        api_key=effective_api_key,
        chat_api_key=effective_api_key,
        model=model,
        gen_fig_model=gen_fig_model,
        style=style,
        all_edited_down=True,
        ref_img=ref_img_path,
    )
    state = Paper2FigureState(
        request=request,
        messages=[],
        result_path=str(run_dir),
        pagecontent=[{"ppt_img_path": image_path} for image_path in image_paths],
    )
    state = await run_workflow("paper2ppt_parallel_consistent_style", state)

    primary_output = _existing_path(
        getattr(state, "ppt_pptx_path", None),
        getattr(state, "ppt_path", None),
        next((str(path) for path in run_dir.rglob("*.pptx")), ""),
    )
    return _finalize_result(
        skill_name="ppt2polish",
        run_dir=run_dir,
        primary_output=primary_output,
        metadata={
            "input_type": "PPTX",
            "style": style,
            "reference_image": ref_img_path,
            "slide_count": len(image_paths),
            "workflow": "paper2ppt_parallel_consistent_style",
        },
        error=None if primary_output else "PPTX output not found",
    )
