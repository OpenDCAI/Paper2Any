#!/usr/bin/env python3
"""
PPT2Polish CLI - Beautify existing PPT files

Usage:
    # Basic beautification
    python script/run_ppt2polish_cli.py --input old_presentation.pptx --style "学术风格，简洁大方" --api-key sk-xxx

    # With reference image for consistent style
    python script/run_ppt2polish_cli.py --input old_presentation.pptx --style "现代简约风格" --ref-img reference_style.png
"""

import argparse
import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script.cli_env import (
    find_output_artifacts,
    load_project_env,
    resolve_cli_image_credentials,
    resolve_cli_text_credentials,
)
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse
from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2ppt_wf_api

load_project_env()

log = get_logger(__name__)


def _state_get(state, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="PPT2Polish CLI - Beautify existing PPT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic beautification
  python script/run_ppt2polish_cli.py --input old_presentation.pptx --style "学术风格，简洁大方" --api-key sk-xxx

  # With reference image for consistent style
  python script/run_ppt2polish_cli.py --input old_presentation.pptx --style "现代简约风格" --ref-img reference_style.png

Environment Variables:
  DF_API_URL    - Default LLM API URL
  DF_API_KEY    - Default API key
  DF_MODEL      - Default text model name
"""
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Input PPT/PPTX file path"
    )

    # Optional arguments
    parser.add_argument(
        "--api-url",
        help="LLM API URL (default: from env DF_API_URL)"
    )

    parser.add_argument(
        "--api-key",
        help="LLM API key (default: from env DF_API_KEY)"
    )

    parser.add_argument(
        "--image-api-url",
        help="Image generation API URL (default: from env DF_IMAGE_API_URL)"
    )

    parser.add_argument(
        "--image-api-key",
        help="Image generation API key (default: from env DF_IMAGE_API_KEY)"
    )

    parser.add_argument(
        "--model",
        default=settings.PAPER2PPT_CONTENT_MODEL,
        help=f"Text model name (default: {settings.PAPER2PPT_CONTENT_MODEL})"
    )

    parser.add_argument(
        "--gen-fig-model",
        default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
        help=f"Image generation model (default: {settings.PAPER2PPT_IMAGE_GEN_MODEL})"
    )

    parser.add_argument(
        "--style",
        default="现代简约风格",
        help="Target style description (default: 现代简约风格)"
    )

    parser.add_argument(
        "--ref-img",
        help="Reference image for style consistency (optional)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/ppt2polish/{timestamp})"
    )

    parser.add_argument(
        "--language",
        default="zh",
        choices=["zh", "en"],
        help="Output language (default: zh)"
    )

    return parser.parse_args()


def validate_input_file(file_path: str) -> Path:
    """Validate input file exists and has correct extension"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() not in [".pptx", ".ppt"]:
        raise ValueError(f"Invalid file type. Expected .pptx or .ppt, got {path.suffix}")

    return path.resolve()


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "ppt2polish" / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def convert_pptx_to_pdf(pptx_path: Path, output_dir: Path) -> Path:
    """
    Convert PPTX to PDF using LibreOffice

    Returns:
        Path to the generated PDF file
    """
    pdf_path = output_dir / "temp_slides.pdf"

    log.info("Converting PPTX to PDF...")
    log.info("Input: %s", pptx_path)
    log.info("Output: %s", pdf_path)

    try:
        # Try using LibreOffice command line
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(pptx_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")

        # LibreOffice creates PDF with same name as input
        generated_pdf = output_dir / f"{pptx_path.stem}.pdf"
        if generated_pdf.exists():
            generated_pdf.rename(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not created: {pdf_path}")

        log.info("PDF created: %s", pdf_path)
        return pdf_path

    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice not found. Please install LibreOffice:\n"
            "  Ubuntu/Debian: sudo apt-get install libreoffice\n"
            "  macOS: brew install --cask libreoffice\n"
            "  Or use: sudo apt-get install unoconv"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("PPTX to PDF conversion timed out (>5 minutes)")


def convert_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[str]:
    """
    Convert PDF to image sequence using pdf2image

    Returns:
        List of image file paths
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "pdf2image not installed. Please install:\n"
            "  pip install pdf2image\n"
            "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
            "  macOS: brew install poppler"
        )

    log.info("Converting PDF to images...")

    # Create images subdirectory
    images_dir = output_dir / "slide_images"
    images_dir.mkdir(exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(str(pdf_path), dpi=300)

    image_paths = []
    for i, image in enumerate(images):
        image_path = images_dir / f"slide_{i:03d}.png"
        image.save(str(image_path), "PNG")
        image_paths.append(str(image_path))

    log.info("Created %s slide images", len(image_paths))
    return image_paths


async def run_ppt2polish_workflow(args, image_paths: list[str], output_dir: Path) -> Paper2PPTResponse:
    """Execute PPT2Polish workflow via the paper2ppt workflow adapter."""

    api_url, api_key = resolve_cli_text_credentials(args.api_url, args.api_key)
    image_api_url, image_api_key = resolve_cli_image_credentials(
        args.image_api_url,
        args.image_api_key,
        fallback_url=api_url,
        fallback_key=api_key,
    )

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

    # Validate reference image if provided
    ref_img_path = None
    if args.ref_img:
        ref_img_path = Path(args.ref_img)
        if not ref_img_path.exists():
            raise FileNotFoundError(f"Reference image not found: {args.ref_img}")
        ref_img_path = str(ref_img_path.resolve())

    req = Paper2PPTRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        image_api_url=image_api_url,
        image_api_key=image_api_key,
        model=args.model,
        gen_fig_model=args.gen_fig_model,
        credential_scope="ppt2polish",
        style=args.style,
        ref_img=ref_img_path or "",
        language=args.language,
        page_count=len(image_paths),
        input_type="FIGURE",
        email="cli_ppt2polish@paper2any.local",
    )

    pagecontent = [{"ppt_img_path": img_path} for img_path in image_paths]

    log.info("%s", "=" * 60)
    log.info("PPT2Polish Workflow Starting")
    log.info("%s", "=" * 60)
    log.info("Number of Slides: %s", len(image_paths))
    log.info("Output Directory: %s", output_dir)
    log.info("Workflow: paper2ppt via workflow adapter")
    log.info("Style: %s", args.style)
    if ref_img_path:
        log.info("Reference Image: %s", ref_img_path)
    log.info("%s", "=" * 60)

    generate_resp = await run_paper2ppt_wf_api(
        req=req,
        pagecontent=pagecontent,
        result_path=str(output_dir),
        get_down=None,
    )
    if not generate_resp.success:
        raise RuntimeError("PPT2Polish workflow failed")

    finalize_req = req.model_copy(update={"all_edited_down": True})
    final_resp = await run_paper2ppt_wf_api(
        req=finalize_req,
        pagecontent=pagecontent,
        result_path=generate_resp.result_path or str(output_dir),
        get_down=None,
    )
    if not final_resp.success:
        raise RuntimeError("PPT2Polish finalize stage failed")

    artifact_root = Path(final_resp.result_path or generate_resp.result_path or output_dir)
    artifact_candidates = []
    for value in [final_resp.ppt_pptx_path, final_resp.ppt_pdf_path]:
        if value:
            p = Path(value)
            if p.exists():
                artifact_candidates.append(p.resolve())
    artifact_candidates.extend(find_output_artifacts(artifact_root, ("*.pptx", "*.pdf")))
    if not artifact_candidates:
        raise RuntimeError(
            f"PPT2Polish finished without final PPT/PDF artifacts under {artifact_root}"
        )
    return final_resp


def print_results(final_state: Paper2PPTResponse, output_dir: Path):
    """Print workflow results"""
    log.info("%s", "=" * 60)
    log.info("PPT2Polish Workflow Completed Successfully")
    log.info("%s", "=" * 60)
    log.info("Output Directory: %s", output_dir)

    # Check for PPT PDF file
    ppt_pdf_path = final_state.ppt_pdf_path
    if ppt_pdf_path and os.path.exists(ppt_pdf_path):
        log.info("Beautified PPT (PDF): %s", ppt_pdf_path)

    ppt_pptx_path = final_state.ppt_pptx_path
    if ppt_pptx_path and os.path.exists(ppt_pptx_path):
        log.info("Beautified PPT (PPTX): %s", ppt_pptx_path)

    # Check for slide images directory
    ppt_pages_dir = output_dir / "ppt_pages"
    if ppt_pages_dir.exists():
        slide_count = len(list(ppt_pages_dir.glob("page_*.png")))
        log.info("Individual Slides: %s images in %s", slide_count, ppt_pages_dir)

    log.info("%s", "=" * 60)


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()

        # Validate input file
        input_path = validate_input_file(args.input)

        # Create output directory
        output_dir = create_output_dir(args)

        log.info("%s", "=" * 60)
        log.info("PPT2Polish - Step 1: Convert PPTX to Images")
        log.info("%s", "=" * 60)

        # Step 1: Convert PPTX to PDF
        pdf_path = convert_pptx_to_pdf(input_path, output_dir)

        # Step 2: Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path, output_dir)

        log.info("%s", "=" * 60)
        log.info("PPT2Polish - Step 2: Beautify Slides")
        log.info("%s", "=" * 60)

        # Step 3: Run workflow to beautify slides
        final_state = asyncio.run(run_ppt2polish_workflow(args, image_paths, output_dir))

        # Print results
        print_results(final_state, output_dir)

        return 0

    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    except ValueError as e:
        log.error("%s", e)
        return 1
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    except Exception as e:
        log.exception("Workflow execution failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
