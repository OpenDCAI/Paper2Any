#!/usr/bin/env python3
"""
PDF2PPT CLI - One-click PDF to PPT conversion

Usage:
    # Basic conversion (no AI)
    python script/run_pdf2ppt_cli.py --input slides.pdf

    # With AI enhancement
    python script/run_pdf2ppt_cli.py --input slides.pdf --use-ai-edit --api-key sk-xxx
"""

import argparse
import asyncio
import os
import sys
import time
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
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse
from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api
from dataflow_agent.utils import get_project_root

load_project_env()

log = get_logger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="PDF2PPT CLI - Convert PDF to editable PPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (no AI)
  python script/run_pdf2ppt_cli.py --input slides.pdf

  # With AI enhancement
  python script/run_pdf2ppt_cli.py --input slides.pdf --use-ai-edit --api-key sk-xxx

  # Custom style
  python script/run_pdf2ppt_cli.py --input slides.pdf --use-ai-edit --style "学术风格"

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
        help="Input PDF file path"
    )

    # Optional arguments
    parser.add_argument(
        "--use-ai-edit",
        action="store_true",
        help="Enable AI enhancement (default: False)"
    )

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
        "--ocr-api-url",
        help="OCR/VLM API URL (default: from env PAPER2DRAWIO_OCR_API_URL)"
    )

    parser.add_argument(
        "--ocr-api-key",
        help="OCR/VLM API key (default: from env PAPER2DRAWIO_OCR_API_KEY)"
    )

    parser.add_argument(
        "--model",
        default=settings.PDF2PPT_DEFAULT_MODEL,
        help=f"Text model name (default: {settings.PDF2PPT_DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--gen-fig-model",
        default=settings.PDF2PPT_DEFAULT_IMAGE_MODEL,
        help=f"Image generation model (default: {settings.PDF2PPT_DEFAULT_IMAGE_MODEL})"
    )

    parser.add_argument(
        "--language",
        default="zh",
        choices=["zh", "en"],
        help="Output language (default: zh)"
    )

    parser.add_argument(
        "--style",
        default="现代简约风格",
        help="Style description (default: 现代简约风格)"
    )

    parser.add_argument(
        "--page-count",
        type=int,
        default=8,
        help="Target page count (default: 8)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/pdf2ppt/{timestamp})"
    )

    return parser.parse_args()


def validate_input_file(file_path: str) -> Path:
    """Validate input file exists and has correct extension"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file type. Expected .pdf, got {path.suffix}")

    return path.resolve()


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "pdf2ppt" / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_pdf2ppt_workflow(args, input_path: Path, output_dir: Path) -> Paper2PPTResponse:
    """Execute PDF2PPT workflow via the backend workflow adapter."""

    if args.ocr_api_url:
        os.environ["PAPER2DRAWIO_OCR_API_URL"] = args.ocr_api_url
    if args.ocr_api_key:
        os.environ["PAPER2DRAWIO_OCR_API_KEY"] = args.ocr_api_key

    api_url, api_key = resolve_cli_text_credentials(args.api_url, args.api_key)
    image_api_url, image_api_key = resolve_cli_image_credentials(
        args.image_api_url,
        args.image_api_key,
        fallback_url=api_url,
        fallback_key=api_key,
    )

    if args.use_ai_edit and not image_api_key:
        raise ValueError(
            "Image API key is required when --use-ai-edit is enabled. "
            "Provide via --image-api-key/DF_IMAGE_API_KEY or reuse a compatible DF_API_KEY."
        )

    req = Paper2PPTRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        image_api_url=image_api_url,
        image_api_key=image_api_key,
        model=args.model,
        gen_fig_model=args.gen_fig_model,
        language=args.language,
        style=args.style,
        page_count=args.page_count,
        input_type="PDF",
        input_content=str(input_path),
        email="cli_pdf2ppt@paper2any.local",
        credential_scope="pdf2ppt",
        use_ai_edit=args.use_ai_edit,
    )

    log.info("%s", "=" * 60)
    log.info("PDF2PPT Workflow Starting")
    log.info("%s", "=" * 60)
    log.info("Input PDF: %s", input_path)
    log.info("Output Directory: %s", output_dir)
    log.info("Workflow: pdf2ppt_qwenvl via workflow adapter")
    log.info("AI Enhancement: %s", "Enabled" if args.use_ai_edit else "Disabled")
    log.info("Style: %s", args.style)
    log.info("Language: %s", args.language)
    log.info("%s", "=" * 60)

    final_resp = await run_pdf2ppt_wf_api(req, result_path=output_dir)
    if not final_resp.success:
        raise RuntimeError("PDF2PPT workflow failed")

    artifact_root = Path(final_resp.result_path or output_dir)
    artifact_candidates = []
    for value in [final_resp.ppt_pptx_path, final_resp.ppt_pdf_path]:
        if value:
            p = Path(value)
            if p.exists():
                artifact_candidates.append(p.resolve())
    artifact_candidates.extend(find_output_artifacts(artifact_root, ("*.pptx", "*.pdf")))
    if not artifact_candidates:
        raise RuntimeError(
            f"PDF2PPT workflow finished without final PPT/PDF artifacts under {artifact_root}"
        )
    return final_resp


def print_results(final_state: Paper2PPTResponse, output_dir: Path):
    """Print workflow results"""
    log.info("%s", "=" * 60)
    log.info("PDF2PPT Workflow Completed Successfully")
    log.info("%s", "=" * 60)
    log.info("Output Directory: %s", output_dir)

    if final_state.ppt_pptx_path and Path(final_state.ppt_pptx_path).exists():
        log.info("PPT File: %s", final_state.ppt_pptx_path)
    if final_state.ppt_pdf_path and Path(final_state.ppt_pdf_path).exists():
        log.info("PDF File: %s", final_state.ppt_pdf_path)

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

        # Run workflow
        final_state = asyncio.run(run_pdf2ppt_workflow(args, input_path, output_dir))

        # Print results
        print_results(final_state, output_dir)

        return 0

    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    except ValueError as e:
        log.error("%s", e)
        return 1
    except Exception as e:
        log.exception("Workflow execution failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
