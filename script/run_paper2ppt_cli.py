#!/usr/bin/env python3
"""
Paper2PPT CLI - Convert papers to PPT presentations

Usage:
    # Basic usage
    python script/run_paper2ppt_cli.py --input paper.pdf --api-key sk-xxx --page-count 15

    # With custom style
    python script/run_paper2ppt_cli.py --input paper.pdf --style "北京大学风格；英文；学术风格" --language zh
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
from dataflow_agent.utils import get_project_root
from fastapi_app.config import settings
from fastapi_app.schemas import Paper2PPTRequest, Paper2PPTResponse
from fastapi_app.workflow_adapters.wa_paper2ppt import (
    run_paper2page_content_wf_api,
    run_paper2ppt_wf_api,
)

load_project_env()

log = get_logger(__name__)


def _state_get(state, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper2PPT CLI - Convert papers to PPT presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script/run_paper2ppt_cli.py --input paper.pdf --api-key sk-xxx --page-count 15

  # With custom style
  python script/run_paper2ppt_cli.py --input paper.pdf --style "北京大学风格；英文；学术风格" --language zh

  # Long paper mode
  python script/run_paper2ppt_cli.py --input long_paper.pdf --use-long-paper --page-count 60

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
        help="Input file (PDF/PPTX) or text content"
    )

    # Optional arguments
    parser.add_argument(
        "--input-type",
        choices=["PDF", "TEXT", "TOPIC", "PPTX"],
        help="Input type (auto-detect if not specified)"
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
        "--model",
        default=settings.PAPER2PPT_OUTLINE_MODEL,
        help=f"Text/outline model name (default: {settings.PAPER2PPT_OUTLINE_MODEL})"
    )

    parser.add_argument(
        "--gen-fig-model",
        default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
        help=f"Image generation model (default: {settings.PAPER2PPT_IMAGE_GEN_MODEL})"
    )

    parser.add_argument(
        "--language",
        default="zh",
        choices=["zh", "en"],
        help="Output language (default: zh)"
    )

    parser.add_argument(
        "--style",
        default="",
        help="PPT style description (default: empty)"
    )

    parser.add_argument(
        "--page-count",
        type=int,
        default=10,
        help="Target page count (default: 10)"
    )

    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
        help="Aspect ratio (default: 16:9)"
    )

    parser.add_argument(
        "--use-long-paper",
        action="store_true",
        help="Use long paper workflow (for >50 pages)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/paper2ppt/{timestamp})"
    )

    return parser.parse_args()


def detect_input_type(input_str: str) -> str:
    """Auto-detect input type from file extension or content"""
    path = Path(input_str)

    # If file doesn't exist, assume it's text
    if not path.exists():
        return "TEXT"

    ext = path.suffix.lower()
    if ext == ".pdf":
        return "PDF"
    elif ext in [".pptx", ".ppt"]:
        return "PPTX"
    else:
        return "TEXT"


def validate_input(input_str: str, input_type: str) -> tuple[str, str]:
    """
    Validate input and return (input_content, resolved_input_type)
    """
    if input_type in ["PDF", "PPTX"]:
        path = Path(input_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_str}")

        # Validate extension
        ext = path.suffix.lower()
        if input_type == "PDF" and ext != ".pdf":
            raise ValueError(f"Expected PDF file, got {ext}")
        elif input_type == "PPTX" and ext not in [".pptx", ".ppt"]:
            raise ValueError(f"Expected PPTX file, got {ext}")

        return str(path.resolve()), input_type
    else:
        # TEXT or TOPIC input
        return input_str, input_type


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "paper2ppt" / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_paper2ppt_workflow(args, input_content: str, input_type: str, output_dir: Path) -> Paper2PPTResponse:
    """Execute Paper2PPT workflow via the same adapters used by the backend API."""

    # Get API configuration
    api_url, api_key = resolve_cli_text_credentials(args.api_url, args.api_key)
    image_api_url, image_api_key = resolve_cli_image_credentials(
        args.image_api_url,
        args.image_api_key,
        fallback_url=api_url,
        fallback_key=api_key,
    )

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

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
        input_type=input_type,
        input_content=input_content,
        aspect_ratio=args.aspect_ratio,
        use_long_paper=bool(args.use_long_paper),
        email="cli_paper2ppt@paper2any.local",
        credential_scope="paper2ppt",
    )

    log.info("%s", "=" * 60)
    log.info("Paper2PPT Workflow Starting (2-Step Process)")
    log.info("%s", "=" * 60)
    log.info("Input Type: %s", input_type)
    if input_type in ["PDF", "PPTX"]:
        log.info("Input File: %s", input_content)
    else:
        log.info("Input Text: %s", f"{input_content[:100]}..." if len(input_content) > 100 else input_content)
    log.info("Output Directory: %s", output_dir)
    log.info("Style: %s", args.style)
    log.info("Page Count: %s", args.page_count)
    log.info("Language: %s", args.language)
    log.info("Aspect Ratio: %s", args.aspect_ratio)
    log.info("%s", "=" * 60)

    # Step 1: Generate page content (outline)
    log.info("Step 1/2: Generating page content outline...")
    pagecontent_resp = await run_paper2page_content_wf_api(req, result_path=output_dir)
    if not pagecontent_resp.success:
        raise RuntimeError("Paper2PPT page-content stage failed")

    log.info("Step 1 completed: Page content generated")
    pagecontent_len = len(pagecontent_resp.pagecontent or [])
    log.info("Generated %s pages", pagecontent_len)

    # Step 2: Generate PPT from page content
    log.info("Step 2/3: Generating PPT slides...")
    generate_resp = await run_paper2ppt_wf_api(
        req=req,
        pagecontent=pagecontent_resp.pagecontent or [],
        result_path=pagecontent_resp.result_path or str(output_dir),
        get_down=None,
    )
    if not generate_resp.success:
        raise RuntimeError("Paper2PPT generation stage failed")

    # Step 3: Finalize export (PPTX/PDF)
    log.info("Step 3/3: Finalizing PPT/PDF export...")
    finalize_req = req.model_copy(update={"all_edited_down": True})
    final_resp = await run_paper2ppt_wf_api(
        req=finalize_req,
        pagecontent=pagecontent_resp.pagecontent or [],
        result_path=generate_resp.result_path or pagecontent_resp.result_path or str(output_dir),
        get_down=None,
    )
    if not final_resp.success:
        raise RuntimeError("Paper2PPT finalize stage failed")

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
            f"Paper2PPT workflow finished without final PPT/PDF artifacts under {artifact_root}"
        )

    log.info("Step 2 completed: PPT generated")
    return final_resp


def print_results(final_state: Paper2PPTResponse, output_dir: Path):
    """Print workflow results"""
    log.info("%s", "=" * 60)
    log.info("Paper2PPT Workflow Completed Successfully")
    log.info("%s", "=" * 60)
    log.info("Output Directory: %s", output_dir)

    # Check for PPT PDF file
    ppt_pdf_path = final_state.ppt_pdf_path
    if ppt_pdf_path and os.path.exists(ppt_pdf_path):
        log.info("PPT PDF File: %s", ppt_pdf_path)

    # Check for editable PPTX file
    ppt_pptx_path = final_state.ppt_pptx_path
    if ppt_pptx_path and os.path.exists(ppt_pptx_path):
        log.info("PPT PPTX File: %s", ppt_pptx_path)

    # Check for page content JSON
    pagecontent = final_state.pagecontent
    if pagecontent:
        log.info("Page Content: %s pages generated", len(pagecontent))

    log.info("%s", "=" * 60)


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()

        # Auto-detect input type if not specified
        input_type = args.input_type or detect_input_type(args.input)

        # Validate input
        input_content, input_type = validate_input(args.input, input_type)

        # Create output directory
        output_dir = create_output_dir(args)

        # Run workflow
        final_state = asyncio.run(run_paper2ppt_workflow(args, input_content, input_type, output_dir))

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
