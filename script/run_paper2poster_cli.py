#!/usr/bin/env python3
"""
Paper2Poster CLI - Convert papers to aesthetic conference posters

Usage:
    # Basic usage
    python script/run_paper2poster_cli.py --input paper.pdf --api-key sk-xxx

    # With custom dimensions and logos
    python script/run_paper2poster_cli.py --input paper.pdf --poster-width 54 --poster-height 36 --logo logo.png --aff-logo aff.png
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2PosterState, Paper2PosterRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper2Poster CLI - Convert papers to aesthetic conference posters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script/run_paper2poster_cli.py --input paper.pdf --api-key sk-xxx

  # With custom dimensions
  python script/run_paper2poster_cli.py --input paper.pdf --poster-width 54 --poster-height 36

  # With logos
  python script/run_paper2poster_cli.py --input paper.pdf --logo logo.png --aff-logo aff.png

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
        help="Input PDF paper file"
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
        "--model",
        default="gpt-4o-2024-08-06",
        help="Text model name (default: gpt-4o-2024-08-06)"
    )

    parser.add_argument(
        "--vision-model",
        default="gpt-4o-2024-08-06",
        help="Vision model name (default: gpt-4o-2024-08-06)"
    )

    parser.add_argument(
        "--poster-width",
        type=float,
        default=54.0,
        help="Poster width in inches (default: 54.0)"
    )

    parser.add_argument(
        "--poster-height",
        type=float,
        default=36.0,
        help="Poster height in inches (default: 36.0)"
    )

    parser.add_argument(
        "--logo",
        default="",
        help="Path to conference/journal logo"
    )

    parser.add_argument(
        "--aff-logo",
        default="",
        help="Path to affiliation logo (for color extraction)"
    )

    parser.add_argument(
        "--url",
        default="",
        help="URL for QR code on poster (optional)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/paper2poster/{timestamp})"
    )

    return parser.parse_args()


def validate_input(input_path: str) -> str:
    """
    Validate input PDF file and return absolute path
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = path.suffix.lower()
    if ext != ".pdf":
        raise ValueError(f"Expected PDF file, got {ext}")

    return str(path.resolve())


def validate_poster_dimensions(width: float, height: float) -> tuple[float, float]:
    """
    Validate poster dimensions and aspect ratio
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Poster dimensions must be positive: {width}x{height}")

    ratio = width / height
    # Check poster ratio: lower bound 1.4 (ISO A paper size), upper bound 2 (human vision limit)
    if ratio > 2.0 or ratio < 1.4:
        raise ValueError(
            f"Poster aspect ratio {ratio:.2f} is out of range. "
            f"Please use a ratio between 1.4 and 2.0 (width/height)"
        )

    return width, height


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        run_id = f"{int(time.time())}-{uuid4().hex[:8]}"
        output_dir = project_root / "outputs" / "cli" / "paper2poster" / run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_paper2poster_workflow(args, input_path: str, output_dir: Path):
    """Execute Paper2Poster workflow"""

    # Get API configuration
    api_url = args.api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("DF_API_KEY", "")

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

    # Build request
    req = Paper2PosterRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=args.model,
        vision_model=args.vision_model,
        poster_width=args.poster_width,
        poster_height=args.poster_height,
        logo_path=args.logo,
        aff_logo_path=args.aff_logo,
        url=args.url,
    )

    # Build state
    state = Paper2PosterState(
        request=req,
        messages=[],
        agent_results={},
        result_path=str(output_dir),
        paper_file=input_path,
        poster_width=args.poster_width,
        poster_height=args.poster_height,
        logo_path=args.logo,
        aff_logo_path=args.aff_logo,
        url=args.url,
    )

    log.info("%s", "=" * 60)
    log.info("Paper2Poster Workflow Starting")
    log.info("%s", "=" * 60)
    log.info("Input PDF: %s", input_path)
    log.info("Output Directory: %s", output_dir)
    log.info("Poster Dimensions: %sx%s inches", args.poster_width, args.poster_height)
    log.info("Aspect Ratio: %.2f", args.poster_width / args.poster_height)
    log.info("Text Model: %s", args.model)
    log.info("Vision Model: %s", args.vision_model)
    if args.logo:
        log.info("Logo: %s", args.logo)
    if args.aff_logo:
        log.info("Affiliation Logo: %s", args.aff_logo)
    log.info("%s", "=" * 60)

    # Run workflow
    log.info("Executing Paper2Poster workflow...")
    log.info("This may take several minutes...")

    state = await run_workflow("paper2poster", state)

    log.info("Paper2Poster workflow completed")

    return state


def print_results(final_state: Paper2PosterState, output_dir: Path):
    """Print workflow results"""
    log.info("%s", "=" * 60)
    log.info("Paper2Poster Workflow Completed Successfully")
    log.info("%s", "=" * 60)
    log.info("Output Directory: %s", output_dir)

    # Check for PPTX file
    pptx_path = getattr(final_state, "output_pptx_path", None)
    if pptx_path and os.path.exists(pptx_path):
        log.info("PPTX File: %s", pptx_path)

    # Check for PNG file
    png_path = getattr(final_state, "output_png_path", None)
    if png_path and os.path.exists(png_path):
        log.info("PNG File: %s", png_path)

    # Check for errors
    errors = getattr(final_state, "errors", [])
    if errors:
        log.warning("Warnings/Errors:")
        for error in errors:
            log.warning("  - %s", error)

    log.info("%s", "=" * 60)


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()

        # Validate input
        input_path = validate_input(args.input)

        # Validate poster dimensions
        width, height = validate_poster_dimensions(args.poster_width, args.poster_height)

        # Create output directory
        output_dir = create_output_dir(args)

        # Run workflow
        final_state = asyncio.run(run_paper2poster_workflow(args, input_path, output_dir))

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
