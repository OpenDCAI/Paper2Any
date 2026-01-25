#!/usr/bin/env python3
"""
Image2PPT CLI - Convert images to editable PPT

Usage:
    # Basic conversion
    python script/run_image2ppt_cli.py --input screenshot.png

    # With AI enhancement
    python script/run_image2ppt_cli.py --input diagram.jpg --use-ai-edit --api-key sk-xxx
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Image2PPT CLI - Convert images to editable PPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python script/run_image2ppt_cli.py --input screenshot.png

  # With AI enhancement
  python script/run_image2ppt_cli.py --input diagram.jpg --use-ai-edit --api-key sk-xxx

  # Custom style
  python script/run_image2ppt_cli.py --input slide.png --use-ai-edit --style "现代简约风格"

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
        help="Input image file path (PNG/JPG/JPEG)"
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
        "--model",
        default="gpt-4o",
        help="Text model name (default: gpt-4o)"
    )

    parser.add_argument(
        "--gen-fig-model",
        default="gemini-2.5-flash-image-preview",
        help="Image generation model (default: gemini-2.5-flash-image-preview)"
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
        default=1,
        help="Target page count (default: 1)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/image2ppt/{timestamp})"
    )

    return parser.parse_args()


def validate_input_file(file_path: str) -> Path:
    """Validate input file exists and has correct extension"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid file type. Expected {valid_extensions}, got {path.suffix}")

    return path.resolve()


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "image2ppt" / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_image2ppt_workflow(args, input_path: Path, output_dir: Path):
    """Execute Image2PPT workflow"""

    # Get API configuration with priority: CLI args > Environment variables > Defaults
    api_url = args.api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("DF_API_KEY", "")

    # Validate API key if AI edit is enabled
    if args.use_ai_edit and not api_key:
        raise ValueError("API key is required when --use-ai-edit is enabled. "
                        "Provide via --api-key or DF_API_KEY environment variable.")

    # Build request
    req = Paper2FigureRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=args.model,
        gen_fig_model=args.gen_fig_model,
        language=args.language,
        style=args.style,
        page_count=args.page_count,
        use_ai_edit=args.use_ai_edit,
        input_type="FIGURE",  # Key difference: set input type to FIGURE
    )

    # Build state
    state = Paper2FigureState(
        request=req,
        messages=[],
        result_path=str(output_dir),
        input_type="FIGURE",  # Set input type to FIGURE
    )

    # Set image file path (use pdf_file attribute for compatibility)
    state.pdf_file = str(input_path)

    # Select workflow based on use_ai_edit flag (reuses PDF2PPT workflows)
    workflow_name = "pdf2ppt_qwenvl" if args.use_ai_edit else "pdf2ppt_parallel"

    print(f"\n{'='*60}")
    print(f"Image2PPT Workflow Starting")
    print(f"{'='*60}")
    print(f"Input Image: {input_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Workflow: {workflow_name}")
    print(f"AI Enhancement: {'Enabled' if args.use_ai_edit else 'Disabled'}")
    print(f"Style: {args.style}")
    print(f"Language: {args.language}")
    print(f"{'='*60}\n")

    # Run workflow
    final_state = await run_workflow(workflow_name, state)

    return final_state


def print_results(final_state: Paper2FigureState, output_dir: Path):
    """Print workflow results"""
    print(f"\n{'='*60}")
    print(f"✓ Image2PPT Workflow Completed Successfully")
    print(f"{'='*60}")
    print(f"Output Directory: {output_dir}")

    ppt_path = getattr(final_state, "ppt_path", None)
    if ppt_path and os.path.exists(ppt_path):
        print(f"PPT File: {ppt_path}")
    else:
        print("Warning: PPT file not found in output")

    print(f"{'='*60}\n")


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
        final_state = asyncio.run(run_image2ppt_workflow(args, input_path, output_dir))

        # Print results
        print_results(final_state, output_dir)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Workflow execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
