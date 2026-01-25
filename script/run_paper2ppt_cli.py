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

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


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
        "--model",
        default="gpt-5.1",
        help="Text model name (default: gpt-5.1)"
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


async def run_paper2ppt_workflow(args, input_content: str, input_type: str, output_dir: Path):
    """Execute Paper2PPT workflow (2-step process)"""

    # Get API configuration
    api_url = args.api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("DF_API_KEY", "")

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

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
        input_type=input_type,
        all_edited_down=True,  # Directly generate final PPT
    )

    # Build state
    state = Paper2FigureState(
        request=req,
        messages=[],
        agent_results={},
        result_path=str(output_dir),
        aspect_ratio=args.aspect_ratio,
    )

    # Set input based on type
    if input_type == "PDF":
        state.paper_file = input_content
    else:
        # For TEXT, TOPIC, or PPTX
        state.paper_file = input_content

    print(f"\n{'='*60}")
    print(f"Paper2PPT Workflow Starting (2-Step Process)")
    print(f"{'='*60}")
    print(f"Input Type: {input_type}")
    if input_type in ["PDF", "PPTX"]:
        print(f"Input File: {input_content}")
    else:
        print(f"Input Text: {input_content[:100]}..." if len(input_content) > 100 else f"Input Text: {input_content}")
    print(f"Output Directory: {output_dir}")
    print(f"Style: {args.style}")
    print(f"Page Count: {args.page_count}")
    print(f"Language: {args.language}")
    print(f"Aspect Ratio: {args.aspect_ratio}")
    print(f"{'='*60}\n")

    # Step 1: Generate page content (outline)
    workflow_name_step1 = "paper2page_content_for_long_paper" if args.use_long_paper else "paper2page_content"
    print(f"Step 1/2: Generating page content outline...")
    print(f"Workflow: {workflow_name_step1}\n")

    state = await run_workflow(workflow_name_step1, state)

    print(f"\n✓ Step 1 completed: Page content generated")
    pagecontent_len = len(getattr(state, "pagecontent", []) or [])
    print(f"Generated {pagecontent_len} pages\n")

    # Step 2: Generate PPT from page content
    workflow_name_step2 = "paper2ppt_parallel_consistent_style"
    print(f"Step 2/2: Generating PPT slides...")
    print(f"Workflow: {workflow_name_step2}\n")

    state = await run_workflow(workflow_name_step2, state)

    print(f"\n✓ Step 2 completed: PPT generated")

    return state


def print_results(final_state: Paper2FigureState, output_dir: Path):
    """Print workflow results"""
    print(f"\n{'='*60}")
    print(f"✓ Paper2PPT Workflow Completed Successfully")
    print(f"{'='*60}")
    print(f"Output Directory: {output_dir}")

    # Check for PPT PDF file
    ppt_pdf_path = getattr(final_state, "ppt_pdf_path", None)
    if ppt_pdf_path and os.path.exists(ppt_pdf_path):
        print(f"PPT PDF File: {ppt_pdf_path}")

    # Check for editable PPTX file
    ppt_pptx_path = getattr(final_state, "ppt_pptx_path", None)
    if ppt_pptx_path and os.path.exists(ppt_pptx_path):
        print(f"PPT PPTX File: {ppt_pptx_path}")

    # Check for page content JSON
    pagecontent = getattr(final_state, "pagecontent", None)
    if pagecontent:
        print(f"Page Content: {len(pagecontent)} pages generated")

    print(f"{'='*60}\n")


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
