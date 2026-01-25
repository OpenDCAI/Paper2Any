#!/usr/bin/env python3
"""
Paper2Figure CLI - Generate scientific figures from papers

Usage:
    # Generate model architecture from PDF
    python script/run_paper2figure_cli.py --input paper.pdf --graph-type model_arch --api-key sk-xxx

    # Generate tech route from text
    python script/run_paper2figure_cli.py --input "Transformer architecture" --input-type TEXT --graph-type tech_route
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


# Workflow mapping for different graph types
WORKFLOW_MAP = {
    "model_arch": "paper2fig_image_only",
    "tech_route": "paper2technical",
    "exp_data": "paper2expfigure",
}


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper2Figure CLI - Generate scientific figures from papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate model architecture from PDF
  python script/run_paper2figure_cli.py --input paper.pdf --graph-type model_arch --api-key sk-xxx

  # Generate tech route from text
  python script/run_paper2figure_cli.py --input "Transformer architecture" --input-type TEXT --graph-type tech_route

  # Generate experimental data figure
  python script/run_paper2figure_cli.py --input paper.pdf --graph-type exp_data

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
        help="Input file (PDF/image) or text content"
    )

    parser.add_argument(
        "--graph-type",
        required=True,
        choices=["model_arch", "tech_route", "exp_data"],
        help="Type of graph to generate"
    )

    # Optional arguments
    parser.add_argument(
        "--input-type",
        choices=["PDF", "TEXT", "FIGURE"],
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
        default="gpt-4o",
        help="Text model name (default: gpt-4o)"
    )

    parser.add_argument(
        "--gen-fig-model",
        default="gemini-2.5-flash-image-preview",
        help="Image generation model (default: gemini-2.5-flash-image-preview)"
    )

    parser.add_argument(
        "--style",
        default="cartoon",
        choices=["cartoon", "realistic"],
        help="Figure style (default: cartoon)"
    )

    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
        help="Output aspect ratio (default: 16:9)"
    )

    parser.add_argument(
        "--complexity",
        default="easy",
        choices=["easy", "mid", "hard"],
        help="Complexity level for model_arch (default: easy)"
    )

    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "zh"],
        help="Output language (default: en)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/paper2figure/{graph_type}/{timestamp})"
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
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        return "FIGURE"
    else:
        return "TEXT"


def validate_input(input_str: str, input_type: str) -> tuple[str, str]:
    """
    Validate input and return (input_content, resolved_input_type)

    Returns:
        - For files: (absolute_path, input_type)
        - For text: (text_content, "TEXT")
    """
    if input_type in ["PDF", "FIGURE"]:
        path = Path(input_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_str}")

        # Validate extension
        ext = path.suffix.lower()
        if input_type == "PDF" and ext != ".pdf":
            raise ValueError(f"Expected PDF file, got {ext}")
        elif input_type == "FIGURE" and ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            raise ValueError(f"Expected image file, got {ext}")

        return str(path.resolve()), input_type
    else:
        # TEXT input
        return input_str, "TEXT"


def create_output_dir(args) -> Path:
    """Create timestamped output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "paper2figure" / args.graph_type / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_paper2figure_workflow(args, input_content: str, input_type: str, output_dir: Path):
    """Execute Paper2Figure workflow"""

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
        figure_complex=args.complexity,
        input_type=input_type,
    )

    # Build state
    state = Paper2FigureState(
        request=req,
        messages=[],
        result_path=str(output_dir),
        aspect_ratio=args.aspect_ratio,
        input_type=input_type,
    )

    # Set input based on type
    if input_type == "PDF":
        state.paper_file = input_content
    elif input_type == "TEXT":
        state.paper_idea = input_content
    elif input_type == "FIGURE":
        state.fig_draft_path = input_content

    # Select workflow based on graph type
    workflow_name = WORKFLOW_MAP[args.graph_type]

    print(f"\n{'='*60}")
    print(f"Paper2Figure Workflow Starting")
    print(f"{'='*60}")
    print(f"Graph Type: {args.graph_type}")
    print(f"Input Type: {input_type}")
    if input_type in ["PDF", "FIGURE"]:
        print(f"Input File: {input_content}")
    else:
        print(f"Input Text: {input_content[:100]}..." if len(input_content) > 100 else f"Input Text: {input_content}")
    print(f"Output Directory: {output_dir}")
    print(f"Workflow: {workflow_name}")
    print(f"Style: {args.style}")
    print(f"Aspect Ratio: {args.aspect_ratio}")
    if args.graph_type == "model_arch":
        print(f"Complexity: {args.complexity}")
    print(f"Language: {args.language}")
    print(f"{'='*60}\n")

    # Run workflow
    final_state = await run_workflow(workflow_name, state)

    return final_state


def print_results(final_state: Paper2FigureState, output_dir: Path, graph_type: str):
    """Print workflow results"""
    print(f"\n{'='*60}")
    print(f"✓ Paper2Figure Workflow Completed Successfully")
    print(f"{'='*60}")
    print(f"Output Directory: {output_dir}")

    # Check for PPT file
    ppt_path = getattr(final_state, "ppt_path", None)
    if ppt_path and os.path.exists(ppt_path):
        print(f"PPT File: {ppt_path}")

    # Check for SVG files (tech_route and exp_data)
    if graph_type in ["tech_route", "exp_data"]:
        svg_file_path = getattr(final_state, "svg_file_path", None)
        if svg_file_path and os.path.exists(svg_file_path):
            print(f"SVG File: {svg_file_path}")

        svg_full_img_path = getattr(final_state, "svg_full_img_path", None)
        if svg_full_img_path and os.path.exists(svg_full_img_path):
            print(f"SVG Image: {svg_full_img_path}")

    # Check for figure draft (model_arch)
    if graph_type == "model_arch":
        fig_draft_path = getattr(final_state, "fig_draft_path", None)
        if fig_draft_path and os.path.exists(fig_draft_path):
            print(f"Figure Draft: {fig_draft_path}")

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
        final_state = asyncio.run(run_paper2figure_workflow(args, input_content, input_type, output_dir))

        # Print results
        print_results(final_state, output_dir, args.graph_type)

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
