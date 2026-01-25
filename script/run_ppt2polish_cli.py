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

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root


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

    print(f"Converting PPTX to PDF...")
    print(f"Input: {pptx_path}")
    print(f"Output: {pdf_path}")

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

        print(f"✓ PDF created: {pdf_path}\n")
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

    print(f"Converting PDF to images...")

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

    print(f"✓ Created {len(image_paths)} slide images\n")
    return image_paths


async def run_ppt2polish_workflow(args, image_paths: list[str], output_dir: Path):
    """Execute PPT2Polish workflow using paper2ppt_parallel_consistent_style"""

    # Get API configuration
    api_url = args.api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("DF_API_KEY", "")

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

    # Validate reference image if provided
    ref_img_path = None
    if args.ref_img:
        ref_img_path = Path(args.ref_img)
        if not ref_img_path.exists():
            raise FileNotFoundError(f"Reference image not found: {args.ref_img}")
        ref_img_path = str(ref_img_path.resolve())

    # Build request
    req = Paper2FigureRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=args.model,
        gen_fig_model=args.gen_fig_model,
        style=args.style,
        all_edited_down=True,  # Directly generate final result
        ref_img=ref_img_path or "",  # Optional reference image for style consistency
    )

    # Build pagecontent as list of image paths
    # Use dict format with ppt_img_path key for compatibility
    pagecontent = [{"ppt_img_path": img_path} for img_path in image_paths]

    # Build state
    state = Paper2FigureState(
        request=req,
        messages=[],
        result_path=str(output_dir),
        pagecontent=pagecontent,
    )

    print(f"\n{'='*60}")
    print(f"PPT2Polish Workflow Starting")
    print(f"{'='*60}")
    print(f"Number of Slides: {len(image_paths)}")
    print(f"Output Directory: {output_dir}")
    print(f"Workflow: paper2ppt_parallel_consistent_style")
    print(f"Style: {args.style}")
    if ref_img_path:
        print(f"Reference Image: {ref_img_path}")
    print(f"{'='*60}\n")

    # Run workflow
    workflow_name = "paper2ppt_parallel_consistent_style"
    final_state = await run_workflow(workflow_name, state)

    return final_state


def print_results(final_state: Paper2FigureState, output_dir: Path):
    """Print workflow results"""
    print(f"\n{'='*60}")
    print(f"✓ PPT2Polish Workflow Completed Successfully")
    print(f"{'='*60}")
    print(f"Output Directory: {output_dir}")

    # Check for PPT PDF file
    ppt_pdf_path = getattr(final_state, "ppt_pdf_path", None)
    if ppt_pdf_path and os.path.exists(ppt_pdf_path):
        print(f"Beautified PPT (PDF): {ppt_pdf_path}")

    # Check for slide images directory
    ppt_pages_dir = output_dir / "ppt_pages"
    if ppt_pages_dir.exists():
        slide_count = len(list(ppt_pages_dir.glob("page_*.png")))
        print(f"Individual Slides: {slide_count} images in {ppt_pages_dir}")

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

        print(f"\n{'='*60}")
        print(f"PPT2Polish - Step 1: Convert PPTX to Images")
        print(f"{'='*60}\n")

        # Step 1: Convert PPTX to PDF
        pdf_path = convert_pptx_to_pdf(input_path, output_dir)

        # Step 2: Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path, output_dir)

        print(f"{'='*60}")
        print(f"PPT2Polish - Step 2: Beautify Slides")
        print(f"{'='*60}\n")

        # Step 3: Run workflow to beautify slides
        final_state = asyncio.run(run_ppt2polish_workflow(args, image_paths, output_dir))

        # Print results
        print_results(final_state, output_dir)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Workflow execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
