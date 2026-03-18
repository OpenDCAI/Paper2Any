#!/usr/bin/env python3
"""
Paper2Video CLI - Convert papers to video

Usage:
    # Basic usage
    python script/run_paper2video_cli.py --input paper.pdf --api-key sk-xxx

    # With custom language / TTS model
    python script/run_paper2video_cli.py --input paper.pdf --language zh --tts-model cosyvoice-v3-flash

    # With reference image and audio (talking head / voice clone)
    python script/run_paper2video_cli.py --input paper.pdf --ref-img avatar.png --ref-audio voice.wav --ref-text "reference script"
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


# Snapshot keys for passing state from step1 to step2 (same as wa_paper2video)
_STATE_SNAPSHOT_KEYS = [
    "result_path",
    "ppt_path",
    "slide_timesteps_path",
    "slide_img_dir",
    "subtitle_and_cursor",
    "subtitle_and_cursor_path",
    "speech_save_dir",
    "cursor_save_path",
    "talking_video_save_dir",
]


def _state_to_snapshot(state: Paper2VideoState | dict) -> dict:
    """Serialize step1 state to dict for step2 (no script_pages)."""
    if isinstance(state, dict):
        req = state.get("request")
        snapshot = {
            "request": asdict(req) if req is not None and hasattr(req, "__dataclass_fields__") else (req if isinstance(req, dict) else {}),
        }
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = state.get(key)
    else:
        snapshot = {"request": asdict(state.request)}
        for key in _STATE_SNAPSHOT_KEYS:
            snapshot[key] = getattr(state, key, None)
    return snapshot


def _state_from_snapshot(snapshot: dict, script_pages: List[dict]) -> Paper2VideoState:
    """Restore state from snapshot and set script_pages; request.script_stage = False."""
    req_dict = snapshot.get("request") or {}
    req_dict = dict(req_dict)
    req_dict["script_stage"] = False
    request = Paper2VideoRequest(**req_dict)
    state = Paper2VideoState(request=request, messages=[])
    for key in _STATE_SNAPSHOT_KEYS:
        if key in snapshot:
            setattr(state, key, snapshot[key])
    state.script_pages = script_pages
    return state


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper2Video CLI - Convert papers to video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script/run_paper2video_cli.py --input paper.pdf --api-key sk-xxx

  # With custom language / TTS model
  python script/run_paper2video_cli.py --input paper.pdf --language zh --tts-model cosyvoice-v3-flash

  # With reference image and audio (talking head)
  python script/run_paper2video_cli.py --input paper.pdf --ref-img avatar.png --ref-audio voice.wav --ref-text "script"

Environment Variables:
  DF_API_URL    - Default LLM API URL
  DF_API_KEY   - Default API key
  DF_TTS_MODEL - Default TTS model name
"""
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input PDF file path",
    )

    parser.add_argument(
        "--api-url",
        help="LLM API URL (default: from env DF_API_URL)",
    )

    parser.add_argument(
        "--api-key",
        help="LLM API key (default: from env DF_API_KEY)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Text/LLM model name (default: gpt-4o)",
    )

    parser.add_argument(
        "--tts-model",
        default="cosyvoice-v3-flash",
        help="TTS model name (default: cosyvoice-v3-flash)",
    )

    parser.add_argument(
        "--language",
        default="en",
        choices=["zh", "en"],
        help="Output language (default: en)",
    )

    parser.add_argument(
        "--ref-img",
        default="",
        help="Reference portrait image path for talking head (optional)",
    )

    parser.add_argument(
        "--ref-audio",
        default="",
        help="Reference audio path for voice clone (optional)",
    )

    parser.add_argument(
        "--ref-text",
        default="",
        help="Reference script text for ref-audio (optional)",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (default: outputs/cli/paper2video/{timestamp})",
    )

    return parser.parse_args()


def validate_input(input_str: str) -> str:
    """Validate input PDF and return resolved path."""
    path = Path(input_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_str}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got {path.suffix}")
    return str(path.resolve())


def create_output_dir(args) -> Path:
    """Create timestamped output directory."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = get_project_root()
        timestamp = int(time.time())
        output_dir = project_root / "outputs" / "cli" / "paper2video" / str(timestamp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_paper2video_workflow(args, paper_pdf_path: str, output_dir: Path):
    """Execute Paper2Video workflow (2-step: generate subtitle, then generate video)."""

    api_url = args.api_url or os.getenv("DF_API_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.getenv("DF_API_KEY", "")

    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

    req = Paper2VideoRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=args.model,
        tts_model=args.tts_model,
        language=args.language,
        paper_pdf_path=paper_pdf_path,
        ref_img_path=args.ref_img or "",
        ref_audio_path=args.ref_audio or "",
        ref_text=args.ref_text or "",
        script_stage=True,
    )

    state = Paper2VideoState(
        request=req,
        messages=[],
        result_path=str(output_dir),
    )

    log.info("%s", "=" * 60)
    log.info("Paper2Video Workflow Starting (2-Step Process)")
    log.info("%s", "=" * 60)
    log.info("Input PDF: %s", paper_pdf_path)
    log.info("Output Directory: %s", output_dir)
    log.info("Language: %s", args.language)
    log.info("TTS Model: %s", args.tts_model)
    if args.ref_img:
        log.info("Ref Image: %s", args.ref_img)
    if args.ref_audio:
        log.info("Ref Audio: %s", args.ref_audio)
    log.info("%s", "=" * 60)

    # Step 1: Generate subtitle / script_pages
    log.info("Step 1/2: Generating subtitle and script pages...")
    log.info("Workflow: paper2video (script_stage=True)")

    state = await run_workflow("paper2video", state)

    script_pages = getattr(state, "script_pages", None) or (state.get("script_pages") if isinstance(state, dict) else [])
    if not isinstance(script_pages, list):
        script_pages = []
    if not script_pages:
        raise ValueError("Step 1 did not produce script_pages. Check PDF and workflow.")

    log.info("Step 1 completed: %s script page(s) generated", len(script_pages))

    # Step 2: Generate video from script_pages
    snapshot = _state_to_snapshot(state)
    state2 = _state_from_snapshot(snapshot, script_pages)
    state2.result_path = str(output_dir)

    log.info("Step 2/2: Generating video...")
    log.info("Workflow: paper2video (script_stage=False)")

    final_state = await run_workflow("paper2video", state2)

    log.info("Step 2 completed: Video generated")

    return final_state


def print_results(final_state: Any, output_dir: Path):
    """Print workflow results."""
    log.info("%s", "=" * 60)
    log.info("Paper2Video Workflow Completed Successfully")
    log.info("%s", "=" * 60)
    log.info("Output Directory: %s", output_dir)

    video_path = getattr(final_state, "video_path", None)
    if isinstance(final_state, dict):
        video_path = video_path or final_state.get("video_path")
    if video_path and os.path.exists(str(video_path)):
        log.info("Video File: %s", video_path)

    script_pages = getattr(final_state, "script_pages", None)
    if isinstance(final_state, dict):
        script_pages = script_pages or final_state.get("script_pages")
    if script_pages:
        log.info("Script Pages: %s page(s)", len(script_pages))

    log.info("%s", "=" * 60)


def main():
    """Main entry point."""
    try:
        args = parse_args()
        paper_pdf_path = validate_input(args.input)
        output_dir = create_output_dir(args)

        final_state = asyncio.run(run_paper2video_workflow(args, paper_pdf_path, output_dir))

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
