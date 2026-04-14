#!/usr/bin/env python3
"""
Paper2PPT Frontend Editable CLI - generate structured editable slides for testing.

Usage:
    python script/run_paper2ppt_frontend_cli.py --input paper.pdf --page-count 20
    python script/run_paper2ppt_frontend_cli.py --input long_paper.pdf --page-count 50 --use-long-paper
    python script/run_paper2ppt_frontend_cli.py --input "LLM agents" --input-type TOPIC --page-count 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from script.cli_env import load_project_env
from dataflow_agent.logger import get_logger
from fastapi_app.config import settings

load_project_env()

log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper2PPT frontend editable CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/run_paper2ppt_frontend_cli.py --input paper.pdf --page-count 12
  python script/run_paper2ppt_frontend_cli.py --input long_paper.pdf --page-count 50 --use-long-paper
  python script/run_paper2ppt_frontend_cli.py --input "Multi-agent systems" --input-type TOPIC --page-count 8
""",
    )
    parser.add_argument("--input", required=True, help="Input PDF path, PPTX path, topic, or text")
    parser.add_argument(
        "--input-type",
        choices=["PDF", "PPTX", "TEXT", "TOPIC"],
        help="Input type (auto-detect if omitted)",
    )
    parser.add_argument("--api-url", help="LLM API URL (default: from env / managed config)")
    parser.add_argument("--api-key", help="LLM API key (default: from env / managed config)")
    parser.add_argument("--credential-scope", default="paper2ppt", help="Managed credential scope")
    parser.add_argument("--email", default="cli_frontend_test@paper2any.local", help="Logical user/email for result path")
    parser.add_argument("--outline-model", default=settings.PAPER2PPT_OUTLINE_MODEL, help="Outline model")
    parser.add_argument("--frontend-model", default=settings.PAPER2PPT_CONTENT_MODEL, help="Frontend editable model")
    parser.add_argument("--image-model", default=settings.PAPER2PPT_IMAGE_GEN_MODEL, help="Image model for include-images mode")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Output language")
    parser.add_argument("--style", default="", help="Style prompt")
    parser.add_argument("--page-count", type=int, default=8, help="Target page count")
    parser.add_argument("--use-long-paper", action="store_true", help="Force long-paper outline workflow")
    parser.add_argument("--include-images", action="store_true", help="Generate/reuse supporting images for editable slides")
    parser.add_argument("--image-style", default="academic_illustration", help="Image style prompt")
    parser.add_argument("--export-pptx", action="store_true", help="Also export editable PPTX using the frontend PptxGenJS exporter")
    parser.add_argument("--asset-base-url", default="http://127.0.0.1:8000", help="Base URL for /outputs assets during PPTX export")
    parser.add_argument("--output-dir", help="Output directory (default: outputs/cli/paper2ppt_frontend/{timestamp})")
    return parser.parse_args()


def detect_input_type(input_str: str) -> str:
    path = Path(input_str)
    if not path.exists():
        return "TEXT"
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "PDF"
    if ext in {".pptx", ".ppt"}:
        return "PPTX"
    return "TEXT"


def validate_input(input_str: str, input_type: str) -> tuple[str, str]:
    if input_type in {"PDF", "PPTX"}:
        path = Path(input_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_str}")
        ext = path.suffix.lower()
        if input_type == "PDF" and ext != ".pdf":
            raise ValueError(f"Expected PDF file, got {ext}")
        if input_type == "PPTX" and ext not in {".pptx", ".ppt"}:
            raise ValueError(f"Expected PPTX file, got {ext}")
        return str(path.resolve()), input_type
    return input_str, input_type


def create_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = int(time.time())
        output_dir = PROJECT_ROOT / "outputs" / "cli" / "paper2ppt_frontend" / str(timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_frontend_workflow(args, input_content: str, input_type: str, output_dir: Path):
    from fastapi_app.schemas import FrontendPPTGenerationRequest, Paper2PPTRequest
    from fastapi_app.services.paper2ppt_frontend_service import Paper2PPTFrontendService
    from fastapi_app.services.managed_api_service import resolve_llm_credentials
    from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2page_content_wf_api

    resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
        args.api_url,
        args.api_key,
        scope=args.credential_scope,
    )

    outline_req = Paper2PPTRequest(
        language=args.language,
        chat_api_url=resolved_chat_api_url,
        credential_scope=args.credential_scope,
        chat_api_key=resolved_api_key,
        api_key=resolved_api_key,
        model=args.outline_model,
        gen_fig_model="",
        input_type=input_type,
        input_content=input_content,
        style=args.style,
        ref_img="",
        email=args.email,
        page_count=args.page_count,
        use_long_paper=bool(args.use_long_paper),
    )

    log.info("%s", "=" * 60)
    log.info("Paper2PPT Frontend Editable CLI")
    log.info("%s", "=" * 60)
    log.info("Input Type: %s", input_type)
    log.info("Page Count: %s", args.page_count)
    log.info("Use Long Paper: %s", args.use_long_paper)
    log.info("Include Images: %s", args.include_images)
    log.info("Output Directory: %s", output_dir)
    log.info("%s", "=" * 60)

    pagecontent_resp = await run_paper2page_content_wf_api(outline_req, result_path=output_dir)
    pagecontent = pagecontent_resp.pagecontent or []
    result_path = pagecontent_resp.result_path or str(output_dir)

    frontend_req = FrontendPPTGenerationRequest(
        result_path=result_path,
        pagecontent=json.dumps(pagecontent, ensure_ascii=False),
        chat_api_url=args.api_url,
        api_key=args.api_key,
        credential_scope=args.credential_scope,
        email=args.email,
        model=args.frontend_model,
        language=args.language,
        style=args.style,
        include_images=args.include_images,
        image_style=args.image_style,
        image_model=args.image_model,
    )

    frontend_service = Paper2PPTFrontendService()
    frontend_resp = await frontend_service.generate_slides(req=frontend_req, request=None)
    slides = frontend_resp.get("slides", []) or []
    theme = frontend_resp.get("theme") or {}

    slides_json_path = output_dir / "frontend_slides.json"
    theme_json_path = output_dir / "frontend_theme.json"
    summary_json_path = output_dir / "frontend_summary.json"
    slides_json_path.write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding="utf-8")
    theme_json_path.write_text(json.dumps(theme, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "success": bool(frontend_resp.get("success")),
                "pagecontent_count": len(pagecontent),
                "slide_count": len(slides),
                "result_path": result_path,
                "slides_json": str(slides_json_path),
                "theme_json": str(theme_json_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("Outline pages: %s", len(pagecontent))
    log.info("Frontend slides: %s", len(slides))
    log.info("Result path: %s", result_path)
    log.info("Slides JSON: %s", slides_json_path)
    log.info("Theme JSON: %s", theme_json_path)

    exported_pptx = None
    if args.export_pptx:
        exported_pptx = (output_dir / "paper2ppt_frontend_editable.pptx").resolve()
        slides_json_arg = slides_json_path.resolve()
        theme_json_arg = theme_json_path.resolve()
        export_cmd = [
            "npx",
            "tsx",
            "scripts/run_paper2ppt_structured_export_cli.ts",
            "--slides-json",
            str(slides_json_arg),
            "--theme-json",
            str(theme_json_arg),
            "--output",
            str(exported_pptx),
            "--asset-base-url",
            args.asset_base_url,
        ]
        log.info("Exporting editable PPTX via frontend CLI: %s", " ".join(export_cmd))
        subprocess.run(
            export_cmd,
            cwd=str(PROJECT_ROOT / "frontend-workflow"),
            check=True,
        )
        log.info("Editable PPTX: %s", exported_pptx)

    return {
        "pagecontent": pagecontent,
        "slides": slides,
        "result_path": result_path,
        "slides_json": str(slides_json_path),
        "theme_json": str(theme_json_path),
        "pptx_path": str(exported_pptx) if exported_pptx else "",
    }


async def main():
    args = parse_args()
    input_type = args.input_type or detect_input_type(args.input)
    input_content, resolved_input_type = validate_input(args.input, input_type)
    output_dir = create_output_dir(args)
    await run_frontend_workflow(args, input_content, resolved_input_type, output_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        raise SystemExit(130)
