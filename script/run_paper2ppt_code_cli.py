#!/usr/bin/env python3
"""
Paper2PPT Code CLI - generate a real editable PPTX from existing pagecontent flow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from script.cli_env import load_project_env, resolve_cli_text_credentials
from dataflow_agent.logger import get_logger
from fastapi_app.config import settings

load_project_env()

log = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper2PPT code editable CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/run_paper2ppt_code_cli.py --input paper.pdf --page-count 12
  python script/run_paper2ppt_code_cli.py --input long_paper.pdf --input-type PDF --page-count 50 --use-long-paper
  python script/run_paper2ppt_code_cli.py --input "Multi-agent systems" --input-type TOPIC --page-count 8
""",
    )
    parser.add_argument("--input", required=True, help="Input PDF/PPTX path, topic, or text")
    parser.add_argument("--input-type", choices=["PDF", "PPTX", "TEXT", "TOPIC"], help="Input type")
    parser.add_argument("--api-url", help="LLM API URL")
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--credential-scope", default="paper2ppt", help="Managed credential scope")
    parser.add_argument("--email", default="cli_code_test@paper2any.local", help="Logical email for outputs")
    parser.add_argument("--outline-model", default=settings.PAPER2PPT_OUTLINE_MODEL, help="Outline model")
    parser.add_argument("--code-model", default=settings.PAPER2PPT_CONTENT_MODEL, help="Editable code model")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Output language")
    parser.add_argument("--style", default="", help="Style prompt")
    parser.add_argument("--page-count", type=int, default=8, help="Target page count")
    parser.add_argument("--use-long-paper", action="store_true", help="Force long-paper outline workflow")
    parser.add_argument("--include-pdf-preview", action="store_true", help="Attempt PDF preview export")
    parser.add_argument("--output-dir", help="Output directory (default: outputs/cli/paper2ppt_code/{timestamp})")
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
        output_dir = PROJECT_ROOT / "outputs" / "cli" / "paper2ppt_code" / str(timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def run_code_workflow(args, input_content: str, input_type: str, output_dir: Path):
    from fastapi_app.schemas import EditablePPTGenerationRequest, Paper2PPTRequest
    from fastapi_app.services.editable_ppt_service import EditablePPTService
    from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2page_content_wf_api

    api_url, api_key = resolve_cli_text_credentials(args.api_url, args.api_key)
    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or DF_API_KEY environment variable.")

    outline_req = Paper2PPTRequest(
        language=args.language,
        chat_api_url=api_url,
        credential_scope=args.credential_scope,
        chat_api_key=api_key,
        api_key=api_key,
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

    pagecontent_resp = await run_paper2page_content_wf_api(outline_req, result_path=output_dir)
    pagecontent = pagecontent_resp.pagecontent or []
    result_path = pagecontent_resp.result_path or str(output_dir)

    editable_chat_api_url, editable_api_key = resolve_cli_text_credentials(args.api_url, args.api_key)
    editable_req = EditablePPTGenerationRequest(
        result_path=result_path,
        pagecontent=json.dumps(pagecontent, ensure_ascii=False),
        chat_api_url=editable_chat_api_url,
        api_key=editable_api_key,
        credential_scope=args.credential_scope,
        email=args.email,
        model=args.code_model,
        language=args.language,
        style=args.style,
        include_pdf_preview=bool(args.include_pdf_preview),
    )

    response = await EditablePPTService().generate_editable_ppt(req=editable_req, request=None)
    log.info("Result path: %s", response.get("result_path"))
    log.info("Editable PPTX: %s", response.get("ppt_pptx_path"))
    if response.get("ppt_pdf_path"):
        log.info("Preview PDF: %s", response.get("ppt_pdf_path"))
    log.info("IR JSON: %s", response.get("ir_path"))
    log.info("Render log: %s", response.get("render_log_path"))
    return response


async def main():
    args = parse_args()
    input_type = args.input_type or detect_input_type(args.input)
    input_content, resolved_input_type = validate_input(args.input, input_type)
    output_dir = create_output_dir(args)
    await run_code_workflow(args, input_content, resolved_input_type, output_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        raise SystemExit(130)
