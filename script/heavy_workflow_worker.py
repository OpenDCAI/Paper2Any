#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PAPER2ANY_SUBPROCESS_WORKER"] = "1"

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def _result_path_from_input(in_data: dict) -> Path | None:
    raw = str(in_data.get("result_path") or "").strip()
    return Path(raw) if raw else None


def _run_paper2figure(in_data: dict) -> dict:
    from fastapi_app.schemas import Paper2FigureRequest
    from fastapi_app.workflow_adapters.wa_paper2figure import run_paper2figure_wf_api_local

    req = Paper2FigureRequest.model_validate(in_data.get("request") or {})
    result_path = _result_path_from_input(in_data)
    response = asyncio.run(run_paper2figure_wf_api_local(req, result_path=result_path))
    return {"success": True, "response": response.model_dump(mode="json")}


def _run_pdf2ppt(in_data: dict) -> dict:
    from fastapi_app.schemas import Paper2PPTRequest
    from fastapi_app.workflow_adapters.wa_pdf2ppt import run_pdf2ppt_wf_api_local

    req = Paper2PPTRequest.model_validate(in_data.get("request") or {})
    result_path = _result_path_from_input(in_data)
    response = asyncio.run(run_pdf2ppt_wf_api_local(req, result_path=result_path))
    return {"success": True, "response": response.model_dump(mode="json")}


def _run_paper2ppt(in_data: dict) -> dict:
    from fastapi_app.schemas import Paper2PPTRequest
    from fastapi_app.workflow_adapters.wa_paper2ppt import run_paper2ppt_wf_api_local

    req = Paper2PPTRequest.model_validate(in_data.get("request") or {})
    response = asyncio.run(
        run_paper2ppt_wf_api_local(
            req=req,
            pagecontent=in_data.get("pagecontent") or [],
            result_path=str(in_data.get("result_path") or "").strip() or None,
            get_down=in_data.get("get_down"),
            edit_page_num=in_data.get("edit_page_num"),
            edit_page_prompt=in_data.get("edit_page_prompt"),
            regenerate_from_outline=bool(in_data.get("regenerate_from_outline", False)),
            auto_fill_generated_pages=bool(in_data.get("auto_fill_generated_pages", True)),
            skip_pages=in_data.get("skip_pages") or [],
        )
    )
    return {"success": True, "response": response.model_dump(mode="json")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Heavy workflow subprocess worker")
    parser.add_argument("--mode", required=True, choices=["paper2figure", "pdf2ppt", "paper2ppt"])
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        output_path.write_text(
            json.dumps({"success": False, "error": f"Input file not found: {input_path}"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1

    try:
        in_data = json.loads(input_path.read_text(encoding="utf-8"))
        if args.mode == "paper2figure":
            result = _run_paper2figure(in_data)
        elif args.mode == "pdf2ppt":
            result = _run_pdf2ppt(in_data)
        else:
            result = _run_paper2ppt(in_data)
    except Exception as exc:
        log.exception("[heavy-workflow-worker] mode=%s failed", args.mode)
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
