#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2PosterRequest, Paper2PosterState
from dataflow_agent.workflow import run_workflow

log = get_logger(__name__)


async def _run_workflow(in_data: dict) -> dict:
    api_key = (in_data.get("api_key") or "").strip()
    api_url = (in_data.get("chat_api_url") or "").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_url:
        os.environ["OPENAI_BASE_URL"] = api_url

    request = Paper2PosterRequest(
        chat_api_url=api_url,
        api_key=api_key,
        chat_api_key=api_key,
        model=in_data.get("model", "gpt-4o-2024-08-06"),
        vision_model=in_data.get("vision_model", "gpt-4o-2024-08-06"),
        poster_width=in_data.get("poster_width", 54.0),
        poster_height=in_data.get("poster_height", 36.0),
        logo_path=in_data.get("logo_path", ""),
        aff_logo_path=in_data.get("aff_logo_path", ""),
        url=in_data.get("url", ""),
    )
    state = Paper2PosterState(
        request=request,
        messages=[],
        result_path=in_data.get("result_path", ""),
        paper_file=in_data.get("paper_file", ""),
        poster_width=in_data.get("poster_width", 54.0),
        poster_height=in_data.get("poster_height", 36.0),
        logo_path=in_data.get("logo_path", ""),
        aff_logo_path=in_data.get("aff_logo_path", ""),
        url=in_data.get("url", ""),
    )

    final_state = await run_workflow("paper2poster", state)
    if isinstance(final_state, dict):
        output_pptx_path = final_state.get("output_pptx_path") or ""
        output_png_path = final_state.get("output_png_path") or ""
        errors = final_state.get("errors") or []
    else:
        output_pptx_path = getattr(final_state, "output_pptx_path", "") or ""
        output_png_path = getattr(final_state, "output_png_path", "") or ""
        errors = getattr(final_state, "errors", []) or []

    success = bool(output_pptx_path)
    message = "Poster generated successfully" if success else ("; ".join(errors) if errors else "Poster generation failed")
    return {
        "success": success,
        "message": message,
        "output_pptx_path": output_pptx_path,
        "output_png_path": output_png_path,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="paper2poster worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    try:
        in_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        output_path.write_text(
            json.dumps({"success": False, "message": f"failed to load input json: {exc}"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1

    try:
        result = asyncio.run(_run_workflow(in_data))
    except Exception as exc:
        log.exception("[paper2poster-worker] workflow failed")
        result = {"success": False, "message": str(exc)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
