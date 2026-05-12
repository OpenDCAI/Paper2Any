from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vendor.presentagent_runtime.contracts import EditablePPTInputRunRequest, EditablePPTRunRequest
    from vendor.presentagent_runtime.env import resolve_runtime_input_request, resolve_runtime_request
    from vendor.presentagent_runtime.runner import run_from_input, run_from_pagecontent
else:
    from .contracts import EditablePPTInputRunRequest, EditablePPTRunRequest
    from .env import resolve_runtime_input_request, resolve_runtime_request
    from .runner import run_from_input, run_from_pagecontent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PresentAgent runtime scaffold.")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--input", help="Input PDF/PPTX path, topic, or text. When set, pagecontent is parsed first.")
    parser.add_argument("--input-type", choices=["PDF", "PPTX", "TEXT", "TOPIC"], default="PDF")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--style", default="")
    parser.add_argument("--page-count", type=int, default=10)
    parser.add_argument("--use-long-paper", action="store_true")
    parser.add_argument("--outline-model", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--vlm-api-url", default="")
    parser.add_argument("--vlm-api-key", default="")
    parser.add_argument("--image-model", default="")
    parser.add_argument("--image-api-url", default="")
    parser.add_argument("--image-api-key", default="")
    parser.add_argument("--credential-scope", default="paper2ppt")
    parser.add_argument("--email", default="")
    parser.add_argument("--enable-llm-codegen", action="store_true")
    parser.add_argument("--include-pdf-preview", dest="include_pdf_preview", action="store_true", default=True)
    parser.add_argument("--no-pdf-preview", dest="include_pdf_preview", action="store_false")
    parser.add_argument("--pagecontent-json", default="[]")
    parser.add_argument("--template", default="blue_style",
                       choices=["blue_style", "pptx_new", "pptx_templete", "paper2ppt_code_editable"],
                       help="选择PPT模板 (默认: blue_style)")
    return parser


def _parse_pagecontent_json(raw_pagecontent: str) -> list[dict[str, Any]]:
    payload = json.loads(raw_pagecontent)
    if not isinstance(payload, list):
        raise ValueError("pagecontent payload must be a JSON list")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("pagecontent payload items must be JSON objects")
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.input:
        runtime_req = EditablePPTInputRunRequest(
            result_path=args.result_path,
            input_type=args.input_type,
            input_content=args.input,
            language=args.language,
            style=args.style,
            page_count=args.page_count,
            use_long_paper=bool(args.use_long_paper),
            outline_model=args.outline_model,
            model=args.model,
            api_url=args.api_url or None,
            api_key=args.api_key or None,
            credential_scope=args.credential_scope,
            email=args.email,
            vlm_model=args.vlm_model,
            vlm_api_url=args.vlm_api_url or None,
            vlm_api_key=args.vlm_api_key or None,
            image_model=args.image_model,
            image_api_url=args.image_api_url or None,
            image_api_key=args.image_api_key or None,
            enable_llm_codegen=bool(args.enable_llm_codegen),
            include_pdf_preview=bool(args.include_pdf_preview),
            template=args.template,
        )
        asyncio.run(
            run_from_input(runtime_req)
        )
        return 0

    pagecontent = _parse_pagecontent_json(args.pagecontent_json)
    runtime_req = EditablePPTRunRequest(
        result_path=args.result_path,
        pagecontent=pagecontent,
        language=args.language,
        style=args.style,
        model=args.model,
        api_url=args.api_url or None,
        api_key=args.api_key or None,
        vlm_model=args.vlm_model,
        vlm_api_url=args.vlm_api_url or None,
        vlm_api_key=args.vlm_api_key or None,
        image_model=args.image_model,
        image_api_url=args.image_api_url or None,
        image_api_key=args.image_api_key or None,
        enable_llm_codegen=bool(args.enable_llm_codegen),
        include_pdf_preview=bool(args.include_pdf_preview),
    )
    run_from_pagecontent(
        runtime_req
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
