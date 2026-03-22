#!/usr/bin/env python3
"""
Paper2PPT CLI - Convert papers to PPT presentations.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from script_lib import JSON_RESULT_PREFIX, run_paper2ppt_job


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper2PPT CLI - Convert papers to PPT presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input file (PDF/PPTX) or text content")
    parser.add_argument("--input-type", choices=["PDF", "TEXT", "TOPIC", "PPTX"], help="Override input type")
    parser.add_argument("--api-url", help="LLM API URL")
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--model", default="gpt-5.1", help="Text model name")
    parser.add_argument("--gen-fig-model", default="gemini-2.5-flash-image-preview", help="Image generation model")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Output language")
    parser.add_argument("--style", default="", help="PPT style description")
    parser.add_argument("--page-count", type=int, default=10, help="Target page count")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"])
    parser.add_argument("--use-long-paper", action="store_true", help="Use long paper workflow")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON result")
    return parser.parse_args()


def print_result(result, as_json: bool):
    payload = result.to_dict()
    if as_json:
        print(f"{JSON_RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}")
        return
    print("Paper2PPT completed")
    print(f"Run directory: {payload['run_dir']}")
    print(f"Primary output: {payload['primary_output'] or '(not found)'}")
    if payload.get("error"):
        print(f"Error: {payload['error']}")


def main():
    try:
        args = parse_args()
        result = asyncio.run(
            run_paper2ppt_job(
                input_value=args.input,
                input_type=args.input_type,
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                gen_fig_model=args.gen_fig_model,
                language=args.language,
                style=args.style,
                page_count=args.page_count,
                aspect_ratio=args.aspect_ratio,
                use_long_paper=args.use_long_paper,
                output_dir=args.output_dir,
            )
        )
        print_result(result, args.json)
        return 0 if result.success else 1
    except Exception as exc:
        if "--json" in sys.argv:
            print(
                f"{JSON_RESULT_PREFIX}{json.dumps({'success': False, 'skill_name': 'paper2ppt', 'error': str(exc)}, ensure_ascii=False)}"
            )
        else:
            print(f"Paper2PPT failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
