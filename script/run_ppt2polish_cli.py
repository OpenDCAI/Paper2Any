#!/usr/bin/env python3
"""
PPT2Polish CLI - Beautify existing PPT files.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from script_lib import JSON_RESULT_PREFIX, run_ppt2polish_job


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPT2Polish CLI - Beautify existing PPT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input PPT/PPTX file path")
    parser.add_argument("--api-url", help="LLM API URL")
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--model", default="gpt-4o", help="Text model name")
    parser.add_argument("--gen-fig-model", default="gemini-2.5-flash-image-preview", help="Image generation model")
    parser.add_argument("--style", default="现代简约风格", help="Target style description")
    parser.add_argument("--ref-img", default="", help="Reference image for style consistency")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON result")
    return parser.parse_args()


def print_result(result, as_json: bool):
    payload = result.to_dict()
    if as_json:
        print(f"{JSON_RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}")
        return
    print("PPT2Polish completed")
    print(f"Run directory: {payload['run_dir']}")
    print(f"Primary output: {payload['primary_output'] or '(not found)'}")
    if payload.get("error"):
        print(f"Error: {payload['error']}")


def main():
    try:
        args = parse_args()
        result = asyncio.run(
            run_ppt2polish_job(
                input_path=args.input,
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                gen_fig_model=args.gen_fig_model,
                style=args.style,
                ref_img=args.ref_img,
                output_dir=args.output_dir,
            )
        )
        print_result(result, args.json)
        return 0 if result.success else 1
    except Exception as exc:
        if "--json" in sys.argv:
            print(
                f"{JSON_RESULT_PREFIX}{json.dumps({'success': False, 'skill_name': 'ppt2polish', 'error': str(exc)}, ensure_ascii=False)}"
            )
        else:
            print(f"PPT2Polish failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
