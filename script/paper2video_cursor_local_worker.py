from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.p2vtool.p2v_tool import _infer_cursor_with_local_model

log = get_logger(__name__)


def _load_payload(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_payload(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper2Video local cursor worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    try:
        payload = _load_payload(args.input_json)
        alloc_conf = str(payload.get("alloc_conf", "") or "").strip()
        if alloc_conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        attn_implementation = str(payload.get("attn_implementation", "") or "").strip()
        if attn_implementation:
            os.environ["PAPER2VIDEO_CURSOR_LOCAL_ATTN_IMPL"] = attn_implementation
        gpu_id = payload.get("gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        model_path = str(payload.get("model_path", "") or "").strip()
        max_new_tokens = int(payload.get("max_new_tokens", 64))
        tasks = payload.get("tasks") or []
        results = []

        for task in tasks:
            slide_idx = int(task.get("slide_idx", 0))
            sentence_idx = int(task.get("sentence_idx", 0))
            prompt = str(task.get("prompt", "") or "")
            cursor_prompt = str(task.get("cursor_prompt", "") or "").strip()
            image_path = str(task.get("image_path", "") or "")
            point = None
            error_message = ""
            try:
                if cursor_prompt.lower() != "no":
                    point = _infer_cursor_with_local_model(
                        cursor_prompt,
                        image_path,
                        model_path=model_path,
                        gpu_id=gpu_id,
                        max_new_tokens=max_new_tokens,
                    )
            except Exception as exc:
                error_message = str(exc)
                log.warning(
                    "[p2v-cursor-local-worker] slide=%s sentence=%s failed: %s",
                    slide_idx,
                    sentence_idx,
                    exc,
                )

            results.append(
                {
                    "slide": slide_idx,
                    "sentence": sentence_idx,
                    "speech_text": prompt,
                    "cursor_prompt": cursor_prompt,
                    "cursor": point,
                    "cursor_backend": "local",
                    "cursor_error": error_message,
                }
            )

        _dump_payload(args.output_json, {"success": True, "results": results})
        return 0
    except Exception as exc:
        _dump_payload(args.output_json, {"success": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
