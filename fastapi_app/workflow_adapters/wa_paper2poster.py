from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


async def run_paper2poster_generate_wf_api(
    *,
    result_path: Path,
    paper_file: str,
    chat_api_url: str,
    api_key: str,
    model: str,
    vision_model: str,
    poster_width: float,
    poster_height: float,
    logo_path: str = "",
    aff_logo_path: str = "",
    url: str = "",
    email: str = "",
) -> Dict[str, Any]:
    """通过独立子进程执行 paper2poster 工作流，隔离 cwd/env/sys.path 变更。"""
    project_root = get_project_root()
    result_root = Path(result_path).resolve()
    worker_dir = result_root / ".worker" / uuid4().hex
    worker_dir.mkdir(parents=True, exist_ok=True)
    input_json = worker_dir / "input.json"
    output_json = worker_dir / "output.json"
    worker_script = project_root / "script" / "paper2poster_worker.py"

    payload = {
        "result_path": str(result_root),
        "paper_file": paper_file,
        "chat_api_url": chat_api_url,
        "api_key": api_key,
        "model": model,
        "vision_model": vision_model,
        "poster_width": poster_width,
        "poster_height": poster_height,
        "logo_path": logo_path,
        "aff_logo_path": aff_logo_path,
        "url": url,
        "email": email,
    }
    input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(worker_script),
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if stdout:
        log.info("[paper2poster-worker stdout]\n%s", stdout.decode("utf-8", errors="ignore").strip())
    if stderr:
        log.warning("[paper2poster-worker stderr]\n%s", stderr.decode("utf-8", errors="ignore").strip())

    if not output_json.is_file():
        message = f"paper2poster worker exited with code {proc.returncode} and produced no output json"
        log.error(message)
        return {"success": False, "message": message}

    try:
        result = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"paper2poster worker returned invalid JSON: {exc}"
        log.error(message)
        return {"success": False, "message": message}

    if proc.returncode != 0 and result.get("success", False):
        result = {
            "success": False,
            "message": f"paper2poster worker exited with code {proc.returncode}",
        }
    return result
