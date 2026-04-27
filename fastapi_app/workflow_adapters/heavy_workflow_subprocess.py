from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

from fastapi_app.utils import resolve_outputs_path

log = get_logger(__name__)

SUBPROCESS_WORKER_ENV = "PAPER2ANY_SUBPROCESS_WORKER"
DISABLE_SUBPROCESS_ENV = "PAPER2ANY_DISABLE_HEAVY_WORKFLOW_SUBPROCESS"
HEAVY_WORKFLOW_PYTHON_ENV = "PAPER2ANY_HEAVY_WORKFLOW_PYTHON"
_LOG_STREAM_LIMIT = 2 * 1024 * 1024


def _sanitize_worker_proxy_env(env: dict[str, str]) -> None:
    """
    The host shell may export ALL_PROXY=socks5h://..., but several SDK/httpx
    call sites inside the heavy workflow worker do not support that scheme.
    Keep HTTP(S) proxy settings intact and drop only the problematic SOCKS
    inheritance for the worker subprocess.
    """
    for key in ("ALL_PROXY", "all_proxy"):
        raw = (env.get(key, "") or "").strip()
        if raw.lower().startswith("socks5h://"):
            env.pop(key, None)


def in_heavy_workflow_subprocess() -> bool:
    return (os.getenv(SUBPROCESS_WORKER_ENV, "") or "").strip() == "1"


def should_use_heavy_workflow_subprocess(*, default: bool = True) -> bool:
    if in_heavy_workflow_subprocess():
        return False
    raw = (os.getenv(DISABLE_SUBPROCESS_ENV, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return False
    return default


def _pick_python_bin() -> str:
    configured_python = (os.getenv(HEAVY_WORKFLOW_PYTHON_ENV, "") or "").strip()
    candidates = [configured_python, sys.executable, "/opt/conda/bin/python3", "python3", "python"]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "No usable Python found for heavy workflow worker. "
        f"Set {HEAVY_WORKFLOW_PYTHON_ENV} to a valid interpreter path."
    )


def _exit_reason(returncode: int) -> str:
    if returncode >= 0:
        return f"code {returncode}"
    signum = -returncode
    try:
        return f"signal {signal.Signals(signum).name}"
    except ValueError:
        return f"signal {signum}"


async def run_heavy_workflow_in_subprocess(
    *,
    mode: str,
    payload: dict[str, Any],
    result_path: Path | None = None,
) -> dict[str, Any]:
    project_root = get_project_root()
    worker_script = project_root / "script" / "heavy_workflow_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"heavy workflow worker script not found: {worker_script}")

    if result_path:
        worker_base_dir = resolve_outputs_path(result_path, must_exist=False, allow_dirs=True)
    else:
        worker_base_dir = (project_root / "outputs" / "_workers" / mode).resolve()
    worker_dir = worker_base_dir / ".worker" / uuid4().hex
    worker_dir.mkdir(parents=True, exist_ok=True)

    input_json = worker_dir / "input.json"
    output_json = worker_dir / "output.json"
    input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    python_bin = _pick_python_bin()
    cmd = [
        python_bin,
        str(worker_script),
        "--mode",
        mode,
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
    ]
    env = os.environ.copy()
    env[SUBPROCESS_WORKER_ENV] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    _sanitize_worker_proxy_env(env)

    log.info("[heavy-workflow:%s] running worker: %s", mode, " ".join(cmd))
    cleanup = True

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_LOG_STREAM_LIMIT,
        )

        async def _forward_stream(stream) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    log.info("[heavy-workflow:%s] %s", mode, text)

        await asyncio.gather(_forward_stream(proc.stdout), _forward_stream(proc.stderr))
        await proc.wait()

        if not output_json.exists():
            cleanup = False
            reason = _exit_reason(proc.returncode)
            raise RuntimeError(
                f"{mode} worker exited with {reason} and produced no output json "
                f"(worker_dir={worker_dir})"
            )

        try:
            out_data = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:
            cleanup = False
            raise RuntimeError(
                f"{mode} worker returned invalid JSON: {exc} (worker_dir={worker_dir})"
            ) from exc

        if proc.returncode != 0:
            cleanup = False
            reason = _exit_reason(proc.returncode)
            err = out_data.get("error") or "unknown error"
            raise RuntimeError(
                f"{mode} worker exited with {reason}: {err} (worker_dir={worker_dir})"
            )

        if not out_data.get("success"):
            cleanup = False
            err = out_data.get("error") or "unknown error"
            raise RuntimeError(f"{mode} worker failed: {err} (worker_dir={worker_dir})")

        return out_data
    finally:
        if cleanup:
            shutil.rmtree(worker_dir, ignore_errors=True)
