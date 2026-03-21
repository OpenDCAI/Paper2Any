#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "sam3_gpu1.log"
PID_FILE = LOG_DIR / "sam3.pid"
ENV_FILE = LOG_DIR / "model_servers.env"

PYTHON = os.environ.get("PAPER2ANY_PYTHON", "/opt/conda/bin/python")
SAM3_HOME = os.environ.get("SAM3_HOME", "/mnt/paper2any/lz/github-proj/Paper2Any/sam3_src")
SAM3_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT_PATH",
    "/mnt/paper2any/lz/github-proj/Paper2Any/models/sam3/sam3.pt",
)
SAM3_BPE = os.environ.get(
    "SAM3_BPE_PATH",
    "/mnt/paper2any/lz/github-proj/Paper2Any/models/sam3/bpe_simple_vocab_16e6.txt.gz",
)
SAM3_GPU = os.environ.get("SAM3_GPU", "1")
SAM3_PORT = os.environ.get("SAM3_PORT", "8021")


def main() -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = SAM3_GPU
    env["SAM3_HOME"] = SAM3_HOME
    env["SAM3_CHECKPOINT_PATH"] = SAM3_CHECKPOINT
    env["SAM3_BPE_PATH"] = SAM3_BPE

    cmd = [
        PYTHON,
        "-m",
        "dataflow_agent.toolkits.model_servers.sam3_server",
        "--host",
        "0.0.0.0",
        "--port",
        SAM3_PORT,
        "--checkpoint",
        SAM3_CHECKPOINT,
        "--bpe",
        SAM3_BPE,
        "--device",
        "cuda",
    ]

    with LOG_FILE.open("ab", buffering=0) as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    PID_FILE.write_text(f"{proc.pid}\n", encoding="utf-8")
    ENV_FILE.write_text(
        "\n".join(
            [
                f"export SAM3_SERVER_URLS=http://127.0.0.1:{SAM3_PORT}",
                f"export SAM3_HOME={SAM3_HOME}",
                f"export SAM3_CHECKPOINT_PATH={SAM3_CHECKPOINT}",
                f"export SAM3_BPE_PATH={SAM3_BPE}",
                f"export PAPER2DRAWIO_SAM3_CHECKPOINT_PATH={SAM3_CHECKPOINT}",
                f"export PAPER2DRAWIO_SAM3_BPE_PATH={SAM3_BPE}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"sam3 exited early with code {proc.returncode}", file=sys.stderr)
        return 1

    print(proc.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
