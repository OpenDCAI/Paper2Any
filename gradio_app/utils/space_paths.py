from __future__ import annotations

import os
import time
from pathlib import Path


def get_persistent_outputs_root() -> Path:
    """
    Hugging Face Spaces persistent storage is mounted at /data.
    If it's available, write outputs there; otherwise fall back to project_root/outputs.
    """
    candidates = []
    # HF Spaces persistent storage
    candidates.append(Path("/data/outputs"))
    # optional user override
    env = os.getenv("DF_OUTPUTS_DIR")
    if env:
        candidates.insert(0, Path(env))

    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            test = c / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return c
        except Exception:
            continue

    # project_root = repo root (…/Paper2Any)
    root = Path(__file__).resolve().parents[2]
    out = (root / "outputs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def new_run_dir(app_name: str) -> Path:
    """
    Create a unique run directory under persistent outputs.
    """
    ts = int(time.time())
    base = get_persistent_outputs_root() / "hf_space" / app_name / str(ts)
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()
