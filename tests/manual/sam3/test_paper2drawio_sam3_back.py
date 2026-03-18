"""
Test script for paper2drawio_sam3 workflow.

This script runs the workflow directly (no CLI). It expects:
- SAM3 HTTP service running on http://127.0.0.1:8001
- Input image: tests/sam3/ori.png

Optional env vars:
- AZURE_OCR_ENDPOINT (default http://localhost:5000)
- TEXT_FORMULA_ENGINE (default none)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def _ensure_repo_on_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # Paper2Any project root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _ensure_workflow_registry(repo_root: Path) -> None:
    """
    Preload dataflow_agent.workflow.registry without importing dataflow_agent.workflow.__init__.
    This avoids auto-import of all wf_*.py (which pulls optional deps).
    """
    import types
    import importlib.util

    wf_pkg_name = "dataflow_agent.workflow"
    if wf_pkg_name not in sys.modules:
        wf_pkg = types.ModuleType(wf_pkg_name)
        wf_pkg.__path__ = [str(repo_root / "dataflow_agent" / "workflow")]
        wf_pkg.__package__ = "dataflow_agent"
        sys.modules[wf_pkg_name] = wf_pkg

    reg_name = "dataflow_agent.workflow.registry"
    if reg_name not in sys.modules:
        reg_path = repo_root / "dataflow_agent" / "workflow" / "registry.py"
        spec = importlib.util.spec_from_file_location(reg_name, reg_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load registry module: {reg_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[reg_name] = mod
        spec.loader.exec_module(mod)


def _check_sam3_health(url: str) -> bool:
    try:
        with urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except URLError:
        return False


def main() -> int:
    repo_root = _ensure_repo_on_path()
    img_path = repo_root / "tests" / "sam3" / "ori.png"

    if not img_path.exists():
        print(f"[ERROR] Input image not found: {img_path}")
        return 1

    # Default envs (can be overridden by user env)
    os.environ.setdefault("AZURE_OCR_ENDPOINT", "http://localhost:5000")
    os.environ.setdefault("TEXT_FORMULA_ENGINE", "none")

    if not _check_sam3_health("http://127.0.0.1:8001/health"):
        print("[ERROR] SAM3 service not ready at http://127.0.0.1:8001/health")
        print("        Please start the service before running this script.")
        return 2

    # Preload workflow registry without importing workflow/__init__
    _ensure_workflow_registry(repo_root)

    # Import the target workflow module via file path to avoid triggering
    # dataflow_agent.workflow.__init__ (which imports all wf_*.py).
    from dataflow_agent.state import Paper2DrawioState
    import importlib.util

    wf_path = repo_root / "dataflow_agent" / "workflow" / "wf_paper2drawio_sam3.py"
    mod_name = "dataflow_agent.workflow.wf_paper2drawio_sam3"
    spec = importlib.util.spec_from_file_location(mod_name, wf_path)
    if spec is None or spec.loader is None:
        print(f"[ERROR] Failed to load workflow module: {wf_path}")
        return 4
    wf_mod = importlib.util.module_from_spec(spec)
    # Ensure module is registered for dataclass type resolution during exec
    sys.modules[mod_name] = wf_mod
    spec.loader.exec_module(wf_mod)
    create_paper2drawio_sam3_graph = getattr(wf_mod, "create_paper2drawio_sam3_graph")

    state = Paper2DrawioState(paper_file=str(img_path))
    graph = create_paper2drawio_sam3_graph().build()
    out = asyncio.run(graph.ainvoke(state))

    # LangGraph may return a dict state; handle both dataclass and dict.
    output_xml = None
    if isinstance(out, dict):
        output_xml = out.get("output_xml_path") or out.get("output_xml")
    else:
        output_xml = getattr(out, "output_xml_path", None) or getattr(out, "output_xml", None)

    print("output_xml:", output_xml)
    if output_xml and Path(output_xml).exists():
        print("[OK] draw.io XML generated.")
        return 0

    print("[ERROR] draw.io XML not generated.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
