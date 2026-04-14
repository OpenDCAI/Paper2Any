from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTERTOOL_ROOT = PROJECT_ROOT / "dataflow_agent" / "toolkits" / "postertool"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(POSTERTOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTERTOOL_ROOT))

from src.agents.renderer import Renderer


def _tokenize_in_subprocess(text: str, queue: mp.Queue) -> None:
    renderer = Renderer()
    queue.put(renderer._tokenize_formatting(text))


def _tokenize_with_timeout(text: str, timeout: float = 2.0) -> list[dict]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_tokenize_in_subprocess, args=(text, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise AssertionError("Renderer tokenizer hung on malformed markdown input")

    assert proc.exitcode == 0
    return queue.get(timeout=1)


def test_renderer_tokenizer_supports_bold_italic_markdown() -> None:
    renderer = Renderer()
    segments = renderer._tokenize_formatting(
        "• ***Guarantees:*** parameter safety, bounded recovery, and **termination (Tmax=10)**"
    )

    assert [segment["text"] for segment in segments] == [
        "• ",
        "Guarantees:",
        " parameter safety, bounded recovery, and ",
        "termination (Tmax=10)",
    ]
    assert segments[1]["bold"] is True and segments[1]["italic"] is True
    assert segments[3]["bold"] is True and segments[3]["italic"] is False


def test_renderer_tokenizer_does_not_hang_on_malformed_markdown() -> None:
    segments = _tokenize_with_timeout(
        "• ***Ablation* headline:** removing logic rules causes large drops"
    )

    joined = "".join(segment["text"] for segment in segments)
    assert "Ablation" in joined
    assert "headline:" in joined
    assert "removing logic rules causes large drops" in joined
