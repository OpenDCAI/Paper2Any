from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataflow_agent.toolkits.p2vtool.p2v_tool as p2v_tool
from dataflow_agent.toolkits.p2vtool.p2v_tool import (
    build_p2v_cursor_backend_config,
    build_p2v_local_tts_config,
    get_default_p2v_cursor_image_path,
    resolve_p2v_cursor_image_path,
)


def test_resolve_p2v_cursor_image_path_falls_back_to_repo_asset() -> None:
    resolved = resolve_p2v_cursor_image_path("/definitely/not/exist/cursor.png")
    assert resolved == get_default_p2v_cursor_image_path()


def test_build_p2v_cursor_backend_config_prefers_vlm(monkeypatch) -> None:
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_MODEL_PATH", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_ENABLED", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_PYTHON", raising=False)
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_API_URL", "http://cursor-vlm.example/v1")
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_VLM_MODEL", "gpt-4o-2024-11-20")

    backend = build_p2v_cursor_backend_config(
        chat_api_url="http://ignored.example/v1",
        api_key="ignored-key",
        default_model="ignored-model",
    )

    assert backend["mode"] == "vlm"
    assert backend["api_url"] == "http://cursor-vlm.example/v1"
    assert backend["api_key"] == "cursor-key"
    assert backend["model"] == "gpt-4o-2024-11-20"


def test_build_p2v_cursor_backend_config_falls_back_to_center(monkeypatch) -> None:
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_MODEL_PATH", raising=False)
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_ENABLED", "off")
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_PYTHON", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_API_URL", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("DF_API_URL", raising=False)
    monkeypatch.delenv("DF_API_KEY", raising=False)

    backend = build_p2v_cursor_backend_config(
        chat_api_url="",
        api_key="",
        default_model="",
    )

    assert backend["mode"] == "center"


def test_build_p2v_cursor_backend_config_supports_isolated_local_python(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "UI-TARS-1.5-7B"
    model_dir.mkdir()

    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_ENABLED", "auto")
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_PYTHON", sys.executable)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_API_URL", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_API_KEY", raising=False)

    backend = build_p2v_cursor_backend_config()

    assert backend["mode"] == "local"
    assert backend["local_model_path"] == str(model_dir)
    assert backend["local_python"] == sys.executable


def test_build_p2v_cursor_backend_config_skips_local_when_worker_env_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "UI-TARS-1.5-7B"
    model_dir.mkdir()

    monkeypatch.setattr(p2v_tool, "_current_env_supports_local_cursor_worker", lambda: False)
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_ENABLED", "auto")
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_LOCAL_MODEL_PATH", str(model_dir))
    monkeypatch.delenv("PAPER2VIDEO_CURSOR_LOCAL_PYTHON", raising=False)
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_API_URL", "http://cursor-vlm.example/v1")
    monkeypatch.setenv("PAPER2VIDEO_CURSOR_API_KEY", "cursor-key")

    backend = build_p2v_cursor_backend_config()

    assert backend["mode"] == "vlm"


def test_build_p2v_local_tts_config_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PAPER2VIDEO_ENABLE_LOCAL_TTS", raising=False)
    monkeypatch.delenv("PAPER2VIDEO_LOCAL_TTS_GPU_IDS", raising=False)

    config = build_p2v_local_tts_config()

    assert config["enabled"] is False
