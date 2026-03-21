from __future__ import annotations

import asyncio
import io
import sys
import types
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fake_mineru_vl_utils = types.ModuleType("mineru_vl_utils")


class _FakeMinerUClient:
    def __init__(self, *args, **kwargs) -> None:
        pass


fake_mineru_vl_utils.MinerUClient = _FakeMinerUClient
sys.modules.setdefault("mineru_vl_utils", fake_mineru_vl_utils)

from dataflow_agent.toolkits.multimodaltool import mineru_tool


class _DummyResponse:
    def __init__(self, status_code: int, json_data=None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = content

    def json(self):
        if self._json_data is None:
            raise ValueError("json() not available")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_run_mineru_pdf_extract_http_remote_api(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%remote mineru test\n")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("result/full.md", "# Remote Parsed\n\nHello MinerU\n")
        zf.writestr("result/images/figure_1.png", b"fake-png-bytes")
    zip_bytes = zip_buffer.getvalue()

    poll_states = iter(
        [
            {
                "code": 0,
                "msg": "ok",
                "data": {
                    "batch_id": "batch-1",
                    "extract_result": [{"file_name": "sample.pdf", "state": "running", "err_msg": ""}],
                },
            },
            {
                "code": 0,
                "msg": "ok",
                "data": {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "sample.pdf",
                            "state": "done",
                            "err_msg": "",
                            "full_zip_url": "https://cdn.example.com/result.zip",
                        }
                    ],
                },
            },
        ]
    )

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/file-urls/batch")
        assert headers["Authorization"] == "Bearer test-token"
        assert json["model_version"] == "vlm"
        return _DummyResponse(
            200,
            {
                "code": 0,
                "msg": "ok",
                "data": {
                    "batch_id": "batch-1",
                    "file_urls": ["https://upload.example.com/file.pdf"],
                },
            },
        )

    def fake_put(url, data=None, timeout=None):
        assert url == "https://upload.example.com/file.pdf"
        assert data.read().startswith(b"%PDF-1.4")
        return _DummyResponse(200)

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/extract-results/batch/batch-1"):
            return _DummyResponse(200, next(poll_states))
        if url == "https://cdn.example.com/result.zip":
            return _DummyResponse(200, content=zip_bytes)
        raise AssertionError(f"unexpected GET url: {url}")

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setenv("MINERU_API_BASE_URL", "https://mineru.net/api/v4")
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    monkeypatch.setenv("MINERU_API_MODEL_VERSION", "vlm")
    monkeypatch.setenv("MINERU_API_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINERU_API_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(mineru_tool.requests, "post", fake_post)
    monkeypatch.setattr(mineru_tool.requests, "put", fake_put)
    monkeypatch.setattr(mineru_tool.requests, "get", fake_get)
    monkeypatch.setattr(mineru_tool.time, "sleep", lambda _: None)
    monkeypatch.setattr(mineru_tool.asyncio, "to_thread", fake_to_thread)

    markdown_text, auto_dir = asyncio.run(
        mineru_tool.run_mineru_pdf_extract_http(
            str(pdf_path),
            str(tmp_path / "outputs"),
        )
    )

    auto_dir_path = Path(auto_dir)
    assert "Hello MinerU" in markdown_text
    assert (auto_dir_path / "full.md").exists()
    assert (auto_dir_path / "sample.md").exists()
    assert (auto_dir_path / "images" / "figure_1.png").exists()
