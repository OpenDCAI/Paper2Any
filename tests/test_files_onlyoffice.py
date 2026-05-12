from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app import utils as outputs_utils
from fastapi_app.config.settings import settings
from fastapi_app.dependencies import auth as auth_module
from fastapi_app.main import create_app
from fastapi_app.middleware import api_key as api_key_module
from fastapi_app.routers import files as files_router
from fastapi_app.services.onlyoffice_file_service import OnlyOfficeFileService


class _FakeOnlyOfficeDownload:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeOnlyOfficeDownload":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_onlyoffice_config_and_callback_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(outputs_utils, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_SERVER_URL", "/onlyoffice")
    monkeypatch.setattr(settings, "ONLYOFFICE_THINKFLOW_PUBLIC_URL", "http://backend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL", "http://frontend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_JWT_SECRET", "")

    pptx_path = tmp_path / "outputs" / "tester" / "paper2ppt" / "123" / "editable.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    pptx_path.write_bytes(b"original-pptx")

    service = OnlyOfficeFileService()
    session_id = "session-1"

    payload = service.get_onlyoffice_config(
        path=str(pptx_path),
        request_base_url="http://backend.local",
        browser_base_url="http://frontend.local",
        editor_session_id=session_id,
    )

    assert payload["enabled"] is True
    assert payload["script_url"] == "/onlyoffice/web-apps/apps/api/documents/api.js"
    assert payload["config"]["document"]["fileType"] == "pptx"
    assert payload["config"]["document"]["key"] == service._onlyoffice_document_key(
        pptx_path,
        editor_session_id=session_id,
    )
    assert "/api/v1/files/onlyoffice/download/" in payload["config"]["document"]["url"]
    assert "/api/v1/files/onlyoffice/callback" in payload["config"]["editorConfig"]["callbackUrl"]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=60: _FakeOnlyOfficeDownload(b"saved-from-onlyoffice"),
    )

    result = service.handle_onlyoffice_callback(
        path=str(pptx_path),
        payload={
            "status": 2,
            "url": "http://document-server.local/download",
            "key": payload["config"]["document"]["key"],
        },
        editor_session_id=session_id,
    )

    assert result == {"error": 0}
    assert pptx_path.read_bytes() == b"saved-from-onlyoffice"


def test_onlyoffice_callback_accepts_same_session_after_file_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(outputs_utils, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_SERVER_URL", "/onlyoffice")
    monkeypatch.setattr(settings, "ONLYOFFICE_THINKFLOW_PUBLIC_URL", "http://backend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL", "http://frontend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_JWT_SECRET", "")

    pptx_path = tmp_path / "outputs" / "tester" / "paper2ppt" / "123" / "editable.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    pptx_path.write_bytes(b"original-pptx")

    service = OnlyOfficeFileService()
    session_id = "session-save"
    payload = service.get_onlyoffice_config(
        path=str(pptx_path),
        request_base_url="http://backend.local",
        browser_base_url="http://frontend.local",
        editor_session_id=session_id,
    )
    document_key = payload["config"]["document"]["key"]
    assert f"document_key={document_key}" in payload["config"]["editorConfig"]["callbackUrl"]

    downloads = iter([b"first-save-from-onlyoffice", b"second-save-from-onlyoffice"])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=60: _FakeOnlyOfficeDownload(next(downloads)),
    )

    first_result = service.handle_onlyoffice_callback(
        path=str(pptx_path),
        payload={
            "status": 6,
            "url": "http://document-server.local/download/1",
            "key": document_key,
        },
        document_key=document_key,
        editor_session_id=session_id,
    )
    assert first_result == {"error": 0}
    assert pptx_path.read_bytes() == b"first-save-from-onlyoffice"

    second_result = service.handle_onlyoffice_callback(
        path=str(pptx_path),
        payload={
            "status": 2,
            "url": "http://document-server.local/download/2",
            "key": document_key,
        },
        document_key=document_key,
        editor_session_id=session_id,
    )

    assert second_result == {"error": 0}
    assert pptx_path.read_bytes() == b"second-save-from-onlyoffice"


def test_onlyoffice_config_prefers_browser_base_url_for_document_download(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(outputs_utils, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_SERVER_URL", "/onlyoffice")
    monkeypatch.setattr(settings, "ONLYOFFICE_THINKFLOW_PUBLIC_URL", "http://backend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL", "http://host.docker.internal:8000")
    monkeypatch.setattr(settings, "ONLYOFFICE_JWT_SECRET", "")

    pptx_path = tmp_path / "outputs" / "tester" / "paper2ppt" / "123" / "editable.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    pptx_path.write_bytes(b"original-pptx")

    service = OnlyOfficeFileService()
    payload = service.get_onlyoffice_config(
        path=str(pptx_path),
        request_base_url="http://backend.local",
        browser_base_url="http://localhost:3000",
        editor_session_id="session-2",
    )

    assert payload["config"]["document"]["url"].startswith(
        "http://host.docker.internal:8000/api/v1/files/onlyoffice/download/"
    )


def test_onlyoffice_callback_rewrites_browser_proxy_download_url(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(outputs_utils, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_SERVER_URL", "/onlyoffice")
    monkeypatch.setattr(settings, "ONLYOFFICE_THINKFLOW_PUBLIC_URL", "http://backend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL", "http://frontend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE", "http://127.0.0.1:8082")
    monkeypatch.setattr(settings, "ONLYOFFICE_JWT_SECRET", "")

    pptx_path = tmp_path / "outputs" / "tester" / "paper2ppt" / "123" / "editable.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    pptx_path.write_bytes(b"original-pptx")

    service = OnlyOfficeFileService()
    session_id = "session-proxy"
    payload = service.get_onlyoffice_config(
        path=str(pptx_path),
        request_base_url="http://backend.local",
        browser_base_url="http://frontend.local",
        editor_session_id=session_id,
    )
    document_key = payload["config"]["document"]["key"]
    requested_urls: list[str] = []

    def fake_urlopen(url: str, timeout: int = 60) -> _FakeOnlyOfficeDownload:
        requested_urls.append(url)
        return _FakeOnlyOfficeDownload(b"saved-from-proxy-cache")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = service.handle_onlyoffice_callback(
        path=str(pptx_path),
        payload={
            "status": 2,
            "url": "http://localhost:13000/onlyoffice/cache/files/data/doc/output.pptx/output.pptx?md5=abc",
            "key": document_key,
        },
        document_key=document_key,
        editor_session_id=session_id,
    )

    assert result == {"error": 0}
    assert requested_urls == ["http://127.0.0.1:8082/cache/files/data/doc/output.pptx/output.pptx?md5=abc"]
    assert pptx_path.read_bytes() == b"saved-from-proxy-cache"


def test_files_routes_fallback_to_system_user_without_supabase(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_key_module, "API_KEY", "test-key")
    monkeypatch.setattr(outputs_utils, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(files_router, "OUTPUTS_ROOT", (tmp_path / "outputs").resolve())
    monkeypatch.setattr(auth_module, "get_supabase_client", lambda: None)
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_SERVER_URL", "/onlyoffice")
    monkeypatch.setattr(settings, "ONLYOFFICE_THINKFLOW_PUBLIC_URL", "http://backend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL", "http://frontend.local")
    monkeypatch.setattr(settings, "ONLYOFFICE_JWT_SECRET", "")

    client = TestClient(create_app())

    upload_response = client.post(
        "/api/v1/files/upload",
        headers={"X-API-Key": "test-key"},
        files={
            "file": (
                "paper2ppt_html_editable.pptx",
                b"pptx-bytes",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        },
        data={"workflow_type": "paper2ppt"},
    )

    assert upload_response.status_code == 200, upload_response.text
    upload_data = upload_response.json()
    assert upload_data["success"] is True
    assert "/outputs/system/paper2ppt/" in upload_data["file_path"]
    saved_path = Path(upload_data["file_path"])
    assert saved_path.exists()

    history_response = client.get("/api/v1/files/history", headers={"X-API-Key": "test-key"})
    assert history_response.status_code == 200, history_response.text
    history_data = history_response.json()
    assert history_data["success"] is True
    assert any(item["file_name"] == "paper2ppt_html_editable.pptx" for item in history_data["files"])

    onlyoffice_response = client.get(
        "/api/v1/files/onlyoffice/config",
        headers={"X-API-Key": "test-key"},
        params={"path": upload_data["file_path"], "browser_base_url": "http://frontend.local"},
    )
    assert onlyoffice_response.status_code == 200, onlyoffice_response.text
    onlyoffice_data = onlyoffice_response.json()
    assert onlyoffice_data["enabled"] is True
    assert onlyoffice_data["config"]["document"]["title"] == "paper2ppt_html_editable.pptx"
