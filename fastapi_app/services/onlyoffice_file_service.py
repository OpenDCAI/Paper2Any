from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Dict
import urllib.parse
import urllib.request

from fastapi import HTTPException
from fastapi.responses import FileResponse

from fastapi_app.config.settings import settings
from fastapi_app.utils import resolve_outputs_path


class OnlyOfficeFileService:
    """Build and handle ONLYOFFICE sessions for PPTX files stored under outputs."""

    def get_onlyoffice_config(
        self,
        *,
        path: str,
        request_base_url: str,
        browser_base_url: str = "",
        editor_session_id: str = "",
    ) -> Dict[str, Any]:
        document_server_url = str(settings.ONLYOFFICE_DOCUMENT_SERVER_URL or "").strip().rstrip("/")
        if not document_server_url:
            return {"enabled": False, "reason": "ONLYOFFICE_DOCUMENT_SERVER_URL is not configured"}

        pptx_path = self._resolve_pptx_path(path)
        callback_base_url = str(settings.ONLYOFFICE_THINKFLOW_PUBLIC_URL or request_base_url or "").strip().rstrip("/")
        document_base_url = str(
            settings.ONLYOFFICE_DOCUMENT_DOWNLOAD_BASE_URL
            or callback_base_url
            or request_base_url
            or browser_base_url
            or ""
        ).strip().rstrip("/")
        if not callback_base_url or not document_base_url:
            raise HTTPException(status_code=500, detail="ONLYOFFICE public URL is required")

        editor_session_id = str(editor_session_id or "").strip()
        document_key = self._onlyoffice_document_key(pptx_path, editor_session_id=editor_session_id)
        document_url = self._onlyoffice_document_download_url(
            path=pptx_path,
            document_key=document_key,
            base_url=document_base_url,
            editor_session_id=editor_session_id,
        )
        callback_url = self._onlyoffice_callback_url(
            path=pptx_path,
            document_key=document_key,
            base_url=callback_base_url,
            editor_session_id=editor_session_id,
        )

        config: Dict[str, Any] = {
            "documentType": "slide",
            "type": "desktop",
            "width": "100%",
            "height": "100%",
            "document": {
                "fileType": "pptx",
                "key": document_key,
                "title": pptx_path.name,
                "url": document_url,
                "permissions": {
                    "edit": True,
                    "download": True,
                    "print": True,
                },
            },
            "editorConfig": {
                "mode": "edit",
                "lang": "zh-CN",
                "callbackUrl": callback_url,
                "user": {
                    "id": "paper2any",
                    "name": "Paper2Any",
                },
                "customization": {
                    "forcesave": True,
                    "autosave": True,
                },
            },
        }
        token = self._onlyoffice_jwt(config)
        if token:
            config["token"] = token

        return {
            "enabled": True,
            "document_server_url": document_server_url,
            "script_url": f"{document_server_url}/web-apps/apps/api/documents/api.js",
            "config": config,
        }

    def get_onlyoffice_document_response(
        self,
        *,
        path: str,
        document_key: str,
        editor_session_id: str = "",
        method: str = "GET",
    ) -> FileResponse:
        pptx_path = self._resolve_pptx_path(path)
        expected_key = self._onlyoffice_document_key(
            pptx_path,
            editor_session_id=str(editor_session_id or "").strip(),
        )
        if str(document_key or "") != expected_key:
            raise HTTPException(status_code=404, detail="Editable PPTX version not found")
        return FileResponse(path=str(pptx_path), filename=pptx_path.name, method=method)

    def handle_onlyoffice_callback(
        self,
        *,
        path: str,
        payload: Dict[str, Any],
        document_key: str = "",
        editor_session_id: str = "",
    ) -> Dict[str, int]:
        status = int(payload.get("status") or 0)
        if status not in {2, 3, 6, 7}:
            return {"error": 0}

        pptx_path = self._resolve_pptx_path(path)
        callback_key = str(document_key or "").strip()
        expected_key = callback_key or self._onlyoffice_document_key(
            pptx_path,
            editor_session_id=str(editor_session_id or "").strip(),
        )
        if not expected_key or str(payload.get("key") or "") != expected_key:
            return {"error": 1}

        download_url = str(payload.get("url") or "").strip()
        if not download_url:
            return {"error": 1}
        download_url = self._rewrite_onlyoffice_download_url(download_url)

        tmp_path = pptx_path.with_suffix(".onlyoffice.tmp")
        try:
            with urllib.request.urlopen(download_url, timeout=60) as response:
                tmp_path.write_bytes(response.read())
            tmp_path.replace(pptx_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"error": 1}
        return {"error": 0}

    def _resolve_pptx_path(self, path: str) -> Path:
        pptx_path = resolve_outputs_path(
            path,
            must_exist=True,
            allow_files=True,
            allow_dirs=False,
        )
        if pptx_path.suffix.lower() != ".pptx":
            raise HTTPException(status_code=400, detail="ONLYOFFICE editing only supports PPTX files")
        return pptx_path

    def _onlyoffice_document_download_url(
        self,
        *,
        path: Path,
        document_key: str,
        base_url: str,
        editor_session_id: str = "",
    ) -> str:
        query_items = {
            "path": str(path),
        }
        if editor_session_id:
            query_items["editor_session_id"] = editor_session_id
        query = urllib.parse.urlencode(query_items)
        return f"{base_url}/api/v1/files/onlyoffice/download/{document_key}.pptx?{query}"

    def _onlyoffice_callback_url(
        self,
        *,
        path: Path,
        document_key: str,
        base_url: str,
        editor_session_id: str = "",
    ) -> str:
        query_items = {
            "path": str(path),
            "document_key": document_key,
        }
        if editor_session_id:
            query_items["editor_session_id"] = editor_session_id
        query = urllib.parse.urlencode(query_items)
        return f"{base_url}/api/v1/files/onlyoffice/callback?{query}"

    def _rewrite_onlyoffice_download_url(self, url: str) -> str:
        server_base = str(settings.ONLYOFFICE_SERVER_DOWNLOAD_URL_BASE or "").strip().rstrip("/")
        if not server_base:
            return url

        parsed = urllib.parse.urlparse(url)
        if not parsed.path.startswith("/onlyoffice/"):
            return url

        base = urllib.parse.urlparse(server_base)
        rewritten_path = parsed.path[len("/onlyoffice") :]
        return urllib.parse.urlunparse(
            (
                base.scheme,
                base.netloc,
                rewritten_path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def _onlyoffice_document_key(self, path: Path, *, editor_session_id: str = "") -> str:
        stat = path.stat()
        raw = (
            "paper2any-onlyoffice-v1:"
            f"{editor_session_id}:"
            f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _onlyoffice_jwt(self, payload: Dict[str, Any]) -> str:
        secret = str(settings.ONLYOFFICE_JWT_SECRET or "").strip()
        if not secret:
            return ""

        header = {"alg": "HS256", "typ": "JWT"}

        def encode(data: Dict[str, Any]) -> bytes:
            raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return base64.urlsafe_b64encode(raw).rstrip(b"=")

        signing_input = encode(header) + b"." + encode(payload)
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return (signing_input + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode("ascii")
