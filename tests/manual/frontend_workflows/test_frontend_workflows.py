from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / "fastapi_app" / ".env"
if load_dotenv and ENV_FILE.is_file():
    load_dotenv(ENV_FILE, override=False)


CaseRunner = Callable[["SmokeConfig", requests.Session], dict[str, Any]]


@dataclass
class SmokeConfig:
    base_url: str
    backend_api_key: str
    llm_api_url: str
    llm_api_key: str
    image_gen_model: str
    image2drawio_vlm_model: str
    paper2drawio_model: str
    paper2video_model: str
    paper2video_tts_model: str
    paper2video_tts_voice_name: str
    connect_timeout: float
    read_timeout: float
    output_dir: Path

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        output_dir: Path | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 1800.0,
    ) -> "SmokeConfig":
        backend_api_key = (
            os.getenv("BACKEND_API_KEY", "").strip()
            or os.getenv("VITE_API_KEY", "").strip()
        )
        if not backend_api_key:
            raise RuntimeError("Missing BACKEND_API_KEY/VITE_API_KEY in environment or fastapi_app/.env")

        llm_api_url = (
            os.getenv("DF_API_URL", "").strip()
            or os.getenv("DEFAULT_LLM_API_URL", "").strip()
        )
        llm_api_key = os.getenv("DF_API_KEY", "").strip()
        if not llm_api_url or not llm_api_key:
            raise RuntimeError("Missing DF_API_URL/DEFAULT_LLM_API_URL or DF_API_KEY in environment")

        ts = int(time.time())
        resolved_output_dir = output_dir or (PROJECT_ROOT / "outputs" / "manual" / "frontend_workflows" / str(ts))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            base_url=(base_url or os.getenv("FRONTEND_WORKFLOW_TEST_BASE_URL", "http://127.0.0.1:8000")).rstrip("/"),
            backend_api_key=backend_api_key,
            llm_api_url=llm_api_url,
            llm_api_key=llm_api_key,
            image_gen_model=os.getenv("FRONTEND_WORKFLOW_TEST_IMAGE_MODEL", "gemini-3-pro-image-preview").strip(),
            image2drawio_vlm_model=os.getenv("FRONTEND_WORKFLOW_TEST_IMAGE2DRAWIO_VLM_MODEL", "gpt-4o").strip(),
            paper2drawio_model=os.getenv("FRONTEND_WORKFLOW_TEST_PAPER2DRAWIO_MODEL", "claude-sonnet-4-5-20250929").strip(),
            paper2video_model=os.getenv("FRONTEND_WORKFLOW_TEST_PAPER2VIDEO_MODEL", "gpt-4o").strip(),
            paper2video_tts_model=os.getenv("FRONTEND_WORKFLOW_TEST_PAPER2VIDEO_TTS_MODEL", "cosyvoice-v3-flash").strip(),
            paper2video_tts_voice_name=os.getenv("FRONTEND_WORKFLOW_TEST_PAPER2VIDEO_TTS_VOICE", "longanyang").strip(),
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            output_dir=resolved_output_dir,
        )

    def timeout(self, read_timeout: float | None = None) -> tuple[float, float]:
        return (self.connect_timeout, read_timeout or self.read_timeout)


def _fixture(path: str) -> Path:
    resolved = PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Fixture not found: {resolved}")
    return resolved


def _session(cfg: SmokeConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({"X-API-Key": cfg.backend_api_key})
    return session


def _healthcheck(cfg: SmokeConfig, session: requests.Session) -> None:
    resp = session.get(f"{cfg.base_url}/health", timeout=cfg.timeout(30.0))
    if resp.status_code != 200:
        raise AssertionError(f"Backend health check failed: {resp.status_code} {resp.text[:500]}")


def _save_text(cfg: SmokeConfig, name: str, content: str) -> Path:
    path = cfg.output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _save_bytes(cfg: SmokeConfig, name: str, content: bytes) -> Path:
    path = cfg.output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    except Exception:
        return resp.text[:1000]


def _expect_ok(resp: requests.Response, *, case_name: str) -> None:
    if resp.ok:
        return
    raise AssertionError(f"[{case_name}] HTTP {resp.status_code}: {_error_message(resp)}")


def _expect_json_success(resp: requests.Response, *, case_name: str) -> dict[str, Any]:
    _expect_ok(resp, case_name=case_name)
    data = resp.json()
    if not isinstance(data, dict):
        raise AssertionError(f"[{case_name}] Expected JSON object, got: {type(data)}")
    if data.get("success") is False:
        raise AssertionError(f"[{case_name}] success=false: {json.dumps(data, ensure_ascii=False)}")
    return data


def run_image2ppt(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    image_path = _fixture("tests/test_02.png")
    with image_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/image2ppt/generate",
            data={
                "email": "manual_frontend_smoke",
                "use_ai_edit": "true",
                "chat_api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "gen_fig_model": cfg.image_gen_model,
            },
            files={"image_file": (image_path.name, f, "image/png")},
            timeout=cfg.timeout(),
        )
    _expect_ok(resp, case_name="image2ppt")
    if not resp.content:
        raise AssertionError("[image2ppt] Empty PPT response body")
    output_path = _save_bytes(cfg, "image2ppt/output.pptx", resp.content)
    return {"case": "image2ppt", "pptx_path": str(output_path), "size_bytes": len(resp.content)}


def run_pdf2ppt(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    pdf_path = _fixture("tests/test_03.pdf")
    with pdf_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/pdf2ppt/generate",
            data={
                "email": "manual_frontend_smoke",
                "use_ai_edit": "true",
                "chat_api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "gen_fig_model": cfg.image_gen_model,
            },
            files={"pdf_file": (pdf_path.name, f, "application/pdf")},
            timeout=cfg.timeout(),
        )
    _expect_ok(resp, case_name="pdf2ppt")
    if not resp.content:
        raise AssertionError("[pdf2ppt] Empty PPT response body")
    output_path = _save_bytes(cfg, "pdf2ppt/output.pptx", resp.content)
    return {"case": "pdf2ppt", "pptx_path": str(output_path), "size_bytes": len(resp.content)}


def run_image2drawio(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    image_path = _fixture("tests/test_02.png")
    with image_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/image2drawio/generate",
            data={
                "chat_api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "gen_fig_model": cfg.image_gen_model,
                "vlm_model": cfg.image2drawio_vlm_model,
                "email": "manual_frontend_smoke",
            },
            files={"image_file": (image_path.name, f, "image/png")},
            timeout=cfg.timeout(),
        )
    data = _expect_json_success(resp, case_name="image2drawio")
    xml_content = (data.get("xml_content") or "").strip()
    if "<mxfile" not in xml_content:
        raise AssertionError("[image2drawio] Missing drawio xml_content")
    xml_path = _save_text(cfg, "image2drawio/output.drawio", xml_content)
    return {"case": "image2drawio", "xml_path": str(xml_path), "used_model": data.get("used_model")}


def run_paper2drawio_pdf(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    pdf_path = _fixture("tests/test_03.pdf")
    with pdf_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/paper2drawio/generate",
            data={
                "chat_api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "model": cfg.paper2drawio_model,
                "input_type": "PDF",
                "diagram_type": "auto",
                "diagram_style": "default",
                "language": "zh",
                "enable_vlm_validation": "false",
                "email": "manual_frontend_smoke",
            },
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=cfg.timeout(),
        )
    data = _expect_json_success(resp, case_name="paper2drawio_pdf")
    xml_content = (data.get("xml_content") or "").strip()
    if "<mxfile" not in xml_content:
        raise AssertionError("[paper2drawio_pdf] Missing drawio xml_content")
    xml_path = _save_text(cfg, "paper2drawio_pdf/output.drawio", xml_content)
    return {"case": "paper2drawio_pdf", "xml_path": str(xml_path), "used_model": data.get("used_model")}


def run_paper2figure_model_arch(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    pdf_path = _fixture("tests/test_03.pdf")
    with pdf_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/paper2figure/generate-json",
            data={
                "img_gen_model_name": cfg.image_gen_model,
                "chat_api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "input_type": "file",
                "email": "manual_frontend_smoke",
                "file_kind": "pdf",
                "graph_type": "model_arch",
                "language": "zh",
                "style": "cartoon",
                "figure_complex": "easy",
                "resolution": "2K",
            },
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=cfg.timeout(),
        )
    data = _expect_json_success(resp, case_name="paper2figure_model_arch")
    output_files = data.get("all_output_files") or []
    if not output_files:
        raise AssertionError("[paper2figure_model_arch] all_output_files is empty")
    json_path = _save_text(
        cfg,
        "paper2figure_model_arch/response.json",
        json.dumps(data, ensure_ascii=False, indent=2),
    )
    return {
        "case": "paper2figure_model_arch",
        "response_json": str(json_path),
        "output_files_count": len(output_files),
    }


def run_paper2video_subtitle(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    pptx_fixture = os.getenv("FRONTEND_WORKFLOW_TEST_PAPER2VIDEO_INPUT", "tests/paper2ppt_editable.pptx").strip()
    pptx_path = _fixture(pptx_fixture)
    with pptx_path.open("rb") as f:
        resp = session.post(
            f"{cfg.base_url}/api/v1/paper2video/generate-subtitle",
            data={
                "email": "manual_frontend_smoke",
                "api_key": cfg.llm_api_key,
                "chat_api_url": cfg.llm_api_url,
                "model": cfg.paper2video_model,
                "tts_model": cfg.paper2video_tts_model,
                "tts_voice_name": cfg.paper2video_tts_voice_name,
                "language": "zh",
                "talking_model": "liveportrait",
                "avatar_preset": "avatar2",
            },
            files={
                "file": (
                    pptx_path.name,
                    f,
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            },
            timeout=cfg.timeout(1200.0),
        )
    data = _expect_json_success(resp, case_name="paper2video_subtitle")
    script_pages = data.get("script_pages") or []
    if not script_pages:
        raise AssertionError("[paper2video_subtitle] Missing script_pages")
    non_empty_script_count = sum(
        1
        for item in script_pages
        if (item.get("script_text") or item.get("scriptText") or "").strip()
    )
    if non_empty_script_count == 0:
        raise AssertionError("[paper2video_subtitle] script_pages returned, but all script_text values are empty")
    json_path = _save_text(
        cfg,
        "paper2video/subtitle_response.json",
        json.dumps(data, ensure_ascii=False, indent=2),
    )
    return {
        "case": "paper2video_subtitle",
        "response_json": str(json_path),
        "result_path": data.get("result_path"),
        "script_pages": script_pages,
        "state_snapshot": data.get("state_snapshot"),
    }


def run_paper2video_full(cfg: SmokeConfig, session: requests.Session) -> dict[str, Any]:
    subtitle_result = run_paper2video_subtitle(cfg, session)
    resp = session.post(
        f"{cfg.base_url}/api/v1/paper2video/generate-video",
        data={
            "result_path": subtitle_result["result_path"],
            "script_pages": json.dumps(
                [
                    {
                        "page_num": item.get("page_num", item.get("pageNum", idx)),
                        "script_text": item.get("script_text", item.get("scriptText", "")),
                    }
                    for idx, item in enumerate(subtitle_result["script_pages"])
                ],
                ensure_ascii=False,
            ),
            "state_snapshot": json.dumps(subtitle_result["state_snapshot"], ensure_ascii=False),
            "email": "manual_frontend_smoke",
        },
        timeout=cfg.timeout(1800.0),
    )
    data = _expect_json_success(resp, case_name="paper2video_full")
    video_ref = data.get("video_url") or data.get("video_path") or ""
    if not video_ref:
        raise AssertionError("[paper2video_full] Missing video_url/video_path")
    json_path = _save_text(
        cfg,
        "paper2video/video_response.json",
        json.dumps(data, ensure_ascii=False, indent=2),
    )
    return {
        "case": "paper2video_full",
        "response_json": str(json_path),
        "video_ref": video_ref,
        "subtitle_result_path": subtitle_result["result_path"],
    }


CASE_REGISTRY: dict[str, CaseRunner] = {
    "image2ppt": run_image2ppt,
    "pdf2ppt": run_pdf2ppt,
    "image2drawio": run_image2drawio,
    "paper2drawio_pdf": run_paper2drawio_pdf,
    "paper2figure_model_arch": run_paper2figure_model_arch,
    "paper2video_subtitle": run_paper2video_subtitle,
    "paper2video_full": run_paper2video_full,
}


def run_cases(cfg: SmokeConfig, case_names: list[str]) -> list[dict[str, Any]]:
    session = _session(cfg)
    _healthcheck(cfg, session)
    results = []
    for case_name in case_names:
        runner = CASE_REGISTRY[case_name]
        started = time.time()
        result = runner(cfg, session)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        results.append(result)
    return results


def _selected_case_names(raw_case_names: list[str] | None) -> list[str]:
    if not raw_case_names:
        return list(CASE_REGISTRY.keys())

    selected: list[str] = []
    for raw in raw_case_names:
        for item in raw.split(","):
            name = item.strip()
            if not name:
                continue
            if name == "all":
                selected.extend(CASE_REGISTRY.keys())
                continue
            if name not in CASE_REGISTRY:
                raise KeyError(f"Unknown case: {name}. Available: {', '.join(CASE_REGISTRY)}")
            selected.append(name)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


@pytest.mark.parametrize("case_name", list(CASE_REGISTRY.keys()), ids=list(CASE_REGISTRY.keys()))
def test_frontend_workflow_case(case_name: str) -> None:
    cfg = SmokeConfig.from_env()
    results = run_cases(cfg, [case_name])
    assert len(results) == 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual frontend-workflow smoke tests against the live backend API.")
    parser.add_argument(
        "--case",
        action="append",
        help=f"Case name or comma-separated case list. Available: {', '.join(CASE_REGISTRY.keys())}, all",
    )
    parser.add_argument("--base-url", default=None, help="Backend base URL, default: http://127.0.0.1:8000")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="HTTP connect timeout seconds")
    parser.add_argument("--read-timeout", type=float, default=1800.0, help="HTTP read timeout seconds")
    parser.add_argument("--output-dir", default=None, help="Directory for saving smoke-test artifacts")
    args = parser.parse_args(argv)

    case_names = _selected_case_names(args.case)
    cfg = SmokeConfig.from_env(
        base_url=args.base_url,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    results = run_cases(cfg, case_names)
    print(json.dumps({"output_dir": str(cfg.output_dir), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
