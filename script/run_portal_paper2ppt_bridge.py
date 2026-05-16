#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from script.cli_env import find_output_artifacts, load_project_env

load_project_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ENV_FILE = PROJECT_ROOT / "frontend-workflow" / ".env"


def _load_extra_env(env_file: Path) -> None:
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_extra_env(FRONTEND_ENV_FILE)

from fastapi_app.config import settings
from fastapi_app.schemas import FrontendPPTGenerationRequest, Paper2PPTRequest
from fastapi_app.services.managed_api_service import (
    resolve_image_generation_credentials,
    resolve_llm_credentials,
    resolve_model_name,
)
from fastapi_app.services.paper2ppt_frontend_service import Paper2PPTFrontendService
from fastapi_app.workflow_adapters.wa_paper2ppt import (
    run_paper2page_content_wf_api,
    run_paper2ppt_wf_api,
)


def _print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portal bridge for Paper2PPT workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    outline = subparsers.add_parser("outline")
    outline.add_argument("--payload-json", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--payload-json", required=True)
    return parser.parse_args()


def _load_payload(path_str: str) -> dict[str, Any]:
    payload_path = Path(path_str).expanduser().resolve()
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _resolve_text_credentials(payload: dict[str, Any]) -> tuple[str, str]:
    api_url = (
        str(payload.get("api_url") or "").strip()
        or os.getenv("DF_API_URL", "").strip()
        or os.getenv("PORTAL_LLM_API_URL", "").strip()
        or "https://api.openai.com/v1"
    )
    api_key = (
        str(payload.get("api_key") or "").strip()
        or os.getenv("DF_API_KEY", "").strip()
        or os.getenv("PORTAL_LLM_API_KEY", "").strip()
    )
    return api_url, api_key


def _resolve_image_credentials(payload: dict[str, Any], api_url: str, api_key: str) -> tuple[str, str]:
    image_api_url = (
        str(payload.get("image_api_url") or "").strip()
        or os.getenv("DF_IMAGE_API_URL", "").strip()
        or os.getenv("PORTAL_IMAGE_API_URL", "").strip()
        or api_url
    )
    image_api_key = (
        str(payload.get("image_api_key") or "").strip()
        or os.getenv("DF_IMAGE_API_KEY", "").strip()
        or os.getenv("PORTAL_IMAGE_API_KEY", "").strip()
        or api_key
    )
    return image_api_url, image_api_key


def _normalize_input(source_mode: str, source: str) -> tuple[str, str]:
    if source_mode != "uploaded_file":
        return "TEXT", source
    candidate = Path(source).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"source file not found: {candidate}")
    suffix = candidate.suffix.lower()
    if suffix == ".pdf":
        return "PDF", str(candidate)
    if suffix in {".ppt", ".pptx"}:
        return "PPTX", str(candidate)
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "TEXT", candidate.read_text(encoding="utf-8", errors="ignore")
    return "TEXT", candidate.read_text(encoding="utf-8", errors="ignore")


def _build_request(
    payload: dict[str, Any],
    *,
    input_type: str,
    input_content: str,
) -> Paper2PPTRequest:
    api_url, api_key = _resolve_text_credentials(payload)
    image_api_url, image_api_key = _resolve_image_credentials(payload, api_url, api_key)
    if not api_key:
        raise ValueError("Paper2PPT requires api_key/DF_API_KEY/PORTAL_LLM_API_KEY.")
    return Paper2PPTRequest(
        language=str(payload.get("language") or "zh"),
        chat_api_url=api_url,
        chat_api_key=api_key,
        api_key=api_key,
        image_api_url=image_api_url,
        image_api_key=image_api_key,
        model=str(payload.get("model") or os.getenv("PORTAL_PAPER2ANY_MODEL") or settings.PAPER2PPT_DEFAULT_MODEL),
        gen_fig_model=str(payload.get("gen_fig_model") or os.getenv("PORTAL_PAPER2ANY_IMAGE_MODEL") or settings.PAPER2PPT_DEFAULT_IMAGE_MODEL),
        input_type=input_type,
        input_content=input_content,
        aspect_ratio=str(payload.get("aspect_ratio") or "16:9"),
        style=str(payload.get("style") or "academic"),
        use_long_paper=bool(payload.get("use_long_paper", False)),
        email="portal_paper2ppt@paper2any.local",
        page_count=int(payload.get("page_count") or 10),
        all_edited_down=bool(payload.get("all_edited_down", False)),
    )


def _normalize_pagecontent(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item if isinstance(item, dict) else {"content": item} for item in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item if isinstance(item, dict) else {"content": item} for item in parsed]
    raise ValueError("pagecontent must be a JSON list or a JSON-encoded list string")


def _normalize_frontend_outputs(result_path: Path, response: dict[str, Any]) -> dict[str, Any]:
    slides = response.get("slides", []) if isinstance(response.get("slides"), list) else []
    theme = response.get("theme", {}) if isinstance(response.get("theme"), dict) else {}
    slides_json_path = result_path / "frontend_slides.portal.json"
    theme_json_path = result_path / "frontend_theme.portal.json"
    preview_json_path = result_path / "portal_ppt_preview.json"
    slides_json_path.write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding="utf-8")
    theme_json_path.write_text(json.dumps(theme, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_json_path.write_text(
        json.dumps(
            {
                "slides": slides,
                "theme": theme,
                "result_path": str(result_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "slides_json_path": str(slides_json_path),
        "theme_json_path": str(theme_json_path),
        "preview_json_path": str(preview_json_path),
        "slide_count": len(slides),
    }


def _export_frontend_pptx(result_path: Path, slides_json_path: str, theme_json_path: str) -> str:
    output_path = (result_path / "paper2ppt_frontend_editable.pptx").resolve()
    export_cmd = [
        "npx",
        "tsx",
        "scripts/run_paper2ppt_structured_export_cli.ts",
        "--slides-json",
        slides_json_path,
        "--theme-json",
        theme_json_path,
        "--output",
        str(output_path),
        "--asset-base-url",
        os.getenv("PORTAL_PAPER2ANY_ASSET_BASE_URL", "http://127.0.0.1:8000"),
    ]
    subprocess.run(
        export_cmd,
        cwd=str(PROJECT_ROOT / "frontend-workflow"),
        env={**os.environ},
        check=True,
    )
    return str(output_path)


async def _run_outline(payload: dict[str, Any]) -> int:
    source_mode = str(payload.get("source_mode") or "uploaded_file")
    source = str(payload.get("source") or "").strip()
    if not source:
        raise ValueError("outline stage requires source")
    input_type, input_content = _normalize_input(source_mode, source)
    req = _build_request(payload, input_type=input_type, input_content=input_content)

    result_path_raw = str(payload.get("result_path") or "").strip()
    result_path = Path(result_path_raw).resolve() if result_path_raw else None
    response = await run_paper2page_content_wf_api(req, result_path=result_path)
    return _print_json(
        {
            "ok": bool(response.success),
            "result_path": response.result_path,
            "pagecontent": response.pagecontent,
            "pagecontent_count": len(response.pagecontent or []),
        }
    )


def _select_primary_output(output_variant: str, *, pptx_path: str, pdf_path: str) -> str:
    if output_variant == "editable":
        return pptx_path or pdf_path
    return pdf_path or pptx_path


async def _generate_frontend_preview(payload: dict[str, Any], *, pagecontent: list[dict[str, Any]], result_path: Path) -> dict[str, Any]:
    credential_scope = str(payload.get("credential_scope") or "paper2ppt")
    api_url = str(payload.get("api_url") or "").strip() or None
    api_key = str(payload.get("api_key") or "").strip() or None
    image_api_url = str(payload.get("image_api_url") or "").strip() or None
    image_api_key = str(payload.get("image_api_key") or "").strip() or None
    resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(api_url, api_key, scope=credential_scope)
    resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
        image_api_url,
        image_api_key,
        scope=credential_scope,
    )
    frontend_req = FrontendPPTGenerationRequest(
        result_path=str(result_path),
        pagecontent=json.dumps(pagecontent, ensure_ascii=False),
        chat_api_url=resolved_chat_api_url,
        api_key=resolved_api_key,
        credential_scope=credential_scope,
        email="portal_paper2ppt@paper2any.local",
        model=resolve_model_name(
            str(payload.get("frontend_model") or payload.get("model") or ""),
            managed_default=settings.PAPER2PPT_CONTENT_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_MODEL,
        ),
        language=str(payload.get("language") or "zh"),
        style=str(payload.get("style") or "academic"),
        include_images=bool(payload.get("include_images", False)),
        image_style=str(payload.get("image_style") or "academic_illustration"),
        image_model=resolve_model_name(
            str(payload.get("gen_fig_model") or ""),
            managed_default=settings.PAPER2PPT_IMAGE_GEN_MODEL,
            fallback_default=settings.PAPER2PPT_DEFAULT_IMAGE_MODEL,
        ),
        image_api_url=resolved_image_api_url,
        image_api_key=resolved_image_api_key,
    )
    service = Paper2PPTFrontendService()
    response = await service.generate_slides(req=frontend_req, request=None)
    outputs = _normalize_frontend_outputs(result_path, response)
    exported_pptx = ""
    try:
        exported_pptx = _export_frontend_pptx(
            result_path,
            outputs["slides_json_path"],
            outputs["theme_json_path"],
        )
    except Exception:
        exported_pptx = ""
    outputs["editable_pptx_path"] = exported_pptx
    return outputs


async def _run_generate(payload: dict[str, Any]) -> int:
    result_path = str(payload.get("result_path") or "").strip()
    if not result_path:
        raise ValueError("generate stage requires result_path")

    source = str(payload.get("source") or "").strip() or "Portal Paper2PPT"
    source_mode = str(payload.get("source_mode") or "text_outline")
    input_type, input_content = _normalize_input(source_mode, source) if source else ("TEXT", "Portal Paper2PPT")
    req = _build_request(payload, input_type=input_type, input_content=input_content)

    pagecontent = _normalize_pagecontent(payload.get("pagecontent") or "")
    if not pagecontent:
        raise ValueError("generate stage requires non-empty pagecontent")

    generate_req = req.model_copy(update={"all_edited_down": False})
    generate_resp = await run_paper2ppt_wf_api(
        req=generate_req,
        pagecontent=pagecontent,
        result_path=result_path,
        get_down=None,
    )

    final_req = req.model_copy(update={"all_edited_down": True})
    final_resp = await run_paper2ppt_wf_api(
        req=final_req,
        pagecontent=pagecontent,
        result_path=generate_resp.result_path or result_path,
        get_down=None,
    )

    final_result_path = final_resp.result_path or generate_resp.result_path or result_path
    result_root = Path(final_result_path).expanduser().resolve()
    pptx_path = str(final_resp.ppt_pptx_path or "").strip()
    pdf_path = str(final_resp.ppt_pdf_path or "").strip()

    if not pptx_path:
        pptx_candidates = find_output_artifacts(result_root, ("*.pptx", "*.ppt"))
        if pptx_candidates:
            pptx_path = str(pptx_candidates[0])
    if not pdf_path:
        pdf_candidates = find_output_artifacts(result_root, ("*.pdf",))
        if pdf_candidates:
            pdf_path = str(pdf_candidates[0])

    output_variant = str(payload.get("output_variant") or "editable")
    frontend_outputs: dict[str, Any] = {}
    if output_variant == "editable":
        frontend_outputs = await _generate_frontend_preview(payload, pagecontent=pagecontent, result_path=result_root)
        if not pptx_path:
            pptx_path = str(frontend_outputs.get("editable_pptx_path") or "").strip()

    primary_output = _select_primary_output(output_variant, pptx_path=pptx_path, pdf_path=pdf_path)
    ok = bool(primary_output)

    all_output_files = [
        str(path)
        for path in find_output_artifacts(
            result_root,
            ("*.pptx", "*.ppt", "*.pdf", "*.json", "*.png", "*.jpg", "*.jpeg"),
        )
    ]
    for extra_key in ("slides_json_path", "theme_json_path", "preview_json_path"):
        extra_path = str(frontend_outputs.get(extra_key) or "").strip()
        if extra_path and extra_path not in all_output_files:
            all_output_files.append(extra_path)

    return _print_json(
        {
            "ok": ok,
            "variant": output_variant,
            "result_path": str(result_root),
            "primary_output": primary_output,
            "ppt_pptx_path": pptx_path,
            "ppt_pdf_path": pdf_path,
            "pagecontent": final_resp.pagecontent or pagecontent,
            "all_output_files": all_output_files,
            "slides_json_path": frontend_outputs.get("slides_json_path"),
            "theme_json_path": frontend_outputs.get("theme_json_path"),
            "preview_json_path": frontend_outputs.get("preview_json_path"),
            "slide_count": frontend_outputs.get("slide_count", len(pagecontent)),
            "error": None if ok else f"no output artifact found under {result_root}",
        }
    )


async def _async_main() -> int:
    args = _parse_args()
    payload = _load_payload(args.payload_json)
    if args.command == "outline":
        return await _run_outline(payload)
    return await _run_generate(payload)


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except Exception as exc:
        return _print_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
