from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from dataflow_agent.agentroles import create_react_agent
from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async
from dataflow_agent.toolkits.multimodaltool.ppt_tool import (
    convert_images_dir_to_pdf_and_full_slide_ppt,
)
from fastapi_app.config import settings
from fastapi_app.schemas import (
    FrontendPPTExportRequest,
    FrontendPPTGenerationRequest,
    FrontendPPTReviewRequest,
)
from fastapi_app.services.managed_api_service import (
    resolve_image_generation_credentials,
    resolve_llm_credentials,
)
from fastapi_app.utils import _from_outputs_url, _to_outputs_url, resolve_outputs_path

log = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FIELD_PLACEHOLDER_RE = re.compile(r"\{\{(?:field|list):([a-zA-Z0-9_]+)\}\}")
_IMAGE_PLACEHOLDER_RE = re.compile(r"\{\{image:([a-zA-Z0-9_]+)\}\}")
_ATTRIBUTE_RE = re.compile(r'([^\s"\'<>/=]+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)
_FORBIDDEN_HTML_RE = re.compile(
    r"<\s*(script|iframe|img|video|audio|canvas|svg)\b|on[a-z]+\s*=",
    re.IGNORECASE,
)
_FORBIDDEN_CSS_RE = re.compile(
    r"@import|url\s*\(|(?:^|[,{])\s*(?:body|html|:root|#root)\b|position\s*:\s*fixed",
    re.IGNORECASE,
)
_SLIDE_GEN_SEMAPHORE = asyncio.Semaphore(4)
_IMAGE_GEN_SEMAPHORE = asyncio.Semaphore(2)
_THEME_FILENAME = "deck_theme.json"
_REFERENCE_SLIDE_LIMIT = 3
_DEFAULT_VISUAL_KEY = "main_visual"
_DEFAULT_VISUAL_KEYS = ("main_visual", "secondary_visual")
_MAX_INLINE_VISUAL_ASSETS = 2
_PREVIEW_MAX_SIDE = 1280
_PREVIEW_SMALL_FILE_BYTES = 900 * 1024
_PREVIEW_JPEG_QUALITY = 82
_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_LANCZOS = _PIL_RESAMPLING.LANCZOS


class Paper2PPTFrontendService:
    def __init__(self) -> None:
        from fastapi_app.services.paper2ppt_service import Paper2PPTService

        self._paper2ppt_service = Paper2PPTService()

    async def generate_slides(
        self,
        req: FrontendPPTGenerationRequest,
        request: Request | None,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        pagecontent = self._paper2ppt_service._parse_pagecontent_json(req.pagecontent)
        if not pagecontent:
            raise HTTPException(status_code=400, detail="pagecontent is required")

        slides_dir = base_dir / "frontend_slide_specs"
        slides_dir.mkdir(parents=True, exist_ok=True)

        credential_scope = self._paper2ppt_service._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )
        resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )

        deck_theme = await self._load_or_create_deck_theme(
            slides_dir=slides_dir,
            pagecontent=pagecontent,
            chat_api_url=resolved_chat_api_url,
            api_key=resolved_api_key,
            model=req.model,
            language=req.language,
            style=req.style,
        )
        current_slide = self._parse_json_text(req.current_slide, "current_slide")

        if req.page_id is not None:
            if req.page_id < 0 or req.page_id >= len(pagecontent):
                raise HTTPException(status_code=400, detail="page_id out of range")
            generated_slide = await self._generate_single_slide(
                base_dir=base_dir,
                slides_dir=slides_dir,
                pagecontent=pagecontent,
                slide_index=req.page_id,
                chat_api_url=resolved_chat_api_url,
                api_key=resolved_api_key,
                model=req.model,
                language=req.language,
                style=req.style,
                include_images=req.include_images,
                image_style=req.image_style,
                image_model=req.image_model,
                image_api_url=resolved_image_api_url,
                image_api_key=resolved_image_api_key,
                edit_prompt=req.edit_prompt,
                current_slide=current_slide,
                theme=deck_theme,
            )
            self._write_slide_spec(slides_dir, generated_slide)
            self._sync_deck_manifest(slides_dir)
            response_slide = self._externalize_slide_assets(generated_slide, request, base_dir=base_dir)
            return {
                "success": True,
                "slides": [response_slide],
                "result_path": str(base_dir),
                "theme": deck_theme,
                "parallel_generation": True,
            }

        skip_set: set[int] = set()
        if req.skip_slides:
            try:
                parsed = json.loads(req.skip_slides)
                if isinstance(parsed, list):
                    skip_set = {
                        int(item)
                        for item in parsed
                        if isinstance(item, (int, str)) and str(item).strip().isdigit()
                    }
            except (json.JSONDecodeError, TypeError, ValueError):
                skip_set = set()

        reused_slides: list[dict] = []
        if skip_set:
            log.info("[frontend] Incremental mode: skip_slides=%s", sorted(skip_set))
            valid_skip_set: set[int] = set()
            for idx in sorted(skip_set):
                spec_path = slides_dir / f"page_{idx:03d}.json"
                if not spec_path.exists():
                    log.warning("[frontend] Spec not found for slide %s, will regenerate", idx)
                    continue
                try:
                    content = await asyncio.to_thread(spec_path.read_text, encoding="utf-8")
                    reused_slides.append(json.loads(content))
                    valid_skip_set.add(idx)
                    log.info("[frontend] Reusing existing spec for slide %s", idx)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[frontend] Failed to load spec for slide %s: %s", idx, exc)
            skip_set = valid_skip_set

        tasks = [
            self._generate_single_slide(
                base_dir=base_dir,
                slides_dir=slides_dir,
                pagecontent=pagecontent,
                slide_index=index,
                chat_api_url=resolved_chat_api_url,
                api_key=resolved_api_key,
                model=req.model,
                language=req.language,
                style=req.style,
                include_images=req.include_images,
                image_style=req.image_style,
                image_model=req.image_model,
                image_api_url=resolved_image_api_url,
                image_api_key=resolved_image_api_key,
                edit_prompt=None,
                current_slide=None,
                theme=deck_theme,
            )
            for index in range(len(pagecontent))
            if index not in skip_set
        ]
        generated_slides = await asyncio.gather(*tasks)
        ordered_slides = sorted(
            list(generated_slides) + reused_slides,
            key=lambda item: int(item.get("page_num", 0)),
        )

        for slide in ordered_slides:
            self._write_slide_spec(slides_dir, slide)
        self._sync_deck_manifest(slides_dir)
        response_slides = [self._externalize_slide_assets(slide, request, base_dir=base_dir) for slide in ordered_slides]

        return {
            "success": True,
            "slides": response_slides,
            "result_path": str(base_dir),
            "theme": deck_theme,
            "parallel_generation": True,
        }

    async def export_slides(
        self,
        req: FrontendPPTExportRequest,
        screenshots: Sequence[UploadFile],
        request: Request | None,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        slides = self._paper2ppt_service._parse_pagecontent_json(req.slides)
        if not slides:
            raise HTTPException(status_code=400, detail="slides is required")
        if not screenshots:
            raise HTTPException(status_code=400, detail="screenshots are required")
        if len(screenshots) != len(slides):
            raise HTTPException(
                status_code=400,
                detail=f"slides count ({len(slides)}) does not match screenshots count ({len(screenshots)})",
            )

        specs_dir = base_dir / "frontend_slide_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        (specs_dir / "frontend_slides.edited.json").write_text(
            json.dumps(slides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        image_dir = base_dir / "frontend_ppt_pages"
        image_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in image_dir.glob("page_*.png"):
            stale_file.unlink(missing_ok=True)

        ordered_files = sorted(
            screenshots,
            key=lambda item: self._extract_page_index(item.filename or ""),
        )
        for index, screenshot in enumerate(ordered_files):
            target_path = image_dir / f"page_{index:03d}.png"
            target_path.write_bytes(await screenshot.read())

        export_dir = base_dir / "frontend_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = export_dir / "paper2ppt_frontend.pdf"
        pptx_path = export_dir / "paper2ppt_frontend.pptx"

        convert_images_dir_to_pdf_and_full_slide_ppt(
            input_dir=str(image_dir),
            output_pdf_path=str(pdf_path),
            output_pptx_path=str(pptx_path),
        )

        response = {
            "success": True,
            "result_path": str(base_dir),
            "ppt_pdf_path": str(pdf_path),
            "ppt_pptx_path": str(pptx_path),
        }

        if request is not None:
            response["ppt_pdf_path"] = _to_outputs_url(str(pdf_path), request)
            response["ppt_pptx_path"] = _to_outputs_url(str(pptx_path), request)
            response["all_output_files"] = self._paper2ppt_service._collect_output_files_as_urls(
                str(base_dir),
                request,
            )
        else:
            response["all_output_files"] = []

        return response

    async def review_slide(
        self,
        req: FrontendPPTReviewRequest,
        screenshot: UploadFile,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(req.result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        slide = self._parse_json_text(req.slide, "slide")
        if slide is None:
            raise HTTPException(status_code=400, detail="slide is required")

        credential_scope = self._paper2ppt_service._resolve_credential_scope(req.credential_scope)
        resolved_chat_api_url, resolved_api_key = resolve_llm_credentials(
            req.chat_api_url,
            req.api_key,
            scope=credential_scope,
        )

        screenshot_bytes = await screenshot.read()
        if not screenshot_bytes:
            raise HTTPException(status_code=400, detail="screenshot is empty")

        theme = self._load_deck_theme(base_dir / "frontend_slide_specs") or self._build_fallback_theme(
            language=req.language,
            style="",
        )
        local_layout_issues = self._parse_string_list(req.layout_issues)
        mime_type = screenshot.content_type or "image/png"
        data_url = f"data:{mime_type};base64,{base64.b64encode(screenshot_bytes).decode('utf-8')}"

        try:
            review_payload = await self._call_llm_json(
                chat_api_url=resolved_chat_api_url,
                api_key=resolved_api_key,
                model=settings.PAPER2PPT_VLM_MODEL or settings.PAPER2PPT_CONTENT_MODEL,
                messages=self._build_review_messages(
                    slide=slide,
                    theme=theme,
                    language=req.language,
                    data_url=data_url,
                    local_layout_issues=local_layout_issues,
                ),
                temperature=0.1,
                max_tokens=900,
                timeout_seconds=float(max(30, int(settings.PAPER2PPT_VLM_TIMEOUT_SECONDS or 90))),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[Paper2PPTFrontendService] Visual review degraded to local layout checks for %s: %s",
                str(slide.get("title") or slide.get("page_num") or "slide"),
                exc,
            )
            fallback_summary = (
                "视觉检查模型暂时不可用，已改用本地布局检测结果。"
                if str(req.language or "").lower().startswith("zh")
                else "Visual review model unavailable; fell back to local layout checks."
            )
            normalized = self._normalize_review_payload(
                payload={
                    "passed": not local_layout_issues,
                    "summary": fallback_summary,
                    "issues": [],
                    "repair_prompt": "",
                },
                slide=slide,
                local_layout_issues=local_layout_issues,
            )
            normalized["degraded"] = True
            normalized["warning"] = str(type(exc).__name__)
            normalized["success"] = True
            return normalized

        normalized = self._normalize_review_payload(payload=review_payload, slide=slide, local_layout_issues=local_layout_issues)
        normalized["success"] = True
        return normalized

    async def upload_asset(
        self,
        *,
        result_path: str,
        asset_key: str,
        upload: UploadFile,
        request: Request | None,
    ) -> Dict[str, Any]:
        base_dir = self._paper2ppt_service.resolve_result_path(result_path)
        if not base_dir.exists():
            raise HTTPException(status_code=400, detail=f"result_path not exists: {base_dir}")

        key = self._slugify(asset_key or _DEFAULT_VISUAL_KEY) or _DEFAULT_VISUAL_KEY
        suffix = Path(upload.filename or "").suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise HTTPException(status_code=400, detail="unsupported image format")

        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail="uploaded image is empty")

        target_dir = base_dir / "frontend_assets" / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = (target_dir / f"{key}_{uuid4().hex}{suffix}").resolve()
        target_path.write_bytes(payload)

        asset = self._finalize_visual_asset(
            base_dir=base_dir,
            asset={
                "key": key,
                "label": key.replace("_", " ").title(),
                "src": str(target_path),
                "alt": Path(upload.filename or target_path.name).stem,
                "source_type": "upload",
                "storage_path": str(target_path),
            },
        )
        return {
            "success": True,
            "asset": self._externalize_asset(asset, request, base_dir=base_dir),
            "result_path": str(base_dir),
        }

    async def _generate_single_slide(
        self,
        *,
        base_dir: Path,
        slides_dir: Path,
        pagecontent: List[Dict[str, Any]],
        slide_index: int,
        chat_api_url: str,
        api_key: str,
        model: str,
        language: str,
        style: str,
        include_images: bool,
        image_style: str,
        image_model: Optional[str],
        image_api_url: str,
        image_api_key: str,
        edit_prompt: Optional[str],
        current_slide: Optional[Dict[str, Any]],
        theme: Dict[str, Any],
    ) -> Dict[str, Any]:
        outline_item = pagecontent[slide_index]
        visual_assets = await self._prepare_visual_assets(
            base_dir=base_dir,
            outline_item=outline_item,
            slide_index=slide_index,
            include_images=include_images,
            image_style=image_style,
            image_model=image_model,
            image_api_url=image_api_url,
            image_api_key=image_api_key,
            chat_api_url=chat_api_url,
            api_key=api_key,
            model=model,
            theme=theme,
            current_slide=current_slide,
        )
        fallback_slide = self._build_fallback_slide(
            outline_item=outline_item,
            slide_index=slide_index,
            slide_count=len(pagecontent),
            theme=theme,
            visual_assets=visual_assets,
        )
        reference_slides = (
            self._load_reference_slides(
                slides_dir=slides_dir,
                exclude_page_num=slide_index + 1,
            )
            if (current_slide or edit_prompt)
            else []
        )
        deck_identity = self._build_deck_identity_summary(theme)
        messages = self._build_messages(
            outline_item=outline_item,
            slide_index=slide_index,
            slide_count=len(pagecontent),
            language=language,
            style=style,
            edit_prompt=edit_prompt,
            current_slide=current_slide,
            theme=theme,
            deck_identity=deck_identity,
            reference_slides=reference_slides,
            visual_assets=visual_assets,
        )

        try:
            async with _SLIDE_GEN_SEMAPHORE:
                raw_payload = await self._call_llm_json(
                    chat_api_url=chat_api_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=0.28 if ((current_slide or edit_prompt) and reference_slides) else 0.32 if (current_slide or edit_prompt) else 0.45,
                    max_tokens=3400,
                )
            normalized = self._normalize_slide_payload(
                payload=raw_payload,
                outline_item=outline_item,
                slide_index=slide_index,
                slide_count=len(pagecontent),
                theme=theme,
                visual_assets=visual_assets,
            )
            return normalized
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[Paper2PPTFrontendService] Falling back to default slide for page %s: %s",
                slide_index,
                exc,
            )
            fallback_slide["generation_note"] = (
                f"Fallback template used because frontend code generation failed: {exc}"
            )
            return fallback_slide

    async def _call_llm_json(
        self,
        *,
        chat_api_url: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.55,
        max_tokens: int = 3200,
        timeout_seconds: float = 180.0,
    ) -> Dict[str, Any]:
        api_url = chat_api_url.rstrip("/")
        target_url = api_url if api_url.endswith("/chat/completions") else f"{api_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        timeout = httpx.Timeout(timeout=timeout_seconds, connect=min(20.0, timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(target_url, json=payload, headers=headers)
        if response.status_code != 200:
            body = response.text[:400]
            raise HTTPException(
                status_code=502,
                detail=f"frontend slide generation failed ({response.status_code}): {body}",
            )

        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"invalid LLM response json: {exc}") from exc

        content = self._extract_message_content(data)
        parsed = self._extract_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object")
        return parsed

    def _extract_message_content(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("LLM response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                text_value = item.get("text")
                if isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                    parts.append(text_value["value"])
            return "\n".join(parts)
        return str(content)

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            match = _JSON_BLOCK_RE.search(text)
            if not match:
                raise
            return json.loads(match.group(0))

    async def _load_or_create_deck_theme(
        self,
        *,
        slides_dir: Path,
        pagecontent: List[Dict[str, Any]],
        chat_api_url: str,
        api_key: str,
        model: str,
        language: str,
        style: str,
    ) -> Dict[str, Any]:
        existing_theme = self._load_deck_theme(slides_dir)
        if existing_theme is not None:
            return existing_theme

        try:
            theme = await self._generate_deck_theme(
                pagecontent=pagecontent,
                chat_api_url=chat_api_url,
                api_key=api_key,
                model=model,
                language=language,
                style=style,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[Paper2PPTFrontendService] Failed to generate deck theme: %s", exc)
            theme = self._build_fallback_theme(language=language, style=style)

        self._write_deck_theme(slides_dir, theme)
        return theme

    async def _generate_deck_theme(
        self,
        *,
        pagecontent: List[Dict[str, Any]],
        chat_api_url: str,
        api_key: str,
        model: str,
        language: str,
        style: str,
    ) -> Dict[str, Any]:
        outline_summary = [
            {
                "page_num": index + 1,
                "title": item.get("title", ""),
                "layout_description": item.get("layout_description", ""),
                "key_points": (item.get("key_points") or [])[:3],
            }
            for index, item in enumerate(pagecontent[:12])
        ]
        payload = await self._call_llm_json(
            chat_api_url=chat_api_url,
            api_key=api_key,
            model=model,
            messages=self._build_theme_messages(
                outline_summary=outline_summary,
                language=language,
                style=style,
            ),
            temperature=0.3,
            max_tokens=1400,
        )
        return self._normalize_theme_payload(payload, language=language, style=style)

    async def _prepare_visual_assets(
        self,
        *,
        base_dir: Path,
        outline_item: Dict[str, Any],
        slide_index: int,
        include_images: bool,
        image_style: str,
        image_model: Optional[str],
        image_api_url: str,
        image_api_key: str,
        chat_api_url: str,
        api_key: str,
        model: str,
        theme: Dict[str, Any],
        current_slide: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        current_assets = self._normalize_visual_assets(
            current_slide.get("visual_assets") or current_slide.get("visualAssets") or []
            if isinstance(current_slide, dict)
            else [],
            base_dir=base_dir,
        )
        if current_assets:
            return current_assets

        if not include_images:
            return []

        asset_refs = self._collect_outline_asset_refs(outline_item)
        if asset_refs:
            paper_assets = await self._resolve_outline_assets(
                base_dir=base_dir,
                asset_refs=asset_refs,
                outline_item=outline_item,
                slide_index=slide_index,
                image_style=image_style,
                chat_api_url=chat_api_url,
                api_key=api_key,
                model=model,
            )
            if paper_assets:
                return paper_assets

        image_prompt = self._build_visual_asset_prompt(
            outline_item=outline_item,
            slide_index=slide_index,
            image_style=image_style,
            theme=theme,
        )
        generated_asset = await self._generate_visual_asset(
            base_dir=base_dir,
            slide_index=slide_index,
            prompt=image_prompt,
            image_style=image_style,
            image_model=image_model,
            image_api_url=image_api_url,
            image_api_key=image_api_key,
            outline_item=outline_item,
        )
        if generated_asset is not None:
            return [generated_asset]

        return [
            {
                "key": _DEFAULT_VISUAL_KEY,
                "label": "Main Visual",
                "src": "",
                "alt": str(outline_item.get("title") or f"Slide {slide_index + 1} visual").strip(),
                "source_type": "generated",
                "storage_path": "",
                "prompt": image_prompt,
                "style": image_style,
            }
        ]

    async def _resolve_outline_assets(
        self,
        *,
        base_dir: Path,
        asset_refs: List[str],
        outline_item: Dict[str, Any],
        slide_index: int,
        image_style: str,
        chat_api_url: str,
        api_key: str,
        model: str,
    ) -> List[Dict[str, Any]]:
        resolved_assets: List[Dict[str, Any]] = []
        for asset_index, asset_ref in enumerate(asset_refs[:_MAX_INLINE_VISUAL_ASSETS]):
            normalized_ref = str(asset_ref or "").strip()
            if not normalized_ref:
                continue

            if self._is_table_asset_ref(normalized_ref):
                resolved_asset_path = await self._resolve_table_asset_path(
                    base_dir=base_dir,
                    asset_ref=normalized_ref,
                    chat_api_url=chat_api_url,
                    api_key=api_key,
                    model=model,
                )
            else:
                resolved_asset_path = self._resolve_asset_path(base_dir=base_dir, asset_ref=normalized_ref)

            if not resolved_asset_path or not Path(resolved_asset_path).exists():
                continue

            resolved_assets.append(
                self._finalize_visual_asset(
                    base_dir=base_dir,
                    asset={
                        "key": self._build_visual_asset_key(asset_index),
                        "label": self._build_visual_asset_label(normalized_ref, asset_index),
                        "src": resolved_asset_path,
                        "alt": str(outline_item.get("title") or f"Slide {slide_index + 1} visual").strip(),
                        "source_type": "paper_asset",
                        "storage_path": resolved_asset_path,
                        "prompt": "",
                        "style": image_style,
                    },
                )
            )
        return resolved_assets

    async def _generate_visual_asset(
        self,
        *,
        base_dir: Path,
        slide_index: int,
        prompt: str,
        image_style: str,
        image_model: Optional[str],
        image_api_url: str,
        image_api_key: str,
        outline_item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not image_api_url or not image_api_key:
            return None

        target_dir = base_dir / "frontend_assets" / "generated"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = (target_dir / f"page_{slide_index:03d}_{_DEFAULT_VISUAL_KEY}.png").resolve()
        model_name = image_model or settings.PAPER2PPT_IMAGE_GEN_MODEL or settings.PAPER2PPT_DEFAULT_IMAGE_MODEL
        api_base = re.sub(r"/chat/completions/?$", "", image_api_url.rstrip("/"), flags=re.IGNORECASE)

        try:
            async with _IMAGE_GEN_SEMAPHORE:
                await generate_or_edit_and_save_image_async(
                    prompt=prompt,
                    save_path=str(target_path),
                    api_url=api_base,
                    api_key=image_api_key,
                    model=model_name,
                    use_edit=False,
                    aspect_ratio="16:9",
                    resolution="2K",
                    timeout=300,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[Paper2PPTFrontendService] Failed to generate frontend visual asset for page %s: %s",
                slide_index,
                exc,
            )
            return None

        return self._finalize_visual_asset(
            base_dir=base_dir,
            asset={
                "key": _DEFAULT_VISUAL_KEY,
                "label": "Main Visual",
                "src": str(target_path),
                "alt": str(outline_item.get("title") or f"Slide {slide_index + 1} visual").strip(),
                "source_type": "generated",
                "storage_path": str(target_path),
                "prompt": prompt,
                "style": image_style,
            },
        )

    def _build_visual_asset_prompt(
        self,
        *,
        outline_item: Dict[str, Any],
        slide_index: int,
        image_style: str,
        theme: Dict[str, Any],
    ) -> str:
        style_map = {
            "academic_illustration": "clean academic illustration with publication-grade composition",
            "realistic": "realistic but presentation-friendly illustration",
            "sci_fi": "restrained sci-fi research visual with clean lighting",
            "flat_infographic": "flat infographic-style illustration with simple shapes",
        }
        key_points = [
            str(item).strip()
            for item in (outline_item.get("key_points") or [])
            if str(item).strip()
        ][:4]
        palette = theme.get("palette") or {}
        return (
            "Create one supporting image for an academic presentation slide. "
            f"Page topic: {str(outline_item.get('title') or f'Slide {slide_index + 1}').strip()}. "
            f"Layout intent: {str(outline_item.get('layout_description') or '').strip()}. "
            f"Key points: {'; '.join(key_points) if key_points else 'keep it concise and presentation-friendly'}. "
            f"Visual style: {style_map.get(image_style, image_style or 'academic illustration')}. "
            f"Preferred palette anchors: background {palette.get('bg', '#0b1020')}, accent {palette.get('accent', '#f59e0b')}, text contrast {palette.get('text', '#e2e8f0')}. "
            "The image must fit inside a 16:9 slide-side visual panel. "
            "Do not put any text, letters, labels, logos, equations, UI chrome, watermark, or slide-like layout in the image. "
            "Focus on one clear subject or scene that supports the slide narrative."
        )

    def _collect_outline_asset_refs(self, outline_item: Dict[str, Any]) -> List[str]:
        collected: List[str] = []

        def _push(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for key in (
                    "asset_ref",
                    "assetRef",
                    "path",
                    "src",
                    "storage_path",
                    "storagePath",
                    "ref",
                    "name",
                ):
                    if key in value:
                        _push(value.get(key))
                        return
                return
            if isinstance(value, list):
                for item in value:
                    _push(item)
                return

            raw = _from_outputs_url(str(value or "").strip())
            if not raw:
                return
            parts = [part.strip() for part in re.split(r"[,\n]+", raw) if part.strip()]
            for part in parts:
                normalized = part.strip().strip('"').strip("'")
                if not normalized or normalized.lower() in {"null", "none", "n/a"}:
                    continue
                if normalized not in collected:
                    collected.append(normalized)

        for key in (
            "asset_ref",
            "assetRef",
            "asset",
            "asset_refs",
            "assetRefs",
            "assets",
            "visual_assets",
            "visualAssets",
        ):
            _push(outline_item.get(key))

        return collected[:_MAX_INLINE_VISUAL_ASSETS]

    def _build_visual_asset_key(self, asset_index: int) -> str:
        if 0 <= asset_index < len(_DEFAULT_VISUAL_KEYS):
            return _DEFAULT_VISUAL_KEYS[asset_index]
        return f"visual_{asset_index + 1}"

    def _build_visual_asset_label(self, asset_ref: str, asset_index: int) -> str:
        if self._is_table_asset_ref(asset_ref):
            return "Paper Table" if asset_index == 0 else f"Paper Table {asset_index + 1}"
        return "Main Visual" if asset_index == 0 else f"Supporting Visual {asset_index + 1}"

    def _is_table_asset_ref(self, asset_ref: Any) -> bool:
        text = str(asset_ref or "").strip().lower()
        return bool(text and re.search(r"\btable(?:[_\s-]*\d+)?\b", text))

    def _normalize_table_asset_key(self, asset_ref: Any) -> str:
        text = str(asset_ref or "").strip()
        if not text:
            return ""
        match = re.search(r"(\d+)", text)
        if match:
            return f"table_{match.group(1)}"
        return self._slugify(text) or text.lower().replace(" ", "_")

    async def _resolve_table_asset_path(
        self,
        *,
        base_dir: Path,
        asset_ref: str,
        chat_api_url: str,
        api_key: str,
        model: str,
    ) -> str:
        table_key = self._normalize_table_asset_key(asset_ref)
        if not table_key:
            return ""

        for root in (
            base_dir / "tables",
            base_dir / "table_images",
            base_dir / "input" / "auto" / "images",
        ):
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = root / f"{table_key}{ext}"
                if candidate.exists():
                    return str(candidate.resolve())

        for root in (base_dir / "tables", base_dir / "table_images", base_dir / "input" / "auto" / "images"):
            if not root.exists():
                continue
            matches = sorted(root.glob(f"{table_key}.*"))
            if matches:
                return str(matches[0].resolve())

        return await self._extract_table_asset(
            base_dir=base_dir,
            asset_ref=asset_ref,
            chat_api_url=chat_api_url,
            api_key=api_key,
            model=model,
        )

    async def _extract_table_asset(
        self,
        *,
        base_dir: Path,
        asset_ref: str,
        chat_api_url: str,
        api_key: str,
        model: str,
    ) -> str:
        mineru_output, mineru_root = self._load_mineru_context(base_dir)
        if not mineru_output:
            return ""

        try:
            state = Paper2FigureState(
                request=Paper2FigureRequest(
                    language="zh",
                    chat_api_url=chat_api_url or "",
                    chat_api_key=api_key or "",
                    api_key=api_key or "",
                    model=model or "gpt-5.1",
                )
            )
            state.result_path = str(base_dir)
            state.mineru_root = mineru_root
            state.minueru_output = mineru_output
            state.asset_ref = asset_ref

            agent = create_react_agent(
                name="table_extractor",
                model_name=model or None,
                temperature=0.1,
                max_retries=6,
                parser_type="json",
            )
            final_state = await agent.execute(state=state)
            table_img_path = str(getattr(final_state, "table_img_path", "") or "").strip()
            if table_img_path and Path(table_img_path).exists():
                return str(Path(table_img_path).resolve())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[Paper2PPTFrontendService] Failed to extract table asset %s for frontend slide: %s",
                asset_ref,
                exc,
            )
        return ""

    def _load_mineru_context(self, base_dir: Path) -> tuple[str, str]:
        primary_root = base_dir / "input" / "auto"
        search_roots = [primary_root]
        if base_dir.exists():
            for child in sorted(base_dir.glob("*/auto")):
                if child not in search_roots:
                    search_roots.append(child)

        for root in search_roots:
            if not root.exists():
                continue
            md_files = sorted(root.glob("*.md"))
            if not md_files:
                continue
            try:
                return md_files[0].read_text(encoding="utf-8"), str(root.resolve())
            except Exception:  # noqa: BLE001
                continue

        return "", str(primary_root.resolve())

    def _resolve_asset_path(self, *, base_dir: Path, asset_ref: str) -> str:
        raw = _from_outputs_url(str(asset_ref or "").strip())
        if not raw:
            return ""

        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            try:
                return str(resolve_outputs_path(candidate, must_exist=True, allow_files=True))
            except HTTPException:
                return ""

        search_paths = [
            base_dir / candidate,
            base_dir / "input" / candidate,
            base_dir / "input" / "auto" / candidate,
            base_dir / "input" / "auto" / "images" / candidate.name,
        ]
        for path in search_paths:
            if path.exists():
                return str(path.resolve())

        filename = candidate.name
        if filename:
            for root in [base_dir / "input" / "auto" / "images", base_dir / "input" / "auto", base_dir]:
                if not root.exists():
                    continue
                matches = list(root.rglob(filename))
                if matches:
                    return str(matches[0].resolve())

        return ""

    def _normalize_visual_assets(
        self,
        raw_assets: Any,
        *,
        base_dir: Path,
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw_assets, list):
            return []

        normalized: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for index, raw_asset in enumerate(raw_assets):
            if not isinstance(raw_asset, dict):
                continue
            key = self._slugify(raw_asset.get("key") or f"{_DEFAULT_VISUAL_KEY}_{index + 1}") or f"{_DEFAULT_VISUAL_KEY}_{index + 1}"
            if key in seen_keys:
                continue
            src = str(
                raw_asset.get("storage_path")
                or raw_asset.get("storagePath")
                or raw_asset.get("src")
                or ""
            ).strip()
            resolved_src = self._resolve_asset_path(base_dir=base_dir, asset_ref=src) if src else ""
            source_type = str(raw_asset.get("source_type") or raw_asset.get("sourceType") or "generated").strip()
            if source_type not in {"generated", "paper_asset", "upload"}:
                source_type = "generated"
            normalized.append(
                self._finalize_visual_asset(
                    base_dir=base_dir,
                    asset={
                        "key": key,
                        "label": str(raw_asset.get("label") or key.replace("_", " ").title()).strip(),
                        "src": resolved_src or "",
                        "alt": str(raw_asset.get("alt") or raw_asset.get("label") or key).strip(),
                        "source_type": source_type,
                        "storage_path": resolved_src or "",
                        "preview_storage_path": str(
                            raw_asset.get("preview_storage_path")
                            or raw_asset.get("previewStoragePath")
                            or raw_asset.get("preview_src")
                            or raw_asset.get("previewSrc")
                            or ""
                        ).strip(),
                        "original_src": str(raw_asset.get("original_src") or raw_asset.get("originalSrc") or "").strip(),
                        "prompt": str(raw_asset.get("prompt") or "").strip(),
                        "style": str(raw_asset.get("style") or "").strip(),
                    },
                )
            )
            seen_keys.add(key)
        return normalized

    def _build_theme_messages(
        self,
        *,
        outline_summary: List[Dict[str, Any]],
        language: str,
        style: str,
    ) -> List[Dict[str, str]]:
        system_prompt = """
You are defining a single deck-level visual theme for an academic HTML/CSS presentation.
Return JSON only. No markdown. No explanation.

Schema:
{
  "theme_name": "short id",
  "visual_mood": "one sentence",
  "palette": {
    "bg": "#0b1020",
    "panel": "rgba(15,23,42,0.92)",
    "primary": "#7dd3fc",
    "secondary": "#38bdf8",
    "accent": "#f59e0b",
    "text": "#e2e8f0",
    "muted": "#94a3b8"
  },
  "typography": {
    "title_font_stack": "font stack",
    "body_font_stack": "font stack",
    "eyebrow_size": 18,
    "title_size": 56,
    "summary_size": 26,
    "body_size": 24
  },
  "layout_rules": ["rule 1", "rule 2"],
  "component_rules": ["rule 1", "rule 2"],
  "theme_lock": {
    "must_keep": ["rule 1", "rule 2"],
    "preferred_layout_patterns": ["hero_with_side_card"],
    "component_signature": "one short sentence",
    "avoid": ["rule 1", "rule 2"]
  },
  "footer_text": "deck footer",
  "section_label_template": "Slide {page_num:02d}/{slide_count:02d}"
}

Requirements:
1. Theme must fit text-first academic slides on a 1600x900 canvas.
2. Use restrained, professional colors and a single coherent component language.
3. Keep typography practical. Titles should stay below 60px, body text below 28px.
4. Avoid references to images, charts, SVG, or external assets.
5. Optimize for consistency across all slides in the same deck.
6. The theme_lock must be concrete enough to prevent per-slide drift during later regeneration.
7. If style_prompt contains explicit color or material directions, translate them into the palette instead of ignoring them.
8. Do not default to cyan/teal accents unless the style_prompt clearly asks for them.
""".strip()

        user_payload = {
            "language": language,
            "style_prompt": style or "",
            "outline_summary": outline_summary,
        }

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Create one deck theme for this outline summary:\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

    def _build_messages(
        self,
        *,
        outline_item: Dict[str, Any],
        slide_index: int,
        slide_count: int,
        language: str,
        style: str,
        edit_prompt: Optional[str],
        current_slide: Optional[Dict[str, Any]],
        theme: Dict[str, Any],
        deck_identity: Dict[str, Any],
        reference_slides: List[Dict[str, Any]],
        visual_assets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_prompt = """
You are an expert academic slide frontend engineer.
Generate a single 16:9 presentation slide as HTML/CSS for a browser-based PPT editor.

Hard requirements:
1. Return JSON only. No markdown fences. No explanation.
2. Output schema:
{
  "title": "short string",
  "html_template": "HTML string",
  "css_code": "CSS string",
  "editable_fields": [
    {"key": "title", "label": "Title", "type": "text", "value": "..."},
    {"key": "summary", "label": "Summary", "type": "textarea", "value": "..."},
    {"key": "key_points", "label": "Key Points", "type": "list", "items": ["...", "..."]}
  ],
  "generation_note": "one short sentence"
}
3. Every visible text in html_template must come from placeholders only:
   - text/textarea fields: {{field:key}}
   - list fields: {{list:key}}
   - controlled images, when required: {{image:key}}
4. css_code must only target .slide-root and its descendants.
5. Do not use external assets, remote fonts, raw image URLs, svg, canvas, script, iframe, video or img tags.
6. The slide must fit inside a 1600x900 canvas with safe margins and no overflow.
7. Use the supplied deck theme so every page looks like the same presentation family.
8. Treat theme_lock as non-negotiable. Do not invent a new palette family, component language, or typography system.
9. Keep titles within 2 lines, with title font 42-60px and body text 18-28px.
10. Prefer grid/flex layouts over brittle absolute positioning.
11. If visual_assets are supplied, reserve layout space and place them using {{image:key}} placeholders. Never write a raw <img> tag yourself.
12. If visual_assets are empty, build a text-first slide using editable text blocks and CSS decoration only.
13. The HTML must contain a single .slide-root root element.
14. If reference deck slides are provided, preserve their shared component grammar, spacing rhythm, and card treatment.
15. Never put {{field:...}}, {{list:...}}, or {{image:...}} placeholders inside HTML attributes like aria-label, title, alt, data-*, href, or style. Placeholders may only appear in element content.
""".strip()

        outline_payload = {
            "slide_index_1based": slide_index + 1,
            "slide_count": slide_count,
            "language": language,
            "style_prompt": style or "",
            "outline_title": outline_item.get("title", ""),
            "layout_description": outline_item.get("layout_description", ""),
            "key_points": outline_item.get("key_points", []),
            "visual_assets": [
                {
                    "key": asset.get("key"),
                    "label": asset.get("label"),
                    "source_type": asset.get("source_type"),
                    "alt": asset.get("alt"),
                }
                for asset in visual_assets
            ],
            "deck_theme": theme,
            "theme_lock": theme.get("theme_lock") or self._build_theme_lock(theme),
        }

        user_sections = [
            "Create a slide based on this outline JSON:",
            json.dumps(outline_payload, ensure_ascii=False, indent=2),
            "Deck identity summary that must stay stable across the whole deck:",
            json.dumps(deck_identity, ensure_ascii=False, indent=2),
            (
                "Keep the slide text-editable and visually consistent with the shared deck theme. "
                "If visual_assets exist, include them as controlled image slots. "
                "If space is tight, simplify layout and tighten spacing instead of enlarging the canvas."
            ),
        ]

        if reference_slides:
            user_sections.extend(
                [
                    "Reference deck slides. Reuse their component grammar instead of inventing a new one:",
                    json.dumps(reference_slides, ensure_ascii=False, indent=2),
                ]
            )

        if current_slide:
            user_sections.extend(
                [
                    "Current slide JSON for reference:",
                    json.dumps(current_slide, ensure_ascii=False, indent=2),
                    (
                        "Preserve the same deck component grammar, accent usage, title rhythm, and spacing language "
                        "from the current slide unless the revision request explicitly changes structure."
                    ),
                ]
            )
        if edit_prompt:
            user_sections.append(f"Revision request: {edit_prompt}")

        user_sections.append(
            "Ensure the editable_fields fully cover all meaningful visible text shown on the slide."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

    def _build_review_messages(
        self,
        *,
        slide: Dict[str, Any],
        theme: Dict[str, Any],
        language: str,
        data_url: str,
        local_layout_issues: List[str],
    ) -> List[Dict[str, Any]]:
        system_prompt = """
You are a strict visual QA reviewer for 16:9 academic presentation slides.
Review a rendered slide screenshot and return JSON only. No markdown. No explanation.

Schema:
{
  "passed": true,
  "summary": "one short sentence",
  "issues": ["issue 1", "issue 2"],
  "repair_prompt": "precise instruction for regenerating the slide while keeping the same content and deck theme"
}

Check for:
1. Overflow, clipping, text too large, crowded spacing, broken alignment.
2. Missing hierarchy or inconsistent typography.
3. Visual inconsistency with the provided deck theme.
4. Weak use of the 16:9 canvas.

If there are any meaningful problems, set passed=false and provide a concrete repair_prompt.
""".strip()

        review_context = {
            "language": language,
            "deck_theme": theme,
            "slide_overview": self._summarize_slide_for_review(slide),
            "local_layout_issues": local_layout_issues[:6],
        }

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Review this rendered academic slide and keep the same content hierarchy during repair.\n\n"
                            f"{json.dumps(review_context, ensure_ascii=False, indent=2)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

    def _summarize_slide_for_review(self, slide: Dict[str, Any]) -> Dict[str, Any]:
        editable_fields = slide.get("editable_fields") or slide.get("editableFields") or []
        summarized_fields: List[Dict[str, Any]] = []
        if isinstance(editable_fields, list):
            for field in editable_fields[:10]:
                if not isinstance(field, dict):
                    continue
                field_type = str(field.get("type") or "text").strip()
                entry: Dict[str, Any] = {
                    "key": str(field.get("key") or "").strip(),
                    "type": field_type,
                }
                if field_type == "list":
                    entry["items"] = [
                        str(item).strip()
                        for item in (field.get("items") or [])
                        if str(item).strip()
                    ][:5]
                else:
                    entry["value"] = str(field.get("value") or "").strip()[:280]
                summarized_fields.append(entry)

        visual_assets = slide.get("visual_assets") or slide.get("visualAssets") or []
        summarized_assets: List[Dict[str, str]] = []
        if isinstance(visual_assets, list):
            for asset in visual_assets[:4]:
                if not isinstance(asset, dict):
                    continue
                summarized_assets.append(
                    {
                        "key": str(asset.get("key") or "").strip(),
                        "label": str(asset.get("label") or "").strip(),
                        "source_type": str(asset.get("source_type") or asset.get("sourceType") or "").strip(),
                    }
                )

        return {
            "page_num": slide.get("page_num") or slide.get("pageNum"),
            "title": str(slide.get("title") or "").strip(),
            "editable_fields": summarized_fields,
            "visual_assets": summarized_assets,
        }

    def _normalize_slide_payload(
        self,
        *,
        payload: Dict[str, Any],
        outline_item: Dict[str, Any],
        slide_index: int,
        slide_count: int,
        theme: Dict[str, Any],
        visual_assets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback_slide = self._build_fallback_slide(
            outline_item=outline_item,
            slide_index=slide_index,
            slide_count=slide_count,
            theme=theme,
            visual_assets=visual_assets,
        )
        html_template = payload.get("html_template") or payload.get("html") or ""
        css_code = payload.get("css_code") or payload.get("css") or ""
        if not isinstance(html_template, str) or not isinstance(css_code, str):
            return fallback_slide
        if len(html_template) > 16000 or len(css_code) > 20000:
            return fallback_slide
        if _FORBIDDEN_HTML_RE.search(html_template) or _FORBIDDEN_CSS_RE.search(css_code):
            return fallback_slide

        normalized_html = self._sanitize_html_template(html_template)
        normalized_css = self._sanitize_css(css_code, theme=theme)
        editable_fields = self._normalize_fields(
            payload.get("editable_fields"),
            outline_item=outline_item,
            slide_index=slide_index,
        )
        if not editable_fields:
            return fallback_slide

        normalized_html, attribute_warnings = self._sanitize_attribute_placeholders(
            normalized_html,
            editable_fields,
        )
        if attribute_warnings:
            log.warning(
                "[Paper2PPTFrontendService] Sanitized attribute placeholders for page %s: %s",
                slide_index + 1,
                ", ".join(attribute_warnings),
            )

        field_keys = {field["key"] for field in editable_fields}
        placeholders = set(_FIELD_PLACEHOLDER_RE.findall(normalized_html))
        image_placeholders = set(_IMAGE_PLACEHOLDER_RE.findall(normalized_html))
        asset_keys = {str(asset.get("key") or "").strip() for asset in visual_assets if str(asset.get("key") or "").strip()}
        if not placeholders:
            return fallback_slide
        if not placeholders.issubset(field_keys):
            return fallback_slide
        if image_placeholders and not image_placeholders.issubset(asset_keys):
            return fallback_slide
        if visual_assets and not image_placeholders:
            return fallback_slide

        title_value = (
            self._find_field_value(editable_fields, "title")
            or outline_item.get("title")
            or f"Slide {slide_index + 1}"
        )
        return {
            "slide_id": str(payload.get("slide_id") or slide_index + 1),
            "page_num": slide_index + 1,
            "title": str(payload.get("title") or title_value),
            "html_template": normalized_html,
            "css_code": normalized_css,
            "editable_fields": editable_fields,
            "visual_assets": visual_assets,
            "generation_note": str(payload.get("generation_note") or ""),
            "status": "done",
        }

    def _normalize_review_payload(
        self,
        *,
        payload: Dict[str, Any],
        slide: Dict[str, Any],
        local_layout_issues: List[str],
    ) -> Dict[str, Any]:
        issues = [
            str(item).strip()
            for item in (payload.get("issues") or [])
            if str(item).strip()
        ]
        combined_issues: List[str] = []
        for issue in [*local_layout_issues, *issues]:
            if issue and issue not in combined_issues:
                combined_issues.append(issue)

        passed = bool(payload.get("passed")) and not combined_issues
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary = "未发现明显版式问题。" if passed else "检测到需要修复的版式问题。"

        repair_prompt = str(payload.get("repair_prompt") or "").strip()
        if not passed and not repair_prompt:
            slide_title = str(slide.get("title") or "current slide").strip()
            repair_prompt = (
                f"Keep the same deck theme and the same slide topic '{slide_title}'. "
                "Fix overflow, oversized text, spacing, alignment, and readability issues. "
                "Preserve the editable text fields and keep the slide inside a clean 16:9 canvas. "
                f"Specific issues: {'; '.join(combined_issues) if combined_issues else 'general layout cleanup'}."
            )

        return {
            "passed": passed,
            "summary": summary,
            "issues": combined_issues,
            "repair_prompt": repair_prompt,
        }

    def _normalize_fields(
        self,
        raw_fields: Any,
        *,
        outline_item: Dict[str, Any],
        slide_index: int,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        outline_points = [
            str(item).strip()
            for item in (outline_item.get("key_points") or [])
            if str(item).strip()
        ]

        if isinstance(raw_fields, list):
            for raw_field in raw_fields:
                if not isinstance(raw_field, dict):
                    continue
                key = self._slugify(raw_field.get("key") or raw_field.get("label") or "")
                if not key or key in seen_keys:
                    continue
                field_type = str(raw_field.get("type") or "text").strip().lower()
                if field_type not in {"text", "textarea", "list"}:
                    field_type = "text"
                label = str(raw_field.get("label") or key.replace("_", " ").title())
                if field_type == "list":
                    items = [
                        str(item).strip()
                        for item in (raw_field.get("items") or [])
                        if str(item).strip()
                    ]
                    if not items:
                        items = outline_points[:4]
                    normalized.append(
                        {
                            "key": key,
                            "label": label,
                            "type": "list",
                            "value": "",
                            "items": items,
                        }
                    )
                else:
                    value = str(raw_field.get("value") or "").strip()
                    normalized.append(
                        {
                            "key": key,
                            "label": label,
                            "type": field_type,
                            "value": value,
                            "items": [],
                        }
                    )
                seen_keys.add(key)

        if "title" not in seen_keys:
            normalized.append(
                {
                    "key": "title",
                    "label": "Title",
                    "type": "text",
                    "value": str(outline_item.get("title") or f"Slide {slide_index + 1}"),
                    "items": [],
                }
            )
        if "summary" not in seen_keys:
            normalized.append(
                {
                    "key": "summary",
                    "label": "Summary",
                    "type": "textarea",
                    "value": str(
                        (outline_points[0] if outline_points else outline_item.get("layout_description") or "")
                    ).strip(),
                    "items": [],
                }
            )
        if "key_points" not in seen_keys:
            normalized.append(
                {
                    "key": "key_points",
                    "label": "Key Points",
                    "type": "list",
                    "value": "",
                    "items": outline_points[:4] or ["Summarize the key contribution"],
                }
            )
        return normalized

    def _build_fallback_theme(self, *, language: str, style: str) -> Dict[str, Any]:
        footer_text = "Paper2Any Frontend PPT"
        section_label_template = (
            "第 {page_num:02d}/{slide_count:02d} 页"
            if language.strip().lower().startswith("zh")
            else "Slide {page_num:02d}/{slide_count:02d}"
        )
        visual_mood = (
            style.strip()
            or (
                "Academic storytelling with calm contrast, concise hierarchy, and consistent card components."
            )
        )
        palette = self._resolve_palette_from_style(style)
        return {
            "theme_name": "scholarly_signal",
            "visual_mood": visual_mood,
            "palette": palette,
            "typography": {
                "title_font_stack": 'Georgia, "Times New Roman", serif',
                "body_font_stack": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
                "eyebrow_size": 18,
                "title_size": 56,
                "summary_size": 26,
                "body_size": 24,
            },
            "layout_rules": [
                "Keep 72px+ safe margins around major content.",
                "Prefer one dominant text area plus one supporting card or metrics block.",
                "Avoid more than two visual columns in a single slide.",
                "Reserve a quiet footer area for page identity or takeaway.",
            ],
            "component_rules": [
                "Use rounded cards with subtle borders and a restrained glow.",
                "Use one accent color only for emphasis, not for large fills.",
                "Keep text hierarchy clear with title, summary, and supporting bullets.",
            ],
            "theme_lock": {
                "must_keep": [
                    "Use only the deck palette colors for fills, borders, and emphasis.",
                    "Keep the same serif title style and sans body style across the deck.",
                    "Keep rounded translucent cards and a quiet footer treatment.",
                ],
                "preferred_layout_patterns": [
                    "hero_with_side_card",
                    "split_insight_grid",
                    "stacked_cards",
                    "timeline_overview",
                ],
                "component_signature": "Rounded refined cards, restrained accent usage, thin borders, and quiet academic spacing.",
                "avoid": [
                    "Do not introduce unrelated bright color families.",
                    "Do not use more than two main columns.",
                    "Do not use oversized billboard titles or poster-like full-bleed blocks.",
                ],
            },
            "footer_text": footer_text,
            "section_label_template": section_label_template,
        }

    def _resolve_palette_from_style(self, style: str) -> Dict[str, str]:
        style_text = (style or "").strip().lower()

        palette_presets = [
            (
                ("terracotta", "ivory", "象牙", "暖白", "赤陶", "赭"),
                {
                    "bg": "#f4efe6",
                    "panel": "rgba(255, 249, 241, 0.88)",
                    "primary": "#8c3b2a",
                    "secondary": "#d0a77d",
                    "accent": "#b85c38",
                    "text": "#2d2018",
                    "muted": "#6c5b4c",
                },
            ),
            (
                ("midnight", "navy", "午夜蓝", "海军蓝", "深海军", "electric blue"),
                {
                    "bg": "#0f172a",
                    "panel": "rgba(15, 23, 42, 0.92)",
                    "primary": "#93c5fd",
                    "secondary": "#60a5fa",
                    "accent": "#3b82f6",
                    "text": "#e5eefc",
                    "muted": "#b7c3d7",
                },
            ),
            (
                ("burgundy", "parchment", "酒红", "米白", "纸感", "墨黑"),
                {
                    "bg": "#f8f2e7",
                    "panel": "rgba(255, 248, 240, 0.9)",
                    "primary": "#7f1d1d",
                    "secondary": "#b45353",
                    "accent": "#991b1b",
                    "text": "#231815",
                    "muted": "#705c55",
                },
            ),
            (
                ("forest", "olive", "森林绿", "橄榄", "沙金", "sand gold"),
                {
                    "bg": "#f4f1e8",
                    "panel": "rgba(248, 245, 237, 0.9)",
                    "primary": "#355e3b",
                    "secondary": "#7c8f4e",
                    "accent": "#c89b5d",
                    "text": "#1f2a22",
                    "muted": "#5c685d",
                },
            ),
            (
                ("orange", "亮橙", "黑白灰", "monochrome"),
                {
                    "bg": "#f6f6f5",
                    "panel": "rgba(255, 255, 255, 0.9)",
                    "primary": "#2f2f34",
                    "secondary": "#71717a",
                    "accent": "#f97316",
                    "text": "#111111",
                    "muted": "#60646c",
                },
            ),
            (
                ("plum", "mist pink", "深紫红", "雾粉", "银灰"),
                {
                    "bg": "#f5eef2",
                    "panel": "rgba(255, 248, 251, 0.9)",
                    "primary": "#5c2346",
                    "secondary": "#9d6381",
                    "accent": "#c08497",
                    "text": "#241823",
                    "muted": "#6b5967",
                },
            ),
        ]

        for keywords, palette in palette_presets:
            if any(keyword in style_text for keyword in keywords):
                return palette

        return {
            "bg": "#0b1020",
            "panel": "rgba(15, 23, 42, 0.92)",
            "primary": "#7dd3fc",
            "secondary": "#38bdf8",
            "accent": "#f59e0b",
            "text": "#e2e8f0",
            "muted": "#94a3b8",
        }

    def _normalize_theme_payload(
        self,
        payload: Dict[str, Any],
        *,
        language: str,
        style: str,
    ) -> Dict[str, Any]:
        fallback = self._build_fallback_theme(language=language, style=style)
        palette_raw = payload.get("palette") or payload.get("color_palette") or {}
        typography_raw = payload.get("typography") or {}
        theme_lock_raw = payload.get("theme_lock") or {}

        def _clean_text(value: Any, default: str) -> str:
            text = str(value or "").strip()
            return text or default

        def _clean_color(value: Any, default: str) -> str:
            text = str(value or "").strip()
            return text or default

        def _clean_int(value: Any, default: int, min_value: int, max_value: int) -> int:
            try:
                parsed = int(float(value))
            except Exception:  # noqa: BLE001
                return default
            return max(min_value, min(max_value, parsed))

        def _clean_list(value: Any, defaults: List[str], limit: int = 6) -> List[str]:
            if isinstance(value, list):
                cleaned = [str(item).strip() for item in value if str(item).strip()]
                if cleaned:
                    return cleaned[:limit]
            return defaults[:limit]

        layout_rules = [
            str(item).strip()
            for item in (payload.get("layout_rules") or [])
            if str(item).strip()
        ][:6]
        component_rules = [
            str(item).strip()
            for item in (payload.get("component_rules") or [])
            if str(item).strip()
        ][:6]

        return {
            "theme_name": _clean_text(payload.get("theme_name"), fallback["theme_name"]),
            "visual_mood": _clean_text(payload.get("visual_mood"), fallback["visual_mood"]),
            "palette": {
                "bg": _clean_color(palette_raw.get("bg"), fallback["palette"]["bg"]),
                "panel": _clean_color(palette_raw.get("panel"), fallback["palette"]["panel"]),
                "primary": _clean_color(palette_raw.get("primary"), fallback["palette"]["primary"]),
                "secondary": _clean_color(palette_raw.get("secondary"), fallback["palette"]["secondary"]),
                "accent": _clean_color(palette_raw.get("accent"), fallback["palette"]["accent"]),
                "text": _clean_color(palette_raw.get("text"), fallback["palette"]["text"]),
                "muted": _clean_color(palette_raw.get("muted"), fallback["palette"]["muted"]),
            },
            "typography": {
                "title_font_stack": _clean_text(
                    typography_raw.get("title_font_stack"),
                    fallback["typography"]["title_font_stack"],
                ),
                "body_font_stack": _clean_text(
                    typography_raw.get("body_font_stack"),
                    fallback["typography"]["body_font_stack"],
                ),
                "eyebrow_size": _clean_int(
                    typography_raw.get("eyebrow_size"),
                    fallback["typography"]["eyebrow_size"],
                    12,
                    24,
                ),
                "title_size": _clean_int(
                    typography_raw.get("title_size"),
                    fallback["typography"]["title_size"],
                    42,
                    60,
                ),
                "summary_size": _clean_int(
                    typography_raw.get("summary_size"),
                    fallback["typography"]["summary_size"],
                    20,
                    30,
                ),
                "body_size": _clean_int(
                    typography_raw.get("body_size"),
                    fallback["typography"]["body_size"],
                    18,
                    28,
                ),
            },
            "layout_rules": layout_rules or fallback["layout_rules"],
            "component_rules": component_rules or fallback["component_rules"],
            "theme_lock": {
                "must_keep": _clean_list(
                    theme_lock_raw.get("must_keep"),
                    fallback["theme_lock"]["must_keep"],
                ),
                "preferred_layout_patterns": _clean_list(
                    theme_lock_raw.get("preferred_layout_patterns"),
                    fallback["theme_lock"]["preferred_layout_patterns"],
                ),
                "component_signature": _clean_text(
                    theme_lock_raw.get("component_signature"),
                    fallback["theme_lock"]["component_signature"],
                ),
                "avoid": _clean_list(
                    theme_lock_raw.get("avoid"),
                    fallback["theme_lock"]["avoid"],
                ),
            },
            "footer_text": _clean_text(payload.get("footer_text"), fallback["footer_text"]),
            "section_label_template": _clean_text(
                payload.get("section_label_template"),
                fallback["section_label_template"],
            ),
        }

    def _build_theme_lock(self, theme: Dict[str, Any]) -> Dict[str, Any]:
        fallback = self._build_fallback_theme(language="zh", style="")
        theme_lock = theme.get("theme_lock")
        if isinstance(theme_lock, dict):
            return {
                "must_keep": [
                    str(item).strip()
                    for item in (theme_lock.get("must_keep") or [])
                    if str(item).strip()
                ] or fallback["theme_lock"]["must_keep"],
                "preferred_layout_patterns": [
                    str(item).strip()
                    for item in (theme_lock.get("preferred_layout_patterns") or [])
                    if str(item).strip()
                ] or fallback["theme_lock"]["preferred_layout_patterns"],
                "component_signature": str(
                    theme_lock.get("component_signature")
                    or fallback["theme_lock"]["component_signature"]
                ).strip(),
                "avoid": [
                    str(item).strip()
                    for item in (theme_lock.get("avoid") or [])
                    if str(item).strip()
                ] or fallback["theme_lock"]["avoid"],
            }
        return fallback["theme_lock"]

    def _build_deck_identity_summary(self, theme: Dict[str, Any]) -> Dict[str, Any]:
        palette = theme.get("palette") or {}
        typography = theme.get("typography") or {}
        theme_lock = self._build_theme_lock(theme)
        return {
            "theme_name": str(theme.get("theme_name") or "deck_theme").strip(),
            "visual_mood": str(theme.get("visual_mood") or "").strip(),
            "palette_anchor": {
                "bg": str(palette.get("bg") or "").strip(),
                "primary": str(palette.get("primary") or "").strip(),
                "accent": str(palette.get("accent") or "").strip(),
                "text": str(palette.get("text") or "").strip(),
            },
            "typography_anchor": {
                "title_font_stack": str(typography.get("title_font_stack") or "").strip(),
                "body_font_stack": str(typography.get("body_font_stack") or "").strip(),
                "title_size": typography.get("title_size"),
                "body_size": typography.get("body_size"),
            },
            "must_keep": theme_lock.get("must_keep") or [],
            "preferred_layout_patterns": theme_lock.get("preferred_layout_patterns") or [],
            "component_signature": theme_lock.get("component_signature") or "",
            "avoid": theme_lock.get("avoid") or [],
        }

    def _externalize_asset(
        self,
        asset: Dict[str, Any],
        request: Request | None,
        *,
        base_dir: Path,
    ) -> Dict[str, Any]:
        normalized = self._finalize_visual_asset(base_dir=base_dir, asset=asset)
        storage_path = str(
            normalized.get("storage_path")
            or normalized.get("storagePath")
            or normalized.get("original_src")
            or normalized.get("originalSrc")
            or normalized.get("src")
            or ""
        ).strip()
        preview_storage_path = str(
            normalized.get("preview_storage_path")
            or normalized.get("previewStoragePath")
            or normalized.get("preview_src")
            or normalized.get("previewSrc")
            or ""
        ).strip()
        if storage_path:
            try:
                resolved_storage = str(resolve_outputs_path(storage_path, must_exist=False, allow_files=True))
            except HTTPException:
                resolved_storage = ""
            try:
                resolved_preview = (
                    str(resolve_outputs_path(preview_storage_path, must_exist=False, allow_files=True))
                    if preview_storage_path
                    else ""
                )
            except HTTPException:
                resolved_preview = ""
            if not resolved_preview and resolved_storage:
                resolved_preview = self._ensure_preview_asset(base_dir=base_dir, original_path=resolved_storage)
            normalized["storage_path"] = resolved_storage
            normalized["preview_storage_path"] = resolved_preview
            normalized["original_src"] = _to_outputs_url(resolved_storage, request) if (request is not None and resolved_storage) else resolved_storage
            normalized["preview_src"] = _to_outputs_url(resolved_preview, request) if (request is not None and resolved_preview) else (resolved_preview or normalized["original_src"])
            normalized["src"] = normalized["preview_src"] or normalized["original_src"]
        else:
            normalized["src"] = str(normalized.get("src") or "").strip()
            normalized["storage_path"] = ""
            normalized["preview_storage_path"] = ""
            normalized["preview_src"] = normalized["src"]
            normalized["original_src"] = normalized["src"]
        return normalized

    def _externalize_slide_assets(
        self,
        slide: Dict[str, Any],
        request: Request | None,
        *,
        base_dir: Path,
    ) -> Dict[str, Any]:
        normalized = dict(slide)
        raw_assets = normalized.get("visual_assets") or []
        if isinstance(raw_assets, list):
            normalized["visual_assets"] = [
                self._externalize_asset(asset, request, base_dir=base_dir)
                for asset in raw_assets
                if isinstance(asset, dict)
            ]
        return normalized

    def _finalize_visual_asset(
        self,
        *,
        base_dir: Path,
        asset: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = dict(asset)
        raw_storage_path = str(
            normalized.get("storage_path")
            or normalized.get("storagePath")
            or normalized.get("original_src")
            or normalized.get("originalSrc")
            or normalized.get("src")
            or ""
        ).strip()
        resolved_storage = self._resolve_asset_path(base_dir=base_dir, asset_ref=raw_storage_path) if raw_storage_path else ""

        raw_preview_path = str(
            normalized.get("preview_storage_path")
            or normalized.get("previewStoragePath")
            or normalized.get("preview_src")
            or normalized.get("previewSrc")
            or ""
        ).strip()
        resolved_preview = self._resolve_asset_path(base_dir=base_dir, asset_ref=raw_preview_path) if raw_preview_path else ""

        if resolved_storage:
            if not resolved_preview or not Path(resolved_preview).exists():
                resolved_preview = self._ensure_preview_asset(base_dir=base_dir, original_path=resolved_storage)
            normalized["storage_path"] = resolved_storage
            normalized["preview_storage_path"] = resolved_preview or ""
            normalized["original_src"] = resolved_storage
            normalized["preview_src"] = resolved_preview or resolved_storage
            normalized["src"] = resolved_preview or resolved_storage
        else:
            normalized["storage_path"] = ""
            normalized["preview_storage_path"] = ""
            normalized["original_src"] = str(normalized.get("original_src") or normalized.get("originalSrc") or normalized.get("src") or "").strip()
            normalized["preview_src"] = str(normalized.get("preview_src") or normalized.get("previewSrc") or normalized.get("src") or "").strip()
            normalized["src"] = normalized["preview_src"] or normalized["original_src"]
        return normalized

    def _ensure_preview_asset(
        self,
        *,
        base_dir: Path,
        original_path: str,
    ) -> str:
        source_path = Path(str(original_path or "").strip())
        if not source_path.exists() or not source_path.is_file():
            return ""
        if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return str(source_path)

        try:
            source_stat = source_path.stat()
        except OSError:
            return str(source_path)

        try:
            with Image.open(source_path) as original_img:
                original_img = ImageOps.exif_transpose(original_img)
                if max(original_img.size) <= _PREVIEW_MAX_SIDE and source_stat.st_size <= _PREVIEW_SMALL_FILE_BYTES:
                    return str(source_path)

                preview_root = base_dir / "frontend_assets" / "previews"
                preview_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1(
                    f"{source_path}|{source_stat.st_size}|{source_stat.st_mtime_ns}".encode("utf-8")
                ).hexdigest()[:16]
                has_alpha = self._image_has_alpha(original_img)
                preview_ext = ".png" if has_alpha else ".jpg"
                preview_path = (preview_root / f"{source_path.stem}_{digest}{preview_ext}").resolve()
                if preview_path.exists():
                    return str(preview_path)

                preview_img = original_img.copy()
                preview_img.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE), _PIL_LANCZOS)
                if has_alpha:
                    preview_img = preview_img.convert("RGBA")
                    preview_img.save(preview_path, format="PNG", optimize=True)
                else:
                    preview_img = preview_img.convert("RGB")
                    preview_img.save(
                        preview_path,
                        format="JPEG",
                        quality=_PREVIEW_JPEG_QUALITY,
                        optimize=True,
                        progressive=True,
                    )
                return str(preview_path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning(
                "[Paper2PPTFrontendService] Failed to build preview for %s: %s",
                source_path,
                exc,
            )
            return str(source_path)

    def _image_has_alpha(self, image: Image.Image) -> bool:
        bands = image.getbands()
        if "A" in bands:
            return True
        if image.mode == "P":
            return "transparency" in image.info
        return False

    def _load_reference_slides(
        self,
        *,
        slides_dir: Path,
        exclude_page_num: int,
    ) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []
        for path in sorted(slides_dir.glob("page_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue
            page_num = int(payload.get("page_num") or 0)
            if page_num <= 0 or page_num == exclude_page_num:
                continue
            references.append(payload)

        if len(references) > _REFERENCE_SLIDE_LIMIT:
            step = max(1, len(references) // _REFERENCE_SLIDE_LIMIT)
            references = references[::step][:_REFERENCE_SLIDE_LIMIT]

        return [self._summarize_reference_slide(slide) for slide in references]

    def _summarize_reference_slide(self, slide: Dict[str, Any]) -> Dict[str, Any]:
        html_template = str(slide.get("html_template") or "")
        css_code = str(slide.get("css_code") or "")
        editable_fields = slide.get("editable_fields") or []
        return {
            "page_num": int(slide.get("page_num") or 0),
            "title": str(slide.get("title") or "").strip(),
            "field_keys": [
                str(field.get("key") or "").strip()
                for field in editable_fields
                if isinstance(field, dict) and str(field.get("key") or "").strip()
            ][:10],
            "html_outline": self._extract_html_outline(html_template),
            "component_classes": self._extract_component_classes(html_template, css_code),
            "css_selectors": self._extract_css_selectors(css_code),
        }

    def _extract_html_outline(self, html_template: str, limit: int = 12) -> List[str]:
        cleaned = re.sub(r"\{\{(?:field|list):[^}]+\}\}", "field", html_template)
        cleaned = re.sub(r"\s+", " ", cleaned)
        outline = re.findall(r"<([a-z0-9]+)(?:[^>]*class=['\"]([^'\"]+)['\"])?", cleaned, flags=re.IGNORECASE)
        rows: List[str] = []
        for tag, class_name in outline:
            tag_name = tag.lower()
            class_token = ""
            if class_name:
                class_token = "." + ".".join(
                    item
                    for item in class_name.strip().split()
                    if item and not item.startswith("ppt-inline-editable")
                )
            value = f"{tag_name}{class_token}"
            if value not in rows:
                rows.append(value)
            if len(rows) >= limit:
                break
        return rows

    def _extract_component_classes(self, html_template: str, css_code: str, limit: int = 10) -> List[str]:
        tokens = re.findall(r"class=['\"]([^'\"]+)['\"]", html_template, flags=re.IGNORECASE)
        selector_tokens = re.findall(r"\.([a-zA-Z0-9_-]+)", css_code)
        ranked: List[str] = []
        for raw_group in tokens:
            for token in raw_group.split():
                token = token.strip()
                if not token or token == "slide-root" or token.startswith("ppt-inline-editable"):
                    continue
                if token not in ranked:
                    ranked.append(token)
        for token in selector_tokens:
            token = token.strip()
            if not token or token == "slide-root" or token.startswith("ppt-inline-editable"):
                continue
            if token not in ranked:
                ranked.append(token)
        return ranked[:limit]

    def _extract_css_selectors(self, css_code: str, limit: int = 8) -> List[str]:
        selectors = re.findall(r"([^{]+)\{", css_code)
        cleaned: List[str] = []
        for selector in selectors:
            normalized = " ".join(selector.split())
            normalized = re.sub(r"\s*,\s*", ", ", normalized)
            if not normalized:
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _build_fallback_slide(
        self,
        *,
        outline_item: Dict[str, Any],
        slide_index: int,
        slide_count: int,
        theme: Dict[str, Any],
        visual_assets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        palette = theme.get("palette") or self._build_fallback_theme(language="zh", style="")["palette"]
        typography = theme.get("typography") or {}
        visual_assets = (visual_assets or [])[:_MAX_INLINE_VISUAL_ASSETS]
        has_visual = bool(visual_assets)
        has_multi_visual = len(visual_assets) > 1
        key_points = [
            str(item).strip()
            for item in (outline_item.get("key_points") or [])
            if str(item).strip()
        ][:4]
        summary = key_points[0] if key_points else str(outline_item.get("layout_description") or "").strip()
        takeaway = key_points[-1] if key_points else "Refine the narrative in the editor"
        section_template = str(theme.get("section_label_template") or "Slide {page_num:02d}/{slide_count:02d}")
        try:
            eyebrow = section_template.format(page_num=slide_index + 1, slide_count=slide_count)
        except Exception:  # noqa: BLE001
            eyebrow = f"Slide {slide_index + 1:02d}/{slide_count:02d}"

        visual_markup = ""
        if has_visual:
            visual_markup = "\n".join(
                f'        <div class="visual-shell visual-shell-{asset_index + 1}">{{{{image:{asset.get("key") or self._build_visual_asset_key(asset_index)}}}}}</div>'
                for asset_index, asset in enumerate(visual_assets)
            )

        html_template = """
<div class="slide-root">
  <div class="slide-shell">
    <div class="grid-layer"></div>
    <div class="hero">
      <div class="hero-copy">
        <div class="eyebrow">{{field:eyebrow}}</div>
        <h1 class="title">{{field:title}}</h1>
        <p class="summary">{{field:summary}}</p>
        """ + (
            """
        <ul class="bullet-list compact">{{list:key_points}}</ul>
"""
            if has_visual
            else ""
        ) + """
      </div>
      """ + (
            """
      <div class="visual-card """ + ("visual-card-grid" if has_multi_visual else "") + """">
""" + visual_markup + """
      </div>
"""
            if has_visual
            else """
      <div class="stat-card">
        <div class="card-label">{{field:points_label}}</div>
        <ul class="bullet-list">{{list:key_points}}</ul>
      </div>
"""
        ) + """
    </div>
    <div class="footer-row">
      <div class="takeaway-card">
        <div class="takeaway-label">{{field:takeaway_label}}</div>
        <p class="takeaway-text">{{field:takeaway}}</p>
      </div>
      <div class="footer-tag">{{field:footer}}</div>
    </div>
  </div>
</div>
""".strip()

        css_code = f"""
.slide-root {{
  width: 100%;
  height: 100%;
  background:
    radial-gradient(circle at top right, {palette["secondary"]}33 0%, transparent 28%),
    radial-gradient(circle at bottom left, {palette["accent"]}22 0%, transparent 32%),
    {palette["bg"]};
  color: {palette["text"]};
  overflow: hidden;
}}
.slide-root * {{
  box-sizing: border-box;
}}
.slide-shell {{
  position: relative;
  width: 100%;
  height: 100%;
  padding: 68px 72px;
}}
.grid-layer {{
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px);
  background-size: 48px 48px;
  opacity: 0.22;
}}
.hero {{
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: {'1.08fr 0.92fr' if has_visual else '1.5fr 0.95fr'};
  gap: 28px;
  height: calc(100% - 120px);
}}
.hero-copy {{
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 20px;
}}
.eyebrow {{
  display: inline-flex;
  align-self: flex-start;
  padding: 8px 14px;
  border-radius: 999px;
  background: {palette["secondary"]}22;
  border: 1px solid {palette["primary"]}55;
  color: {palette["primary"]};
  font-size: {int(typography.get("eyebrow_size") or 18)}px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.title {{
  margin: 0;
  max-width: 880px;
  font-size: {int(typography.get("title_size") or 56)}px;
  line-height: 1.04;
  letter-spacing: -0.04em;
  font-family: {typography.get("title_font_stack") or 'Georgia, "Times New Roman", serif'};
}}
.summary {{
  margin: 0;
  max-width: 840px;
  font-size: {int(typography.get("summary_size") or 26)}px;
  line-height: 1.42;
  color: {palette["muted"]};
  white-space: pre-wrap;
  font-family: {typography.get("body_font_stack") or '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'};
}}
.stat-card, .takeaway-card, .visual-card {{
  border-radius: 28px;
  border: 1px solid {palette["primary"]}30;
  background: {palette["panel"]};
  box-shadow: 0 30px 60px rgba(15, 23, 42, 0.35);
  backdrop-filter: blur(10px);
}}
.stat-card {{
  align-self: center;
  padding: 28px;
}}
.visual-card {{
  padding: 18px;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}}
.visual-card.visual-card-grid {{
  justify-content: stretch;
}}
.visual-shell {{
  width: 100%;
  height: 100%;
  min-height: 384px;
  border-radius: 22px;
  overflow: hidden;
}}
.visual-card.visual-card-grid .visual-shell {{
  flex: 1 1 0;
  min-height: 160px;
}}
.visual-card.visual-card-grid .visual-shell-1 {{
  min-height: 236px;
}}
.card-label, .takeaway-label {{
  font-size: {int(typography.get("eyebrow_size") or 18)}px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {palette["primary"]};
  margin-bottom: 14px;
  font-weight: 700;
}}
.bullet-list {{
  margin: 0;
  padding-left: 26px;
  display: grid;
  gap: 14px;
  font-size: {int(typography.get("body_size") or 24)}px;
  line-height: 1.35;
  font-family: {typography.get("body_font_stack") or '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'};
}}
.bullet-list li {{
  color: {palette["text"]};
}}
.bullet-list.compact {{
  max-width: 720px;
  gap: 10px;
  font-size: {max(18, int(typography.get("body_size") or 24) - 2)}px;
}}
.footer-row {{
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 1.4fr auto;
  align-items: end;
  gap: 18px;
}}
.takeaway-card {{
  padding: 24px 28px;
}}
.takeaway-text {{
  margin: 0;
  font-size: {int(typography.get("body_size") or 24)}px;
  line-height: 1.4;
  color: {palette["text"]};
  white-space: pre-wrap;
  font-family: {typography.get("body_font_stack") or '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'};
}}
.footer-tag {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 220px;
  padding: 14px 18px;
  border-radius: 999px;
  border: 1px solid {palette["accent"]}55;
  color: {palette["accent"]};
  font-size: {int(typography.get("eyebrow_size") or 18)}px;
  font-weight: 700;
  background: rgba(15, 23, 42, 0.45);
}}
""".strip()

        editable_fields = [
            {
                "key": "eyebrow",
                "label": "Eyebrow",
                "type": "text",
                "value": eyebrow,
                "items": [],
            },
            {
                "key": "title",
                "label": "Title",
                "type": "text",
                "value": str(outline_item.get("title") or f"Slide {slide_index + 1}"),
                "items": [],
            },
            {
                "key": "summary",
                "label": "Summary",
                "type": "textarea",
                "value": summary,
                "items": [],
            },
            {
                "key": "key_points",
                "label": "Key Points",
                "type": "list",
                "value": "",
                "items": key_points or ["Summarize the page content here"],
            },
            {
                "key": "takeaway_label",
                "label": "Takeaway Label",
                "type": "text",
                "value": "Takeaway",
                "items": [],
            },
            {
                "key": "takeaway",
                "label": "Takeaway",
                "type": "textarea",
                "value": takeaway,
                "items": [],
            },
            {
                "key": "footer",
                "label": "Footer",
                "type": "text",
                "value": str(theme.get("footer_text") or "Paper2Any Frontend PPT"),
                "items": [],
            },
        ]
        if not has_visual:
            editable_fields.insert(
                3,
                {
                    "key": "points_label",
                    "label": "Points Label",
                    "type": "text",
                    "value": "Key Points",
                    "items": [],
                },
            )

        return {
            "slide_id": str(slide_index + 1),
            "page_num": slide_index + 1,
            "title": str(outline_item.get("title") or f"Slide {slide_index + 1}"),
            "html_template": html_template,
            "css_code": css_code,
            "editable_fields": editable_fields,
            "visual_assets": visual_assets,
            "generation_note": "Built-in fallback template",
            "status": "done",
        }

    def _sanitize_html_template(self, html_template: str) -> str:
        cleaned = re.sub(r"<\s*/?\s*(html|head|body)\b[^>]*>", "", html_template, flags=re.IGNORECASE)
        cleaned = re.sub(r"<script[\s\S]*?</script>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\son[a-z]+\s*=\s*(['\"]).*?\1", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\sstyle\s*=\s*(['\"]).*?\1", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = cleaned.strip()
        if 'class="slide-root"' not in cleaned and "class='slide-root'" not in cleaned:
            cleaned = f'<div class="slide-root">{cleaned}</div>'
        return cleaned

    def _sanitize_attribute_placeholders(
        self,
        html_template: str,
        editable_fields: Sequence[Dict[str, Any]],
    ) -> tuple[str, list[str]]:
        field_map = {
            str(field.get("key") or "").strip(): field
            for field in editable_fields
            if isinstance(field, dict) and str(field.get("key") or "").strip()
        }
        warnings: list[str] = []

        def _replace_attr(match: re.Match[str]) -> str:
            attr_name, quote, attr_value = match.groups()
            next_value = attr_value

            def _replace_field(token_match: re.Match[str]) -> str:
                field_key = str(token_match.group(1) or "").strip()
                field = field_map.get(field_key)
                if field is None:
                    return ""
                if str(field.get("type") or "") == "list":
                    raw_value = " • ".join(
                        str(item).strip()
                        for item in (field.get("items") or [])
                        if str(item).strip()
                    )
                else:
                    raw_value = str(field.get("value") or "")
                return html.escape(" ".join(raw_value.split()), quote=True)

            next_value = re.sub(r"\{\{field:([a-zA-Z0-9_]+)\}\}", _replace_field, next_value)
            next_value = re.sub(r"\{\{list:([a-zA-Z0-9_]+)\}\}", _replace_field, next_value)
            next_value = re.sub(r"\{\{image:([a-zA-Z0-9_]+)\}\}", "", next_value)

            if next_value != attr_value:
                warnings.append(attr_name)
                return f"{attr_name}={quote}{next_value}{quote}"
            return match.group(0)

        sanitized = _ATTRIBUTE_RE.sub(_replace_attr, html_template)
        return sanitized, sorted(set(warnings))

    def _sanitize_css(self, css_code: str, *, theme: Dict[str, Any]) -> str:
        cleaned = re.sub(r"/\*[\s\S]*?\*/", "", css_code)
        cleaned = re.sub(r"@import[^;]+;", "", cleaned, flags=re.IGNORECASE)

        def _clamp_font_size(match: re.Match[str]) -> str:
            prefix, value_raw, unit = match.groups()
            try:
                value = float(value_raw)
            except Exception:  # noqa: BLE001
                return match.group(0)
            if unit == "px":
                value = max(12.0, min(72.0, value))
                value_text = f"{value:.2f}".rstrip("0").rstrip(".")
            else:
                value = max(0.75, min(4.5, value))
                value_text = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{prefix}{value_text}{unit}"

        cleaned = re.sub(
            r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(px|rem)",
            _clamp_font_size,
            cleaned,
            flags=re.IGNORECASE,
        )
        guard_css = f"""
.slide-root {{
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  color: {(theme.get("palette") or {}).get("text", "#e2e8f0")};
}}
.slide-root * {{
  box-sizing: border-box;
}}
""".strip()
        return f"{cleaned.strip()}\n{guard_css}".strip()

    def _find_field_value(self, fields: Sequence[Dict[str, Any]], key: str) -> str:
        for field in fields:
            if field.get("key") == key and isinstance(field.get("value"), str):
                return field["value"]
        return ""

    def _load_deck_theme(self, slides_dir: Path) -> Optional[Dict[str, Any]]:
        theme_path = slides_dir / _THEME_FILENAME
        if not theme_path.exists():
            return None
        try:
            payload = json.loads(theme_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        return self._normalize_theme_payload(payload, language="zh", style="")

    def _write_deck_theme(self, slides_dir: Path, theme: Dict[str, Any]) -> None:
        (slides_dir / _THEME_FILENAME).write_text(
            json.dumps(theme, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_slide_spec(self, slides_dir: Path, slide: Dict[str, Any]) -> None:
        page_num = int(slide.get("page_num") or 0)
        target_path = slides_dir / f"page_{page_num - 1:03d}.json"
        target_path.write_text(
            json.dumps(slide, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _sync_deck_manifest(self, slides_dir: Path) -> None:
        slides: List[Dict[str, Any]] = []
        for path in sorted(slides_dir.glob("page_*.json")):
            try:
                slides.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
        manifest = {
            "theme": self._load_deck_theme(slides_dir),
            "slides": slides,
        }
        (slides_dir / "frontend_slides.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _parse_json_text(self, raw_text: Optional[str], field_name: str) -> Optional[Dict[str, Any]]:
        if raw_text is None or not raw_text.strip():
            return None
        try:
            data = json.loads(raw_text)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid {field_name} json: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
        return data

    def _parse_string_list(self, raw_text: Optional[str]) -> List[str]:
        if raw_text is None or not raw_text.strip():
            return []
        stripped = raw_text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:  # noqa: BLE001
            pass
        return [line.strip() for line in stripped.splitlines() if line.strip()]

    def _slugify(self, raw_value: Any) -> str:
        text = str(raw_value or "").strip().lower()
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")

    def _extract_page_index(self, filename: str) -> int:
        match = re.search(r"(\d+)", filename or "")
        if not match:
            return 10_000
        return int(match.group(1))
