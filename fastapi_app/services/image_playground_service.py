from __future__ import annotations

import asyncio
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException, Request

from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async
from fastapi_app.config import settings
from fastapi_app.config.pricing import get_workflow_cost
from fastapi_app.dependencies import AuthUser
from fastapi_app.services.billing_service import BillingService
from fastapi_app.services.managed_api_service import resolve_image_generation_credentials
from fastapi_app.utils import _to_outputs_url, get_outputs_root

log = get_logger(__name__)

IMAGE_PLAYGROUND_ALLOWED_MODELS = (
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gpt-image-2",
    "gpt-image-2-all",
)
IMAGE_PLAYGROUND_ALLOWED_BATCH_COUNTS = (1, 2, 4, 8, 16)
IMAGE_PLAYGROUND_BATCH_CONCURRENCY = 4
IMAGE_PLAYGROUND_GEMINI_FLASH_ASPECT_RATIOS = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
)
IMAGE_PLAYGROUND_GEMINI_PRO_ASPECT_RATIOS = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
IMAGE_PLAYGROUND_GEMINI_FLASH_RESOLUTIONS = ("1K", "2K", "4K")
IMAGE_PLAYGROUND_GEMINI_PRO_RESOLUTIONS = ("1K", "2K", "4K")
IMAGE_PLAYGROUND_GPT_IMAGE_SIZES = (
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "1152x2048",
)
IMAGE_PLAYGROUND_GPT_IMAGE_QUALITIES = ("auto", "low", "medium", "high")
IMAGE_PLAYGROUND_VARIANT_CUES = (
    "Use a modular card-based composition with strong section separation.",
    "Emphasize a left-to-right pipeline layout with clear directional flow.",
    "Favor a hero-centric composition with one dominant focal module and supporting annotations.",
    "Prefer a grid-based infographic layout with balanced whitespace and tighter visual hierarchy.",
    "Use a publication-style diagram layout with restrained visual effects and stronger labels.",
    "Lean into a poster-like visual with bold contrast, large title space, and simpler supporting elements.",
    "Favor a zoomed-in composition that highlights the most important subsystem or mechanism.",
    "Use a wider panoramic composition with clearer stage transitions and grouped components.",
)
_PREVIEW_MAX_SIDE = 320
_PREVIEW_JPEG_QUALITY = 28
_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_LANCZOS = _PIL_RESAMPLING.LANCZOS


class ImagePlaygroundService:
    def __init__(self) -> None:
        self.outputs_root = get_outputs_root()
        self.unit_cost = max(1, int(get_workflow_cost("image_playground", default=1)))

    def _resolve_model(self, requested_model: Optional[str]) -> str:
        normalized = (requested_model or "").strip()
        fallback = (settings.IMAGE_PLAYGROUND_DEFAULT_IMAGE_MODEL or "").strip() or IMAGE_PLAYGROUND_ALLOWED_MODELS[0]
        resolved = normalized or fallback
        if resolved not in IMAGE_PLAYGROUND_ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail="Unsupported image playground model")
        return resolved

    def _resolve_batch_count(self, batch_count: Optional[int]) -> int:
        if batch_count is None:
            return 1
        resolved = int(batch_count)
        if resolved not in IMAGE_PLAYGROUND_ALLOWED_BATCH_COUNTS:
            raise HTTPException(status_code=400, detail="Unsupported image playground batch count")
        return resolved

    def _user_dir(self, user: Optional[AuthUser]) -> str:
        if user and not getattr(user, "is_anonymous", False):
            return user.id
        return "default"

    def _create_run_dir(self, user: Optional[AuthUser]) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.outputs_root / self._user_dir(user) / "image_playground" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _pick_allowed(self, requested: Optional[str], *, allowed: tuple[str, ...], default: str) -> str:
        normalized = (requested or "").strip()
        if normalized:
            if normalized not in allowed:
                raise HTTPException(status_code=400, detail=f"Unsupported image playground option: {normalized}")
            return normalized
        return default

    def _generation_kwargs(
        self,
        model: str,
        *,
        aspect_ratio: Optional[str],
        resolution: Optional[str],
        size: Optional[str],
        quality: Optional[str],
    ) -> dict[str, Any]:
        if model == "gemini-3.1-flash-image-preview":
            return {
                "aspect_ratio": self._pick_allowed(
                    aspect_ratio,
                    allowed=IMAGE_PLAYGROUND_GEMINI_FLASH_ASPECT_RATIOS,
                    default="16:9",
                ),
                "resolution": self._pick_allowed(
                    resolution,
                    allowed=IMAGE_PLAYGROUND_GEMINI_FLASH_RESOLUTIONS,
                    default="2K",
                ),
                "timeout": 120,
            }
        if model == "gemini-3-pro-image-preview":
            return {
                "aspect_ratio": self._pick_allowed(
                    aspect_ratio,
                    allowed=IMAGE_PLAYGROUND_GEMINI_PRO_ASPECT_RATIOS,
                    default="16:9",
                ),
                "resolution": self._pick_allowed(
                    resolution,
                    allowed=IMAGE_PLAYGROUND_GEMINI_PRO_RESOLUTIONS,
                    default="2K",
                ),
                "timeout": 300,
            }
        if model == "gpt-image-2":
            return {
                "size": self._pick_allowed(
                    size,
                    allowed=IMAGE_PLAYGROUND_GPT_IMAGE_SIZES,
                    default="2048x1152",
                ),
                "quality": self._pick_allowed(
                    quality,
                    allowed=IMAGE_PLAYGROUND_GPT_IMAGE_QUALITIES,
                    default="medium",
                ),
                "output_format": "png",
                "timeout": 360,
            }
        if model == "gpt-image-2-all":
            return {"response_format": "b64_json", "timeout": 120}
        raise HTTPException(status_code=400, detail="Unsupported image playground model")

    def _build_variant_prompt(self, prompt: str, index: int, batch_count: int) -> str:
        if batch_count <= 1:
            return prompt
        cue = IMAGE_PLAYGROUND_VARIANT_CUES[index % len(IMAGE_PLAYGROUND_VARIANT_CUES)]
        return (
            f"{prompt}\n\n"
            "Variant guidance:\n"
            f"- This is variant {index + 1} of {batch_count}.\n"
            "- Keep the same topic, language requirement, and scientific intent.\n"
            "- Produce one distinct composition compared with sibling variants.\n"
            f"- {cue}\n"
            "- Do not make a collage, contact sheet, or multiple alternatives inside one image."
        )

    def _image_has_alpha(self, image: Image.Image) -> bool:
        bands = image.getbands()
        if "A" in bands:
            return True
        if image.mode == "P":
            return "transparency" in image.info
        return False

    def _build_preview_image(self, run_dir: Path, source_path: Path, index: int) -> Path:
        preview_root = run_dir / "frontend_assets" / "previews"
        preview_root.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source_path) as original_img:
                original_img = ImageOps.exif_transpose(original_img)
                preview_img = original_img.copy()
                preview_img.thumbnail((_PREVIEW_MAX_SIDE, _PREVIEW_MAX_SIDE), _PIL_LANCZOS)
                has_alpha = self._image_has_alpha(preview_img)
                preview_ext = ".png" if has_alpha else ".jpg"
                preview_path = preview_root / f"image_{index + 1:02d}_preview{preview_ext}"
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
                return preview_path
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("[image_playground] failed to build preview for %s: %s", source_path, exc)
            return source_path

    def _build_zip_archive(self, run_dir: Path, successful_items: list[dict[str, Any]]) -> Optional[Path]:
        if len(successful_items) <= 1:
            return None
        zip_path = run_dir / f"image_playground_batch_{run_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in successful_items:
                file_path = Path(str(item["file_path"]))
                if file_path.exists():
                    zf.write(file_path, arcname=file_path.name)
        return zip_path if zip_path.exists() else None

    def _build_item_payload(self, *, index: int, output_path: Path, preview_path: Path, request: Request) -> dict[str, Any]:
        return {
            "index": index + 1,
            "image_url": _to_outputs_url(str(output_path), request),
            "preview_url": _to_outputs_url(str(preview_path), request),
            "file_path": str(output_path),
            "preview_path": str(preview_path),
            "file_name": output_path.name,
            "preview_file_name": preview_path.name,
            "variant_label": f"Variant {index + 1}",
        }

    def _write_meta(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def generate(
        self,
        *,
        prompt: str,
        model: Optional[str],
        chat_api_url: Optional[str],
        api_key: Optional[str],
        template_key: Optional[str],
        domain_key: Optional[str],
        aspect_ratio: Optional[str],
        resolution: Optional[str],
        size: Optional[str],
        quality: Optional[str],
        batch_count: Optional[int],
        request: Request,
        user: Optional[AuthUser],
    ) -> dict[str, Any]:
        normalized_prompt = (prompt or "").strip()
        if not normalized_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        resolved_model = self._resolve_model(model)
        resolved_batch_count = self._resolve_batch_count(batch_count)
        resolved_image_api_url, resolved_image_api_key = resolve_image_generation_credentials(
            chat_api_url,
            api_key,
            scope="image_playground",
        )
        if not resolved_image_api_url or not resolved_image_api_key:
            raise HTTPException(status_code=400, detail="Image generation credentials are required")

        generation_kwargs = self._generation_kwargs(
            resolved_model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            size=size,
            quality=quality,
        )
        run_dir = self._create_run_dir(user)
        request_meta_path = run_dir / "request.json"
        self._write_meta(
            request_meta_path,
            {
                "prompt": normalized_prompt,
                "model": resolved_model,
                "template_key": (template_key or "").strip(),
                "domain_key": (domain_key or "").strip(),
                "aspect_ratio": (aspect_ratio or "").strip(),
                "resolution": (resolution or "").strip(),
                "size": (size or "").strip(),
                "quality": (quality or "").strip(),
                "batch_count": resolved_batch_count,
                "unit_cost": self.unit_cost,
                "requested_points": self.unit_cost * resolved_batch_count,
            },
        )

        semaphore = asyncio.Semaphore(IMAGE_PLAYGROUND_BATCH_CONCURRENCY)

        async def _generate_single(index: int) -> dict[str, Any]:
            output_path = run_dir / f"image_{index + 1:02d}.png"
            variant_prompt = self._build_variant_prompt(normalized_prompt, index, resolved_batch_count)
            try:
                async with semaphore:
                    await generate_or_edit_and_save_image_async(
                        prompt=variant_prompt,
                        save_path=str(output_path),
                        api_url=resolved_image_api_url,
                        api_key=resolved_image_api_key,
                        model=resolved_model,
                        use_edit=False,
                        **generation_kwargs,
                    )
                if not output_path.exists():
                    raise RuntimeError("image generation finished without output")
                preview_path = self._build_preview_image(run_dir, output_path, index)
                return {
                    "success": True,
                    "index": index,
                    "prompt": variant_prompt,
                    "payload": self._build_item_payload(
                        index=index,
                        output_path=output_path,
                        preview_path=preview_path,
                        request=request,
                    ),
                }
            except Exception as exc:
                log.exception("[image_playground] variant %s failed", index + 1)
                return {
                    "success": False,
                    "index": index,
                    "prompt": variant_prompt,
                    "error": str(exc),
                }

        raw_results = await asyncio.gather(*[_generate_single(index) for index in range(resolved_batch_count)])
        successful_items = [item["payload"] for item in raw_results if item.get("success")]
        failed_items = [
            {
                "index": int(item.get("index", 0)) + 1,
                "error": str(item.get("error") or "Unknown error"),
            }
            for item in raw_results
            if not item.get("success")
        ]
        success_count = len(successful_items)
        failed_count = len(failed_items)
        zip_path = self._build_zip_archive(run_dir, successful_items)
        billing_warning: Optional[str] = None

        result_meta_path = run_dir / "result.json"
        self._write_meta(
            result_meta_path,
            {
                "success_count": success_count,
                "failed_count": failed_count,
                "images": successful_items,
                "errors": failed_items,
                "zip_path": str(zip_path) if zip_path else "",
            },
        )

        if success_count <= 0:
            detail = "All image generation attempts failed."
            if failed_items:
                preview_errors = "; ".join(item["error"] for item in failed_items[:2])
                detail = f"{detail} {preview_errors}"
            raise HTTPException(status_code=502, detail=detail)

        charge_amount = self.unit_cost * success_count
        if user and not getattr(user, "is_anonymous", False) and charge_amount > 0:
            event_key = f"image_playground_batch_{user.id}_{run_dir.name}"
            try:
                BillingService().consume_workflow(
                    workflow_type="image_playground",
                    amount=charge_amount,
                    user=user,
                    guest_id=getattr(request.state, "guest_id", None),
                    event_key=event_key,
                )
            except HTTPException as exc:
                detail = str(exc.detail) if getattr(exc, "detail", None) else str(exc)
                billing_warning = (
                    "Images were generated successfully, but automatic billing did not complete. "
                    f"Please refresh your balance. ({detail})"
                )
                log.error("[image_playground] billing failed for run=%s: %s", run_dir.name, detail)
            except Exception as exc:  # pragma: no cover - defensive logging
                billing_warning = (
                    "Images were generated successfully, but automatic billing did not complete. "
                    "Please refresh your balance."
                )
                log.exception("[image_playground] unexpected billing failure for run=%s: %s", run_dir.name, exc)

        first_item = successful_items[0]
        zip_url = _to_outputs_url(str(zip_path), request) if zip_path else ""
        log.info(
            "[image_playground] generated model=%s success=%s failed=%s run=%s",
            resolved_model,
            success_count,
            failed_count,
            run_dir,
        )
        return {
            "success": True,
            "image_url": first_item["image_url"],
            "file_path": first_item["file_path"],
            "file_name": first_item["file_name"],
            "model": resolved_model,
            "prompt": normalized_prompt,
            "workflow_type": "image_playground",
            "images": successful_items,
            "batch_count": resolved_batch_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "zip_path": zip_url,
            "zip_file_name": zip_path.name if zip_path else "",
            "billing_warning": billing_warning,
        }
