from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from dataflow_agent.toolkits.sam3_import_helper import add_site_packages_for_packages


def _ensure_sam3_importable() -> None:
    import sys

    add_site_packages_for_packages(("iopath",))

    candidates = []
    sam3_home = os.environ.get("SAM3_HOME", "").strip()
    if sam3_home:
        candidates.append(Path(sam3_home))

    project_root = Path(__file__).resolve().parents[3]
    candidates.append(project_root / "models" / "sam3-official" / "sam3")

    for path in candidates:
        if path.exists() and path.is_dir():
            p = str(path.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)


_ensure_sam3_importable()
from sam3.model_builder import build_sam3_image_model  # type: ignore
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore


class PredictRequest(BaseModel):
    image_path: str = Field(..., description="Path to the image that the server can read")
    prompts: List[str] = Field(..., min_items=1, description="Text prompts for SAM3")
    return_masks: bool = Field(False, description="Whether to return mask data")
    mask_format: Literal["rle", "png"] = Field(
        "rle", description="Mask format: run-length encoding or base64 png"
    )
    score_threshold: Optional[float] = Field(None, description="Override score threshold")
    epsilon_factor: Optional[float] = Field(None, description="Override polygon epsilon factor")
    min_area: Optional[int] = Field(None, description="Override minimum polygon area")


class PredictResponse(BaseModel):
    image_size: Dict[str, int]
    results: List[Dict]


def _encode_mask_rle(mask: np.ndarray) -> str:
    flat = mask.reshape(-1).astype(np.uint8)
    runs: List[int] = []
    last_val = flat[0]
    length = 1
    for val in flat[1:]:
        if val == last_val:
            length += 1
        else:
            runs.append(length)
            length = 1
            last_val = val
    runs.append(length)
    return ",".join(str(x) for x in runs)


def _encode_mask_png(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _extract_polygon(binary_mask: np.ndarray, epsilon_factor: float) -> Tuple[List[List[int]], float]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], 0.0

    max_cnt = None
    max_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area > max_area:
            max_area = area
            max_cnt = cnt

    if max_cnt is None or max_area <= 0:
        return [], 0.0

    epsilon = epsilon_factor * cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    if approx is None or len(approx) < 3:
        return [], 0.0
    return approx.reshape(-1, 2).tolist(), max_area


def _calculate_area(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


class Sam3Runtime:
    def __init__(
        self,
        checkpoint_path: str,
        bpe_path: Optional[str] = None,
        score_threshold: float = 0.5,
        epsilon_factor: float = 0.02,
        min_area: int = 100,
        device: str = "cuda",
        cache_size: int = 2,
    ) -> None:
        self.score_threshold = score_threshold
        self.epsilon_factor = epsilon_factor
        self.min_area = min_area

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
        )
        self.processor = Sam3Processor(self.model, device=device)

        self.cache_size = cache_size
        self.state_cache: OrderedDict[str, Dict] = OrderedDict()
        self.cache_lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()

    async def _get_image_state(self, image_path: str) -> Dict:
        async with self.cache_lock:
            if image_path in self.state_cache:
                self.state_cache.move_to_end(image_path)
                return self.state_cache[image_path]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")
        canvas_size = pil_image.size

        image_state = self.processor.set_image(pil_image)
        cache_item = {
            "image_state": image_state,
            "canvas_size": canvas_size,
        }

        async with self.cache_lock:
            self.state_cache[image_path] = cache_item
            if len(self.state_cache) > self.cache_size:
                self.state_cache.popitem(last=False)
        return cache_item

    def _build_detection(
        self,
        prompt: str,
        score: float,
        bbox: List[int],
        polygon: List[List[int]],
        mask_payload: Optional[str],
        mask_format: Optional[str],
        mask_shape: Optional[List[int]],
    ) -> Dict:
        item: Dict = {
            "prompt": prompt,
            "score": score,
            "bbox": bbox,
            "polygon": polygon,
            "area": _calculate_area(bbox),
        }
        if mask_payload is not None and mask_format is not None and mask_shape is not None:
            item["mask"] = {
                "data": mask_payload,
                "format": mask_format,
                "shape": mask_shape,
            }
        return item

    async def predict(self, payload: PredictRequest) -> PredictResponse:
        async with self.inference_lock:
            cache_item = await self._get_image_state(payload.image_path)
            state = cache_item["image_state"]
            canvas_w, canvas_h = cache_item["canvas_size"]

            score_threshold = payload.score_threshold or self.score_threshold
            epsilon_factor = payload.epsilon_factor or self.epsilon_factor
            min_area = payload.min_area or self.min_area

            all_results: List[Dict] = []

            for prompt in payload.prompts:
                self.processor.reset_all_prompts(state)
                result_state = self.processor.set_text_prompt(prompt=prompt, state=state)
                masks = result_state.get("masks", [])
                boxes = result_state.get("boxes", [])
                scores = result_state.get("scores", [])

                if masks is None or len(masks) == 0:
                    continue

                num_masks = masks.shape[0] if isinstance(masks, torch.Tensor) else len(masks)
                for i in range(num_masks):
                    score_val = scores[i]
                    score_val = score_val.item() if hasattr(score_val, "item") else float(score_val)
                    if score_val < score_threshold:
                        continue

                    box = boxes[i]
                    bbox = box.detach().cpu().numpy().tolist() if isinstance(box, torch.Tensor) else box
                    bbox = [int(v) for v in bbox]

                    mask = masks[i]
                    binary_mask = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
                    if binary_mask.ndim > 2:
                        binary_mask = binary_mask.squeeze()
                    binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255

                    polygon, polygon_area = _extract_polygon(binary_mask, epsilon_factor)
                    if len(polygon) == 0 or polygon_area < min_area:
                        continue

                    mask_payload = None
                    mask_shape = None
                    if payload.return_masks:
                        mask_shape = [binary_mask.shape[0], binary_mask.shape[1]]
                        if payload.mask_format == "png":
                            mask_payload = _encode_mask_png(binary_mask)
                        else:
                            mask_payload = _encode_mask_rle(binary_mask)

                    all_results.append(
                        self._build_detection(
                            prompt=prompt,
                            score=score_val,
                            bbox=bbox,
                            polygon=polygon,
                            mask_payload=mask_payload,
                            mask_format=payload.mask_format if payload.return_masks else None,
                            mask_shape=mask_shape,
                        )
                    )

            return PredictResponse(
                image_size={"width": canvas_w, "height": canvas_h},
                results=all_results,
            )


def create_app(runtime: Sam3Runtime) -> FastAPI:
    app = FastAPI(title="SAM3 Model Server", version="1.0.0")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        try:
            return await runtime.predict(request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a persistent SAM3 HTTP service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("SAM3_CHECKPOINT_PATH", str(Path(__file__).resolve().parents[3] / "models" / "sam3" / "sam3.pt")),
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--bpe",
        default=os.getenv("SAM3_BPE_PATH", str(Path(__file__).resolve().parents[3] / "models" / "sam3" / "bpe_simple_vocab_16e6.txt.gz")),
        help="Path to SAM3 BPE file",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--epsilon-factor", type=float, default=0.02)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device id")
    parser.add_argument("--cache-size", type=int, default=2, help="LRU cache size for encoded images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runtime = Sam3Runtime(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe,
        score_threshold=args.score_threshold,
        epsilon_factor=args.epsilon_factor,
        min_area=args.min_area,
        device=args.device,
        cache_size=args.cache_size,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, workers=1)
