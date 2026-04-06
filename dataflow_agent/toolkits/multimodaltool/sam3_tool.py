from __future__ import annotations

import base64
import io
import itertools
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
from PIL import Image
import requests

from dataflow_agent.toolkits.image2drawio import bbox_iou_px


class Sam3ServiceClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        return resp.status_code == 200

    def predict(
        self,
        image_path: str,
        prompts: List[str],
        return_masks: bool = False,
        mask_format: Literal["rle", "png"] = "rle",
        score_threshold: Optional[float] = None,
        epsilon_factor: Optional[float] = None,
        min_area: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image_path": image_path,
            "prompts": prompts,
            "return_masks": return_masks,
            "mask_format": mask_format,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if epsilon_factor is not None:
            payload["epsilon_factor"] = epsilon_factor
        if min_area is not None:
            payload["min_area"] = min_area

        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class Sam3ServicePool:
    def __init__(self, endpoints: Sequence[str], timeout: int = 120) -> None:
        if not endpoints:
            raise ValueError("At least one endpoint is required")
        self.clients = [Sam3ServiceClient(url, timeout=timeout) for url in endpoints]
        self._lock = threading.Lock()
        self._cursor = itertools.cycle(range(len(self.clients)))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        with self._lock:
            client_index = next(self._cursor)
        return self.clients[client_index].predict(*args, **kwargs)

    def health(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for client in self.clients:
            try:
                status[client.base_url] = client.health()
            except Exception:
                status[client.base_url] = False
        return status


@dataclass
class Sam3PredictRun:
    group: str
    prompts: List[str]
    score_threshold: Optional[float] = None
    min_area: Optional[int] = None
    epsilon_factor: Optional[float] = None


def _bbox_area(b: List[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _read_env_value_from_fastapi_env(env_name: str) -> str:
    env_file = Path(__file__).resolve().parents[3] / "fastapi_app" / ".env"
    if not env_file.is_file():
        return ""

    prefix = f"{env_name}="
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith(prefix):
                continue
            return line[len(prefix):].strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def get_sam3_endpoints(
    env_names: Sequence[str] = ("SAM3_SERVER_URLS", "SAM3_ENDPOINTS"),
    default_endpoint: str = "http://127.0.0.1:8001",
) -> List[str]:
    for env_name in env_names:
        raw = os.getenv(env_name, "").strip()
        if raw:
            endpoints = [u.strip() for u in raw.split(",") if u.strip()]
            if endpoints:
                return endpoints

    for env_name in env_names:
        raw = _read_env_value_from_fastapi_env(env_name)
        if raw:
            endpoints = [u.strip() for u in raw.split(",") if u.strip()]
            if endpoints:
                return endpoints

    if default_endpoint:
        return [default_endpoint]
    return []


def get_sam3_client(
    endpoints: Optional[Sequence[str]] = None,
    timeout: int = 120,
) -> Optional[Any]:
    resolved = list(endpoints) if endpoints is not None else get_sam3_endpoints()
    resolved = [x for x in resolved if x]
    if not resolved:
        return None
    if len(resolved) == 1:
        return Sam3ServiceClient(resolved[0], timeout=timeout)
    return Sam3ServicePool(resolved, timeout=timeout)


def decode_sam3_mask(mask_obj: Dict[str, Any]) -> Optional[np.ndarray]:
    if not mask_obj:
        return None
    fmt = mask_obj.get("format")
    data = mask_obj.get("data")
    shape = mask_obj.get("shape")
    if not fmt or data is None:
        return None
    if fmt == "png":
        try:
            raw = base64.b64decode(data)
            img = Image.open(io.BytesIO(raw)).convert("L")
            arr = np.array(img)
            return (arr > 0).astype(np.uint8)
        except Exception:
            return None
    if fmt == "rle" and shape:
        try:
            h, w = int(shape[0]), int(shape[1])
            runs = [int(x) for x in str(data).split(",") if x]
            flat = np.zeros(sum(runs), dtype=np.uint8)
            val = 0
            idx = 0
            for r in runs:
                if r <= 0:
                    continue
                if val == 1:
                    flat[idx:idx + r] = 1
                idx += r
                val = 1 - val
            if flat.size < h * w:
                flat = np.pad(flat, (0, h * w - flat.size), constant_values=0)
            flat = flat[: h * w]
            return flat.reshape((h, w))
        except Exception:
            return None
    return None


def dedup_sam3_results_across_groups(
    items: List[Dict[str, Any]],
    group_config: Dict[str, Dict[str, Any]],
    dedup_iou: float = 0.7,
    arrow_dedup_iou: float = 0.85,
    shape_image_iou: float = 0.6,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    sorted_items = sorted(
        items,
        key=lambda x: (group_config.get(x.get("group", ""), {}).get("priority", 1), x.get("score", 0)),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    dropped = set()

    for i, item_i in enumerate(sorted_items):
        if i in dropped:
            continue
        kept.append(item_i)
        bbox_i = item_i.get("bbox")
        if not bbox_i:
            continue
        group_i = item_i.get("group", "")

        for j in range(i + 1, len(sorted_items)):
            if j in dropped:
                continue
            item_j = sorted_items[j]
            bbox_j = item_j.get("bbox")
            if not bbox_j:
                continue
            group_j = item_j.get("group", "")

            iou = bbox_iou_px(bbox_i, bbox_j)
            if iou < 0.1:
                continue

            if (group_i == "arrow" or group_j == "arrow") and iou > arrow_dedup_iou:
                dropped.add(j)
                continue

            if iou > shape_image_iou:
                is_shape_image = (
                    (group_i == "shape" and group_j == "image")
                    or (group_i == "image" and group_j == "shape")
                )
                if is_shape_image:
                    if group_i == "shape":
                        if item_i in kept:
                            kept.remove(item_i)
                        kept.append(item_j)
                        dropped.add(j)
                        break
                    dropped.add(j)
                    continue

            if iou > dedup_iou:
                dropped.add(j)

    return kept


def filter_sam3_items_contained_by_images(
    items: List[Dict[str, Any]],
    image_groups: Optional[Sequence[str]] = None,
    contain_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    if not items:
        return items
    image_group_set = set(image_groups or ["image"])
    to_remove = set()
    for i, a in enumerate(items):
        if i in to_remove:
            continue
        bbox_a = a.get("bbox")
        if not bbox_a:
            continue
        area_a = _bbox_area(bbox_a)
        group_a = a.get("group", "")
        for j, b in enumerate(items):
            if i == j or j in to_remove:
                continue
            bbox_b = b.get("bbox")
            if not bbox_b:
                continue
            area_b = _bbox_area(bbox_b)
            group_b = b.get("group", "")
            if area_a <= 0 or area_b <= 0:
                continue
            xA = max(bbox_a[0], bbox_b[0])
            yA = max(bbox_a[1], bbox_b[1])
            xB = min(bbox_a[2], bbox_b[2])
            yB = min(bbox_a[3], bbox_b[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter <= 0:
                continue
            if area_a > area_b:
                contain = inter / float(area_b)
                if contain > contain_threshold and group_a in image_group_set:
                    to_remove.add(j)
            else:
                contain = inter / float(area_a)
                if contain > contain_threshold and group_b in image_group_set:
                    to_remove.add(i)
                    break
    return [it for k, it in enumerate(items) if k not in to_remove]


def run_sam3_predict_runs(
    client: Any,
    image_path: str,
    runs: Sequence[Sam3PredictRun],
    return_masks: bool = True,
    mask_format: Literal["rle", "png"] = "png",
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for run in runs:
        try:
            resp = client.predict(
                image_path=image_path,
                prompts=run.prompts,
                return_masks=return_masks,
                mask_format=mask_format,
                score_threshold=run.score_threshold,
                epsilon_factor=run.epsilon_factor,
                min_area=run.min_area,
            )
        except Exception:
            continue
        results = resp.get("results", []) or []
        for item in results:
            item["group"] = run.group
        all_results.extend(results)
    return all_results


def predict_sam3_groups(
    client: Any,
    image_path: str,
    group_prompts: Dict[str, List[str]],
    group_config: Dict[str, Dict[str, Any]],
    extra_runs: Optional[Sequence[Sam3PredictRun]] = None,
    dedup_iou: float = 0.7,
    arrow_dedup_iou: float = 0.85,
    shape_image_iou: float = 0.6,
    contain_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    runs: List[Sam3PredictRun] = []
    for group, prompts in group_prompts.items():
        cfg = group_config.get(group, {})
        runs.append(
            Sam3PredictRun(
                group=group,
                prompts=prompts,
                score_threshold=cfg.get("score_threshold"),
                min_area=cfg.get("min_area"),
            )
        )
    if extra_runs:
        runs.extend(extra_runs)

    all_results = run_sam3_predict_runs(client=client, image_path=image_path, runs=runs)
    all_results = dedup_sam3_results_across_groups(
        all_results,
        group_config=group_config,
        dedup_iou=dedup_iou,
        arrow_dedup_iou=arrow_dedup_iou,
        shape_image_iou=shape_image_iou,
    )
    all_results = filter_sam3_items_contained_by_images(
        all_results,
        image_groups=["image"],
        contain_threshold=contain_threshold,
    )
    return all_results
