from __future__ import annotations

"""
Unified image segmentation tool wrappers.

This module provides:
- SAM (Segment Anything) via ultralytics.SAM
- YOLOv8 instance segmentation via ultralytics.YOLO
- Semantic segmentation via Hugging Face transformers (e.g., SegFormer)
- Classical graph-based segmentation via Felzenszwalb (scikit-image)

Design philosophy:
- Similar to mineru_tool.py: expose simple, stateless function APIs
- Internally cache heavy model objects to avoid repeated initialization
- Normalize outputs into Python dict / list structures that are easy to consume

# 函数一览（中文说明）：
# _ensure_ultralytics_sam_available: 校验 ultralytics.SAM 是否可用，否则抛出安装提示。
# _ensure_ultralytics_yolo_available: 校验 ultralytics.YOLO 是否可用，否则抛出安装提示。
# _ensure_hf_pipeline_available: 校验 transformers.pipeline 是否可用，否则抛出安装提示。
# _ensure_skimage_available: 校验 scikit-image 是否可用，否则抛出安装提示。
# _ensure_matplotlib_available: 校验 matplotlib 是否可用，否则抛出安装提示。
# _load_image_pil: 从路径读取图片并转为 RGB 的 PIL.Image。
# _get_image_size: 返回图片的宽高 (width, height)。
# _get_sam_model: 懒加载并缓存指定 checkpoint 的 SAM 模型。
# free_sam_model: 显式释放指定 checkpoint 的 SAM 模型并清理 CUDA 显存。
# run_sam_auto: 对单张图片运行 SAM 自动分割，返回每个实例的 mask、归一化 bbox 等信息。
# run_sam_auto_batch: 对多张图片批量运行 SAM 自动分割，按图片返回实例列表。
# _get_yolo_model: 懒加载并缓存指定权重和设备的 YOLOv8 分割模型。
# run_yolov8_seg: 对单张图片运行 YOLOv8 实例分割，返回带类别标签和分数的实例信息。
# run_yolov8_seg_batch: 对多张图片批量运行 YOLOv8 实例分割，按图片返回实例列表。
# _get_hf_seg_pipeline: 懒加载并缓存 Hugging Face 语义分割 pipeline。
# run_hf_semantic_seg: 使用 HF pipeline 对单张图片做语义分割，返回每个类别的前景掩膜。
# run_hf_semantic_seg_batch: 对多张图片批量做语义分割，按图片返回类别掩膜列表。
# run_felzenszwalb: 调用 Felzenszwalb 图分割算法，返回标签图或每个 segment 的布尔掩膜。
# save_felzenszwalb_visualization: 运行 Felzenszwalb 并保存带分割边界的可视化图片。
# save_sam_instances: 将 SAM 分割得到的每个实例按 bbox 或 RGBA mask 截图后保存为单独图片。
# 过滤/后处理函数：
# filter_sam_items_by_area_and_score: 按最小面积、最小得分过滤 SAM 实例。
# bbox_iou: 计算两个归一化 bbox 的 IoU。
# mask_iou: 计算两个布尔 mask 的 IoU。
# nms_sam_items_by_bbox: 基于 bbox IoU 的 SAM 实例 NMS 去重。
# nms_sam_items_by_mask: 基于 mask IoU 的 SAM 实例 NMS 去重。
# topk_sam_items: 只保留 Top-K 个 SAM 实例。
# postprocess_sam_items: 统一封装上述步骤的 SAM 实例后处理函数。

"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional
import base64
import requests
import random
import zlib
import os

import numpy as np
from PIL import Image

# Optional imports (lazy usage, we guard them at call-time)
try:
    from ultralytics import SAM as UltralyticsSAM
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    UltralyticsSAM = None  # type: ignore
    YOLO = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional dependency
    hf_pipeline = None  # type: ignore

# scikit-image & matplotlib for classical segmentation
try:
    from skimage import io as skio, segmentation
except Exception:  # pragma: no cover - optional dependency
    skio = None  # type: ignore
    segmentation = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

# torch 仅用于显式释放 CUDA 显存（free_sam_model 等场景）
try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------------------------------------------------------------
# 0. Simple global caches to avoid re-loading heavy models
# -----------------------------------------------------------------------------
_SAM_MODELS: Dict[str, Any] = {}
_YOLO_MODELS: Dict[tuple, Any] = {}
_HF_SEG_PIPELINES: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _ensure_ultralytics_sam_available() -> None:
    if UltralyticsSAM is None:
        raise ImportError(
            "ultralytics.SAM is not available. Please install `ultralytics`:\n"
            "    pip install ultralytics\n"
        )


def _ensure_ultralytics_yolo_available() -> None:
    if YOLO is None:
        raise ImportError(
            "ultralytics.YOLO is not available. Please install `ultralytics`:\n"
            "    pip install ultralytics\n"
        )


def _ensure_hf_pipeline_available() -> None:
    if hf_pipeline is None:
        raise ImportError(
            "transformers.pipeline is not available. Please install transformers:\n"
            "    pip install transformers\n"
        )


def _ensure_skimage_available() -> None:
    if skio is None or segmentation is None:
        raise ImportError(
            "scikit-image is not available. Please install scikit-image:\n"
            "    pip install scikit-image\n"
        )


def _ensure_matplotlib_available() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is not available. Please install matplotlib:\n"
            "    pip install matplotlib\n"
        )


def _load_image_pil(image_path: str) -> Image.Image:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    return Image.open(p).convert("RGB")


def _get_image_size(image_path: str) -> tuple[int, int]:
    img = _load_image_pil(image_path)
    return img.size  # (width, height)


# -----------------------------------------------------------------------------
# 1. SAM (Segment Anything Model) via ultralytics.SAM
# -----------------------------------------------------------------------------
def _get_sam_model(
    checkpoint: str = "sam_b.pt",
):
    """
    Lazy-load and cache ultralytics.SAM model.
    """
    _ensure_ultralytics_sam_available()
    key = checkpoint
    if key not in _SAM_MODELS:
        # Note: in current ultralytics versions SAM(...) does not accept device= argument here.
        # Device is controlled at inference time, e.g. model(img, device="cuda").
        _SAM_MODELS[key] = UltralyticsSAM(checkpoint)
    return _SAM_MODELS[key]


def free_sam_model(checkpoint: str = "sam_b.pt") -> None:
    """
    显式释放指定 checkpoint 对应的 SAM 模型，并尽量清理 CUDA 显存。

    典型用法：
        from dataflow_agent.toolkits.imtool.sam_tool import free_sam_model

        # 在 workflow 正常结束或发生 OOM 后调用
        free_sam_model("/abs/path/to/sam_b.pt")

    注意：
    - 仅影响当前 Python 进程持有的 SAM 模型，不会影响其他 GPU 进程。
    - 若当前环境中没有安装 torch，则只能删除 Python 层的引用，
      无法调用 torch.cuda.empty_cache()。
    """
    global _SAM_MODELS
    m = _SAM_MODELS.pop(checkpoint, None)
    if m is not None:
        # 尝试清理 ultralytics 内部引用
        try:
            # 一些 ultralytics 模型提供 model.to("cpu") 等接口，此处最好切到 CPU 再删除
            if hasattr(m, "model"):
                try:
                    m.model.to("cpu")
                except Exception:
                    pass
        except Exception:
            pass

        del m

    # 若 torch 可用，则尝试清空 CUDA 缓存，减轻显存压力
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            # 防御式处理，确保不会因为清理失败影响主流程
            pass


def run_sam_auto(
    image_path: str,
    checkpoint: str = "sam_b.pt",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run automatic segmentation on a single image using SAM (ultralytics backend).

    Parameters
    ----------
    image_path : str
        Path to the input image.
    checkpoint : str, optional
        SAM checkpoint to use, by default "sam_b.pt".
    device : str, optional
        Device string for torch (e.g. "cuda", "cpu"), by default "cuda".

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains at least:
            - mask: np.ndarray[H, W] bool
            - bbox: [x1, y1, x2, y2] in normalized coordinates [0,1]
            - score: float or None
            - area: int (number of True pixels in mask)
    """
    model = _get_sam_model(checkpoint=checkpoint)
    # In recent ultralytics versions, SAM models can receive `device` at call time.
    # If your installed version does not support this, you can remove `device=device`.
    try:
        results = model(image_path, device=device)  # ultralytics will load image internally
    except TypeError:
        # Fallback: older/newer API without device argument
        results = model(image_path)

    width, height = _get_image_size(image_path)

    all_items: List[Dict[str, Any]] = []
    for r in results:
        # r.masks: ultralytics Masks object or None
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)

        if masks is None:
            continue

        mask_data = getattr(masks, "data", None)
        if mask_data is None:
            continue

        mask_np = mask_data.cpu().numpy()  # [N, H, W]
        n_instances = mask_np.shape[0]

        # boxes may be None (SAM auto masks can be box-less); handle gracefully
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # [N, 4], in pixels
            scores = getattr(boxes, "conf", None)
            scores_np = scores.cpu().numpy() if scores is not None else None
        else:
            xyxy = np.zeros((n_instances, 4), dtype=np.float32)
            scores_np = None

        for i in range(n_instances):
            m = mask_np[i] > 0.5  # bool
            area = int(m.sum())

            x1, y1, x2, y2 = xyxy[i]
            # clamp & normalize
            x1 = float(max(0, min(width, x1))) / max(width, 1)
            x2 = float(max(0, min(width, x2))) / max(width, 1)
            y1 = float(max(0, min(height, y1))) / max(height, 1)
            y2 = float(max(0, min(height, y2))) / max(height, 1)
            bbox_norm = [x1, y1, x2, y2]

            score = float(scores_np[i]) if scores_np is not None else None

            all_items.append(
                {
                    "mask": m,
                    "bbox": bbox_norm,
                    "score": score,
                    "area": area,
                }
            )

    return all_items


def run_sam_auto_server(
    image_path: str,
    server_urls: Union[str, List[str]],
    checkpoint: str = "sam_b.pt",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run SAM auto segmentation via remote/local server.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    server_urls : str or List[str]
        Single URL or list of URLs for the SAM model server(s).
        e.g. "http://localhost:8001" or ["http://localhost:8001", "http://localhost:8002"]
    checkpoint : str, optional
        SAM checkpoint, by default "sam_b.pt".
    device : str, optional
        Device string, by default "cuda".

    Returns
    -------
    List[Dict[str, Any]]
        Same structure as run_sam_auto.
    """
    if isinstance(server_urls, str):
        urls = [server_urls]
    else:
        urls = list(server_urls)
    
    if not urls:
        raise ValueError("No server URLs provided")
    
    # Simple random load balancing
    base_url = random.choice(urls)
    api_url = f"{base_url.rstrip('/')}/predict"
    
    # Ensure image path is absolute for server to access
    abs_image_path = os.path.abspath(image_path)
    
    payload = {
        "image_path": abs_image_path,
        "checkpoint": checkpoint,
        "device": device
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        items_raw = data.get("items", [])
        all_items: List[Dict[str, Any]] = []
        
        for it in items_raw:
            # Deserialize mask
            mask_b64 = it.get("mask_b64")
            mask_shape = it.get("mask_shape")
            
            if mask_b64 and mask_shape:
                try:
                    compressed_bytes = base64.b64decode(mask_b64)
                    # Decompress using zlib
                    mask_bytes = zlib.decompress(compressed_bytes)
                    # Reconstruct numpy bool array
                    # Note: mask_bytes is flattened bool bytes
                    mask_flat = np.frombuffer(mask_bytes, dtype=bool)
                    mask = mask_flat.reshape(tuple(mask_shape))
                except Exception as e:
                    # Fallback for uncompressed data (backward compatibility) or decompression error
                    try:
                        mask_bytes = base64.b64decode(mask_b64)
                        mask_flat = np.frombuffer(mask_bytes, dtype=bool)
                        mask = mask_flat.reshape(tuple(mask_shape))
                    except Exception:
                        print(f"Failed to decode/decompress mask: {e}")
                        mask = None
            else:
                mask = None
            
            all_items.append({
                "mask": mask,
                "bbox": it.get("bbox"),
                "score": it.get("score"),
                "area": it.get("area", 0)
            })
            
        return all_items
        
    except Exception as e:
        # Log error or handle retry logic here if needed
        raise RuntimeError(f"Failed to call SAM server at {api_url}: {e}")


def run_sam_auto_batch(
    image_paths: List[str],
    checkpoint: str = "sam_b.pt",
    device: str = "cuda",
) -> List[List[Dict[str, Any]]]:
    """
    Batch automatic segmentation using SAM.

    Parameters
    ----------
    image_paths : list[str]
        List of image paths.
    checkpoint : str, optional
        SAM checkpoint to use, by default "sam_b.pt".
    device : str, optional
        Device string, by default "cuda".

    Returns
    -------
    List[List[Dict[str, Any]]]
        One list of items per input image, see `run_sam_auto`.
    """
    return [run_sam_auto(p, checkpoint=checkpoint, device=device) for p in image_paths]


# -----------------------------------------------------------------------------
# 1.1 SAM post-processing helpers (过滤 / 去重 / Top-K)
# -----------------------------------------------------------------------------
def filter_sam_items_by_area_and_score(
    items: List[Dict[str, Any]],
    min_area: int = 0,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    简单按面积和得分过滤 SAM 实例。

    Parameters
    ----------
    items : List[Dict[str, Any]]
        run_sam_auto 返回的实例列表。
    min_area : int, optional
        保留的最小像素面积，默认 0（不过滤）。
    min_score : float, optional
        保留的最小得分阈值（当存在 score 时），默认 0.0。

    Returns
    -------
    List[Dict[str, Any]]
        过滤后的实例列表。
    """
    filtered: List[Dict[str, Any]] = []
    for it in items:
        area = int(it.get("area", 0))
        score = it.get("score", None)
        if area < min_area:
            continue
        if score is not None and float(score) < float(min_score):
            continue
        filtered.append(it)
    return filtered


def bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """
    计算两个归一化 bbox 的 IoU。

    Parameters
    ----------
    box1, box2 : [x1, y1, x2, y2] in [0, 1]

    Returns
    -------
    float
        IoU 值，范围 [0, 1]。
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    计算两个布尔 mask 的 IoU。
    """
    if m1.shape != m2.shape:
        # 简单兜底：形状不一致时不计算 IoU
        return 0.0

    m1_bool = m1.astype(bool)
    m2_bool = m2.astype(bool)

    inter = np.logical_and(m1_bool, m2_bool).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(m1_bool, m2_bool).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def nms_sam_items_by_bbox(
    items: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    score_key: str = "score",  # "score" | "area"
) -> List[Dict[str, Any]]:
    """
    基于 bbox IoU 的 Non-Maximum Suppression（NMS）去重。

    - 按 score_key（score 或 area）从大到小排序；
    - 依次保留与已有保留框 IoU 小于阈值的实例。

    Parameters
    ----------
    items : List[Dict[str, Any]]
        run_sam_auto 返回的实例列表。
    iou_threshold : float, optional
        IoU 阈值，超过则认为“太重叠”，默认 0.5。
    score_key : str, optional
        用于排序和优先保留的字段，默认 "score"，可选 "area"。

    Returns
    -------
    List[Dict[str, Any]]
        经过 NMS 去重后的实例列表。
    """

    def _score(it: Dict[str, Any]) -> float:
        if score_key == "score":
            v = it.get("score")
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        # fallback to area
        return float(it.get("area", 0))

    # 从高分/大面积到低分/小面积排序
    items_sorted = sorted(items, key=_score, reverse=True)

    kept: List[Dict[str, Any]] = []
    for it in items_sorted:
        bbox = it.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        keep = True
        for kept_it in kept:
            kept_bbox = kept_it.get("bbox")
            if not kept_bbox or len(kept_bbox) != 4:
                continue
            iou = bbox_iou(bbox, kept_bbox)
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(it)

    return kept


def nms_sam_items_by_mask(
    items: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """
    基于 mask IoU 的 Non-Maximum Suppression（NMS）去重。

    - IoU 计算使用像素级 mask，更精确但速度稍慢；
    - 其他逻辑与 nms_sam_items_by_bbox 类似。

    Parameters
    ----------
    items : List[Dict[str, Any]]
        run_sam_auto 返回的实例列表。
    iou_threshold : float, optional
        IoU 阈值，默认 0.5。
    score_key : str, optional
        用于排序的字段，默认 "score"，可选 "area"。

    Returns
    -------
    List[Dict[str, Any]]
        经过 NMS 去重后的实例列表。
    """

    def _score(it: Dict[str, Any]) -> float:
        if score_key == "score":
            v = it.get("score")
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return float(it.get("area", 0))

    items_sorted = sorted(items, key=_score, reverse=True)
    kept: List[Dict[str, Any]] = []

    for it in items_sorted:
        m = it.get("mask")
        if m is None:
            continue
        m_arr = np.array(m)
        if m_arr.dtype != bool:
            m_arr = m_arr > 0

        keep = True
        for kept_it in kept:
            m2 = kept_it.get("mask")
            if m2 is None:
                continue
            m2_arr = np.array(m2)
            if m2_arr.dtype != bool:
                m2_arr = m2_arr > 0

            # 形状不同则视作 IoU=0（不去重）
            if m_arr.shape != m2_arr.shape:
                continue

            iou = mask_iou(m_arr, m2_arr)
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(it)

    return kept


def topk_sam_items(
    items: List[Dict[str, Any]],
    k: int = 0,
    sort_key: str = "area",  # "area" | "score"
) -> List[Dict[str, Any]]:
    """
    只保留 Top-K 个 SAM 实例。

    Parameters
    ----------
    items : List[Dict[str, Any]]
        run_sam_auto 返回的实例列表。
    k : int, optional
        要保留的实例数量；小于等于 0 时不截断，默认 0。
    sort_key : str, optional
        排序依据，默认 "area"，可选 "score"。

    Returns
    -------
    List[Dict[str, Any]]
        截断后的实例列表。
    """
    if k is None or k <= 0:
        return items

    def _score(it: Dict[str, Any]) -> float:
        if sort_key == "score":
            v = it.get("score")
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return float(it.get("area", 0))

    items_sorted = sorted(items, key=_score, reverse=True)
    return items_sorted[:k]


def postprocess_sam_items(
    items: List[Dict[str, Any]],
    min_area: int = 0,
    min_score: float = 0.0,
    iou_threshold: float = 0.0,
    top_k: Optional[int] = None,
    nms_by: str = "bbox",  # "bbox" | "mask"
    score_key_for_nms: str = "score",  # "score" | "area"
    sort_key_for_topk: str = "area",  # "area" | "score"
) -> List[Dict[str, Any]]:
    """
    统一封装 SAM 实例的后处理流程（过滤 + NMS + Top-K）。

    典型用法：
        items = run_sam_auto(...)
        items = postprocess_sam_items(
            items,
            min_area=200,
            min_score=0.3,
            iou_threshold=0.6,
            top_k=20,
            nms_by="bbox",
        )

    Parameters
    ----------
    items : List[Dict[str, Any]]
        run_sam_auto 返回的实例列表。
    min_area : int, optional
        最小保留面积，默认 0（不过滤）。
    min_score : float, optional
        最小保留得分（当存在 score），默认 0.0。
    iou_threshold : float, optional
        若大于 0，则按该阈值进行 NMS 去重；小于等于 0 时不做 NMS。
    top_k : Optional[int], optional
        若给定且 > 0，则在 NMS 后只保留 Top-K，默认 None（不截断）。
    nms_by : {"bbox", "mask"}, optional
        NMS 的 IoU 计算方式，默认 "bbox"。
    score_key_for_nms : {"score", "area"}, optional
        NMS 时的排序依据，默认 "score"。
    sort_key_for_topk : {"area", "score"}, optional
        Top-K 时的排序依据，默认 "area"。

    Returns
    -------
    List[Dict[str, Any]]
        经过后处理的 SAM 实例列表。
    """
    # 1) 面积 / 得分过滤
    out = filter_sam_items_by_area_and_score(
        items,
        min_area=min_area,
        min_score=min_score,
    )

    # 2) NMS 去重
    if iou_threshold and iou_threshold > 0.0:
        if nms_by == "mask":
            out = nms_sam_items_by_mask(
                out,
                iou_threshold=float(iou_threshold),
                score_key=score_key_for_nms,
            )
        else:
            out = nms_sam_items_by_bbox(
                out,
                iou_threshold=float(iou_threshold),
                score_key=score_key_for_nms,
            )

    # 3) Top-K 截断
    if top_k is not None and top_k > 0:
        out = topk_sam_items(
            out,
            k=int(top_k),
            sort_key=sort_key_for_topk,
        )

    return out


# -----------------------------------------------------------------------------
# 2. YOLOv8 Instance Segmentation via ultralytics.YOLO
# -----------------------------------------------------------------------------
def _get_yolo_model(
    weights: str = "yolov8n-seg.pt",
    device: str = "cuda",
):
    """
    Lazy-load and cache ultralytics.YOLO model.
    """
    _ensure_ultralytics_yolo_available()
    key = (weights, device)
    if key not in _YOLO_MODELS:
        _YOLO_MODELS[key] = YOLO(weights).to(device)
    return _YOLO_MODELS[key]


def run_yolov8_seg(
    image_path: str,
    weights: str = "yolov8n-seg.pt",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 instance segmentation on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    weights : str, optional
        Weights file or model name, by default "yolov8n-seg.pt".
    device : str, optional
        Device string (e.g. "cuda", "cpu"), by default "cuda".

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains:
            - mask: np.ndarray[H, W] bool
            - bbox: [x1, y1, x2, y2] normalized
            - label: str
            - score: float
    """
    model = _get_yolo_model(weights=weights, device=device)
    results = model(image_path)

    width, height = _get_image_size(image_path)
    all_items: List[Dict[str, Any]] = []

    for r in results:
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", None) or {}

        if masks is None or boxes is None:
            continue

        mask_data = masks.data.cpu().numpy()  # [N, H, W]
        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        conf = boxes.conf.cpu().numpy()  # [N]
        cls = boxes.cls.cpu().numpy()  # [N]

        n_instances = mask_data.shape[0]
        for i in range(n_instances):
            m = mask_data[i] > 0.5  # bool

            x1, y1, x2, y2 = xyxy[i]
            x1 = float(max(0, min(width, x1))) / max(width, 1)
            x2 = float(max(0, min(width, x2))) / max(width, 1)
            y1 = float(max(0, min(height, y1))) / max(height, 1)
            y2 = float(max(0, min(height, y2))) / max(height, 1)
            bbox_norm = [x1, y1, x2, y2]

            c = int(cls[i])
            label = names.get(c, str(c))
            score = float(conf[i])

            all_items.append(
                {
                    "mask": m,
                    "bbox": bbox_norm,
                    "label": label,
                    "score": score,
                }
            )

    return all_items


def run_yolov8_seg_batch(
    image_paths: List[str],
    weights: str = "yolov8n-seg.pt",
    device: str = "cuda",
) -> List[List[Dict[str, Any]]]:
    """
    Batch instance segmentation using YOLOv8.

    Returns
    -------
    List[List[Dict[str, Any]]]
        One list of items per input image, see `run_yolov8_seg`.
    """
    return [run_yolov8_seg(p, weights=weights, device=device) for p in image_paths]


# -----------------------------------------------------------------------------
# 3. Hugging Face Semantic Segmentation (e.g., SegFormer)
# -----------------------------------------------------------------------------
def _get_hf_seg_pipeline(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
):
    """
    Lazy-load and cache HF image-segmentation pipeline.
    """
    _ensure_hf_pipeline_available()
    if model_name not in _HF_SEG_PIPELINES:
        _HF_SEG_PIPELINES[model_name] = hf_pipeline(
            "image-segmentation",
            model=model_name,
        )
    return _HF_SEG_PIPELINES[model_name]


def run_hf_semantic_seg(
    image_path: str,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> List[Dict[str, Any]]:
    """
    Run semantic segmentation via Hugging Face pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model_name : str, optional
        HF model name, by default "nvidia/segformer-b0-finetuned-ade-512-512".

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains:
            - label: str
            - score: float (if provided by pipeline)
            - mask: np.ndarray[H, W] bool    # foreground region of that label
    """
    seg_pipe = _get_hf_seg_pipeline(model_name=model_name)
    img = _load_image_pil(image_path)
    results = seg_pipe(img)

    # results is usually a list of dict:
    #   {"label": ..., "score": ..., "mask": PIL.Image or np.array}
    normed: List[Dict[str, Any]] = []
    for r in results:
        label = r.get("label")
        score = float(r.get("score", 0.0))

        mask = r.get("mask")
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = np.array(mask)

        # Convert to bool mask: non-zero as True
        if mask_np.dtype != bool:
            m_bool = mask_np > 0
        else:
            m_bool = mask_np

        normed.append(
            {
                "label": label,
                "score": score,
                "mask": m_bool,
            }
        )

    return normed


def run_hf_semantic_seg_batch(
    image_paths: List[str],
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> List[List[Dict[str, Any]]]:
    """
    Batch semantic segmentation via Hugging Face pipeline.

    Returns
    -------
    List[List[Dict[str, Any]]]
        One list of items per image, see `run_hf_semantic_seg`.
    """
    return [run_hf_semantic_seg(p, model_name=model_name) for p in image_paths]


# -----------------------------------------------------------------------------
# 4. Classical segmentation: Felzenszwalb graph-based segmentation
# -----------------------------------------------------------------------------
def run_felzenszwalb(
    image_path: str,
    scale: float = 100.0,
    sigma: float = 0.5,
    min_size: int = 50,
    return_type: str = "labels",  # "labels" | "masks"
) -> Dict[str, Any]:
    """
    Perform Felzenszwalb's efficient graph-based segmentation.

    Parameters
    ----------
    image_path: str
        Path to the input image.
    scale: float, optional
        Higher means larger clusters, by default 100.0.
    sigma: float, optional
        Gaussian smoothing parameter, by default 0.5.
    min_size: int, optional
        Minimum component size, by default 50.
    return_type: {"labels", "masks"}, optional
        - "labels": return an integer label map [H, W]
        - "masks": return a list of bool masks for each segment id

    Returns
    -------
    Dict[str, Any]
        If return_type == "labels":
            {
                "segments": np.ndarray[H, W] int,
                "num_segments": int,
            }
        If return_type == "masks":
            {
                "masks": list[np.ndarray[H, W] bool],
                "labels": list[int],
            }
    """
    _ensure_skimage_available()

    img = skio.imread(image_path)
    segments = segmentation.felzenszwalb(
        img, scale=scale, sigma=sigma, min_size=min_size
    )
    unique_labels = np.unique(segments)

    if return_type == "labels":
        return {
            "segments": segments,
            "num_segments": int(unique_labels.size),
        }

    if return_type == "masks":
        masks: List[np.ndarray] = []
        labels: List[int] = []
        for lab in unique_labels:
            m = segments == lab
            masks.append(m)
            labels.append(int(lab))
        return {
            "masks": masks,
            "labels": labels,
        }

    raise ValueError(f"Unsupported return_type: {return_type!r}, must be 'labels' or 'masks'.")


def save_felzenszwalb_visualization(
    image_path: str,
    output_path: str,
    scale: float = 100.0,
    sigma: float = 0.5,
    min_size: int = 50,
) -> str:
    """
    Run Felzenszwalb segmentation and save a visualization with boundaries.

    Parameters
    ----------
    image_path: str
        Path to the input image.
    output_path: str
        Path to save the visualization PNG (or other image format).
    scale: float, optional
        Higher means larger clusters, by default 100.0.
    sigma: float, optional
        Gaussian smoothing parameter, by default 0.5.
    min_size: int, optional
        Minimum component size, by default 50.

    Returns
    -------
    str
        Absolute path to the saved visualization image.
    """
    _ensure_skimage_available()
    _ensure_matplotlib_available()

    img = skio.imread(image_path)
    segments = segmentation.felzenszwalb(
        img, scale=scale, sigma=sigma, min_size=min_size
    )
    vis = segmentation.mark_boundaries(img, segments)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(vis)
    plt.tight_layout(pad=0)
    plt.savefig(out_p, bbox_inches="tight", pad_inches=0)
    plt.close()

    return str(out_p.resolve())


# -----------------------------------------------------------------------------
# 5. Save SAM instances as images
# -----------------------------------------------------------------------------
def save_sam_instances(
    image_path: str,
    items: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    prefix: str = "sam_inst_",
    mode: str = "bbox",  # "bbox" | "rgba"
) -> List[str]:
    """
    Save each SAM instance (from run_sam_auto) as a separate image file.

    Parameters
    ----------
    image_path : str
        Original image path.
    items : List[Dict[str, Any]]
        The list returned by `run_sam_auto`, each item must contain:
          - "mask": np.ndarray[H, W] bool
          - "bbox": [x1, y1, x2, y2] normalized
    output_dir : str | Path
        Directory to save cropped images. Will be created if not exists.
    prefix : str, optional
        Filename prefix, e.g. "sam_inst_".
    mode : {"bbox", "rgba"}, optional
        - "bbox": crop rectangular region by bbox from original image (RGB).
        - "rgba": apply mask as alpha to original image (RGBA), then crop bbox.

    Returns
    -------
    List[str]
        List of absolute paths to saved instance images.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original image as RGBA for consistent processing
    img = _load_image_pil(image_path).convert("RGBA")
    width, height = img.size

    saved_paths: List[str] = []

    for idx, item in enumerate(items):
        mask = item.get("mask")
        bbox = item.get("bbox")
        if mask is None or bbox is None:
            continue

        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        if mask.dtype != bool:
            m_bool = mask > 0
        else:
            m_bool = mask

        # bbox is normalized [x1, y1, x2, y2]
        x1_norm, y1_norm, x2_norm, y2_norm = bbox
        left = max(0, min(width, int(round(x1_norm * width))))
        top = max(0, min(height, int(round(y1_norm * height))))
        right = max(0, min(width, int(round(x2_norm * width))))
        bottom = max(0, min(height, int(round(y2_norm * height))))

        if right <= left or bottom <= top:
            continue

        # Ensure mask size matches image size
        if m_bool.shape[:2] != (height, width):
            # If shapes mismatch, try simple resize via PIL (nearest)
            m_img = Image.fromarray(m_bool.astype(np.uint8) * 255)
            m_img = m_img.resize((width, height), resample=Image.NEAREST)
            m_bool = np.array(m_img) > 0

        if mode == "bbox":
            # Simple rectangular crop from original image (no transparency)
            rgb_img = img.convert("RGB")
            patch = rgb_img.crop((left, top, right, bottom))
        elif mode == "rgba":
            # Apply mask as alpha to original image, then crop
            rgba = np.array(img)  # H x W x 4
            alpha = rgba[:, :, 3]
            alpha[~m_bool] = 0
            rgba[:, :, 3] = alpha
            masked_img = Image.fromarray(rgba)
            patch = masked_img.crop((left, top, right, bottom))
        else:
            raise ValueError(f"Unsupported mode: {mode!r}, must be 'bbox' or 'rgba'.")

        out_path = out_dir / f"{prefix}{idx}.png"
        patch.save(out_path)
        saved_paths.append(str(out_path.resolve()))

    return saved_paths


def segment_layout_boxes(
    image_path: str,
    output_dir: str,
    checkpoint: str = "sam_b.pt",
    device: str = "cuda",
    min_area: int = 0,
    min_score: float = 0.0,
    iou_threshold: float = 0.5,
    top_k: Optional[int] = None,
    nms_by: str = "bbox",
) -> List[Dict[str, Any]]:
    """
    针对空框模板图做 SAM 分割，过滤 + NMS + 裁剪，返回每个框的 bbox + patch PNG 路径。

    该函数主要服务于 paper2figure_with_sam 工作流中的“背景框架层”生成：
    - 输入为二次编辑后的空框模板图（fig_layout_path）；
    - 不关心具体语义，只需要稳定的矩形/箭头布局；
    - 输出 items 将在后续被转换为 SVG / EMF 并按 bbox 映射回 PPT。
    """
    # 1) SAM 自动分割
    items = run_sam_auto(image_path, checkpoint=checkpoint, device=device)

    # 2) 过滤 + NMS + Top-K
    items = postprocess_sam_items(
        items,
        min_area=min_area,
        min_score=min_score,
        iou_threshold=iou_threshold,
        top_k=top_k,
        nms_by=nms_by,
        score_key_for_nms="score",
        sort_key_for_topk="area",
    )

    # 3) 将每个实例按 bbox 裁剪为 PNG 小图
    saved_paths = save_sam_instances(
        image_path=image_path,
        items=items,
        output_dir=output_dir,
        prefix="layout_",
        mode="bbox",
    )

    # 4) 绑定 png_path & type
    for i, p in enumerate(saved_paths):
        if i >= len(items):
            break
        items[i]["png_path"] = p
        # 标记为布局框，和 MinerU 的 type 区分开
        items[i]["type"] = "layout_box"

    return items


def segment_layout_boxes_server(
    image_path: str,
    output_dir: str,
    server_urls: Union[str, List[str]],
    checkpoint: str = "sam_b.pt",
    device: str = "cuda",
    min_area: int = 0,
    min_score: float = 0.0,
    iou_threshold: float = 0.5,
    top_k: Optional[int] = None,
    nms_by: str = "bbox",
) -> List[Dict[str, Any]]:
    """
    Server version of segment_layout_boxes.
    """
    # 1) SAM 自动分割 (Remote)
    items = run_sam_auto_server(
        image_path, 
        server_urls=server_urls, 
        checkpoint=checkpoint, 
        device=device
    )

    # 2) 过滤 + NMS + Top-K
    items = postprocess_sam_items(
        items,
        min_area=min_area,
        min_score=min_score,
        iou_threshold=iou_threshold,
        top_k=top_k,
        nms_by=nms_by,
        score_key_for_nms="score",
        sort_key_for_topk="area",
    )

    # 3) 将每个实例按 bbox 裁剪为 PNG 小图
    saved_paths = save_sam_instances(
        image_path=image_path,
        items=items,
        output_dir=output_dir,
        prefix="layout_",
        mode="bbox",
    )

    # 4) 绑定 png_path & type
    for i, p in enumerate(saved_paths):
        if i >= len(items):
            break
        items[i]["png_path"] = p
        # 标记为布局框，和 MinerU 的 type 区分开
        items[i]["type"] = "layout_box"

    return items


# -----------------------------------------------------------------------------
# 6. Simple demo main for quick testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick manual test entry.

    Usage (from repository root, adjust PYTHONPATH accordingly):
        python -m dataflow_agent.toolkits.imtool.sam_tool

    It will:
    - Load the specified PNG as input.
    - Try SAM auto segmentation (if ultralytics + SAM weights are available),
      and save overlay visualization.
    - Run Felzenszwalb graph-based segmentation and save boundary visualization.

    All outputs are written to:
        /DataFlow-Agent/outputs
    """
    import os
    import cv2
    from dataflow_agent.utils import get_project_root

    # 1. Input/output paths
    img_path = f"{get_project_root()}/tests/image.png"
    out_dir = Path(f"{get_project_root()}/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Demo] Input image: {img_path}")
    print(f"[Demo] Output dir : {out_dir}")

    # 2. Read image (BGR for OpenCV visualization)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # 3. Test SAM automatic segmentation (if available)
    try:
        items = run_sam_auto(
            img_path,
            checkpoint="sam_b.pt",  # assumes this file is available or will be auto-downloaded
            device="cuda",          # change to "cpu" if no GPU
        )

        # Apply default post-processing for demo:
        # - remove tiny masks
        # - suppress highly-overlapping instances
        # - keep at most 20 instances
        items = postprocess_sam_items(
            items,
            min_area=200,
            min_score=0.0,
            iou_threshold=0.3,
            top_k=20,
            nms_by="bbox",
        )

        print(f"[SAM] Got {len(items)} masks after post-processing")

        # 3.1 Save each SAM instance as a separate image
        sam_inst_dir = out_dir / "sam_instances"
        saved_paths = save_sam_instances(
            image_path=img_path,
            items=items,
            output_dir=sam_inst_dir,
            prefix="sam_inst_",
            mode="bbox",  # or "rgba"
        )
        print(f"[SAM] {len(saved_paths)} instance images saved to dir: {sam_inst_dir}")
        for p in saved_paths:
            print("   ", p)

        # 3.2 Keep original overlay visualization
        if items:
            overlay = img_bgr.copy()
            # Visualize at most first 10 masks
            for i, item in enumerate(items[:10]):
                m = item["mask"].astype(np.uint8) * 255  # HxW
                color = np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
                color_mask = np.zeros_like(overlay)
                color_mask[m > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

            sam_vis_path = out_dir / "sam_masks_overlay.png"
            cv2.imwrite(str(sam_vis_path), overlay)
            print(f"[SAM] Overlay saved to: {sam_vis_path}")
        else:
            print("[SAM] No masks returned.")
    except ImportError as e:
        print("[SAM] Skipped due to missing dependency:", e)
    except Exception as e:
        print("[SAM] Error while running SAM demo:", e)

    # 4. Test Felzenszwalb segmentation (classical graph-based)
    try:
        res = run_felzenszwalb(
            img_path,
            scale=100.0,
            sigma=0.5,
            min_size=50,
            return_type="labels",
        )
        segments = res["segments"]
        num_segments = res["num_segments"]
        print(f"[Felzenszwalb] num_segments = {num_segments}")

        # Visualize boundaries using skimage + OpenCV
        from skimage import segmentation as _seg

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis = _seg.mark_boundaries(img_rgb, segments)
        vis_bgr = cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        felz_vis_path = out_dir / "felzenszwalb_boundaries.png"
        cv2.imwrite(str(felz_vis_path), vis_bgr)
        print(f"[Felzenszwalb] Boundary visualization saved to: {felz_vis_path}")
    except ImportError as e:
        print("[Felzenszwalb] Skipped due to missing dependency:", e)
    except Exception as e:
        print("[Felzenszwalb] Error while running Felzenszwalb demo:", e)
