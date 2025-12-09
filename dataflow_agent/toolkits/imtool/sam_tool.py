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
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional

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
    image_path : str
        Path to the input image.
    scale : float, optional
        Higher means larger clusters, by default 100.0.
    sigma : float, optional
        Gaussian smoothing parameter, by default 0.5.
    min_size : int, optional
        Minimum component size, by default 50.
    return_type : {"labels", "masks"}, optional
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
    image_path : str
        Path to the input image.
    output_path : str
        Path to save the visualization PNG (or other image format).
    scale : float, optional
        Higher means larger clusters, by default 100.0.
    sigma : float, optional
        Gaussian smoothing parameter, by default 0.5.
    min_size : int, optional
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
        /home/ubuntu/liuzhou/myproj/dev/DataFlow-Agent/outputs
    """
    import os
    import cv2

    # 1. Input/output paths
    img_path = ""
    out_dir = Path("")
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
        print(f"[SAM] Got {len(items)} masks")

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
