# ================================================================
# BRIA-RMBG 2.0 高质量抠图工具（离线版）
# - 模型：RMBG 2.0（ONNX）
# - 依赖：onnxruntime, pillow, numpy
# ================================================================

from __future__ import annotations
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

# 若依赖放在 ./deps 目录（推荐本地隔离）
DEPS_DIR = Path(__file__).resolve().parent / "deps"
if DEPS_DIR.exists():
    sys.path.append(str(DEPS_DIR))

import onnxruntime as ort

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models" / "RMBG-2.0" / "onnx"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_DIR = CURRENT_DIR / "bg_removed"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用 RMBG-2.0 的主模型
DEFAULT_MODEL_PATH = MODELS_DIR / "model.onnx"


class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量抠图"""

    def __init__(self, model_path: str | None = None, output_dir: str | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 初始化 ONNXRuntime 推理会话
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def remove_background(self, image_path: str) -> str:
        """输入图片路径，输出抠图后透明 PNG 文件路径"""
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # 模型期望输入分辨率（1024x1024）
        side = 1024
        img_rs = img.resize((side, side), Image.BICUBIC)

        # 归一化 + 调整维度
        arr = np.asarray(img_rs).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)

        # 模型推理
        input_name = self.session.get_inputs()[0].name
        pred = self.session.run(None, {input_name: arr})[0]

        # RMBG 2.0 输出掩码（可能带边缘增强）
        mask = pred[0, 0]
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)

        # 调整掩码尺寸回原图
        m = Image.fromarray(mask, "L").resize((orig_w, orig_h), Image.BICUBIC)

        # 合并 RGBA
        rgba = img.convert("RGBA")
        r, g, b, _ = rgba.split()
        out = Image.merge("RGBA", (r, g, b, m))

        # 轻微平滑边缘，增强自然感
        out = out.filter(ImageFilter.SMOOTH_MORE)

        name = Path(image_path).stem
        output_path = self.output_dir / f"{name}_rmbg2_removed.png"
        out.save(output_path)
        print(f"抠图完成: {output_path}")
        return str(output_path)


def local_tool_for_bg_remove(req: dict) -> str:
    """暴露统一接口"""
    remover = BriaRMBG2Remover(
        model_path=req.get("model_path"), output_dir=req.get("output_dir")
    )
    return remover.remove_background(req["image_path"])


def get_bg_remove_desc(lang: str = "zh") -> str:
    """中文说明"""
    return (
        "使用 BRIA-RMBG 2.0 模型执行高质量抠图，自动去除背景并输出带透明通道的 PNG 文件。"
        "支持多种输入格式（JPG/PNG/WebP），输出文件默认保存在同目录的 bg_removed/ 下。"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BRIA-RMBG 2.0 高质量抠图工具")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("--model_path", default=None, help="模型路径（可选）")
    parser.add_argument("--output_dir", default=None, help="输出目录（可选）")
    args = parser.parse_args()

    out = local_tool_for_bg_remove(
        {
            "image_path": args.image_path,
            "model_path": args.model_path,
            "output_dir": args.output_dir,
        }
    )
    print(out)
