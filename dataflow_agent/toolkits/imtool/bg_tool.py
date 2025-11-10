# ================================================================
# BRIA-RMBG 2.0 高质量抠图工具
# - 模型：RMBG 2.0（ONNX）
# - 依赖：onnxruntime, pillow, numpy
# ================================================================

from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import onnxruntime as ort
import shutil

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "onnx" / "model.onnx"
OUTPUT_DIR = CURRENT_DIR


def ensure_model():
    """确保 RMBG 模型存在，如不存在则自动从 ModelScope 下载"""
    if MODEL_PATH.exists():
        print(f"模型已存在: {MODEL_PATH}")
        return

    print("未检测到模型，正在从 ModelScope 下载 onnx/model.onnx ...")
    os.system(
        f"cd '{CURRENT_DIR}' && "
        "modelscope download --model AI-ModelScope/RMBG-2.0 onnx/model.onnx"
    )

    # 清理临时目录
    temp_dir = CURRENT_DIR / ".____temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    #Step 1: 在当前目录搜索
    found = list(CURRENT_DIR.rglob("model.onnx"))

    #Step 2: 若未找到，则在 ModelScope 缓存目录搜索
    if not found:
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub"
        if cache_dir.exists():
            found = list(cache_dir.rglob("model.onnx"))

    if not found:
        raise FileNotFoundError("模型下载失败：未在本地或缓存中找到 model.onnx 文件。")

    # 找到最新的 model.onnx 文件
    src = max(found, key=lambda p: p.stat().st_mtime)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 若模型不在标准路径，则复制过去
    if src.resolve() != MODEL_PATH.resolve():
        shutil.copy2(src, MODEL_PATH)
        print(f"检测到模型实际位置: {src}")
        print(f"已复制到标准路径: {MODEL_PATH}")
    else:
        print(f"模型下载完成: {MODEL_PATH}")


class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量抠图"""

    def __init__(self, model_path: str | None = None, output_dir: str | None = None):
        ensure_model()
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

        # 优先使用 GPU，否则退回 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)


    def remove_background(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        side = 1024
        img_rs = img.resize((side, side), Image.BICUBIC)
        arr = np.asarray(img_rs).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]

        input_name = self.session.get_inputs()[0].name
        pred = self.session.run(None, {input_name: arr})[0]

        mask = np.clip(pred[0, 0], 0, 1)
        mask = (mask * 255).astype(np.uint8)
        m = Image.fromarray(mask, "L").resize((orig_w, orig_h), Image.BICUBIC)

        rgba = img.convert("RGBA")
        r, g, b, _ = rgba.split()
        out = Image.merge("RGBA", (r, g, b, m)).filter(ImageFilter.SMOOTH_MORE)

        out_path = self.output_dir / f"{Path(image_path).stem}_bg_removed.png"
        out.save(out_path)
        print(f"抠图完成: {out_path}")
        return str(out_path)


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
