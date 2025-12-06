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
# import onnxruntime as ort
# import shutil
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "onnx" / "model.onnx"
OUTPUT_DIR = CURRENT_DIR


def ensure_model(model_path: Path):
    """
    若 model_path 不存在，则从 ModelScope 下载 RMBG-2.0 到该路径。
    """
    if model_path.exists():
        print(f"模型已存在: {model_path}")
        return

    print("未检测到模型文件，正在下载 RMBG-2.0 权重...")

    # 确保目录存在
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 直接下载到目标目录
    cmd = (
        f"modelscope download "
        f"--model AI-ModelScope/RMBG-2.0 "
        f"--local_dir '{model_path.parent}' "
    )
    os.system(cmd)

    # 检查下载是否成功
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型下载失败：未找到 {model_path}。\n"
            "请检查 ModelScope 或手动下载。"
        )

    print(f"模型已成功下载到: {model_path}")

class BriaRMBG2Remover:
    def __init__(self, model_path: Path = None, output_dir: Path = None):
        """
        model_path: 本地 RMBG-2.0 模型目录（包含 model 文件）
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

        ensure_model(self.model_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        print(f"加载本地权重: {model_path}")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_path,
            trust_remote_code=True
        ).eval().to(device)

        # Transform pipeline
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def remove_background(self, image_path: str) -> str:
        """
        对输入图像进行背景抠图，并保存到 output_dir
        """
        image_path = Path(image_path)
        print(f"开始抠图: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform_image(image).unsqueeze(0).to(self.device)

        # Predict mask
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        # Resize mask back to original size
        mask = pred_pil.resize(image.size)

        # Apply alpha mask
        out = image.copy()
        out.putalpha(mask)

        # Save output
        out_path = self.output_dir / f"{image_path.stem}_bg_removed.png"
        out.save(out_path)

        print(f"抠图完成: {out_path}")
        return str(out_path)


# class BriaRMBG2Remover:
#     """使用 BRIA-RMBG 2.0 模型进行高质量抠图"""

#     def __init__(self, model_path: str | None = None, output_dir: str | None = None):
#         self.model_path = Path(model_path) if model_path else MODEL_PATH
#         self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

#         ensure_model(self.model_path)

#         # 优先使用 GPU，否则退回 CPU
#         providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         self.session = ort.InferenceSession(str(self.model_path), providers=providers)


#     def remove_background(self, image_path: str) -> str:
#         img = Image.open(image_path).convert("RGB")
#         orig_w, orig_h = img.size

#         side = 1024
#         img_rs = img.resize((side, side), Image.BICUBIC)
#         arr = np.asarray(img_rs).astype(np.float32) / 255.0
#         arr = arr.transpose(2, 0, 1)[None, ...]

#         input_name = self.session.get_inputs()[0].name
#         pred = self.session.run(None, {input_name: arr})[0]

#         mask = np.clip(pred[0, 0], 0, 1)
#         mask = (mask * 255).astype(np.uint8)
#         m = Image.fromarray(mask, "L").resize((orig_w, orig_h), Image.BICUBIC)

#         rgba = img.convert("RGBA")
#         r, g, b, _ = rgba.split()
#         out = Image.merge("RGBA", (r, g, b, m)).filter(ImageFilter.SMOOTH_MORE)

#         out_path = self.output_dir / f"{Path(image_path).stem}_bg_removed.png"
#         out.save(out_path)
#         print(f"抠图完成: {out_path}")
#         return str(out_path)


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
