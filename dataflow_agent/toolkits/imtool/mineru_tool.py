# Before Using, do these:

# hf download opendatalab/MinerU2.5-2509-1.2B --local-dir opendatalab/MinerU2.5-2509-1.2B

# With vllm>=0.10.1, you can use following command to serve the model. The logits processor is used to support no_repeat_ngram_size sampling param, which can help the model to avoid generating repeated content.

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port> \
#   --logits-processors mineru_vl_utils:MinerULogitsProcessor
# If you are using vllm<0.10.1, no_repeat_ngram_size sampling param is not supported. You still can serve the model without logits processor:

# vllm serve models/MinerU2.5-2509-1.2B \
#     --host 127.0.0.1 \
#     --port 8001 \
#     --logits-processors mineru_vl_utils:MinerULogitsProcessor \
#     --gpu-memory-utilization 0.6

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port>


from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional
import os
import shutil
import subprocess

from PIL import Image
from mineru_vl_utils import MinerUClient


# ---------------------------------------
# 1. two_step_extract (sync)
# ---------------------------------------
def run_two_step_extract(image_path: str, port: int):
    """同步调用 MinerU two_step_extract，处理单张图片并返回结构化结果。"""
    image = Image.open(image_path)
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return client.two_step_extract(image)


# ---------------------------------------
# 2. batch_two_step_extract (sync)
# ---------------------------------------
def run_batch_two_step_extract(image_paths: list[str], port: int):
    """同步批量调用 MinerU two_step_extract，处理多张图片并返回结果列表。"""
    images = [Image.open(p) for p in image_paths]
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return client.batch_two_step_extract(images)


# ---------------------------------------
# 3. aio_two_step_extract (async)
# ---------------------------------------
async def run_aio_two_step_extract(image_path: str, port: int):
    """异步调用 MinerU two_step_extract，处理单张图片并返回结构化结果。"""
    image = Image.open(image_path)
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return await client.aio_two_step_extract(image)


# ---------------------------------------
# 4. aio_batch_two_step_extract (async)
# ---------------------------------------
async def run_aio_batch_two_step_extract(image_paths: list[str], port: int):
    """异步批量调用 MinerU two_step_extract，处理多张图片并返回结果列表。"""
    images = [Image.open(p) for p in image_paths]
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return await client.aio_batch_two_step_extract(images)


# ---------------------------------------
# 5. 根据 MinerU bbox & type 裁剪原图
# ---------------------------------------
def crop_mineru_blocks_by_type(
    image_path: str,
    blocks: List[Dict[str, Any]],
    target_type: Optional[Union[str, Sequence[str]]] = None,
    output_dir: str = "",
    prefix: str = "",
) -> List[str]:
    """
    根据 MinerU two_step_extract / aio_two_step_extract 的结构化结果，
    按指定 type 的 bbox 从整张图片中裁剪子图并保存到输出目录。

    参数:
        image_path: 原始图片路径 (如技术路线图 PNG)
        blocks: MinerU 返回的 list[dict] 结果
        target_type: 需要裁剪的块类型，如 "title" / "text" / "image" / "footer"
        output_dir: 输出目录路径，不存在会自动创建
        prefix: 输出文件名前缀，可选

    返回:
        所有成功保存的裁剪图片的绝对路径列表
    """
    img = Image.open(image_path)
    width, height = img.size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    # target_type 为 None 时不过滤，返回所有 block
    if target_type is None:
        target_types = None
    elif isinstance(target_type, str):
        target_types = {target_type}
    else:
        target_types = set(target_type)

    for idx, block in enumerate(blocks):
        block_type = block.get("type")
        # 只有在显式指定了 target_type 时才进行过滤
        if target_types is not None and block_type not in target_types:
            continue

        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = bbox

        # 将归一化坐标 [0,1] 转为像素坐标，并做边界裁剪
        left = max(0, min(width, int(round(x1_norm * width))))
        top = max(0, min(height, int(round(y1_norm * height))))
        right = max(0, min(width, int(round(x2_norm * width))))
        bottom = max(0, min(height, int(round(y2_norm * height))))

        # 无效 bbox 跳过
        if right <= left or bottom <= top:
            continue

        cropped = img.crop((left, top, right, bottom))

        # 使用实际 block_type 命名，便于区分不同类型
        safe_block_type = block_type or "unknown"
        filename = f"{prefix}{safe_block_type}_{idx}.png"
        out_path = out_dir / filename
        cropped.save(out_path)

        saved_paths.append(str(out_path.resolve()))

    return saved_paths

def run_mineru_pdf_extract(
    pdf_path: str,
    output_dir: str = "",
    source: str = "modelscope",
    mineru_executable: Optional[str] = None,
):
    """
    使用 MinerU 命令行方式提取 PDF 中的结构化内容，

    参数:
        pdf_path: PDF 文件路径
        output_dir: 输出目录路径，不存在会自动创建
        source: 下载模型的源，可选 modelscope、huggingface
        mineru_executable: mineru 可执行文件路径，
            - 不传时：优先从环境变量 MINERU_CMD 中读取，
              若没有则从 PATH 中查找 'mineru'
            - 传入绝对路径时：直接使用该路径

    返回:
        解析的所有图片、markdown格式的内容
    """
    # 1. 解析 mineru 可执行路径
    if mineru_executable is None:
        mineru_executable = (
            os.environ.get("MINERU_CMD")  # 环境变量优先
            or shutil.which("mineru")     # 当前 env 的命令
        )
        if mineru_executable is None:
            raise RuntimeError(
                "未找到 `mineru` 可执行文件，请确保：\n"
                "1) 已在当前环境安装 MinerU，并且 `mineru` 在 PATH 中；或\n"
                "2) 设置环境变量 MINERU_CMD 指向 mineru 可执行文件；或\n"
                "3) 调用 run_mineru_pdf_extract 时显式传入 mineru_executable 参数。"
            )

    mineru_cmd = [
        str(mineru_executable),
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "--source",
        source,
    ]

    # 2. 可选：自动创建 output_dir
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3. 执行命令
    subprocess.run(
        mineru_cmd,
        shell=False,
        check=True,
        text=True,
        stderr=None,
        stdout=None,
    )



def crop_mineru_blocks_with_meta(
    image_path: str,
    blocks: List[Dict[str, Any]],
    target_type: Optional[Union[str, Sequence[str]]] = None,
    output_dir: str = "",
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    与 ``crop_mineru_blocks_by_type`` 类似，但返回包含元信息的列表，
    方便后续根据 MinerU 的 bbox 在 PPT 中按比例还原布局。

    返回的每个元素包含:
        - block_index: 在原始 blocks 列表中的索引
        - type: MinerU 块类型
        - bbox: 原始归一化 bbox [x1, y1, x2, y2]
        - png_path: 裁剪得到的小图 PNG 绝对路径
    """
    img = Image.open(image_path)
    width, height = img.size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # target_type 为 None 时不过滤，返回所有 block
    if target_type is None:
        target_types = None
    elif isinstance(target_type, str):
        target_types = {target_type}
    else:
        target_types = set(target_type)

    for idx, block in enumerate(blocks):
        block_type = block.get("type")
        # 只有在显式指定了 target_type 时才进行过滤
        if target_types is not None and block_type not in target_types:
            continue

        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = bbox

        # 将归一化坐标 [0,1] 转为像素坐标，并做边界裁剪
        left = max(0, min(width, int(round(x1_norm * width))))
        top = max(0, min(height, int(round(y1_norm * height))))
        right = max(0, min(width, int(round(x2_norm * width))))
        bottom = max(0, min(height, int(round(y2_norm * height))))

        if right <= left or bottom <= top:
            continue

        cropped = img.crop((left, top, right, bottom))

        safe_block_type = block_type or "unknown"
        filename = f"{prefix}{safe_block_type}_{idx}.png"
        out_path = out_dir / filename
        cropped.save(out_path)

        results.append(
            {
                "block_index": idx,
                "type": block_type,
                "bbox": bbox,
                "png_path": str(out_path.resolve()),
            }
        )

    return results


def svg_to_emf(svg_path: str, emf_path: str, dpi: int = 600) -> str:
    """
    使用 Inkscape 将 SVG 文件转换为 EMF 矢量图，返回生成的 EMF 路径。
    使用 Inkscape 将 SVG 转换为 EMF 矢量图。

    依赖
    ----
    - 系统需安装 Inkscape，并且 `inkscape` 在 PATH 中可直接调用。

    参数
    ----
    svg_path:
        输入 SVG 文件路径。
    emf_path:
        输出 EMF 文件路径。

    返回
    ----
    str
        生成的 EMF 文件的绝对路径。

    异常
    ----
    FileNotFoundError
        当输入 SVG 文件不存在时。
    RuntimeError
        当 Inkscape 调用失败或未生成输出文件时。
    """
    svg_p = Path(svg_path)
    if not svg_p.exists():
        raise FileNotFoundError(f"输入 SVG 不存在: {svg_p}")

    emf_p = Path(emf_path)
    emf_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        # inkscape input.svg --export-filename=output.emf
        result = subprocess.run(
            [
                "inkscape",
                str(svg_p),
                "--export-filename",
                str(emf_p),
                "--export-text-to-path",
                f"--export-dpi={dpi}"
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "调用 Inkscape 失败：系统中可能未安装 `inkscape` 可执行文件，"
            "请先安装 Inkscape 并确保其在 PATH 中。"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Inkscape 转换失败，返回码 {result.returncode}：\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    if not emf_p.exists():
        raise RuntimeError(f"Inkscape 运行后未发现输出 EMF 文件: {emf_p}")

    return str(emf_p.resolve())


# ---------------------------------------
# 6. 递归 MinerU 拆图 + 坐标映射 (HTTP 版)
# ---------------------------------------
def _crop_image_by_norm_bbox(
    image_path: str,
    bbox: Sequence[float],
    output_dir: Union[str, Path],
    prefix: str = "",
    index: int = 0,
) -> str:
    """
    按归一化 bbox [x1,y1,x2,y2] 从 image_path 裁剪出子图并保存，返回绝对路径。
    """
    img = Image.open(image_path)
    width, height = img.size

    x1_norm, y1_norm, x2_norm, y2_norm = bbox
    left = max(0, min(width, int(round(x1_norm * width))))
    top = max(0, min(height, int(round(y1_norm * height))))
    right = max(0, min(width, int(round(x2_norm * width))))
    bottom = max(0, min(height, int(round(y2_norm * height))))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clamp: {bbox}")

    cropped = img.crop((left, top, right, bottom))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    filename = f"{prefix}{stem}_{index}.png"
    out_path = out_dir / filename
    cropped.save(out_path)

    return str(out_path.resolve())


async def recursive_mineru_layout(
    image_path: str,
    port: int,
    max_depth: int = 2,
    current_depth: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    block_types_for_subimage: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    使用 MinerU HTTP two_step_extract，递归拆图并将所有最底层块映射到
    最顶层图的归一化坐标系。

    返回的每个元素形如:
        {
            "type": str,
            "bbox": [x1, y1, x2, y2],   # 相对于最顶层图的归一化坐标
            "png_path": str | None,     # 对应子图路径（图像/表格等）
            "text": str | None,         # 文本内容（若有）
            "depth": int,               # 所在递归深度
        }
    """
    if current_depth > max_depth:
        return []

    # 默认在原图同目录下创建一个 mineru_recursive 子目录
    if output_dir is None:
        base = Path(image_path).with_suffix("")
        output_dir = base.parent / f"{base.stem}_mineru_recursive"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 默认哪些类型会继续拆成子图
    if block_types_for_subimage is None:
        block_types_for_subimage = ["image", "img", "table", "figure"]

    # 1. 当前层 MinerU 调用
    blocks = await run_aio_two_step_extract(image_path=image_path, port=port)

    leaf_items: List[Dict[str, Any]] = []

    # blocks 结构假定为 List[Dict]，包含 type / bbox / text 等
    for idx, blk in enumerate(blocks):
        blk_type = blk.get("type")
        bbox = blk.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # 保证归一化 bbox 在 [0,1] 内，大致裁剪
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y2 = max(0.0, min(1.0, float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        norm_bbox = [x1, y1, x2, y2]

        # 如果是需要继续拆的图块类型，裁剪子图并递归
        if blk_type in block_types_for_subimage and current_depth < max_depth:
            try:
                sub_img_path = _crop_image_by_norm_bbox(
                    image_path=image_path,
                    bbox=norm_bbox,
                    output_dir=out_dir / "sub_images",
                    prefix=f"depth{current_depth}_blk{idx}_",
                    index=idx,
                )
            except Exception:
                # 裁剪失败则当成叶子块处理
                leaf_items.append(
                    {
                        "type": blk_type,
                        "bbox": norm_bbox,
                        "png_path": None,
                        "text": blk.get("text") or blk.get("content"),
                        "depth": current_depth,
                    }
                )
                continue

            # 子图内部是完整的 [0,1] 坐标系，需要映射回当前图的 norm_bbox
            sub_items = await recursive_mineru_layout(
                image_path=sub_img_path,
                port=port,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                output_dir=out_dir,
                block_types_for_subimage=block_types_for_subimage,
            )

            pw = norm_bbox[2] - norm_bbox[0]
            ph = norm_bbox[3] - norm_bbox[1]
            for si in sub_items:
                sb = si.get("bbox")
                if not sb or len(sb) != 4:
                    continue
                sx1, sy1, sx2, sy2 = sb
                nx1 = norm_bbox[0] + sx1 * pw
                ny1 = norm_bbox[1] + sy1 * ph
                nx2 = norm_bbox[0] + sx2 * pw
                ny2 = norm_bbox[1] + sy2 * ph
                si["bbox"] = [nx1, ny1, nx2, ny2]
                leaf_items.append(si)
        else:
            # 文本或其他不再下钻的类型，直接当作叶子
            leaf_items.append(
                {
                    "type": blk_type,
                    "bbox": norm_bbox,
                    "png_path": None,
                    "text": blk.get("text") or blk.get("content"),
                    "depth": current_depth,
                }
            )

    return leaf_items
