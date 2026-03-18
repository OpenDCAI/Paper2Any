from __future__ import annotations

"""
paper2figure 工作流适配器。

职责：接收 FastAPI 路由层传来的 Paper2FigureRequest，
根据 input_type × graph_type 两个维度选择对应的 workflow，
执行后将结果封装成 Paper2FigureResponse 返回。

支持的组合：
  input_type: PDF | TEXT | FIGURE
  graph_type: model_arch | tech_route | exp_data | (其他兜底)
"""

import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow

from fastapi_app.schemas import Paper2FigureRequest, Paper2FigureResponse

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_serializable(obj: Any) -> Any:
    """递归将任意对象转成可 JSON 序列化的结构，无法处理的类型用 str 兜底。"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_final_state_json(final_state: dict, out_dir: Path, filename: str = "final_state.json") -> None:
    """将 final_state 序列化后保存到 out_dir/filename，用于调试。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"final_state 已保存到 {out_path}")


def _resolve_image_to_local(image_path_or_url: str, project_root: Path) -> str:
    """
    将图片路径/URL 统一转换为本地绝对路径。

    前端传来的图片路径有两种形式：
      - 完整 URL：http://localhost:8000/outputs/xxx/yyy.png
      - 相对路径：/outputs/xxx/yyy.png
    两者都包含 /outputs/，统一提取后半段拼成本地绝对路径。
    如果文件不存在或转换失败，原样返回，由后续 workflow 自行处理。
    """
    if not image_path_or_url or "/outputs/" not in image_path_or_url:
        return image_path_or_url

    try:
        relative_path = image_path_or_url.split("/outputs/", 1)[1]
        relative_path = relative_path.split("?", 1)[0]          # 去掉查询参数
        relative_path = urllib.parse.unquote(relative_path)      # 解码 URL 编码
        local_path = project_root / "outputs" / relative_path
        if local_path.exists():
            log.info(f"[paper2figure] 图片路径转换: {image_path_or_url} -> {local_path}")
            return str(local_path)
        else:
            log.warning(f"[paper2figure] 本地文件不存在: {local_path}")
    except Exception as e:
        log.warning(f"[paper2figure] 图片路径转换失败: {e}")

    return image_path_or_url


def _resolve_workflow(graph_type: str, input_type: str, edit_prompt: str) -> tuple[str, str]:
    """
    根据 graph_type 和 input_type 决定使用哪个 workflow 及对应的 task 目录名。

    model_arch 有一个特殊子分支：
      - input_type=FIGURE 且没有 edit_prompt → 用户想把图片直接转成 PPT，走 pdf2ppt_qwenvl
      - 其他情况 → 走标准图像生成 paper2fig_image_only

    返回: (wf_name, task_name)
    """
    if graph_type == "model_arch":
        if input_type == "FIGURE" and not edit_prompt:
            # 图片转 PPT 模式（无编辑提示词，直接转换）
            return "pdf2ppt_qwenvl", "paper2fig_ppt"
        else:
            # 标准模型架构图生成
            return "paper2fig_image_only", "paper2fig"
    elif graph_type == "tech_route":
        return "paper2technical", "paper2tec"
    elif graph_type == "exp_data":
        return "paper2expfigure", "paper2exp"
    else:
        # 兜底 workflow
        return "paper2fig_with_sam", "paper2fig"


def _build_result_root(result_path: Path | None, project_root: Path, email: str, task_name: str, ts: str) -> Path:
    """
    确定本次任务的输出根目录。

    优先使用调用方传入的 result_path（路由层统一分配）；
    未传入时按 outputs/{email}/{task_name}/{ts} 自动生成。
    """
    if result_path:
        return Path(result_path).resolve()

    user_dir = email or ""
    return (project_root / "outputs" / user_dir / task_name / ts).resolve()


def _get_state_attr(state: Any, key: str, default: str = "") -> str:
    """兼容 final_state 为 dict 或 State 对象两种情况，安全取值并转为 str。"""
    if isinstance(state, dict):
        return str(state.get(key, default) or default)
    return str(getattr(state, key, default) or default)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def run_paper2figure_wf_api(req: Paper2FigureRequest, result_path: Path | None = None) -> Paper2FigureResponse:
    """
    paper2figure 工作流主入口。

    Args:
        req: 前端通过 FormData 映射而来的请求参数，关键字段：
             - input_type:   "PDF" | "TEXT" | "FIGURE"
             - input_content: PDF 路径 / 纯文本 / 图片路径或 URL
             - graph_type:   "model_arch" | "tech_route" | "exp_data"
        result_path: 可选，由路由层指定的输出目录；未指定时自动生成。

    Returns:
        Paper2FigureResponse，包含生成的 PPT/SVG 文件路径列表。
    """
    project_root = get_project_root()
    tmps_dir = project_root / "dataflow_agent" / "tmps"
    tmps_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 1. 初始化 State，根据 input_type 设置对应的输入字段
    # ------------------------------------------------------------------
    state = Paper2FigureState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.aspect_ratio = req.aspect_ratio

    if req.input_type == "PDF":
        # PDF 模式：直接把文件路径交给 workflow 解析
        state.paper_file = req.input_content

    elif req.input_type == "TEXT":
        # 文本模式：用户输入的文字描述作为创作素材
        state.paper_idea = req.input_content

    elif req.input_type == "FIGURE":
        # 图片编辑/重绘模式：前端传来的可能是 URL 或相对路径，需转为本地绝对路径
        local_image = _resolve_image_to_local(req.input_content, project_root)
        req.input_content = local_image          # 同步更新 req，确保 workflow 内部也能读到
        state.request.prev_image = local_image   # workflow 读取上一张图的字段
        state.fig_draft_path = local_image       # wf_paper2expfigure 使用的草图路径
        state.paper_idea = "Image Edit Mode"     # 防止某些节点因 paper_idea 为空而报错

    else:
        raise TypeError(f"不支持的 input_type: {req.input_type}，可选值: PDF, TEXT, FIGURE")

    # ------------------------------------------------------------------
    # 2. 根据 graph_type × input_type 选择 workflow 和输出目录
    # ------------------------------------------------------------------
    wf_name, task_name = _resolve_workflow(req.graph_type, req.input_type, req.edit_prompt)
    result_root = _build_result_root(result_path, project_root, req.email, task_name, ts)
    result_root.mkdir(parents=True, exist_ok=True)

    state.result_path = str(result_root)
    state.mask_detail_level = 2
    log.info(f"[paper2figure] workflow={wf_name}, result_path={result_root}")

    # ------------------------------------------------------------------
    # 3. 技术路线图专属参数（参考图 + 二次编辑提示词）
    # ------------------------------------------------------------------
    if req.graph_type == "tech_route":
        if req.reference_image_path:
            # 用户上传的参考图，VLM 会据此生成相似风格的技术路线图
            state.temp_data["reference_image_path"] = req.reference_image_path
            log.info(f"[paper2figure] 参考图: {req.reference_image_path}")
        if req.tech_route_edit_prompt:
            # 用户对已生成图的二次编辑指令
            state.temp_data["tech_route_edit_prompt"] = req.tech_route_edit_prompt
            log.info(f"[paper2figure] 二次编辑提示词: {req.tech_route_edit_prompt}")

    # ------------------------------------------------------------------
    # 4. 执行 workflow
    # ------------------------------------------------------------------
    log.info(f"[paper2figure] language={req.language}, palette={req.tech_route_palette!r}")
    final_state: Paper2FigureState = await run_workflow(wf_name, state)

    # 保存完整 state 到 tmps/ 供调试
    save_final_state_json(to_serializable(final_state), out_dir=tmps_dir / ts)
    log.info(f"[paper2figure] 完成，ppt_path={final_state['ppt_path']}")

    # ------------------------------------------------------------------
    # 5. 收集输出文件，构造响应
    # ------------------------------------------------------------------
    ppt_filename = _get_state_attr(final_state, "ppt_path")

    # SVG 相关路径仅 tech_route 会有值，其他类型返回空字符串
    svg_filename        = _get_state_attr(final_state, "svg_file_path")
    svg_image_filename  = _get_state_attr(final_state, "svg_img_path")
    # 黑白版：优先取专用字段，没有则回退到普通 svg
    svg_bw_filename       = _get_state_attr(final_state, "svg_bw_file_path") or svg_filename
    svg_bw_image_filename = _get_state_attr(final_state, "svg_bw_img_path") or svg_image_filename
    # 彩色版
    svg_color_filename       = _get_state_attr(final_state, "svg_color_file_path")
    svg_color_image_filename = _get_state_attr(final_state, "svg_color_img_path")

    # 扫描输出目录，收集所有 PPTX / PNG / SVG 文件供前端展示
    all_output_files: list[str] = []
    try:
        for p in Path(state.result_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".pptx", ".png", ".svg"}:
                all_output_files.append(str(p))
    except Exception as e:
        log.warning(f"[paper2figure] 收集输出文件列表失败: {e}")

    return Paper2FigureResponse(
        success=True,
        ppt_filename=ppt_filename,
        svg_filename=svg_filename,
        svg_image_filename=svg_image_filename,
        svg_bw_filename=svg_bw_filename,
        svg_bw_image_filename=svg_bw_image_filename,
        svg_color_filename=svg_color_filename,
        svg_color_image_filename=svg_color_image_filename,
        all_output_files=all_output_files,
    )
