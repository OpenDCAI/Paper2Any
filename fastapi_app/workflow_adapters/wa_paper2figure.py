from __future__ import annotations

"""
paper2figure 工作流封装。

从原来的单文件 workflow_adapters.py / paper2_modules.py 拆分而来，
只保留与 paper2figure 相关的封装逻辑。
"""

import json
import time
from pathlib import Path
from typing import Any

from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow

from fastapi_app.schemas import Paper2FigureRequest, Paper2FigureResponse

log = get_logger(__name__)


# -------------------------- Paper2Figure ----------------------------


def to_serializable(obj: Any):
    """递归将对象转成可 JSON 序列化结构"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_final_state_json(
    final_state: dict, out_dir: Path, filename: str = "final_state.json"
) -> None:
    """
    直接把 final_state 用 json.dump 存到 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
    遇到无法序列化的对象用 str 兜底。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    print(f"final_state 已保存到 {out_path}")


async def run_paper2figure_wf_api(req: Paper2FigureRequest) -> Paper2FigureResponse:
    """
    根据 graph_type 选择不同 workflow，并拆分输出目录。

    入参 req 通常由 FastAPI 路由层（如 paper2any.generate_paper2figure）
    根据前端 FormData 映射而来：
      - input_type: "PDF" / "TEXT" / "FIGURE"
      - input_content: 文件路径或纯文本
      - graph_type: "model_arch" | "tech_route" | "exp_data"
    """
    # -------- 基础路径与输出目录 -------- #
    project_root: Path = get_project_root()
    tmps_dir: Path = project_root / "dataflow_agent" / "tmps"
    tmps_dir.mkdir(parents=True, exist_ok=True)

    state = Paper2FigureState(request=req, messages=[])
    state.temp_data["round"] = 0

    # 根据 input_type / input_content 设置具体输入
    if req.input_type == "PDF":
        state.paper_file = req.input_content
    elif req.input_type == "TEXT":
        state.paper_idea = req.input_content
    elif req.input_type == "FIGURE":
        state.fig_draft_path = req.input_content
    else:
        raise TypeError("Invalid input type. Available input type: PDF, TEXT, FIGURE.")

    # 其它控制参数
    state.aspect_ratio = req.aspect_ratio

    # -------- 按 graph_type 决定 workflow + 输出根目录 -------- #
    ts = time.strftime("%Y%m%d_%H%M%S")
    graph_type = req.graph_type

    if graph_type == "model_arch":
        wf_name = "paper2fig_with_sam"
        result_root = project_root / "outputs" / req.invite_code / "paper2fig" / ts
    elif graph_type == "tech_route":
        wf_name = "paper2technical"
        result_root = project_root / "outputs" / req.invite_code / "paper2tec" / ts
    elif graph_type == "exp_data":
        wf_name = "paper2expfigure"
        result_root = project_root / "outputs" / req.invite_code / "paper2exp" / ts
    else:
        wf_name = "paper2fig_with_sam"
        result_root = project_root / "outputs" / req.invite_code / "paper2fig" / ts

    result_root.mkdir(parents=True, exist_ok=True)
    state.result_path = str(result_root)
    log.critical(f"[paper2figure] result_path: {state.result_path} !!!!!!!!\n")
    state.mask_detail_level = 2

    # -------- 异步执行 -------- #
    log.critical(f"[paper2figure] req: {req} !!!!!!!!\n")
    final_state: Paper2FigureState = await run_workflow(wf_name, state)

    # -------- 保存最终 State -------- #
    serializable_state = to_serializable(final_state)
    save_final_state_json(
        final_state=serializable_state,
        out_dir=tmps_dir / ts,
    )

    log.info(f"[paper2figure] Results saved in directory: {state.result_path}")
    log.info(f"[paper2figure]: {final_state['ppt_path']}")

    # -------- 构造响应：根据 graph_type 返回不同字段 -------- #
    ppt_filename = str(final_state["ppt_path"])

    # 默认空字符串，避免 None 影响前端
    svg_filename = ""
    svg_image_filename = ""

    try:
        # final_state 可能是 State 或 dict，两种方式都考虑
        if isinstance(final_state, dict):
            svg_filename = str(final_state.get("svg_file_path", "") or "")
            svg_image_filename = str(final_state.get("svg_img_path", "") or "")
        else:
            svg_filename = str(getattr(final_state, "svg_file_path", "") or "")
            svg_image_filename = str(getattr(final_state, "svg_img_path", "") or "")
    except Exception as e:  # pragma: no cover - 仅日志兜底
        log.warning(f"[paper2figure] 提取 SVG 路径失败: {e}")
        svg_filename = ""
        svg_image_filename = ""

    # 收集本次任务输出目录下的所有 PPTX / PNG / SVG 文件绝对路径
    all_output_files: list[str] = []
    try:
        result_root_path = Path(state.result_path)
        if result_root_path.exists():
            for p in result_root_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".pptx", ".png", ".svg"}:
                    all_output_files.append(str(p))
    except Exception as e:  # pragma: no cover
        log.warning(f"[paper2figure] 收集输出文件列表失败: {e}")

    return Paper2FigureResponse(
        success=True,
        ppt_filename=ppt_filename,
        svg_filename=svg_filename,
        svg_image_filename=svg_image_filename,
        all_output_files=all_output_files,
    )
