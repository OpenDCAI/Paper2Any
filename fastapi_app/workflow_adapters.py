from __future__ import annotations

"""
将 dataflow_agent.workflow.* 中的工作流封装为可供 FastAPI 路由调用的纯 Python 函数。

这里主要参考 gradio_app 中已有的封装逻辑：
- gradio_app.pages.operator_write.run_operator_write_pipeline
- gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List

import json
import time

from dataflow_agent.state import Paper2FigureState, DFRequest, DFState, Paper2FigureRequest as DF_Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import (
    create_pipeline_graph,
)
from dataflow_agent.workflow.wf_pipeline_write import create_operator_write_graph

from .schemas import (
    OperatorWriteRequest,
    OperatorWriteResponse,
    PipelineRecommendRequest,
    PipelineRecommendResponse,
    Paper2FigureRequest,
    Paper2FigureResponse,
)

log = get_logger(__name__)




# ------------------- 算子编写工作流封装 -------------------


async def run_operator_write_pipeline_api(
    req: OperatorWriteRequest,
) -> OperatorWriteResponse:
    """
    基于 wf_pipeline_write.create_operator_write_graph 的算子编写封装。

    对标 gradio_app.pages.operator_write 中的 run_operator_write_pipeline，
    但输入输出均使用 Pydantic 模型，方便 FastAPI 直接返回。
    """
    # 设置环境变量
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    else:
        # 若未显式提供，则回落到环境变量或一个 dummy key
        req.api_key = os.getenv("DF_API_KEY", "sk-dummy")

    # 处理默认 json_file
    projdir = get_project_root()
    json_file = req.json_file or f"{projdir}/tests/test.jsonl"

    # 构造 DFRequest / DFState
    df_req = DFRequest(
        language=req.language,
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model,
        target=req.target,
        need_debug=req.need_debug,
        max_debug_rounds=req.max_debug_rounds,
        json_file=json_file,
    )
    state = DFState(request=df_req, messages=[])

    # 设置输出路径（如果提供）
    if req.output_path:
        state.temp_data["pipeline_file_path"] = req.output_path

    # 设置类别
    if req.category:
        state.temp_data["category"] = req.category

    # 初始化调试轮次
    state.temp_data["round"] = 0

    # 构建并执行工作流图
    graph = create_operator_write_graph().build()
    # 递归限制与 Gradio 版本保持一致：主链 4 步 + 每轮 5 步 * 轮次 + buffer 5
    recursion_limit = 4 + 5 * req.max_debug_rounds + 5
    final_state = await graph.ainvoke(
        state,
        config={"recursion_limit": recursion_limit},
    )

    # ---------- 提取结果（参考 gradio_app/pages/operator_write.py） ----------
    matched_ops: List[str] = []
    code_str: str = ""
    execution_result: Dict[str, Any] = {}
    debug_runtime: Dict[str, Any] = {}
    agent_results: Dict[str, Any] = {}

    # 提取匹配的算子
    try:
        if isinstance(final_state, dict):
            matched = final_state.get("matched_ops", [])
            if not matched:
                matched = (
                    final_state.get("agent_results", {})
                    .get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        else:
            matched = getattr(final_state, "matched_ops", [])
            if not matched and hasattr(final_state, "agent_results"):
                matched = (
                    final_state.agent_results.get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        matched_ops = list(matched or [])
    except Exception as e:  # pragma: no cover - 仅日志
        log.warning(f"[operator_write] 提取匹配算子失败: {e}")
        matched_ops = []

    # 提取生成的代码
    try:
        if isinstance(final_state, dict):
            temp_data = final_state.get("temp_data", {})
            code_str = (
                temp_data.get("pipeline_code", "") if isinstance(temp_data, dict) else ""
            )
        else:
            temp_data = getattr(final_state, "temp_data", {})
            code_str = (
                temp_data.get("pipeline_code", "") if isinstance(temp_data, dict) else ""
            )
    except Exception as e:  # pragma: no cover
        log.warning(f"[operator_write] 提取代码失败: {e}")
        code_str = ""

    # 提取执行结果
    try:
        if isinstance(final_state, dict):
            exec_res = final_state.get("execution_result", {}) or {}
            if not exec_res or ("success" not in exec_res):
                exec_res = (
                    final_state.get("agent_results", {})
                    .get("operator_executor", {})
                    .get("results", {})
                    or exec_res
                )
        else:
            exec_res = getattr(final_state, "execution_result", {}) or {}
            if (not exec_res or ("success" not in exec_res)) and hasattr(
                final_state, "agent_results"
            ):
                exec_res = (
                    final_state.agent_results.get("operator_executor", {}).get(
                        "results", {}
                    )
                    or exec_res
                )
        execution_result = dict(exec_res or {})
    except Exception as e:  # pragma: no cover
        log.warning(f"[operator_write] 提取执行结果失败: {e}")
        execution_result = {}

    # 提取调试运行时信息
    try:
        if isinstance(final_state, dict):
            dbg = (final_state.get("temp_data") or {}).get("debug_runtime")
        else:
            dbg = getattr(final_state, "temp_data", {}).get("debug_runtime")
        debug_runtime = dict(dbg or {})
    except Exception as e:  # pragma: no cover
        log.warning(f"[operator_write] 提取调试信息失败: {e}")
        debug_runtime = {}

    # 提取 agent_results
    try:
        if isinstance(final_state, dict):
            agent_results = dict(final_state.get("agent_results", {}) or {})
        else:
            agent_results = dict(getattr(final_state, "agent_results", {}) or {})
    except Exception as e:  # pragma: no cover
        log.warning(f"[operator_write] 提取 agent_results 失败: {e}")
        agent_results = {}

    # 构建日志信息（与 Gradio 版本类似，方便前端直接展示）
    log_lines: List[str] = []
    log_lines.append("==== 算子编写结果 ====")
    log_lines.append(f"\n匹配到的算子数量: {len(matched_ops)}")
    if matched_ops:
        log_lines.append(f"匹配的算子: {matched_ops}")

    log_lines.append(f"\n生成的代码长度: {len(code_str)} 字符")

    if execution_result:
        success_flag = execution_result.get("success", False)
        log_lines.append(f"\n执行成功: {success_flag}")
        if not success_flag:
            stderr = execution_result.get("stderr", "") or execution_result.get(
                "traceback", ""
            )
            if stderr:
                log_lines.append(f"\n错误信息:\n{stderr[:500]}")

    if debug_runtime:
        log_lines.append("\n==== 调试信息 ====")
        input_key = debug_runtime.get("input_key")
        available_keys = debug_runtime.get("available_keys", [])
        if input_key:
            log_lines.append(f"选择的输入键: {input_key}")
        if available_keys:
            log_lines.append(f"可用键: {available_keys}")
        stdout = debug_runtime.get("stdout", "")
        stderr = debug_runtime.get("stderr", "")
        if stdout:
            log_lines.append(f"\n标准输出:\n{stdout[:1000]}")
        if stderr:
            log_lines.append(f"\n标准错误:\n{stderr[:1000]}")

    log_text = "\n".join(log_lines)

    return OperatorWriteResponse(
        success=True,
        code=code_str or "",
        matched_ops=matched_ops,
        execution_result=execution_result,
        debug_runtime=debug_runtime,
        agent_results=agent_results,
        log=log_text,
    )


# ------------------- 流水线推荐工作流封装 -------------------


async def run_pipeline_recommend_api(
    req: PipelineRecommendRequest,
) -> PipelineRecommendResponse:
    """
    基于 wf_pipeline_recommend_extract_json.create_pipeline_graph 的封装。

    对标 gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow。
    """
    # 环境变量设置
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
        os.environ["DF_API_URL"] = req.chat_api_url

    project_root: Path = get_project_root()
    tmps_dir: Path = project_root / "dataflow_agent" / "tmps"

    # 对 session_id 做一次 URL-safe 的 base64 编码，确保目录名安全
    session_id_encoded = base64.urlsafe_b64encode(req.session_id.encode()).decode()
    session_dir: Path = tmps_dir / session_id_encoded
    session_dir.mkdir(parents=True, exist_ok=True)

    python_file_path = session_dir / "pipeline.py"

    df_req = DFRequest(
        language="en",
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model_name,
        json_file=req.json_file,
        target=req.target,
        python_file_path=str(python_file_path),
        need_debug=req.need_debug,
        session_id=session_id_encoded,
        max_debug_rounds=req.max_debug_rounds,
        chat_api_url_for_embeddings=req.chat_api_url_for_embeddings,
        embedding_model_name=req.embedding_model_name,
        update_rag_content=req.update_rag_content,
    )

    state = DFState(request=df_req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = True

    graph = create_pipeline_graph().build()
    final_state = await graph.ainvoke(state)

    # 对齐原实现：execution_result 使用 debug_history
    if isinstance(final_state, dict):
        debug_history = dict(final_state.get("debug_history", {}) or {})
        agent_results = dict(final_state.get("agent_results", {}) or {})
    else:
        debug_history = dict(getattr(final_state, "debug_history", {}) or {})
        agent_results = dict(getattr(final_state, "agent_results", {}) or {})

    return PipelineRecommendResponse(
        success=True,
        python_file=str(df_req.python_file_path),
        execution_result=debug_history,
        agent_results=agent_results,
    )

# --------------------------Paper2Figure----------------------------

# ====================== 通用工具函数 ====================== #
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

def save_final_state_json(final_state: dict, out_dir: Path, filename: str = "final_state.json") -> None:
    """
    直接把 final_state 用 json.dump 存到 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
    遇到无法序列化的对象用 str 兜底。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    print(f"final_state 已保存到 {out_path}")

# ====================== 主函数 ====================== #
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

    # -------- 映射到 dataflow_agent.state.Paper2FigureRequest -------- #
    # df_req = DF_Paper2FigureRequest(
    #     language=req.language,
    #     chat_api_url=req.chat_api_url,
    #     api_key=req.api_key or req.chat_api_key,
    #     model=req.model,
    # )
    # 透传额外字段
    # df_req.input_type = req.input_type
    # df_req.chat_api_key = req.chat_api_key
    # df_req.input_content = req.input_content
    # df_req.gen_fig_model = req.gen_fig_model
    # df_req.bg_rm_model = req.bg_rm_model
    # df_req["graph_type"] = req.graph_type
    # df_req["style"] = req.style

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
        result_root = project_root / "outputs" / "paper2fig" / ts
    elif graph_type == "tech_route":
        wf_name = "paper2technical"
        result_root = project_root / "outputs" / "paper2tec" / ts
    elif graph_type == "exp_data":
        # TODO: 后续接入 paper2exp workflow
        wf_name = "paper2fig_with_sam"
        result_root = project_root / "outputs" / "paper2exp" / ts
    else:
        wf_name = "paper2fig_with_sam"
        result_root = project_root / "outputs" / "paper2fig" / ts

    result_root.mkdir(parents=True, exist_ok=True)
    state.result_path = str(result_root)
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

    return Paper2FigureResponse(
        success=True,
        ppt_filename=ppt_filename,
        svg_filename=svg_filename,
        svg_image_filename=svg_image_filename,
    )
