from __future__ import annotations
import json
from dataflow_agent.state import DFState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
    local_tool_for_get_match_operator_code,
    search_operator_by_description,
    get_operator_code_by_name,
    _get_operators_by_rag_with_scores,
    _determine_match_quality,
    _generate_match_warning,
    MATCH_QUALITY_THRESHOLDS,
)

from dataflow_agent.agentroles.data_agents.refine import (
    create_refine_target_analyzer,
    create_refine_planner,
    create_json_pipeline_refiner,
)
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import robust_parse_json

log = get_logger(__name__)

def create_pipeline_refine_graph() -> GenericGraphBuilder:
    """
    flow: target_analyzer -> refine_planner -> pipeline_refiner
    仅修改 state.pipeline_structure_code（JSON），每一步使用 LLM 分析与生成。
    已实现：
    1) target_analyzer 返回所需子操作（含 step_id/action/desc/position_hint），并对每个子操作分别做RAG匹配top1算子与代码；
    2) 聚合匹配结果为 op_context（按 step_id 对齐）传递给 pipeline_refiner；
    3) 移除独立的 match_op 节点。

    """
    builder = GenericGraphBuilder(state_model=DFState, entry_point="refine_target_analyzer")

    # --------------------- refine_target_analyzer -------------------------
    @builder.pre_tool("purpose", "refine_target_analyzer")
    def _purpose(state: DFState) -> str:
        return local_tool_for_get_purpose(state.request)

    @builder.pre_tool("get_pipeline_code", "refine_target_analyzer")
    def get_pipeline_code(state: DFState) -> str:
        return json.dumps(state.pipeline_structure_code or {}, ensure_ascii=False, indent=2)

    @builder.pre_tool("pipeline_nodes_summary", "refine_target_analyzer")
    def _pipeline_nodes_summary(state: DFState):
        nodes = (state.pipeline_structure_code or {}).get("nodes", [])
        summary = []
        for n in nodes or []:
            run_cfg = ((n or {}).get("config", {}) or {}).get("run", {}) or {}
            summary.append({
                "id": n.get("id"),
                "name": n.get("name"),
                "type": n.get("type"),
                "run": {k: run_cfg.get(k) for k in ["input_key", "output_key"] if k in run_cfg}
            })
        return summary

    async def target_analyzer_node(s: DFState) -> DFState:
        agent = create_refine_target_analyzer()
        s2 = await agent.execute(s, use_agent=False)

        # 基于目标分析输出的子操作描述，逐条做RAG匹配top1算子与代码
        try:
            intent = s2.agent_results.get("refine_target_analyzer", {}).get("results", {}) or {}
            subs = intent.get("needed_operators_desc", []) or []
            # 兼容 dict 形式
            if isinstance(subs, dict):
                subs = [
                    {
                        "step_id": k,
                        **({} if not isinstance(v, dict) else v),
                        **({"desc": v} if isinstance(v, str) else {}),
                    }
                    for k, v in subs.items()
                ]
            op_contexts = []
            # 准备公共算子目录
            cat = s2.category.get("category") if isinstance(s2.category, dict) else None
            data_type = cat or "Default"
            operator_catalog = get_operator_content_str(data_type=data_type)

            for item in subs:
                desc = (item or {}).get("desc") or ""
                step_id = (item or {}).get("step_id") or ""
                action = (item or {}).get("action") or ""
                if not desc or not step_id:
                    continue

                # 使用带分数的 RAG 搜索，获取相似度和匹配质量
                matched_name = None
                code_block = ""
                match_quality = "low"
                max_score = 0.0
                warning_msg = None
                matched_operators = []

                try:
                    # 调用 RAG 搜索，返回带分数的结果
                    matched_operators = _get_operators_by_rag_with_scores(desc, top_k=4)

                    if matched_operators:
                        # 获取最高分数和匹配质量
                        max_score = max(op.get("similarity_score", 0.0) for op in matched_operators)
                        match_quality = _determine_match_quality(max_score)
                        warning_msg = _generate_match_warning(desc, max_score, match_quality)

                        # 获取最佳匹配的算子名称
                        matched_name = matched_operators[0].get("name") if matched_operators else None

                        # 获取算子源码
                        if matched_name:
                            try:
                                code_block = local_tool_for_get_match_operator_code({"match_operators": [matched_name]}) or ""
                            except Exception:
                                code_block = ""

                        log.info(f"[RAG匹配] 需求: '{desc}' -> 最佳匹配: {matched_name}, "
                                f"相似度: {max_score:.3f}, 质量: {match_quality}")
                except Exception as e:
                    log.warning(f"[RAG匹配] 搜索失败: {e}")
                    matched_name = None
                    code_block = ""

                op_contexts.append({
                    "step_id": step_id,
                    "action": action,
                    "matched_name": matched_name,
                    "matched_operators": matched_operators,  # 所有候选算子（含分数）
                    "max_similarity_score": max_score,
                    "match_quality": match_quality,
                    "warning": warning_msg,
                    "code_snippet": code_block,
                })

            # 汇总写入 agent_results，供后续节点作为 op_context 使用
            s2.agent_results["op_contexts"] = op_contexts
        except Exception:
            # 容错：不中断流程
            s2.agent_results["op_contexts"] = []

        return s2

    # --------------------- refine_planner -------------------------
    @builder.pre_tool("intent", "refine_planner")
    def _intent(state: DFState):
        try:
            return state.agent_results.get("refine_target_analyzer", {}).get("results", {})
        except Exception:
            return {"raw_target": getattr(state.request, "target", "")}

    @builder.pre_tool("pipeline_nodes_summary", "refine_planner")
    def _planner_summary(state: DFState):
        return _pipeline_nodes_summary(state)

    @builder.pre_tool("op_context", "refine_planner")
    def _planner_opctx(state: DFState):
        return state.agent_results.get("op_contexts", [])

    async def refine_planner_node(s: DFState) -> DFState:
        agent = create_refine_planner()
        s2 = await agent.execute(s, use_agent=False)
        return s2

    # --------------------- pipeline_refiner -------------------------
    async def pipeline_refiner_node(s: DFState) -> DFState:
        # 创建 ToolManager 并注册工具
        tm = ToolManager()

        # 注册前置工具（原本通过 @builder.pre_tool 注册的）
        tm.register_pre_tool(
            name="pipeline_json",
            role="pipeline_refiner",
            func=lambda: json.dumps(s.pipeline_structure_code or {}, ensure_ascii=False, indent=2)
        )
        tm.register_pre_tool(
            name="modification_plan",
            role="pipeline_refiner",
            func=lambda: (
                s.agent_results.get("refine_planner", {}).get("results", {}).get("modification_plan")
                if isinstance(s.agent_results.get("refine_planner", {}).get("results", {}), dict)
                   and isinstance(s.agent_results.get("refine_planner", {}).get("results", {}).get("modification_plan"), list)
                else s.agent_results.get("refine_planner", {}).get("results", {})
            )
        )
        tm.register_pre_tool(
            name="op_context",
            role="pipeline_refiner",
            func=lambda: s.agent_results.get("op_contexts", [])
        )

        # 注册后置工具（RAG 搜索工具）
        tm.register_post_tool(search_operator_by_description, role="pipeline_refiner")
        tm.register_post_tool(get_operator_code_by_name, role="pipeline_refiner")

        # 创建 agent 并传入 tool_manager
        agent = create_json_pipeline_refiner(tool_manager=tm)
        # 使用 use_agent=True 启用 graph agent 模式，让 LLM 可以调用 RAG 工具
        s2 = await agent.execute(s, use_agent=True)
        # 直接覆盖写回（按需求不做校验）
        try:
            result = s2.agent_results.get("pipeline_refiner", {}).get("results", {})
            # 1) 直接字典形式
            if isinstance(result, dict) and result.get("nodes") and result.get("edges"):
                s2.pipeline_structure_code = result
            else:
                # 2) 回退：尝试解析 raw 文本中的 JSON
                raw_txt = ""
                if isinstance(result, dict) and "raw" in result:
                    raw_txt = result.get("raw") or ""
                elif isinstance(result, str):
                    raw_txt = result
                if raw_txt:
                    try:
                        parsed = robust_parse_json(raw_txt)
                        if isinstance(parsed, dict) and parsed.get("nodes") and parsed.get("edges"):
                            s2.pipeline_structure_code = parsed
                    except Exception:
                        pass
        except Exception:
            pass
        return s2

    nodes = {
        "refine_target_analyzer": target_analyzer_node,
        "refine_planner": refine_planner_node,
        "pipeline_refiner": pipeline_refiner_node,
    }
    edges = [("refine_target_analyzer", "refine_planner"), ("refine_planner", "pipeline_refiner")]
    builder.add_nodes(nodes).add_edges(edges)
    return builder

