from __future__ import annotations

import asyncio, os, json, pathlib, dataclasses
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
    local_tool_for_get_categories,
    get_otherinfo_code
)
from dataflow_agent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
    post_process_combine_pipeline_result,
    get_operators_by_rag,
    get_operators_info_by_names,
)
# from dataflow_agent.toolkits.pipetool.pipe_tools import parse_pipeline_file
import dataflow_agent.toolkits.pipetool.pipe_tools as pt
from dataflow_agent.agentroles.classifier import create_classifier
from dataflow_agent.agentroles.target_parser import create_target_parser
from dataflow_agent.agentroles.recommender import create_recommender
from dataflow_agent.agentroles.pipelinebuilder import create_pipeline_builder
from dataflow_agent.agentroles.debugger import create_code_debugger
from dataflow_agent.agentroles.rewriter import create_rewriter
from dataflow_agent.agentroles.inforequester import create_info_requester
from dataflow_agent.agentroles.exporter import create_exporter


from dataflow_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.utils import get_project_root
from dataflow_agent.logger import get_logger
MAX_DEBUG_ROUNDS = 3
log = get_logger(__name__)

PROJDIR = get_project_root()

# ======================================================================
# create_pipeline_graph
# ======================================================================
def create_pipeline_graph() -> GenericGraphBuilder:
    # ★ 1. 图入口：classifier
    builder = GenericGraphBuilder(state_model=DFState, entry_point="classifier")

    # ------------------------------------------------------------------
    # Ⅰ. 工具注册
    # ------------------------------------------------------------------
    # -------- classifier 前置工具 --------
    @builder.pre_tool("sample", "classifier")
    def cls_get_sample(state: DFState):
        # 取 2 条样本
        return local_tool_for_sample(state.request, sample_size=2)["samples"]

    @builder.pre_tool("categories", "classifier")
    def cls_get_categories(state: DFState):
        return local_tool_for_get_categories()

    # -------- target_parser 前置工具 --------
    @builder.pre_tool("target", "target_parser")
    def tp_get_target(state: DFState):
        return state.request.target

    # -------- recommender 前置工具 --------
    @builder.pre_tool("sample", "exporter")
    @builder.pre_tool("sample", "recommender")
    def rec_get_sample(state: DFState):
        # 推荐器只拿 1 条样本
        return local_tool_for_sample(state.request, sample_size=1)["samples"]

    @builder.pre_tool("target", "recommender")
    def rec_get_target(state: DFState):
        # 返回 target_parser 拆解出的算子描述列表
        operator_descriptions = state.temp_data.get('operator_descriptions', [])
        return operator_descriptions

    @builder.pre_tool("operator", "recommender")
    def rec_get_operator(state: DFState):
        # 使用 target_parser_node 中 RAG 检索出的算子作为候选集
        split_ops = state.temp_data.get('split_ops', [])
        if split_ops:
            return get_operators_info_by_names(split_ops)
        # 兜底：如果没有 RAG 结果，使用默认全量算子
        return get_operator_content_str(data_type="Default")

    class GetOpInput(BaseModel):
        oplist: list = Field(description="list['xxx']的算子列表")

    @builder.post_tool("recommender")
    @tool(args_schema=GetOpInput)
    def combine_pipeline(oplist: list) -> str:
        """Combine pipeline post tool for recommender"""
        return post_process_combine_pipeline_result(oplist)

    # -------- debugger / rewriter 前置工具 --------
    @builder.pre_tool("pipeline_code", "code_debugger")
    def get_pipeline_code_for_debug(state: DFState):
        return state.temp_data.get("pipeline_code", "")

    @builder.pre_tool("error_trace", "code_debugger")
    def get_error_trace_for_debug(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("stdout", "")

    @builder.pre_tool("pipeline_code", "rewriter")
    def get_pipeline_code_for_rewrite(state: DFState):
        return state.temp_data.get("pipeline_code", "")

    @builder.pre_tool("error_trace", "rewriter")
    def get_error_trace_for_rewrite(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("stdout", "")

    @builder.pre_tool("debug_reason", "rewriter")
    def get_debug_reason(state: DFState):
        return state.code_debug_result.get("reason", "")

    @builder.pre_tool("data_sample", "rewriter")
    def get_data_sample(state: DFState):
        return local_tool_for_sample(state.request, sample_size=1)["samples"]
    
    @builder.pre_tool("other_info", "rewriter")
    def get_other_info(state: DFState):
        return state.temp_data.get("other_info_summary", "")
    
    # --------  inforequest 前置工具 --------
    @builder.pre_tool("pipeline_code", "info_requester")
    def ir_pipeline_code(state: DFState):
        return state.temp_data.get("pipeline_code", "")

    @builder.pre_tool("error_trace", "info_requester")
    def ir_error_trace(state: DFState):
        return state.execution_result.get("stderr", "") \
            or state.execution_result.get("stdout", "")
    
    class ModuleListInput(BaseModel):
        module_list: list = Field(
            description="List of dotted-path python modules or file paths"
        )
    @builder.post_tool("info_requester")
    @tool(args_schema=ModuleListInput)
    def fetch_other_info(module_list: list) -> dict:
        """Return source code for requested modules"""
        log.error(f'fetch_other_info：{module_list}')
        return get_otherinfo_code(module_list)

    @builder.pre_tool('nodes_info', "exporter")
    def get_nodes_info(s: DFState):
        return s.temp_data.get("structure_code", {}).get("nodes", [])
    

    # ------------------------------------------------------------------
    # Ⅱ. 节点实现
    # ------------------------------------------------------------------
    async def classifier_node(s: DFState) -> DFState:
        """Classifier 节点"""
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        classifier = create_classifier(tool_manager=get_tool_manager())
        s = await classifier.execute(s, use_agent=False)
        return s

    async def target_parser_node(s: DFState) -> DFState:
        """目标意图理解节点 - 将 target 拆解为算子描述列表，并进行 RAG 检索"""
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        
        parser = create_target_parser(tool_manager=get_tool_manager())
        s = await parser.execute(s, use_agent=False)
        
        operator_descriptions = s.temp_data.get('operator_descriptions', [])
        log.info(f"[target_parser_node] 拆解算子描述: {operator_descriptions}")
        
        # 在意图解析后立即进行 RAG 检索，获取候选算子
        if operator_descriptions:
            s.temp_data['split_ops'] = get_operators_by_rag(
                search_queries=operator_descriptions,
                top_k=2,
                faiss_index_path=f"{PROJDIR}/dataflow_agent/resources/faiss_cache/all_ops.index",
                base_url=f"{s.request.chat_api_url}embeddings" if s.request.chat_api_url_for_embeddings == "" else s.request.chat_api_url_for_embeddings,
                model_name=s.request.embedding_model_name
            )
            # 展平结果
            s.temp_data['split_ops'] = [item for sub in s.temp_data['split_ops'] for item in sub]
            log.info(f"[target_parser_node] RAG 检索结果: {s.temp_data['split_ops']}")
        else:
            s.temp_data['split_ops'] = []
        
        return s

    # --- recommender 子图同原来 ---
    async def build_recommender_subgraph(init_state: DFState):
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        tm = get_tool_manager()
        recommender = create_recommender(tool_manager=tm)
        init_state = await recommender.execute(init_state, use_agent=True)

        rec_inst = init_state.temp_data["recommender_instance"]
        pre_results = init_state.temp_data["pre_tool_results"]
        post_tools = rec_inst.get_post_tools()

        sg = StateGraph(DFState)
        sg.add_node("assistant", rec_inst.create_assistant_node_func(init_state, pre_results))
        if post_tools:
            sg.add_node("tools", ToolNode(post_tools))
            sg.add_conditional_edges("assistant", tools_condition)
            sg.add_edge("tools", "assistant")
        sg.set_entry_point("assistant")
        return sg.compile()

    async def build_info_requester_subgraph(init_state: DFState):
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        tm = get_tool_manager()
        requester = create_info_requester(tool_manager=tm, tool_mode = "auto")
        init_state = await requester.execute(init_state, use_agent=True)

        req_inst   = init_state.temp_data["info_requester_instance"]
        pre_result = init_state.temp_data["pre_tool_results"]
        post_tools = req_inst.get_post_tools()

        sg = StateGraph(DFState)
        sg.add_node("assistant", req_inst.create_assistant_node_func(init_state, pre_result))
        if post_tools:
            sg.add_node("tools", ToolNode(post_tools))
            sg.add_conditional_edges("assistant", tools_condition)
            sg.add_edge("tools", "assistant")
        sg.set_entry_point("assistant")
        return sg.compile()

    async def recommender_node(s: DFState) -> DFState:
        from dataflow.utils.registry import OPERATOR_REGISTRY  
        _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}

        rec_graph = await build_recommender_subgraph(s)
        result = await rec_graph.ainvoke(s)
        if isinstance(result, dict):
            for k, v in result.items():         
                setattr(s, k, v)
        else:
            for f in dataclasses.fields(DFState):
                setattr(s, f.name, getattr(result, f.name))
        rec_val = s.get("recommendation", {})
        ops_list = rec_val.get("ops", []) if isinstance(rec_val, dict) else rec_val
        if not ops_list:
            raise ValueError("Recommender 没有返回有效算子列表")
        from dataflow_agent.toolkits.optool.op_tools import get_operators_by_rag
        
        # 直接使用公共索引进行RAG检索
        ops_list = get_operators_by_rag(search_queries=ops_list, 
                                        top_k=1, 
                                        faiss_index_path=f"{PROJDIR}/dataflow_agent/resources/faiss_cache/all_ops.index",
                                        base_url=f"{s.request.chat_api_url}embeddings" if s.request.chat_api_url_for_embeddings == "" else s.request.chat_api_url_for_embeddings,
                                        model_name = s.request.embedding_model_name)
        ops_list = [item for sub in ops_list for item in sub]   # 展平

        # 如果要求RAG实时更新，检测未注册的算子；如有则删除索引文件并重跑RAG
        if s.request.update_rag_content:
            missing_ops = [op for op in ops_list if op not in _NAME2CLS]
            if missing_ops:
                log.warning("[recommender] 检测到未注册算子 %s ，重新生成索引", missing_ops)

                # 删除索引文件(.index & .meta)
                index_path = f"{PROJDIR}/dataflow_agent/resources/faiss_cache/all_ops.index"
                for suffix in (".index", ".index.meta"):
                    p = pathlib.Path(index_path + suffix) if suffix.endswith(".meta") else pathlib.Path(index_path)
                    if p.exists():
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass

                # 重新检索（此时 get_operators_by_rag 会自动重新构建索引）
                ops_list = get_operators_by_rag(search_queries=ops_list, 
                                        top_k=1, 
                                        faiss_index_path=index_path,
                                        base_url=f"{s.request.chat_api_url}embeddings" if s.request.chat_api_url_for_embeddings == "" else s.request.chat_api_url_for_embeddings,
                                        model_name = s.request.embedding_model_name)
                ops_list = [item for sub in ops_list for item in sub]

        log.warning(f"[recommender_node + RAG ] 推荐算子列表：{ops_list}")
        s["recommendation"] = ops_list
        s.agent_results['recommender']['results']['ops'] = ops_list
        return s

    async def builder_node(s: DFState) -> DFState:
        builder_agent = create_pipeline_builder()
        skip = bool(s.temp_data.get("rewritten", False))
        log.warning(f"[builder_node] skip_assemble = {skip}, rewritten = {s.temp_data.get('rewritten', False)}")
        return await builder_agent.execute(
            s,
            skip_assemble=skip,
            file_path= s.request.python_file_path,
            assembler_kwargs={"file_path": s.request.json_file, "chat_api_url": s.request.chat_api_url, "cache_dir": s.request.cache_dir},
        )

    async def debugger_node(s: DFState) -> DFState:
        """
        • 运行代码调试器  
        • 把本轮 execution_result 和 code_debug_result 一起写入 debug_history  
        • 自增轮次 round/debug_round
        """

        round_id = s.temp_data.get("round", 0)
        log.info(f"[debugger_node] 当前是第 {round_id} 轮调试，准备调试…")

        debugger = create_code_debugger(tool_manager=get_tool_manager())
        s = await debugger.execute(s, use_agent=True)   # <-- 得到最新 execution_result & code_debug_result

        # -------- 保存历史 --------
        s.debug_history[round_id] = {
            "execution_result": dict(s.execution_result),
            "code_debug_result": dict(s.code_debug_result),
        }

        # s.temp_data["round"] = round_id + 1
        # s.debug_round = round_id + 1         
        return s

    async def rewriter_node(s: DFState) -> DFState:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager())
        return await rewriter.execute(s, use_agent=True)

    def after_rewrite_node(s: DFState) -> DFState:

        rewriter = create_rewriter(tool_manager=get_tool_manager())
        return rewriter.after_rewrite(s)
    
    async def info_requester_node(s: DFState) -> DFState:
        info_graph = await build_info_requester_subgraph(s)
        result = await info_graph.ainvoke(s)
        if isinstance(result, dict):
            for k, v in result.items():
                setattr(s, k, v)
        else:
            import dataclasses
            for f in dataclasses.fields(DFState):
                setattr(s, f.name, getattr(result, f.name))
        return s

    async def exporter_node(s: DFState) -> DFState:
        """
        将 builder 生成的 python 流水线文件解析为 JSON 结构并落盘
        """
        py_path = s.temp_data.get("pipeline_file_path") or s.request.python_file_path
        if not py_path:
            raise RuntimeError("exporter_node: 找不到 pipeline 源文件")
        
        graph = pt.parse_pipeline_file(py_path)
        s.temp_data["structure_code"] = graph
        s.pipeline_structure_code = graph

        exporter = create_exporter(
            tool_manager=get_tool_manager(),    
            react_mode=True, 
            react_max_retries=3
        )
        s = await exporter.execute(s)
        out_path = pathlib.Path(py_path).with_suffix(".json")
        if s.request.need_debug == False:
            # 不需要调试模式，使用 nodes_info 自动生成边
            print(s.nodes_info)
            graph = pt.build_edges_from_nodes(s.nodes_info, out_path)
        else:
            # 需要调试模式，解析 Python 文件
            out_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"[exporter] structure saved -> {out_path}")
        return s


    # ------------------------------------------------------------------
    # Ⅲ. 条件函数
    # ------------------------------------------------------------------
    def builder_condition(s: DFState):
        if s.request.need_debug:
            # ① 仅当调试阶段成功，才回到 builder
            if (
                s.execution_result.get("success")
                and s.temp_data.pop("debug_sample_file", None)  
            ): 
                log.warning('再次进入循环builder！！')
                return "builder"

            # ② 正式流程成功 → exporter
            if s.execution_result.get("success"):
                return "exporter"

            # ③ debug 轮次达到上限 → 结束
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "code_debugger"
        else:
            # 非调试模式，成功或失败都直接去 exporter
            return "exporter"

    # ------------------------------------------------------------------
    # Ⅳ. 组图
    # ------------------------------------------------------------------
    nodes = {
        "classifier": classifier_node,
        "target_parser": target_parser_node,
        "recommender": recommender_node,
        "builder": builder_node,
        "code_debugger": debugger_node,
        "info_requester": info_requester_node,   
        "rewriter": rewriter_node,
        "after_rewrite": after_rewrite_node,
        "exporter": exporter_node,           
    }

    edges = [
        ("classifier", "target_parser"),
        ("target_parser", "recommender"),
        ("recommender", "builder"),
        ("code_debugger", "info_requester"),  
        ("info_requester", "rewriter"),       
        ("rewriter", "after_rewrite"),
        ("after_rewrite", "builder"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"builder": builder_condition})
    return builder

# 暴露出去
from dataflow_agent.workflow.registry import register

@register("pipelinerecommend")
def factory():
    return create_pipeline_graph()
