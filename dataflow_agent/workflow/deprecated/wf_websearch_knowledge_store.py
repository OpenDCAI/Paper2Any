"""
websearch_knowledge_store workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
根据用户提供的 Agent 设计，基于 LangGraph / GenericGraphBuilder
搭建一个「只包含节点与流转逻辑」的基础图。

全局状态字段（见 `WebsearchKnowledgeState`）：
- input_urls: 用户初始输入的 URL 列表
- research_routes: 研究计划队列
- raw_data_store: 原始数据仓库
- knowledge_base_summary: 最终结构化知识总结

节点角色：
- planner          : 全知指挥官
- initial_analyzer : 初始分析师
- web_researcher   : 外勤研究员
- chief_curator    : 首席馆长


"""

from __future__ import annotations

from typing import List

from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import WebsearchKnowledgeState

log = get_logger(__name__)


@register("websearch_knowledge_store")
def create_websearch_knowledge_store_graph() -> GenericGraphBuilder:  # noqa: N802
    """
    Workflow factory: dfa run --wf websearch_knowledge_store
    
    仅负责：
    - 声明全局状态类型 `WebsearchKnowledgeState`
    - 搭建节点与流转关系（基于 LangGraph 条件边）

    """
    builder = GenericGraphBuilder(
        state_model=WebsearchKnowledgeState,
        entry_point="planner",  # 流程从 Planner 开始
    )

    async def planner_node(state: WebsearchKnowledgeState) -> WebsearchKnowledgeState:
        """
        Planner Node (全知指挥官)
        
        职责：
        - 管理任务队列 research_routes
        - 处理 Researcher 的产出并入库
        - 设定下一个执行任务 current_task
        """
        from dataflow_agent.agentroles.paper2any_agents.websearch_planner import create_websearch_planner_agent
        planner = create_websearch_planner_agent()
        await planner.run(state)
        return state

    async def initial_analyzer_node(state: WebsearchKnowledgeState) -> WebsearchKnowledgeState:
        """
        Initial Analyzer (初始分析师)
        
        职责：
        - 访问 Input URLs
        - 提取正文并保存 DOM
        - 产出初始研究路线 research_routes
        """
        from dataflow_agent.agentroles.paper2any_agents.websearch_initial_analyzer import create_websearch_initial_analyzer_agent
        analyzer = create_websearch_initial_analyzer_agent()
        await analyzer.run(state)
        return state

    async def web_researcher_node(state: WebsearchKnowledgeState) -> WebsearchKnowledgeState:
        """
        Web Researcher (外勤研究员)
        
        职责：
        - 依据 current_task 进行联网深度搜索
        - 抓取新网页内容，保存 DOM
        """
        from dataflow_agent.agentroles.paper2any_agents.websearch_researcher import create_websearch_researcher_agent
        researcher = create_websearch_researcher_agent()
        await researcher.run(state)
        return state

    async def chief_curator_node(state: WebsearchKnowledgeState) -> WebsearchKnowledgeState:
        """
        Chief Curator (首席馆长)
        
        职责：
        - 读取 Raw Data Store 全量数据
        - 生成最终结构化知识总结 knowledge_base_summary
        """
        from dataflow_agent.agentroles.paper2any_agents.websearch_curator import create_websearch_curator_agent
        curator = create_websearch_curator_agent()
        await curator.run(state)
        return state

    # 简单终止节点：直接回传状态
    def _end_node(state: WebsearchKnowledgeState) -> WebsearchKnowledgeState:
        log.debug("[_end_] reached end of websearch_knowledge_store workflow.")
        return state

    # ==============================================================
    # 注册 nodes
    # ==============================================================
    nodes = {
        "planner": planner_node,
        "initial_analyzer": initial_analyzer_node,
        "web_researcher": web_researcher_node,
        "chief_curator": chief_curator_node,
        "_end_": _end_node,
    }

    builder.add_nodes(nodes)

    # ==============================================================
    # EDGES: 非条件边（执行完之后统一回到 planner）
    # ==============================================================
    edges: List[tuple[str, str]] = [
        ("initial_analyzer", "planner"),
        ("web_researcher", "planner"),
        ("chief_curator", "planner"),
    ]
    builder.add_edges(edges)

    # ==============================================================
    # CONDITIONAL EDGES: Planner 的流转逻辑
    # ==============================================================
    def planner_condition(state: WebsearchKnowledgeState) -> str:
        """
        Planner 条件路由逻辑，对应用户描述中的：
        - 判断逻辑 A: Research Routes 为空？Input URLs 有内容吗？
        - 判断逻辑 B: Research Routes 里有未执行的计划吗？
        - 判断逻辑 C: 计划都执行完了吗？内容尚未清洗入库吗？
        - 判断逻辑 D: 计划都执行完了吗？内容都清洗入库了吗？
        
        约定（仅作图阶段的简化）：
        - `research_routes` 作为待执行计划队列：
          - 非空 → 代表还有未执行计划
          - 由各节点自行维护入队 / 出队
        - `raw_data_store` 非空且 `knowledge_base_summary` 为空 → 代表需要 Chief Curator 清洗
        - `knowledge_base_summary` 非空 → 代表已完成清洗入库
        """
        # 从主 state 或 request 上获取用户输入 URL
        input_urls = state.input_urls or getattr(state.request, "input_urls", [])
        research_routes = state.research_routes
        raw_data_store = state.raw_data_store
        knowledge_base_summary = state.knowledge_base_summary

        # 判断逻辑 B: Research Routes 里有未执行的计划吗？
        if research_routes:
            # 有计划就优先执行计划（执行 Web Researcher）
            return "web_researcher"

        # 判断逻辑 A: Research Routes 为空，Input URLs 有内容吗？
        if not research_routes and input_urls and not raw_data_store:
            # 还没有做过初始分析，但有 URL，可以进入 Initial Analyzer
            return "initial_analyzer"

        # 判断逻辑 C: 计划都执行完，但内容尚未清洗入库
        if (not research_routes) and raw_data_store and not knowledge_base_summary:
            return "chief_curator"

        # 判断逻辑 D: 计划都执行完，内容都清洗入库
        if (not research_routes) and knowledge_base_summary:
            return "_end_"

        # 兜底：如果没有 URL 且无计划也无数据，直接结束
        return "_end_"

    builder.add_conditional_edges({"planner": planner_condition})

    return builder






