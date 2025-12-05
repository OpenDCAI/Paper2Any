"""
OperatorQA Agent - 算子问答 Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01 15:05:06
生成位置: dataflow_agent/agentroles/common_agents/operator_qa_agent.py

本文件实现了基于 Agentic RAG 的算子问答 Agent，支持：
1. 自然语言查询算子功能
2. 查询算子参数含义
3. 查看算子源码
4. 多轮对话

架构设计：
- OperatorRAGService: 封装所有 RAG 相关逻辑，便于后续升级为 Agentic RAG
- OperatorQAAgent: Agent 核心类，负责对话管理和 LLM 交互
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.graphbuilder.message_history import AdvancedMessageHistory

log = get_logger(__name__)


# ==============================================================================
# OperatorRAGService - RAG 服务类（解耦设计，便于后续升级为 Agentic RAG）
# ==============================================================================
class OperatorRAGService:
    """
    算子 RAG 检索服务
    
    封装所有 RAG 相关逻辑，与 Agent 解耦，便于后续升级为更复杂的 Agentic RAG。
    
    功能：
    - search(query): 向量检索相关算子
    - get_operator_info(names): 获取算子基本信息
    - get_operator_source(name): 获取算子源码
    - get_operator_params(name): 获取参数详情
    """
    
    def __init__(
        self,
        ops_json_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_url: str = "http://123.129.219.111:3000/v1/embeddings",
        api_key: Optional[str] = None,
    ):
        """
        初始化 RAG 服务
        
        Args:
            ops_json_path: 算子 JSON 文件路径，默认使用项目内置路径
            faiss_index_path: FAISS 索引文件路径，默认使用项目内置路径
            embedding_model: embedding 模型名称
            embedding_api_url: embedding API 地址
            api_key: API 密钥
        """
        from dataflow_agent.utils import get_project_root
        
        project_root = get_project_root()
        self.ops_json_path = ops_json_path or str(
            project_root / "dataflow_agent/toolkits/resources/ops.json"
        )
        self.faiss_index_path = faiss_index_path or str(
            project_root / "dataflow_agent/resources/faiss_cache/all_ops.index"
        )
        self.embedding_model = embedding_model
        self.embedding_api_url = embedding_api_url
        self.api_key = api_key or os.getenv("DF_API_KEY")
        
        self._searcher = None  # 懒加载
        
    def _get_searcher(self):
        """获取或创建 RAG 检索器（懒加载）"""
        if self._searcher is None:
            from dataflow_agent.toolkits.optool.op_tools import RAGOperatorSearch
            self._searcher = RAGOperatorSearch(
                ops_json_path=self.ops_json_path,
                faiss_index_path=self.faiss_index_path,
                model_name=self.embedding_model,
                base_url=self.embedding_api_url,
                api_key=self.api_key,
            )
        return self._searcher
    
    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
    ) -> Union[List[str], List[List[str]]]:
        """
        向量检索相关算子
        
        Args:
            query: 查询字符串或查询列表
            top_k: 返回 top-k 个结果
            
        Returns:
            算子名称列表
        """
        try:
            searcher = self._get_searcher()
            return searcher.search(query, top_k=top_k)
        except Exception as e:
            log.error(f"RAG 检索失败: {e}")
            return [] if isinstance(query, str) else [[] for _ in query]
    
    def get_operator_info(self, names: List[str]) -> str:
        """
        获取算子基本信息（名称、描述、分类）
        
        Args:
            names: 算子名称列表
            
        Returns:
            JSON 格式的算子信息字符串
        """
        from dataflow_agent.toolkits.optool.op_tools import get_operators_info_by_names
        try:
            return get_operators_info_by_names(names)
        except Exception as e:
            log.error(f"获取算子信息失败: {e}")
            return "[]"
    
    def get_operator_source(self, name: str) -> str:
        """
        获取算子源码
        
        Args:
            name: 算子名称
            
        Returns:
            源码字符串
        """
        from dataflow_agent.toolkits.optool.op_tools import get_operator_source_by_name
        try:
            return get_operator_source_by_name(name)
        except Exception as e:
            log.error(f"获取算子源码失败: {e}")
            return f"# 无法获取算子 '{name}' 的源码: {e}"
    
    def get_operator_params(self, name: str) -> Dict[str, Any]:
        """
        获取算子参数详情（init 和 run 参数）
        
        Args:
            name: 算子名称
            
        Returns:
            包含 init 和 run 参数的字典
        """
        import json
        try:
            info_str = self.get_operator_info([name])
            info_list = json.loads(info_str)
            if info_list and len(info_list) > 0:
                op_info = info_list[0]
                return op_info.get("parameter", {"init": [], "run": []})
            return {"init": [], "run": []}
        except Exception as e:
            log.error(f"获取算子参数失败: {e}")
            return {"init": [], "run": [], "error": str(e)}
    
    def search_and_get_info(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        综合检索：先向量检索，再获取详细信息
        
        Args:
            query: 查询字符串
            top_k: 返回 top-k 个结果
            
        Returns:
            包含检索结果和详细信息的字典
        """
        import json
        
        # 1. 向量检索
        matched_names = self.search(query, top_k=top_k)
        
        # 2. 获取详细信息
        if matched_names:
            info_str = self.get_operator_info(matched_names)
            try:
                info_list = json.loads(info_str)
            except:
                info_list = []
        else:
            info_list = []
        
        return {
            "query": query,
            "matched_operators": matched_names,
            "operator_details": info_list,
        }


# ==============================================================================
# OperatorQAAgent - Agent 核心类
# ==============================================================================
@register("operator_qa")
class OperatorQAAgent(BaseAgent):
    """
    算子问答 Agent
    
    支持功能：
    1. 自然语言查询算子功能（"我想过滤掉缺失值用哪个算子？"）
    2. 查询特定算子做什么（"df.filter_by 是干嘛的？"）
    3. 查询算子参数含义（"算子的 run 函数里面参数是什么意思？"）
    4. 查看算子源码
    5. 多轮对话
    """
    
    def __init__(
        self,
        tool_manager: Optional[ToolManager] = None,
        rag_service: Optional[OperatorRAGService] = None,
        **kwargs,
    ):
        """
        初始化 OperatorQAAgent
        
        Args:
            tool_manager: 工具管理器
            rag_service: RAG 服务实例，如果不传则自动创建
            **kwargs: 传递给 BaseAgent 的其他参数
        """
        # 启用多轮对话支持
        kwargs.setdefault("ignore_history", False)
        super().__init__(tool_manager=tool_manager, **kwargs)
        
        # 组合 RAG 服务
        self.rag_service = rag_service or OperatorRAGService()
    
    # ---------- 工厂方法 ----------
    @classmethod
    def create(
        cls,
        tool_manager: Optional[ToolManager] = None,
        rag_service: Optional[OperatorRAGService] = None,
        **kwargs,
    ):
        return cls(tool_manager=tool_manager, rag_service=rag_service, **kwargs)

    # ---------- 基本配置 ----------
    @property
    def role_name(self) -> str:
        return "operator_qa"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_operator_qa"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_operator_qa"

    # ---------- Prompt 参数 ----------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据前置工具结果构造 prompt 参数
        
        RAG 检索已改为后置工具，由 LLM 自主决定是否调用。
        对话历史由 BaseAgent 的 AdvancedMessageHistory 自动管理。
        
        Args:
            pre_tool_results: 前置工具执行结果
            
        Returns:
            prompt 参数字典
        """
        return {
            "user_query": pre_tool_results.get("user_query", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认前置工具结果"""
        return {
            "user_query": "",
        }

    # ---------- 结果写回 ----------
    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将推理结果写回 MainState"""
        # 存储 QA 结果
        state.operator_qa_result = result
        super().update_state_result(state, result, pre_tool_results)
    
    # ---------- RAG 相关方法（供外部调用） ----------
    def search_operators(self, query: str, top_k: int = 5) -> List[str]:
        """检索相关算子"""
        return self.rag_service.search(query, top_k=top_k)
    
    def get_operator_info(self, names: List[str]) -> str:
        """获取算子信息"""
        return self.rag_service.get_operator_info(names)
    
    def get_operator_source(self, name: str) -> str:
        """获取算子源码"""
        return self.rag_service.get_operator_source(name)
    
    def get_operator_params(self, name: str) -> Dict[str, Any]:
        """获取算子参数"""
        return self.rag_service.get_operator_params(name)



# ==============================================================================
# Helper APIs
# ==============================================================================
async def operator_qa(
    state: MainState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    rag_service: Optional[OperatorRAGService] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    use_agent: bool = False,
    **kwargs,
) -> MainState:
    """
    OperatorQA 的异步入口
    
    Args:
        state: 主状态对象
        model_name: 模型名称
        tool_manager: 工具管理器实例
        rag_service: RAG 服务实例
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        tool_mode: 工具调用模式
        react_mode: 是否启用 ReAct 推理模式
        react_max_retries: ReAct 模式下最大重试次数
        parser_type: 解析器类型
        parser_config: 解析器配置字典
        use_agent: 是否使用 agent 模式
        **kwargs: 其他传递给 execute 的参数
        
    Returns:
        更新后的 MainState 对象
    """
    agent = OperatorQAAgent(
        tool_manager=tool_manager,
        rag_service=rag_service,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_operator_qa_agent(
    tool_manager: Optional[ToolManager] = None,
    rag_service: Optional[OperatorRAGService] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    tool_mode: str = "auto",
    react_mode: bool = False,
    react_max_retries: int = 3,
    parser_type: str = "json",
    parser_config: Optional[Dict[str, Any]] = None,
    message_history: Optional["AdvancedMessageHistory"] = None,
    **kwargs,
) -> OperatorQAAgent:
    """
    创建 OperatorQAAgent 实例
    
    Args:
        tool_manager: 工具管理器
        rag_service: RAG 服务实例
        model_name: 模型名称
        temperature: 采样温度
        max_tokens: 最大 token 数
        tool_mode: 工具模式
        react_mode: ReAct 模式
        react_max_retries: 最大重试次数
        parser_type: 解析器类型
        parser_config: 解析器配置
        **kwargs: 其他参数
        
    Returns:
        OperatorQAAgent 实例
    """
    return OperatorQAAgent.create(
        tool_manager=tool_manager,
        rag_service=rag_service,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode=tool_mode,
        react_mode=react_mode,
        react_max_retries=react_max_retries,
        parser_type=parser_type,
        parser_config=parser_config,
        message_history=message_history,
        **kwargs,
    )
