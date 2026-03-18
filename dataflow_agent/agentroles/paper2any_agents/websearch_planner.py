"""
Websearch Planner Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
负责 Web 搜索知识入库流程中的「Planner / 全知指挥官」角色。

职责：
1. 监控研究进度：检查 Web Researcher 的产出，并将其汇总到 Raw Data Store。
2. 维护任务队列：从 research_routes 中弹出已完成的任务，并设定下一个 current_task。
3. 动态规划（可选）：后续可扩展利用 LLM 分析当前进度，动态增删任务。
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

from dataflow_agent.state import MainState, WebsearchKnowledgeState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

# OpenAI 依赖
from openai import AsyncOpenAI
import httpx

log = get_logger(__name__)


@register("websearch_planner")
class WebsearchPlannerAgent(BaseAgent):
    """
    Websearch Planner Agent（完整实现版）
    """

    def __init__(
        self,
        tool_manager: Optional[ToolManager] = None,
        llm_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(tool_manager=tool_manager, **kwargs)
        
        self.llm_config = llm_config or {
            "base_url": os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            "api_key": os.getenv("DF_API_KEY", "sk-xxx"),
            "model": os.getenv("THIRD_PARTY_MODEL", "gpt-4o"),
        }
        
        api_key = self.llm_config.get("api_key")
        base_url = self.llm_config.get("base_url")
        model = self.llm_config.get("model")
        
        http_client = httpx.AsyncClient(trust_env=False)
        self.llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        self.llm_model = model
        log.info(f"🔗 LLM Initialized with Base URL: {base_url}")

    @classmethod
    def create(
        cls,
        tool_manager: Optional[ToolManager] = None,
        **kwargs,
    ) -> "WebsearchPlannerAgent":
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "websearch_planner"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_websearch_planner"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_websearch_planner"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造 Planner 所需的 prompt 参数。
        """
        return {
            "pre_tool_results": pre_tool_results,
        }

    async def run(self, state: MainState, **kwargs) -> Dict[str, Any]:
        """
        执行规划任务
        
        1. 检查 researcher 结果，同步数据到 raw_data_store
        2. 维护 research_routes 队列
        3. 设定 state.current_task 供 researcher 使用
        """
        log.info(f"[{self.role_name}] 开始执行规划逻辑...")
        
        if not isinstance(state, WebsearchKnowledgeState):
            log.error("State 类型错误，必须为 WebsearchKnowledgeState")
            return {"status": "failed", "reason": "Invalid state type"}

        # 1. 检查 Web Researcher 的执行产出
        researcher_results = state.agent_results.get("websearch_researcher", {})
        
        # 如果有成功的 researcher 结果且未被 Planner 处理过
        if (researcher_results and 
            researcher_results.get("result", {}).get("status") == "success" and 
            not researcher_results.get("planner_processed", False)):
            
            log.info("发现未处理的 Web Researcher 产出，正在同步到 Raw Data Store...")
            res_data = researcher_results["result"]
            storage_path_str = res_data.get("storage_path")
            
            if storage_path_str:
                storage_path = Path(storage_path_str)
                dom_snapshots_dir = storage_path / "dom_snapshots"
                
                # --- 同步 DOM 快照（HTML）---
                if dom_snapshots_dir.exists():
                    new_records_count = 0
                    for dom_file in dom_snapshots_dir.glob("*.html"):
                        # 检查是否已存在（根据路径判断）
                        if any(r.get("dom_filepath") == str(dom_file) for r in state.raw_data_store):
                            continue
                            
                        record = {
                            "url": "extracted_from_research", # 理想情况下 researcher 应该返回具体 URL
                            "timestamp": datetime.now().isoformat(),
                            "dom_filepath": str(dom_file),
                            "source": "websearch_researcher",
                            "task": getattr(state, "current_task", "Unknown Task")
                        }
                        state.raw_data_store.append(record)
                        new_records_count += 1
                    log.info(f"成功同步 {new_records_count} 条 DOM 快照记录到 Raw Data Store")
                
                # --- 同步 final_resources 中的 PDF 文件 ---
                final_resources_dir = storage_path / "final_resources"
                if final_resources_dir.exists():
                    pdf_records_count = 0
                    for pdf_file in final_resources_dir.glob("*.pdf"):
                        # 检查是否已存在（根据 pdf_filepath 判断）
                        if any(r.get("pdf_filepath") == str(pdf_file) for r in state.raw_data_store):
                            continue
                        
                        record = {
                            "url": f"pdf://{pdf_file.name}",
                            "timestamp": datetime.now().isoformat(),
                            "pdf_filepath": str(pdf_file),
                            "source": "websearch_researcher",
                            "task": getattr(state, "current_task", "Unknown Task"),
                            "type": "pdf"
                        }
                        state.raw_data_store.append(record)
                        pdf_records_count += 1
                    if pdf_records_count > 0:
                        log.info(f"成功同步 {pdf_records_count} 条 PDF 记录到 Raw Data Store")

            # 任务队列维护：既然 Researcher 成功完成了当前任务，就将其从队列中弹出
            if state.research_routes:
                done_task = state.research_routes.pop(0)
                log.info(f"已完成并移除研究子任务: {done_task}")
            
            # 标记该 researcher 结果已处理
            researcher_results["planner_processed"] = True

        # 2. 设定下一个任务
        if state.research_routes:
            # 设定下一个要执行的任务给 researcher 节点看
            state["current_task"] = state.research_routes[0]
            log.info(f"设定下一个 current_task: {state['current_task']}")
        else:
            # 如果没有待执行任务，清理 current_task
            if hasattr(state, "current_task"):
                state["current_task"] = None
            log.info("研究任务队列已空")

        # 3. 构建返回结果
        result_payload = {
            "status": "success",
            "remaining_tasks_count": len(state.research_routes),
            "raw_data_store_count": len(state.raw_data_store),
            "next_task": state.get("current_task")
        }
        
        self.update_state_result(state, result_payload, {})
        return result_payload

    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        将结果写回到 state.agent_results。
        """
        if getattr(state, "agent_results", None) is not None:
            state.agent_results[self.role_name] = {
                "result": result,
                "pre_tool_results": pre_tool_results,
            }
        log.debug(f"[{self.role_name}] result written to state.agent_results.")
        super().update_state_result(state, result, pre_tool_results)


def create_websearch_planner_agent(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> WebsearchPlannerAgent:
    """
    便捷创建函数。
    """
    return WebsearchPlannerAgent.create(tool_manager=tool_manager, **kwargs)


