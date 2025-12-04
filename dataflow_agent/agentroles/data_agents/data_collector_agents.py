# dataflow_agent/agentroles/data_collector_agents.py

from __future__ import annotations
import json
from typing import Any, Dict, Optional

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.state import WebCrawlState
from dataflow_agent.toolkits.datatool.log_manager import LogManager, log_agent_input_output
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class DownloadMethodDecisionAgent(BaseAgent):
    """下载方法决策器 - 决定使用哪种方法下载数据"""
    async def execute(self, state: WebCrawlState, logger: LogManager, user_original_request: str, current_task_objective: str, search_keywords: str) -> Dict[str, Any]:
        log.info("\n--- 下载方法决策器 ---")
        # 记录输入
        inputs = {
            "user_original_request": user_original_request,
            "current_task_objective": current_task_objective,
            "search_keywords": search_keywords,
            "state_initial_request": state.initial_request
        }
        log_agent_input_output("DownloadMethodDecisionAgent", inputs, logger=logger)
        
        system_prompt = self.prompt_gen.render("system_prompt_for_download_method_decision")
        human_prompt = self.prompt_gen.render("task_prompt_for_download_method_decision", 
                                              user_original_request=user_original_request,
                                              current_task_objective=current_task_objective,
                                              keywords=search_keywords)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("download_method_decision_raw_response", response.content)
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            decision = json.loads(clean_response)
            log.info(f"下载方法决策: {decision.get('method')} - {decision.get('reasoning')}")
            logger.log_data("download_method_decision_parsed", decision, is_json=True)
            # 记录输出
            log_agent_input_output("DownloadMethodDecisionAgent", inputs, decision, logger=logger)
            return decision
        except Exception as e:
            log.error(f"解析下载方法决策时出错: {e}\n原始响应: {response.content}")
            fallback_result = {"method": "web_crawl", "reasoning": "解析失败，使用默认的web爬取方法", "keywords_for_hf": [], "fallback_method": "huggingface"}
            log_agent_input_output("DownloadMethodDecisionAgent", inputs, fallback_result, logger=logger)
            return fallback_result

class HuggingFaceDecisionAgent(BaseAgent):
    
    async def execute(self, search_results: Dict[str, List[Dict]], objective: str, logger: LogManager, message: str = "", max_dataset_size: int = None) -> str | None:
        log.info("\n--- HuggingFace 决策器 ---")
        # 记录输入
        inputs = {
            "objective": objective,
            "message": message,
            "max_dataset_size": max_dataset_size,
            "search_results_count": sum(len(v) for v in search_results.values()) if search_results else 0
        }
        log_agent_input_output("HuggingFaceDecisionAgent", inputs, logger=logger)
        
        if not search_results or all(not v for v in search_results.values()):
            log.info("[HuggingFace Decision] 搜索结果为空，无法决策。")
            return None

        system_prompt = self.prompt_gen.render("system_prompt_for_huggingface_decision")
        human_prompt = self.prompt_gen.render("task_prompt_for_huggingface_decision",
                                              objective=objective,
                                              message=message,
                                              search_results=json.dumps(search_results, indent=2, ensure_ascii=False))

        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        log_name = f"hf_decision_{objective.replace(' ', '_')}"
        logger.log_data(f"{log_name}_raw_response", response.content)

        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            decision = json.loads(clean_response)
            selected_id = decision.get("selected_dataset_id")
            
            if selected_id:
                log.info(f"[HuggingFace Decision] 决策: {selected_id}. 原因: {decision.get('reasoning')}")
                log_agent_input_output("HuggingFaceDecisionAgent", inputs, selected_id, logger=logger)
                return selected_id
            else:
                log.info(f"[HuggingFace Decision] 决策: 无合适的数据集。原因: {decision.get('reasoning')}")
                log_agent_input_output("HuggingFaceDecisionAgent", inputs, None, logger=logger)
                return None
        except Exception as e:
            log.error(f"[HuggingFace Decision] 解析决策时出错: {e}\n原始响应: {response.content}")
            log_agent_input_output("HuggingFaceDecisionAgent", inputs, None, logger=logger)
            return None

class KaggleDecisionAgent(BaseAgent):
    """Kaggle数据集决策器 - 使用模型选择最合适的Kaggle数据集"""
    
    async def execute(self, search_results: Dict[str, List[Dict]], objective: str, logger: LogManager, message: str = "", max_dataset_size: int = None) -> str | None:
        log.info("\n--- Kaggle 决策器 ---")
        
        if not search_results or all(not v for v in search_results.values()):
            log.info("[Kaggle Decision] 搜索结果为空，无法决策。")
            return None

        system_prompt = self.prompt_gen.render("system_prompt_for_kaggle_decision")
        human_prompt = self.prompt_gen.render("task_prompt_for_kaggle_decision",
                                              objective=objective,
                                              message=message,
                                              max_dataset_size=max_dataset_size if max_dataset_size else "None",
                                              search_results=json.dumps(search_results, indent=2, ensure_ascii=False, default=lambda o: getattr(o, "name", str(o))))

        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        log_name = f"kaggle_decision_{objective.replace(' ', '_')}"
        logger.log_data(f"{log_name}_raw_response", response.content)

        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            decision = json.loads(clean_response)
            selected_id = decision.get("selected_dataset_id")
            
            if selected_id:
                log.info(f"[Kaggle Decision] 决策: {selected_id}. 原因: {decision.get('reasoning')}")
                return selected_id
            else:
                log.info(f"[Kaggle Decision] 决策: 无合适的数据集。原因: {decision.get('reasoning')}")
                return None
        except Exception as e:
            log.info(f"[Kaggle Decision] 解析决策时出错: {e}\n原始响应: {response.content}")
            return None

class DatasetDetailReaderAgent(BaseAgent):
    """数据集详情读取器 - 读取并分析数据集的详细信息，特别是HF数据集"""
    
    async def execute(self, dataset_id: str, dataset_type: str, dataset_info: Dict[str, Any], logger: LogManager, max_dataset_size: int = None) -> Dict[str, Any]:
        log.info(f"\n--- 数据集详情读取器 ({dataset_type}) ---")
        log.info(f"正在分析数据集: {dataset_id}")
        
        system_prompt = self.prompt_gen.render("system_prompt_for_dataset_detail_reader")
        human_prompt = self.prompt_gen.render("task_prompt_for_dataset_detail_reader",
                                              dataset_id=dataset_id,
                                              dataset_type=dataset_type,
                                              max_dataset_size=max_dataset_size if max_dataset_size else "None",
                                              dataset_info=json.dumps(dataset_info, indent=2, ensure_ascii=False))

        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        log_name = f"dataset_detail_{dataset_type}_{dataset_id.replace('/', '_')}"
        logger.log_data(f"{log_name}_raw_response", response.content)

        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            detail = json.loads(clean_response)
            log.info(f"[Dataset Detail Reader] 分析完成: {detail.get('summary', 'N/A')}")
            logger.log_data(f"{log_name}_parsed", detail, is_json=True)
            return detail
        except Exception as e:
            log.info(f"[Dataset Detail Reader] 解析响应时出错: {e}\n原始响应: {response.content}")
            return {
                "dataset_id": dataset_id,
                "size_bytes": None,
                "meets_size_limit": True,  # 默认假设满足限制
                "summary": f"解析失败: {e}"
            }

class TaskDecomposer(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager) -> WebCrawlState:
        log.info("\n--- decomposer ---")
        system_prompt = self.prompt_gen.render("system_prompt_for_task_decomposer")
        human_prompt = self.prompt_gen.render("task_prompt_for_task_decomposer", 
                                              request=state.initial_request)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("1_decomposer_raw_response", response.content)
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            plan = json.loads(clean_response)
            state.sub_tasks = plan.get("sub_tasks", [])
            # 保存任务分解器提供的清晰用户需求描述
            state.user_message = plan.get("message", state.initial_request)
            log.info(f"任务计划已生成，包含 {len(state.sub_tasks)} 个步骤。")
            logger.log_data("1_decomposer_parsed_plan", plan, is_json=True)
        except Exception as e:
            log.info(f"解析任务计划时出错: {e}\n原始响应: {response.content}")
        return state