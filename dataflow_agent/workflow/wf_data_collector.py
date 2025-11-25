# dataflow_agent/workflow/wf_data_collector.py

from __future__ import annotations
import os
import asyncio
import json
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright
from langgraph.graph import StateGraph, START, END

from dataflow_agent.state import WebCrawlState
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.webresearch import (
    ToolManager,
    URLFilter,
    WebPageReader,
    Executor,
    Supervisor,
    WebResearchAgent,
    QueryGeneratorAgent,
)
from dataflow_agent.agentroles.download_manager import DownloadManager
from dataflow_agent.agentroles.data_collector_agents import (
    DownloadMethodDecisionAgent,
    HuggingFaceDecisionAgent,
    KaggleDecisionAgent,
    DatasetDetailReaderAgent,
    TaskDecomposer,
)
from dataflow_agent.toolkits.datatool import (
    LogManager,
    RAGManager,
    HuggingFaceDatasetManager,
    KaggleDatasetManager,
    PaddleDatasetManager,
)
from dataflow_agent.workflow.registry import register

log = get_logger(__name__)

class WebCrawlOrchestrator:
    def __init__(self, 
                 api_base_url: str,
                 api_key: str,
                 model_name: str = "gpt-4o",
                 download_dir: str = "./downloaded_data5", 
                 dataset_size_categories: List[str] = None,
                 max_crawl_cycles_per_task: int = 5,
                 max_crawl_cycles_for_research: int = 15,
                 search_engine: str = "tavily",
                 use_jina_reader: bool=True,
                 enable_rag: bool = True,
                 concurrent_pages: int = 5,
                 disable_cache: bool = True,
                 temp_base_dir: str = None,
                 max_dataset_size: int = None,
                 max_download_subtasks: int = None,
                 rag_api_base_url: str | None = None,
                 rag_api_key: str | None = None,
                 rag_embed_model: str | None = None,
                 tavily_api_key: str | None = None):
        """
        初始化 WebCrawlOrchestrator
        
        Args:
            disable_cache: 如果为 True，将完全禁用 HuggingFace 和 Kaggle 的缓存，
                           使用临时目录并在下载后自动清理，避免占用任何磁盘空间。
                           也可以通过环境变量 DF_DISABLE_CACHE=true 来启用。
            max_download_subtasks: 下载子任务执行数量的上限；传入 None 表示不限制。
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        # if isinstance(download_dir, str) and download_dir.startswith("\\\\?\\"):
        #     cleaned_download_dir = download_dir[4:]
        #     log.info(f"[Orchestrator] 检测到下载目录存在 '\\\\?\\' 前缀，已自动移除: {cleaned_download_dir}")
        #     download_dir = cleaned_download_dir
        self.download_dir = download_dir
        self.max_crawl_cycles_per_task = max_crawl_cycles_per_task
        self.max_crawl_cycles_for_research = max_crawl_cycles_for_research
        self.search_engine = search_engine
        self.use_jina_reader = use_jina_reader
        self.rag_api_base_url = rag_api_base_url or api_base_url
        self.rag_api_key = rag_api_key or api_key
        self.rag_embed_model = rag_embed_model
        self.enable_rag = enable_rag and bool(self.rag_api_base_url) and bool(self.rag_api_key)
        self.rag_collection_name = "rag_collection"
        self.concurrent_pages = concurrent_pages
        self.max_dataset_size = max_dataset_size
        self.max_download_subtasks = max_download_subtasks
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key
        os.makedirs(self.download_dir, exist_ok=True)
        
        # 统一控制临时目录，避免写入系统 /tmp
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir or os.path.join(self.download_dir, ".tmp")
        os.makedirs(self.temp_base_dir, exist_ok=True)
        os.environ.setdefault("TMPDIR", self.temp_base_dir)

        # --- 缓存目录统一控制 ---
        cache_root = os.path.join(os.path.abspath(self.download_dir), ".cache")
        os.makedirs(cache_root, exist_ok=True)
        hf_cache_root = os.path.join(cache_root, "hf")
        os.makedirs(hf_cache_root, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache_root)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache_root, "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache_root, "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache_root, "transformers"))

        kaggle_cache_root = os.path.join(cache_root, "kaggle")
        os.makedirs(kaggle_cache_root, exist_ok=True)
        os.environ.setdefault("KAGGLE_HUB_CACHE", kaggle_cache_root)
        os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.join(kaggle_cache_root, "config"))
        os.makedirs(os.environ["KAGGLE_CONFIG_DIR"], exist_ok=True)

        # 检查是否禁用缓存（优先使用参数，其次使用环境变量）
        if not disable_cache:
            disable_cache = os.getenv("DF_DISABLE_CACHE", "false").lower() in ("true", "1", "yes")
        
        log.info(f"[Orchestrator] 配置:")
        log.info(f"  - 模型: {model_name}")
        log.info(f"  - 搜索引擎: {search_engine.upper()}")
        log.info(f"  - Jina Reader: {'启用' if use_jina_reader else '禁用'}")
        log.info(f"  - RAG 增强: {'启用' if enable_rag else '禁用'}")
        log.info(f"  - 并行页面数: {concurrent_pages}")
        log.info(f"  - 禁用缓存: {'是 (下载后自动清理临时文件)' if disable_cache else '否 (缓存将保存在项目目录)'}")
        
        self.task_decomposer = TaskDecomposer(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.url_filter = URLFilter(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.web_page_reader = WebPageReader(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.executor = Executor()
        self.supervisor = Supervisor(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.download_decision_agent = DownloadMethodDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.logger = LogManager()
        self.web_research_agent = WebResearchAgent(
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            concurrent_pages=self.concurrent_pages,
            web_page_reader=self.web_page_reader,
            executor=self.executor,
            supervisor=self.supervisor,
        )
        
        # *** 在 Orchestrator 中初始化 HF 工具和新 Agent ***
        # 如果未禁用缓存，设置缓存目录到 download_dir 下的 .cache 文件夹，避免占用系统盘
        cache_base_dir = None if disable_cache else os.path.join(os.path.abspath(self.download_dir), ".cache")
        
        self.hf_decision_agent = HuggingFaceDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.kaggle_decision_agent = KaggleDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.dataset_detail_reader = DatasetDetailReaderAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.hf_manager = HuggingFaceDatasetManager(max_retries=2, retry_delay=5, size_categories=dataset_size_categories, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # Kaggle / Paddle 管理器
        self.kaggle_manager = KaggleDatasetManager(search_engine=self.search_engine, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # self.paddle_manager = PaddleDatasetManager(search_engine=self.search_engine)  # 已注释：Paddle数据集获取功能
        
        # [--- QueryGenerator Agent 初始化 ---]
        self.query_generator = QueryGeneratorAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        
        # 下载管理器
        self.download_manager = DownloadManager(
            hf_manager=self.hf_manager,
            kaggle_manager=self.kaggle_manager,
            hf_decision_agent=self.hf_decision_agent,
            kaggle_decision_agent=self.kaggle_decision_agent,
            dataset_detail_reader=self.dataset_detail_reader,
            logger=self.logger,
            max_dataset_size=self.max_dataset_size,
            download_dir=self.download_dir,
        )
        
        # [--- RAG 管理器初始化 ---]
        self.rag_manager = None
        if self.enable_rag:
            log.info("[Orchestrator] 初始化 RAG 管理器...")
            try:
                self.rag_manager = RAGManager(
                    api_base_url=self.rag_api_base_url,
                    api_key=self.rag_api_key,
                    embed_model=self.rag_embed_model,
                    persist_directory=os.path.join(self.download_dir, "rag_db"),
                    collection_name=self.rag_collection_name
                )
            except ValueError as e:
                log.warning(f"[Orchestrator] 初始化 RAG 管理器失败: {e}")
                self.enable_rag = False
        else:
            if enable_rag and not self.enable_rag:
                log.warning("[Orchestrator] RAG 已禁用：缺少 RAG API 配置。")
        
    def _remaining_download_slots(self, state: WebCrawlState) -> Optional[int]:
        """计算剩余可执行的下载子任务数量。"""
        limit = state.max_download_subtasks
        if limit is None:
            return None
        remaining = limit - state.completed_download_tasks
        return remaining if remaining > 0 else 0

    def _apply_download_limit_to_state(self, state: WebCrawlState) -> None:
        """根据上限裁剪待执行的下载子任务。"""
        remaining = self._remaining_download_slots(state)
        if remaining is None:
            return

        downloads_removed = 0
        if remaining <= 0:
            downloads_removed = sum(1 for task in state.sub_tasks if task.get("type") == "download")
            if downloads_removed:
                log.info(f"[Orchestrator] 下载子任务上限已达，移除 {downloads_removed} 个待执行的下载子任务。")
            state.sub_tasks = [task for task in state.sub_tasks if task.get("type") != "download"]
            return

        filtered_tasks: List[Dict[str, Any]] = []
        kept_downloads = 0
        for task in state.sub_tasks:
            if task.get("type") == "download":
                if kept_downloads >= remaining:
                    downloads_removed += 1
                    continue
                kept_downloads += 1
            filtered_tasks.append(task)

        if downloads_removed:
            log.info(f"[Orchestrator] 应用下载子任务上限，裁剪 {downloads_removed} 个多余的下载子任务。")
        state.sub_tasks = filtered_tasks

    async def _process_single_url(self, page, url: str, state: WebCrawlState, 
                                   task_objective: str, task_type: str, 
                                   is_research_phase: bool) -> Dict[str, Any]:
        """
        处理单个URL的辅助方法
        """
        result = {
            'success': False,
            'url': url,
            'crawled_entry': None,
            'new_urls': [],
            'download_successful': False
        }
        
        try:
            # 读取网页并分析
            action_plan = await self.web_page_reader.execute(
                state, self.logger, page=page, url=url, 
                current_objective=task_objective,
                is_research=is_research_phase
            )
            
            # 创建临时state来执行操作
            temp_state = WebCrawlState(
                initial_request=state.initial_request,
                download_dir=state.download_dir,
                search_engine=state.search_engine,
                use_jina_reader=state.use_jina_reader,
                rag_manager=state.rag_manager,
                enable_rag=state.enable_rag,
                max_crawl_cycles_per_task=state.max_crawl_cycles_per_task,
                max_crawl_cycles_for_research=state.max_crawl_cycles_for_research
            )
            temp_state.crawled_data = []
            temp_state.url_queue = []
            temp_state.visited_urls = state.visited_urls.copy()
            temp_state.download_successful_for_current_task = False
            
            # 执行操作
            temp_state = await self.executor.execute(
                temp_state, action_plan, source_url=url, page=page, 
                current_task_type=task_type
            )
            
            # 收集结果
            result['success'] = True
            if temp_state.crawled_data:
                result['crawled_entry'] = temp_state.crawled_data[0] if len(temp_state.crawled_data) == 1 else {
                    'multiple_entries': temp_state.crawled_data
                }
            result['new_urls'] = temp_state.url_queue
            result['download_successful'] = temp_state.download_successful_for_current_task
            
        except Exception as e:
            log.info(f"[并行处理] 处理URL时出错 ({url}): {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    async def _process_urls_parallel(self, context, urls: List[str], state: WebCrawlState,
                                      task_objective: str, task_type: str, 
                                      is_research_phase: bool) -> List[Dict[str, Any]]:
        """
        并行处理多个URL
        """
        async def process_with_page(url: str) -> Dict[str, Any]:
            """为每个URL创建独立的page并处理"""
            page = await context.new_page()
            try:
                return await self._process_single_url(
                    page, url, state, task_objective, task_type, is_research_phase
                )
            finally:
                await page.close()
        
        # 并行处理URL
        tasks = [process_with_page(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.info(f"[并行处理] URL处理异常 ({urls[i]}): {result}")
                processed_results.append({
                    'success': False,
                    'url': urls[i],
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def run(self, initial_request: str):
        self.logger.new_run()
        log.info("启动多阶段数据爬取流程...")
        log.info(f"搜索引擎: {self.search_engine.upper()}")
        log.info(f"Jina Reader: {'启用' if self.use_jina_reader else '禁用'}")
        log.info(f"RAG 增强: {'启用' if self.rag_manager else '禁用'}")
        
        state = WebCrawlState(
            initial_request=initial_request, 
            download_dir=self.download_dir,
            search_engine=self.search_engine,  # [--- 传入搜索引擎选择 ---]
            use_jina_reader=self.use_jina_reader,  # [--- 传入 Jina Reader 选项 ---]
            rag_manager=self.rag_manager,  # [--- 传入 RAG 管理器 ---]
            enable_rag=self.enable_rag,  # [--- 传入 RAG 启用状态 ---]
            max_crawl_cycles_per_task=self.max_crawl_cycles_per_task,
            max_crawl_cycles_for_research=self.max_crawl_cycles_for_research,
            max_dataset_size=self.max_dataset_size,
            max_download_subtasks=self.max_download_subtasks
        )
        state.request.max_download_subtasks = self.max_download_subtasks
        self.logger.log_data("0_initial_request", {
            "request": initial_request,
            "search_engine": self.search_engine,
            "use_jina_reader": self.use_jina_reader,
            "enable_rag": self.enable_rag
        }, is_json=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            page = await context.new_page()

            try:
                state = await self.task_decomposer.execute(state, self.logger)

                if not state.sub_tasks:
                    log.info("未能生成任何子任务，流程终止。")
                    return

                while state.sub_tasks:
                    if not state.sub_tasks:
                        break
                    current_task = state.sub_tasks.pop(0)
                    task_type = current_task.get("type")
                    task_objective = current_task.get("objective")
                    search_keywords = current_task.get("search_keywords", task_objective)

                    log.info(f"\n\n{'='*50}\n开始执行子任务: [{task_type.upper()}] - {task_objective}\n{'='*50}")
                    state.reset_for_new_task()
                    if task_type == "download" and state.research_summary.get("new_sub_tasks"):
                        new_tasks = state.research_summary.pop("new_sub_tasks", [])
                        if new_tasks:
                            log.info(f"根据研究总结，将泛化下载任务替换为 {len(new_tasks)} 个具体任务。")
                            remaining_after_fallback: List[Dict[str, Any]] = []
                            removed = False
                            for task in state.sub_tasks:
                                if not removed and task.get("type") == "download":
                                    removed = True
                                    continue
                                remaining_after_fallback.append(task)
                            state.sub_tasks = new_tasks + remaining_after_fallback
                            self._apply_download_limit_to_state(state)
                            continue
                        else:
                            log.info(f"研究阶段未发现具体下载目标，执行默认的下载任务作为兜底。") 

                    if task_type == "research":
                        await self.web_research_agent.execute(
                            state,
                            context=context,
                            logger=self.logger,
                            objective=task_objective,
                            search_keywords=search_keywords,
                        )
                        new_download_tasks = state.research_summary.get("new_sub_tasks", []) or []
                        if new_download_tasks:
                            log.info(f"[Research] 添加 {len(new_download_tasks)} 个新下载子任务，并移除默认兜底任务。")
                            remaining_after_fallback: List[Dict[str, Any]] = []
                            removed = False
                            for task in state.sub_tasks:
                                if not removed and task.get("type") == "download":
                                    removed = True
                                    continue
                                remaining_after_fallback.append(task)
                            state.sub_tasks = new_download_tasks + remaining_after_fallback
                            state.research_summary["new_sub_tasks"] = []
                        current_task["status"] = "completed"
                        self._apply_download_limit_to_state(state)
                        state.completed_sub_tasks.append(current_task)
                        log.info(f"子任务 [{task_type.upper()}] 完成。状态: {current_task.get('status', 'completed')}")
                        continue

                    download_success = False
                    abort_download_task = False

                    if task_type == "download":
                        log.info("正在获取 HuggingFace 搜索关键词（默认优先HF）...")
                        # 传入用户的原始需求和当前子任务信息
                        user_original_request = getattr(state, "user_message", None) or state.initial_request
                        decision = await self.download_decision_agent.execute(
                            state, self.logger, user_original_request, task_objective, search_keywords
                        )
                        hf_keywords = decision.get("keywords_for_hf", []) or []
                        if not hf_keywords:
                            if isinstance(search_keywords, (list, tuple)):
                                hf_keywords = [
                                    kw
                                    for kw in search_keywords
                                    if isinstance(kw, str) and kw.strip()
                                ] or [task_objective]
                            else:
                                if isinstance(search_keywords, str) and search_keywords.strip():
                                    hf_keywords = [search_keywords]
                                else:
                                    hf_keywords = [task_objective]

                        state.download_successful_for_current_task = False
                        manager_outcome = await self.download_manager.execute(
                            state,
                            context=context,
                            task_objective=task_objective,
                            search_keywords=search_keywords,
                            hf_keywords=hf_keywords,
                        )
                        download_success = manager_outcome.success
                        abort_download_task = manager_outcome.abort_due_to_size
                        state.download_successful_for_current_task = download_success

                    # --- Web Crawl Logic ---
                    if (
                        task_type == "download"
                        and not download_success
                        and not abort_download_task
                    ): 
                        log.info(f"使用关键词进行搜索: '{search_keywords}'")
                        query_kw = (
                            ", ".join(search_keywords)
                            if isinstance(search_keywords, (list, tuple))
                            else search_keywords
                        )
                        search_results = await ToolManager.search_web(
                            query_kw, search_engine=state.search_engine
                        )
                        state.search_results_text = search_results
                        self.logger.log_data(
                            f"2_search_results_{task_objective.replace(' ', '_')}",
                            search_results,
                        )

                        state = await self.url_filter.execute(
                            state, logger=self.logger, is_research=False
                        )

                        max_cycles = state.max_crawl_cycles_per_task

                        while state.url_queue and state.current_cycle < max_cycles:
                            batch_urls: List[str] = []
                            while (
                                len(batch_urls) < self.concurrent_pages and state.url_queue
                            ):
                                current_url = state.url_queue.pop(0)
                                if current_url not in state.visited_urls:
                                    batch_urls.append(current_url)
                                    state.visited_urls.add(current_url)

                            if not batch_urls:
                                break

                            log.info(f"\n[并行处理] 开始处理 {len(batch_urls)} 个网页...")

                            results = await self._process_urls_parallel(
                                context,
                                batch_urls,
                                state,
                                task_objective,
                                task_type,
                                False,
                            )
                            for result in results:
                                if result["success"]:
                                    crawled_entry = result.get("crawled_entry")
                                    if crawled_entry:
                                        if "multiple_entries" in crawled_entry:
                                            state.crawled_data.extend(
                                                crawled_entry["multiple_entries"]
                                            )
                                        else:
                                            state.crawled_data.append(crawled_entry)

                                    for new_url in result.get("new_urls", []):
                                        if (
                                            new_url not in state.visited_urls
                                            and new_url not in state.url_queue
                                        ):
                                            state.url_queue.append(new_url)
                                    if result.get("download_successful"):
                                        state.download_successful_for_current_task = True

                            state = await self.supervisor.execute(
                                state,
                                self.logger,
                                current_objective=task_objective,
                                is_research=False,
                            )
                            if state.download_successful_for_current_task:
                                log.info(
                                    f"子任务 '{task_objective}' 的下载目标已完成，提前结束爬取循环。"
                                )
                                break
                            log.info(
                                f"[并行处理] 批次完成，剩余 {len(state.url_queue)} 个URL待处理"
                            )
                    
                    if task_type == "download":
                        if state.download_successful_for_current_task:
                            current_task['status'] = 'completed_successfully'
                        else:
                            current_task['status'] = 'failed_due_to_size_limit' if abort_download_task else 'failed_to_download'
                        state.completed_download_tasks += 1
                        self._apply_download_limit_to_state(state)
                    else:
                        current_task['status'] = 'completed'

                    state.completed_sub_tasks.append(current_task)
                    log.info(f"子任务 [{task_type.upper()}] 完成。状态: {current_task.get('status', 'N/A')}")

            finally:
                await browser.close()

        log.info("\n任务执行完毕!")
        downloaded_files = [d for d in state.crawled_data if d.get('type') == 'file' or d.get('type') == 'huggingface_dataset']
        log.info(f"最终收集到的文件 ({len(downloaded_files)} 个): {json.dumps(downloaded_files, indent=2, ensure_ascii=False)}")
        
        final_state_log = {}
        for k, v in state.__dict__.items():
            if isinstance(v, set):
                final_state_log[k] = list(v)
            elif k == 'rag_manager':
                if v:
                    final_state_log['rag_statistics'] = v.get_statistics()
                else:
                    final_state_log['rag_statistics'] = None
            elif hasattr(v, '__dict__') and not isinstance(v, (str, int, float, bool, list, dict, tuple)):
                final_state_log[k] = f"<{v.__class__.__name__} object>"
            else:
                final_state_log[k] = v
        
        self.logger.log_data("7_final_state", final_state_log, is_json=True)
        return state

    async def run_with_langgraph(self, initial_request: str):
        """
        使用 LangGraph 实现的网页爬取流程（功能与 run() 完全一致）
        """
        self.logger.new_run()
        log.info("启动多阶段数据爬取流程（LangGraph版本）...")
        log.info(f"搜索引擎: {self.search_engine.upper()}")
        log.info(f"Jina Reader: {'启用' if self.use_jina_reader else '禁用'}")
        log.info(f"RAG 增强: {'启用' if self.rag_manager else '禁用'}")
        
        state = WebCrawlState(
            initial_request=initial_request, 
            download_dir=self.download_dir,
            search_engine=self.search_engine,
            use_jina_reader=self.use_jina_reader,
            rag_manager=self.rag_manager,
            enable_rag=self.enable_rag,
            max_crawl_cycles_per_task=self.max_crawl_cycles_per_task,
            max_crawl_cycles_for_research=self.max_crawl_cycles_for_research,
            max_dataset_size=self.max_dataset_size,
            max_download_subtasks=self.max_download_subtasks
        )
        state.request.max_download_subtasks = self.max_download_subtasks
        self.logger.log_data("0_initial_request", {
            "request": initial_request,
            "search_engine": self.search_engine,
            "use_jina_reader": self.use_jina_reader,
            "enable_rag": self.enable_rag
        }, is_json=True)

        # 使用闭包变量存储 playwright context 和当前任务信息（避免 LangGraph 传递 state 时丢失）
        playwright_context: Optional[Any] = None
        playwright_browser: Optional[Any] = None
        current_task_info: Optional[Dict[str, Any]] = None
        skip_routing: bool = False

        async def task_decomposition_node(state: WebCrawlState) -> WebCrawlState:
            """任务分解节点"""
            log.info("=== [LangGraph] 任务分解节点 ===")
            state = await self.task_decomposer.execute(state, self.logger)
            if not state.sub_tasks:
                log.info("未能生成任何子任务，流程终止。")
                state.is_finished = True
            return state

        async def check_has_tasks(state: WebCrawlState) -> str:
            """检查是否还有任务需要处理"""
            nonlocal skip_routing
            
            if state.is_finished or not state.sub_tasks:
                return "end"
            # 如果设置了跳过路由标记，直接回到 process_task（用于处理任务替换的情况）
            if skip_routing:
                skip_routing = False
                return "process_task"
            return "process_task"

        async def process_task_node(state: WebCrawlState) -> WebCrawlState:
            """处理单个子任务节点"""
            nonlocal current_task_info
            
            if not state.sub_tasks:
                state.is_finished = True
                current_task_info = None
                return state

            current_task = state.sub_tasks.pop(0)
            task_type = current_task.get("type")
            task_objective = current_task.get("objective")
            search_keywords = current_task.get("search_keywords", task_objective)

            log.info(f"\n\n{'='*50}\n开始执行子任务: [{task_type.upper()}] - {task_objective}\n{'='*50}")
            state.reset_for_new_task()

            # 处理 research 任务生成的新下载任务
            if task_type == "download" and state.research_summary.get("new_sub_tasks"):
                new_tasks = state.research_summary.pop("new_sub_tasks", [])
                if new_tasks:
                    log.info(f"根据研究总结，将泛化下载任务替换为 {len(new_tasks)} 个具体任务。")
                    remaining_after_fallback: List[Dict[str, Any]] = []
                    removed = False
                    for task in state.sub_tasks:
                        if not removed and task.get("type") == "download":
                            removed = True
                            continue
                        remaining_after_fallback.append(task)
                    state.sub_tasks = new_tasks + remaining_after_fallback
                    self._apply_download_limit_to_state(state)
                    # 清除当前任务信息，因为当前任务被跳过
                    current_task_info = None
                    skip_routing = True
                    return state
                else:
                    log.info(f"研究阶段未发现具体下载目标，执行默认的下载任务作为兜底。")

            # 存储当前任务信息到闭包变量（这样不会被 LangGraph 丢失）
            current_task_info = {
                "task": current_task,
                "type": task_type,
                "objective": task_objective,
                "search_keywords": search_keywords
            }

            return state

        async def route_task_type(state: WebCrawlState) -> str:
            """根据任务类型路由"""
            nonlocal current_task_info
            
            if current_task_info is None:
                log.warning("[LangGraph] route_task_type: current_task_info 未设置，返回 complete_task")
                return "complete_task"
            
            task_type = current_task_info["type"]
            if task_type == "research":
                return "research_task"
            elif task_type == "download":
                return "download_task"
            else:
                return "complete_task"

        async def research_task_node(state: WebCrawlState) -> WebCrawlState:
            """Research 任务处理节点"""
            nonlocal current_task_info, playwright_context
            
            log.info("=== [LangGraph] Research 任务节点 ===")
            
            if current_task_info is None:
                log.error("[LangGraph] research_task_node: current_task_info 未设置，跳过")
                return state
            
            if playwright_context is None:
                log.error("[LangGraph] research_task_node: playwright_context 未设置，跳过")
                return state
            
            context = playwright_context
            await self.web_research_agent.execute(
                state,
                context=context,
                logger=self.logger,
                objective=current_task_info["objective"],
                search_keywords=current_task_info["search_keywords"],
            )
            new_download_tasks = state.research_summary.get("new_sub_tasks", []) or []
            if new_download_tasks:
                log.info(f"[Research] 添加 {len(new_download_tasks)} 个新下载子任务，并移除默认兜底任务。")
                remaining_after_fallback: List[Dict[str, Any]] = []
                removed = False
                for task in state.sub_tasks:
                    if not removed and task.get("type") == "download":
                        removed = True
                        continue
                    remaining_after_fallback.append(task)
                state.sub_tasks = new_download_tasks + remaining_after_fallback
                state.research_summary["new_sub_tasks"] = []
            
            # 更新任务状态并添加到完成列表
            if current_task_info:
                current_task_info["task"]["status"] = "completed"
                self._apply_download_limit_to_state(state)
                state.completed_sub_tasks.append(current_task_info["task"])
                log.info(f"子任务 [RESEARCH] 完成。状态: {current_task_info['task'].get('status', 'completed')}")
            return state

        async def download_task_node(state: WebCrawlState) -> WebCrawlState:
            """Download 任务处理节点"""
            nonlocal current_task_info, playwright_context
            
            log.info("=== [LangGraph] Download 任务节点 ===")
            
            if current_task_info is None:
                log.error("[LangGraph] download_task_node: current_task_info 未设置，跳过")
                return state
            
            if playwright_context is None:
                log.error("[LangGraph] download_task_node: playwright_context 未设置，跳过")
                return state
            
            context = playwright_context
            task_objective = current_task_info["objective"]
            search_keywords = current_task_info["search_keywords"]

            log.info("正在获取 HuggingFace 搜索关键词（默认优先HF）...")
            user_original_request = getattr(state, "user_message", None) or state.initial_request
            decision = await self.download_decision_agent.execute(
                state, self.logger, user_original_request, task_objective, search_keywords
            )
            hf_keywords = decision.get("keywords_for_hf", []) or []
            if not hf_keywords:
                if isinstance(search_keywords, (list, tuple)):
                    hf_keywords = [
                        kw
                        for kw in search_keywords
                        if isinstance(kw, str) and kw.strip()
                    ] or [task_objective]
                else:
                    if isinstance(search_keywords, str) and search_keywords.strip():
                        hf_keywords = [search_keywords]
                    else:
                        hf_keywords = [task_objective]

            state.download_successful_for_current_task = False
            manager_outcome = await self.download_manager.execute(
                state,
                context=context,
                task_objective=task_objective,
                search_keywords=search_keywords,
                hf_keywords=hf_keywords,
            )
            download_success = manager_outcome.success
            abort_download_task = manager_outcome.abort_due_to_size
            state.download_successful_for_current_task = download_success

            # 如果下载失败且未因大小限制中止，进行网页爬取
            if not download_success and not abort_download_task:
                log.info(f"使用关键词进行搜索: '{search_keywords}'")
                query_kw = (
                    ", ".join(search_keywords)
                    if isinstance(search_keywords, (list, tuple))
                    else search_keywords
                )
                search_results = await ToolManager.search_web(
                    query_kw, search_engine=state.search_engine
                )
                state.search_results_text = search_results
                self.logger.log_data(
                    f"2_search_results_{task_objective.replace(' ', '_')}",
                    search_results,
                )

                state = await self.url_filter.execute(
                    state, logger=self.logger, is_research=False
                )

                max_cycles = state.max_crawl_cycles_per_task

                while state.url_queue and state.current_cycle < max_cycles:
                    batch_urls: List[str] = []
                    while (
                        len(batch_urls) < self.concurrent_pages and state.url_queue
                    ):
                        current_url = state.url_queue.pop(0)
                        if current_url not in state.visited_urls:
                            batch_urls.append(current_url)
                            state.visited_urls.add(current_url)

                    if not batch_urls:
                        break

                    log.info(f"\n[并行处理] 开始处理 {len(batch_urls)} 个网页...")

                    results = await self._process_urls_parallel(
                        context,
                        batch_urls,
                        state,
                        task_objective,
                        getattr(state, '_task_type', 'download'),  # 使用 getattr 确保安全
                        False,
                    )
                    for result in results:
                        if result["success"]:
                            crawled_entry = result.get("crawled_entry")
                            if crawled_entry:
                                if "multiple_entries" in crawled_entry:
                                    state.crawled_data.extend(
                                        crawled_entry["multiple_entries"]
                                    )
                                else:
                                    state.crawled_data.append(crawled_entry)

                            for new_url in result.get("new_urls", []):
                                if (
                                    new_url not in state.visited_urls
                                    and new_url not in state.url_queue
                                ):
                                    state.url_queue.append(new_url)
                            if result.get("download_successful"):
                                state.download_successful_for_current_task = True

                    state = await self.supervisor.execute(
                        state,
                        self.logger,
                        current_objective=task_objective,
                        is_research=False,
                    )
                    if state.download_successful_for_current_task:
                        log.info(
                            f"子任务 '{task_objective}' 的下载目标已完成，提前结束爬取循环。"
                        )
                        break
                    log.info(
                        f"[并行处理] 批次完成，剩余 {len(state.url_queue)} 个URL待处理"
                    )

            # 更新任务状态
            if current_task_info:
                if state.download_successful_for_current_task:
                    current_task_info["task"]['status'] = 'completed_successfully'
                else:
                    current_task_info["task"]['status'] = 'failed_due_to_size_limit' if abort_download_task else 'failed_to_download'
                state.completed_download_tasks += 1
                self._apply_download_limit_to_state(state)
                state.completed_sub_tasks.append(current_task_info["task"])
                log.info(f"子任务 [DOWNLOAD] 完成。状态: {current_task_info['task'].get('status', 'N/A')}")
            return state


        async def complete_task_node(state: WebCrawlState) -> WebCrawlState:
            """完成任务节点（处理未知任务类型）"""
            nonlocal current_task_info
            
            if current_task_info is None:
                log.warning("[LangGraph] complete_task_node: current_task_info 未设置，无法完成任务")
                return state
            
            current_task_info["task"]['status'] = 'completed'
            state.completed_sub_tasks.append(current_task_info["task"])
            task_type = current_task_info.get("type", 'UNKNOWN')
            log.info(f"子任务 [{task_type.upper()}] 完成。状态: {current_task_info['task'].get('status', 'N/A')}")
            return state

        async def finalize_node(state: WebCrawlState) -> WebCrawlState:
            """最终化节点"""
            log.info("\n任务执行完毕!")
            downloaded_files = [d for d in state.crawled_data if d.get('type') == 'file' or d.get('type') == 'huggingface_dataset']
            log.info(f"最终收集到的文件 ({len(downloaded_files)} 个): {json.dumps(downloaded_files, indent=2, ensure_ascii=False)}")
            
            final_state_log = {}
            for k, v in state.__dict__.items():
                if k.startswith('_'):
                    continue  # 跳过临时字段
                if isinstance(v, set):
                    final_state_log[k] = list(v)
                elif k == 'rag_manager':
                    if v:
                        final_state_log['rag_statistics'] = v.get_statistics()
                    else:
                        final_state_log['rag_statistics'] = None
                elif hasattr(v, '__dict__') and not isinstance(v, (str, int, float, bool, list, dict, tuple)):
                    final_state_log[k] = f"<{v.__class__.__name__} object>"
                else:
                    final_state_log[k] = v
            
            self.logger.log_data("7_final_state", final_state_log, is_json=True)
            return state

        # 构建 LangGraph 工作流
        graph_builder = StateGraph(WebCrawlState)
        
        # 添加节点
        graph_builder.add_node("task_decomposition", task_decomposition_node)
        graph_builder.add_node("process_task", process_task_node)
        graph_builder.add_node("research_task", research_task_node)
        graph_builder.add_node("download_task", download_task_node)
        graph_builder.add_node("complete_task", complete_task_node)
        graph_builder.add_node("finalize", finalize_node)

        # 添加边
        graph_builder.add_edge(START, "task_decomposition")
        graph_builder.add_conditional_edges(
            "task_decomposition",
            check_has_tasks,
            {
                "end": "finalize",
                "process_task": "process_task"
            }
        )
        async def route_after_process_task(state: WebCrawlState) -> str:
            """在 process_task 之后的路由函数"""
            nonlocal current_task_info, skip_routing
            
            # 如果设置了跳过路由标记，直接回到 process_task
            if skip_routing:
                skip_routing = False
                return "process_task"
            # 否则根据任务类型路由
            return await route_task_type(state)

        graph_builder.add_conditional_edges(
            "process_task",
            route_after_process_task,
            {
                "research_task": "research_task",
                "download_task": "download_task",
                "complete_task": "complete_task",
                "process_task": "process_task"  # 允许直接回到 process_task
            }
        )
        graph_builder.add_conditional_edges(
            "research_task",
            check_has_tasks,
            {
                "end": "finalize",
                "process_task": "process_task"
            }
        )
        graph_builder.add_conditional_edges(
            "download_task",
            check_has_tasks,
            {
                "end": "finalize",
                "process_task": "process_task"
            }
        )
        graph_builder.add_conditional_edges(
            "complete_task",
            check_has_tasks,
            {
                "end": "finalize",
                "process_task": "process_task"
            }
        )
        graph_builder.add_edge("finalize", END)

        # 编译图
        graph = graph_builder.compile()

        # 执行工作流（在 playwright context 中）
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            page = await context.new_page()
            
            try:
                # 将 playwright context 和 browser 存储到闭包变量中
                playwright_context = context
                playwright_browser = browser
                
                result = await graph.ainvoke(state)
                
                # 确保返回的是 WebCrawlState 对象，而不是字典
                if isinstance(result, WebCrawlState):
                    state = result
                elif isinstance(result, dict):
                    # 如果返回的是字典，更新原始 state 对象
                    log.warning("[LangGraph] graph.ainvoke 返回了字典，正在转换为 WebCrawlState")
                    # 从字典中更新 state 的字段
                    for key, value in result.items():
                        if key.startswith('_'):
                            continue  # 跳过临时字段
                        if hasattr(state, key):
                            # 特殊处理集合类型
                            if key == 'visited_urls' and isinstance(value, list):
                                setattr(state, key, set(value))
                            else:
                                setattr(state, key, value)
                else:
                    log.warning(f"[LangGraph] graph.ainvoke 返回了未知类型: {type(result)}，使用原始 state")
            finally:
                await browser.close()
                # 清理闭包变量
                playwright_context = None
                playwright_browser = None
                current_task_info = None
                skip_routing = False

        return state