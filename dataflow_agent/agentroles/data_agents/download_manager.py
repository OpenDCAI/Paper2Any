from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from langgraph.graph import StateGraph

from dataflow_agent.logger import get_logger
from dataflow_agent.state import WebCrawlState

log = get_logger(__name__)


@dataclass
class DownloadContext:
    task_objective: str
    search_keywords: str | List[str]
    hf_keywords: List[str]
    playwright_context: any


@dataclass
class DownloadOutcome:
    success: bool = False
    abort_due_to_size: bool = False


class DownloadManager:
    """
    负责 orchestrate 下载流程（HuggingFace → Kaggle），使用 LangGraph 串联。
    """

    def __init__(
        self,
        *,
        hf_manager,
        kaggle_manager,
        hf_decision_agent,
        kaggle_decision_agent,
        dataset_detail_reader,
        logger,
        max_dataset_size: Optional[int],
        download_dir: str,
    ):
        self.hf_manager = hf_manager
        self.kaggle_manager = kaggle_manager
        self.hf_decision_agent = hf_decision_agent
        self.kaggle_decision_agent = kaggle_decision_agent
        self.dataset_detail_reader = dataset_detail_reader
        self.logger = logger
        self.max_dataset_size = max_dataset_size
        self.download_dir = download_dir

    async def execute(
        self,
        state: WebCrawlState,
        *,
        context,
        task_objective: str,
        search_keywords: str | List[str],
        hf_keywords: List[str],
    ) -> DownloadOutcome:
        ctx = DownloadContext(
            task_objective=task_objective,
            search_keywords=search_keywords,
            hf_keywords=hf_keywords,
            playwright_context=context,
        )
        outcome = DownloadOutcome()
        graph = self._build_graph(ctx, outcome)
        result = await graph.ainvoke(state)
        return outcome if isinstance(result, WebCrawlState) else outcome

    def _build_graph(self, ctx: DownloadContext, outcome: DownloadOutcome):
        sg = StateGraph(WebCrawlState)

        async def hf_attempt(state: WebCrawlState) -> WebCrawlState:
            await self._try_huggingface(state, ctx, outcome)
            return state

        async def kaggle_attempt(state: WebCrawlState) -> WebCrawlState:
            if outcome.success or outcome.abort_due_to_size:
                return state
            await self._try_kaggle(state, ctx, outcome)
            return state

        def hf_condition(state: WebCrawlState):
            return "__end__" if state.download_successful_for_current_task else "kaggle"

        def kaggle_condition(state: WebCrawlState):
            return "__end__"

        sg.add_node("huggingface", hf_attempt)
        sg.add_node("kaggle", kaggle_attempt)
        sg.add_conditional_edges("huggingface", hf_condition)
        sg.add_conditional_edges("kaggle", kaggle_condition)
        sg.set_entry_point("huggingface")
        return sg.compile()

    async def _try_huggingface(
        self, state: WebCrawlState, ctx: DownloadContext, outcome: DownloadOutcome
    ) -> None:
        if not ctx.hf_keywords:
            log.info("[DownloadManager] 未提供 HuggingFace 关键词，跳过 HF 下载。")
            return
        try:
            search_results = await self.hf_manager.search_datasets(
                ctx.hf_keywords, max_results=5
            )
        except Exception as exc:
            log.info(f"[DownloadManager] HuggingFace 搜索失败: {exc}")
            return

        selected_id = await self.hf_decision_agent.execute(
            search_results,
            ctx.task_objective,
            self.logger,
            message="",
            max_dataset_size=self.max_dataset_size,
        )
        if not selected_id:
            log.info("[DownloadManager] HuggingFace 未选择合适数据集。")
            return

        dataset_info = {}
        for res_list in search_results.values():
            for dataset in res_list:
                if dataset.get("id") == selected_id:
                    dataset_info = dataset
                    break
            if dataset_info:
                break

        try:
            detail_result = await self.dataset_detail_reader.execute(
                dataset_id=selected_id,
                dataset_type="huggingface",
                dataset_info=dataset_info,
                logger=self.logger,
                max_dataset_size=self.max_dataset_size,
            )
        except Exception as exc:
            log.info(f"[DownloadManager] 详情读取失败: {exc}")
            detail_result = None

        if (
            self.max_dataset_size
            and detail_result
            and not detail_result.get("meets_size_limit", True)
        ):
            log.info(
                f"[DownloadManager] HuggingFace 数据集 {selected_id} 超过大小限制，终止该任务。"
            )
            outcome.abort_due_to_size = True
            return

        hf_dir = os.path.join(self.download_dir, "hf_datasets")
        save_path = await self.hf_manager.download_dataset(selected_id, hf_dir)
        if save_path:
            state.crawled_data.append(
                {
                    "source_url": f"https://huggingface.co/datasets/{selected_id}",
                    "local_path": save_path,
                    "type": "huggingface_dataset",
                    "dataset_id": selected_id,
                    "dataset_info": dataset_info,
                    "detail_analysis": detail_result,
                }
            )
            log.info(f"[DownloadManager] HuggingFace 数据集下载成功: {selected_id}")
            state.download_successful_for_current_task = True
            outcome.success = True
        else:
            log.info(f"[DownloadManager] HuggingFace 数据集下载失败: {selected_id}")

    async def _try_kaggle(
        self, state: WebCrawlState, ctx: DownloadContext, outcome: DownloadOutcome
    ) -> None:
        keywords = ctx.hf_keywords or (
            [ctx.task_objective]
            if isinstance(ctx.task_objective, str)
            else [str(ctx.task_objective)]
        )

        try:
            search_results = await self.kaggle_manager.search_datasets(
                keywords, max_results=5
            )
        except Exception as exc:
            log.info(f"[DownloadManager] Kaggle 搜索失败: {exc}")
            return

        if not search_results:
            log.info("[DownloadManager] Kaggle 未返回候选数据集。")
            return

        selected_id = await self.kaggle_decision_agent.execute(
            search_results,
            ctx.task_objective,
            self.logger,
            message="",
            max_dataset_size=self.max_dataset_size,
        )
        if not selected_id:
            log.info("[DownloadManager] Kaggle 未选择合适数据集。")
            return

        dataset_info = {}
        for res_list in search_results.values():
            for dataset in res_list:
                if dataset.get("id") == selected_id:
                    dataset_info = dataset
                    break
            if dataset_info:
                break

        try:
            detail_result = await self.dataset_detail_reader.execute(
                dataset_id=selected_id,
                dataset_type="kaggle",
                dataset_info=dataset_info,
                logger=self.logger,
                max_dataset_size=self.max_dataset_size,
            )
        except Exception as exc:
            log.info(f"[DownloadManager] Kaggle 数据集详情读取失败: {exc}")
            detail_result = None

        if (
            self.max_dataset_size
            and detail_result
            and not detail_result.get("meets_size_limit", True)
        ):
            log.info(
                f"[DownloadManager] Kaggle 数据集 {selected_id} 超过大小限制，终止该任务。"
            )
            outcome.abort_due_to_size = True
            return

        try:
            download_page = await ctx.playwright_context.new_page()
        except Exception as exc:
            log.info(f"[DownloadManager] 创建 Playwright 页面失败: {exc}")
            return

        try:
            save_dir = os.path.join(self.download_dir, "kaggle_datasets")
            saved_path = await self.kaggle_manager.try_download(
                download_page, selected_id, save_dir
            )
        finally:
            await download_page.close()

        if saved_path:
            state.crawled_data.append(
                {
                    "source_url": f"https://www.kaggle.com/datasets/{selected_id}",
                    "local_path": saved_path,
                    "type": "kaggle_dataset",
                    "dataset_id": selected_id,
                    "dataset_info": dataset_info,
                    "detail_analysis": detail_result,
                }
            )
            log.info(f"[DownloadManager] Kaggle 数据集下载成功: {selected_id}")
            state.download_successful_for_current_task = True
            outcome.success = True
        else:
            log.info(f"[DownloadManager] Kaggle 数据集下载失败: {selected_id}")

