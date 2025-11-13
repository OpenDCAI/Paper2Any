from __future__ import annotations
import os

# 在导入 huggingface_hub / datasets 之前，优先设置 HF_ENDPOINT，确保所有内部请求走镜像
# 支持通过环境变量 DF_HF_ENDPOINT 覆盖（例如 https://hf-mirror.com）
_df_hf_endpoint = os.getenv("DF_HF_ENDPOINT")
if _df_hf_endpoint and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = _df_hf_endpoint
import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import shutil
from playwright.async_api import async_playwright, Page
import tenacity
import requests.exceptions

import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataflow_agent.state import WebCrawlState
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.webresearch import (
    BaseAgent,
    ToolManager,
    URLFilter,
    WebPageReader,
    Executor,
    Supervisor,
    WebResearchAgent,
    QueryGeneratorAgent,
)
from dataflow_agent.agentroles.download_manager import DownloadManager
from dataflow_agent.toolkits.webatool import check_if_download_link, download_file

log = get_logger(__name__)

def log_agent_input_output(agent_name: str, inputs: Dict[str, Any], outputs: Any = None, logger: Any = None):
    """
    记录Agent的输入和输出
    
    Args:
        agent_name: Agent名称
        inputs: 输入参数字典
        outputs: 输出结果
        logger: LogManager实例（可选）
    """
    # 记录到标准日志
    log.info(f"[Agent Input] {agent_name}: {json.dumps(inputs, indent=2, ensure_ascii=False, default=str)}")
    if outputs is not None:
        # 对于大型输出，只记录摘要
        if isinstance(outputs, (dict, list)) and len(str(outputs)) > 1000:
            output_summary = f"<{type(outputs).__name__} with {len(outputs) if isinstance(outputs, (list, dict)) else 'N/A'} items>"
            log.info(f"[Agent Output] {agent_name}: {output_summary}")
        else:
            log.info(f"[Agent Output] {agent_name}: {json.dumps(outputs, indent=2, ensure_ascii=False, default=str)}")
    
    # 如果提供了LogManager，也记录到文件
    if logger:
        logger.log_data(f"{agent_name}_input", inputs, is_json=True)
        if outputs is not None:
            logger.log_data(f"{agent_name}_output", outputs, is_json=True)

class LogManager:
    """负责为每次运行创建日志目录并保存每一步的数据。"""
    def __init__(self, base_dir="logs"):
        self.run_dir = ""
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir

    def new_run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        log.info(f"日志将保存在: {self.run_dir}")

    def log_data(self, step_name: str, data: Any, is_json: bool = False):
        if not self.run_dir:
            log.warning("LogManager尚未初始化，无法记录日志。")
            return
        safe_step_name = re.sub(r'[\\/*?:"<>|]', "", step_name)
        safe_step_name = re.sub(r'\s+', '_', safe_step_name).strip("_") or "log"
        extension = ".json" if is_json else ".txt"
        filename = os.path.join(self.run_dir, f"{safe_step_name}{extension}")
        content = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            log.error(f"写入日志文件 {filename} 时出错: {e}")

# --- RAG管理器 ---
class RAGManager:

    def __init__(
        self,
        api_base_url: str | None,
        api_key: str | None,
        *,
        embed_model: str | None = None,
        persist_directory: str = "./rag_db",
        reset: bool = False,
        collection_name: str = "rag_collection",
    ):
        resolved_api_base = api_base_url or os.getenv("RAG_API_URL")
        resolved_api_key = api_key or os.getenv("RAG_API_KEY")
        resolved_embed_model = embed_model or os.getenv("RAG_EMB_MODEL") or "text-embedding-3-large"

        if not resolved_api_base or not resolved_api_key:
            raise ValueError("RAG 初始化失败：缺少 API 基础地址或 API Key。请在调用阶段传入或设置环境变量。")

        log.info(f"[RAG] 初始化 RAG 管理器，存储目录: {persist_directory}，模型: {resolved_embed_model}")
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=resolved_api_base,
            openai_api_key=resolved_api_key,
            model=resolved_embed_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.collection_name = collection_name
        self.document_count = 0
        # 仅在明确要求时重置
        if reset and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        # 预先初始化一个持久化的空集合，便于后续追加与检索
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            log.error(f"[RAG] 初始化向量存储失败: {e}")
            self.vectorstore = None
        # 去重集合，避免重复块污染召回
        self._seen_hashes = set()
    
    async def add_webpage_content(self, url: str, text_content: str, metadata: Dict[str, Any] = None):

        if not text_content or len(text_content.strip()) < 50:
            log.info(f"[RAG] 跳过内容过短的网页: {url}")
            return
        try:
            log.info(f"[RAG] 正在添加网页内容: {url} (长度: {len(text_content)} 字符)")
            # 基础清洗
            cleaned = re.sub(r"\s+", " ", text_content).strip()
            chunks = self.text_splitter.split_text(cleaned)
            log.info(f"[RAG] 文本已分成 {len(chunks)} 个块")
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk or len(chunk.strip()) < 80:
                    continue
                # 内容去重
                import hashlib
                digest = hashlib.sha1(chunk.strip().encode("utf-8")).hexdigest()
                if digest in self._seen_hashes:
                    continue
                self._seen_hashes.add(digest)
                doc_metadata = {
                    "source_url": url,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    doc_metadata.update(metadata)
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
            if not documents:
                log.warning(f"[RAG] 清洗/去重后无有效文档块可添加: {url}")
                return
            if self.vectorstore is None:
                # 兜底：如果之前初始化失败，则在首次添加时创建
                self.vectorstore = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                await asyncio.to_thread(self.vectorstore.add_documents, documents)
            # 立即持久化，保证下次运行可用
            try:
                await asyncio.to_thread(self.vectorstore.persist)
            except Exception as e:
                log.error(f"[RAG] 持久化失败: {e}")
            self.document_count += len(documents)
            log.info(f"[RAG] 成功添加 {len(documents)} 个文档块，总计: {self.document_count} 块")
        except Exception as e:
            log.error(f"[RAG] 添加网页内容时出错 ({url}): {e}")
    
    async def get_context_for_single_query(self, query: str, max_chars: int = 18000) -> str:
        """获取单个查询的上下文"""
        if self.vectorstore is None:
            log.warning("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            log.info(f"[RAG] 检索查询: {query[:50]}...")
            mmr_docs = await asyncio.to_thread(
                self.vectorstore.max_marginal_relevance_search,
                query,
                k=15,
                fetch_k=60,
                lambda_mult=0.5
            )
            
            # 构建上下文
            context_parts = []
            total_chars = 0
            seen_urls = set()
            for doc in mmr_docs:
                source_url = doc.metadata.get("source_url", "unknown")
                content = doc.page_content
                if source_url not in seen_urls:
                    header = f"\n--- Source: {source_url} ---\n"
                    context_parts.append(header)
                    total_chars += len(header)
                    seen_urls.add(source_url)
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        context_parts.append(content[:remaining] + "...[truncated]")
                    break
                context_parts.append(content + "\n")
                total_chars += len(content) + 1
            
            context = "".join(context_parts)
            log.info(f"[RAG] 查询检索完成: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            log.error(f"[RAG] 检索查询 '{query}' 时出错: {e}")
            return ""
    
    async def get_context_for_analysis(self, objective: str, max_chars: int = 20000, queries: List[str] = None) -> str:
        """获取用于分析的上下文，支持多个查询（合并结果）"""
        if self.vectorstore is None:
            log.warning("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            # 如果没有提供查询，使用原始objective
            if queries is None or len(queries) == 0:
                queries = [objective]
            
            log.info(f"[RAG] 使用 {len(queries)} 个查询进行检索")
            all_docs = []
            seen_doc_hashes = set()
            
            # 对每个查询进行检索
            for i, query in enumerate(queries, 1):
                log.info(f"[RAG] 查询 {i}/{len(queries)}: {query[:50]}...")
                try:
                    mmr_docs = await asyncio.to_thread(
                        self.vectorstore.max_marginal_relevance_search,
                        query,
                        k=15,  # 每个查询检索更少的文档，但多个查询会合并
                        fetch_k=60,
                        lambda_mult=0.5
                    )
                    # 去重：使用内容hash避免重复
                    import hashlib
                    for doc in mmr_docs:
                        doc_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
                        if doc_hash not in seen_doc_hashes:
                            seen_doc_hashes.add(doc_hash)
                            all_docs.append(doc)
                except Exception as e:
                    log.error(f"[RAG] 查询 '{query}' 检索失败: {e}")
            
            log.info(f"[RAG] 合并后共获得 {len(all_docs)} 个去重文档")
            
            # 构建上下文
            context_parts = []
            total_chars = 0
            seen_urls = set()
            for doc in all_docs:
                source_url = doc.metadata.get("source_url", "unknown")
                content = doc.page_content
                if source_url not in seen_urls:
                    header = f"\n--- Source: {source_url} ---\n"
                    context_parts.append(header)
                    total_chars += len(header)
                    seen_urls.add(source_url)
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        context_parts.append(content[:remaining] + "...[truncated]")
                    break
                context_parts.append(content + "\n")
                total_chars += len(content) + 1
            
            context = "".join(context_parts)
            log.info(f"[RAG] 生成分析上下文: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            log.error(f"[RAG] 检索内容时出错: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:

        return {
            "total_documents": self.document_count,
            "vectorstore_ready": self.vectorstore is not None
        }

class HuggingFaceDatasetManager:

    
    def __init__(self, max_retries: int = 2, retry_delay: int = 5, size_categories: List[str] = None, cache_dir: str = None, disable_cache: bool = False, temp_base_dir: str = None):
        self.hf_endpoint = 'https://hf-mirror.com'
        self.max_retries = max_retries
        self.retry_delay = retry_delay # seconds
        self.size_categories = size_categories  # e.g., ["n<1K", "1K<n<10K", "10K<n<100K"]
        self.disable_cache = disable_cache
        os.environ['HF_ENDPOINT'] = self.hf_endpoint
        
        # 允许通过 DF_TEMP_DIR 或传参指定临时目录基准，避免写入系统 /tmp
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir
        if self.temp_base_dir:
            os.makedirs(self.temp_base_dir, exist_ok=True)

        # 如果禁用缓存，使用可控的临时目录并在下载后清理
        if disable_cache:
            import tempfile
            temp_cache = tempfile.mkdtemp(prefix="hf_cache_", dir=self.temp_base_dir)
            os.environ['HF_HOME'] = temp_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(temp_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(temp_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(temp_cache, "transformers")
            self._temp_cache_dir = temp_cache
            log.info(f"[HuggingFace] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
        elif cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            # 设置 HuggingFace 相关的缓存环境变量
            hf_cache = os.path.join(cache_dir, "hf_cache")
            os.makedirs(hf_cache, exist_ok=True)
            os.environ['HF_HOME'] = hf_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(hf_cache, "transformers")
            self._temp_cache_dir = None
            log.info(f"[HuggingFace] 缓存目录已设置为: {hf_cache} (避免占用系统盘)")
        else:
            # 如果未指定，使用默认的项目目录
            default_cache = os.path.join(os.getcwd(), ".cache", "hf")
            os.makedirs(default_cache, exist_ok=True)
            os.environ['HF_HOME'] = default_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(default_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(default_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(default_cache, "transformers")
            self._temp_cache_dir = None
            log.info(f"[HuggingFace] 使用默认缓存目录: {default_cache}")
        
        log.info(f"[HuggingFace] 初始化，最大重试次数: {self.max_retries}, 延迟: {self.retry_delay}s (线性增长), 数据集大小类别: {self.size_categories if self.size_categories else '不限制'}")

        # 延迟导入 HuggingFace 依赖，确保上方已正确设置缓存相关环境变量
        from huggingface_hub import HfApi, snapshot_download
        from datasets import get_dataset_config_names

        self.hf_api = HfApi(endpoint=self.hf_endpoint)
        self._snapshot_download = snapshot_download
        self._get_dataset_config_names = get_dataset_config_names

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:

        if isinstance(e, (
            ConnectionResetError, 
            ConnectionRefusedError, 
            requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError
        )):
            return True

        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [502, 503, 504]:
            return True
            
        error_str = str(e).lower()
        if any(err_msg in error_str for err_msg in [
            "10054", 
            "connection reset by peer",
            "timeout", 
            "serviceunavailable" 
        ]):
             return True
             
        return False

    async def _retry_async_thread(self, func, *args, **kwargs):

        def log_retry_attempt(retry_state: tenacity.RetryCallState):
            attempt = retry_state.attempt_number
            exception = retry_state.outcome.exception()
            log.info(f"[HuggingFace] 发生可重试网络错误 (Attempt {attempt}/{self.max_retries}): {exception}")

        retryer = tenacity.AsyncRetrying(

            stop=tenacity.stop_after_attempt(self.max_retries),
        
            wait=tenacity.wait_incrementing(start=self.retry_delay, increment=self.retry_delay),
            retry=tenacity.retry_if_exception(self._is_retryable_error),
            before_sleep=log_retry_attempt,
            reraise=True  
        )

        async def func_to_retry():
            return await asyncio.to_thread(func, *args, **kwargs)

        try:
            return await retryer(func_to_retry)
        
        except tenacity.RetryError as e:
            log.info(f"[HuggingFace] 所有 {self.max_retries} 次重试均失败。")
            if e.last_attempt and e.last_attempt.failed:
                raise e.last_attempt.exception
            else:
                raise Exception(f"HuggingFace操作失败 ({func.__name__})，但未捕获到特定异常。")
        
        except Exception as e:
            log.info(f"[HuggingFace] 发生不可重试错误: {e}")
            raise e 

    
    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> Dict[str, List[Dict]]:
        results = {}
        
        for keyword in keywords:
            try:
                log.info(f"[HuggingFace] 搜索关键词: '{keyword}'")

                datasets = await self._retry_async_thread(
                    self.hf_api.list_datasets, 
                    search=keyword, 
                    limit=max_results,
                    # size_categories=self.size_categories
                )
                
                results[keyword] = []
                for dataset in datasets:
                    # 尝试获取数据集大小信息
                    dataset_size = None
                    try:
                        # 尝试从dataset对象获取大小信息
                        if hasattr(dataset, 'siblings'):
                            # 计算所有文件的总大小
                            total_size = 0
                            for sibling in getattr(dataset, 'siblings', []):
                                if hasattr(sibling, 'size') and sibling.size:
                                    total_size += sibling.size
                            if total_size > 0:
                                dataset_size = total_size
                        # 如果siblings中没有，尝试从其他属性获取
                        if not dataset_size and hasattr(dataset, 'size'):
                            dataset_size = getattr(dataset, 'size', None)
                    except Exception as e:
                        # 如果获取大小失败，继续使用None
                        pass
                    
                    results[keyword].append({
                        "id": dataset.id,
                        "title": getattr(dataset, 'title', dataset.id),
                        "description": getattr(dataset, 'description', ''),
                        "downloads": getattr(dataset, 'downloads', 0),
                        "tags": getattr(dataset, 'tags', []),
                        "size": dataset_size  # 数据集大小（字节），可能为None
                    })
                
                log.info(f"[HuggingFace] 找到 {len(results[keyword])} 个数据集")
                
            except Exception as e:
                log.info(f"[HuggingFace] 搜索关键词 '{keyword}' 时出错 (经过重试后): {e}")
                results[keyword] = []
        
        return results
    
    # -----------------------------------------------------------------
    # vvvvvvvvvvvv   修改后的 download_dataset 方法   vvvvvvvvvvvv
    # -----------------------------------------------------------------
    async def download_dataset(self, dataset_id: str, save_dir: str) -> str | None:
        try:
            log.info(f"[HuggingFace] 开始下载数据集: {dataset_id}")
            dataset_dir = os.path.join(save_dir, dataset_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            config_to_load = None
            try:
                log.info(f"[HuggingFace] 正在检查 {dataset_id} 的配置...")
                
                configs = await self._retry_async_thread(
                    self._get_dataset_config_names,
                    path=dataset_id,
                    # base_url=self.hf_endpoint,  # 显式传入镜像端点
                    # token=self.hf_api.token 
                )
                
                if configs:
                    config_to_load = configs[0] 
                    log.info(f"[HuggingFace] 数据集 {dataset_id} 有 {len(configs)} 个配置. 自动选择第一个: {config_to_load}")
                else:
                    log.info(f"[HuggingFace] 数据集 {dataset_id} 没有特定的配置.")
            
            except Exception as e:
                log.info(f"[HuggingFace] 检查配置时出错 (将跳过配置检查，直接下载): {e}")
                config_to_load = None
            
            # --- 核心修改：使用 snapshot_download 替换 load_dataset ---
            log.info(f"[HuggingFace] 开始下载 {dataset_id} 的所有文件...")
            
            returned_path = await self._retry_async_thread(
                self._snapshot_download, 
                repo_id=dataset_id,
                local_dir=dataset_dir,
                repo_type="dataset",             # 明确告知是数据集
                force_download=True,           # 相当于 download_mode="force_redownload"
                # local_dir_use_symlinks=False,  # 推荐设置，避免Windows或跨设备问题
                endpoint=self.hf_endpoint      # 显式传入镜像端点，确保重试时使用镜像
                # token=self.hf_api.token      # 如果需要私有库，可以传入
            )
            # --- 修改结束 ---
            
            # 如果禁用了缓存，下载完成后清理临时缓存目录
            if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
                try:
                    if os.path.exists(self._temp_cache_dir):
                        shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                        log.info(f"[HuggingFace] 已清理临时缓存目录: {self._temp_cache_dir}")
                except Exception as e:
                    log.info(f"[HuggingFace] 清理临时缓存目录时出错: {e} (可忽略)")
            
            config_str = f"(配置: {config_to_load})" if config_to_load else "(默认配置)"
            log.info(f"[HuggingFace] 数据集 {dataset_id} {config_str} *文件*下载成功，保存至 {returned_path}")
            return returned_path
            
        except Exception as e:
            log.info(f"[HuggingFace] 下载数据集 {dataset_id} 失败 (经过重试后): {e}")
            return None

class KaggleDatasetManager:

    def __init__(self, search_engine: str = "tavily", cache_dir: str = None, disable_cache: bool = False, temp_base_dir: str = None):
        self.search_engine = search_engine
        self.api = None
        self.disable_cache = disable_cache
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir
        if self.temp_base_dir:
            os.makedirs(self.temp_base_dir, exist_ok=True)
        
        # 如果禁用缓存，使用可控的临时目录并在下载后清理
        if disable_cache:
            import tempfile
            temp_cache = tempfile.mkdtemp(prefix="kaggle_cache_", dir=self.temp_base_dir)
            os.environ['KAGGLE_HUB_CACHE'] = temp_cache
            kaggle_config = os.path.join(temp_cache, "config")
            os.makedirs(kaggle_config, exist_ok=True)
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = temp_cache
            log.info(f"[Kaggle] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
        elif cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            kaggle_cache = os.path.join(cache_dir, "kaggle_cache")
            os.makedirs(kaggle_cache, exist_ok=True)
            # 设置 kagglehub 缓存目录（通过环境变量）
            os.environ['KAGGLE_HUB_CACHE'] = kaggle_cache
            # Kaggle API 默认下载到指定目录，但可能还会有元数据缓存
            # 设置 KAGGLE_CONFIG_DIR 可以控制配置和缓存位置
            kaggle_config = os.path.join(kaggle_cache, "config")
            os.makedirs(kaggle_config, exist_ok=True)
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = None
            log.info(f"[Kaggle] 缓存目录已设置为: {kaggle_cache} (避免占用系统盘)")
        else:
            # 如果未指定，使用默认的项目目录
            default_cache = os.path.join(os.getcwd(), ".cache", "kaggle")
            os.makedirs(default_cache, exist_ok=True)
            os.environ['KAGGLE_HUB_CACHE'] = default_cache
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                kaggle_config = os.path.join(default_cache, "config")
                os.makedirs(kaggle_config, exist_ok=True)
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = None
            log.info(f"[Kaggle] 使用默认缓存目录: {default_cache}")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            self.api = KaggleApi()
            self.api.authenticate()
            log.info("[Kaggle] 已使用 KaggleApi 进行认证。")
        except Exception as e:
            log.info(f"[Kaggle] KaggleApi 初始化/认证失败: {e}. 请配置 ~/.kaggle/kaggle.json 或设置 KAGGLE_USERNAME/KAGGLE_KEY。将无法使用 Kaggle API。")

    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> Dict[str, List[Dict]]:
        """搜索Kaggle数据集，返回详细信息（类似HF格式）"""
        if not self.api:
            log.info("[Kaggle] 未初始化 KaggleApi，跳过 Kaggle 搜索。")
            return {}
        results = {}
        try:
            # KaggleApi 不支持并发调用，这里串行合并结果
            for kw in keywords:
                try:
                    # 设置 60 秒超时
                    items = await asyncio.wait_for(
                        asyncio.to_thread(self.api.dataset_list, search=kw),
                        timeout=60.0
                    )
                    results[kw] = []
                    for it in (items or [])[:max_results]:
                        # it.ref 格式如 owner/slug
                        ref = getattr(it, 'ref', None) or f"{getattr(it, 'ownerSlug', '')}/{getattr(it, 'datasetSlug', '')}"
                        if ref and '/' in ref:
                            total_size = getattr(it, 'totalBytes', 0) or getattr(it, 'total_bytes', 0)
                            if not total_size and self.api:
                                try:
                                    files_resp = await asyncio.wait_for(
                                        asyncio.to_thread(self.api.dataset_list_files, ref),
                                        timeout=30.0
                                    )
                                    if files_resp:
                                        files = getattr(files_resp, 'files', None) or []
                                        size_acc = 0
                                        for f in files:
                                            size_acc += getattr(f, 'totalBytes', 0) or getattr(f, 'fileSize', 0) or getattr(f, 'size', 0)
                                        if size_acc > 0:
                                            total_size = size_acc
                                except asyncio.TimeoutError:
                                    log.info(f"[Kaggle] 获取文件大小超时: {ref}")
                                except Exception as size_err:
                                    log.info(f"[Kaggle] 获取 {ref} 文件大小失败: {size_err}")
                            # 提取详细信息，并将复杂对象转换为字符串
                            raw_tags = getattr(it, 'tags', [])
                            try:
                                tags_list = [getattr(t, 'name', str(t)) for t in (raw_tags or [])]
                            except Exception:
                                tags_list = []
                            dataset_info = {
                                "id": ref,
                                "title": getattr(it, 'title', ref),
                                "description": getattr(it, 'description', ''),
                                "downloads": getattr(it, 'usabilityRating', 0),
                                "size": total_size,
                                "tags": tags_list,
                                "owner": getattr(it, 'ownerSlug', ''),
                                "url": f"https://www.kaggle.com/datasets/{ref}"
                            }
                            results[kw].append(dataset_info)
                except asyncio.TimeoutError:
                    log.info(f"[Kaggle] 搜索 '{kw}' 超时（60秒），跳过")
                    results[kw] = []
                except Exception as e:
                    log.info(f"[Kaggle] 搜索 '{kw}' 出错: {e}")
                    results[kw] = []
        except Exception as e:
            log.info(f"[Kaggle] 搜索失败: {e}")
            return {}
        
        log.info(f"[Kaggle] API 搜索汇总结果: {sum(len(v) for v in results.values())} 个候选")
        return results

    @staticmethod
    def _to_ref(s: str) -> str | None:
        # 支持 URL 或者 owner/slug 形式
        s = (s or '').strip()
        if not s:
            return None
        if 'kaggle.com/datasets/' in s:
            m = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?#]+)", s)
            if not m:
                return None
            return f"{m.group(1)}/{m.group(2)}"
        # 直接是 ref
        if '/' in s and len(s.split('/')) == 2:
            return s
        return None

    async def try_download(self, page: Page, dataset_identifier: str, save_dir: str) -> str | None:
        os.makedirs(save_dir, exist_ok=True)
        ref = self._to_ref(dataset_identifier)
        if not ref:
            log.info(f"[Kaggle] 无法解析数据集标识: {dataset_identifier}")
            return None
        
        # 优先使用 kagglehub
        # 注意：kagglehub 可能会在缓存目录留下文件，但我们已经设置了环境变量
        try:
            import kagglehub  # type: ignore
            log.info(f"[Kaggle] 优先使用 kagglehub 下载: {ref}")
            path = await asyncio.to_thread(kagglehub.dataset_download, ref)
            if path and os.path.exists(path):
                log.info(f"[Kaggle] kagglehub 下载完成: {path}")
                # 如果 kagglehub 下载的路径不在 save_dir 中，尝试将文件移动到指定目录
                # 这样可以避免在缓存目录留下文件
                if os.path.abspath(path) != os.path.abspath(save_dir):
                    try:
                        # 如果是文件，移动到 save_dir；如果是目录，复制内容
                        if os.path.isfile(path):
                            dest_path = os.path.join(save_dir, os.path.basename(path))
                            shutil.move(path, dest_path)
                            log.info(f"[Kaggle] 已移动文件到指定目录: {dest_path}")
                            return dest_path
                        elif os.path.isdir(path):
                            # 如果是目录，复制内容到 save_dir
                            for item in os.listdir(path):
                                src_item = os.path.join(path, item)
                                dst_item = os.path.join(save_dir, item)
                                if os.path.isdir(src_item):
                                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src_item, dst_item)
                            log.info(f"[Kaggle] 已复制内容到指定目录: {save_dir}")
                            # 如果禁用了缓存，删除原始缓存目录
                            if self.disable_cache:
                                try:
                                    shutil.rmtree(path, ignore_errors=True)
                                    log.info(f"[Kaggle] 已清理 kagglehub 缓存目录: {path}")
                                except Exception as e:
                                    log.info(f"[Kaggle] 清理缓存目录时出错: {e} (可忽略)")
                            return save_dir
                    except Exception as move_e:
                        log.info(f"[Kaggle] 移动/复制文件时出错: {move_e}，返回原始路径")
                return path
            log.info("[Kaggle] kagglehub 返回无效路径。")
        except Exception as e:
            log.info(f"[Kaggle] kagglehub 失败或未安装: {e}，尝试使用 KaggleApi。")
        
        # 如果 kagglehub 失败或未安装，尝试使用 KaggleApi（直接下载到指定目录，避免系统盘缓存）
        if self.api:
            try:
                log.info(f"[Kaggle] 使用 KaggleApi 下载: {ref} (直接下载到 {save_dir}，避免系统盘缓存)")
                # 设置 60 秒超时
                await asyncio.wait_for(
                    asyncio.to_thread(self.api.dataset_download_files, ref, path=save_dir, unzip=True, quiet=False),
                    timeout=60.0
                )
                log.info(f"[Kaggle] 下载完成并解压至: {save_dir}")
                return save_dir
            except asyncio.TimeoutError:
                log.info(f"[Kaggle] API 下载超时（60秒），失败")
            except Exception as e:
                log.info(f"[Kaggle] API 下载失败: {e}")
        else:
            log.info("[Kaggle] 未初始化 KaggleApi。")
        
        # 如果禁用了缓存，清理临时缓存目录
        if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
            try:
                if os.path.exists(self._temp_cache_dir):
                    shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                    log.info(f"[Kaggle] 已清理临时缓存目录: {self._temp_cache_dir}")
            except Exception as e:
                log.info(f"[Kaggle] 清理临时缓存目录时出错: {e} (可忽略)")
        
        # 最后兜底失败
        return None

class PaddleDatasetManager:

    def __init__(self, search_engine: str = "tavily"):
        self.search_engine = search_engine
        log.info(f"[Paddle] 初始化 (search_engine={self.search_engine})")

    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> List[str]:
        urls: List[str] = []
        for kw in keywords:
            try:
                query = f"site:paddlepaddle.org.cn {kw} 数据集"
                text = await ToolManager.search_web(query, search_engine=self.search_engine)
                found = ToolManager._extract_urls_from_markdown(text)
                # 只保留 Paddle 官方域名上的链接
                filtered = [u for u in found if "paddlepaddle.org.cn" in u]
                urls.extend(filtered)
            except Exception as e:
                log.info(f"[Paddle] 搜索 '{kw}' 出错: {e}")
        dedup = list(dict.fromkeys(urls))[:max_results]
        log.info(f"[Paddle] 搜索汇总结果: {len(dedup)} 个候选")
        return dedup

    async def try_download(self, page: Page, dataset_page_url: str, save_dir: str) -> str | None:
        os.makedirs(save_dir, exist_ok=True)
        try:
            content = await ToolManager._read_with_jina_reader(dataset_page_url)
            urls = content.get("urls", []) if content else []
            candidates = [u for u in urls if any(u.lower().endswith(ext) for ext in [".zip", ".csv", ".tar", ".gz", ".parquet"])]
            candidates = list(dict.fromkeys(candidates))
            log.info(f"[Paddle] 页面解析得到 {len(candidates)} 个下载候选链接")
            for u in candidates:
                path = await ToolManager.download_file(page, u, save_dir)
                if path:
                    return path
        except Exception as e:
            log.info(f"[Paddle] 解析页面失败: {e}")
        return None
class DownloadMethodDecisionAgent(BaseAgent):
    """下载方法决策器 - 决定使用哪种方法下载数据"""
    async def execute(self, state: WebCrawlState, logger: LogManager, current_objective: str, search_keywords: str) -> Dict[str, Any]:
        log.info("\n--- 下载方法决策器 ---")
        # 记录输入
        inputs = {
            "current_objective": current_objective,
            "search_keywords": search_keywords,
            "state_initial_request": state.initial_request
        }
        log_agent_input_output("DownloadMethodDecisionAgent", inputs, logger=logger)
        
        system_prompt = self.prompt_gen.render("system_prompt_for_download_method_decision")
        human_prompt = self.prompt_gen.render("task_prompt_for_download_method_decision", 
                                              objective=current_objective, 
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
                        decision = await self.download_decision_agent.execute(
                            state, self.logger, task_objective, search_keywords
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

async def main():
    # 从环境变量获取 API 配置
    api_base_url = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
    api_key = os.getenv("DF_API_KEY")
    model_name = os.getenv("DF_MODEL", "gpt-4o")
    
    if not api_key:
        log.info("错误: 请在运行脚本前设置 DF_API_KEY 环境变量。")
        return
    

    user_request = "帮我找一些代码的数据集"
    
    # dataset_size_categories 可选值: ["n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "1M<n<10M", "10M<n<100M", "100M<n<1B", "n>1B"]
    # search_engine 可选值: "tavily", "duckduckgo", "jina"
    orchestrator = WebCrawlOrchestrator(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        dataset_size_categories=["1K<n<10K"],
        max_crawl_cycles_per_task=10,  # 下载任务的最大循环次数
        max_crawl_cycles_for_research=15,  # research阶段的最大循环次数，允许访问更多网站
        search_engine="tavily",  # 选择搜索引擎: tavily, duckduckgo, jina
        use_jina_reader=True,  # 是否使用 Jina Reader 提取网页结构化内容（Markdown格式，快速）
        enable_rag=True,  # 是否启用 RAG 增强（无论使用哪种解析方法，都用 RAG 精炼内容）
        concurrent_pages=5,  # 并行处理的页面数量，可根据网络和机器性能调整（建议3-10）
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    await orchestrator.run(user_request)

if __name__ == "__main__":
    asyncio.run(main())