
from __future__ import annotations
import os

# 在导入 huggingface_hub / datasets 之前，优先设置 HF_ENDPOINT，确保所有内部请求走镜像
# 支持通过环境变量 DF_HF_ENDPOINT 覆盖（例如 https://hf-mirror.com）
_df_hf_endpoint = os.getenv("DF_HF_ENDPOINT")
if _df_hf_endpoint and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = _df_hf_endpoint
import asyncio
import json
import httpx
import re
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import shutil
from playwright.async_api import async_playwright, Page, Error as PlaywrightError
from urllib.parse import urljoin
import tenacity
import requests.exceptions
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_community.tools import DuckDuckGoSearchRun

import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import WebCrawlState
from dataflow_agent.logger import get_logger

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

class BaseAgent(ABC):
    _prompt_generator = None
    
    def __init__(self, model_name: str = "gpt-4o", api_base_url: str = None, api_key: str = None):
        if not api_base_url:
            raise ValueError("错误：必须提供 api_base_url 参数！")
        if not api_key:
            raise ValueError("错误：必须提供 api_key 参数！")
        self.llm = ChatOpenAI(model=model_name, temperature=0.1, openai_api_base=api_base_url, openai_api_key=api_key, max_tokens=4096)
        
        # 初始化 prompt generator（只初始化一次）
        if BaseAgent._prompt_generator is None:
            BaseAgent._prompt_generator = PromptsTemplateGenerator(
                output_language="zh",
                python_modules=["prompts_repo"]
            )
        self.prompt_gen = BaseAgent._prompt_generator

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass

    def _create_messages(self, system_prompt: str, human_prompt: str) -> List:
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

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
class ToolManager:
    @staticmethod
    async def search_web(query: str, search_engine: str = "tavily") -> str:

        if isinstance(query, (list, tuple)):
            query = ", ".join([str(x) for x in query if x])
        elif not isinstance(query, str):
            query = str(query)
        
        log.info(f"[Search] 使用 {search_engine.upper()} 搜索: '{query}'")
 
        if search_engine.lower() == "jina":
            return await ToolManager._jina_search(query)
        elif search_engine.lower() == "duckduckgo":
            return await ToolManager._duckduckgo_search(query)
        else:  
            return await ToolManager._tavily_search(query)
    
    @staticmethod
    async def _tavily_search(query: str) -> str:
        tavily_api_key = os.getenv("TAVILY_API_KEY", "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")
        if not tavily_api_key:
            log.info("[Tavily] API Key 未设置，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)
        
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": False,
                        "include_raw_content": False,
                        "max_results": 30
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                
                if not results:
                    log.info("[Tavily] 无结果，回退到 DuckDuckGo")
                    return await ToolManager._duckduckgo_search(query)
                
                formatted = [
                    f"标题: {item.get('title', '无标题')}\n"
                    f"URL: {item.get('url', '')}\n"
                    f"摘要: {item.get('content', '')}\n---"
                    for item in results
                ]
                log.info(f"[Tavily] 搜索完成，找到 {len(results)} 个结果")
                return "\n".join(formatted)
        except Exception as e:
            log.info(f"[Tavily] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)
    
    @staticmethod
    async def _duckduckgo_search(query: str) -> str:
        """DuckDuckGo 搜索"""
        try:
            search_tool = DuckDuckGoSearchRun()
            result_text = await asyncio.to_thread(search_tool.run, query)
            log.info(f"[DuckDuckGo] 搜索完成")
            return result_text
        except Exception as e:
            log.info(f"[DuckDuckGo] 搜索错误: {e}")
            return ""
    
    @staticmethod
    async def _jina_search(query: str) -> str:

        try:
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"https://s.jina.ai/{encoded_query}"
            
            log.info(f"[Jina Search] 搜索查询: {query}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    search_url,
                    headers={
                        "Accept": "application/json", 
                        "X-Return-Format": "markdown"  
                    }
                )
                resp.raise_for_status()

                try:
                    data = resp.json()
                    if isinstance(data, dict) and "data" in data:
                        results = data.get("data", [])
                        if results:
                            formatted = []
                            for item in results[:10]: 
                                title = item.get("title", "无标题")
                                url = item.get("url", "")
                                content = item.get("content", "") or item.get("description", "")
                                formatted.append(f"标题: {title}\nURL: {url}\n摘要: {content}\n---")
                            log.info(f"[Jina Search] 搜索完成，找到 {len(formatted)} 个结果")
                            return "\n".join(formatted)
                except:
                    text_content = resp.text
                    if text_content:
                        log.info(f"[Jina Search] 搜索完成（文本模式）")
                        return text_content[:15000]  
                
                log.info("[Jina Search] 无搜索结果，回退到 DuckDuckGo")
                return await ToolManager._duckduckgo_search(query)
                
        except Exception as e:
            log.info(f"[Jina Search] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)

    @staticmethod
    async def read_web_page(page: Page, url: str, use_jina_reader: bool = False) -> Dict[str, Any]:

        if use_jina_reader:
            return await ToolManager._read_with_jina_reader(url)

        log.info(f"[Playwright] 正在读取网页: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
                element.decompose()
            text_content = soup.get_text(separator='\n', strip=True)
            base_url = await page.evaluate("() => document.baseURI")
            raw_urls = [a.get('href', '').strip() for a in soup.find_all('a', href=True) if a.get('href', '').strip() and not a.get('href').startswith(('javascript:', 'mailto:'))]
            urls = [urljoin(base_url, raw_url) for raw_url in raw_urls]
            log.info(f"[Playwright] 网页读取成功: {url}")
            return {"urls": urls, "text": text_content}
        except PlaywrightError as e:
            log.info(f"[Playwright] 读取网页时出错: {e}")
            return {"urls": [], "text": f"错误: 无法访问页面 {url} ({e})"}
        except Exception as e:
            log.info(f"读取网页时发生未知错误: {e}")
            return {"urls": [], "text": f"错误: 读取页面 {url} 时发生未知错误。"}
    
    @staticmethod
    async def _read_with_jina_reader(url: str) -> Dict[str, Any]:

        log.info(f"[Jina Reader] 正在提取网页: {url}")
        try:
            jina_url = f"https://r.jina.ai/{url}"
            
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(
                    jina_url,
                    headers={
                        "Accept": "text/plain",  
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                resp.raise_for_status()

                text_response = resp.text

                structured_content = ToolManager._parse_jina_text_format(text_response, url)

                markdown_content = structured_content.get("markdown", "")

                warning = structured_content.get("warning", "")
                if warning:
                    log.info(f"[Jina Reader] 警告: {warning}")
                    if "blocked" in warning.lower() or "403" in warning or "forbidden" in warning.lower():
                        log.info(f"[Jina Reader] 网页被封禁，无法提取内容")
                        return {
                            "urls": [],
                            "text": f"无法访问该页面: {warning}",
                            "structured_content": structured_content
                        }
                

                urls = ToolManager._extract_urls_from_markdown(markdown_content)
                
                log.info(f"[Jina Reader] 提取成功: {len(markdown_content)} 字符, {len(urls)} 个链接")
                
                return {
                    "urls": urls,
                    "text": markdown_content,
                    "structured_content": structured_content
                }
                    
        except httpx.HTTPStatusError as e:
            log.info(f"[Jina Reader] HTTP错误 {e.response.status_code}: {e}")
            return {
                "urls": [],
                "text": f"HTTP错误: {e.response.status_code}",
                "structured_content": None
            }
        except Exception as e:
            log.info(f"[Jina Reader] 提取失败: {e}")
            return {
                "urls": [],
                "text": f"Jina Reader 错误: {str(e)}",
                "structured_content": None
            }
    
    @staticmethod
    def _parse_jina_text_format(text: str, original_url: str) -> Dict[str, Any]:

        structured = {
            "title": "",
            "url_source": original_url,
            "warning": "",
            "markdown": "",
            "url": original_url
        }
        
        lines = text.split('\n')
        current_section = None
        markdown_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Title:"):
                structured["title"] = line[6:].strip()
                i += 1
                continue
            
            if line.startswith("URL Source:"):
                structured["url_source"] = line[11:].strip()
                i += 1
                continue

            if line.startswith("Warning:"):
                structured["warning"] = line[8:].strip()
                i += 1
                continue
            

            if line == "Markdown Content:":
                current_section = "markdown"
                i += 1

                while i < len(lines):
                    markdown_lines.append(lines[i])
                    i += 1
                break
            
            i += 1
        
        structured["markdown"] = '\n'.join(markdown_lines).strip()
        
        return structured
    
    @staticmethod
    def _extract_urls_from_markdown(markdown_text: str) -> List[str]:

        import re
        urls = []
        
        markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        markdown_links = re.findall(markdown_link_pattern, markdown_text)
        for text, url in markdown_links:
            if url and not url.startswith('#'): 
                urls.append(url)

        plain_url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        plain_urls = re.findall(plain_url_pattern, markdown_text)
        urls.extend(plain_urls)

        unique_urls = list(dict.fromkeys(urls))[:50]
        
        return unique_urls
    
    @staticmethod
    async def check_if_download_link(url: str) -> Dict[str, Any]:

        result = {
            "is_download": False,
            "reason": "",
            "content_type": "",
            "filename": ""
        }
        
        # 1. 快速检查：URL 扩展名
        common_file_extensions = [
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz', 
            '.csv', '.xlsx', '.xls', '.json', '.xml', '.tsv',  
            '.pdf', '.doc', '.docx', '.txt', '.md', 
            '.jpg', '.jpeg', '.png', '.gif', '.svg',
            '.mp4', '.avi', '.mov', '.mp3', '.wav',  
            '.exe', '.msi', '.dmg', '.deb', '.rpm',  
            '.parquet', '.arrow', '.h5', '.hdf5', '.pkl'  
        ]
        
        url_lower = url.lower()
        for ext in common_file_extensions:
            if url_lower.endswith(ext) or f"{ext}?" in url_lower or f"{ext}#" in url_lower:
                result["is_download"] = True
                result["reason"] = f"URL包含文件扩展名: {ext}"
                result["filename"] = url.split('/')[-1].split('?')[0].split('#')[0]
                return result
        

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.head(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })

                content_disposition = response.headers.get("Content-Disposition", "")
                if content_disposition and "attachment" in content_disposition.lower():
                    result["is_download"] = True
                    result["reason"] = "Content-Disposition 包含 attachment"
                    if "filename=" in content_disposition:
                        import re
                        filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
                        if filename_match:
                            result["filename"] = filename_match.group(1).strip('\'"')
                    return result

                content_type = response.headers.get("Content-Type", "").lower()
                result["content_type"] = content_type

                downloadable_types = [
                    "application/octet-stream",
                    "application/zip",
                    "application/x-zip-compressed",
                    "application/x-rar-compressed",
                    "application/pdf",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument",
                    "application/x-tar",
                    "application/gzip",
                    "application/x-gzip",
                    "text/csv",
                    "application/json",
                    "application/xml",
                    "image/",
                    "video/",
                    "audio/"
                ]
                
                for dtype in downloadable_types:
                    if dtype in content_type:
                        result["is_download"] = True
                        result["reason"] = f"Content-Type 是可下载类型: {content_type}"
                        return result
                
                if "text/html" in content_type:
                    result["is_download"] = False
                    result["reason"] = "Content-Type 是 HTML 页面，非文件下载"
                    return result
                
        except Exception as e:
            result["reason"] = f"HEAD 请求失败: {e}，无法确定"
            return result
        
        result["reason"] = "无法确定是否为下载链接"
        return result

    @staticmethod
    async def download_file(page: Page, url: str, save_dir: str) -> str | None:
        log.info(f"[Playwright] 准备从 {url} 下载文件")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            download_page = await page.context.new_page()
            async with download_page.expect_download(timeout=12000) as download_info: 
                try:
                    await download_page.goto(url, timeout=60000)
                except PlaywrightError as e:
                    if "Download is starting" in str(e) or "navigation" in str(e):
                        log.info(f"下载已通过导航或重定向触发。")
                        pass
                    else:
                        raise e
            download = await download_info.value
            try:
                await download_page.close()
            except Exception as close_e:
                log.info(f"关闭下载页面时出错（可忽略）: {close_e}")
            suggested_filename = download.suggested_filename
            save_path = os.path.join(save_dir, suggested_filename)
            log.info(f"文件 '{suggested_filename}' 正在保存中...")
            temp_file_path = await download.path()
            if not temp_file_path:
                    log.info(f"[Playwright] 下载失败，未能获取临时文件路径。")
                    await download.delete() 
                    return None
            shutil.move(temp_file_path, save_path)
            log.info(f"[Playwright] 下载完成: {save_path}")
            return save_path
        except Exception as e:
            log.info(f" [Playwright] 下载过程中发生意外错误 ({url}): {e}")
            try:
                await download_page.close()
            except:
                pass
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
        
class SubTaskRefinerAgent(BaseAgent):
    """在 research 结束后，对生成的子任务列表进行一次 LLM 筛选去重/一致性检查。"""
    async def execute(self, message: str, sub_tasks: list[dict], logger: LogManager) -> list[dict]:
        log.info("\n--- 子任务精炼器 ---")
        system_prompt = self.prompt_gen.render("system_prompt_for_subtask_refiner")
        try:
            sub_tasks_json = json.dumps(sub_tasks, indent=2, ensure_ascii=False)
        except Exception:
            # 兜底：将不可序列化对象转字符串
            sub_tasks_json = json.dumps(sub_tasks, indent=2, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o)))
        human_prompt = self.prompt_gen.render(
            "task_prompt_for_subtask_refiner",
            message=message or "",
            sub_tasks=sub_tasks_json
        )
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("subtask_refiner_raw_response", response.content)
        try:
            clean = response.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean)
            filtered = data.get("filtered_sub_tasks", [])
            if isinstance(filtered, list):
                log.info(f"[SubTaskRefiner] 过滤后子任务数: {len(filtered)}")
                logger.log_data("subtask_refiner_parsed", filtered, is_json=True)
                return filtered
        except Exception as e:
            log.info(f"[SubTaskRefiner] 解析失败: {e}\n原始响应: {response.content}")
        # 解析失败则返回原列表
        return sub_tasks

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

class QueryGeneratorAgent(BaseAgent):
    """查询生成器 - 为RAG检索生成多样的英文查询"""
    async def execute(self, objective: str, message: str, logger: LogManager) -> List[str]:
        log.info("\n--- 查询生成器 ---")
        system_prompt = self.prompt_gen.render("system_prompt_for_query_generator")
        human_prompt = self.prompt_gen.render("task_prompt_for_query_generator",
                                              objective=objective,
                                              message=message)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("query_generator_raw_response", response.content)
        
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(clean_response)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                log.info(f"生成了 {len(queries)} 个检索查询: {queries}")
                logger.log_data("query_generator_parsed", queries, is_json=True)
                return queries
            else:
                log.info(f"查询生成格式错误，返回空列表")
                return []
        except Exception as e:
            log.info(f"解析查询生成响应时出错: {e}\n原始响应: {response.content}")
            return []

class SummaryAgent(BaseAgent):
    def _apply_download_limit_to_state(self, state: WebCrawlState) -> None:
        """在研究阶段内部裁剪下载子任务，避免超过上限。"""
        if not hasattr(state, "sub_tasks") or state.sub_tasks is None:
            return

        limit = getattr(state, "max_download_subtasks", None)
        if limit is None:
            return

        completed = getattr(state, "completed_download_tasks", 0) or 0
        remaining = limit - completed

        if remaining <= 0:
            downloads_removed = sum(1 for task in state.sub_tasks if task.get("type") == "download")
            if downloads_removed:
                log.info(f"[Summary] 下载子任务上限已达，移除 {downloads_removed} 个待执行的下载子任务。")
            state.sub_tasks = [task for task in state.sub_tasks if task.get("type") != "download"]
            return

        filtered_tasks: List[Dict[str, Any]] = []
        kept_downloads = 0
        downloads_removed = 0
        for task in state.sub_tasks:
            if task.get("type") == "download":
                if kept_downloads >= remaining:
                    downloads_removed += 1
                    continue
                kept_downloads += 1
            filtered_tasks.append(task)

        if downloads_removed:
            log.info(f"[Summary] 应用下载子任务上限，裁剪 {downloads_removed} 个多余的下载子任务。")
        state.sub_tasks = filtered_tasks

    async def execute(self, state: WebCrawlState, logger: LogManager, research_objective: str, query_generator: QueryGeneratorAgent = None) -> WebCrawlState:
        log.info("\n--- 总结与规划（使用RAG增强，多次查询多次生成） ---")
        
        # 收集所有生成的新子任务
        all_new_sub_tasks = []
        all_summaries = []
        
        # [--- 使用 RAG 获取精炼的相关内容 ---]
        if state.enable_rag and state.rag_manager:
            log.info("[Summary] 使用 RAG 检索相关内容...")
            
            # 生成多样化的英文查询
            queries = None
            if query_generator:
                try:
                    queries = await query_generator.execute(
                        objective=research_objective,
                        message=getattr(state, "user_message", ""),
                        logger=logger
                    )
                except Exception as e:
                    log.info(f"[Summary] 查询生成失败: {e}，使用单一查询")
            
            # 如果没有生成查询，使用原始objective作为单查询
            if not queries or len(queries) == 0:
                queries = [research_objective]
            
            log.info(f"[Summary] 将对 {len(queries)} 个查询分别检索并生成子任务")
            
            # 对每个查询分别检索并生成子任务
            for query_idx, query in enumerate(queries, 1):
                log.info(f"\n[Summary] === 处理查询 {query_idx}/{len(queries)}: {query[:50]}... ===")
                
                # 获取当前查询的上下文
                relevant_context = await state.rag_manager.get_context_for_single_query(
                    query=query,
                    max_chars=18000
                )
                
                if not relevant_context:
                    log.info(f"[Summary] 查询 {query_idx} 检索结果为空，跳过")
                    continue
                
                # 收集当前已有的download子任务列表（包括之前生成的，使用集合去重）
                # 使用集合来存储任务的唯一标识，避免重复
                seen_task_keys = set()
                existing_download_tasks = []
                
                # 从state.sub_tasks中收集已有的download任务
                if hasattr(state, "sub_tasks") and state.sub_tasks:
                    for task in state.sub_tasks:
                        if task.get("type") == "download":
                            task_key = (task.get("objective", ""), task.get("search_keywords", ""))
                            if task_key not in seen_task_keys:
                                seen_task_keys.add(task_key)
                                existing_download_tasks.append({
                                    "objective": task.get("objective", ""),
                                    "search_keywords": task.get("search_keywords", "")
                                })
                
                # 也要包含本次循环中已生成的新子任务
                for new_task in all_new_sub_tasks:
                    task_key = (new_task.get("objective", ""), new_task.get("search_keywords", ""))
                    if task_key not in seen_task_keys:
                        seen_task_keys.add(task_key)
                        existing_download_tasks.append({
                            "objective": new_task.get("objective", ""),
                            "search_keywords": new_task.get("search_keywords", "")
                        })
                
                existing_subtasks_str = json.dumps(existing_download_tasks, indent=2, ensure_ascii=False) if existing_download_tasks else "[]"
                
                # 调用模型生成子任务
                system_prompt = self.prompt_gen.render("system_prompt_for_summary_agent")
                human_prompt = self.prompt_gen.render("task_prompt_for_summary_agent",
                                                      objective=research_objective,
                                                      message=getattr(state, "user_message", ""),
                                                      existing_subtasks=existing_subtasks_str,
                                                      context=relevant_context)
                messages = self._create_messages(system_prompt, human_prompt)
                response = await self.llm.ainvoke(messages)
                log_name = f"summary_query_{query_idx}_{research_objective.replace(' ', '_')}"
                logger.log_data(f"{log_name}_raw_response", response.content)
                logger.log_data(f"{log_name}_query", query)
                logger.log_data(f"{log_name}_context_used", relevant_context)
                logger.log_data(f"{log_name}_existing_subtasks", existing_download_tasks, is_json=True)
                
                try:
                    clean_response = response.content.strip().replace("```json", "").replace("```", "")
                    summary_plan = json.loads(clean_response)
                    
                    new_tasks = summary_plan.get("new_sub_tasks", [])
                    summary_text = summary_plan.get("summary", "")
                    
                    if new_tasks:
                        log.info(f"[Summary] 查询 {query_idx} 生成了 {len(new_tasks)} 个新子任务")
                        # 将新任务添加到all_new_sub_tasks
                        all_new_sub_tasks.extend(new_tasks)
                        
                        # 将新子任务添加到state.sub_tasks中，以便下次查询时能避免重复
                        # 添加时也做去重，避免state.sub_tasks中有重复任务
                        if not hasattr(state, "sub_tasks") or state.sub_tasks is None:
                            state.sub_tasks = []
                        
                        # 使用集合跟踪已有的任务键
                        existing_task_keys = set()
                        for task in state.sub_tasks:
                            if task.get("type") == "download":
                                task_key = (task.get("objective", ""), task.get("search_keywords", ""))
                                existing_task_keys.add(task_key)
                        
                        # 只添加不重复的新任务
                        for new_task in new_tasks:
                            task_key = (new_task.get("objective", ""), new_task.get("search_keywords", ""))
                            if task_key not in existing_task_keys:
                                state.sub_tasks.append(new_task)
                                existing_task_keys.add(task_key)
                    
                    if summary_text:
                        all_summaries.append(f"[Query {query_idx}: {query[:30]}...] {summary_text}")
                    
                    logger.log_data(f"{log_name}_parsed_plan", summary_plan, is_json=True)
                except Exception as e:
                    log.info(f"[Summary] 查询 {query_idx} 解析响应时出错: {e}\n原始响应: {response.content}")
            
            # 记录RAG统计
            rag_stats = state.rag_manager.get_statistics()
            log.info(f"[Summary] RAG统计: {rag_stats}")
            logger.log_data("rag_statistics", rag_stats, is_json=True)
            
        else:
            # RAG未启用，使用传统方法（单次生成）
            log.info("[Summary] RAG 未启用，使用传统方法...")
            all_text = "\n\n---\n\n".join([data['text_content'] for data in state.crawled_data if 'text_content' in data])
            relevant_context = all_text[:18000] if all_text else ""
            
            if not relevant_context:
                log.info("研究阶段未能收集到任何文本内容，无法生成新任务。")
                return state
            
            # 收集现有的download子任务列表
            existing_download_tasks = []
            if hasattr(state, "sub_tasks") and state.sub_tasks:
                for task in state.sub_tasks:
                    if task.get("type") == "download":
                        existing_download_tasks.append({
                            "objective": task.get("objective", ""),
                            "search_keywords": task.get("search_keywords", "")
                        })
            existing_subtasks_str = json.dumps(existing_download_tasks, indent=2, ensure_ascii=False) if existing_download_tasks else "[]"
            
            system_prompt = self.prompt_gen.render("system_prompt_for_summary_agent")
            human_prompt = self.prompt_gen.render("task_prompt_for_summary_agent",
                                                  objective=research_objective,
                                                  message=getattr(state, "user_message", ""),
                                                  existing_subtasks=existing_subtasks_str,
                                                  context=relevant_context)
            messages = self._create_messages(system_prompt, human_prompt)
            response = await self.llm.ainvoke(messages)
            log_name = f"summary_and_plan_{research_objective.replace(' ', '_')}"
            logger.log_data(f"{log_name}_raw_response", response.content)
            logger.log_data(f"{log_name}_context_used", relevant_context)
            try:
                clean_response = response.content.strip().replace("```json", "").replace("```", "")
                summary_plan = json.loads(clean_response)
                all_new_sub_tasks = summary_plan.get("new_sub_tasks", [])
                all_summaries = [summary_plan.get("summary", "No summary")]
                logger.log_data(f"{log_name}_parsed_plan", summary_plan, is_json=True)
            except Exception as e:
                log.info(f"解析总结规划响应时出错: {e}\n原始响应: {response.content}")
        
        # 将所有结果合并到research_summary
        state.research_summary = {
            "new_sub_tasks": all_new_sub_tasks,
            "summary": "\n".join(all_summaries) if all_summaries else "No summary"
        }
        self._apply_download_limit_to_state(state)
        
        log.info(f"\n总结与规划完成:")
        log.info(f"  - 总共生成了 {len(all_new_sub_tasks)} 个新的下载任务")
        if all_summaries:
            summary_preview = "\n".join(all_summaries)[:300]
            log.info(f"  - 摘要预览: {summary_preview}..." if len("\n".join(all_summaries)) > 300 else f"  - 摘要: {summary_preview}")
        
        return state

# --- Agent ---
class URLFilter(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, is_research: bool = False, **kwargs) -> WebCrawlState:
        log.info("\n--- flitter ---")
        if not state.search_results_text:
            log.info("没有搜索结果文本可供筛选。")
            return state
        
        # Research 阶段需要更多URL
        url_count_instruction = "尽可能多地提取URL（至少10-15个）" if is_research else "提取最相关的URL（5-10个）"
        
        system_prompt = self.prompt_gen.render("system_prompt_for_url_filter",
                                               url_count_instruction=url_count_instruction)
        human_prompt = self.prompt_gen.render("task_prompt_for_url_filter",
                                              request=state.initial_request,
                                              search_results=state.search_results_text)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("3_url_filter_raw_response", response.content)
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean_response)
            urls = result.get("selected_urls", [])
            state.filtered_urls = urls
            state.url_queue.extend(urls)
            log.info(f"URL筛选完成。待爬取栈: {state.url_queue}")
            logger.log_data("3_url_filter_parsed_output", result, is_json=True)
        except Exception as e:
            log.info(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
        return state

# --- Agent---
class WebPageReader(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, page: Page, url: str, current_objective: str, is_research: bool = False) -> Dict[str, Any]:
        log.info(f"\n--- web_reader (目标: {current_objective}) ---")
        log.info(f"--- 正在分析 URL: {url} ---")
        
        page_data = await ToolManager.read_web_page(page, url, use_jina_reader=state.use_jina_reader)
        safe_url_filename = f"4_read_page_{url.replace('://', '_').replace('/', '_')}"
        logger.log_data(safe_url_filename, page_data, is_json=True)
        
        text_content = page_data.get("text", "")
        if text_content:
            crawl_entry = {"source_url": url, "text_content": text_content}
            # 如果使用了 Jina Reader，保存结构化内容
            if "structured_content" in page_data and page_data["structured_content"]:
                crawl_entry["structured_content"] = page_data["structured_content"]
            state.crawled_data.append(crawl_entry)
            
            # [--- RAG 集成：在 research 阶段将内容添加到 RAG 知识库（无论使用哪种解析方法） ---]
            if is_research and state.enable_rag and state.rag_manager:
                await state.rag_manager.add_webpage_content(
                    url=url,
                    text_content=text_content,
                    metadata={
                        "objective": current_objective,
                        "extraction_method": "jina_reader" if state.use_jina_reader else "playwright"
                    }
                )

        system_prompt = self.prompt_gen.render("system_prompt_for_webpage_reader")
        compact_text = page_data.get("text", "")[:16000]
        discovered_urls = page_data.get("urls", [])[:100]
        urls_block = "\n".join(discovered_urls)
        
        human_prompt = self.prompt_gen.render("task_prompt_for_webpage_reader",
                                              objective=current_objective,
                                              urls_block=urls_block,
                                              text_content=compact_text)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data(f"{safe_url_filename}_raw_response", response.content)
        
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            action_plan = json.loads(clean_response)
            log.info(f"页面分析完毕。计划行动: {action_plan.get('action')}, 描述: {action_plan.get('description')}")
            if action_plan.get('action') == 'download':
                 log.info(f"发现 {len(action_plan.get('urls', []))} 个下载链接。")
            logger.log_data(f"{safe_url_filename}_parsed_output", action_plan, is_json=True)
            return action_plan
        except Exception as e:
            log.info(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
            return {"action": "dead_end", "description": "Failed to parse LLM response."}


class Executor:

    def __init__(self):
        pass
    
    async def execute(self, state: WebCrawlState, action_plan: Dict[str, Any], source_url: str, page: Page, current_task_type: str) -> WebCrawlState:
        log.info(f"\n--- executor (任务类型: {current_task_type}) ---")
        action = action_plan.get("action")
        
        if action == "download":
            if current_task_type != "download":
                log.info(f"在 '{current_task_type}' 阶段发现下载链接，已忽略。URL: {action_plan.get('urls', 'N/A')}")
                return state

            success = await self._execute_web_download(state, action_plan, source_url, page)
            
            if success:
                state.download_successful_for_current_task = True
        
        elif action == "navigate":
            url_to_navigate = action_plan.get("url")
            if not url_to_navigate: return state
            full_url = urljoin(source_url, url_to_navigate) 
            if full_url not in state.visited_urls and full_url not in state.url_queue:
                state.url_queue.append(full_url)
                log.info(f"执行成功: 将新URL入栈: {full_url}")

        elif action == "dead_end":
            log.info("执行: 到达死胡同，无操作。")
        
        return state
    
    async def _execute_web_download(self, state: WebCrawlState, action_plan: Dict[str, Any], source_url: str, page: Page) -> bool:
        """执行网页下载（支持多文件，包含下载链接检查）"""
        urls_to_download = action_plan.get("urls", [])
        if not urls_to_download: 
            log.info("网页下载失败：LLM未在action_plan中提供 'urls' 列表。")
            return False
        
        download_succeeded_at_least_once = False
        
        for url_to_download in urls_to_download:
            if not url_to_download or not isinstance(url_to_download, str):
                log.info(f"跳过无效的下载条目: {url_to_download}")
                continue
                    
            try:
                full_url = urljoin(source_url, url_to_download)
                
                # [---检查是否是文件下载链接 ---]
                log.info(f"\n[检查] 正在验证 URL 是否为下载链接: {full_url}")
                check_result = await ToolManager.check_if_download_link(full_url)
                
                log.info(f"[检查结果] 是否为下载链接: {check_result['is_download']}")
                log.info(f"[检查结果] 原因: {check_result['reason']}")
                if check_result.get('content_type'):
                    log.info(f"[检查结果] Content-Type: {check_result['content_type']}")
                if check_result.get('filename'):
                    log.info(f"[检查结果] 文件名: {check_result['filename']}")

                if check_result['is_download'] == False and "HTML 页面" in check_result['reason']:
                    log.info(f"跳过非文件链接: {full_url}")
                    continue

                if not check_result['is_download']:
                    log.info(f"非下载链接，跳过: {full_url} ({check_result['reason']})")
                    continue

                if state.max_dataset_size:
                    content_length = 0
                    try:
                        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                            head_resp = await client.head(full_url, headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                            })
                            content_length = int(head_resp.headers.get("Content-Length", 0))
                    except Exception as size_err:
                        log.info(f"检查文件大小失败，将继续下载: {size_err}")
                    if content_length and content_length > state.max_dataset_size:
                        log.info(f"跳过下载：文件大小 {content_length} 字节超过限制 {state.max_dataset_size} 字节 -> {full_url}")
                        continue

                log.info(f"✓ 确认为文件下载链接，开始下载...")

                final_save_path = await ToolManager.download_file(page, full_url, save_dir=state.download_dir)

                if final_save_path:
                    state.crawled_data.append({
                        "source_url": full_url,
                        "local_path": final_save_path,
                        "type": "file",
                        "check_result": check_result
                    })
                    log.info(f"网页下载成功: 已下载文件到 {final_save_path}")
                    download_succeeded_at_least_once = True
                else:
                    log.info(f"网页下载失败: 从 {full_url} 下载文件失败。")
                        
            except Exception as e:
                full_url_str = "N/A"
                try:
                    full_url_str = urljoin(source_url, str(url_to_download))
                except Exception:
                    pass
                log.info(f"下载 {url_to_download} (Full: {full_url_str}) 时发生异常: {e}")
        
        return download_succeeded_at_least_once
    
class Supervisor(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, current_objective: str, is_research: bool = False) -> WebCrawlState:
        log.info("\n--- supervisor ---")
        state.current_cycle += 1
        # Research 阶段允许更多循环次数
        max_cycles = state.max_crawl_cycles_for_research if is_research else state.max_crawl_cycles_per_task
        if not state.url_queue or state.current_cycle >= max_cycles:
            log.info(f"已完成 {state.current_cycle} 个循环 (最大: {max_cycles})")
            return state
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
        self.summary_agent = SummaryAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.url_filter = URLFilter(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.web_page_reader = WebPageReader(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.executor = Executor()
        self.supervisor = Supervisor(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.download_decision_agent = DownloadMethodDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.logger = LogManager()
        
        # *** 在 Orchestrator 中初始化 HF 工具和新 Agent ***
        # 如果未禁用缓存，设置缓存目录到 download_dir 下的 .cache 文件夹，避免占用系统盘
        cache_base_dir = None if disable_cache else os.path.join(os.path.abspath(self.download_dir), ".cache")
        
        self.hf_decision_agent = HuggingFaceDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.kaggle_decision_agent = KaggleDecisionAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.dataset_detail_reader = DatasetDetailReaderAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.subtask_refiner = SubTaskRefinerAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        self.hf_manager = HuggingFaceDatasetManager(max_retries=2, retry_delay=5, size_categories=dataset_size_categories, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # Kaggle / Paddle 管理器
        self.kaggle_manager = KaggleDatasetManager(search_engine=self.search_engine, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # self.paddle_manager = PaddleDatasetManager(search_engine=self.search_engine)  # 已注释：Paddle数据集获取功能
        
        # [--- QueryGenerator Agent 初始化 ---]
        self.query_generator = QueryGeneratorAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        
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
                            state.sub_tasks = new_tasks + state.sub_tasks
                            self._apply_download_limit_to_state(state)
                            continue
                        else:
                            log.info(f"研究阶段未发现具体下载目标，执行默认的下载任务作为兜底。") 

                    download_method = "huggingface"
                    hf_keywords = []
                    fallback_method = "web_crawl"
                    
                    if task_type == "download":
                        log.info("正在获取 HuggingFace 搜索关键词（默认优先HF）...")
                        decision = await self.download_decision_agent.execute(
                            state, self.logger, task_objective, search_keywords
                        )
                        # 只用于提取关键词，策略固定为先HF后Web
                        hf_keywords = decision.get("keywords_for_hf", [])
                        if not hf_keywords:
                            # 回退：从当前任务关键词/目标生成
                            if isinstance(search_keywords, (list, tuple)):
                                hf_keywords = [kw for kw in search_keywords if isinstance(kw, str) and kw.strip()] or [task_objective]
                            else:
                                hf_keywords = [search_keywords] if isinstance(search_keywords, str) and search_keywords.strip() else [task_objective]
                        log.info(f"HuggingFace搜索关键词: {hf_keywords}")

                    download_success = False
                    abort_download_task = False
                    
                    # 始终先尝试 HuggingFace："搜索 -> 决策 -> 下载"
                    if task_type == "download":
                        log.info("优先尝试使用 HuggingFace 方法下载...")
                        selected_id = None
                        hf_search_results = {}
                        try:
                            # 1. Search
                            log.info(f"[HuggingFace] 正在搜索关键词: {hf_keywords}")
                            hf_search_results = await self.hf_manager.search_datasets(hf_keywords, max_results=5)
                            
                            # 2. Decide
                            selected_id = await self.hf_decision_agent.execute(
                                hf_search_results, task_objective, self.logger, 
                                message=getattr(state, "user_message", ""),
                                max_dataset_size=self.max_dataset_size
                            )
                            
                            # 3. 读取数据集详情（如果选择了数据集）
                            detail_result = None
                            if selected_id:
                                log.info(f"[HuggingFace] 已选择数据集: {selected_id}，开始读取详细信息...")
                                try:
                                    # 获取数据集基本信息
                                    dataset_info = {}
                                    for res_list in hf_search_results.values():
                                        for d in res_list:
                                            if d['id'] == selected_id:
                                                dataset_info = d
                                                break
                                    
                                    if not dataset_info:
                                        log.info(f"[HuggingFace] 警告: 未找到数据集 {selected_id} 的详细信息")
                                        dataset_info = {"id": selected_id}
                                    
                                    # 使用详情读取器分析数据集
                                    log.info(f"[HuggingFace] 调用详情读取器分析数据集...")
                                    detail_result = await self.dataset_detail_reader.execute(
                                        dataset_id=selected_id,
                                        dataset_type="huggingface",
                                        dataset_info=dataset_info,
                                        logger=self.logger,
                                        max_dataset_size=self.max_dataset_size
                                    )
                                    
                                    # 检查是否满足大小限制
                                    if self.max_dataset_size:
                                        log.info(f"[HuggingFace] 检查大小限制: 最大允许 {self.max_dataset_size} 字节")
                                        if detail_result and not detail_result.get("meets_size_limit", True):
                                            log.info(f"[HuggingFace] 数据集 {selected_id} 超过大小限制 ({self.max_dataset_size} 字节)，终止该下载子任务")
                                            abort_download_task = True
                                            selected_id = None
                                        elif detail_result:
                                            dataset_size = detail_result.get("size_bytes")
                                            if dataset_size:
                                                log.info(f"[HuggingFace] 数据集大小: {dataset_size} 字节，符合限制")
                                            else:
                                                log.info(f"[HuggingFace] 警告: 无法获取数据集大小，默认允许下载")
                                    else:
                                        log.info(f"[HuggingFace] 未设置大小限制，继续下载")
                                except Exception as detail_e:
                                    log.info(f"[HuggingFace] 读取数据集详情时出错: {detail_e}")
                                    import traceback
                                    traceback.print_exc()
                                    # 如果详情读取失败，但设置了大小限制，为了安全起见，跳过下载
                                    if self.max_dataset_size:
                                        log.info(f"[HuggingFace] 由于无法验证大小且设置了限制，终止该下载子任务")
                                        abort_download_task = True
                                        selected_id = None
                            
                            # 4. Download
                            if selected_id:
                                hf_download_dir = os.path.join(state.download_dir, "hF_datasets")
                                save_path = await self.hf_manager.download_dataset(selected_id, hf_download_dir)
                                
                                if save_path:
                                    # 记录下载的数据集信息
                                    dataset_info = {}
                                    for res_list in hf_search_results.values():
                                        for d in res_list:
                                            if d['id'] == selected_id:
                                                dataset_info = d
                                                break
                                    
                                    state.crawled_data.append({
                                        "source_url": f"https://huggingface.co/datasets/{selected_id}",
                                        "local_path": save_path, 
                                        "type": "huggingface_dataset",
                                        "dataset_id": selected_id,
                                        "dataset_info": dataset_info,
                                        "detail_analysis": detail_result
                                    })
                                    log.info(f"[HuggingFace] 数据集下载成功: {selected_id}")
                                    download_success = True
                                else:
                                    log.info(f"[HuggingFace] LLM选择的数据集 {selected_id} 下载失败。")
                            
                        except Exception as e:
                            log.info(f"[HuggingFace] HF下载流程发生意外错误: {e}")
                            download_success = False
                        if download_success:
                            state.download_successful_for_current_task = True

                    # 若 HF 未成功，尝试 Kaggle（但如果因大小限制已中止，则不再尝试）
                    if task_type == "download" and not state.download_successful_for_current_task and not abort_download_task:
                        try:
                            log.info("尝试使用 Kaggle 源下载...")
                            # 1. Search (返回详细信息)
                            kaggle_search_results = await self.kaggle_manager.search_datasets(hf_keywords or [task_objective], max_results=5)
                            
                            if kaggle_search_results:
                                # 2. Decide (使用模型选择)
                                selected_kaggle_id = await self.kaggle_decision_agent.execute(
                                    kaggle_search_results, task_objective, self.logger,
                                    message=getattr(state, "user_message", ""),
                                    max_dataset_size=self.max_dataset_size
                                )
                                
                                if selected_kaggle_id:
                                    # 3. 读取数据集详情
                                    kaggle_dataset_info = {}
                                    for res_list in kaggle_search_results.values():
                                        for d in res_list:
                                            if d['id'] == selected_kaggle_id:
                                                kaggle_dataset_info = d
                                                break
                                    
                                    # 使用详情读取器分析数据集
                                    kaggle_detail_result = await self.dataset_detail_reader.execute(
                                        dataset_id=selected_kaggle_id,
                                        dataset_type="kaggle",
                                        dataset_info=kaggle_dataset_info,
                                        logger=self.logger,
                                        max_dataset_size=self.max_dataset_size
                                    )
                                    
                                    # 检查是否满足大小限制
                                    if self.max_dataset_size and kaggle_detail_result and not kaggle_detail_result.get("meets_size_limit", True):
                                        log.info(f"[Kaggle] 数据集 {selected_kaggle_id} 超过大小限制 ({self.max_dataset_size} 字节)，跳过下载")
                                    else:
                                        # 4. Download
                                        dl_page = await context.new_page()
                                        try:
                                            save_dir = os.path.join(state.download_dir, "kaggle_datasets")
                                            saved = await self.kaggle_manager.try_download(dl_page, selected_kaggle_id, save_dir)
                                            if saved:
                                                state.crawled_data.append({
                                                    "source_url": f"https://www.kaggle.com/datasets/{selected_kaggle_id}",
                                                    "local_path": saved,
                                                    "type": "kaggle_dataset",
                                                    "dataset_id": selected_kaggle_id,
                                                    "dataset_info": kaggle_dataset_info,
                                                    "detail_analysis": kaggle_detail_result
                                                })
                                                log.info(f"[Kaggle] 下载成功: {saved}")
                                                state.download_successful_for_current_task = True
                                        finally:
                                            await dl_page.close()
                                else:
                                    log.info("[Kaggle] 模型未选择任何数据集。")
                            else:
                                log.info("[Kaggle] 未找到候选数据集。")
                        except Exception as e:
                            log.info(f"[Kaggle] 下载流程发生意外错误: {e}")

                    # 若 Kaggle 未成功，尝试 Paddle（已注释）
                    # if task_type == "download" and not state.download_successful_for_current_task:
                    #     try:
                    #         log.info("尝试使用 Paddle 源下载...")
                    #         paddle_urls = await self.paddle_manager.search_datasets(hf_keywords or [task_objective], max_results=5)
                    #         if paddle_urls:
                    #             dl_page = await context.new_page()
                    #             try:
                    #                 for ds_url in paddle_urls:
                    #                     log.info(f"[Paddle] 候选: {ds_url}")
                    #                     save_dir = os.path.join(state.download_dir, "paddle_datasets")
                    #                     saved = await self.paddle_manager.try_download(dl_page, ds_url, save_dir)
                    #                     if saved:
                    #                         state.crawled_data.append({
                    #                             "source_url": ds_url,
                    #                             "local_path": saved,
                    #                             "type": "paddle_dataset"
                    #                         })
                    #                         log.info(f"[Paddle] 下载成功: {saved}")
                    #                         state.download_successful_for_current_task = True
                    #                         break
                    #             finally:
                    #                 await dl_page.close()
                    #         else:
                    #             log.info("[Paddle] 未找到候选数据集。")
                    #     except Exception as e:
                    #         log.info(f"[Paddle] 下载流程发生意外错误: {e}")
                    
                    # --- Web Crawl Logic ---
                    if task_type == "research" or (task_type == "download" and not state.download_successful_for_current_task):
                        
                        is_research_phase = (task_type == "research")
                        
                        log.info(f"使用关键词进行搜索: '{search_keywords}'")
                        # 防御：关键词可能为列表
                        query_kw = ", ".join(search_keywords) if isinstance(search_keywords, (list, tuple)) else search_keywords
                        # [--- 使用 state 中的搜索引擎配置 ---]
                        search_results = await ToolManager.search_web(query_kw, search_engine=state.search_engine)
                        state.search_results_text = search_results
                        self.logger.log_data(f"2_search_results_{task_objective.replace(' ', '_')}", search_results)

                        # Research 阶段：直接从搜索结果中提取 URL 入队，跳过 URL 筛选
                        if is_research_phase:
                            try:
                                direct_urls = ToolManager._extract_urls_from_markdown(search_results) if search_results else []
                                # 防止过多：保留前 30 个即可
                                direct_urls = direct_urls[:30]
                                state.filtered_urls = direct_urls
                                for u in direct_urls:
                                    if u not in state.visited_urls and u not in state.url_queue:
                                        state.url_queue.append(u)
                                self.logger.log_data("3_url_extracted_directly_for_research", {
                                    "count": len(direct_urls),
                                    "urls": direct_urls
                                }, is_json=True)
                                log.info(f"Research阶段：已直接加入 {len(direct_urls)} 个URL到待爬取队列（跳过筛选）。")
                            except Exception as e:
                                log.info(f"Research阶段直接提取URL时出错: {e}")
                        else:
                            state = await self.url_filter.execute(state, logger=self.logger, is_research=is_research_phase)

                        max_cycles = state.max_crawl_cycles_for_research if is_research_phase else state.max_crawl_cycles_per_task
                        
                        while state.url_queue and state.current_cycle < max_cycles:
                            # 获取一批URL进行并行处理
                            batch_urls = []
                            while len(batch_urls) < self.concurrent_pages and state.url_queue:
                                current_url = state.url_queue.pop(0)
                                if current_url not in state.visited_urls:
                                    batch_urls.append(current_url)
                                    state.visited_urls.add(current_url)
                            
                            if not batch_urls:
                                break
                            
                            log.info(f"\n[并行处理] 开始处理 {len(batch_urls)} 个网页...")
                            
                            # 并行处理这批URL
                            results = await self._process_urls_parallel(
                                context, batch_urls, state, task_objective, 
                                task_type, is_research_phase
                            )
                            for result in results:
                                if result['success']:
                                    crawled_entry = result.get('crawled_entry')
                                    if crawled_entry:
                                        if 'multiple_entries' in crawled_entry:
                                            state.crawled_data.extend(crawled_entry['multiple_entries'])
                                        else:
                                            state.crawled_data.append(crawled_entry)

                                    for new_url in result.get('new_urls', []):
                                        if new_url not in state.visited_urls and new_url not in state.url_queue:
                                            state.url_queue.append(new_url)
                                    if result.get('download_successful'):
                                        state.download_successful_for_current_task = True
                            
                            state = await self.supervisor.execute(state, self.logger, current_objective=task_objective, is_research=is_research_phase)
                            if state.download_successful_for_current_task:
                                log.info(f"子任务 '{task_objective}' 的下载目标已完成，提前结束爬取循环。")
                                break
                            log.info(f"[并行处理] 批次完成，剩余 {len(state.url_queue)} 个URL待处理")
                    # 已经先尝试过 HuggingFace，这里不再二次回退到 HF
                    
                    if task_type == "research":
                        state = await self.summary_agent.execute(state, self.logger, research_objective=task_objective, query_generator=self.query_generator)
                        # 研究阶段完成后，对新生成的下载子任务统一做一次筛选去重
                        try:
                            new_tasks = state.research_summary.get("new_sub_tasks", []) if hasattr(state, "research_summary") else []
                            if new_tasks:
                                refined = await self.subtask_refiner.execute(
                                    message=getattr(state, "user_message", ""),
                                    sub_tasks=new_tasks,
                                    logger=self.logger
                                )
                                state.research_summary["new_sub_tasks"] = refined
                                self._apply_download_limit_to_state(state)
                                log.info(f"[Research] 子任务精炼完成：{len(new_tasks)} -> {len(refined)}")
                        except Exception as e:
                            log.info(f"[Research] 子任务精炼失败: {e}")
                    
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