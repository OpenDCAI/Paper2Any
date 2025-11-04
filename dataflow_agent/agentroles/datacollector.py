
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
from typing import Any, Dict, List
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

from huggingface_hub import HfApi,snapshot_download
from datasets import load_dataset, get_dataset_config_names
import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import WebCrawlState

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
        print(f"日志将保存在: {self.run_dir}")

    def log_data(self, step_name: str, data: Any, is_json: bool = False):
        if not self.run_dir:
            print("LogManager尚未初始化，无法记录日志。")
            return
        safe_step_name = re.sub(r'[\\/*?:"<>|]', "", step_name)
        extension = ".json" if is_json else ".txt"
        filename = os.path.join(self.run_dir, f"{safe_step_name}{extension}")
        content = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"写入日志文件 {filename} 时出错: {e}")

# --- RAG管理器 ---
class RAGManager:

    def __init__(self, api_base_url: str, api_key: str, persist_directory: str = "./rag_db", reset: bool = False, collection_name: str = "rag_collection"):
        print(f"[RAG] 初始化 RAG 管理器，存储目录: {persist_directory}")
        embed_model = os.getenv("RAG_EMB_MODEL", "text-embedding-3-large")
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=os.getenv("RAG_API_URL"),
            openai_api_key=os.getenv("RAG_API_KEY"),
            model=embed_model
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
            print(f"[RAG] 初始化向量存储失败: {e}")
            self.vectorstore = None
        # 去重集合，避免重复块污染召回
        self._seen_hashes = set()
    
    async def add_webpage_content(self, url: str, text_content: str, metadata: Dict[str, Any] = None):

        if not text_content or len(text_content.strip()) < 50:
            print(f"[RAG] 跳过内容过短的网页: {url}")
            return
        try:
            print(f"[RAG] 正在添加网页内容: {url} (长度: {len(text_content)} 字符)")
            # 基础清洗
            cleaned = re.sub(r"\s+", " ", text_content).strip()
            chunks = self.text_splitter.split_text(cleaned)
            print(f"[RAG] 文本已分成 {len(chunks)} 个块")
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
                print(f"[RAG] 清洗/去重后无有效文档块可添加: {url}")
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
                print(f"[RAG] 持久化失败: {e}")
            self.document_count += len(documents)
            print(f"[RAG] 成功添加 {len(documents)} 个文档块，总计: {self.document_count} 块")
        except Exception as e:
            print(f"[RAG] 添加网页内容时出错 ({url}): {e}")
    
    async def get_context_for_single_query(self, query: str, max_chars: int = 18000) -> str:
        """获取单个查询的上下文"""
        if self.vectorstore is None:
            print("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            print(f"[RAG] 检索查询: {query[:50]}...")
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
            print(f"[RAG] 查询检索完成: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            print(f"[RAG] 检索查询 '{query}' 时出错: {e}")
            return ""
    
    async def get_context_for_analysis(self, objective: str, max_chars: int = 20000, queries: List[str] = None) -> str:
        """获取用于分析的上下文，支持多个查询（合并结果）"""
        if self.vectorstore is None:
            print("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            # 如果没有提供查询，使用原始objective
            if queries is None or len(queries) == 0:
                queries = [objective]
            
            print(f"[RAG] 使用 {len(queries)} 个查询进行检索")
            all_docs = []
            seen_doc_hashes = set()
            
            # 对每个查询进行检索
            for i, query in enumerate(queries, 1):
                print(f"[RAG] 查询 {i}/{len(queries)}: {query[:50]}...")
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
                    print(f"[RAG] 查询 '{query}' 检索失败: {e}")
            
            print(f"[RAG] 合并后共获得 {len(all_docs)} 个去重文档")
            
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
            print(f"[RAG] 生成分析上下文: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            print(f"[RAG] 检索内容时出错: {e}")
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

    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5, size_categories: List[str] = None, cache_dir: str = None, disable_cache: bool = False, temp_base_dir: str = None):
        self.hf_endpoint = 'https://hf-mirror.com'
        self.hf_api = HfApi(endpoint=self.hf_endpoint)
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
            print(f"[HuggingFace] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
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
            print(f"[HuggingFace] 缓存目录已设置为: {hf_cache} (避免占用系统盘)")
        else:
            # 如果未指定，使用默认的项目目录
            default_cache = os.path.join(os.getcwd(), ".cache", "hf")
            os.makedirs(default_cache, exist_ok=True)
            os.environ['HF_HOME'] = default_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(default_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(default_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(default_cache, "transformers")
            self._temp_cache_dir = None
            print(f"[HuggingFace] 使用默认缓存目录: {default_cache}")
        
        print(f"[HuggingFace] 初始化，最大重试次数: {self.max_retries}, 延迟: {self.retry_delay}s (线性增长), 数据集大小类别: {self.size_categories if self.size_categories else '不限制'}")

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
            print(f"[HuggingFace] 发生可重试网络错误 (Attempt {attempt}/{self.max_retries}): {exception}")

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
            print(f"[HuggingFace] 所有 {self.max_retries} 次重试均失败。")
            if e.last_attempt and e.last_attempt.failed:
                raise e.last_attempt.exception
            else:
                raise Exception(f"HuggingFace操作失败 ({func.__name__})，但未捕获到特定异常。")
        
        except Exception as e:
            print(f"[HuggingFace] 发生不可重试错误: {e}")
            raise e 

    
    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> Dict[str, List[Dict]]:
        results = {}
        
        for keyword in keywords:
            try:
                print(f"[HuggingFace] 搜索关键词: '{keyword}'")

                datasets = await self._retry_async_thread(
                    self.hf_api.list_datasets, 
                    search=keyword, 
                    limit=max_results,
                    # size_categories=self.size_categories
                )
                
                results[keyword] = []
                for dataset in datasets:
                    results[keyword].append({
                        "id": dataset.id,
                        "title": getattr(dataset, 'title', dataset.id),
                        "description": getattr(dataset, 'description', ''),
                        "downloads": getattr(dataset, 'downloads', 0),
                        "tags": getattr(dataset, 'tags', [])
                    })
                
                print(f"[HuggingFace] 找到 {len(results[keyword])} 个数据集")
                
            except Exception as e:
                print(f"[HuggingFace] 搜索关键词 '{keyword}' 时出错 (经过重试后): {e}")
                results[keyword] = []
        
        return results
    
    # -----------------------------------------------------------------
    # vvvvvvvvvvvv   修改后的 download_dataset 方法   vvvvvvvvvvvv
    # -----------------------------------------------------------------
    async def download_dataset(self, dataset_id: str, save_dir: str) -> str | None:
        try:
            print(f"[HuggingFace] 开始下载数据集: {dataset_id}")
            dataset_dir = os.path.join(save_dir, dataset_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            config_to_load = None
            try:
                print(f"[HuggingFace] 正在检查 {dataset_id} 的配置...")
                
                configs = await self._retry_async_thread(
                    get_dataset_config_names,
                    path=dataset_id,
                    # base_url=self.hf_endpoint,  # 显式传入镜像端点
                    # token=self.hf_api.token 
                )
                
                if configs:
                    config_to_load = configs[0] 
                    print(f"[HuggingFace] 数据集 {dataset_id} 有 {len(configs)} 个配置. 自动选择第一个: {config_to_load}")
                else:
                    print(f"[HuggingFace] 数据集 {dataset_id} 没有特定的配置.")
            
            except Exception as e:
                print(f"[HuggingFace] 检查配置时出错 (将跳过配置检查，直接下载): {e}")
                config_to_load = None
            
            # --- 核心修改：使用 snapshot_download 替换 load_dataset ---
            print(f"[HuggingFace] 开始下载 {dataset_id} 的所有文件...")
            
            returned_path = await self._retry_async_thread(
                snapshot_download, 
                repo_id=dataset_id,
                local_dir=dataset_dir,
                repo_type="dataset",             # 明确告知是数据集
                force_download=True,           # 相当于 download_mode="force_redownload"
                local_dir_use_symlinks=False,  # 推荐设置，避免Windows或跨设备问题
                endpoint=self.hf_endpoint      # 显式传入镜像端点，确保重试时使用镜像
                # token=self.hf_api.token      # 如果需要私有库，可以传入
            )
            # --- 修改结束 ---
            
            # 如果禁用了缓存，下载完成后清理临时缓存目录
            if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
                try:
                    if os.path.exists(self._temp_cache_dir):
                        shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                        print(f"[HuggingFace] 已清理临时缓存目录: {self._temp_cache_dir}")
                except Exception as e:
                    print(f"[HuggingFace] 清理临时缓存目录时出错: {e} (可忽略)")
            
            config_str = f"(配置: {config_to_load})" if config_to_load else "(默认配置)"
            print(f"[HuggingFace] 数据集 {dataset_id} {config_str} *文件*下载成功，保存至 {returned_path}")
            return returned_path
            
        except Exception as e:
            print(f"[HuggingFace] 下载数据集 {dataset_id} 失败 (经过重试后): {e}")
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
            print(f"[Kaggle] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
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
            print(f"[Kaggle] 缓存目录已设置为: {kaggle_cache} (避免占用系统盘)")
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
            print(f"[Kaggle] 使用默认缓存目录: {default_cache}")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            self.api = KaggleApi()
            self.api.authenticate()
            print("[Kaggle] 已使用 KaggleApi 进行认证。")
        except Exception as e:
            print(f"[Kaggle] KaggleApi 初始化/认证失败: {e}. 请配置 ~/.kaggle/kaggle.json 或设置 KAGGLE_USERNAME/KAGGLE_KEY。将无法使用 Kaggle API。")

    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> List[str]:
        if not self.api:
            print("[Kaggle] 未初始化 KaggleApi，跳过 Kaggle 搜索。")
            return []
        refs: List[str] = []
        try:
            # KaggleApi 不支持并发调用，这里串行合并结果
            for kw in keywords:
                try:
                    items = await asyncio.to_thread(self.api.dataset_list, search=kw)
                    for it in (items or [])[:max_results]:
                        # it.ref 格式如 owner/slug
                        ref = getattr(it, 'ref', None) or f"{getattr(it, 'ownerSlug', '')}/{getattr(it, 'datasetSlug', '')}"
                        if ref and '/' in ref:
                            refs.append(ref)
                except Exception as e:
                    print(f"[Kaggle] 搜索 '{kw}' 出错: {e}")
        except Exception as e:
            print(f"[Kaggle] 搜索失败: {e}")
        # 去重并限量
        dedup = list(dict.fromkeys(refs))[:max_results]
        print(f"[Kaggle] API 搜索汇总结果: {len(dedup)} 个候选")
        return dedup

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
            print(f"[Kaggle] 无法解析数据集标识: {dataset_identifier}")
            return None
        
        # 优先使用 KaggleApi（直接下载到指定目录，避免系统盘缓存）
        if self.api:
            try:
                print(f"[Kaggle] 使用 KaggleApi 下载: {ref} (直接下载到 {save_dir}，避免系统盘缓存)")
                await asyncio.to_thread(self.api.dataset_download_files, ref, path=save_dir, unzip=True, quiet=True)
                print(f"[Kaggle] 下载完成并解压至: {save_dir}")
                return save_dir
            except Exception as e:
                print(f"[Kaggle] API 下载失败: {e}")
        else:
            print("[Kaggle] 未初始化 KaggleApi，尝试使用 kagglehub。")
        
        # 如果 KaggleApi 失败或未初始化，尝试 kagglehub
        # 注意：kagglehub 可能会在缓存目录留下文件，但我们已经设置了环境变量
        try:
            import kagglehub  # type: ignore
            print(f"[Kaggle] 使用 kagglehub 下载: {ref}")
            path = await asyncio.to_thread(kagglehub.dataset_download, ref)
            if path and os.path.exists(path):
                print(f"[Kaggle] kagglehub 下载完成: {path}")
                # 如果 kagglehub 下载的路径不在 save_dir 中，尝试将文件移动到指定目录
                # 这样可以避免在缓存目录留下文件
                if os.path.abspath(path) != os.path.abspath(save_dir):
                    try:
                        # 如果是文件，移动到 save_dir；如果是目录，复制内容
                        if os.path.isfile(path):
                            dest_path = os.path.join(save_dir, os.path.basename(path))
                            shutil.move(path, dest_path)
                            print(f"[Kaggle] 已移动文件到指定目录: {dest_path}")
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
                            print(f"[Kaggle] 已复制内容到指定目录: {save_dir}")
                            # 如果禁用了缓存，删除原始缓存目录
                            if self.disable_cache:
                                try:
                                    shutil.rmtree(path, ignore_errors=True)
                                    print(f"[Kaggle] 已清理 kagglehub 缓存目录: {path}")
                                except Exception as e:
                                    print(f"[Kaggle] 清理缓存目录时出错: {e} (可忽略)")
                            return save_dir
                    except Exception as move_e:
                        print(f"[Kaggle] 移动/复制文件时出错: {move_e}，返回原始路径")
                return path
            print("[Kaggle] kagglehub 返回无效路径。")
        except Exception as e:
            print(f"[Kaggle] kagglehub 失败或未安装: {e}")
        
        # 如果禁用了缓存，清理临时缓存目录
        if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
            try:
                if os.path.exists(self._temp_cache_dir):
                    shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                    print(f"[Kaggle] 已清理临时缓存目录: {self._temp_cache_dir}")
            except Exception as e:
                print(f"[Kaggle] 清理临时缓存目录时出错: {e} (可忽略)")
        
        # 最后兜底失败
        return None

class PaddleDatasetManager:

    def __init__(self, search_engine: str = "tavily"):
        self.search_engine = search_engine
        print(f"[Paddle] 初始化 (search_engine={self.search_engine})")

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
                print(f"[Paddle] 搜索 '{kw}' 出错: {e}")
        dedup = list(dict.fromkeys(urls))[:max_results]
        print(f"[Paddle] 搜索汇总结果: {len(dedup)} 个候选")
        return dedup

    async def try_download(self, page: Page, dataset_page_url: str, save_dir: str) -> str | None:
        os.makedirs(save_dir, exist_ok=True)
        try:
            content = await ToolManager._read_with_jina_reader(dataset_page_url)
            urls = content.get("urls", []) if content else []
            candidates = [u for u in urls if any(u.lower().endswith(ext) for ext in [".zip", ".csv", ".tar", ".gz", ".parquet"])]
            candidates = list(dict.fromkeys(candidates))
            print(f"[Paddle] 页面解析得到 {len(candidates)} 个下载候选链接")
            for u in candidates:
                path = await ToolManager.download_file(page, u, save_dir)
                if path:
                    return path
        except Exception as e:
            print(f"[Paddle] 解析页面失败: {e}")
        return None
class ToolManager:
    @staticmethod
    async def search_web(query: str, search_engine: str = "tavily") -> str:

        if isinstance(query, (list, tuple)):
            query = ", ".join([str(x) for x in query if x])
        elif not isinstance(query, str):
            query = str(query)
        
        print(f"[Search] 使用 {search_engine.upper()} 搜索: '{query}'")
 
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
            print("[Tavily] API Key 未设置，回退到 DuckDuckGo")
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
                    print("[Tavily] 无结果，回退到 DuckDuckGo")
                    return await ToolManager._duckduckgo_search(query)
                
                formatted = [
                    f"标题: {item.get('title', '无标题')}\n"
                    f"URL: {item.get('url', '')}\n"
                    f"摘要: {item.get('content', '')}\n---"
                    for item in results
                ]
                print(f"[Tavily] 搜索完成，找到 {len(results)} 个结果")
                return "\n".join(formatted)
        except Exception as e:
            print(f"[Tavily] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)
    
    @staticmethod
    async def _duckduckgo_search(query: str) -> str:
        """DuckDuckGo 搜索"""
        try:
            search_tool = DuckDuckGoSearchRun()
            result_text = await asyncio.to_thread(search_tool.run, query)
            print(f"[DuckDuckGo] 搜索完成")
            return result_text
        except Exception as e:
            print(f"[DuckDuckGo] 搜索错误: {e}")
            return ""
    
    @staticmethod
    async def _jina_search(query: str) -> str:

        try:
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"https://s.jina.ai/{encoded_query}"
            
            print(f"[Jina Search] 搜索查询: {query}")
            
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
                            print(f"[Jina Search] 搜索完成，找到 {len(formatted)} 个结果")
                            return "\n".join(formatted)
                except:
                    text_content = resp.text
                    if text_content:
                        print(f"[Jina Search] 搜索完成（文本模式）")
                        return text_content[:15000]  
                
                print("[Jina Search] 无搜索结果，回退到 DuckDuckGo")
                return await ToolManager._duckduckgo_search(query)
                
        except Exception as e:
            print(f"[Jina Search] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)

    @staticmethod
    async def read_web_page(page: Page, url: str, use_jina_reader: bool = False) -> Dict[str, Any]:

        if use_jina_reader:
            return await ToolManager._read_with_jina_reader(url)

        print(f"[Playwright] 正在读取网页: {url}")
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
            print(f"[Playwright] 网页读取成功: {url}")
            return {"urls": urls, "text": text_content}
        except PlaywrightError as e:
            print(f"[Playwright] 读取网页时出错: {e}")
            return {"urls": [], "text": f"错误: 无法访问页面 {url} ({e})"}
        except Exception as e:
            print(f"读取网页时发生未知错误: {e}")
            return {"urls": [], "text": f"错误: 读取页面 {url} 时发生未知错误。"}
    
    @staticmethod
    async def _read_with_jina_reader(url: str) -> Dict[str, Any]:

        print(f"[Jina Reader] 正在提取网页: {url}")
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
                    print(f"[Jina Reader] 警告: {warning}")
                    if "blocked" in warning.lower() or "403" in warning or "forbidden" in warning.lower():
                        print(f"[Jina Reader] 网页被封禁，无法提取内容")
                        return {
                            "urls": [],
                            "text": f"无法访问该页面: {warning}",
                            "structured_content": structured_content
                        }
                

                urls = ToolManager._extract_urls_from_markdown(markdown_content)
                
                print(f"[Jina Reader] 提取成功: {len(markdown_content)} 字符, {len(urls)} 个链接")
                
                return {
                    "urls": urls,
                    "text": markdown_content,
                    "structured_content": structured_content
                }
                    
        except httpx.HTTPStatusError as e:
            print(f"[Jina Reader] HTTP错误 {e.response.status_code}: {e}")
            return {
                "urls": [],
                "text": f"HTTP错误: {e.response.status_code}",
                "structured_content": None
            }
        except Exception as e:
            print(f"[Jina Reader] 提取失败: {e}")
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
        print(f"[Playwright] 准备从 {url} 下载文件")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            download_page = await page.context.new_page()
            async with download_page.expect_download(timeout=12000) as download_info: 
                try:
                    await download_page.goto(url, timeout=6000)
                except PlaywrightError as e:
                    if "Download is starting" in str(e) or "navigation" in str(e):
                        print(f"下载已通过导航或重定向触发。")
                        pass
                    else:
                        raise e
            download = await download_info.value
            try:
                await download_page.close()
            except Exception as close_e:
                print(f"关闭下载页面时出错（可忽略）: {close_e}")
            suggested_filename = download.suggested_filename
            save_path = os.path.join(save_dir, suggested_filename)
            print(f"文件 '{suggested_filename}' 正在保存中...")
            temp_file_path = await download.path()
            if not temp_file_path:
                    print(f"[Playwright] 下载失败，未能获取临时文件路径。")
                    await download.delete() 
                    return None
            shutil.move(temp_file_path, save_path)
            print(f"[Playwright] 下载完成: {save_path}")
            return save_path
        except Exception as e:
            print(f" [Playwright] 下载过程中发生意外错误 ({url}): {e}")
            try:
                await download_page.close()
            except:
                pass
            return None

class DownloadMethodDecisionAgent(BaseAgent):
    """下载方法决策器 - 决定使用哪种方法下载数据"""
    async def execute(self, state: WebCrawlState, logger: LogManager, current_objective: str, search_keywords: str) -> Dict[str, Any]:
        print("\n--- 下载方法决策器 ---")
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
            print(f"下载方法决策: {decision.get('method')} - {decision.get('reasoning')}")
            logger.log_data("download_method_decision_parsed", decision, is_json=True)
            return decision
        except Exception as e:
            print(f"解析下载方法决策时出错: {e}\n原始响应: {response.content}")
            return {"method": "web_crawl", "reasoning": "解析失败，使用默认的web爬取方法", "keywords_for_hf": [], "fallback_method": "huggingface"}

class HuggingFaceDecisionAgent(BaseAgent):
    
    async def execute(self, search_results: Dict[str, List[Dict]], objective: str, logger: LogManager, message: str = "") -> str | None:
        print("\n--- HuggingFace 决策器 ---")
        
        if not search_results or all(not v for v in search_results.values()):
            print("[HuggingFace Decision] 搜索结果为空，无法决策。")
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
                print(f"[HuggingFace Decision] 决策: {selected_id}. 原因: {decision.get('reasoning')}")
                return selected_id
            else:
                print(f"[HuggingFace Decision] 决策: 无合适的数据集。原因: {decision.get('reasoning')}")
                return None
        except Exception as e:
            print(f"[HuggingFace Decision] 解析决策时出错: {e}\n原始响应: {response.content}")
            return None
        
class TaskDecomposer(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager) -> WebCrawlState:
        print("\n--- decomposer ---")
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
            print(f"任务计划已生成，包含 {len(state.sub_tasks)} 个步骤。")
            logger.log_data("1_decomposer_parsed_plan", plan, is_json=True)
        except Exception as e:
            print(f"解析任务计划时出错: {e}\n原始响应: {response.content}")
        return state

class QueryGeneratorAgent(BaseAgent):
    """查询生成器 - 为RAG检索生成多样的英文查询"""
    async def execute(self, objective: str, message: str, logger: LogManager) -> List[str]:
        print("\n--- 查询生成器 ---")
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
                print(f"生成了 {len(queries)} 个检索查询: {queries}")
                logger.log_data("query_generator_parsed", queries, is_json=True)
                return queries
            else:
                print(f"查询生成格式错误，返回空列表")
                return []
        except Exception as e:
            print(f"解析查询生成响应时出错: {e}\n原始响应: {response.content}")
            return []

class SummaryAgent(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, research_objective: str, query_generator: QueryGeneratorAgent = None) -> WebCrawlState:
        print("\n--- 总结与规划（使用RAG增强，多次查询多次生成） ---")
        
        # 收集所有生成的新子任务
        all_new_sub_tasks = []
        all_summaries = []
        
        # [--- 使用 RAG 获取精炼的相关内容 ---]
        if state.enable_rag and state.rag_manager:
            print("[Summary] 使用 RAG 检索相关内容...")
            
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
                    print(f"[Summary] 查询生成失败: {e}，使用单一查询")
            
            # 如果没有生成查询，使用原始objective作为单查询
            if not queries or len(queries) == 0:
                queries = [research_objective]
            
            print(f"[Summary] 将对 {len(queries)} 个查询分别检索并生成子任务")
            
            # 对每个查询分别检索并生成子任务
            for query_idx, query in enumerate(queries, 1):
                print(f"\n[Summary] === 处理查询 {query_idx}/{len(queries)}: {query[:50]}... ===")
                
                # 获取当前查询的上下文
                relevant_context = await state.rag_manager.get_context_for_single_query(
                    query=query,
                    max_chars=18000
                )
                
                if not relevant_context:
                    print(f"[Summary] 查询 {query_idx} 检索结果为空，跳过")
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
                        print(f"[Summary] 查询 {query_idx} 生成了 {len(new_tasks)} 个新子任务")
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
                    print(f"[Summary] 查询 {query_idx} 解析响应时出错: {e}\n原始响应: {response.content}")
            
            # 记录RAG统计
            rag_stats = state.rag_manager.get_statistics()
            print(f"[Summary] RAG统计: {rag_stats}")
            logger.log_data("rag_statistics", rag_stats, is_json=True)
            
        else:
            # RAG未启用，使用传统方法（单次生成）
            print("[Summary] RAG 未启用，使用传统方法...")
            all_text = "\n\n---\n\n".join([data['text_content'] for data in state.crawled_data if 'text_content' in data])
            relevant_context = all_text[:18000] if all_text else ""
            
            if not relevant_context:
                print("研究阶段未能收集到任何文本内容，无法生成新任务。")
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
                print(f"解析总结规划响应时出错: {e}\n原始响应: {response.content}")
        
        # 将所有结果合并到research_summary
        state.research_summary = {
            "new_sub_tasks": all_new_sub_tasks,
            "summary": "\n".join(all_summaries) if all_summaries else "No summary"
        }
        
        print(f"\n总结与规划完成:")
        print(f"  - 总共生成了 {len(all_new_sub_tasks)} 个新的下载任务")
        if all_summaries:
            summary_preview = "\n".join(all_summaries)[:300]
            print(f"  - 摘要预览: {summary_preview}..." if len("\n".join(all_summaries)) > 300 else f"  - 摘要: {summary_preview}")
        
        return state

# --- Agent ---
class URLFilter(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, is_research: bool = False, **kwargs) -> WebCrawlState:
        print("\n--- flitter ---")
        if not state.search_results_text:
            print("没有搜索结果文本可供筛选。")
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
            print(f"URL筛选完成。待爬取栈: {state.url_queue}")
            logger.log_data("3_url_filter_parsed_output", result, is_json=True)
        except Exception as e:
            print(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
        return state

# --- Agent---
class WebPageReader(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, page: Page, url: str, current_objective: str, is_research: bool = False) -> Dict[str, Any]:
        print(f"\n--- web_reader (目标: {current_objective}) ---")
        print(f"--- 正在分析 URL: {url} ---")
        
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
            print(f"页面分析完毕。计划行动: {action_plan.get('action')}, 描述: {action_plan.get('description')}")
            if action_plan.get('action') == 'download':
                 print(f"发现 {len(action_plan.get('urls', []))} 个下载链接。")
            logger.log_data(f"{safe_url_filename}_parsed_output", action_plan, is_json=True)
            return action_plan
        except Exception as e:
            print(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
            return {"action": "dead_end", "description": "Failed to parse LLM response."}


class Executor:

    def __init__(self):
        pass
    
    async def execute(self, state: WebCrawlState, action_plan: Dict[str, Any], source_url: str, page: Page, current_task_type: str) -> WebCrawlState:
        print(f"\n--- executor (任务类型: {current_task_type}) ---")
        action = action_plan.get("action")
        
        if action == "download":
            if current_task_type != "download":
                print(f"在 '{current_task_type}' 阶段发现下载链接，已忽略。URL: {action_plan.get('urls', 'N/A')}")
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
                print(f"执行成功: 将新URL入栈: {full_url}")

        elif action == "dead_end":
            print("执行: 到达死胡同，无操作。")
        
        return state
    
    async def _execute_web_download(self, state: WebCrawlState, action_plan: Dict[str, Any], source_url: str, page: Page) -> bool:
        """执行网页下载（支持多文件，包含下载链接检查）"""
        urls_to_download = action_plan.get("urls", [])
        if not urls_to_download: 
            print("网页下载失败：LLM未在action_plan中提供 'urls' 列表。")
            return False
        
        download_succeeded_at_least_once = False
        
        for url_to_download in urls_to_download:
            if not url_to_download or not isinstance(url_to_download, str):
                print(f"跳过无效的下载条目: {url_to_download}")
                continue
                    
            try:
                full_url = urljoin(source_url, url_to_download)
                
                # [---检查是否是文件下载链接 ---]
                print(f"\n[检查] 正在验证 URL 是否为下载链接: {full_url}")
                check_result = await ToolManager.check_if_download_link(full_url)
                
                print(f"[检查结果] 是否为下载链接: {check_result['is_download']}")
                print(f"[检查结果] 原因: {check_result['reason']}")
                if check_result.get('content_type'):
                    print(f"[检查结果] Content-Type: {check_result['content_type']}")
                if check_result.get('filename'):
                    print(f"[检查结果] 文件名: {check_result['filename']}")

                if check_result['is_download'] == False and "HTML 页面" in check_result['reason']:
                    print(f"跳过非文件链接: {full_url}")
                    continue

                if check_result['is_download'] or "无法确定" in check_result['reason']:
                    print(f"✓ 确认为文件下载链接，开始下载...")

                    final_save_path = await ToolManager.download_file(page, full_url, save_dir=state.download_dir) 
                    
                    if final_save_path:
                        state.crawled_data.append({
                            "source_url": full_url,
                            "local_path": final_save_path,
                            "type": "file",
                            "check_result": check_result
                        })
                        print(f"网页下载成功: 已下载文件到 {final_save_path}")
                        download_succeeded_at_least_once = True 
                    else:
                        print(f"网页下载失败: 从 {full_url} 下载文件失败。")
                        
            except Exception as e:
                full_url_str = "N/A"
                try:
                    full_url_str = urljoin(source_url, str(url_to_download))
                except Exception:
                    pass
                print(f"下载 {url_to_download} (Full: {full_url_str}) 时发生异常: {e}")
        
        return download_succeeded_at_least_once
    
class Supervisor(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, current_objective: str, is_research: bool = False) -> WebCrawlState:
        print("\n--- supervisor ---")
        state.current_cycle += 1
        # Research 阶段允许更多循环次数
        max_cycles = state.max_crawl_cycles_for_research if is_research else state.max_crawl_cycles_per_task
        if not state.url_queue or state.current_cycle >= max_cycles:
            print(f"已完成 {state.current_cycle} 个循环 (最大: {max_cycles})")
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
                 temp_base_dir: str = None):
        """
        初始化 WebCrawlOrchestrator
        
        Args:
            disable_cache: 如果为 True，将完全禁用 HuggingFace 和 Kaggle 的缓存，
                           使用临时目录并在下载后自动清理，避免占用任何磁盘空间。
                           也可以通过环境变量 DF_DISABLE_CACHE=true 来启用。
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.download_dir = download_dir
        self.max_crawl_cycles_per_task = max_crawl_cycles_per_task
        self.max_crawl_cycles_for_research = max_crawl_cycles_for_research
        self.search_engine = search_engine
        self.use_jina_reader = use_jina_reader
        self.enable_rag = enable_rag
        self.concurrent_pages = concurrent_pages
        os.makedirs(self.download_dir, exist_ok=True)
        
        # 统一控制临时目录，避免写入系统 /tmp
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir or os.path.join(self.download_dir, ".tmp")
        os.makedirs(self.temp_base_dir, exist_ok=True)
        os.environ.setdefault("TMPDIR", self.temp_base_dir)

        # 检查是否禁用缓存（优先使用参数，其次使用环境变量）
        if not disable_cache:
            disable_cache = os.getenv("DF_DISABLE_CACHE", "false").lower() in ("true", "1", "yes")
        
        print(f"[Orchestrator] 配置:")
        print(f"  - 模型: {model_name}")
        print(f"  - 搜索引擎: {search_engine.upper()}")
        print(f"  - Jina Reader: {'启用' if use_jina_reader else '禁用'}")
        print(f"  - RAG 增强: {'启用' if enable_rag else '禁用'}")
        print(f"  - 并行页面数: {concurrent_pages}")
        print(f"  - 禁用缓存: {'是 (下载后自动清理临时文件)' if disable_cache else '否 (缓存将保存在项目目录)'}")
        
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
        self.hf_manager = HuggingFaceDatasetManager(max_retries=3, retry_delay=5, size_categories=dataset_size_categories, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # Kaggle / Paddle 管理器
        self.kaggle_manager = KaggleDatasetManager(search_engine=self.search_engine, cache_dir=cache_base_dir, disable_cache=disable_cache, temp_base_dir=self.temp_base_dir)
        # self.paddle_manager = PaddleDatasetManager(search_engine=self.search_engine)  # 已注释：Paddle数据集获取功能
        
        # [--- QueryGenerator Agent 初始化 ---]
        self.query_generator = QueryGeneratorAgent(model_name=self.model_name, api_base_url=self.api_base_url, api_key=self.api_key)
        
        # [--- RAG 管理器初始化 ---]
        self.rag_manager = None
        if enable_rag:
            print("[Orchestrator] 初始化 RAG 管理器...")
            self.rag_manager = RAGManager(
                api_base_url=self.api_base_url,
                api_key=self.api_key,
                persist_directory=os.path.join(self.download_dir, "rag_db")
            )
        
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
            print(f"[并行处理] 处理URL时出错 ({url}): {e}")
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
                print(f"[并行处理] URL处理异常 ({urls[i]}): {result}")
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
        print("启动多阶段数据爬取流程...")
        print(f"搜索引擎: {self.search_engine.upper()}")
        print(f"Jina Reader: {'启用' if self.use_jina_reader else '禁用'}")
        print(f"RAG 增强: {'启用' if self.rag_manager else '禁用'}")
        
        state = WebCrawlState(
            initial_request=initial_request, 
            download_dir=self.download_dir,
            search_engine=self.search_engine,  # [--- 传入搜索引擎选择 ---]
            use_jina_reader=self.use_jina_reader,  # [--- 传入 Jina Reader 选项 ---]
            rag_manager=self.rag_manager,  # [--- 传入 RAG 管理器 ---]
            enable_rag=self.enable_rag,  # [--- 传入 RAG 启用状态 ---]
            max_crawl_cycles_per_task=self.max_crawl_cycles_per_task,
            max_crawl_cycles_for_research=self.max_crawl_cycles_for_research
        )
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
                    print("未能生成任何子任务，流程终止。")
                    return

                while state.sub_tasks:
                    current_task = state.sub_tasks.pop(0)
                    task_type = current_task.get("type")
                    task_objective = current_task.get("objective")
                    search_keywords = current_task.get("search_keywords", task_objective)

                    print(f"\n\n{'='*50}\n开始执行子任务: [{task_type.upper()}] - {task_objective}\n{'='*50}")
                    state.reset_for_new_task()
                    if task_type == "download" and state.research_summary.get("new_sub_tasks"):
                        new_tasks = state.research_summary.pop("new_sub_tasks", [])
                        if new_tasks:
                            print(f"根据研究总结，将泛化下载任务替换为 {len(new_tasks)} 个具体任务。")
                            state.sub_tasks = new_tasks + state.sub_tasks
                            continue
                        else:
                            print(f"研究阶段未发现具体下载目标，执行默认的下载任务作为兜底。") 

                    download_method = "huggingface"
                    hf_keywords = []
                    fallback_method = "web_crawl"
                    
                    if task_type == "download":
                        print("正在获取 HuggingFace 搜索关键词（默认优先HF）...")
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
                        print(f"HuggingFace搜索关键词: {hf_keywords}")

                    download_success = False
                    
                    # 始终先尝试 HuggingFace："搜索 -> 决策 -> 下载"
                    if task_type == "download":
                        print("优先尝试使用 HuggingFace 方法下载...")
                        selected_id = None
                        hf_search_results = {}
                        try:
                            # 1. Search
                            print(f"[HuggingFace] 正在搜索关键词: {hf_keywords}")
                            hf_search_results = await self.hf_manager.search_datasets(hf_keywords, max_results=5)
                            
                            # 2. Decide
                            selected_id = await self.hf_decision_agent.execute(
                                hf_search_results, task_objective, self.logger, message=getattr(state, "user_message", "")
                            )
                            
                            # 3. Download
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
                                        "dataset_info": dataset_info
                                    })
                                    print(f"[HuggingFace] 数据集下载成功: {selected_id}")
                                    download_success = True
                                else:
                                    print(f"[HuggingFace] LLM选择的数据集 {selected_id} 下载失败。")
                            
                        except Exception as e:
                            print(f"[HuggingFace] HF下载流程发生意外错误: {e}")
                            download_success = False
                        if download_success:
                            state.download_successful_for_current_task = True

                    # 若 HF 未成功，尝试 Kaggle
                    if task_type == "download" and not state.download_successful_for_current_task:
                        try:
                            print("尝试使用 Kaggle 源下载...")
                            kaggle_urls = await self.kaggle_manager.search_datasets(hf_keywords or [task_objective], max_results=5)
                            if kaggle_urls:
                                dl_page = await context.new_page()
                                try:
                                    for ds_url in kaggle_urls:
                                        print(f"[Kaggle] 候选: {ds_url}")
                                        save_dir = os.path.join(state.download_dir, "kaggle_datasets")
                                        saved = await self.kaggle_manager.try_download(dl_page, ds_url, save_dir)
                                        if saved:
                                            state.crawled_data.append({
                                                "source_url": ds_url,
                                                "local_path": saved,
                                                "type": "kaggle_dataset"
                                            })
                                            print(f"[Kaggle] 下载成功: {saved}")
                                            state.download_successful_for_current_task = True
                                            break
                                finally:
                                    await dl_page.close()
                            else:
                                print("[Kaggle] 未找到候选数据集。")
                        except Exception as e:
                            print(f"[Kaggle] 下载流程发生意外错误: {e}")

                    # 若 Kaggle 未成功，尝试 Paddle（已注释）
                    # if task_type == "download" and not state.download_successful_for_current_task:
                    #     try:
                    #         print("尝试使用 Paddle 源下载...")
                    #         paddle_urls = await self.paddle_manager.search_datasets(hf_keywords or [task_objective], max_results=5)
                    #         if paddle_urls:
                    #             dl_page = await context.new_page()
                    #             try:
                    #                 for ds_url in paddle_urls:
                    #                     print(f"[Paddle] 候选: {ds_url}")
                    #                     save_dir = os.path.join(state.download_dir, "paddle_datasets")
                    #                     saved = await self.paddle_manager.try_download(dl_page, ds_url, save_dir)
                    #                     if saved:
                    #                         state.crawled_data.append({
                    #                             "source_url": ds_url,
                    #                             "local_path": saved,
                    #                             "type": "paddle_dataset"
                    #                         })
                    #                         print(f"[Paddle] 下载成功: {saved}")
                    #                         state.download_successful_for_current_task = True
                    #                         break
                    #             finally:
                    #                 await dl_page.close()
                    #         else:
                    #             print("[Paddle] 未找到候选数据集。")
                    #     except Exception as e:
                    #         print(f"[Paddle] 下载流程发生意外错误: {e}")
                    
                    # --- Web Crawl Logic ---
                    if task_type == "research" or (task_type == "download" and not state.download_successful_for_current_task):
                        
                        is_research_phase = (task_type == "research")
                        
                        print(f"使用关键词进行搜索: '{search_keywords}'")
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
                                print(f"Research阶段：已直接加入 {len(direct_urls)} 个URL到待爬取队列（跳过筛选）。")
                            except Exception as e:
                                print(f"Research阶段直接提取URL时出错: {e}")
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
                            
                            print(f"\n[并行处理] 开始处理 {len(batch_urls)} 个网页...")
                            
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
                                print(f"子任务 '{task_objective}' 的下载目标已完成，提前结束爬取循环。")
                                break
                            print(f"[并行处理] 批次完成，剩余 {len(state.url_queue)} 个URL待处理")
                    # 已经先尝试过 HuggingFace，这里不再二次回退到 HF
                    
                    if task_type == "research":
                        state = await self.summary_agent.execute(state, self.logger, research_objective=task_objective, query_generator=self.query_generator)
                    
                    if task_type == "download":
                        if state.download_successful_for_current_task:
                            current_task['status'] = 'completed_successfully'
                        else:
                            current_task['status'] = 'failed_to_download'
                    else:
                        current_task['status'] = 'completed'

                    state.completed_sub_tasks.append(current_task)
                    print(f"子任务 [{task_type.upper()}] 完成。状态: {current_task.get('status', 'N/A')}")

            finally:
                await browser.close()

        print("\n任务执行完毕!")
        downloaded_files = [d for d in state.crawled_data if d.get('type') == 'file' or d.get('type') == 'huggingface_dataset']
        print(f"最终收集到的文件 ({len(downloaded_files)} 个): {json.dumps(downloaded_files, indent=2, ensure_ascii=False)}")
        
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
        print("错误: 请在运行脚本前设置 DF_API_KEY 环境变量。")
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
        concurrent_pages=5  # 并行处理的页面数量，可根据网络和机器性能调整（建议3-10）
    )
    await orchestrator.run(user_request)

if __name__ == "__main__":
    asyncio.run(main())