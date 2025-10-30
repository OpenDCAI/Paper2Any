
from __future__ import annotations
import os
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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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

    def __init__(self, api_base_url: str, api_key: str, persist_directory: str = "./rag_db"):
        print(f"[RAG] 初始化 RAG 管理器，存储目录: {persist_directory}")
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=api_base_url,
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.document_count = 0
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
    
    async def add_webpage_content(self, url: str, text_content: str, metadata: Dict[str, Any] = None):

        if not text_content or len(text_content.strip()) < 50:
            print(f"[RAG] 跳过内容过短的网页: {url}")
            return
        try:
            print(f"[RAG] 正在添加网页内容: {url} (长度: {len(text_content)} 字符)")
            chunks = self.text_splitter.split_text(text_content)
            print(f"[RAG] 文本已分成 {len(chunks)} 个块")
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "source_url": url,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    doc_metadata.update(metadata)
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
            if self.vectorstore is None:
                self.vectorstore = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                await asyncio.to_thread(self.vectorstore.add_documents, documents)
            self.document_count += len(documents)
            print(f"[RAG] 成功添加 {len(documents)} 个文档块，总计: {self.document_count} 块")
        except Exception as e:
            print(f"[RAG] 添加网页内容时出错 ({url}): {e}")
    
    async def get_context_for_analysis(self, objective: str, max_chars: int = 15000) -> str:
        """获取用于分析的上下文"""
        if self.vectorstore is None:
            print("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            print(f"[RAG] 正在检索与 '{objective[:50]}...' 相关的内容")
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                objective,
                k=20
            )
            context_parts = []
            total_chars = 0
            seen_urls = set()
            for doc, score in results:
                source_url = doc.metadata.get("source_url", "unknown")
                content = doc.page_content
                if source_url not in seen_urls:
                    header = f"\n--- 来源: {source_url} (相关度: {score:.4f}) ---\n"
                    context_parts.append(header)
                    total_chars += len(header)
                    seen_urls.add(source_url)
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        context_parts.append(content[:remaining] + "...[已截断]")
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
    
    def __init__(self, model_name: str = "gpt-4o"):
        api_base_url ="http://123.129.219.111:3000/v1"
        api_key = os.getenv("DF_API_KEY")
        if not api_key: raise ValueError("错误：请先设置 DF_API_KEY 环境变量！")
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

    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5, size_categories: List[str] = None):
        self.hf_endpoint = 'https://hf-mirror.com'
        self.hf_api = HfApi(endpoint=self.hf_endpoint)
        self.max_retries = max_retries
        self.retry_delay = retry_delay # seconds
        self.size_categories = size_categories  # e.g., ["n<1K", "1K<n<10K", "10K<n<100K"]
        os.environ['HF_ENDPOINT'] = self.hf_endpoint
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
                    size_categories=self.size_categories
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
                    # token=self.hf_api.token 
                )
                
                if configs:
                    config_to_load = configs[0] 
                    print(f"[HuggingFace] 数据集 {dataset_id} 有 {len(configs)} 个配置. 自动选择第一个: {config_to_load}")
                else:
                    print(f"[HuggingFace] 数据集 {dataset_id} 没有特定的配置.")
            
            except Exception as e:
                print(f"[HuggingFace] 检查配置时出错 (将尝试无配置下载): {e}")
                if "Invalid pattern" in str(e):
                    print(f"[HuggingFace] 检测到 'Invalid pattern' 错误，终止此数据集下载。")
                    raise e 
                config_to_load = None
            
            # --- 核心修改：使用 snapshot_download 替换 load_dataset ---
            print(f"[HuggingFace] 开始下载 {dataset_id} 的所有文件...")
            
            returned_path = await self._retry_async_thread(
                snapshot_download, 
                repo_id=dataset_id,
                local_dir=dataset_dir,
                repo_type="dataset",             # 明确告知是数据集
                force_download=True,           # 相当于 download_mode="force_redownload"
                local_dir_use_symlinks=False   # 推荐设置，避免Windows或跨设备问题
                # token=self.hf_api.token      # 如果需要私有库，可以传入
            )
            # --- 修改结束 ---
            
            config_str = f"(配置: {config_to_load})" if config_to_load else "(默认配置)"
            print(f"[HuggingFace] 数据集 {dataset_id} {config_str} *文件*下载成功，保存至 {returned_path}")
            return returned_path
            
        except Exception as e:
            print(f"[HuggingFace] 下载数据集 {dataset_id} 失败 (经过重试后): {e}")
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

    async def execute(self, search_results: Dict[str, List[Dict]], objective: str, logger: LogManager) -> str | None:
        print("\n--- HuggingFace 决策器 ---")
        
        if not search_results or all(not v for v in search_results.values()):
            print("[HuggingFace Decision] 搜索结果为空，无法决策。")
            return None

        system_prompt = self.prompt_gen.render("system_prompt_for_huggingface_decision")
        human_prompt = self.prompt_gen.render("task_prompt_for_huggingface_decision",
                                              objective=objective,
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
            print(f"任务计划已生成，包含 {len(state.sub_tasks)} 个步骤。")
            logger.log_data("1_decomposer_parsed_plan", plan, is_json=True)
        except Exception as e:
            print(f"解析任务计划时出错: {e}\n原始响应: {response.content}")
        return state

class SummaryAgent(BaseAgent):
    async def execute(self, state: WebCrawlState, logger: LogManager, research_objective: str) -> WebCrawlState:
        print("\n--- 总结与规划（使用RAG增强） ---")
        
        # [--- 使用 RAG 获取精炼的相关内容 ---]
        if state.enable_rag and state.rag_manager:
            print("[Summary] 使用 RAG 检索相关内容...")
            relevant_context = await state.rag_manager.get_context_for_analysis(
                objective=research_objective,
                max_chars=18000
            )
            if not relevant_context:
                print("[Summary] RAG 检索未能找到相关内容，使用原始方法...")
                all_text = "\n\n---\n\n".join([data['text_content'] for data in state.crawled_data if 'text_content' in data])
                relevant_context = all_text[:18000] if all_text else ""
            else:
                rag_stats = state.rag_manager.get_statistics()
                print(f"[Summary] RAG统计: {rag_stats}")
                logger.log_data("rag_statistics", rag_stats, is_json=True)
        else:
            print("[Summary] RAG 未启用，使用传统方法...")
            all_text = "\n\n---\n\n".join([data['text_content'] for data in state.crawled_data if 'text_content' in data])
            relevant_context = all_text[:18000] if all_text else ""
        
        if not relevant_context:
            print("研究阶段未能收集到任何文本内容，无法生成新任务。")
            return state

        system_prompt = self.prompt_gen.render("system_prompt_for_summary_agent")
        human_prompt = self.prompt_gen.render("task_prompt_for_summary_agent",
                                              objective=research_objective,
                                              context=relevant_context)
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        log_name = f"summary_and_plan_{research_objective.replace(' ', '_')}"
        logger.log_data(f"{log_name}_raw_response", response.content)
        logger.log_data(f"{log_name}_context_used", relevant_context)
        try:
            clean_response = response.content.strip().replace("```json", "").replace("```", "")
            summary_plan = json.loads(clean_response)
            state.research_summary = summary_plan
            new_task_count = len(summary_plan.get("new_sub_tasks", []))
            summary_text = summary_plan.get("summary", "无摘要")
            print(f"总结与规划完成:")
            print(f"  - 摘要: {summary_text[:200]}..." if len(summary_text) > 200 else f"  - 摘要: {summary_text}")
            print(f"  - 生成了 {new_task_count} 个新的下载任务")
            logger.log_data(f"{log_name}_parsed_plan", summary_plan, is_json=True)
        except Exception as e:
            print(f"解析总结规划响应时出错: {e}\n原始响应: {response.content}")
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
                 download_dir: str = "./downloaded_data5", 
                 dataset_size_categories: List[str] = None,
                 max_crawl_cycles_per_task: int = 5,
                 max_crawl_cycles_for_research: int = 15,
                 search_engine: str = "tavily",
                 use_jina_reader: bool=True,
                 enable_rag: bool = True,
                 concurrent_pages: int = 5):
        """
        初始化 WebCrawlOrchestrator
        """
        self.download_dir = download_dir
        self.max_crawl_cycles_per_task = max_crawl_cycles_per_task
        self.max_crawl_cycles_for_research = max_crawl_cycles_for_research
        self.search_engine = search_engine
        self.use_jina_reader = use_jina_reader
        self.enable_rag = enable_rag
        self.concurrent_pages = concurrent_pages
        os.makedirs(self.download_dir, exist_ok=True)
        
        print(f"[Orchestrator] 配置:")
        print(f"  - 搜索引擎: {search_engine.upper()}")
        print(f"  - Jina Reader: {'启用' if use_jina_reader else '禁用'}")
        print(f"  - RAG 增强: {'启用' if enable_rag else '禁用'}")
        print(f"  - 并行页面数: {concurrent_pages}")
        
        self.task_decomposer = TaskDecomposer()
        self.summary_agent = SummaryAgent()
        self.url_filter = URLFilter()
        self.web_page_reader = WebPageReader()
        self.executor = Executor()
        self.supervisor = Supervisor()
        self.download_decision_agent = DownloadMethodDecisionAgent()
        self.logger = LogManager()
        
        # *** 在 Orchestrator 中初始化 HF 工具和新 Agent ***
        self.hf_decision_agent = HuggingFaceDecisionAgent()
        self.hf_manager = HuggingFaceDatasetManager(max_retries=3, retry_delay=5, size_categories=dataset_size_categories)
        
        # [--- RAG 管理器初始化 ---]
        self.rag_manager = None
        if enable_rag:
            api_base_url = "http://123.129.219.111:3000/v1"
            api_key = os.getenv("DF_API_KEY")
            if api_key:
                print("[Orchestrator] 初始化 RAG 管理器...")
                self.rag_manager = RAGManager(
                    api_base_url=api_base_url,
                    api_key=api_key,
                    persist_directory=os.path.join(self.download_dir, "rag_db")
                )
            else:
                print("[Orchestrator] 警告: 未找到 DF_API_KEY，RAG 功能将被禁用。")
        
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

                    download_method = "web_crawl"
                    hf_keywords = []
                    fallback_method = "huggingface"
                    
                    if task_type == "download":
                        print("正在决策最佳下载方法...")
                        decision = await self.download_decision_agent.execute(
                            state, self.logger, task_objective, search_keywords
                        )
                        download_method = decision.get("method", "web_crawl")
                        hf_keywords = decision.get("keywords_for_hf", [])
                        fallback_method = decision.get("fallback_method", "huggingface")
                        print(f"决策结果: 主要方法={download_method}, 备用方法={fallback_method}")
                        if hf_keywords:
                            print(f"HuggingFace搜索关键词: {hf_keywords}")

                    download_success = False
                    
                    # [--- HF 逻辑修改点： "搜索 -> 决策 -> 下载" ---]
                    if task_type == "download" and download_method == "huggingface":
                        print("尝试使用HuggingFace方法下载...")
                        selected_id = None
                        hf_search_results = {}
                        try:
                            # 1. Search
                            print(f"[HuggingFace] 正在搜索关键词: {hf_keywords}")
                            hf_search_results = await self.hf_manager.search_datasets(hf_keywords, max_results=5)
                            
                            # 2. Decide
                            selected_id = await self.hf_decision_agent.execute(
                                hf_search_results, task_objective, self.logger
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
                        
                        # --- Fallback Logic for HF -> Web Crawl ---
                        if not download_success and fallback_method == "web_crawl":
                            print("HuggingFace下载失败，回退到网页爬取方法...")
                            download_method = "web_crawl" # 强制设为 web_crawl
                        elif download_success:
                            state.download_successful_for_current_task = True
                    
                    # --- Web Crawl Logic ---
                    if task_type == "research" or (task_type == "download" and download_method == "web_crawl" and not state.download_successful_for_current_task):
                        
                        is_research_phase = (task_type == "research")
                        
                        print(f"使用关键词进行搜索: '{search_keywords}'")
                        # 防御：关键词可能为列表
                        query_kw = ", ".join(search_keywords) if isinstance(search_keywords, (list, tuple)) else search_keywords
                        # [--- 使用 state 中的搜索引擎配置 ---]
                        search_results = await ToolManager.search_web(query_kw, search_engine=state.search_engine)
                        state.search_results_text = search_results
                        self.logger.log_data(f"2_search_results_{task_objective.replace(' ', '_')}", search_results)
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
                    # --- Fallback Logic for Web Crawl -> HuggingFace ---
                    if task_type == "download" and not state.download_successful_for_current_task and fallback_method == "huggingface":
                        print("\n网页爬取方法未能成功下载，尝试回退到HuggingFace方法...")
                        selected_id = None
                        hf_search_results = {}
                        try:
                            # 如果没有预定义的HF关键词，从任务目标中生成
                            if not hf_keywords:
                                hf_keywords = [task_objective]
                            print(f"[HuggingFace Fallback] 正在搜索关键词: {hf_keywords}")
                            hf_search_results = await self.hf_manager.search_datasets(hf_keywords, max_results=5)
                            
                            selected_id = await self.hf_decision_agent.execute(
                                hf_search_results, task_objective, self.logger
                            )
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
                                    print(f"[HuggingFace Fallback] 数据集下载成功: {selected_id}")
                                    state.download_successful_for_current_task = True
                                else:
                                    print(f"[HuggingFace Fallback] LLM选择的数据集 {selected_id} 下载失败。")
                            else:
                                print("[HuggingFace Fallback] 未找到合适的数据集。")
                        except Exception as e:
                            print(f"[HuggingFace Fallback] HF备用下载流程发生意外错误: {e}")
                    
                    if task_type == "research":
                        state = await self.summary_agent.execute(state, self.logger, research_objective=task_objective)
                    
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
    if "DF_API_KEY" not in os.environ:
        print("错误: 请在运行脚本前设置 DF_API_KEY 环境变量。")
        return
    

    user_request = "帮我找一些代码的数据集"
    
    # dataset_size_categories 可选值: ["n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "1M<n<10M", "10M<n<100M", "100M<n<1B", "n>1B"]
    # search_engine 可选值: "tavily", "duckduckgo", "jina"
    orchestrator = WebCrawlOrchestrator(
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
    os.environ.setdefault("DF_API_URL", "http://123.129.219.111:3000/v1")
    asyncio.run(main())