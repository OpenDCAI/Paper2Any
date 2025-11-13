from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from playwright.async_api import Error as PlaywrightError, Page

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import WebCrawlState
from dataflow_agent.toolkits.webatool import check_if_download_link, download_file

try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:  # pragma: no cover - optional dependency
    DuckDuckGoSearchRun = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from dataflow_agent.agentroles.datacollector import LogManager

log = get_logger(__name__)


class BaseAgent(ABC):
    _prompt_generator: Optional[PromptsTemplateGenerator] = None

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if not api_base_url:
            raise ValueError("错误：必须提供 api_base_url 参数！")
        if not api_key:
            raise ValueError("错误：必须提供 api_key 参数！")

        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            openai_api_base=api_base_url,
            openai_api_key=api_key,
            max_tokens=4096,
        )

        if BaseAgent._prompt_generator is None:
            BaseAgent._prompt_generator = PromptsTemplateGenerator(
                output_language="zh", python_modules=["prompts_repo"]
            )
        self.prompt_gen = BaseAgent._prompt_generator

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:  # pragma: no cover - interface
        pass

    def _create_messages(self, system_prompt: str, human_prompt: str) -> List:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]


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
        if search_engine.lower() == "duckduckgo":
            return await ToolManager._duckduckgo_search(query)
        return await ToolManager._tavily_search(query)

    @staticmethod
    async def _tavily_search(query: str) -> str:
        tavily_api_key = os.getenv(
            "TAVILY_API_KEY", "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"
        )
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
                        "max_results": 30,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])

                if not results:
                    log.info("[Tavily] 无结果，回退到 DuckDuckGo")
                    return await ToolManager._duckduckgo_search(query)

                formatted = [
                    "\n".join(
                        [
                            f"标题: {item.get('title', '无标题')}",
                            f"URL: {item.get('url', '')}",
                            f"摘要: {item.get('content', '')}",
                            "---",
                        ]
                    )
                    for item in results
                ]
                log.info(f"[Tavily] 搜索完成，找到 {len(results)} 个结果")
                return "\n".join(formatted)
        except Exception as e:  # pragma: no cover - 网络异常
            log.info(f"[Tavily] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)

    @staticmethod
    async def _duckduckgo_search(query: str) -> str:
        if DuckDuckGoSearchRun is None:  # pragma: no cover - 备用逻辑
            log.info("[DuckDuckGo] 依赖缺失，返回空结果")
            return ""
        try:
            search_tool = DuckDuckGoSearchRun()
            result_text = await asyncio.to_thread(search_tool.run, query)
            log.info("[DuckDuckGo] 搜索完成")
            return result_text
        except Exception as e:  # pragma: no cover - 外部依赖异常
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
                        "X-Return-Format": "markdown",
                    },
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
                                content = item.get("content", "") or item.get(
                                    "description", ""
                                )
                                formatted.append(
                                    f"标题: {title}\nURL: {url}\n摘要: {content}\n---"
                                )
                            log.info(
                                f"[Jina Search] 搜索完成，找到 {len(formatted)} 个结果"
                            )
                            return "\n".join(formatted)
                except Exception:
                    text_content = resp.text
                    if text_content:
                        log.info("[Jina Search] 搜索完成（文本模式）")
                        return text_content[:15000]

                log.info("[Jina Search] 无搜索结果，回退到 DuckDuckGo")
                return await ToolManager._duckduckgo_search(query)

        except Exception as e:  # pragma: no cover - 网络异常
            log.info(f"[Jina Search] 错误: {e}，回退到 DuckDuckGo")
            return await ToolManager._duckduckgo_search(query)

    @staticmethod
    async def read_web_page(
        page: Page, url: str, use_jina_reader: bool = False
    ) -> Dict[str, Any]:
        if use_jina_reader:
            return await ToolManager._read_with_jina_reader(url)

        log.info(f"[Playwright] 正在读取网页: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            html_content = await page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                element.decompose()
            text_content = soup.get_text(separator="\n", strip=True)
            base_url = await page.evaluate("() => document.baseURI")
            raw_urls = [
                a.get("href", "").strip()
                for a in soup.find_all("a", href=True)
                if a.get("href", "").strip()
                and not a.get("href").startswith(("javascript:", "mailto:"))
            ]
            urls = [urljoin(base_url, raw_url) for raw_url in raw_urls]
            log.info(f"[Playwright] 网页读取成功: {url}")
            return {"urls": urls, "text": text_content}
        except PlaywrightError as e:
            log.info(f"[Playwright] 读取网页时出错: {e}")
            return {"urls": [], "text": f"错误: 无法访问页面 {url} ({e})"}
        except Exception as e:  # pragma: no cover - 未知异常
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
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        ),
                    },
                )
                resp.raise_for_status()

                text_response = resp.text

                structured_content = ToolManager._parse_jina_text_format(
                    text_response, url
                )
                markdown_content = structured_content.get("markdown", "")

                warning = structured_content.get("warning", "")
                if warning:
                    log.info(f"[Jina Reader] 警告: {warning}")
                    if (
                        "blocked" in warning.lower()
                        or "403" in warning
                        or "forbidden" in warning.lower()
                    ):
                        log.info("[Jina Reader] 网页被封禁，无法提取内容")
                        return {
                            "urls": [],
                            "text": f"无法访问该页面: {warning}",
                            "structured_content": structured_content,
                        }

                urls = ToolManager._extract_urls_from_markdown(markdown_content)

                log.info(
                    f"[Jina Reader] 提取成功: {len(markdown_content)} 字符, {len(urls)} 个链接"
                )

                return {
                    "urls": urls,
                    "text": markdown_content,
                    "structured_content": structured_content,
                }

        except httpx.HTTPStatusError as e:  # pragma: no cover - HTTP 异常
            log.info(f"[Jina Reader] HTTP错误 {e.response.status_code}: {e}")
            return {
                "urls": [],
                "text": f"HTTP错误: {e.response.status_code}",
                "structured_content": None,
            }
        except Exception as e:  # pragma: no cover - 网络异常
            log.info(f"[Jina Reader] 提取失败: {e}")
            return {
                "urls": [],
                "text": f"Jina Reader 错误: {str(e)}",
                "structured_content": None,
            }

    @staticmethod
    def _parse_jina_text_format(text: str, original_url: str) -> Dict[str, Any]:
        structured = {
            "title": "",
            "url_source": original_url,
            "warning": "",
            "markdown": "",
            "url": original_url,
        }

        lines = text.split("\n")
        markdown_lines: List[str] = []

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
                i += 1
                while i < len(lines):
                    markdown_lines.append(lines[i])
                    i += 1
                break

            i += 1

        structured["markdown"] = "\n".join(markdown_lines).strip()
        return structured

    @staticmethod
    def _extract_urls_from_markdown(markdown_text: str) -> List[str]:
        urls: List[str] = []

        markdown_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        markdown_links = re.findall(markdown_link_pattern, markdown_text)
        for _text, url in markdown_links:
            if url and not url.startswith("#"):
                urls.append(url)

        plain_url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
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
            "filename": "",
        }

        common_file_extensions = [
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".bz2",
            ".xz",
            ".csv",
            ".xlsx",
            ".xls",
            ".json",
            ".xml",
            ".tsv",
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".md",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".mp4",
            ".avi",
            ".mov",
            ".mp3",
            ".wav",
            ".exe",
            ".msi",
            ".dmg",
            ".deb",
            ".rpm",
            ".parquet",
            ".arrow",
            ".h5",
            ".hdf5",
            ".pkl",
        ]

        url_lower = url.lower()
        for ext in common_file_extensions:
            if (
                url_lower.endswith(ext)
                or f"{ext}?" in url_lower
                or f"{ext}#" in url_lower
            ):
                result["is_download"] = True
                result["reason"] = f"URL包含文件扩展名: {ext}"
                result["filename"] = url.split("/")[-1].split("?")[0].split("#")[0]
                return result

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.head(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        )
                    },
                )

                content_disposition = response.headers.get("Content-Disposition", "")
                if content_disposition and "attachment" in content_disposition.lower():
                    result["is_download"] = True
                    result["reason"] = "Content-Disposition 包含 attachment"
                    if "filename=" in content_disposition:
                        filename_match = re.search(
                            r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)',
                            content_disposition,
                        )
                        if filename_match:
                            result["filename"] = filename_match.group(1).strip("'\"")
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
                    "audio/",
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

        except Exception as e:  # pragma: no cover - 网络异常
            result["reason"] = f"HEAD 请求失败: {e}，无法确定"
            return result

        result["reason"] = "无法确定是否为下载链接"
        return result

    @staticmethod
    async def download_file(page: Page, url: str, save_dir: str) -> Optional[str]:
        log.info(f"[Playwright] 准备从 {url} 下载文件")
        os.makedirs(save_dir, exist_ok=True)

        download_page = await page.context.new_page()
        try:
            async with download_page.expect_download(timeout=12000) as download_info:
                try:
                    await download_page.goto(url, timeout=60000)
                except PlaywrightError as e:
                    if "Download is starting" in str(e) or "navigation" in str(e):
                        log.info("下载已通过导航或重定向触发。")
                    else:
                        raise e
            download = await download_info.value
            try:
                await download_page.close()
            except Exception as close_e:  # pragma: no cover - best effort
                log.info(f"关闭下载页面时出错（可忽略）: {close_e}")
            suggested_filename = download.suggested_filename
            save_path = os.path.join(save_dir, suggested_filename)
            log.info(f"文件 '{suggested_filename}' 正在保存中...")
            temp_file_path = await download.path()
            if not temp_file_path:
                log.info("[Playwright] 下载失败，未能获取临时文件路径。")
                await download.delete()
                return None
            shutil.move(temp_file_path, save_path)
            log.info(f"[Playwright] 下载完成: {save_path}")
            return save_path
        except Exception as e:  # pragma: no cover - 下载异常
            log.info(f"[Playwright] 下载过程中发生意外错误 ({url}): {e}")
            try:
                await download_page.close()
            except Exception:
                pass
            return None


class QueryGeneratorAgent(BaseAgent):
    """查询生成器 - 为RAG检索生成多样的英文查询"""

    async def execute(
        self, objective: str, message: str, logger: "LogManager"
    ) -> List[str]:
        log.info("\n--- 查询生成器 ---")
        system_prompt = self.prompt_gen.render("system_prompt_for_query_generator")
        human_prompt = self.prompt_gen.render(
            "task_prompt_for_query_generator", objective=objective, message=message
        )
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("query_generator_raw_response", response.content)

        try:
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "")
            )
            queries = json.loads(clean_response)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                log.info(f"生成了 {len(queries)} 个检索查询: {queries}")
                logger.log_data("query_generator_parsed", queries, is_json=True)
                return queries
            log.info("查询生成格式错误，返回空列表")
            return []
        except Exception as e:  # pragma: no cover - JSON 解析异常
            log.info(f"解析查询生成响应时出错: {e}\n原始响应: {response.content}")
            return []


class URLSelectorAgent(BaseAgent):
    """根据研究目标挑选高优先级 URL"""

    def __init__(self, *, top_k: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.default_top_k = top_k

    async def execute(
        self,
        *,
        objective: str,
        candidate_urls: List[str],
        current_depth: int,
        max_depth: int,
        top_k: Optional[int] = None,
        logger: Optional["LogManager"] = None,
    ) -> List[str]:
        if not candidate_urls:
            return []
        top_k = top_k or self.default_top_k
        system_prompt = (
            "你是一名网络研究助手，需要根据研究目标挑选下一步访问的网页。"
            "请评估候选 URL 与研究目标的相关性、信息增量和层次覆盖，并输出一个 JSON 数组，"
            "包含不超过 top_k 个最优先访问的 URL，按优先级从高到低排列。"
        )
        human_prompt = (
            f"研究目标: {objective}\n"
            f"当前深度: {current_depth} / 最大深度: {max_depth}\n"
            f"top_k: {top_k}\n"
            "候选 URL 列表:\n"
            + "\n".join(candidate_urls)
            + "\n\n请仅返回 JSON 数组，例如: [\"url1\", \"url2\"]."
        )
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        if logger:
            logger.log_data("url_selector_raw_response", response.content)
        try:
            clean = response.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean)
            if isinstance(data, list):
                selected = [u for u in data if isinstance(u, str)]
                return selected[:top_k]
        except Exception as e:  # pragma: no cover - JSON 解析异常
            log.info(f"[URLSelector] 解析失败: {e}\n原始响应: {response.content}")
        return []


class SummaryAgent(BaseAgent):
    def _apply_download_limit_to_state(self, state: WebCrawlState) -> None:
        if not hasattr(state, "sub_tasks") or state.sub_tasks is None:
            return

        limit = getattr(state, "max_download_subtasks", None)
        if limit is None:
            return

        completed = getattr(state, "completed_download_tasks", 0) or 0
        remaining = limit - completed

        if remaining <= 0:
            downloads_removed = sum(
                1 for task in state.sub_tasks if task.get("type") == "download"
            )
            if downloads_removed:
                log.info(
                    f"[Summary] 下载子任务上限已达，移除 {downloads_removed} 个待执行的下载子任务。"
                )
            state.sub_tasks = [
                task for task in state.sub_tasks if task.get("type") != "download"
            ]
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
            log.info(
                f"[Summary] 应用下载子任务上限，裁剪 {downloads_removed} 个多余的下载子任务。"
            )
        state.sub_tasks = filtered_tasks

    async def execute(
        self,
        state: WebCrawlState,
        logger: "LogManager",
        research_objective: str,
        query_generator: Optional[QueryGeneratorAgent] = None,
    ) -> WebCrawlState:
        log.info("\n--- 总结与规划（使用RAG增强，多次查询多次生成） ---")

        all_new_sub_tasks: List[Dict[str, Any]] = []
        all_summaries: List[str] = []

        if state.enable_rag and state.rag_manager:
            log.info("[Summary] 使用 RAG 检索相关内容...")

            queries: Optional[List[str]] = None
            if query_generator:
                try:
                    queries = await query_generator.execute(
                        objective=research_objective,
                        message=getattr(state, "user_message", ""),
                        logger=logger,
                    )
                except Exception as e:  # pragma: no cover - LLM 异常
                    log.info(f"[Summary] 查询生成失败: {e}，使用单一查询")

            if not queries:
                queries = [research_objective]

            log.info(f"[Summary] 将对 {len(queries)} 个查询分别检索并生成子任务")

            for query_idx, query in enumerate(queries, 1):
                log.info(
                    f"\n[Summary] === 处理查询 {query_idx}/{len(queries)}: {query[:50]}... ==="
                )

                relevant_context = await state.rag_manager.get_context_for_single_query(
                    query=query, max_chars=18000
                )

                if not relevant_context:
                    log.info(f"[Summary] 查询 {query_idx} 检索结果为空，跳过")
                    continue

                seen_task_keys = set()
                existing_download_tasks = []

                if getattr(state, "sub_tasks", None):
                    for task in state.sub_tasks:
                        if task.get("type") == "download":
                            task_key = (
                                task.get("objective", ""),
                                task.get("search_keywords", ""),
                            )
                            if task_key not in seen_task_keys:
                                seen_task_keys.add(task_key)
                                existing_download_tasks.append(
                                    {
                                        "objective": task.get("objective", ""),
                                        "search_keywords": task.get(
                                            "search_keywords", ""
                                        ),
                                    }
                                )

                for new_task in all_new_sub_tasks:
                    task_key = (
                        new_task.get("objective", ""),
                        new_task.get("search_keywords", ""),
                    )
                    if task_key not in seen_task_keys:
                        seen_task_keys.add(task_key)
                        existing_download_tasks.append(
                            {
                                "objective": new_task.get("objective", ""),
                                "search_keywords": new_task.get(
                                    "search_keywords", ""
                                ),
                            }
                        )

                existing_subtasks_str = (
                    json.dumps(existing_download_tasks, indent=2, ensure_ascii=False)
                    if existing_download_tasks
                    else "[]"
                )

                system_prompt = self.prompt_gen.render(
                    "system_prompt_for_summary_agent"
                )
                human_prompt = self.prompt_gen.render(
                    "task_prompt_for_summary_agent",
                    objective=research_objective,
                    message=getattr(state, "user_message", ""),
                    existing_subtasks=existing_subtasks_str,
                    context=relevant_context,
                )
                messages = self._create_messages(system_prompt, human_prompt)
                response = await self.llm.ainvoke(messages)
                log_name = (
                    f"summary_query_{query_idx}_{research_objective.replace(' ', '_')}"
                )
                logger.log_data(f"{log_name}_raw_response", response.content)
                logger.log_data(f"{log_name}_query", query)
                logger.log_data(f"{log_name}_context_used", relevant_context)
                logger.log_data(
                    f"{log_name}_existing_subtasks", existing_download_tasks, is_json=True
                )

                try:
                    clean_response = (
                        response.content.strip()
                        .replace("```json", "")
                        .replace("```", "")
                    )
                    summary_plan = json.loads(clean_response)

                    new_tasks = summary_plan.get("new_sub_tasks", [])
                    summary_text = summary_plan.get("summary", "")

                    if new_tasks:
                        log.info(
                            f"[Summary] 查询 {query_idx} 生成了 {len(new_tasks)} 个新子任务"
                        )
                        all_new_sub_tasks.extend(new_tasks)

                        if getattr(state, "sub_tasks", None) is None:
                            state.sub_tasks = []

                        existing_task_keys = {
                            (
                                task.get("objective", ""),
                                task.get("search_keywords", ""),
                            )
                            for task in state.sub_tasks
                            if task.get("type") == "download"
                        }

                        for new_task in new_tasks:
                            task_key = (
                                new_task.get("objective", ""),
                                new_task.get("search_keywords", ""),
                            )
                            if task_key not in existing_task_keys:
                                state.sub_tasks.append(new_task)
                                existing_task_keys.add(task_key)

                    if summary_text:
                        all_summaries.append(
                            f"[Query {query_idx}: {query[:30]}...] {summary_text}"
                        )

                    logger.log_data(f"{log_name}_parsed_plan", summary_plan, is_json=True)
                except Exception as e:  # pragma: no cover - JSON 解析异常
                    log.info(
                        f"[Summary] 查询 {query_idx} 解析响应时出错: {e}\n原始响应: {response.content}"
                    )

            rag_stats = state.rag_manager.get_statistics()
            log.info(f"[Summary] RAG统计: {rag_stats}")
            logger.log_data("rag_statistics", rag_stats, is_json=True)

        else:
            log.info("[Summary] RAG 未启用，使用传统方法...")
            all_text = "\n\n---\n\n".join(
                [
                    data["text_content"]
                    for data in state.crawled_data
                    if "text_content" in data
                ]
            )
            relevant_context = all_text[:18000] if all_text else ""

            if not relevant_context:
                log.info("研究阶段未能收集到任何文本内容，无法生成新任务。")
                return state

            existing_download_tasks = []
            if getattr(state, "sub_tasks", None):
                for task in state.sub_tasks:
                    if task.get("type") == "download":
                        existing_download_tasks.append(
                            {
                                "objective": task.get("objective", ""),
                                "search_keywords": task.get("search_keywords", ""),
                            }
                        )
            existing_subtasks_str = (
                json.dumps(existing_download_tasks, indent=2, ensure_ascii=False)
                if existing_download_tasks
                else "[]"
            )

            system_prompt = self.prompt_gen.render(
                "system_prompt_for_summary_agent"
            )
            human_prompt = self.prompt_gen.render(
                "task_prompt_for_summary_agent",
                objective=research_objective,
                message=getattr(state, "user_message", ""),
                existing_subtasks=existing_subtasks_str,
                context=relevant_context,
            )
            messages = self._create_messages(system_prompt, human_prompt)
            response = await self.llm.ainvoke(messages)
            log_name = f"summary_and_plan_{research_objective.replace(' ', '_')}"
            logger.log_data(f"{log_name}_raw_response", response.content)
            logger.log_data(f"{log_name}_context_used", relevant_context)
            try:
                clean_response = (
                    response.content.strip().replace("```json", "").replace("```", "")
                )
                summary_plan = json.loads(clean_response)
                all_new_sub_tasks = summary_plan.get("new_sub_tasks", [])
                all_summaries = [summary_plan.get("summary", "No summary")]
                logger.log_data(
                    f"{log_name}_parsed_plan", summary_plan, is_json=True
                )
            except Exception as e:  # pragma: no cover - JSON 解析异常
                log.info(f"解析总结规划响应时出错: {e}\n原始响应: {response.content}")

        state.research_summary = {
            "new_sub_tasks": all_new_sub_tasks,
            "summary": "\n".join(all_summaries) if all_summaries else "No summary",
        }
        self._apply_download_limit_to_state(state)

        log.info("\n总结与规划完成:")
        log.info(f"  - 总共生成了 {len(all_new_sub_tasks)} 个新的下载任务")
        if all_summaries:
            summary_preview = "\n".join(all_summaries)
            log.info(
                f"  - 摘要预览: {summary_preview[:300]}..."
                if len(summary_preview) > 300
                else f"  - 摘要: {summary_preview}"
            )

        return state


class URLFilter(BaseAgent):
    async def execute(
        self, state: WebCrawlState, logger: "LogManager", is_research: bool = False, **_
    ) -> WebCrawlState:
        log.info("\n--- flitter ---")
        if not state.search_results_text:
            log.info("没有搜索结果文本可供筛选。")
            return state

        url_count_instruction = (
            "尽可能多地提取URL（至少10-15个）"
            if is_research
            else "提取最相关的URL（5-10个）"
        )

        system_prompt = self.prompt_gen.render(
            "system_prompt_for_url_filter", url_count_instruction=url_count_instruction
        )
        human_prompt = self.prompt_gen.render(
            "task_prompt_for_url_filter",
            request=state.initial_request,
            search_results=state.search_results_text,
        )
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data("3_url_filter_raw_response", response.content)
        try:
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(clean_response)
            urls = result.get("selected_urls", [])
            state.filtered_urls = urls
            state.url_queue.extend(urls)
            log.info(f"URL筛选完成。待爬取栈: {state.url_queue}")
            logger.log_data("3_url_filter_parsed_output", result, is_json=True)
        except Exception as e:  # pragma: no cover - JSON 解析异常
            log.info(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
        return state


class WebPageReader(BaseAgent):
    async def execute(
        self,
        state: WebCrawlState,
        logger: "LogManager",
        page: Page,
        url: str,
        current_objective: str,
        is_research: bool = False,
    ) -> Dict[str, Any]:
        log.info(f"\n--- web_reader (目标: {current_objective}) ---")
        log.info(f"--- 正在分析 URL: {url} ---")

        page_data = await ToolManager.read_web_page(
            page, url, use_jina_reader=state.use_jina_reader
        )
        safe_url_filename = f"4_read_page_{url.replace('://', '_').replace('/', '_')}"
        logger.log_data(safe_url_filename, page_data, is_json=True)

        text_content = page_data.get("text", "")
        if text_content:
            crawl_entry: Dict[str, Any] = {"source_url": url, "text_content": text_content}
            if "structured_content" in page_data and page_data["structured_content"]:
                crawl_entry["structured_content"] = page_data["structured_content"]
            state.crawled_data.append(crawl_entry)

            if is_research and state.enable_rag and state.rag_manager:
                await state.rag_manager.add_webpage_content(
                    url=url,
                    text_content=text_content,
                    metadata={
                        "objective": current_objective,
                        "extraction_method": (
                            "jina_reader" if state.use_jina_reader else "playwright"
                        ),
                    },
                )

        system_prompt = self.prompt_gen.render("system_prompt_for_webpage_reader")
        compact_text = page_data.get("text", "")[:16000]
        discovered_urls = page_data.get("urls", [])[:100]
        urls_block = "\n".join(discovered_urls)

        human_prompt = self.prompt_gen.render(
            "task_prompt_for_webpage_reader",
            objective=current_objective,
            urls_block=urls_block,
            text_content=compact_text,
        )
        messages = self._create_messages(system_prompt, human_prompt)
        response = await self.llm.ainvoke(messages)
        logger.log_data(f"{safe_url_filename}_raw_response", response.content)

        try:
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "")
            )
            action_plan = json.loads(clean_response)
            action_plan["discovered_urls"] = discovered_urls
            log.info(
                f"页面分析完毕。计划行动: {action_plan.get('action')}, 描述: {action_plan.get('description')}"
            )
            if action_plan.get("action") == "download":
                log.info(f"发现 {len(action_plan.get('urls', []))} 个下载链接。")
            logger.log_data(
                f"{safe_url_filename}_parsed_output", action_plan, is_json=True
            )
            return action_plan
        except Exception as e:  # pragma: no cover - JSON 解析异常
            log.info(f"解析LLM响应时出错: {e}\n原始响应: {response.content}")
            return {"action": "dead_end", "description": "Failed to parse LLM response."}


class Executor:
    def __init__(self) -> None:
        pass

    async def execute(
        self,
        state: WebCrawlState,
        action_plan: Dict[str, Any],
        source_url: str,
        page: Page,
        current_task_type: str,
    ) -> WebCrawlState:
        log.info(f"\n--- executor (任务类型: {current_task_type}) ---")
        action = action_plan.get("action")

        if action == "download":
            if current_task_type != "download":
                log.info(
                    f"在 '{current_task_type}' 阶段发现下载链接，已忽略。URL: {action_plan.get('urls', 'N/A')}"
                )
                return state

            success = await self._execute_web_download(
                state, action_plan, source_url, page
            )

            if success:
                state.download_successful_for_current_task = True

        elif action == "navigate":
            url_to_navigate = action_plan.get("url")
            if not url_to_navigate:
                return state
            full_url = urljoin(source_url, url_to_navigate)
            if full_url not in state.visited_urls and full_url not in state.url_queue:
                state.url_queue.append(full_url)
                log.info(f"执行成功: 将新URL入栈: {full_url}")

        elif action == "dead_end":
            log.info("执行: 到达死胡同，无操作。")

        return state

    async def _execute_web_download(
        self, state: WebCrawlState, action_plan: Dict[str, Any], source_url: str, page: Page
    ) -> bool:
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

                log.info(f"\n[检查] 正在验证 URL 是否为下载链接: {full_url}")
                check_result = await check_if_download_link(full_url)

                log.info(f"[检查结果] 是否为下载链接: {check_result['is_download']}")
                log.info(f"[检查结果] 原因: {check_result['reason']}")
                if check_result.get("content_type"):
                    log.info(f"[检查结果] Content-Type: {check_result['content_type']}")
                if check_result.get("filename"):
                    log.info(f"[检查结果] 文件名: {check_result['filename']}")

                if (
                    check_result["is_download"] is False
                    and "HTML 页面" in check_result["reason"]
                ):
                    log.info(f"跳过非文件链接: {full_url}")
                    continue

                if not check_result["is_download"]:
                    log.info(f"非下载链接，跳过: {full_url} ({check_result['reason']})")
                    continue

                if state.max_dataset_size:
                    content_length = 0
                    try:
                        async with httpx.AsyncClient(
                            timeout=15.0, follow_redirects=True
                        ) as client:
                            head_resp = await client.head(
                                full_url,
                                headers={
                                    "User-Agent": (
                                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36"
                                    )
                                },
                            )
                            content_length = int(head_resp.headers.get("Content-Length", 0))
                    except Exception as size_err:  # pragma: no cover - 网络异常
                        log.info(f"检查文件大小失败，将继续下载: {size_err}")
                    if content_length and content_length > state.max_dataset_size:
                        log.info(
                            f"跳过下载：文件大小 {content_length} 字节超过限制 {state.max_dataset_size} 字节 -> {full_url}"
                        )
                        continue

                log.info("✓ 确认为文件下载链接，开始下载...")

                final_save_path = await download_file(
                    page, full_url, save_dir=state.download_dir
                )

                if final_save_path:
                    state.crawled_data.append(
                        {
                            "source_url": full_url,
                            "local_path": final_save_path,
                            "type": "file",
                            "check_result": check_result,
                        }
                    )
                    log.info(f"网页下载成功: 已下载文件到 {final_save_path}")
                    download_succeeded_at_least_once = True
                else:
                    log.info(f"网页下载失败: 从 {full_url} 下载文件失败。")

            except Exception as e:  # pragma: no cover - 捕获所有下载异常
                try:
                    full_url_str = urljoin(source_url, str(url_to_download))
                except Exception:
                    full_url_str = "N/A"
                log.info(f"下载 {url_to_download} (Full: {full_url_str}) 时发生异常: {e}")

        return download_succeeded_at_least_once


class Supervisor(BaseAgent):
    async def execute(
        self,
        state: WebCrawlState,
        logger: "LogManager",
        current_objective: str,
        is_research: bool = False,
    ) -> WebCrawlState:
        log.info("\n--- supervisor ---")
        state.current_cycle += 1
        max_cycles = (
            state.max_crawl_cycles_for_research
            if is_research
            else state.max_crawl_cycles_per_task
        )
        if not state.url_queue or state.current_cycle >= max_cycles:
            log.info(f"已完成 {state.current_cycle} 个循环 (最大: {max_cycles})")
        return state


class SubTaskRefinerAgent(BaseAgent):
    async def execute(
        self, message: str, sub_tasks: List[Dict[str, Any]], logger: "LogManager"
    ) -> List[Dict[str, Any]]:
        log.info("\n--- 子任务精炼器 ---")
        system_prompt = self.prompt_gen.render("system_prompt_for_subtask_refiner")
        try:
            sub_tasks_json = json.dumps(sub_tasks, indent=2, ensure_ascii=False)
        except Exception:
            sub_tasks_json = json.dumps(
                sub_tasks,
                indent=2,
                ensure_ascii=False,
                default=lambda o: getattr(o, "__dict__", str(o)),
            )
        human_prompt = self.prompt_gen.render(
            "task_prompt_for_subtask_refiner", message=message or "", sub_tasks=sub_tasks_json
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
        except Exception as e:  # pragma: no cover - JSON 解析异常
            log.info(f"[SubTaskRefiner] 解析失败: {e}\n原始响应: {response.content}")
        return sub_tasks


class WebResearchAgent(BaseAgent):
    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        api_base_url: Optional[str],
        api_key: Optional[str],
        concurrent_pages: int = 5,
        summary_agent: Optional[SummaryAgent] = None,
        web_page_reader: Optional[WebPageReader] = None,
        executor: Optional[Executor] = None,
        supervisor: Optional[Supervisor] = None,
        query_generator: Optional[QueryGeneratorAgent] = None,
        subtask_refiner: Optional[SubTaskRefinerAgent] = None,
    ) -> None:
        super().__init__(
            model_name=model_name, api_base_url=api_base_url, api_key=api_key
        )
        init_kwargs = {
            "model_name": model_name,
            "api_base_url": api_base_url,
            "api_key": api_key,
        }
        self.concurrent_pages = concurrent_pages
        self.summary_agent = summary_agent or SummaryAgent(**init_kwargs)
        self.web_page_reader = web_page_reader or WebPageReader(**init_kwargs)
        self.executor = executor or Executor()
        self.supervisor = supervisor or Supervisor(**init_kwargs)
        self.query_generator = query_generator or QueryGeneratorAgent(**init_kwargs)
        self.subtask_refiner = subtask_refiner or SubTaskRefinerAgent(**init_kwargs)
        self.top_k_urls = 3
        self.max_tree_depth = 3
        self.url_selector = URLSelectorAgent(top_k=self.top_k_urls, **init_kwargs)

    async def execute(
        self,
        state: WebCrawlState,
        *,
        context,
        logger: "LogManager",
        objective: str,
        search_keywords: str | List[str],
    ) -> WebCrawlState:
        state.temp_data["url_tree_queue"] = []
        graph = self._build_graph(
            context=context,
            logger=logger,
            objective=objective,
            search_keywords=search_keywords,
        )
        result_state = await graph.ainvoke(state)
        return result_state if isinstance(result_state, WebCrawlState) else state

    async def process_urls_parallel(
        self,
        context,
        entries: List[Dict[str, Any]],
        state: WebCrawlState,
        task_objective: str,
        task_type: str,
        is_research_phase: bool,
        logger: "LogManager",
    ) -> List[Dict[str, Any]]:
        async def process_with_page(entry: Dict[str, Any]) -> Dict[str, Any]:
            url = entry["url"]
            page = await context.new_page()
            try:
                return await self._process_single_url(
                    page,
                    url,
                    state,
                    task_objective,
                    task_type,
                    is_research_phase,
                    logger,
                    entry,
                )
            finally:
                await page.close()

        tasks = [process_with_page(entry) for entry in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.info(f"[并行处理] URL处理异常 ({entries[i]['url']}): {result}")
                processed_results.append(
                    {
                        "success": False,
                        "url": entries[i]["url"],
                        "error": str(result),
                        "entry": entries[i],
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def _build_graph(
        self,
        *,
        context,
        logger: "LogManager",
        objective: str,
        search_keywords: str | List[str],
    ):
        sg = StateGraph(WebCrawlState)

        async def search_node(state: WebCrawlState) -> WebCrawlState:
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
            logger.log_data(
                f"2_search_results_{objective.replace(' ', '_')}", search_results
            )
            return state

        async def seed_urls_node(state: WebCrawlState) -> WebCrawlState:
            try:
                direct_urls = (
                    ToolManager._extract_urls_from_markdown(state.search_results_text)
                    if state.search_results_text
                    else []
                )
                direct_urls = direct_urls[:30]
                state.filtered_urls = direct_urls
                queue = [{"url": url, "depth": 0} for url in direct_urls]
                state.temp_data["url_tree_queue"] = queue
                state.url_queue = [item["url"] for item in queue]
                logger.log_data(
                    "3_url_extracted_directly_for_research",
                    {
                        "count": len(direct_urls),
                        "urls": direct_urls,
                    },
                    is_json=True,
                )
                log.info(
                    f"Research阶段：首层共有 {len(direct_urls)} 个候选 URL，将全部入队。"
                )
            except Exception as e:  # pragma: no cover - 解析异常
                log.info(f"Research阶段直接提取URL时出错: {e}")
            return state

        def seed_condition(state: WebCrawlState) -> str:
            queue = state.temp_data.get("url_tree_queue") or []
            return "crawl" if queue else "summary"

        async def crawl_node(state: WebCrawlState) -> WebCrawlState:
            queue: List[Dict[str, Any]] = state.temp_data.get("url_tree_queue", [])
            batch_entries: List[Dict[str, Any]] = []
            while queue and len(batch_entries) < self.concurrent_pages:
                entry = queue.pop(0)
                url = entry["url"]
                if url in state.visited_urls:
                    continue
                state.visited_urls.add(url)
                batch_entries.append(entry)

            if not batch_entries:
                state.temp_data["url_tree_queue"] = queue
                return state

            log.info(f"\n[并行处理] 开始处理 {len(batch_entries)} 个网页...")
            results = await self.process_urls_parallel(
                context,
                batch_entries,
                state,
                task_objective=objective,
                task_type="research",
                is_research_phase=True,
                logger=logger,
            )
            existing_urls = {entry["url"] for entry in queue}
            for result in results:
                if result.get("success"):
                    crawled_entry = result.get("crawled_entry")
                    if crawled_entry:
                        if "multiple_entries" in crawled_entry:
                            state.crawled_data.extend(crawled_entry["multiple_entries"])
                        else:
                            state.crawled_data.append(crawled_entry)

                    entry = result.get("entry") or {}
                    depth = entry.get("depth", 0)
                    discovered_urls = result.get("discovered_urls", [])
                    if discovered_urls and depth + 1 < self.max_tree_depth:
                        selected_children = await self.url_selector.execute(
                            objective=objective,
                            candidate_urls=discovered_urls,
                            current_depth=depth + 1,
                            max_depth=self.max_tree_depth,
                            top_k=max(self.top_k_urls, 0),
                            logger=logger,
                        )
                        if not selected_children:
                            selected_children = discovered_urls[: self.top_k_urls]
                        for child_url in selected_children:
                            if (
                                child_url not in state.visited_urls
                                and child_url not in existing_urls
                            ):
                                queue.append({"url": child_url, "depth": depth + 1})
                                existing_urls.add(child_url)

                    if result.get("download_successful"):
                        state.download_successful_for_current_task = True

            await self.supervisor.execute(
                state, logger, current_objective=objective, is_research=True
            )
            state.temp_data["url_tree_queue"] = queue
            state.url_queue = [entry["url"] for entry in queue]
            log.info(
                f"[并行处理] 批次完成，剩余 {len(queue)} 个URL待处理 (cycle={state.current_cycle})"
            )
            return state

        def crawl_condition(state: WebCrawlState) -> str:
            max_cycles = state.max_crawl_cycles_for_research
            queue = state.temp_data.get("url_tree_queue") or []
            if (
                queue
                and state.current_cycle < max_cycles
                and not state.download_successful_for_current_task
            ):
                return "crawl"
            return "summary"

        async def summary_node(state: WebCrawlState) -> WebCrawlState:
            return await self.summary_agent.execute(
                state,
                logger,
                research_objective=objective,
                query_generator=self.query_generator,
            )

        async def refine_node(state: WebCrawlState) -> WebCrawlState:
            try:
                new_tasks = state.research_summary.get("new_sub_tasks", []) if hasattr(state, "research_summary") else []
                if new_tasks:
                    refined = await self.subtask_refiner.execute(
                        message=getattr(state, "user_message", ""),
                        sub_tasks=new_tasks,
                        logger=logger,
                    )
                    state.research_summary["new_sub_tasks"] = refined
                    self.summary_agent._apply_download_limit_to_state(state)
                    log.info(f"[Research] 子任务精炼完成：{len(new_tasks)} -> {len(refined)}")
            except Exception as e:  # pragma: no cover - LLM异常
                log.info(f"[Research] 子任务精炼失败: {e}")
            return state

        sg.add_node("search", search_node)
        sg.add_node("seed_urls", seed_urls_node)
        sg.add_node("crawl", crawl_node)
        sg.add_node("summary", summary_node)
        sg.add_node("refine", refine_node)

        sg.add_edge("search", "seed_urls")
        sg.add_conditional_edges("seed_urls", seed_condition)
        sg.add_conditional_edges("crawl", crawl_condition)
        sg.add_edge("summary", "refine")

        sg.set_entry_point("search")
        return sg.compile()

    async def _process_single_url(
        self,
        page: Page,
        url: str,
        state: WebCrawlState,
        task_objective: str,
        task_type: str,
        is_research_phase: bool,
        logger: "LogManager",
        queue_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "url": url,
            "crawled_entry": None,
            "new_urls": [],
            "download_successful": False,
        }

        try:
            action_plan = await self.web_page_reader.execute(
                state,
                logger,
                page=page,
                url=url,
                current_objective=task_objective,
                is_research=is_research_phase,
            )
            discovered_urls = action_plan.get("discovered_urls", [])

            temp_state = WebCrawlState(
                initial_request=state.initial_request,
                download_dir=state.download_dir,
                search_engine=state.search_engine,
                use_jina_reader=state.use_jina_reader,
                rag_manager=state.rag_manager,
                enable_rag=state.enable_rag,
                max_crawl_cycles_per_task=state.max_crawl_cycles_per_task,
                max_crawl_cycles_for_research=state.max_crawl_cycles_for_research,
            )
            temp_state.crawled_data = []
            temp_state.url_queue = []
            temp_state.visited_urls = state.visited_urls.copy()
            temp_state.download_successful_for_current_task = False

            temp_state = await self.executor.execute(
                temp_state,
                action_plan,
                source_url=url,
                page=page,
                current_task_type=task_type,
            )

            result["success"] = True
            if temp_state.crawled_data:
                result["crawled_entry"] = (
                    temp_state.crawled_data[0]
                    if len(temp_state.crawled_data) == 1
                    else {"multiple_entries": temp_state.crawled_data}
                )
            result["new_urls"] = temp_state.url_queue
            result["download_successful"] = temp_state.download_successful_for_current_task
            result["discovered_urls"] = discovered_urls
            if queue_entry is not None:
                result["entry"] = queue_entry

        except Exception as e:  # pragma: no cover - 网络/LLM异常
            log.info(f"[并行处理] 处理URL时出错 ({url}): {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

