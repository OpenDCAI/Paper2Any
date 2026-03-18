from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse, unquote

# Dataflow Agent 依赖
from dataflow_agent.state import MainState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

from playwright.async_api import async_playwright, Page, BrowserContext, Browser, Playwright

from playwright_stealth import Stealth


from openai import AsyncOpenAI

log = get_logger(__name__)


class DomFetcher:
    """网页DOM数据抓取类"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.proxy_config = self._get_proxy_config()
    
    def _get_proxy_config(self) -> Optional[Dict[str, str]]:
        """从环境变量读取代理配置"""
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        
        if not http_proxy and not https_proxy:
            all_proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
            if all_proxy:
                if all_proxy.startswith("socks5h://"):
                    http_proxy = all_proxy.replace("socks5h://", "http://")
                else:
                    http_proxy = all_proxy
        
        proxy_url = http_proxy or https_proxy
        
        if proxy_url:
            parsed = urlparse(proxy_url)
            server = f"{parsed.scheme}://{parsed.netloc}"
            no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy", "localhost,127.0.0.1,::1")
            
            return {
                "server": server,
                "bypass": no_proxy
            }
        
        return None
    
    async def fetch_html(self, url: str, wait_time: int = 3) -> Optional[str]:
        log.info(f"🌐 正在访问页面: {url}")
        try:
            log.info("🕵️ 正在应用反爬虫绕过技术 (playwright-stealth)...")
            async with Stealth().use_async(async_playwright()) as p:
                return await self._process_page(p, url, wait_time)
        except Exception as e:
            log.error(f"❌ 抓取失败: {str(e)}")
            import traceback
            log.error(traceback.format_exc())
            return None
    
    async def _process_page(self, p, url: str, wait_time: int) -> str:
        launch_args = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ]
        }
        
        if self.proxy_config:
            launch_args["proxy"] = self.proxy_config
            log.info(f"🕵️ DomFetcher 使用代理: {self.proxy_config['server']}")
        
        browser = await p.chromium.launch(**launch_args)
        
        context_options = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        }
        
        if self.proxy_config:
            context_options["proxy"] = self.proxy_config
        
        context = await browser.new_context(**context_options)
        page = await context.new_page()
        
        try:
            log.info(f"📄 正在加载页面...")
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
                log.info("✅ 网络请求已稳定")
            except Exception:
                log.warning("⚠️ 网络空闲等待超时，继续执行...")
            
            if wait_time > 0:
                log.info(f"⏳ 额外等待 {wait_time} 秒...")
                await asyncio.sleep(wait_time)
            
            html_content = await page.content()
            log.info(f"✅ HTML获取成功，长度: {len(html_content)} 字符")
            
            return html_content
            
        finally:
            await browser.close()



class PlaywrightToolKit:
    def __init__(self, headless: bool = True, use_proxy: Optional[bool] = None):
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.base_download_dir: Optional[str] = None
        
        if use_proxy is None:
            has_env_proxy = bool(
                os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or
                os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or
                os.getenv("ALL_PROXY") or os.getenv("all_proxy")
            )
            self.use_proxy = has_env_proxy
        else:
            self.use_proxy = use_proxy
        
        self.proxy_config = self._get_proxy_config() if self.use_proxy else None
    
    def _get_proxy_config(self) -> Optional[Dict[str, str]]:
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        
        if not http_proxy and not https_proxy:
            all_proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
            if all_proxy:
                if all_proxy.startswith("socks5h://"):
                    http_proxy = all_proxy.replace("socks5h://", "http://")
                else:
                    http_proxy = all_proxy
        
        proxy_url = http_proxy or https_proxy
        
        if proxy_url:
            parsed = urlparse(proxy_url)
            server = f"{parsed.scheme}://{parsed.netloc}"
            no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy", "localhost,127.0.0.1,::1")
            return {
                "server": server,
                "bypass": no_proxy
            }
        
        return {
            "server": "http://127.0.0.1:7890",
            "bypass": "localhost,127.0.0.1,0.0.0.0"
        }

    async def start(self):
        if not self.playwright:
            self.playwright = await async_playwright().start()
        if not self.browser:
            launch_args = {
                "headless": self.headless,
                "args": ["--no-sandbox", "--disable-setuid-sandbox"]
            }
            if self.use_proxy:
                 launch_args["proxy"] = self.proxy_config
            self.browser = await self.playwright.chromium.launch(**launch_args)
            
            context_options = {
                "viewport": {"width": 1280, "height": 800},
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "ignore_https_errors": True
            }
            
            if self.use_proxy and self.proxy_config:
                context_options["proxy"] = self.proxy_config
                log.info(f"✅ 已配置浏览器代理: {self.proxy_config['server']}")
            else:
                log.info("⚠️  未使用代理")
            
            self.context = await self.browser.new_context(**context_options)
            self.page = await self.context.new_page()

    async def _ensure_browser(self):
        if not self.page:
            await self.start()

    async def close(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        self.page = None

    async def _wait_and_stabilize(self):
        try:
            if self.page:
                await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                await self.page.wait_for_load_state("networkidle", timeout=2000)
        except Exception:
            pass
        await asyncio.sleep(1)

    async def navigate(self, url: str) -> str:
        try:
            await self._ensure_browser()
            await self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            await self._wait_and_stabilize()
            title = await self.page.title()
            return f"Success: Navigated to {url}. Page Title: {title}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def get_accessibility_tree(self, use_accessibility_tree: bool = True) -> str:
        try:
            await self._ensure_browser()
            if use_accessibility_tree:
                snapshot = await self.page.locator("body").aria_snapshot()
                if not snapshot:
                    return "Empty accessibility tree (aria_snapshot)."
                return snapshot
            snapshot = await self.page.content()
            if not snapshot:
                return "Empty DOM (raw DOM)."
            return snapshot
        except Exception as e:
            return f"Error getting snapshot: {str(e)}"

    async def click_element(self, element_text: str) -> str:
        await self._ensure_browser()
        log.info(f"🖱️  Attempting to click element with text: '{element_text}'")
        
        strategies = [
            { "name": "get_by_role('button')", "locator": lambda: self.page.get_by_role("button", name=element_text).first },
            { "name": "get_by_role('link')", "locator": lambda: self.page.get_by_role("link", name=element_text).first },
            { "name": "button:text=", "locator": lambda: self.page.locator(f"button:text={element_text}").first },
            { "name": "button:has-text()", "locator": lambda: self.page.locator(f"button:has-text('{element_text}')").first },
            { "name": "summary:text=", "locator": lambda: self.page.locator(f"summary:text={element_text}").first },
            { "name": "summary:has-text()", "locator": lambda: self.page.locator(f"summary:has-text('{element_text}')").first },
            { "name": "text=", "locator": lambda: self.page.locator(f"text={element_text}").first },
            { "name": ":has-text()", "locator": lambda: self.page.locator(f":has-text('{element_text}')").first },
            { "name": "CSS [aria-label]", "locator": lambda: self.page.locator(f"[aria-label='{element_text}']").first },
            { "name": "CSS [aria-label*='...']", "locator": lambda: self.page.locator(f"[aria-label*='{element_text[:20]}']").first if len(element_text) >= 5 else None, "skip_if": lambda: len(element_text) < 5 },
            { "name": "XPath contains", "locator": lambda: self.page.locator(f"xpath=//*[contains(text(), '{element_text}')]").first if "'" not in element_text else None, "skip_if": lambda: "'" in element_text },
            { "name": "XPath button contains", "locator": lambda: self.page.locator(f"xpath=//button[contains(text(), '{element_text}')]").first if "'" not in element_text else None, "skip_if": lambda: "'" in element_text },
        ]
        
        errors = []
        for strategy in strategies:
            try:
                if "skip_if" in strategy and strategy["skip_if"](): continue
                locator = strategy["locator"]()
                if locator is None: continue
                
                count = await locator.count()
                if count > 0:
                    if await locator.is_visible():
                        try:
                            await locator.wait_for(state="visible", timeout=3000)
                            await locator.click(timeout=5000)
                            await self._wait_and_stabilize()
                            return f"Success: Clicked element '{element_text}' using strategy '{strategy['name']}'"
                        except Exception as click_error:
                            errors.append(f"{strategy['name']}: 点击失败 - {str(click_error)}")
                    else:
                        errors.append(f"{strategy['name']}: 元素存在但不可见")
                else:
                    errors.append(f"{strategy['name']}: 未找到元素")
            except Exception as e:
                errors.append(f"{strategy['name']}: 执行异常 - {str(e)}")
        
        error_summary = "\n".join([f"  - {err}" for err in errors[:5]])
        return f"Error: Could not click element '{element_text}'.\n{error_summary}"

    async def input_text(self, element_label_or_placeholder: str, text: str) -> str:
        await self._ensure_browser()
        log.info(f"⌨️  Attempting to input '{text}' into field: '{element_label_or_placeholder}'")
        
        locate_strategies = [
            { "name": "get_by_placeholder", "locator": lambda: self.page.get_by_placeholder(element_label_or_placeholder).first },
            { "name": "get_by_label", "locator": lambda: self.page.get_by_label(element_label_or_placeholder).first },
            { "name": "get_by_role('textbox')", "locator": lambda: self.page.get_by_role("textbox", name=element_label_or_placeholder).first },
            { "name": "get_by_role('combobox')", "locator": lambda: self.page.get_by_role("combobox", name=element_label_or_placeholder).first },
            { "name": "input[placeholder*='...']", "locator": lambda: self.page.locator(f"input[placeholder*='{element_label_or_placeholder[:10]}']").first if len(element_label_or_placeholder) >= 5 else None, "skip_if": lambda: len(element_label_or_placeholder) < 5 },
            { "name": "容器查找 input", "locator": lambda: self.page.locator(f":text('{element_label_or_placeholder}') >> xpath=.. >> input").first if "'" not in element_label_or_placeholder else None, "skip_if": lambda: "'" in element_label_or_placeholder },
        ]
        
        target_locator = None
        for strategy in locate_strategies:
            try:
                if "skip_if" in strategy and strategy["skip_if"](): continue
                locator = strategy["locator"]()
                if locator and await locator.count() > 0 and await locator.is_visible():
                    target_locator = locator
                    log.info(f"  ✅ Found input using: {strategy['name']}")
                    break
            except:
                continue

        if not target_locator:
            return f"Error: Could not find input field matching '{element_label_or_placeholder}'."

        input_strategies = [
            { "name": "fill()", "func": lambda loc: self._input_using_fill(loc, text) },
            { "name": "type()", "func": lambda loc: self._input_using_type(loc, text) },
            { "name": "keyboard", "func": lambda loc: self._input_using_keyboard_char_by_char(loc, text) },
        ]

        for strategy in input_strategies:
            try:
                result = await strategy["func"](target_locator)
                if result.startswith("Success"):
                    await self._wait_and_stabilize()
                    return f"{result} using {strategy['name']}"
            except:
                continue

        return f"Error: All input methods failed for '{element_label_or_placeholder}'."

    async def _input_using_fill(self, locator, text: str) -> str:
        await locator.fill("")
        await locator.fill(text)
        await locator.press("Enter")
        return f"Success: Input '{text}' using fill()"

    async def _input_using_type(self, locator, text: str) -> str:
        await locator.fill("")
        await locator.type(text, delay=50)
        await locator.press("Enter")
        return f"Success: Input '{text}' using type()"

    async def _input_using_keyboard_char_by_char(self, locator, text: str) -> str:
        await locator.wait_for(state="visible", timeout=3000)
        await locator.click()
        for char in text:
            await self.page.keyboard.type(char, delay=30)
        await self.page.keyboard.press("Enter")
        return f"Success: Input '{text}' using keyboard"

    async def go_back(self) -> str:
        try:
            await self._ensure_browser()
            await self.page.go_back()
            await self._wait_and_stabilize()
            title = await self.page.title()
            return f"Success: Navigated back. Current page title: {title}"
        except Exception as e:
            return f"Error going back: {str(e)}"

    async def download_resource(self, url: str, download_dir: Optional[str] = None) -> str:
        """下载资源文件（信任 LLM 传入的完整 URL）"""
        try:
            if not url or not url.strip():
                return "Error: URL is required."
            
            log.info(f"📥 开始下载资源: {url}")
            await self._ensure_browser()
            
            if download_dir is None:
                download_dir = self.base_download_dir if self.base_download_dir else os.path.join(os.getcwd(), "downloads")
            
            os.makedirs(download_dir, exist_ok=True)
            
            parsed_url = urlparse(url)
            path = unquote(parsed_url.path)
            suggested_filename = os.path.basename(path) if path else "download"
            
            # 尝试通过 HEAD 请求获取文件名，但不修改 URL
            try:
                response = await self.context.request.head(url, timeout=30000)
                content_disposition = response.headers.get('content-disposition', '')
                if content_disposition:
                    import re
                    filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
                    if filename_match:
                        suggested_filename = filename_match.group(1).strip('"\'')
                await response.dispose()
            except:
                pass
            
            if not suggested_filename or suggested_filename == 'download' or '.' not in suggested_filename:
                # 如果没有后缀，自动补全pdf（仅作为文件名兜底，不改变请求URL）
                suggested_filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            save_path = os.path.join(download_dir, suggested_filename)
            if os.path.exists(save_path):
                name, ext = os.path.splitext(suggested_filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(download_dir, f"{name}_{timestamp}{ext}")
            
            log.info(f"⬇️  正在下载文件...")
            # 直接使用传入的 URL，不做任何硬修改
            response = await self.context.request.get(url, timeout=60000)
            
            if response.status >= 400:
                error_text = await response.text()
                await response.dispose()
                return f"Error: HTTP {response.status} - {error_text[:200]}"
            
            content = await response.body()
            await response.dispose()
            
            with open(save_path, 'wb') as f:
                f.write(content)
            
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            return f"Success: Downloaded resource to {save_path} (Size: {file_size_mb:.2f} MB)"
        except Exception as e:
            log.error(f"❌ 下载过程中发生错误: {str(e)}")
            return f"Error downloading resource from {url}: {str(e)}"

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        tool_map = {
            "navigate": self.navigate,
            "get_accessibility_tree": self.get_accessibility_tree,
            "click_element": self.click_element,
            "input_text": self.input_text,
            "go_back": self.go_back,
            "download_resource": self.download_resource
        }
        if tool_name not in tool_map:
            return f"Error: Tool '{tool_name}' not found."
        return await tool_map[tool_name](**arguments)

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "navigate",
                "description": "Navigate to a specific URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to navigate to"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "get_accessibility_tree",
                "description": "Get the current page structure to read text and identify elements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "use_accessibility_tree": {
                            "type": "boolean",
                            "description": "True 使用 aria_snapshot 过滤树；False 使用原始 DOM 快照",
                            "default": True
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "click_element",
                "description": "Click an element by its visible text content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_text": {"type": "string", "description": "The visible text on the element to click"}
                    },
                    "required": ["element_text"]
                }
            },
            {
                "name": "input_text",
                "description": "Input text into a field identified by its Placeholder text or Label name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_label_or_placeholder": {"type": "string", "description": "The placeholder text or label name"},
                        "text": {"type": "string", "description": "Text to input"}
                    },
                    "required": ["element_label_or_placeholder", "text"]
                }
            },
            {
                "name": "go_back",
                "description": "Navigate back to the previous page",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "download_resource",
                "description": "Download a resource file. **CRITICAL**: You must provide the *final direct download URL*. The tool will NOT fix URLs for you. If you are downloading a paper (e.g., arXiv), you MUST manually convert the abstract URL to a PDF URL (e.g. change 'abs' to 'pdf' and add '.pdf') BEFORE calling this tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The complete direct download URL. Example: verify that 'https://arxiv.org/abs/2506.21506' is changed to 'https://arxiv.org/pdf/2506.21506.pdf' before inputting here."},
                        "download_dir": {"type": "string", "description": "Optional directory"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "terminate",
                "description": "Finish the task",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

# ==========================================
# 2. 智能体逻辑 (Agent Logic)
# ==========================================

class WebAgent:
    def __init__(self, toolkit: PlaywrightToolKit, llm_config: Optional[Dict] = None, dom_save_dir: Optional[Path] = None):
        self.toolkit = toolkit
        self.action_history: List[Dict[str, Any]] = []
        self.accessibility_trees: List[Dict[str, Any]] = []
        self.consecutive_failures = 0
        self.dom_save_dir = dom_save_dir
        self.visited_urls: set = set()
        self.dom_fetcher = DomFetcher(headless=toolkit.headless) if hasattr(toolkit, 'headless') else DomFetcher(headless=True)
        
        self.llm_config = llm_config or {
            "base_url": os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            "api_key": os.getenv("DF_API_KEY", "sk-xxx"),
            "model": os.getenv("THIRD_PARTY_MODEL", "gpt-4o"),
        }
        api_key = llm_config.get("api_key")
        base_url = llm_config.get("base_url")
        model = llm_config.get("model")
        
        http_client = httpx.AsyncClient(trust_env=False)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        self.model = model
        log.info(f"🔗 LLM Initialized with Base URL: {base_url}")

    def _construct_prompt(self, task: str, current_tree: str) -> str:
        if self.action_history:
            history_lines = []
            for i, record in enumerate(self.action_history, 1):
                step_num = record.get("step", i)
                action = record.get("action", "")
                result = record.get("result", "")
                success = record.get("success", False)
                status = "✅ SUCCESS" if success else "❌ FAILED"
                history_lines.append(f"Step {step_num} [{status}]: {action}")
                history_lines.append(f"  Result: {result}")
            history_str = "\n".join(history_lines)
        else:
            history_str = "None (Start of task)"
        
        tools_schema = self.toolkit.get_tools_schema()
        tools_schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False)
        
        failure_warning = ""
        if self.consecutive_failures >= 2:
            failure_warning = f"\n⚠️  WARNING: {self.consecutive_failures} consecutive failures detected. Consider using go_back()."
        
        return f"""
You are an autonomous web automation agent.

Your Goal: {task}

--- ACTION SPACE (Available Tools) ---

{tools_schema_str}

--- ACTION HISTORY ---

{history_str}

{failure_warning}

--- CURRENT ACCESSIBILITY TREE ---

{current_tree}

--- FEW-SHOT EXAMPLES (URL CORRECTION) ---

**IMPORTANT**: When using `download_resource`, you MUST manually correct URLs from abstract/view pages to direct file links. The tool does NOT do this for you.

Example 1 (arXiv Paper):
User Goal: "Download the paper at https://arxiv.org/abs/2506.21506"
Observation: The URL is an abstract page (.../abs/...), not a PDF.
Thought: "I need to download the PDF. The tool requires a direct link. I will convert 'abs' to 'pdf' and add the extension."
Tool Call:
{{
  "thought": "I am converting the abstract URL to a direct PDF URL to ensure successful download.",
  "tool": "download_resource",
  "args": {{
    "url": "https://arxiv.org/pdf/2506.21506.pdf"
  }}
}}

Example 2 (General PDF):
Observation: Link text says "Download Manual", href is "http://example.com/manual" (redirects to PDF).
Thought: "It is safer to assume the file ends in .pdf for the tool."
Tool Call:
{{
  "thought": "Appending .pdf to ensure the tool handles it as a file.",
  "tool": "download_resource",
  "args": {{
    "url": "http://example.com/manual.pdf"
  }}
}}

--- INSTRUCTIONS ---

1. Review ACTION HISTORY. If actions failed, try a different approach.
2. Choose a tool from ACTION SPACE.
3. **CRITICAL**: BEFORE calling `download_resource`, verify the URL in your thought process. 
   - If it is an arXiv link (`/abs/`), YOU MUST change it to `/pdf/` and ensure it ends in `.pdf`.
   - Do not use "hard" methods in the tool; use your intelligence to format the URL correctly in the `args`.
4. AVOID CAPTCHA sites (Google Scholar, Stack Overflow).
5. Output JSON format ONLY with "thought", "tool", and "args".
"""

    async def _call_real_llm(self, prompt: str) -> str:
        log.info("🤔 Agent is thinking...")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful web assistant. You always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"❌ OpenAI API Error: {e}")
            return "{}"

    async def _get_current_url(self) -> str:
        try:
            await self.toolkit._ensure_browser()
            if self.toolkit.page:
                return self.toolkit.page.url
        except:
            pass
        return "Unknown"

    async def _save_dom_if_new_url(self, url: str, step: int) -> Optional[str]:
        if not self.dom_save_dir: return None
        normalized_url = url.split('#')[0]
        if normalized_url in self.visited_urls: return None
        self.visited_urls.add(normalized_url)
        
        dom_dir = self.dom_save_dir / "dom_snapshots"
        dom_dir.mkdir(exist_ok=True)
        
        try:
            html_content = await self.dom_fetcher.fetch_html(url, wait_time=3)
            if html_content:
                from urllib.parse import urlparse
                parsed = urlparse(normalized_url)
                domain = parsed.netloc.replace('.', '_')
                path = parsed.path.replace('/', '_').strip('_') or 'index'
                filename = f"step_{step:03d}_{domain}_{path[:50]}.html"
                filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
                filepath = dom_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html_content)
                return str(filepath)
            return None
        except Exception:
            return None

    async def run(self, task: str, max_steps: int = 12) -> str:
        log.info(f"🚀 Starting Real Task: {task}")
        final_summary = "Task executed but no summary returned."
        
        for i in range(max_steps):
            log.info(f"\n--- Step {i+1} ---")
            tree = await self.toolkit.get_accessibility_tree()
            current_url = await self._get_current_url()
            
            if current_url != "Unknown":
                await self._save_dom_if_new_url(current_url, i + 1)
            
            tree_record = {
                "step": i + 1,
                "timestamp": datetime.now().isoformat(),
                "url": current_url,
                "accessibility_tree_snippet": tree[:5000], 
                "task": task
            }
            self.accessibility_trees.append(tree_record)
            
            prompt = self._construct_prompt(task, tree)
            response_str = await self._call_real_llm(prompt)
            
            try:
                action_data = json.loads(response_str)
                thought = action_data.get("thought", "No thought")
                tool_name = action_data.get("tool")
                args = action_data.get("args", {})
                
                log.info(f"🧠 Thought: {thought}")
                log.info(f"🛠️  Action: {tool_name} {args}")
                
                if tool_name == "terminate":
                    log.info("✅ Agent completed the task.")
                    final_summary = thought or "Task completed successfully."
                    break
                
                if not tool_name or not str(tool_name).strip():
                    log.warning("⚠️ LLM 返回的 tool 为空（可能超时或 API 异常），本步跳过")
                    result = "Error: No tool returned (API timeout or empty response)."
                    self.consecutive_failures += 1
                    self.action_history.append({
                        "step": i + 1,
                        "action": "(empty)",
                        "result": result,
                        "success": False,
                        "thought": thought
                    })
                    continue
                
                result = await self.toolkit.execute_tool(tool_name, args)
                log.info(f"📝 Result: {result}")
                
                if tool_name == "navigate" and result.startswith("Success"):
                    await asyncio.sleep(1)
                    new_url = await self._get_current_url()
                    if new_url != "Unknown":
                        await self._save_dom_if_new_url(new_url, i + 1)
                
                is_success = result.startswith("Success")
                if is_success:
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                
                self.action_history.append({
                    "step": i + 1,
                    "action": f"{tool_name}({args})",
                    "result": result,
                    "success": is_success,
                    "thought": thought
                })
                
            except json.JSONDecodeError:
                log.error(f"❌ JSON Error: {response_str}")
                self.consecutive_failures += 1
            except Exception as e:
                log.error(f"❌ Exec Error: {e}")
                self.consecutive_failures += 1
        
        return final_summary

    def save_accessibility_trees(self, filepath: Optional[str] = None) -> str:
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"accessibility_trees_{timestamp}.json"
        try:
            full_record = {
                "accessibility_trees": self.accessibility_trees,
                "action_history": self.action_history,
                "consecutive_failures": self.consecutive_failures
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_record, f, indent=2, ensure_ascii=False)
            return filepath
        except:
            return ""

@register("websearch_researcher")
class WebsearchResearcherAgent(BaseAgent):
    """Websearch Researcher Agent"""

    def __init__(self, tool_manager: Optional[ToolManager] = None, llm_config: Optional[Dict] = None, **kwargs):
        super().__init__(tool_manager=tool_manager, **kwargs)
        self.llm_config = llm_config or {
            "base_url": os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            "api_key": os.getenv("DF_API_KEY", "sk-xxx"),
            "model": os.getenv("THIRD_PARTY_MODEL", "gpt-4o"),
        }
        self.output_dir = Path("./raw_data_store")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs) -> "WebsearchResearcherAgent":
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "websearch_researcher"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_websearch_researcher"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_websearch_researcher"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"pre_tool_results": pre_tool_results}

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    async def run(self, state: MainState, **kwargs) -> Dict[str, Any]:
        task_description = getattr(state, "current_task", "Find relevant financial data.")
        log.info(f"[WebsearchResearcher] Task: {task_description}")

        timestamp = int(time.time())
        safe_task_name = "".join([c for c in task_description[:15] if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_') or "task"
        session_dir = self.output_dir / f"{timestamp}_{safe_task_name}"
        session_dir.mkdir(exist_ok=True)

        headless_mode = os.getenv("HEADLESS", "true").lower() == "true"
        use_proxy_env = os.getenv("USE_PROXY", "").lower()
        use_proxy = True if use_proxy_env == "true" else (False if use_proxy_env == "false" else None)
        
        toolkit = PlaywrightToolKit(headless=headless_mode, use_proxy=use_proxy)
        final_resources_dir = session_dir / "final_resources"
        final_resources_dir.mkdir(exist_ok=True)
        toolkit.base_download_dir = str(final_resources_dir)

        final_summary = "Task executed but no summary returned."

        try:
            await toolkit.start()
            agent = WebAgent(toolkit=toolkit, llm_config=self.llm_config, dom_save_dir=session_dir)
            final_summary = await agent.run(task_description, max_steps=12)
            
            try:
                trees_file = agent.save_accessibility_trees(str(session_dir / "accessibility_trees.json"))
                if trees_file: log.info(f"Accessibility trees saved to: {trees_file}")
            except Exception as e:
                log.warning(f"Failed to save accessibility trees: {e}")

            result_payload = {
                "summary": final_summary,
                "storage_path": str(session_dir),
                "captured_files_count": 0,
                "captured_files": [],
                "status": "success"
            }
        except Exception as e:
            log.error(f"[WebsearchResearcher] Execution failed: {e}", exc_info=True)
            result_payload = {"error": str(e), "status": "failed"}
        finally:
            await toolkit.close()

        if hasattr(state, "agent_results") and state.agent_results is not None:
            state.agent_results[self.role_name] = {"result": result_payload, "pre_tool_results": self.get_default_pre_tool_results()}
        
        return result_payload

def create_websearch_researcher_agent(tool_manager: Optional[ToolManager] = None, **kwargs) -> WebsearchResearcherAgent:
    return WebsearchResearcherAgent.create(tool_manager=tool_manager, **kwargs)