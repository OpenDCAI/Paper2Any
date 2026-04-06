"""
Websearch Initial Analyzer Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
对应「Initial Analyzer / 初始分析师」角色：
- 访问 Input URLs（冷启动）
- 把原始正文 / 多模态引用写入 Raw Data Store
- 产出研究路线 Research Routes

实现功能：
- 使用 DomFetcher 抓取网页 DOM 数据
- 保存 DOM 到 raw_data_store 目录
- 调用 mineruhtml API 提取网页正文
- 使用 LLM 分析正文生成子任务列表
"""

from __future__ import annotations

import os
import json
import time
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

from dataflow_agent.state import MainState, WebsearchKnowledgeState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

# 导入 DomFetcher（从 websearch_researcher 中复用）
from dataflow_agent.agentroles.paper2any_agents.websearch_researcher import DomFetcher

# OpenAI 依赖
from openai import AsyncOpenAI

log = get_logger(__name__)


@register("websearch_initial_analyzer")
class WebsearchInitialAnalyzerAgent(BaseAgent):
    """
    Websearch Initial Analyzer Agent（完整实现版）
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
        
        self.mineruhtml_url = os.getenv("MINERUHTML_API_URL", "http://localhost:7771")
        self.output_dir = Path("./raw_data_store")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        headless_mode = os.getenv("HEADLESS", "true").lower() == "true"
        self.dom_fetcher = DomFetcher(headless=headless_mode)
        
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
    ) -> "WebsearchInitialAnalyzerAgent":
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "websearch_initial_analyzer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_websearch_initial_analyzer"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_websearch_initial_analyzer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造初始分析需要的 prompt 参数。
        """
        return {
            "pre_tool_results": pre_tool_results,
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {}

    def _extract_input_urls_from_state(self, state: MainState) -> List[str]:
        """从 state 中提取 input_urls"""
        if isinstance(state, WebsearchKnowledgeState):
            # 优先从 state.input_urls 获取
            urls = state.input_urls or []
            if not urls:
                # 从 request.input_urls 获取
                urls = getattr(state.request, "input_urls", []) or []
            return urls
        else:
            # 兼容其他 State 类型
            urls = getattr(state, "input_urls", []) or []
            if not urls:
                request = getattr(state, "request", None)
                if request:
                    urls = getattr(request, "input_urls", []) or []
            return urls

    async def _fetch_and_save_dom(self, url: str, session_dir: Path, url_index: int) -> Optional[str]:
        """
        抓取并保存网页 DOM
        
        Returns:
            保存的 HTML 文件路径，失败返回 None
        """
        try:
            log.info(f"🌐 [{url_index}] 正在抓取网页 DOM: {url}")
            
            # 抓取 HTML
            html_content = await self.dom_fetcher.fetch_html(url, wait_time=3)
            
            if not html_content:
                log.warning(f"⚠️ [{url_index}] DOM 抓取失败: {url}")
                return None
            
            # 保存到 dom_snapshots 目录
            dom_dir = session_dir / "dom_snapshots"
            dom_dir.mkdir(exist_ok=True)
            
            # 生成安全的文件名
            parsed = urlparse(url)
            domain = parsed.netloc.replace('.', '_')
            path = parsed.path.replace('/', '_').strip('_') or 'index'
            if len(path) > 50:
                path = path[:50]
            
            filename = f"url_{url_index:02d}_{domain}_{path}.html"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
            
            filepath = dom_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            log.info(f"✅ [{url_index}] DOM 已保存至: {filepath}")
            return str(filepath)
            
        except Exception as e:
            log.error(f"❌ [{url_index}] 保存 DOM 时出错: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None

    async def _extract_content_with_mineruhtml(self, html_content: str, url: str) -> Optional[str]:
        """
        调用 mineruhtml API 提取网页正文
        
        API 规范:
        - URL:   http://localhost:7771/extract  (这里用 self.mineruhtml_url 做前缀)
        - Method: POST
        - Content-Type: application/json
        - Request JSON: { "html": "<完整 HTML 字符串>" }   （字段名为 html）
        - Response JSON: { "main_html": "<提取出的正文 HTML>" }
        """
        try:
            log.info("📄 正在调用 mineruhtml API 提取正文...")

            api_url = f"{self.mineruhtml_url}/extract"
            payload = {
                "html": html_content,
            }

            # mineruhtml 端推理可能较慢，这里将超时时间放宽到 3000 秒（5 分钟）
            async with httpx.AsyncClient(timeout=3000.0) as client:
                response = await client.post(
                    api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code != 200:
                log.error(f"❌ mineruhtml API 返回错误: {response.status_code}")
                try:
                    log.error(f"错误详情: {response.text[:500]}")
                except Exception:
                    pass
                return None

            try:
                result = response.json()
            except json.JSONDecodeError:
                log.error("❌ mineruhtml 返回的不是合法 JSON")
                return None

            extracted_content = result.get("main_html", "")

            if extracted_content and extracted_content.strip():
                log.info(f"✅ 正文提取成功，长度: {len(extracted_content)} 字符")
                return extracted_content

            log.warning("⚠️ mineruhtml 返回的 main_html 为空")
            return None

        except Exception as e:
            log.error(f"❌ 调用 mineruhtml API 时出错: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None

    async def _analyze_content_and_generate_tasks(
        self, 
        url: str, 
        extracted_content: str
    ) -> List[str]:
        """
        使用 LLM 分析正文内容，生成子任务列表
        
        Args:
            url: 原始 URL
            extracted_content: 提取的正文内容
            
        Returns:
            子任务列表
        """
        try:
            log.info(f"🤔 正在使用 LLM 分析内容并生成子任务...")
            
            # 截断内容（避免超出 token 限制）
            max_content_length = 8000
            if len(extracted_content) > max_content_length:
                extracted_content = extracted_content[:max_content_length] + "\n\n[... 内容已截断 ...]"
            
            prompt = f"""
你是一个专业的知识研究规划师。请分析以下网页内容，识别其中的核心知识点，并生成**研究型子任务**列表。

【核心目标】子任务将由 WebAgent 自动执行（搜索、访问网页、阅读内容），后续由知识提取 Agent 从研究结果中提取结构化知识。
因此，每个子任务必须是一个**可执行的调研指令**，聚焦于某个知识点的深入调研。

## 子任务类型和格式

1. **[概念调研]** - 调研某个核心概念/术语的定义、原理和机制
   格式：`[概念调研] 调研<概念名称>的<具体调研方向>`

2. **[对比调研]** - 调研多种方法/技术/方案的异同与优劣
   格式：`[对比调研] 对比<方法A>与<方法B>在<维度>上的异同`

3. **[溯源调研]** - 调研某个方法/理论的来源、发展脉络和关键文献
   格式：`[溯源调研] 调研<主题>的<具体溯源方向>`

4. **[技术调研]** - 调研某个具体技术/实现的详细工作机制
   格式：`[技术调研] 调研<技术名称>的<具体技术方向>`

5. **[应用调研]** - 调研某技术/方法在特定领域的实际应用和效果
   格式：`[应用调研] 调研<技术/方法>在<领域>中的应用方式和效果`

## Few-shot 示例

**示例正文：**
扩散模型介绍：2020 年提出的 DDPM（Denoising Diffusion Probabilistic Model）开启了扩散模型的热潮。扩散模型通过从噪声中采样来生成目标数据，包含前向过程（逐步加噪）和反向过程（通过 U-Net 逐步去噪还原图片）。代码实现基于 MindSpore，包含正弦位置编码、Attention 与 Residual Block、GaussianDiffusion 以及引入 EMA 优化的 Trainer。参考论文《Denoising Diffusion Probabilistic Models》，代码仓库 GitHub: lvyufeng/denoising-diffusion-mindspore。

**示例输出：**
{{
    "tasks": [
        "[概念调研] 调研 DDPM 扩散模型前向加噪和反向去噪的数学原理及噪声调度策略",
        "[技术调研] 调研 U-Net 在扩散模型中作为去噪网络的架构设计和跳跃连接机制",
        "[对比调研] 对比扩散模型与 GAN、VAE 在生成质量、训练稳定性和推理速度上的优劣",
        "[溯源调研] 调研扩散模型从 Sohl-Dickstein 非平衡热力学到 DDPM 再到 Score-based models 的发展脉络",
        "[技术调研] 调研 EMA 指数移动平均在扩散模型训练中的优化机制和衰减率选择",
        "[概念调研] 调研 DDIM 加速采样将马尔可夫过程转化为非马尔可夫过程的原理"
    ],
    "reasoning": "从知识提取角度分析：1) DDPM 数学原理是核心基础知识；2) U-Net 架构是关键技术组件；3) 与 GAN/VAE 对比能建立生成模型知识体系；4) 溯源研究构建发展脉络；5) EMA 是重要训练优化技术；6) DDIM 是采样加速的关键方法。"
}}

---

## 当前任务

网页 URL: {url}

网页正文内容：
{extracted_content}

请根据网页内容，生成 3-8 个**研究型子任务**。设计原则：

1. **可执行性**：每个任务是 WebAgent 可直接执行的调研指令（去搜索、阅读、收集信息）
2. **面向知识提取**：调研方向必须能产出可结构化的知识（定义、原理、对比、流程等）
3. **深度聚焦**：每个任务聚焦一个明确的知识点，深入调研而非泛泛浏览
4. **知识覆盖**：任务之间应覆盖不同知识维度（概念、对比、溯源、技术、应用）
5. **严格遵循格式**：使用 [概念调研]/[对比调研]/[溯源调研]/[技术调研]/[应用调研] 标签

输出格式：请返回一个 JSON 对象，格式如下：
{{
    "tasks": [
        "[概念调研] 调研...",
        "[对比调研] 对比...",
        "[溯源调研] 调研...",
        "[技术调研] 调研...",
        "[应用调研] 调研...",
        ...
    ],
    "reasoning": "从知识提取角度说明为什么选择这些调研方向"
}}

只返回 JSON，不要其他内容。
"""
            
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识研究规划师，擅长从内容中识别核心知识点并设计面向知识提取的研究型任务。你总是返回有效的 JSON 格式。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            tasks = result_json.get("tasks", [])
            reasoning = result_json.get("reasoning", "")
            
            log.info(f"✅ 生成了 {len(tasks)} 个子任务")
            if reasoning:
                log.info(f"📝 生成理由: {reasoning}")
            
            return tasks
            
        except Exception as e:
            log.error(f"❌ LLM 分析时出错: {e}")
            import traceback
            log.error(traceback.format_exc())
            # 返回默认任务
            return [f"深入研究 {url} 的相关内容"]

    async def run(self, state: MainState, **kwargs) -> Dict[str, Any]:
        """
        执行初始分析任务
        
        流程：
        1. 从 state 获取 input_urls
        2. 对每个 URL：
           a. 抓取 DOM 并保存
           b. 调用 mineruhtml API 提取正文
           c. 保存正文内容
        3. 使用 LLM 分析所有正文，生成子任务列表
        4. 更新 state 的 research_routes 和 raw_data_store
        """
        log.info(f"[WebsearchInitialAnalyzer] 开始执行初始分析任务")
        
        # 1. 提取 input_urls
        input_urls = self._extract_input_urls_from_state(state)
        
        if not input_urls:
            log.warning("[WebsearchInitialAnalyzer] 未找到 input_urls，跳过处理")
            result_payload = {
                "status": "skipped",
                "reason": "No input URLs found",
                "research_routes": [],
                "raw_data_store": []
            }
            self.update_state_result(state, result_payload, self.get_default_pre_tool_results())
            return result_payload
        
        log.info(f"[WebsearchInitialAnalyzer] 找到 {len(input_urls)} 个 URL: {input_urls}")
        
        # 2. 准备存储目录
        timestamp = int(time.time())
        session_dir = self.output_dir / f"{timestamp}_initial_analysis"
        session_dir.mkdir(exist_ok=True)
        
        # 3. 处理每个 URL
        all_extracted_contents = []
        raw_data_records = []
        
        for idx, url in enumerate(input_urls, 1):
            log.info(f"\n{'='*60}")
            log.info(f"处理 URL {idx}/{len(input_urls)}: {url}")
            log.info(f"{'='*60}")
            
            # 3.1 抓取并保存 DOM
            dom_filepath = await self._fetch_and_save_dom(url, session_dir, idx)
            
            # 3.2 读取 HTML 内容（如果保存成功）
            html_content = None
            if dom_filepath:
                try:
                    with open(dom_filepath, "r", encoding="utf-8") as f:
                        html_content = f.read()
                except Exception as e:
                    log.warning(f"⚠️ 读取 DOM 文件失败: {e}")
            
            # 如果 DOM 抓取失败，尝试直接抓取
            if not html_content:
                log.info(f"🔄 重新抓取 HTML 内容...")
                html_content = await self.dom_fetcher.fetch_html(url, wait_time=3)
            
            if not html_content:
                log.error(f"❌ 无法获取 HTML 内容，跳过该 URL")
                continue
            
            # 3.3 调用 mineruhtml API 提取正文
            extracted_content = await self._extract_content_with_mineruhtml(html_content, url)
            
            if not extracted_content:
                log.warning(f"⚠️ 正文提取失败，使用原始 HTML 的前 5000 字符作为正文")
                extracted_content = html_content[:5000]
            
            # 3.4 保存提取的正文
            content_filepath = session_dir / f"extracted_content_url_{idx:02d}.md"
            with open(content_filepath, "w", encoding="utf-8") as f:
                f.write(f"# URL: {url}\n\n")
                f.write(f"提取时间: {datetime.now().isoformat()}\n\n")
                f.write("---\n\n")
                f.write(extracted_content)
            
            log.info(f"✅ 正文已保存至: {content_filepath}")
            
            # 3.5 记录到 raw_data_store
            record = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "dom_filepath": dom_filepath,
                "content_filepath": str(content_filepath),
                "extracted_content_length": len(extracted_content),
                "extracted_content_preview": extracted_content[:200] + "..." if len(extracted_content) > 200 else extracted_content
            }
            raw_data_records.append(record)
            all_extracted_contents.append({
                "url": url,
                "content": extracted_content
            })
        
        # 4. 使用 LLM 分析所有内容，生成子任务列表
        research_routes = []
        
        if all_extracted_contents:
            log.info(f"\n{'='*60}")
            log.info(f"开始分析内容并生成研究子任务...")
            log.info(f"{'='*60}")
            
            # 合并所有内容
            combined_content = "\n\n".join([
                f"## URL {i+1}: {item['url']}\n\n{item['content'][:2000]}"  # 每个 URL 最多 2000 字符
                for i, item in enumerate(all_extracted_contents)
            ])
            
            # 生成子任务
            research_routes = await self._analyze_content_and_generate_tasks(
                url=", ".join([item["url"] for item in all_extracted_contents]),
                extracted_content=combined_content
            )
            
            log.info(f"✅ 生成了 {len(research_routes)} 个研究子任务:")
            for i, task in enumerate(research_routes, 1):
                log.info(f"  {i}. {task}")
        
        # 5. 保存汇总信息
        summary_filepath = session_dir / "analysis_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_urls": input_urls,
            "research_routes": research_routes,
            "raw_data_records": raw_data_records,
            "session_dir": str(session_dir)
        }
        with open(summary_filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        log.info(f"✅ 分析汇总已保存至: {summary_filepath}")
        
        # 6. 更新 state（如果是 WebsearchKnowledgeState）
        if isinstance(state, WebsearchKnowledgeState):
            # 更新 research_routes（直接赋值，因为这是初始分析）
            state.research_routes = research_routes
            
            # 同时保存一份原始任务列表（不会被 planner 修改），供 chief_curator 使用
            state.original_research_routes = research_routes.copy()
            
            # 更新 raw_data_store（追加模式，保留已有数据）
            if not hasattr(state, 'raw_data_store') or state.raw_data_store is None:
                state.raw_data_store = []
            # 追加新记录（避免重复）
            existing_urls = {r.get("url") for r in state.raw_data_store if isinstance(r, dict)}
            for record in raw_data_records:
                if record.get("url") not in existing_urls:
                    state.raw_data_store.append(record)
            
            log.info(f"✅ 已更新 state.research_routes ({len(research_routes)} 个任务)")
            log.info(f"✅ 已保存 state.original_research_routes ({len(state.original_research_routes)} 个原始任务)")
            log.info(f"✅ 已更新 state.raw_data_store (新增 {len(raw_data_records)} 条记录，总计 {len(state.raw_data_store)} 条)")
        
        # 7. 返回结果
        result_payload = {
            "status": "success",
            "session_dir": str(session_dir),
            "input_urls": input_urls,
            "research_routes": research_routes,
            "raw_data_store": raw_data_records,
            "summary_filepath": str(summary_filepath)
        }
        
        self.update_state_result(state, result_payload, self.get_default_pre_tool_results())
        return result_payload

    def update_state_result(
        self,
        state: MainState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """
        把初始分析结果写回 state.agent_results。
        """
        if getattr(state, "agent_results", None) is not None:
            state.agent_results[self.role_name] = {
                "result": result,
                "pre_tool_results": pre_tool_results,
            }
        log.debug("[WebsearchInitialAnalyzerAgent] result written to state.agent_results.")
        super().update_state_result(state, result, pre_tool_results)


def create_websearch_initial_analyzer_agent(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> WebsearchInitialAnalyzerAgent:
    return WebsearchInitialAnalyzerAgent.create(tool_manager=tool_manager, **kwargs)






