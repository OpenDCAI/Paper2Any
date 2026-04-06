from __future__ import annotations

import os
import json
import time
import httpx
import re
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm
import fitz  # PyMuPDF

from dataflow_agent.state import MainState, WebsearchKnowledgeState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.agentroles.cores.registry import register

# OpenAI 依赖
from openai import AsyncOpenAI

log = get_logger(__name__)


@register("websearch_curator")
class WebsearchChiefCuratorAgent(BaseAgent):
    """
    Websearch Chief Curator Agent
    实现：针对每个子任务独立建立知识库 MD 文件，包含多模态资源占位。
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
        log.info(f"🔗 LLM Initialized for Independent Curation: {base_url}")

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs) -> "WebsearchChiefCuratorAgent":
        return cls(tool_manager=tool_manager, **kwargs)

    @property
    def role_name(self) -> str:
        return "websearch_curator"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_websearch_curator"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_websearch_curator"

    def _simplify_html(self, html: str) -> str:
        """HTML 预处理，去除无关标签"""
        try:
            html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
            html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
            html = re.sub(r"", "", html)
            html = re.sub(r"<head[\s\S]*?</head>", "", html, flags=re.IGNORECASE)
        except Exception as e:
            log.warning(f"⚠️ 简化 HTML 出错: {e}")
        return html

    async def _extract_content_with_mineruhtml(self, html_content: str) -> Optional[str]:
        """调用 mineruhtml API 提取正文"""
        try:
            api_url = f"{self.mineruhtml_url}/extract"
            payload = {"html": html_content}
            async with httpx.AsyncClient(timeout=3000.0) as client:
                response = await client.post(api_url, json=payload)
            if response.status_code == 200:
                return response.json().get("main_html", "")
            return None
        except Exception as e:
            log.error(f"❌ mineruhtml API 出错: {e}")
            return None

    def _is_data_uri(self, url: str) -> bool:
        """检查 URL 是否是 Data URI"""
        return url.strip().lower().startswith("data:")

    def _parse_data_uri(self, data_uri: str) -> Tuple[Optional[str], Optional[bytes]]:
        """
        解析 Data URI，返回 (mime_type, decoded_data)
        Data URI 格式: data:[<mediatype>][;base64],<data>
        """
        try:
            # 去除空白字符
            data_uri = data_uri.strip()
            
            # 检查是否是有效的 data URI
            if not data_uri.lower().startswith("data:"):
                return None, None
            
            # 移除 "data:" 前缀
            data_part = data_uri[5:]
            
            # 分离 metadata 和 data
            if "," not in data_part:
                return None, None
            
            metadata, encoded_data = data_part.split(",", 1)
            
            # 解析 mime type 和编码
            mime_type = "application/octet-stream"  # 默认
            is_base64 = False
            
            if metadata:
                parts = metadata.split(";")
                if parts[0]:
                    mime_type = parts[0].lower()
                if "base64" in [p.lower() for p in parts]:
                    is_base64 = True
            
            # 解码数据
            if is_base64:
                # 处理可能的空格（有些 base64 字符串包含空格）
                encoded_data = encoded_data.replace(" ", "+")
                decoded_data = base64.b64decode(encoded_data)
            else:
                # URL 编码的数据
                from urllib.parse import unquote
                decoded_data = unquote(encoded_data).encode("utf-8")
            
            return mime_type, decoded_data
            
        except Exception as e:
            log.warning(f"⚠️ 解析 Data URI 失败: {e}")
            return None, None

    def _get_extension_from_mime(self, mime_type: str) -> str:
        """根据 MIME 类型获取文件扩展名"""
        mime_to_ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
            "image/bmp": ".bmp",
            "image/x-icon": ".ico",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/ogg": ".ogv",
            "audio/mpeg": ".mp3",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
        }
        return mime_to_ext.get(mime_type, ".bin")

    async def _save_data_uri(self, data_uri: str, save_dir: Path) -> Optional[str]:
        """保存 Data URI 中的数据到文件"""
        mime_type, decoded_data = self._parse_data_uri(data_uri)
        
        if decoded_data is None:
            log.warning(f"⚠️ 无法解析 Data URI")
            return None
        
        # 过滤太小的图片（通常是1x1像素的追踪图片）
        if len(decoded_data) < 100:  # 小于 100 字节的数据太小，可能是追踪像素
            log.debug(f"      跳过过小的 Data URI（{len(decoded_data)} 字节）")
            return None
        
        try:
            ext = self._get_extension_from_mime(mime_type)
            filename = f"datauri_{int(time.time() * 1000)}{ext}"
            save_path = save_dir / filename
            
            with open(save_path, "wb") as f:
                f.write(decoded_data)
            
            log.debug(f"      ✅ 保存 Data URI 资源: {filename} ({len(decoded_data)} 字节)")
            return filename
            
        except Exception as e:
            log.warning(f"⚠️ 保存 Data URI 失败: {e}")
            return None

    async def _download_resource(self, url: str, save_dir: Path) -> Optional[str]:
        """下载多媒体资源（支持 HTTP URL 和 Data URI）"""
        # 检查是否是 Data URI
        if self._is_data_uri(url):
            return await self._save_data_uri(url, save_dir)

        if not url.startswith(("http://", "https://")):
            log.warning(f"⚠️ 跳过无效 URL（非 HTTP/HTTPS 且非 Data URI）: {url[:100]}...")
            return None
        
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    parsed = urlparse(url)
                    filename = os.path.basename(parsed.path) or f"res_{int(time.time())}"
                    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
                    save_path = save_dir / filename
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    return filename
            return None
        except Exception as e:
            log.warning(f"⚠️ 下载失败 {url[:100]}...: {e}")
            return None

    async def _process_media_and_placeholders(self, html_content: str, base_url: str, save_dir: Path) -> tuple[str, List[Dict[str, str]]]:
        """替换 HTML 中的媒体为占位符"""
        soup = BeautifulSoup(html_content, "html.parser")
        resources = []
        
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                # 检查是否是 Data URI，如果是则直接使用，否则与基础 URL 拼接
                if self._is_data_uri(src):
                    full_url = src  # Data URI 本身就是完整的数据
                else:
                    full_url = urljoin(base_url, src)
                
                filename = await self._download_resource(full_url, save_dir)
                if filename:
                    img.replace_with(f"[Image: {filename}]")
                    # 对于 Data URI，不记录完整内容（太长），只记录类型
                    url_for_record = "data:..." if self._is_data_uri(src) else full_url
                    resources.append({"type": "image", "filename": filename, "url": url_for_record})
        
        for video in soup.find_all(["video", "source"]):
            src = video.get("src")
            if src:
                # 同样检查 Data URI
                if self._is_data_uri(src):
                    full_url = src
                else:
                    full_url = urljoin(base_url, src)
                
                filename = await self._download_resource(full_url, save_dir)
                if filename:
                    video.replace_with(f"[Video: {filename}]")
                    url_for_record = "data:..." if self._is_data_uri(src) else full_url
                    resources.append({"type": "video", "filename": filename, "url": url_for_record})

        return soup.get_text(separator="\n\n"), resources

    async def _rename_and_describe_images(self, resources: List[Dict[str, str]], resource_dir: Path, page_content: str) -> List[Dict[str, str]]:
        """LLM 辅助重命名图片并生成语义化描述"""
        image_resources = [r for r in resources if r["type"] == "image"]
        if not image_resources: return resources
        
        try:
            image_info = [f"图片{i}: {r['filename']}" for i, r in enumerate(image_resources)]
            prompt = f"根据网页内容：{page_content[:1500]}\n为这些图片生成英文语义文件名(img_xxx.png)和中文描述。JSON格式输出。"
            
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            # 解析并重命名文件（逻辑简化，实际建议增加 JSON 解析鲁棒性）
            # ... 此处逻辑同前 ...
            return resources
        except:
            return resources

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        使用 PyMuPDF (fitz) 从 PDF 文件中提取纯文本。
        
        Args:
            pdf_path: PDF 文件的绝对路径
            
        Returns:
            提取的纯文本内容，失败时返回 None
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            text_parts = []
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            doc.close()
            
            if not text_parts:
                log.warning(f"⚠️ PDF 未提取到文本内容: {pdf_path}")
                return None
            
            full_text = "\n\n".join(text_parts)
            log.info(f"      ✅ PDF 文本提取完成，共 {total_pages} 页，{len(full_text)} 字符")
            return full_text
            
        except Exception as e:
            log.error(f"❌ PDF 文本提取失败 ({pdf_path}): {e}")
            return None

    def _split_into_chunks(self, source_data: List[Dict], max_chars: int = 80000) -> List[List[Dict]]:
        """
        将源数据分成多个 chunk，每个 chunk 的总字符数不超过 max_chars。
        每个源作为一个整体，不会被拆分到不同 chunk 中。
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in source_data:
            item_size = len(item.get("raw_knowledge", "")) + len(item.get("url", "")) + 50  # 50 for formatting
            
            # 如果单个 item 就超过限制，需要截断
            if item_size > max_chars:
                # 如果当前 chunk 有内容，先保存
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                
                # 截断大型内容
                truncated_item = item.copy()
                truncated_item["raw_knowledge"] = item["raw_knowledge"][:max_chars - 100] + "\n\n... [内容已截断] ..."
                chunks.append([truncated_item])
                continue
            
            # 如果加入当前 item 会超过限制，开启新 chunk
            if current_size + item_size > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(item)
            current_size += item_size
        
        # 保存最后一个 chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    async def _extract_knowledge_from_chunk(self, route: str, chunk_data: List[Dict], chunk_idx: int, total_chunks: int) -> str:
        """从单个 chunk 中提取与子任务相关的知识点"""
        context_segments = []
        for item in chunk_data:
            context_segments.append(f"### Source: {item['url']}\n{item['raw_knowledge']}")
        
        chunk_context = "\n\n".join(context_segments)
        
        prompt = f"""
你是一个首席馆长。请从以下参考资料中提取与【研究子任务】相关的知识点。

研究子任务：{route}

参考资料（第 {chunk_idx + 1}/{total_chunks} 批）：
{chunk_context}

任务要求：
1. 仅提取与该子任务直接相关的内容，忽略无关信息。
2. 识别并保留原始数据中的多媒体占位符 [Image: xxx] 或 [Video: xxx]。
3. 知识表述要精炼、专业。
4. 使用 Markdown 格式的要点列表。
5. 如果本批资料中没有与子任务相关的内容，请回复"本批次无相关内容"。

请提取相关知识点：
"""
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": "你是一个专业的技术馆长，擅长从大量资料中提取特定主题的知识。"},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content

    async def _merge_knowledge_points(self, route: str, knowledge_points: List[str]) -> str:
        """将多个 chunk 提取的知识点合并成最终文档"""
        # 过滤掉"无相关内容"的结果
        valid_points = [kp for kp in knowledge_points if "无相关内容" not in kp and len(kp.strip()) > 20]
        
        if not valid_points:
            return f"# {route}\n\n暂无相关知识内容。"
        
        # 如果只有一个有效结果，直接返回
        if len(valid_points) == 1:
            return f"# {route}\n\n{valid_points[0]}"
        
        combined = "\n\n---\n\n".join([f"### 知识点批次 {i+1}\n{kp}" for i, kp in enumerate(valid_points)])
        
        prompt = f"""
你是一个首席馆长。请将以下多批次提取的知识点整合成一份完整、专业的知识文档。

研究子任务：{route}

各批次提取的知识点：
{combined}

任务要求：
1. 去除重复内容，合并相似观点。
2. 按逻辑结构组织内容（如：概述、原理、应用、挑战等）。
3. 保留所有多媒体占位符 [Image: xxx] 或 [Video: xxx]。
4. 使用专业的 Markdown 格式输出。
5. 确保内容完整、结构清晰。

请生成整合后的知识文档：
"""
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": "你是一个专业的技术馆长，擅长知识整合与文档编写。"},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

    async def _curate_single_knowledge_point(self, route: str, source_data: List[Dict], curated_dir: Path) -> str:
        """为每一个子任务独立总结并生成 MD（分 chunk 处理）"""
        log.info(f"🎯 正在针对任务独立建模: {route}")
        
        # 1. 将源数据分成多个 chunk
        chunks = self._split_into_chunks(source_data, max_chars=80000)
        log.info(f"      📦 数据已分成 {len(chunks)} 个批次进行处理")
        
        # 2. 对每个 chunk 提取知识点
        knowledge_points = []
        for idx, chunk in enumerate(chunks):
            log.info(f"      🔄 处理批次 [{idx+1}/{len(chunks)}]，包含 {len(chunk)} 个来源")
            try:
                kp = await self._extract_knowledge_from_chunk(route, chunk, idx, len(chunks))
                knowledge_points.append(kp)
                log.info(f"      ✅ 批次 [{idx+1}/{len(chunks)}] 完成，提取 {len(kp)} 字符")
            except Exception as e:
                log.warning(f"      ⚠️ 批次 [{idx+1}/{len(chunks)}] 处理失败: {e}")
                continue
        
        # 3. 合并所有知识点
        if len(knowledge_points) > 1:
            log.info(f"      🔗 正在合并 {len(knowledge_points)} 个批次的知识点...")
            content = await self._merge_knowledge_points(route, knowledge_points)
        elif len(knowledge_points) == 1:
            content = f"# {route}\n\n{knowledge_points[0]}"
        else:
            content = f"# {route}\n\n暂无相关知识内容。"
        
        # 4. 保存文件
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", route).replace(" ", "_")
        # 限制文件名长度
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        file_path = curated_dir / f"KNOWLEDGE_{safe_name}.md"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        log.info(f"      📝 知识文档已保存，共 {len(content)} 字符")
        return str(file_path)

    async def run(self, state: MainState, **kwargs) -> Dict[str, Any]:
        """主执行逻辑：汇总数据 -> 网页清洗 -> 逐任务独立建模"""
        log.info("=" * 60)
        log.info("🚀 [WebsearchCurator] 开始知识独立建模流程")
        log.info("=" * 60)
        
        if not isinstance(state, WebsearchKnowledgeState):
            log.error("❌ 状态类型错误，期望 WebsearchKnowledgeState")
            return {"status": "failed", "reason": "State Error"}

        raw_data = state.raw_data_store or []
        # 优先使用 original_research_routes（不会被 planner 清空的原始列表）
        # 回退到 research_routes 以兼容旧版
        research_routes = state.original_research_routes or state.research_routes or []
        
        log.info(f"📊 输入数据统计:")
        log.info(f"   - 原始数据源数量: {len(raw_data)}")
        log.info(f"   - 研究子任务数量: {len(research_routes)}")
        if state.original_research_routes:
            log.info(f"   - 使用原始任务列表 (original_research_routes)")
        else:
            log.warning(f"   - ⚠️ original_research_routes 为空，回退使用 research_routes")
        
        if not raw_data:
            log.warning("⚠️ 没有原始数据，跳过处理")
            return {"status": "skipped", "reason": "No raw data"}

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        curated_dir = self.output_dir / f"{timestamp}_curated"
        curated_dir.mkdir(exist_ok=True)
        resource_dir = curated_dir / "resources"
        resource_dir.mkdir(exist_ok=True)
        # 创建正文内容文件夹
        extracted_contents_dir = curated_dir / "extracted_contents"
        extracted_contents_dir.mkdir(exist_ok=True)
        log.info(f"📁 创建输出目录: {curated_dir}")
        log.info(f"📁 正文内容目录: {extracted_contents_dir}")

        # ==================== 阶段 1: 预处理 ====================
        log.info("-" * 60)
        log.info("📥 [阶段 1/2] 开始预处理原始数据...")
        log.info("-" * 60)
        
        processed_sources = []
        skipped_count = 0
        
        pdf_count = 0
        html_count = 0
        
        with tqdm(total=len(raw_data), desc="🔄 预处理数据", unit="条", ncols=100) as pbar:
            for idx, item in enumerate(raw_data):
                url = item.get("url", "未知URL")
                item_type = item.get("type", "")
                pdf_path = item.get("pdf_filepath")
                dom_path = item.get("dom_filepath")
                
                pbar.set_postfix_str(f"处理: {url[:40]}..." if len(url) > 40 else f"处理: {url}")
                
                # ---- PDF 类型处理 ----
                if item_type == "pdf" or pdf_path:
                    if not pdf_path or not os.path.exists(pdf_path):
                        log.warning(f"   ⚠️ [{idx+1}/{len(raw_data)}] 跳过 - PDF文件不存在: {pdf_path}")
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    log.info(f"   📑 [{idx+1}/{len(raw_data)}] 处理PDF: {os.path.basename(pdf_path)}")
                    
                    clean_text = self._extract_text_from_pdf(pdf_path)
                    if not clean_text:
                        log.warning(f"      ⚠️ PDF 文本提取为空，跳过")
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    processed_sources.append({
                        "url": url,
                        "raw_knowledge": clean_text,
                        "resources": []  # PDF 暂不提取内嵌媒体资源
                    })
                    pdf_count += 1
                    pbar.update(1)
                    continue
                
                # ---- HTML 类型处理（原有逻辑）----
                if not dom_path or not os.path.exists(dom_path):
                    log.warning(f"   ⚠️ [{idx+1}/{len(raw_data)}] 跳过 - DOM文件不存在: {dom_path}")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                log.info(f"   📄 [{idx+1}/{len(raw_data)}] 处理网页: {url}")
                
                with open(dom_path, "r", encoding="utf-8") as f:
                    html = f.read()
                log.debug(f"      - 原始 HTML 大小: {len(html)} 字符")
                
                # 简化 HTML
                simplified_html = self._simplify_html(html)
                log.debug(f"      - 简化后 HTML 大小: {len(simplified_html)} 字符")
                
                # 从 DOM 提取媒体资源（在正文提取之前，确保不丢失任何媒体）
                log.info(f"      🖼️ 从 DOM 提取媒体资源...")
                _, resources = await self._process_media_and_placeholders(simplified_html, url, resource_dir)
                log.info(f"      ✅ 媒体处理完成，发现 {len(resources)} 个资源")
                
                # 提取正文
                log.info(f"      🔍 调用 mineruhtml 提取正文...")
                main_html = await self._extract_content_with_mineruhtml(simplified_html) or html
                log.info(f"      ✅ 正文提取完成，大小: {len(main_html)} 字符")
                
                # 从正文提取纯文本内容
                soup = BeautifulSoup(main_html, "html.parser")
                clean_text = soup.get_text(separator="\n\n")
                
                processed_sources.append({
                    "url": url,
                    "raw_knowledge": clean_text,
                    "resources": resources
                })
                html_count += 1
                
                pbar.update(1)
        
        log.info(f"📊 预处理完成统计:")
        log.info(f"   - 成功处理: {len(processed_sources)} 条（HTML: {html_count}, PDF: {pdf_count}）")
        log.info(f"   - 跳过数量: {skipped_count} 个")
        total_resources = sum(len(s["resources"]) for s in processed_sources)
        log.info(f"   - 总媒体资源: {total_resources} 个")

        # ==================== 阶段 2: 知识建模 ====================
        log.info("-" * 60)
        log.info("📝 [阶段 2/2] 开始独立知识建模...")
        log.info("-" * 60)
        
        generated_md_files = []
        
        with tqdm(total=len(research_routes), desc="🧠 知识建模", unit="任务", ncols=100) as pbar:
            for idx, route in enumerate(research_routes):
                route_display = route[:35] + "..." if len(route) > 35 else route
                pbar.set_postfix_str(f"任务: {route_display}")
                
                log.info(f"   🎯 [{idx+1}/{len(research_routes)}] 建模任务: {route}")
                
                file_path = await self._curate_single_knowledge_point(route, processed_sources, curated_dir)
                generated_md_files.append(file_path)
                
                log.info(f"      ✅ 生成文件: {os.path.basename(file_path)}")
                
                pbar.update(1)

        # ==================== 完成汇总 ====================
        log.info("=" * 60)
        log.info("🎉 [WebsearchCurator] 知识建模流程完成!")
        log.info("=" * 60)
        log.info(f"📊 最终统计:")
        log.info(f"   - 生成知识文档: {len(generated_md_files)} 个")
        log.info(f"   - 输出目录: {curated_dir}")
        log.info(f"   - 资源目录: {resource_dir}")
        
        # 更新状态
        state.knowledge_base_summary = f"已完成 {len(generated_md_files)} 个知识点独立建模。"
        
        result_payload = {
            "status": "success",
            "curated_directory": str(curated_dir),
            "files_created": generated_md_files,
            "tasks_processed": len(research_routes),
            "sources_processed": len(processed_sources),
            "sources_skipped": skipped_count,
            "total_resources": total_resources
        }
        
        self.update_state_result(state, result_payload, {})
        return result_payload

    def update_state_result(self, state: MainState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        if hasattr(state, "agent_results"):
            state.agent_results[self.role_name] = {"result": result, "pre_tool_results": pre_tool_results}
        super().update_state_result(state, result, pre_tool_results)


def create_websearch_curator_agent(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> WebsearchChiefCuratorAgent:
    """
    便捷创建函数。
    """
    return WebsearchChiefCuratorAgent.create(tool_manager=tool_manager, **kwargs)