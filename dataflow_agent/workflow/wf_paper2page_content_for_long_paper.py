from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_react_agent, create_simple_agent
from dataflow_agent.agentroles.paper2any_agents.content_expander_agent import create_content_expander
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.utils_markdown_sections import (
    build_section_batches,
    extract_markdown_sections,
    get_safe_outline_input_budget,
    is_probably_english,
)

from dataflow_agent.toolkits.multimodaltool.mineru_tool import run_mineru_pdf_extract_http

log = get_logger(__name__)


def _resolve_outline_model(state: Paper2FigureState) -> str | None:
    request = getattr(state, "request", None)
    request_model = str(getattr(request, "model", "") or "").strip()
    if request_model:
        return request_model

    explicit_outline_model = str(getattr(request, "outline_model", "") or "").strip()
    if explicit_outline_model:
        return explicit_outline_model

    configured_outline_model = os.getenv("PAPER2PPT_OUTLINE_MODEL", "").strip()
    if configured_outline_model:
        return configured_outline_model

    return None

"""
Workflow: paper2page_content_for_long_paper
Description: 专门用于处理长文档（如书籍、长论文、长篇报告）生成大量 PPT 页面的工作流。

Process:
1. Input Routing (_start_ -> _route_input):
   - PDF: 解析 PDF 获取全文 markdown (parse_pdf_pages_long)
   - TEXT: 直接接收文本输入 (prepare_text_input)
   - TOPIC: 根据主题生成长文 (generate_long_content_from_topic)

2. Content Expansion & Consolidation:
   - 对于 TEXT/TOPIC 输入，如果内容不足，会进行迭代扩写 (expand_text_iteratively / generate_long_content_from_topic)。
   - 所有来源的内容最终汇总到 state.long_text (consolidate_long_text)。
   - 再次检查总长度，如果不足目标页数所需字符数，进行补充扩写 (ensure_sufficient_content)。
     * 动态字符数计算：英文 ~3000 chars/page, 中文 ~800 chars/page。

3. Outline Generation (outline_for_long_text):
   - 根据 state.request.page_count (默认为 60) 和总文本长度，计算分批方案。
   - 将长文本切分为多个 batch。
   - 对每个 batch 调用 long_paper_outline_agent 生成对应页面的 outline (generate_outline_for_batch)。
   - 汇总所有批次的页面内容，并进行首尾衔接处理。

4. Output:
   - 生成的页面列表存储在 state.pagecontent。
"""

# ============================================================
# 辅助函数
# ============================================================

def _ensure_result_path(state: Paper2FigureState) -> str:
    """
    统一本次 workflow 的根输出目录
    """
    raw = getattr(state, "result_path", None)
    if raw:
        return raw

    root = get_project_root()
    ts = int(time.time())
    base_dir = (root / "outputs" / "paper2page_content_long" / str(ts)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


def _abs_path(p: str) -> str:
    if not p:
        return ""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return p


def _calculate_target_chars(target_pages: int, text: str = "") -> int:
    """
    根据页数和语言类型计算目标字符数
    英文：约 3000 chars/page
    中文：约 800 chars/page
    """
    is_en = is_probably_english(text)
    chars_per_page = 3000 if is_en else 800
    target = target_pages * chars_per_page
    # log.info(f"[long_paper] 目标计算: {target_pages}页, 英文={is_en}, 阈值={target} chars")
    return target


def _extract_plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "value",
            "content",
            "summary",
            "title",
            "label",
            "body",
            "description",
            "reason",
            "point",
            "raw",
        )
        for key in preferred_keys:
            extracted = _extract_plain_text(value.get(key))
            if extracted:
                return extracted
        for item in value.values():
            extracted = _extract_plain_text(item)
            if extracted:
                return extracted
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [_extract_plain_text(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    return str(value).strip()


def _normalize_outline_points(value: Any, *, limit: int = 5) -> List[str]:
    if isinstance(value, list):
        items = [_extract_plain_text(item) for item in value]
    else:
        items = [_extract_plain_text(value)]
    cleaned = [item for item in items if item]
    return cleaned[:limit]


def _clip_outline_point(text: str, *, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return (clipped or text[:limit]).rstrip(",;:.- ") + "..."


def _normalize_outline_page_item(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    title = _extract_plain_text(raw.get("title"))
    layout_description = _extract_plain_text(raw.get("layout_description"))
    key_points = _normalize_outline_points(raw.get("key_points"), limit=6)
    asset_ref_text = _extract_plain_text(raw.get("asset_ref"))

    # ReAct 失败或空对象时，不要把错误占位直接透传给前端。
    if raw.get("error") and not title and not key_points:
        return None
    if not title and not layout_description and not key_points and not asset_ref_text:
        return None

    normalized = dict(raw)
    normalized["title"] = title
    normalized["layout_description"] = layout_description
    normalized["key_points"] = key_points
    normalized["asset_ref"] = asset_ref_text or None
    return normalized


def _normalize_outline_pages(items: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in items:
        item = _normalize_outline_page_item(raw)
        if item is not None:
            normalized.append(item)
    return normalized


def _split_batch_text_into_units(content: str) -> List[str]:
    content = _extract_plain_text(content)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content or "") if part.strip()]
    if paragraphs:
        return paragraphs

    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if lines:
        return lines

    collapsed = re.sub(r"\s+", " ", str(content or "")).strip()
    if not collapsed:
        return []

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?\.])\s+", collapsed)
        if part.strip()
    ]
    if len(sentence_parts) > 1:
        return sentence_parts

    chunk_size = 220
    return [collapsed[i:i + chunk_size].strip() for i in range(0, len(collapsed), chunk_size) if collapsed[i:i + chunk_size].strip()]


def _build_fallback_pages_for_batch(
    *,
    batch: Dict[str, Any],
    existing_pages: List[Dict[str, Any]],
    page_budget: int,
    language: str,
) -> List[Dict[str, Any]]:
    if page_budget <= 0:
        return []

    batch_titles = [str(title).strip() for title in (batch.get("section_titles") or []) if str(title).strip()]
    units = _split_batch_text_into_units(str(batch.get("content") or ""))
    if not units:
        units = ["Content summary pending refinement."]

    missing = max(0, page_budget - len(existing_pages))
    if missing <= 0:
        return []

    fallback_pages: List[Dict[str, Any]] = []
    unit_count = len(units)
    chunk_size = max(1, (unit_count + missing - 1) // missing)
    use_chinese = str(language or "").strip().lower().startswith("zh")
    default_heading_prefix = "章节" if use_chinese else "Section"
    closing_title = "感谢聆听" if use_chinese else "Thank You"
    closing_points = ["感谢聆听", "欢迎交流与提问"] if use_chinese else ["Thank you for your attention.", "Questions & Discussion"]
    fallback_layout = (
        "结构化学术内容页，包含一个简洁摘要和若干支持要点，延续前后页叙事。"
        if use_chinese else
        "Structured academic content slide with one concise summary paragraph and supporting bullet points. Preserve narrative continuity with neighboring slides."
    )

    for fallback_idx in range(missing):
        if fallback_idx == missing - 1 and batch.get("is_last"):
            fallback_pages.append({
                "title": closing_title,
                "layout_description": (
                    "结束页，包含简短致谢与答疑提示。"
                    if use_chinese else
                    "Closing page with a concise thank-you message and optional Q&A prompt."
                ),
                "key_points": closing_points,
                "asset_ref": None,
            })
            continue

        start = fallback_idx * chunk_size
        end = min(unit_count, start + chunk_size)
        excerpt_units = units[start:end] or units[-1:]
        excerpt = " ".join(excerpt_units)
        heading = batch_titles[min(fallback_idx, len(batch_titles) - 1)] if batch_titles else f"{default_heading_prefix} {fallback_idx + 1}"
        points = [
            _clip_outline_point(text.strip())
            for text in excerpt_units[:4]
            if text.strip()
        ]
        if not points:
            points = [_clip_outline_point(excerpt[:220].strip())] if excerpt.strip() else ["Expand this section in the editor."]

        fallback_pages.append({
            "title": heading if fallback_idx == 0 else f"{heading} ({fallback_idx + 1})",
            "layout_description": fallback_layout,
            "key_points": points[:5],
            "asset_ref": None,
        })

    return fallback_pages


# ============================================================
# Workflow 工厂函数
# ============================================================

@register("paper2page_content_for_long_paper")
def create_paper2page_content_graph() -> GenericGraphBuilder:
    """
    长文本 Paper2PageContent Workflow
    专门处理长文本（50页+）的 PDF/TEXT/TOPIC 输入
    """
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    # ----------------------------------------------------------------------
    # PRE-TOOLS
    # ----------------------------------------------------------------------
    
    @builder.pre_tool("current_chunk", "long_paper_outline_agent")
    def _get_current_chunk(state: Paper2FigureState):
        """提供当前批次的文本内容"""
        return getattr(state, "current_chunk", "")

    @builder.pre_tool("batch_info", "long_paper_outline_agent")
    def _get_batch_info(state: Paper2FigureState):
        """提供批次信息，用于 prompt 生成"""
        idx = getattr(state, "chunk_index", 0)
        total = getattr(state, "total_chunks", 1)
        pages = getattr(state, "pages_to_generate", 10)
        return {
            "batch_index": idx + 1,
            "total_batches": total,
            "pages_to_generate": pages,
            "is_first": idx == 0,
            "is_last": idx == total - 1,
        }
    @builder.pre_tool("generation_round", "topic_writer")
    def _get_generation_round(state: Paper2FigureState):
        """提供 topic 生成轮次信息"""
        return getattr(state, "generation_round", 0)

    # ----------------------------------------------------------------------
    # Outline Refine Tools (Added for consistency with standard workflow)
    # ----------------------------------------------------------------------
    @builder.pre_tool("outline_feedback", "outline_refine_agent")
    def _get_outline_feedback(state: Paper2FigureState):
        return state.outline_feedback or ""

    @builder.pre_tool("minueru_output", "outline_refine_agent")
    def _get_mineru_markdown_for_refine(state: Paper2FigureState):
        return state.minueru_output or ""

    @builder.pre_tool("text_content", "outline_refine_agent")
    def _get_text_content_for_refine(state: Paper2FigureState):
        return state.text_content or ""

    @builder.pre_tool("pagecontent", "outline_refine_agent")
    def _get_pagecontent_for_refine(state: Paper2FigureState):
        return json.dumps(state.pagecontent or [], ensure_ascii=False)

    @builder.pre_tool("pagecontent_raw", "outline_refine_agent")
    def _get_pagecontent_raw_for_refine(state: Paper2FigureState):
        return state.pagecontent or []

    # ==============================================================
    # NODES
    # ==============================================================
    
    def _start_(state: Paper2FigureState) -> Paper2FigureState:
        """初始化 state"""
        _ensure_result_path(state)
        
        # 初始化字段
        state.minueru_output = state.minueru_output or ""
        state.text_content = state.text_content or ""
        state.pagecontent = state.pagecontent or []
        state.long_text = getattr(state, "long_text", "") or ""
        state.markdown_sections = getattr(state, "markdown_sections", []) or []
        state.current_section_titles = getattr(state, "current_section_titles", []) or []
        if not getattr(state, "max_batch_tokens", 0):
            state.max_batch_tokens = get_safe_outline_input_budget(getattr(state.request, "model", None))
        
        # 设置默认目标页数
        # 1. 优先从 request.page_count 获取
        if state.request and state.request.page_count:
             state.target_pages = state.request.page_count
        # 2. 否则查看 state 中是否有 target_pages
        elif not hasattr(state, "target_pages") or not state.target_pages:
            state.target_pages = 60  # 默认 60 页
        
        log.info(f"[long_paper] 目标页数: {state.target_pages}")
        return state

    async def parse_pdf_pages_long(state: Paper2FigureState) -> Paper2FigureState:
        """
        PDF 长文解析：读取完整 markdown，不做字符限制
        """
        paper_pdf_path = Path(_abs_path(state.paper_file))
        if not paper_pdf_path.exists():
            log.error(f"[long_paper] PDF 文件不存在: {paper_pdf_path}")
            state.long_text = ""
            return state

        result_root = Path(_ensure_result_path(state))
        result_root.mkdir(parents=True, exist_ok=True)

        pdf_stem = paper_pdf_path.stem
        paper_dir = result_root / pdf_stem
        auto_dir = paper_dir / "auto"

        # 触发 MinerU 解析
        if not auto_dir.exists():
            try:
                log.info(f"[long_paper] 开始 MinerU 解析: {paper_pdf_path}")
                mineru_port = int(getattr(state, "mineru_port", 8010) or 8010)
                await run_mineru_pdf_extract_http(
                    str(paper_pdf_path),
                    str(result_root),
                    port=mineru_port,
                )
            except Exception as e:
                log.error(f"[long_paper] MinerU 解析失败: {e}")
                state.long_text = ""
                return state

        auto_dir = (result_root / pdf_stem / "auto").resolve()
        markdown_path = auto_dir / f"{pdf_stem}.md"
        
        if not markdown_path.exists():
            log.error(f"[long_paper] Markdown 文件不存在: {markdown_path}")
            state.long_text = ""
            return state

        try:
            md = markdown_path.read_text(encoding="utf-8")
            log.info(f"[long_paper] 读取完整 markdown: {len(md)} 字符")
        except Exception as e:
            log.error(f"[long_paper] 读取 markdown 失败: {e}")
            md = ""

        # 不做裁剪，保留完整内容
        state.long_text = md
        state.minueru_output = md
        state.mineru_root = str(auto_dir)
        state.markdown_sections = extract_markdown_sections(md)
        
        return state

    async def prepare_text_input(state: Paper2FigureState) -> Paper2FigureState:
        """
        TEXT 输入：准备文本内容
        """
        log.info(f"[long_paper] TEXT 输入长度: {len(state.text_content)} 字符")
        return state

    async def expand_text_iteratively(state: Paper2FigureState) -> Paper2FigureState:
        """
        TEXT 循环扩写：扩写到足够长度
        """
        target_pages = getattr(state, "target_pages", 60)
        current_text = _extract_plain_text(state.text_content)
        
        # 动态计算目标
        target_chars = _calculate_target_chars(target_pages, current_text)
        
        log.info(f"[long_paper] 开始扩写，当前: {len(current_text)} 字符，目标: {target_chars} 字符 ({target_pages}页)")
        
        if len(current_text) >= target_chars:
             log.info(f"[long_paper] 初始长度已满足要求")
             return state

        max_rounds = state.max_rounds
        
        agent = create_simple_agent(
            name = "content_expander",
            model_name=_resolve_outline_model(state),
            temperature=0.7,
            parser_type="text",
        )
        
        for round_num in range(max_rounds):
            state.expansion_round = round_num
            state.text_content = current_text
            
            state = await agent.execute(state=state)
            
            expanded_text = _extract_plain_text(state.text_content)
            if expanded_text:
                current_text = expanded_text
            else:
                log.warning("[long_paper] 扩写结果为空，保留上一轮文本内容")
            
            # 重新计算目标（以防语言变化）
            target_chars = _calculate_target_chars(target_pages, current_text)
            
            log.info(f"[long_paper] 扩写轮次 {round_num + 1}/{max_rounds}: {len(current_text)} / {target_chars} 字符")
            
            if len(current_text) >= target_chars:
                log.info(f"[long_paper] 扩写完成，达到目标长度")
                break
        
        state.text_content = current_text
        return state

    async def generate_long_content_from_topic(state: Paper2FigureState) -> Paper2FigureState:
        """
        TOPIC 多轮生成长文
        """
        target_pages = getattr(state, "target_pages", 60)
        max_rounds = state.max_rounds
        
        current_text = _extract_plain_text(state.text_content)
        target_chars = target_pages * 800 
        
        log.info(f"[long_paper] 从 TOPIC 生成长文，当前: {len(current_text)} 字符")        
        agent = create_simple_agent(
            name="topic_writer",
            model_name=_resolve_outline_model(state),
            parser_type="text",
        )
        for round_num in range(max_rounds):
            state.generation_round = round_num
            state.text_content = current_text

            state = await agent.execute(state=state)
            
            generated_text = _extract_plain_text(state.text_content)
            if generated_text:
                current_text = generated_text
            else:
                log.warning("[long_paper] topic_writer 返回空内容，保留上一轮文本")
            
            # 动态更新目标
            target_chars = _calculate_target_chars(target_pages, current_text)
            log.info(f"[long_paper] 生成轮次 {round_num + 1}/{max_rounds}: {len(current_text)} / {target_chars} 字符")
            if len(current_text) >= target_chars:
                log.info(f"[long_paper] 生成完成，达到目标长度")
                break
        state.text_content = current_text
        return state

    async def outline_refine_agent(state: Paper2FigureState) -> Paper2FigureState:
        """
        outline_refine_agent: refine existing outline based on user feedback.
        """
        agent = create_react_agent(
            name="outline_refine_agent",
            model_name=_resolve_outline_model(state),
            parser_type="json",
            max_retries=5
        )
        state = await agent.execute(state=state)
        return state

    async def consolidate_long_text(state: Paper2FigureState) -> Paper2FigureState:
        """
        统一整合各来源的长文本到 state.long_text
        """
        if state.long_text:
            # PDF 路径已经有 long_text
            log.info(f"[long_paper] 使用 PDF markdown: {len(state.long_text)} 字符")
        elif state.text_content:
            # TEXT/TOPIC 路径使用 text_content
            state.long_text = _extract_plain_text(state.text_content)
            log.info(f"[long_paper] 使用 text_content: {len(state.long_text)} 字符")
        else:
            state.long_text = ""
            log.warning("[long_paper] 没有可用的长文本内容")
        
        state.markdown_sections = extract_markdown_sections(state.long_text)
        log.info(f"[long_paper] 提取到 {len(state.markdown_sections)} 个 section")
        return state

    async def ensure_sufficient_content(state: Paper2FigureState) -> Paper2FigureState:
        """
        确保内容足够长，不够则扩写
        """
        target_pages = getattr(state, "target_pages", 60)
        long_text = state.long_text or ""
        
        # 动态计算目标
        target_chars = _calculate_target_chars(target_pages, long_text)
        
        if len(long_text) >= target_chars:
            log.info(f"[long_paper] 内容充足: {len(long_text)} >= {target_chars} 字符")
            return state
        
        log.info(f"[long_paper] 内容不足({len(long_text)} < {target_chars} chars)，开始补充扩写")
        
        agent = create_content_expander(
            model_name=_resolve_outline_model(state),
            temperature=0.7,
            parser_type="text",
        )
        
        max_rounds = state.max_rounds 
        current_text = long_text
        
        for round_num in range(max_rounds):
            state.expansion_round = round_num
            state.text_content = current_text
            
            state = await agent.execute(state=state)
            
            expanded_text = _extract_plain_text(state.text_content)
            if expanded_text:
                current_text = expanded_text
            else:
                log.warning("[long_paper] 补充扩写结果为空，继续使用已有正文")
            
            # 重新计算目标
            target_chars = _calculate_target_chars(target_pages, current_text)
            
            log.info(f"[long_paper] 补充扩写轮次 {round_num + 1}/{max_rounds}: {len(current_text)} / {target_chars} 字符")
            
            if len(current_text) >= target_chars:
                break
        
        state.long_text = current_text
        log.info(f"[long_paper] 最终扩写后长度: {len(state.long_text)} 字符")
        return state

    async def generate_outline_for_batch(
        state: Paper2FigureState,
        batch: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        为单个批次生成 outline
        """
        # 深拷贝 state 以防止并发修改冲突
        state = copy.deepcopy(state)

        chunk_text = batch.get("content", "") or ""
        batch_idx = int(batch.get("batch_index", 0))
        total_batches = int(batch.get("total_batches", 1))
        pages_to_generate = int(batch.get("pages_to_generate", 1) or 1)
        section_titles = list(batch.get("section_titles", []) or [])

        log.critical(f"[chunk_text: ] {chunk_text[:200]}")
        
        # 临时设置当前批次信息
        state.current_chunk     = chunk_text
        state.chunk_index       = batch_idx
        state.total_chunks      = total_batches
        state.pages_to_generate = pages_to_generate
        state.current_section_titles = section_titles
        state.markdown_sections = list(batch.get("sections", []) or [])
        
        # 显式设置首尾状态，供 Agent 动态选择 Prompt
        state.is_first = (batch_idx == 0)
        state.is_last = (batch_idx == total_batches - 1)
        
        # 调用 long_paper_outline_agent
        agent = create_react_agent(
            name = "long_paper_outline_agent",
            model_name=_resolve_outline_model(state),
            temperature=0.1,
            max_retries=5,
            parser_type="json",
        )
        
        result_state = await agent.execute(state=state)
        
        # 提取生成的页面
        pages = result_state.pagecontent or []
        if not isinstance(pages, list):
            pages = [pages]
        pages = _normalize_outline_pages(pages)
        
        log.info(f"[long_paper] 批次 {batch_idx + 1}/{total_batches} 生成了 {len(pages)} 页")
        return pages

    async def outline_for_long_text(state: Paper2FigureState) -> Paper2FigureState:
        """
        对长文本按目标页数分批生成 outline（并行处理）
        """
        import asyncio
        
        long_text = state.long_text or ""
        target_pages = getattr(state, "target_pages", 60)
        pages_per_batch = state.pages_per_batch  # 每批次目标页数
        max_batch_tokens = max(8_000, int(getattr(state, "max_batch_tokens", 0) or 0))
        
        if not long_text:
            log.error("[long_paper] 没有长文本内容，无法生成 outline")
            state.pagecontent = []
            return state
        
        # 1. 确保内容充足
        target_chars = _calculate_target_chars(target_pages, long_text)
        if len(long_text) < target_chars:
            log.info(f"[long_paper] 内容不足({len(long_text)} < {target_chars})，触发扩写")
            state = await ensure_sufficient_content(state)
            long_text = state.long_text
        
        # 2. 先按 markdown section 切分，再基于 section 组批
        sections = state.markdown_sections or extract_markdown_sections(long_text)
        state.markdown_sections = sections
        if not sections:
            log.error("[long_paper] 未能从长文本中提取 section，无法生成 outline")
            state.pagecontent = []
            return state

        batches = build_section_batches(
            sections,
            target_pages=target_pages,
            pages_per_batch=pages_per_batch,
            max_batch_tokens=max_batch_tokens,
        )
        log.info(
            f"[long_paper] 分 {len(batches)} 批次，目标 {target_pages} 页，"
            f"max_batch_tokens={max_batch_tokens}"
        )
        
        # 3. 并行处理所有批次
        tasks = []
        batch_info = []  # 保存批次信息用于后续处理
        
        for batch in batches:
            batch_idx = int(batch.get("batch_index", 0))
            chunk_text = batch.get("content", "") or ""
            log.info(f"[long_paper] 准备批次 {batch_idx + 1}/{len(batches)}: "
                    f"sections={batch.get('section_titles', [])} ({len(chunk_text)} chars)")
            
            # 创建异步任务
            task = generate_outline_for_batch(
                state=state,
                batch=batch,
            )
            tasks.append(task)
            batch_info.append(batch)
        
        # 4. 并行执行所有任务
        log.info(f"[long_paper] 开始并行执行 {len(tasks)} 个批次...")
        results = await asyncio.gather(*tasks)
        log.info(f"[long_paper] 并行执行完成，收到 {len(results)} 个结果")
        
        normalized_batches: List[tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
        for chunk_pages, batch in zip(results, batch_info):
            page_budget = int(batch.get("pages_to_generate", 1) or 1)
            selected = list(_normalize_outline_pages(chunk_pages)[:page_budget])
            normalized_batches.append((selected, batch))

        if normalized_batches and all(len(selected) == 0 for selected, _ in normalized_batches):
            log.error("[long_paper] 所有批次均未生成有效 outline，拒绝使用全量 fallback 伪造大纲")
            state.pagecontent = []
            setattr(state, "outline_generation_error", "long_paper_outline_agent returned no valid pages for every batch")
            return state

        # 5. 按顺序处理结果
        all_pages = []
        for selected, batch in normalized_batches:
            batch_idx = int(batch.get("batch_index", 0))
            page_budget = int(batch.get("pages_to_generate", 1) or 1)
            raw_count = len(selected)
            if raw_count > page_budget:
                log.warning(
                    f"[long_paper] 批次 {batch_idx + 1}: 生成 {raw_count} 页，"
                    f"按预算保留 {page_budget} 页"
                )
            elif raw_count < page_budget:
                log.warning(
                    f"[long_paper] 批次 {batch_idx + 1}: 生成页数不足 {raw_count}/{page_budget}"
                )
                fallback_pages = _build_fallback_pages_for_batch(
                    batch=batch,
                    existing_pages=selected,
                    page_budget=page_budget,
                    language=getattr(getattr(state, "request", None), "language", "en"),
                )
                if fallback_pages:
                    log.warning(
                        f"[long_paper] 批次 {batch_idx + 1}: 使用 {len(fallback_pages)} 页 fallback 补齐到 {page_budget} 页"
                    )
                    selected.extend(fallback_pages)
            else:
                log.info(f"[long_paper] 批次 {batch_idx + 1}: 生成 {raw_count} 页，符合预算")
            all_pages.extend(selected)

        if len(all_pages) != target_pages:
            log.warning(f"[long_paper] 最终页数 {len(all_pages)} 与目标 {target_pages} 不完全一致")
        
        state.pagecontent = all_pages
        log.info(f"[long_paper] 并行处理完成，最终生成 {len(all_pages)} 页 pagecontent")
        
        return state

    # ==============================================================
    # 路由函数
    # ==============================================================
    
    def _route_input(state: Paper2FigureState) -> str:
        """根据输入类型路由到不同节点"""
        # 优先检查是否有反馈
        feedback = (state.outline_feedback or "").strip()
        if feedback and state.pagecontent:
            log.critical("走 OUTLINE 反馈修订路径 (Long Paper)")
            return "outline_refine_agent"

        t = getattr(state.request, "input_type", None) or getattr(state, "input_type", None) or ""
        t = str(t).upper().strip()
        
        if t == "PDF":
            log.info("[long_paper] 路由: PDF → parse_pdf_pages_long")
            return "parse_pdf_pages_long"
        elif t == "TEXT":
            log.info("[long_paper] 路由: TEXT → prepare_text_input")
            return "prepare_text_input"
        elif t == "TOPIC":
            log.info("[long_paper] 路由: TOPIC → generate_long_content_from_topic")
            return "generate_long_content_from_topic"
        else:
            log.error(f"[long_paper] 无效的 input_type: {t}，仅支持 PDF/TEXT/TOPIC")
            return "_end_"

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    
    nodes = {
        "_start_": _start_,
        
        # PDF 路径
        "parse_pdf_pages_long": parse_pdf_pages_long,
        
        # TEXT 路径
        "prepare_text_input": prepare_text_input,
        "expand_text_iteratively": expand_text_iteratively,
        
        # TOPIC 路径
        "generate_long_content_from_topic": generate_long_content_from_topic,
        
        # 统一处理
        "consolidate_long_text": consolidate_long_text,
        "outline_for_long_text": outline_for_long_text,

        # 修订
        "outline_refine_agent": outline_refine_agent,
        
        "_end_": lambda state: state,
    }

    edges = [
        # Refine → End
        ("outline_refine_agent", "_end_"),

        # PDF → 统一整合
        ("parse_pdf_pages_long", "consolidate_long_text"),
        
        # TEXT → 扩写 → 统一整合
        ("prepare_text_input", "expand_text_iteratively"),
        ("expand_text_iteratively", "consolidate_long_text"),
        
        # TOPIC → 生成 → 统一整合
        ("generate_long_content_from_topic", "consolidate_long_text"),
        
        # 统一整合 → 分批 outline → 结束
        ("consolidate_long_text", "outline_for_long_text"),
        ("outline_for_long_text", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", _route_input)
    
    return builder
