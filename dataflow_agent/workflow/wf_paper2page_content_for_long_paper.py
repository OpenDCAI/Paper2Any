from __future__ import annotations

import os
import time
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

from dataflow_agent.state import Paper2FigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_react_agent, create_simple_agent
from dataflow_agent.agentroles.paper2any_agents.long_paper_outline_agent import create_long_paper_outline_agent
from dataflow_agent.agentroles.paper2any_agents.content_expander_agent import create_content_expander
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

from dataflow_agent.toolkits.imtool.mineru_tool import run_mineru_pdf_extract

log = get_logger(__name__)

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


def _is_english_text(text: str | Any) -> bool:
    """简单判断文本是否主要为英文（ASCII占比 > 80%）"""
    if not text:
        # 默认非英文（中文）以保持较低的字符阈值，避免误判导致过度扩写
        return False
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return False

    # 统计前 5000 个字符即可
    sample = text[:5000]
    ascii_count = sum(1 for c in sample if ord(c) < 128)
    return (ascii_count / len(sample)) > 0.8


def _calculate_target_chars(target_pages: int, text: str = "") -> int:
    """
    根据页数和语言类型计算目标字符数
    英文：约 3000 chars/page
    中文：约 800 chars/page
    """
    is_en = _is_english_text(text)
    chars_per_page = 3000 if is_en else 800
    target = target_pages * chars_per_page
    # log.info(f"[long_paper] 目标计算: {target_pages}页, 英文={is_en}, 阈值={target} chars")
    return target


def split_text_by_chars(text: str, chunk_size: int = 30000) -> List[str]:
    """
    按字符数切分文本，尽量在段落边界切分
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        
        # 向后查找段落边界（双换行符），但不超过500字符
        if end_pos < len(text):
            boundary = text.rfind('\n\n', current_pos, end_pos + 500)
            if boundary > current_pos:
                end_pos = boundary
        
        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return chunks


def calculate_batches(
    total_chars: int,
    target_pages: int,
    pages_per_batch: int = 10
) -> List[Tuple[int, int, int, bool, bool]]:
    """
    计算分批方案
    
    Args:
        total_chars: 总字符数
        target_pages: 目标总页数
        pages_per_batch: 每批次目标页数
    
    Returns:
        [(start_char, end_char, batch_idx, is_first, is_last), ...]
    """
    num_batches = max(1, (target_pages + pages_per_batch - 1) // pages_per_batch)
    chars_per_batch = total_chars // num_batches
    
    batches = []
    for i in range(num_batches):
        start_char = i * chars_per_batch
        end_char = min((i + 1) * chars_per_batch, total_chars)
        is_first = (i == 0)
        is_last = (i == num_batches - 1)
        batches.append((start_char, end_char, i, is_first, is_last))
    
    return batches


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
                run_mineru_pdf_extract(str(paper_pdf_path), str(result_root), "modelscope")
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
        state.mineru_root = str(auto_dir)
        
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
        current_text = state.text_content or ""
        
        # 动态计算目标
        target_chars = _calculate_target_chars(target_pages, current_text)
        
        log.info(f"[long_paper] 开始扩写，当前: {len(current_text)} 字符，目标: {target_chars} 字符 ({target_pages}页)")
        
        if len(current_text) >= target_chars:
             log.info(f"[long_paper] 初始长度已满足要求")
             return state

        max_rounds = state.max_rounds
        
        agent = create_simple_agent(
            name = "content_expander",
            temperature=0.7,
            parser_type="text",
        )
        
        for round_num in range(max_rounds):
            state.expansion_round = round_num
            state.text_content = current_text
            
            state = await agent.execute(state=state)
            
            # 增加类型检查，防止 agent 返回 dict 导致后续切片报错
            # 用户要求：直接把字典当字符串
            current_text = str(state.text_content) if state.text_content else ""
            
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
        
        current_text = state.text_content or ""
        target_chars = target_pages * 800 
        
        log.info(f"[long_paper] 从 TOPIC 生成长文，当前: {len(current_text)} 字符")        
        agent = create_simple_agent(
            name="topic_writer",
            parser_type="text",
        )
        for round_num in range(max_rounds):
            state.generation_round = round_num
            state.text_content = current_text

            state = await agent.execute(state=state)
            
            current_text = str(state.text_content) if state.text_content else ""
            
            # 动态更新目标
            target_chars = _calculate_target_chars(target_pages, current_text)
            log.info(f"[long_paper] 生成轮次 {round_num + 1}/{max_rounds}: {len(current_text)} / {target_chars} 字符")
            if len(current_text) >= target_chars:
                log.info(f"[long_paper] 生成完成，达到目标长度")
                break
        state.text_content = current_text
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
            state.long_text = state.text_content
            log.info(f"[long_paper] 使用 text_content: {len(state.long_text)} 字符")
        else:
            state.long_text = ""
            log.warning("[long_paper] 没有可用的长文本内容")
        
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
            temperature=0.7,
            parser_type="text",
        )
        
        max_rounds = state.max_rounds 
        current_text = long_text
        
        for round_num in range(max_rounds):
            state.expansion_round = round_num
            state.text_content = current_text
            
            state = await agent.execute(state=state)
            
            # 增加类型检查
            # 用户要求：直接把字典当字符串
            current_text = str(state.text_content) if state.text_content else ""
            
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
        chunk_text: str,
        batch_idx: int,
        total_batches: int,
        pages_to_generate: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        为单个批次生成 outline
        """
        # 深拷贝 state 以防止并发修改冲突
        state = copy.deepcopy(state)

        log.critical(f"[chunk_text: ] {chunk_text[:200]}")
        
        # 临时设置当前批次信息
        state.current_chunk     = chunk_text
        state.chunk_index       = batch_idx
        state.total_chunks      = total_batches
        state.pages_to_generate = pages_to_generate
        
        # 显式设置首尾状态，供 Agent 动态选择 Prompt
        state.is_first = (batch_idx == 0)
        state.is_last = (batch_idx == total_batches - 1)
        
        # 调用 long_paper_outline_agent
        agent = create_react_agent(
            name = "long_paper_outline_agent",
            temperature=0.1,
            max_retries=5,
            parser_type="json",
        )
        
        result_state = await agent.execute(state=state)
        
        # 提取生成的页面
        pages = result_state.pagecontent or []
        if not isinstance(pages, list):
            pages = [pages]
        
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
        pages_to_generate = state.pages_to_generate  # 每批次让 agent 生成的页数（含首尾）
        
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
        
        # 2. 计算分批方案
        batches = calculate_batches(len(long_text), target_pages, pages_per_batch)
        log.info(f"[long_paper] 分 {len(batches)} 批次，目标 {target_pages} 页，将并行处理")
        
        # 3. 并行处理所有批次
        tasks = []
        batch_info = []  # 保存批次信息用于后续处理
        
        for start_char, end_char, batch_idx, is_first, is_last in batches:
            chunk_text = long_text[start_char:end_char]
            
            log.info(f"[long_paper] 准备批次 {batch_idx + 1}/{len(batches)}: "
                    f"字符 {start_char}-{end_char} ({len(chunk_text)} chars)")
            
            # 创建异步任务
            task = generate_outline_for_batch(
                state=state,
                chunk_text=chunk_text,
                batch_idx=batch_idx,
                total_batches=len(batches),
                pages_to_generate=pages_to_generate,
            )
            tasks.append(task)
            batch_info.append((batch_idx, is_first, is_last))
        
        # 4. 并行执行所有任务
        log.info(f"[long_paper] 开始并行执行 {len(tasks)} 个批次...")
        results = await asyncio.gather(*tasks)
        log.info(f"[long_paper] 并行执行完成，收到 {len(results)} 个结果")
        
        # 5. 按顺序处理结果
        all_pages = []
        for idx, (chunk_pages, (batch_idx, is_first, is_last)) in enumerate(zip(results, batch_info)):
            # 不再进行裁剪，直接保留所有生成的页面
            selected = chunk_pages
            log.info(f"[long_paper] 批次 {batch_idx + 1}: 生成 {len(chunk_pages)} 页，全部保留")
            all_pages.extend(selected)
        
        # 6. 确保总页数符合要求
        if len(all_pages) > target_pages:
            log.warning(f"[long_paper] 生成页数超出目标({len(all_pages)} > {target_pages})，截断")
            all_pages = all_pages[:target_pages]
        elif len(all_pages) < target_pages:
            log.warning(f"[long_paper] 生成页数不足: {len(all_pages)}/{target_pages}")
        
        state.pagecontent = all_pages
        log.info(f"[long_paper] 并行处理完成，最终生成 {len(all_pages)} 页 pagecontent")
        
        return state

    # ==============================================================
    # 路由函数
    # ==============================================================
    
    def _route_input(state: Paper2FigureState) -> str:
        """根据输入类型路由到不同节点"""
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
        
        "_end_": lambda state: state,
    }

    edges = [
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
