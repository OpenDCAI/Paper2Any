"""
paper2figure research-assisted image workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
针对“仅输入绘图需求描述”的 Paper2Figure 增强流程：

1. 解析绘图需求，生成 arXiv 检索 query。
2. 检索、下载相关论文 PDF，并调用 MinerU 解析。
3. 抽取论文中的图像，使用 VLM 做结构化描述，同时总结 paper idea/method。
4. 分析参考论文 / 图像与当前绘图需求的关联性与参考价值。
5. 生成生图 prompt，并在支持时把相关 figure 作为参考图输入。
6. 使用 critic 评估生成结果；严重问题回到检索增强，局部问题走 image edit。

所有中间产物都会落在 ``state.result_path`` 或自动创建的 ``outputs`` 目录下。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
from openai import AsyncOpenAI

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import Paper2FigureState
from dataflow_agent.toolkits.rebuttal.arxiv import ArxivAgent
from dataflow_agent.workflow.registry import register

log = get_logger(__name__)


def _safe_filename(name: str, limit: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", (name or "").strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "item")[:limit]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def _safe_json_loads(text: str | Dict[str, Any] | List[Any] | None, default: Any) -> Any:
    if text is None:
        return default
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return default
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}|\[.*\]", stripped, flags=re.S)
        if not match:
            return default
        try:
            return json.loads(match.group(0))
        except Exception:
            return default


def _request_dict(state: Paper2FigureState) -> Dict[str, Any]:
    return dict(vars(state.request))


def _get_requirement_text(state: Paper2FigureState) -> str:
    request = state.request
    candidates = [
        getattr(request, "input_content", ""),
        getattr(request, "target", ""),
        getattr(state, "paper_idea", ""),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    raise ValueError("paper2fig_research_image 仅支持文本绘图需求，未找到有效 input_content/target。")


def _ensure_result_path(state: Paper2FigureState) -> Path:
    raw = getattr(state, "result_path", None)
    if raw:
        base_dir = Path(raw).expanduser().resolve()
    else:
        ts = int(time.time())
        project_root = Path(__file__).resolve().parents[2]
        base_dir = (project_root / "outputs" / "paper2figure_research_image" / str(ts)).resolve()
        state.result_path = str(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _ensure_runtime_store(state: Paper2FigureState) -> Dict[str, Any]:
    store = state.temp_data.setdefault("paper2fig_research_image", {})
    store.setdefault("active_papers", [])
    store.setdefault("backlog_papers", [])
    store.setdefault("rounds", [])
    return store


def _make_client(api_url: str, api_key: str) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(api_key=api_key, base_url=api_url, http_client=http_client)
    return client, http_client


async def _call_json_llm(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    client, http_client = _make_client(api_url, api_key)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _safe_json_loads(content, {})
        return parsed if isinstance(parsed, dict) else {}
    finally:
        await http_client.aclose()


async def _call_text_llm(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    client, http_client = _make_client(api_url, api_key)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()
    finally:
        await http_client.aclose()


async def _build_search_plan(state: Paper2FigureState, requirement: str) -> Dict[str, Any]:
    api_url = state.request.chat_api_url
    api_key = state.request.api_key
    model = state.request.get("search_model", state.request.model)
    system_prompt = (
        "你是论文检索规划助手。你要把科研绘图需求转为 arXiv 检索计划。"
        "只返回 JSON。"
    )
    user_prompt = f"""
当前绘图需求：
{requirement}

请输出 JSON：
{{
  "intent": "一句话总结绘图目标",
  "search_queries": ["最多3条 arXiv query"],
  "keywords": ["关键词"],
  "visual_focus": ["需要重点参考的视觉元素"]
}}
"""
    result = await _call_json_llm(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    queries = [q.strip() for q in result.get("search_queries", []) if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [requirement[:180]]
    result["search_queries"] = queries[:3]
    result.setdefault("intent", requirement[:120])
    result.setdefault("keywords", [])
    result.setdefault("visual_focus", [])
    return result


async def _search_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    def _run() -> List[Dict[str, Any]]:
        agent = ArxivAgent(max_results=max_results, download_mode="pdf_first")
        return agent.search_and_analyze(query)

    return await asyncio.to_thread(_run)


def _normalize_pdf_url(paper: Dict[str, Any]) -> str:
    pdf_url = (paper.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url
    abs_url = (paper.get("abs_url") or "").strip()
    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if abs_url and "arxiv.org/abs/" in abs_url:
        return abs_url.replace("/abs/", "/pdf/") + ".pdf"
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    raise ValueError(f"无法为论文生成 PDF 下载地址: {paper}")


async def _download_pdf_file(paper: Dict[str, Any], save_path: Path) -> Path:
    url = _normalize_pdf_url(paper)

    def _run() -> Path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "paper2figure-research/1.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            save_path.write_bytes(response.read())
        return save_path

    return await asyncio.to_thread(_run)


async def _parse_pdf_with_mineru(pdf_path: Path, output_dir: Path, port: int) -> Dict[str, Any]:
    from dataflow_agent.toolkits.multimodaltool.mineru_tool import run_mineru_pdf_extract_http

    markdown_text, auto_dir = await run_mineru_pdf_extract_http(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        port=port,
    )
    auto_dir_path = Path(auto_dir).resolve()
    markdown_path = auto_dir_path / f"{pdf_path.stem}.md"
    figure_paths = sorted((auto_dir_path / "images").glob("*.png"))
    return {
        "markdown_text": markdown_text,
        "auto_dir": str(auto_dir_path),
        "markdown_path": str(markdown_path),
        "figure_paths": [str(path.resolve()) for path in figure_paths],
    }


async def _summarize_paper_content(
    state: Paper2FigureState,
    *,
    requirement: str,
    paper: Dict[str, Any],
    markdown_text: str,
) -> Dict[str, Any]:
    api_url = state.request.chat_api_url
    api_key = state.request.api_key
    model = state.request.get("analysis_model", state.request.model)
    system_prompt = "你是科研论文分析助手。基于论文内容总结 idea、method，并判断其对绘图需求的参考价值。只返回 JSON。"
    user_prompt = f"""
绘图需求：
{requirement}

论文标题：{paper.get('title', '')}
论文摘要：{paper.get('abstract', '')}

论文解析内容（可能被截断）：
{markdown_text[:12000]}

请输出 JSON：
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "idea": "核心思想",
  "method": "方法概要",
  "visual_takeaways": ["对绘图有帮助的结构/布局/元素"],
  "relevance": "与需求的关联性说明"
}}
"""
    result = await _call_json_llm(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    result.setdefault("paper_id", paper.get("paper_id", ""))
    result.setdefault("idea", paper.get("abstract", ""))
    result.setdefault("method", "")
    result.setdefault("visual_takeaways", [])
    result.setdefault("relevance", "")
    return result


async def _describe_figure_image(
    state: Paper2FigureState,
    *,
    requirement: str,
    figure_id: str,
    image_path: str,
    paper_title: str,
) -> Dict[str, Any]:
    prompt = f"""
请分析这张来自论文《{paper_title}》的 figure，并输出 JSON：
{{
  "figure_id": "{figure_id}",
  "summary": "一句话概括图的作用",
  "layout": "版式结构",
  "key_elements": ["图中关键模块/箭头/分区"],
  "style_notes": ["配色/视觉风格/标注风格"],
  "reference_value": "它对当前绘图需求的参考价值",
  "reuse_for": ["layout|content|style|annotation"]
}}

当前绘图需求：{requirement}
"""
    from dataflow_agent.toolkits.multimodaltool.req_understanding import call_image_understanding_async

    result_text = await call_image_understanding_async(
        model=state.request.get("vlm_model", state.request.model),
        messages=[{"role": "user", "content": prompt}],
        api_url=state.request.chat_api_url,
        api_key=state.request.api_key,
        image_path=image_path,
        max_tokens=4096,
        temperature=0.1,
        timeout=180,
    )
    parsed = _safe_json_loads(result_text, {})
    if not isinstance(parsed, dict):
        parsed = {}
    parsed.setdefault("figure_id", figure_id)
    parsed.setdefault("summary", result_text[:300])
    parsed.setdefault("layout", "")
    parsed.setdefault("key_elements", [])
    parsed.setdefault("style_notes", [])
    parsed.setdefault("reference_value", "")
    parsed.setdefault("reuse_for", [])
    parsed["image_path"] = image_path
    return parsed


async def _analyze_reference_bundle(
    state: Paper2FigureState,
    *,
    requirement: str,
    paper_contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    api_url = state.request.chat_api_url
    api_key = state.request.api_key
    model = state.request.get("analysis_model", state.request.model)

    compact_context = []
    for paper in paper_contexts:
        compact_context.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "idea": paper.get("idea_method", {}).get("idea", ""),
                "method": paper.get("idea_method", {}).get("method", ""),
                "visual_takeaways": paper.get("idea_method", {}).get("visual_takeaways", []),
                "figures": [
                    {
                        "figure_id": fig.get("figure_id"),
                        "summary": fig.get("summary"),
                        "layout": fig.get("layout"),
                        "reference_value": fig.get("reference_value"),
                        "reuse_for": fig.get("reuse_for", []),
                    }
                    for fig in paper.get("figure_analyses", [])[:3]
                ],
            }
        )

    system_prompt = "你是科研绘图参考分析助手。请从参考论文中筛选最值得借鉴的内容和图像。只返回 JSON。"
    user_prompt = f"""
绘图需求：
{requirement}

参考论文与 figures：
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

请输出 JSON：
{{
  "reference_summary": "整体参考策略",
  "paper_rankings": [{{"paper_id": "", "score": 0, "reason": ""}}],
  "figure_rankings": [{{"figure_id": "", "score": 0, "reason": "", "reuse_for": ["layout"]}}],
  "recommended_reference_images": ["figure_id"],
  "gaps": ["当前参考还缺什么"]
}}
"""
    result = await _call_json_llm(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    result.setdefault("reference_summary", "")
    result.setdefault("paper_rankings", [])
    result.setdefault("figure_rankings", [])
    result.setdefault("recommended_reference_images", [])
    result.setdefault("gaps", [])
    return result


def _iter_figures(paper_contexts: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for paper in paper_contexts:
        for figure in paper.get("figure_analyses", []):
            yield figure


def _resolve_reference_images(
    paper_contexts: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    max_images: int,
) -> List[str]:
    figure_map = {fig.get("figure_id"): fig for fig in _iter_figures(paper_contexts)}
    picked: List[str] = []
    for figure_id in analysis.get("recommended_reference_images", []):
        image_path = (figure_map.get(figure_id) or {}).get("image_path")
        if image_path and os.path.exists(image_path):
            picked.append(image_path)
    if picked:
        return picked[:max_images]
    for ranked in analysis.get("figure_rankings", []):
        image_path = (figure_map.get(ranked.get("figure_id")) or {}).get("image_path")
        if image_path and os.path.exists(image_path):
            picked.append(image_path)
        if len(picked) >= max_images:
            break
    return picked[:max_images]


async def _compose_generation_plan(
    state: Paper2FigureState,
    *,
    requirement: str,
    paper_contexts: List[Dict[str, Any]],
    reference_analysis: Dict[str, Any],
    previous_feedback: Optional[Dict[str, Any]],
    round_index: int,
) -> Dict[str, Any]:
    api_url = state.request.chat_api_url
    api_key = state.request.api_key
    model = state.request.get("prompt_model", state.request.model)

    compact_context = []
    for paper in paper_contexts[:4]:
        compact_context.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "idea": paper.get("idea_method", {}).get("idea", ""),
                "method": paper.get("idea_method", {}).get("method", ""),
                "figures": [
                    {
                        "figure_id": fig.get("figure_id"),
                        "summary": fig.get("summary"),
                        "layout": fig.get("layout"),
                        "key_elements": fig.get("key_elements", []),
                        "style_notes": fig.get("style_notes", []),
                    }
                    for fig in paper.get("figure_analyses", [])[:3]
                ],
            }
        )

    system_prompt = "你是科研绘图 prompt 设计助手。请根据需求和参考论文，生成高质量图像生成/编辑 prompt。只返回 JSON。"
    user_prompt = f"""
绘图需求：
{requirement}

参考分析：
{json.dumps(reference_analysis, ensure_ascii=False, indent=2)}

参考内容：
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

上一轮 critic 反馈：
{json.dumps(previous_feedback or {}, ensure_ascii=False, indent=2)}

当前是第 {round_index} 轮。

请输出 JSON：
{{
  "final_prompt": "用于文生图的完整 prompt",
  "edit_prompt": "若需要局部优化，给出图像编辑 prompt；否则可为空",
  "negative_prompt": "可选",
  "reference_image_ids": ["推荐送入模型的参考图 figure_id"],
  "prompt_strategy": "说明如何借鉴 method/figure"
}}
"""
    result = await _call_json_llm(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    result.setdefault("final_prompt", requirement)
    result.setdefault("edit_prompt", "")
    result.setdefault("negative_prompt", "")
    result.setdefault("reference_image_ids", [])
    result.setdefault("prompt_strategy", "")
    return result


async def _generate_or_edit_figure(
    state: Paper2FigureState,
    *,
    prompt: str,
    save_path: Path,
    reference_images: List[str],
    previous_image_path: Optional[str],
    use_edit: bool,
) -> str:
    image_api_url = state.request.get("image_api_url", state.request.chat_api_url)
    image_api_key = state.request.get("image_api_key", state.request.api_key)
    image_model = state.request.get("gen_fig_model", getattr(state.request, "gen_fig_model", "gemini-2.5-flash-image-preview"))
    aspect_ratio = getattr(state, "aspect_ratio", None) or state.request.get("aspect_ratio", "16:9")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if reference_images and ("gemini" in image_model.lower()) and not use_edit:
        from dataflow_agent.toolkits.multimodaltool.req_img import gemini_multi_image_edit_async

        await gemini_multi_image_edit_async(
            prompt=prompt,
            image_paths=reference_images,
            save_path=str(save_path),
            api_url=image_api_url,
            api_key=image_api_key,
            model=image_model,
            aspect_ratio=aspect_ratio,
            timeout=300,
        )
        return str(save_path)

    from dataflow_agent.toolkits.multimodaltool.req_img import generate_or_edit_and_save_image_async

    await generate_or_edit_and_save_image_async(
        prompt=prompt,
        save_path=str(save_path),
        api_url=image_api_url,
        api_key=image_api_key,
        model=image_model,
        image_path=previous_image_path,
        use_edit=use_edit,
        aspect_ratio=aspect_ratio,
        timeout=300,
    )
    return str(save_path)


async def _critic_generated_figure(
    state: Paper2FigureState,
    *,
    requirement: str,
    image_path: str,
    prompt_payload: Dict[str, Any],
    reference_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""
请以科研绘图 critic 身份评估生成结果是否满足需求，并输出 JSON：
{{
  "decision": "accept|research_more|local_edit",
  "score": 0,
  "summary": "总体评价",
  "major_issues": ["严重问题"],
  "minor_issues": ["局部问题"],
  "edit_prompt": "若是 local_edit，给出直接可用于图像编辑的 prompt",
  "search_focus": ["若是 research_more，需要补充检索的方向"]
}}

绘图需求：
{requirement}

本轮 prompt：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

参考分析：
{json.dumps(reference_analysis, ensure_ascii=False, indent=2)}
"""
    from dataflow_agent.toolkits.multimodaltool.req_understanding import call_image_understanding_async

    result_text = await call_image_understanding_async(
        model=state.request.get("critic_model", state.request.get("vlm_model", state.request.model)),
        messages=[{"role": "user", "content": prompt}],
        api_url=state.request.chat_api_url,
        api_key=state.request.api_key,
        image_path=image_path,
        max_tokens=4096,
        temperature=0.1,
        timeout=180,
    )
    parsed = _safe_json_loads(result_text, {})
    if not isinstance(parsed, dict):
        parsed = {}
    parsed.setdefault("decision", "accept")
    parsed.setdefault("score", 80)
    parsed.setdefault("summary", result_text[:300])
    parsed.setdefault("major_issues", [])
    parsed.setdefault("minor_issues", [])
    parsed.setdefault("edit_prompt", "")
    parsed.setdefault("search_focus", [])
    return parsed


def _dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for paper in papers:
        key = paper.get("arxiv_id") or paper.get("abs_url") or paper.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


async def _collect_single_paper_context(
    state: Paper2FigureState,
    *,
    requirement: str,
    paper: Dict[str, Any],
    papers_dir: Path,
    max_figures_per_paper: int,
) -> Dict[str, Any]:
    paper_id = paper.get("arxiv_id") or _safe_filename(paper.get("title", "paper"), limit=40)
    paper = dict(paper)
    paper["paper_id"] = paper_id
    paper_dir = papers_dir / _safe_filename(f"{paper_id}_{paper.get('title', '')}")
    pdf_path = paper_dir / "paper.pdf"
    mineru_dir = paper_dir / "mineru"
    figure_analysis_dir = paper_dir / "figure_analyses"
    figure_analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_json(paper_dir / "metadata.json", paper)

    try:
        await _download_pdf_file(paper, pdf_path)
        pdf_status = "downloaded"
    except Exception as exc:
        log.warning(f"[paper2fig_research_image] 下载 PDF 失败: {paper.get('title')} - {exc}")
        pdf_status = f"download_failed: {exc}"

    parse_result: Dict[str, Any] = {
        "markdown_text": "",
        "auto_dir": "",
        "markdown_path": "",
        "figure_paths": [],
    }
    if pdf_path.exists():
        try:
            parse_result = await _parse_pdf_with_mineru(pdf_path, mineru_dir, port=state.mineru_port)
        except Exception as exc:
            log.warning(f"[paper2fig_research_image] MinerU 解析失败: {paper.get('title')} - {exc}")
            _write_text(paper_dir / "mineru_error.txt", str(exc))

    markdown_text = parse_result.get("markdown_text") or paper.get("abstract", "")
    idea_method = await _summarize_paper_content(
        state,
        requirement=requirement,
        paper=paper,
        markdown_text=markdown_text,
    )
    _write_json(paper_dir / "idea_method.json", idea_method)

    figure_analyses: List[Dict[str, Any]] = []
    for index, figure_path in enumerate(parse_result.get("figure_paths", [])[:max_figures_per_paper], start=1):
        figure_id = f"{paper_id}::figure_{index:03d}"
        analysis = await _describe_figure_image(
            state,
            requirement=requirement,
            figure_id=figure_id,
            image_path=figure_path,
            paper_title=paper.get("title", paper_id),
        )
        _write_json(figure_analysis_dir / f"figure_{index:03d}.json", analysis)
        figure_analyses.append(analysis)

    context = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": paper.get("authors", []),
        "published": paper.get("published", ""),
        "pdf_status": pdf_status,
        "pdf_path": str(pdf_path) if pdf_path.exists() else "",
        "mineru_dir": parse_result.get("auto_dir", ""),
        "markdown_path": parse_result.get("markdown_path", ""),
        "idea_method": idea_method,
        "figure_analyses": figure_analyses,
    }
    _write_json(paper_dir / "paper_context.json", context)
    return context


async def _ingest_paper_candidates(
    state: Paper2FigureState,
    *,
    requirement: str,
    candidates: List[Dict[str, Any]],
    papers_dir: Path,
    limit: int,
    max_figures_per_paper: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for paper in candidates[:limit]:
        context = await _collect_single_paper_context(
            state,
            requirement=requirement,
            paper=paper,
            papers_dir=papers_dir,
            max_figures_per_paper=max_figures_per_paper,
        )
        results.append(context)
    return results


@register("paper2fig_research_image")
def create_p2fig_research_image_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=Paper2FigureState, entry_point="_start_")

    async def _start_node(state: Paper2FigureState) -> Paper2FigureState:
        output_root = _ensure_result_path(state)
        requirement = _get_requirement_text(state)
        runtime = _ensure_runtime_store(state)
        runtime["requirement"] = requirement
        runtime["output_root"] = str(output_root)
        state.input_type = "TEXT"
        state.text_content = requirement

        _write_json(output_root / "input" / "request.json", _request_dict(state))
        _write_text(output_root / "input" / "requirement.txt", requirement)
        return state

    async def _search_and_parse_node(state: Paper2FigureState) -> Paper2FigureState:
        output_root = _ensure_result_path(state)
        runtime = _ensure_runtime_store(state)
        requirement = runtime["requirement"]

        query_plan = await _build_search_plan(state, requirement)
        _write_json(output_root / "search" / "query_plan.json", query_plan)

        max_results = int(state.request.get("search_top_k", 4))
        initial_reference_papers = int(state.request.get("initial_reference_papers", 2))
        max_figures_per_paper = int(state.request.get("max_figures_per_paper", 3))

        all_results: List[Dict[str, Any]] = []
        for query in query_plan.get("search_queries", []):
            papers = await _search_arxiv_papers(query, max_results=max_results)
            for paper in papers:
                paper["search_query"] = query
            all_results.extend(papers)

        deduped = _dedupe_papers(all_results)
        _write_json(output_root / "search" / "arxiv_results.json", deduped)

        active_candidates = deduped[:initial_reference_papers]
        backlog_candidates = deduped[initial_reference_papers:]
        papers_dir = output_root / "papers"
        active_contexts = await _ingest_paper_candidates(
            state,
            requirement=requirement,
            candidates=active_candidates,
            papers_dir=papers_dir,
            limit=initial_reference_papers,
            max_figures_per_paper=max_figures_per_paper,
        )

        runtime["query_plan"] = query_plan
        runtime["all_candidates"] = deduped
        runtime["active_papers"] = active_contexts
        runtime["backlog_papers"] = backlog_candidates

        state.agent_results["paper2fig_research_search"] = {
            "status": "success",
            "results": {
                "query_plan": query_plan,
                "candidate_count": len(deduped),
                "active_paper_count": len(active_contexts),
            },
        }
        return state

    async def _reference_analysis_node(state: Paper2FigureState) -> Paper2FigureState:
        output_root = _ensure_result_path(state)
        runtime = _ensure_runtime_store(state)
        requirement = runtime["requirement"]
        active_papers = runtime.get("active_papers", [])

        reference_analysis = await _analyze_reference_bundle(
            state,
            requirement=requirement,
            paper_contexts=active_papers,
        )
        runtime["reference_analysis"] = reference_analysis
        _write_json(output_root / "analysis" / "reference_analysis.json", reference_analysis)

        state.agent_results["paper2fig_research_reference_analysis"] = {
            "status": "success",
            "results": reference_analysis,
        }
        return state

    async def _generation_loop_node(state: Paper2FigureState) -> Paper2FigureState:
        output_root = _ensure_result_path(state)
        runtime = _ensure_runtime_store(state)
        requirement = runtime["requirement"]

        max_rounds = int(state.request.get("max_rounds", 3))
        max_reference_papers = int(state.request.get("max_reference_papers", 4))
        max_reference_images = int(state.request.get("max_reference_images", 3))
        max_figures_per_paper = int(state.request.get("max_figures_per_paper", 3))

        current_image_path: Optional[str] = None
        previous_feedback: Optional[Dict[str, Any]] = None
        accepted = False

        for round_index in range(1, max_rounds + 1):
            active_papers = runtime.get("active_papers", [])
            reference_analysis = runtime.get("reference_analysis", {})
            round_dir = output_root / "generation" / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            prompt_payload = await _compose_generation_plan(
                state,
                requirement=requirement,
                paper_contexts=active_papers,
                reference_analysis=reference_analysis,
                previous_feedback=previous_feedback,
                round_index=round_index,
            )
            _write_json(round_dir / "prompt.json", prompt_payload)

            if previous_feedback and previous_feedback.get("decision") == "local_edit":
                effective_prompt = prompt_payload.get("edit_prompt") or previous_feedback.get("edit_prompt") or prompt_payload.get("final_prompt")
                use_edit = bool(current_image_path)
            else:
                effective_prompt = prompt_payload.get("final_prompt") or requirement
                use_edit = False

            reference_images = _resolve_reference_images(
                active_papers,
                {
                    **reference_analysis,
                    "recommended_reference_images": prompt_payload.get("reference_image_ids")
                    or reference_analysis.get("recommended_reference_images", []),
                },
                max_images=max_reference_images,
            )
            _write_json(round_dir / "reference_images.json", reference_images)

            image_output_path = round_dir / "generated.png"
            current_image_path = await _generate_or_edit_figure(
                state,
                prompt=effective_prompt,
                save_path=image_output_path,
                reference_images=reference_images,
                previous_image_path=current_image_path,
                use_edit=use_edit,
            )

            critic_result = await _critic_generated_figure(
                state,
                requirement=requirement,
                image_path=current_image_path,
                prompt_payload=prompt_payload,
                reference_analysis=reference_analysis,
            )
            _write_json(round_dir / "critic.json", critic_result)

            round_record = {
                "round": round_index,
                "image_path": current_image_path,
                "prompt": prompt_payload,
                "critic": critic_result,
                "reference_count": len(active_papers),
            }
            runtime["rounds"].append(round_record)

            decision = critic_result.get("decision", "accept")
            previous_feedback = critic_result
            if decision == "accept":
                accepted = True
                break

            if decision == "research_more":
                backlog = runtime.get("backlog_papers", [])
                if backlog and len(active_papers) < max_reference_papers:
                    next_contexts = await _ingest_paper_candidates(
                        state,
                        requirement=requirement,
                        candidates=backlog,
                        papers_dir=output_root / "papers",
                        limit=1,
                        max_figures_per_paper=max_figures_per_paper,
                    )
                    runtime["active_papers"] = active_papers + next_contexts
                    runtime["backlog_papers"] = backlog[1:]
                runtime["reference_analysis"] = await _analyze_reference_bundle(
                    state,
                    requirement=requirement,
                    paper_contexts=runtime.get("active_papers", []),
                )
                _write_json(output_root / "analysis" / f"reference_analysis_round_{round_index:02d}.json", runtime["reference_analysis"])

        state.fig_draft_path = current_image_path or ""
        final_payload = {
            "accepted": accepted,
            "final_image_path": state.fig_draft_path,
            "rounds": runtime.get("rounds", []),
            "active_papers": [paper.get("paper_id") for paper in runtime.get("active_papers", [])],
        }
        _write_json(output_root / "final" / "result.json", final_payload)

        state.agent_results["paper2fig_research_image"] = {
            "status": "success",
            "results": final_payload,
        }
        return state

    nodes = {
        "_start_": _start_node,
        "search_parse": _search_and_parse_node,
        "reference_analysis": _reference_analysis_node,
        "generation_loop": _generation_loop_node,
    }
    edges = [
        ("_start_", "search_parse"),
        ("search_parse", "reference_analysis"),
        ("reference_analysis", "generation_loop"),
        ("generation_loop", "__end__"),
    ]
    builder.add_nodes(nodes).add_edges(edges)
    return builder

