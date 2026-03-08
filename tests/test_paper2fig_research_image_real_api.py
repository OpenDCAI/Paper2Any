"""
Paper2Figure research-image workflow 真实 API 联调脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式：
  1) 直接运行（推荐）
     source /opt/conda/etc/profile.d/conda.sh && conda activate pzw-dev
     PYTHONPATH=. python tests/test_paper2fig_research_image_real_api.py

  2) 通过 pytest 运行（默认跳过，需显式开启）
     source /opt/conda/etc/profile.d/conda.sh && conda activate pzw-dev
     RUN_REAL_API_TESTS=1 PYTHONPATH=. pytest tests/test_paper2fig_research_image_real_api.py -q -s

说明：
- 文本分析默认使用：`http://123.129.219.111:3000/v1` + `gpt-4o-mini`
- 图像生成/编辑默认使用：`https://api.apiyi.com/v1` + `gemini-3.1-flash-image-preview`
- 所有中间产物都会写入 `outputs/paper2fig_research_image_real_api/` 下的时间戳目录
- 若本地未安装可选依赖（如 `Pillow` / `pdf2image` / MinerU 依赖），脚本会在运行时报错并打印上下文
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from dataflow_agent.workflow import run_workflow


DEFAULT_TEXT_API_URL = "http://123.129.219.111:3000/v1"
DEFAULT_TEXT_API_KEY = "sk-yBIfI1TcbftVVFy2uLNfvLRQxE9Z4WFjXEfBQbo2rP8lIDqO"
DEFAULT_TEXT_MODEL = "gpt-4o-mini"

DEFAULT_IMAGE_API_URL = "https://api.apiyi.com/v1"
DEFAULT_IMAGE_API_KEY = "sk-1XOECzAbmXWmUplV2f79Eb22E1014cFaA04e42B0A3B6F95d"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"

DEFAULT_REQUIREMENT = (
    "请生成一张科研方法图，展示一个多模态论文理解与科研绘图系统。"
    "图中需要包含：输入论文/需求、arXiv检索、PDF解析、Figure理解、"
    "Method总结、Reference分析、Prompt生成、Image Generation、Critic反馈回路。"
    "整体风格要求专业、简洁、适合论文或技术汇报中展示，文本使用中文。"
)


def _pick(obj, key: str):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def build_request() -> Paper2FigureRequest:
    req = Paper2FigureRequest(
        chat_api_url=os.getenv("P2A_TEXT_API_URL", DEFAULT_TEXT_API_URL),
        api_key=os.getenv("P2A_TEXT_API_KEY", DEFAULT_TEXT_API_KEY),
        model=os.getenv("P2A_TEXT_MODEL", DEFAULT_TEXT_MODEL),
        target=os.getenv("P2A_REQUIREMENT", DEFAULT_REQUIREMENT),
        input_type="TEXT",
        input_content=os.getenv("P2A_REQUIREMENT", DEFAULT_REQUIREMENT),
        gen_fig_model=os.getenv("P2A_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
        vlm_model=os.getenv("P2A_VLM_MODEL", DEFAULT_TEXT_MODEL),
    )

    req.image_api_url = os.getenv("P2A_IMAGE_API_URL", DEFAULT_IMAGE_API_URL)
    req.image_api_key = os.getenv("P2A_IMAGE_API_KEY", DEFAULT_IMAGE_API_KEY)

    req.initial_reference_papers = int(os.getenv("P2A_INITIAL_REFERENCE_PAPERS", "2"))
    req.max_reference_papers = int(os.getenv("P2A_MAX_REFERENCE_PAPERS", "4"))
    req.max_rounds = int(os.getenv("P2A_MAX_ROUNDS", "2"))
    req.max_figures_per_paper = int(os.getenv("P2A_MAX_FIGURES_PER_PAPER", "2"))
    req.max_reference_images = int(os.getenv("P2A_MAX_REFERENCE_IMAGES", "2"))
    req.search_top_k = int(os.getenv("P2A_SEARCH_TOP_K", "4"))
    req.search_model = os.getenv("P2A_SEARCH_MODEL", DEFAULT_TEXT_MODEL)
    req.analysis_model = os.getenv("P2A_ANALYSIS_MODEL", DEFAULT_TEXT_MODEL)
    req.prompt_model = os.getenv("P2A_PROMPT_MODEL", DEFAULT_TEXT_MODEL)
    req.critic_model = os.getenv("P2A_CRITIC_MODEL", DEFAULT_TEXT_MODEL)
    req.language = os.getenv("P2A_LANGUAGE", "zh")
    return req


async def run_real_pipeline() -> tuple[object, Path]:
    req = build_request()
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "paper2fig_research_image_real_api" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    state = Paper2FigureState(
        request=req,
        result_path=str(output_dir),
        messages=[],
        agent_results={},
    )

    print("=" * 88)
    print("🚀 开始执行真实 API 联调")
    print(f"📁 输出目录: {output_dir}")
    print(f"🧠 文本模型: {req.model}")
    print(f"🖼️ 图像模型: {req.gen_fig_model}")
    print(f"🔎 检索数量: top_k={req.search_top_k}, init_refs={req.initial_reference_papers}, max_refs={req.max_reference_papers}")
    print("=" * 88)

    final_state = await run_workflow("paper2fig_research_image", state)

    result = {
        "fig_draft_path": _pick(final_state, "fig_draft_path"),
        "result_path": _pick(final_state, "result_path"),
        "agent_results": _pick(final_state, "agent_results"),
    }
    result_path = output_dir / "real_api_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ 真实 API 联调完成")
    print(f"🖼️ 最终图片: {result['fig_draft_path']}")
    print(f"📦 总输出目录: {result['result_path']}")
    print(f"📝 摘要文件: {result_path}")
    return final_state, output_dir


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_REAL_API_TESTS", "0") != "1",
    reason="真实 API 联调默认跳过；设置 RUN_REAL_API_TESTS=1 后运行",
)
async def test_paper2fig_research_image_real_api() -> None:
    final_state, output_dir = await run_real_pipeline()

    fig_path = Path(_pick(final_state, "fig_draft_path"))
    assert fig_path.exists(), f"最终图片不存在: {fig_path}"
    assert output_dir.exists(), f"输出目录不存在: {output_dir}"
    assert (output_dir / "input" / "request.json").exists()
    assert (output_dir / "search" / "query_plan.json").exists()
    assert (output_dir / "final" / "result.json").exists()


if __name__ == "__main__":
    try:
        asyncio.run(run_real_pipeline())
    except Exception as exc:
        print(f"❌ 真实 API 联调失败: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise
