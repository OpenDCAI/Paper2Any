"""
WebsearchInitialAnalyzerAgent 直接调用测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  python tests/test_websearch_initial_analyzer.py

说明:
- 不使用 pytest，直接调用 initial_analyzer 节点（WebsearchInitialAnalyzerAgent）
- 使用 WebsearchKnowledgeState，模拟 workflow 中的 initial_analyzer_node 调用
- 将返回的完整结果 JSON 保存到 tests/debug_websearch_initial_analyzer_result.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中，避免从 tests 目录运行时 import 失败
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import WebsearchKnowledgeState
from dataflow_agent.agentroles.data_agents.websearch_initial_analyzer import (
    WebsearchInitialAnalyzerAgent,
)


async def run_initial_analyzer_node() -> dict:
    """
    直接调用 initial_analyzer 节点（模拟 workflow 中的 initial_analyzer_node）

    返回:
        dict: 包含执行结果的字典
    """
    # 使用指定的测试 URL
    test_url = "https://ai-bot.cn/webagent/"

    # 构造 WebsearchKnowledgeState（模拟 workflow 中的状态）
    state = WebsearchKnowledgeState()

    # 设置 input_urls（同时写到 state 和 request 中，和 workflow 逻辑保持一致）
    state.input_urls = [test_url]
    state.request.input_urls = [test_url]

    # 初始化 WebsearchInitialAnalyzerAgent（不注入 ToolManager，让 Agent 内部按默认逻辑执行）
    agent = WebsearchInitialAnalyzerAgent.create(tool_manager=None)

    print("🚀 开始执行 initial_analyzer 节点...")
    print(f"🔗 测试 URL: {test_url}")
    print("-" * 80)

    result = await agent.run(state)

    print("-" * 80)
    print("✅ initial_analyzer 节点执行完成")
    print(f"📊 执行状态: {result.get('status', 'unknown')}")

    if result.get("status") == "success":
        print(f"📁 存储路径: {result.get('session_dir', 'N/A')}")
        print(f"📌 研究子任务数量: {len(result.get('research_routes', []))}")
        print(f"📦 原始记录条数: {len(result.get('raw_data_store', []))}")

    # 将结果落盘，便于后续人工检查与回归
    debug_dir = Path("tests")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "debug_websearch_initial_analyzer_result.json"

    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 结果已保存至: {debug_path}")

    # 基础结构校验
    if not isinstance(result, dict):
        print("⚠️  警告: 返回结果不是字典类型")
    elif "status" not in result:
        print("⚠️  警告: 结果中缺少 status 字段")
    elif result.get("status") == "success":
        required_fields = ["session_dir", "research_routes", "raw_data_store"]
        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            print(f"⚠️  警告: 成功时缺少以下字段: {missing_fields}")
        else:
            print("✅ 结果结构校验通过")

    return result


async def main():
    """主函数：直接运行 initial_analyzer 节点"""
    try:
        result = await run_initial_analyzer_node()
        return result
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())


