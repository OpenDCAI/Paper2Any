"""
WebsearchResearcherAgent 直接调用测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  python tests/test_websearch_researcher.py

说明:
- 不使用 pytest，直接调用 research 节点（WebsearchResearcherAgent）
- 真实调用 LLM 与浏览器（依赖环境变量 DF_API_URL、DF_API_KEY 等）
- 使用 WebsearchKnowledgeState，模拟 workflow 中的 web_researcher_node 调用
- 将返回的完整结果 JSON 保存到 tests/debug_websearch_researcher_result.json
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
from dataflow_agent.agentroles.data_agents.websearch_researcher import (
    WebsearchResearcherAgent,
)


async def run_web_researcher_node() -> dict:
    """
    直接调用 research 节点（模拟 workflow 中的 web_researcher_node）
    
    返回:
        dict: 包含执行结果的字典
    """
    # 使用真实的 user_query（可以根据需要修改）
    user_query = "请帮我在网页上查找一篇关于 diffusion model 的公开论文（例如 arXiv），并下载可用的 PDF 或相关附件。"

    # 构造 WebsearchKnowledgeState（模拟 workflow 中的状态）
    state = WebsearchKnowledgeState()
    
    # 设置当前任务（WebsearchResearcherAgent 会从 state.current_task 读取）
    setattr(state, "current_task", user_query)
    
    # 可以设置 research_routes（如果有的话）
    # state.research_routes = ["研究路线1", "研究路线2"]
    
    # 初始化 WebsearchResearcherAgent（不注入 ToolManager，让 Agent 内部按默认逻辑执行）
    agent = WebsearchResearcherAgent.create(tool_manager=None)

    # 直接调用 agent.run 方法（内部会自己构造 LLM 与 Browser）
    print(f"🚀 开始执行 research 节点...")
    print(f"📝 任务描述: {user_query}")
    print("-" * 80)
    
    result = await agent.run(state)
    
    print("-" * 80)
    print(f"✅ Research 节点执行完成")
    print(f"📊 执行状态: {result.get('status', 'unknown')}")
    
    if result.get("status") == "success":
        print(f"📄 摘要: {result.get('summary', 'N/A')[:200]}...")
        print(f"📁 存储路径: {result.get('storage_path', 'N/A')}")
        print(f"📎 捕获文件数: {result.get('captured_files_count', 0)}")
    
    # 将结果落盘，便于后续人工检查与回归
    debug_dir = Path("tests")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "debug_websearch_researcher_result.json"
    
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存至: {debug_path}")
    
    # 基础结构校验
    if not isinstance(result, dict):
        print("⚠️  警告: 返回结果不是字典类型")
    elif "status" not in result:
        print("⚠️  警告: 结果中缺少 status 字段")
    elif result.get("status") == "success":
        required_fields = ["summary", "storage_path", "captured_files"]
        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            print(f"⚠️  警告: 成功时缺少以下字段: {missing_fields}")
        else:
            print("✅ 结果结构校验通过")
    
    return result


async def main():
    """主函数：直接运行 research 节点"""
    try:
        result = await run_web_researcher_node()
        return result
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())


