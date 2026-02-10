"""
WebsearchChiefCuratorAgent 直接调用测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
运行方式:
  python tests/test_websearch_curator.py

说明:
- 不使用 pytest，直接调用 chief_curator 节点（WebsearchChiefCuratorAgent）
- 使用 WebsearchKnowledgeState，模拟 workflow 中的 chief_curator_node 调用
- 使用 tests/raw_data_store 中的真实数据进行测试
- 将返回的完整结果 JSON 保存到 tests/debug_websearch_curator_result.json
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from datetime import datetime

# 确保项目根目录在 sys.path 中
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import WebsearchKnowledgeState
from dataflow_agent.agentroles.data_agents.websearch_curator import (
    WebsearchChiefCuratorAgent,
)


def discover_raw_data_store(raw_data_dir: Path) -> tuple[list[dict], list[str]]:
    """
    扫描 raw_data_store 目录，构建 raw_data_store 和 research_routes

    Returns:
        tuple: (raw_data_store, research_routes)
    """
    raw_data_store = []
    research_routes = []
    
    if not raw_data_dir.exists():
        print(f"⚠️ 目录不存在: {raw_data_dir}")
        return raw_data_store, research_routes

    # 遍历所有子目录
    for session_dir in sorted(raw_data_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        
        # 1. 检查 initial_analysis 目录，读取 research_routes
        if "initial_analysis" in session_dir.name:
            summary_file = session_dir / "analysis_summary.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                
                # 提取 research_routes（只取一次，避免重复）
                if not research_routes:
                    research_routes = summary.get("research_routes", [])
                    print(f"📋 从 {summary_file.name} 读取到 {len(research_routes)} 个研究子任务")
                
                # 从 raw_data_records 构建数据
                for record in summary.get("raw_data_records", []):
                    dom_path = record.get("dom_filepath", "")
                    # 修正相对路径 - 需要基于项目根目录
                    if dom_path and not os.path.isabs(dom_path):
                        full_dom_path = PROJECT_ROOT / dom_path
                    else:
                        full_dom_path = Path(dom_path)
                    
                    if full_dom_path.exists():
                        raw_data_store.append({
                            "url": record.get("url", ""),
                            "dom_filepath": str(full_dom_path),
                            "timestamp": record.get("timestamp", datetime.now().isoformat()),
                            "source": "initial_analysis"
                        })

        # 2. 检查子任务目录的 dom_snapshots
        dom_snapshots_dir = session_dir / "dom_snapshots"
        if dom_snapshots_dir.exists() and dom_snapshots_dir.is_dir():
            # 读取 accessibility_trees.json 获取 URL 信息
            acc_trees_file = session_dir / "accessibility_trees.json"
            url_map = {}
            if acc_trees_file.exists():
                with open(acc_trees_file, "r", encoding="utf-8") as f:
                    acc_data = json.load(f)
                    for tree in acc_data.get("accessibility_trees", []):
                        step = tree.get("step")
                        url = tree.get("url", "")
                        if step and url:
                            url_map[step] = url

            # 遍历 dom 快照
            for html_file in sorted(dom_snapshots_dir.glob("*.html")):
                # 跳过 blank 页面
                if "_blank" in html_file.name:
                    continue
                
                # 从文件名提取 step 编号
                # 格式: step_001_arxiv_org_index.html
                parts = html_file.stem.split("_")
                if len(parts) >= 2 and parts[0] == "step":
                    try:
                        step_num = int(parts[1])
                        url = url_map.get(step_num, f"https://{parts[2]}" if len(parts) > 2 else "unknown")
                    except (ValueError, IndexError):
                        url = "unknown"
                else:
                    url = "unknown"

                raw_data_store.append({
                    "url": url,
                    "dom_filepath": str(html_file),
                    "timestamp": datetime.now().isoformat(),
                    "source": session_dir.name
                })

    print(f"📦 共发现 {len(raw_data_store)} 个原始数据源")
    return raw_data_store, research_routes


async def run_curator_node() -> dict:
    """
    直接调用 chief_curator 节点，使用真实数据
    """
    # 1. 设置真实数据目录
    raw_data_dir = PROJECT_ROOT / "tests" / "raw_data_store"
    
    print("=" * 80)
    print("🔍 正在扫描真实数据目录...")
    print(f"📁 数据目录: {raw_data_dir}")
    print("=" * 80)

    # 2. 发现数据
    raw_data_store, research_routes = discover_raw_data_store(raw_data_dir)
    
    if not raw_data_store:
        print("❌ 未发现任何有效数据")
        return {"status": "failed", "reason": "No raw data found"}

    if not research_routes:
        print("⚠️ 未找到 research_routes，使用默认子任务")
        research_routes = ["分析网页内容并提取关键信息"]

    # 打印发现的数据摘要
    print("\n📊 数据摘要:")
    print(f"  - 原始数据源数量: {len(raw_data_store)}")
    print(f"  - 研究子任务数量: {len(research_routes)}")
    
    print("\n📋 研究子任务列表:")
    for i, route in enumerate(research_routes[:5], 1):  # 只打印前5个
        print(f"  {i}. {route[:60]}..." if len(route) > 60 else f"  {i}. {route}")
    if len(research_routes) > 5:
        print(f"  ... 还有 {len(research_routes) - 5} 个子任务")

    print("\n📄 部分数据源预览:")
    for item in raw_data_store[:3]:  # 只打印前3个
        print(f"  - URL: {item['url'][:50]}...")
        print(f"    DOM: {Path(item['dom_filepath']).name}")
    if len(raw_data_store) > 3:
        print(f"  ... 还有 {len(raw_data_store) - 3} 个数据源")

    # 3. 构造 WebsearchKnowledgeState
    state = WebsearchKnowledgeState()
    state.raw_data_store = raw_data_store
    state.research_routes = research_routes

    # 4. 初始化 WebsearchChiefCuratorAgent
    agent = WebsearchChiefCuratorAgent.create(tool_manager=None)

    print("\n" + "=" * 80)
    print("🚀 开始执行 chief_curator 节点...")
    print("=" * 80)

    # 5. 执行 Agent
    result = await agent.run(state)

    print("\n" + "=" * 80)
    print("✅ chief_curator 节点执行完成")
    print("=" * 80)
    
    print(f"\n📊 执行结果:")
    print(f"  - 状态: {result.get('status', 'unknown')}")

    if result.get("status") == "success":
        print(f"  - 输出目录: {result.get('curated_directory', 'N/A')}")
        print(f"  - 处理任务数: {result.get('tasks_processed', 0)}")
        print(f"  - 生成文件数: {len(result.get('files_created', []))}")
        
        # 打印生成的文件列表
        files = result.get('files_created', [])
        if files:
            print("\n📝 生成的知识文件:")
            for f in files:
                print(f"  - {Path(f).name}")
        
        # 打印生成的总结预览
        summary = getattr(state, "knowledge_base_summary", "")
        if summary:
            print(f"\n📖 知识库状态: {summary}")
    else:
        print(f"  - 原因: {result.get('reason', 'N/A')}")

    # 6. 将结果落盘
    debug_dir = Path("tests")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "debug_websearch_curator_result.json"

    # 添加更多调试信息
    result["_debug"] = {
        "raw_data_count": len(raw_data_store),
        "research_routes_count": len(research_routes),
        "research_routes": research_routes,
        "raw_data_sources": [{"url": d["url"], "source": d.get("source")} for d in raw_data_store[:10]]
    }

    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存至: {debug_path}")

    return result


async def main():
    """主函数"""
    try:
        result = await run_curator_node()
        return result
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
