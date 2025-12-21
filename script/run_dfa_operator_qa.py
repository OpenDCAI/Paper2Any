#!/usr/bin/env python3
"""
OperatorQA 入口脚本 - 算子问答命令行工具
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用方式：
    # 单次查询
    python run_dfa_operator_qa.py --query "我想过滤掉缺失值用哪个算子？"
    
    # 交互模式
    python run_dfa_operator_qa.py --interactive
    
    # 指定模型
    python run_dfa_operator_qa.py --query "..." --model gpt-4-turbo
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import DFRequest, MainState
from dataflow_agent.workflow.wf_operator_qa import create_operator_qa_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DataFlow 算子问答工具 - 通过自然语言查询算子信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 单次查询
    python run_dfa_operator_qa.py --query "我想过滤掉缺失值用哪个算子？"
    
    # 交互模式（多轮对话）
    python run_dfa_operator_qa.py --interactive
    
    # 查看算子源码
    python run_dfa_operator_qa.py --query "给我看看 PromptedFilter 的源码"
    
    # 查询参数含义
    python run_dfa_operator_qa.py --query "PromptedGenerator 的 run 方法参数是什么意思？"
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="查询内容"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启用交互模式（多轮对话）"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="使用的模型名称 (默认: gpt-4o)"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://123.129.219.111:3000/v1/",
        help="Chat API URL"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (默认从环境变量 DF_API_KEY 读取)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="RAG 检索返回的算子数量 (默认: 5)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（JSON 格式）"
    )
    
    return parser.parse_args()


async def run_single_query(
    query: str,
    model: str = "gpt-4o",
    api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    执行单次查询
    
    Args:
        query: 用户查询
        model: 模型名称
        api_url: API URL
        api_key: API Key
        chat_history: 对话历史
        
    Returns:
        查询结果
    """
    # 获取 API Key
    api_key = api_key or os.getenv("DF_API_KEY", "")
    if not api_key:
        log.warning("未设置 API Key，请通过 --api-key 参数或 DF_API_KEY 环境变量设置")
    
    # 构建请求
    req = DFRequest(
        language="zh",
        chat_api_url=api_url,
        api_key=api_key,
        model=model,
        target=query,
    )
    
    # 构建状态
    state = MainState(request=req, messages=[])
    if chat_history:
        state.chat_history = chat_history
    
    # 构建并执行工作流
    log.info(f"正在处理查询: {query}")
    graph_builder = create_operator_qa_graph()
    graph = graph_builder.build()
    
    try:
        final_state = await graph.ainvoke(state)
    except Exception as e:
        log.error(f"执行失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }
    
    # 提取结果
    agent_result = final_state.get("agent_results", {}).get("operator_qa", {})
    results = agent_result.get("results", {})
    
    return {
        "success": True,
        "query": query,
        "answer": results.get("answer", ""),
        "related_operators": results.get("related_operators", []),
        "code_snippet": results.get("code_snippet", ""),
        "follow_up_suggestions": results.get("follow_up_suggestions", []),
        "chat_history": getattr(final_state, "chat_history", []),
    }


async def interactive_mode(
    model: str = "gpt-4o",
    api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: Optional[str] = None,
):
    """
    交互模式 - 多轮对话
    
    通过复用同一个 graph 和 state，实现真正的多轮对话。
    state.messages 会在多轮对话中累积，LLM 能看到完整的对话历史。
    """
    print("\n" + "=" * 60)
    print("  DataFlow 算子问答助手 (交互模式)")
    print("=" * 60)
    print("\n欢迎使用 DataFlow 算子问答助手！")
    print("你可以询问关于 DataFlow 算子的任何问题。")
    print("\n命令:")
    print("  - 输入问题进行查询")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'clear' 清除对话历史")
    print("  - 输入 'history' 查看对话历史")
    print("-" * 60 + "\n")
    
    # 获取 API Key
    api_key = api_key or os.getenv("DF_API_KEY", "")
    if not api_key:
        log.warning("未设置 API Key，请通过 --api-key 参数或 DF_API_KEY 环境变量设置")
    
    # 只创建一次 graph（复用 workflow 工厂函数内的共享变量）
    log.info("初始化 workflow graph...")
    graph_builder = create_operator_qa_graph()
    graph = graph_builder.build()
    
    # 创建一次 state，后续复用（messages 会累积）
    req = DFRequest(
        language="zh",
        chat_api_url=api_url,
        api_key=api_key,
        model=model,
        target="",  # 每次循环更新
    )
    state = MainState(request=req, messages=[])
    
    while True:
        try:
            # 获取用户输入
            query = input("\n🧑 你: ").strip()
            
            if not query:
                continue
            
            # 处理命令
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 再见！")
                break
            
            if query.lower() == "clear":
                # 清除对话历史：重置 state.messages
                state.messages = []
                print("✅ 对话历史已清除")
                continue
            
            if query.lower() == "history":
                if not state.messages:
                    print("📝 对话历史为空")
                else:
                    print(f"\n📝 对话历史 ({len(state.messages)} 条消息):")
                    for i, msg in enumerate(state.messages):
                        role = "🧑 你" if msg.type == "human" else "🤖 助手" if msg.type == "ai" else f"[{msg.type}]"
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"  [{i+1}] {role}: {content}")
                continue
            
            # 更新查询
            state.request.target = query
            
            # 执行查询（复用同一个 state，messages 会累积）
            print("\n⏳ 正在思考...")
            try:
                # graph.ainvoke 返回的是字典，需要更新 state
                final_state_dict = await graph.ainvoke(state)
                
                # 更新 state 的 messages（用于下一轮对话）
                if "messages" in final_state_dict:
                    state.messages = final_state_dict["messages"]
                
                # 更新 agent_results
                if "agent_results" in final_state_dict:
                    state.agent_results = final_state_dict["agent_results"]
                    
            except Exception as e:
                log.error(f"执行失败: {e}")
                print(f"\n❌ 查询失败: {e}")
                continue
            
            # 提取结果（从字典中获取）
            agent_result = final_state_dict.get("agent_results", {}).get("operator_qa", {})
            results = agent_result.get("results", {})
            
            if results:
                # 显示回答
                answer = results.get("answer", "")
                print(f"\n🤖 助手: {answer}")
                
                # 显示信息来源
                source = results.get("source_explanation", "")
                if source:
                    print(f"\n📌 信息来源: {source}")
                
                # 显示相关算子
                related_ops = results.get("related_operators", [])
                if related_ops:
                    print(f"\n📦 相关算子: {', '.join(related_ops)}")
                
                # 显示代码片段
                code_snippet = results.get("code_snippet", "")
                if code_snippet:
                    print(f"\n📄 代码片段:\n{code_snippet[:500]}...")
                
                # 显示后续建议
                suggestions = results.get("follow_up_suggestions", [])
                if suggestions:
                    print("\n💡 你可能还想问:")
                    for suggestion in suggestions[:3]:
                        print(f"   - {suggestion}")
                
                # 显示当前消息数量（调试用）
                log.debug(f"当前消息历史: {len(state.messages)} 条")
            else:
                print(f"\n❌ 未获取到有效结果")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            log.exception(f"发生错误: {e}")
            print(f"\n❌ 发生错误: {e}")


def format_result(result: Dict[str, Any]) -> str:
    """格式化输出结果"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  查询结果")
    lines.append("=" * 60)
    
    lines.append(f"\n📝 问题: {result.get('query', '')}")
    lines.append(f"\n💬 回答:\n{result.get('answer', '无回答')}")
    
    if result.get("related_operators"):
        lines.append(f"\n📦 相关算子: {', '.join(result['related_operators'])}")
    
    if result.get("code_snippet"):
        lines.append(f"\n📄 代码片段:\n{result['code_snippet']}")
    
    if result.get("follow_up_suggestions"):
        lines.append("\n💡 后续建议:")
        for s in result["follow_up_suggestions"]:
            lines.append(f"   - {s}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


async def main():
    """主函数"""
    args = parse_args()
    
    if args.interactive:
        # 交互模式
        await interactive_mode(
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
        )
    elif args.query:
        # 单次查询
        result = await run_single_query(
            query=args.query,
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
        )
        
        # 输出结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到: {args.output}")
        else:
            print(format_result(result))
    else:
        # 无参数时显示帮助
        print("请使用 --query 指定查询内容，或使用 --interactive 进入交互模式")
        print("使用 --help 查看更多选项")


if __name__ == "__main__":
    asyncio.run(main())

