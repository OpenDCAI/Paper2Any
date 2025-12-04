"""
测试 test_graph workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01 20:16:43

运行方式:
  pytest tests/test_test_graph.py -v -s
  或直接: python tests/test_test_graph.py
"""

from __future__ import annotations
import asyncio
import pytest

# ------------ 依赖 -------------
from dataflow_agent.states.test_graph_state import TestGraphState, TestGraphRequest
from dataflow_agent.workflow import run_workflow
# 如果使用了自定义 State，请替换上面的 TestGraphState 导入：
# from dataflow_agent.state import YourCustomState
# --------------------------------


# ============ 核心异步流程 ============
async def run_test_graph_pipeline() -> TestGraphState:
    """
    执行 test_graph 工作流的测试流程
    """
    # TODO: 根据实际需求构造初始状态
    # 1) 如果使用自定义请求对象，在这里构造
    req = TestGraphRequest()
    
    # 2) 初始化状态
    state = TestGraphState(
        messages=[],
        request=req
    )
    
    # TODO: 可以在这里预设一些测试数据
    # state.user_input = "测试输入"
    # state.agent_results = {}

    # 3) 通过注册中心执行工作流
    final_state: TestGraphState = await run_workflow("test_graph", state)
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_test_graph_pipeline():
    """
    测试 test_graph 工作流的完整流程
    """
    final_state = await run_test_graph_pipeline()

    # TODO: 根据实际业务逻辑添加断言
    # 示例断言：
    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "agent_results"), "state 应包含 agent_results"
    
    # -- 检查特定节点的结果 --
    # assert "step1" in final_state.agent_results, "step1 应该执行"
    # assert final_state.agent_results["step1"]["msg"] == "hello step1"
    
    # -- 检查 messages 或其他字段 --
    # assert len(final_state.messages) > 0, "应该有消息记录"

    # -- 调试输出，可按需保留 --
    print("\n=== agent_results ===")
    print(final_state.agent_results)
    
    if hasattr(final_state, "messages") and final_state.messages:
        print("\n=== messages ===")
        for msg in final_state.messages:
            print(f"- {msg}")


# ============ 直接 python 执行 ============
if __name__ == "__main__":
    """
    允许直接运行此文件进行快速测试
    """
    asyncio.run(run_test_graph_pipeline())