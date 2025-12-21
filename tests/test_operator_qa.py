"""
测试 operator_qa workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01 15:05:11

运行方式:
  pytest tests/test_operator_qa.py -v -s
  或直接: python tests/test_operator_qa.py
"""

from __future__ import annotations
import asyncio
import pytest

# ------------ 依赖 -------------
from dataflow_agent.state import xxState, xxRequest
from dataflow_agent.workflow import run_workflow
# 如果使用了自定义 State，请替换上面的 xxState 导入：
# from dataflow_agent.state import YourCustomState
# --------------------------------


# ============ 核心异步流程 ============
async def run_operator_qa_pipeline() -> xxState:
    """
    执行 operator_qa 工作流的测试流程
    """
    # TODO: 根据实际需求构造初始状态
    # 1) 如果使用自定义请求对象，在这里构造
    # req = YourRequest(
    #     param1="value1",
    #     param2="value2",
    # )
    
    # 2) 初始化状态
    state = xxState(
        messages=[],
        # request=req,  # 如果有自定义请求
    )
    
    # TODO: 可以在这里预设一些测试数据
    # state.user_input = "测试输入"
    # state.agent_results = {}

    # 3) 通过注册中心执行工作流
    final_state: xxState = await run_workflow("operator_qa", state)
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_operator_qa_pipeline():
    """
    测试 operator_qa 工作流的完整流程
    """
    final_state = await run_operator_qa_pipeline()

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
    asyncio.run(run_operator_qa_pipeline())