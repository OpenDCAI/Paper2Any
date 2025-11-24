"""
TRJ 功能测试

测试轨迹收集、构建和导出功能
"""

import asyncio
from dataflow_agent.trajectory import (
    TrajectoryManager,
    TrajectoryCollector,
    TrajectoryBuilder,
    TrajectoryExporter,
    StepRole,
    ActionType,
)
from dataflow_agent.state import DFState, DFRequest
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


async def test_basic_trajectory():
    """测试基本的轨迹记录流程"""
    log.info("=" * 60)
    log.info("测试 1: 基本轨迹记录")
    log.info("=" * 60)
    
    # 1. 创建管理器
    manager = TrajectoryManager()
    
    # 2. 准备输入
    inputs = {
        "query": "帮我分析这份销售数据",
        "model": "gpt-4o"
    }
    
    # 3. 开始记录
    manager.start_recording(inputs=inputs)
    
    # 4. 模拟 workflow 执行
    collector = manager.get_collector()
    
    # 步骤 1: Classifier
    collector.on_node_start("classifier", StepRole.AGENT.value, {"query": inputs["query"]})
    collector.on_llm_call(
        model="gpt-4o",
        messages=[{"role": "user", "content": "分类任务"}],
        response='{"category": "data_analysis"}',
        duration_ms=1200.5
    )
    collector.on_node_end(output={"category": "data_analysis"})
    
    # 步骤 2: Recommender
    collector.on_node_start("recommender", StepRole.AGENT.value)
    collector.on_tool_call(
        tool_name="get_operators",
        tool_args={"category": "data_analysis"},
        tool_result=["read_csv", "analyze_data", "plot_chart"],
        duration_ms=50.2
    )
    collector.on_node_end(output={"ops": ["read_csv", "analyze_data"]})
    
    # 步骤 3: Builder
    collector.on_node_start("builder", StepRole.SYSTEM_NODE.value)
    collector.on_node_end(output={"code": "pipeline code here"})
    
    # 5. 创建最终状态
    request = DFRequest(
        target=inputs["query"],
        model=inputs["model"]
    )
    final_state = DFState(request=request)
    final_state.agent_results = {
        "classifier": {"results": {"category": "data_analysis"}},
        "recommender": {"results": {"ops": ["read_csv", "analyze_data"]}},
        "builder": {"results": {"code": "pipeline code here"}}
    }
    
    # 6. 停止记录
    trajectory = manager.stop_recording(
        state=final_state,
        workflow_name="test_workflow"
    )
    
    # 7. 验证
    assert trajectory is not None
    assert trajectory.workflow_name == "test_workflow"
    assert len(trajectory.steps) == 3
    assert trajectory.status == "success"
    
    log.info(f"✓ 轨迹 ID: {trajectory.trace_id}")
    log.info(f"✓ 步骤数: {len(trajectory.steps)}")
    log.info(f"✓ LLM 调用: {trajectory.total_llm_calls}")
    log.info(f"✓ 工具调用: {trajectory.total_tool_calls}")
    
    # 8. 导出
    filepath = manager.export(trajectory, format="json")
    log.info(f"✓ 已导出: {filepath}")
    
    return trajectory


async def test_react_mode_trajectory():
    """测试 ReAct 模式轨迹"""
    log.info("=" * 60)
    log.info("测试 2: ReAct 模式轨迹")
    log.info("=" * 60)
    
    manager = TrajectoryManager()
    manager.start_recording(inputs={"query": "计算 123 + 456"})
    
    collector = manager.get_collector()
    
    # ReAct 循环
    for i in range(2):
        collector.on_node_start(f"agent_step_{i}", StepRole.AGENT.value)
        
        # 思考
        collector.on_thought(f"我需要使用计算器工具来计算 123 + 456")
        
        # 动作
        collector.on_action(
            action_type=ActionType.TOOL_CALL.value,
            action_payload={
                "tool_name": "calculator",
                "tool_args": {"expression": "123 + 456"}
            }
        )
        
        # 观察
        collector.on_observation("计算结果: 579")
        
        collector.on_node_end(output={"result": 579})
    
    # 创建状态
    request = DFRequest(target="计算 123 + 456")
    final_state = DFState(request=request)
    final_state.agent_results = {
        "agent": {"results": {"answer": "579"}}
    }
    
    trajectory = manager.stop_recording(final_state, "react_workflow")
    
    # 验证 ReAct 特征
    assert any(step.thought for step in trajectory.steps)
    assert any(step.observation for step in trajectory.steps)
    assert trajectory.mode == "react"
    
    log.info(f"✓ 模式: {trajectory.mode}")
    log.info(f"✓ 包含思考: {any(step.thought for step in trajectory.steps)}")
    
    # 导出 SFT 格式
    filepath = manager.export(trajectory, format="sft")
    log.info(f"✓ 已导出 SFT 格式: {filepath}")
    
    return trajectory


async def test_batch_export():
    """测试批量导出"""
    log.info("=" * 60)
    log.info("测试 3: 批量导出")
    log.info("=" * 60)
    
    # 创建多个轨迹
    trajectories = []
    
    for i in range(3):
        manager = TrajectoryManager()
        manager.start_recording(inputs={"query": f"任务 {i}"})
        
        collector = manager.get_collector()
        collector.on_node_start(f"node_{i}", StepRole.SYSTEM_NODE.value)
        collector.on_node_end(output={"result": f"output_{i}"})
        
        request = DFRequest(target=f"任务 {i}")
        state = DFState(request=request)
        
        trajectory = manager.stop_recording(state, f"workflow_{i}")
        trajectories.append(trajectory)
    
    # 批量导出
    exporter = TrajectoryExporter()
    
    # 导出为 JSONL
    filepath = exporter.export_to_jsonl(trajectories, mode="raw")
    log.info(f"✓ 批量导出 JSONL: {filepath}")
    
    # 导出统计信息
    stats_path = exporter.export_statistics(trajectories)
    log.info(f"✓ 导出统计信息: {stats_path}")
    
    return trajectories


async def test_feedback():
    """测试用户反馈"""
    log.info("=" * 60)
    log.info("测试 4: 用户反馈")
    log.info("=" * 60)
    
    manager = TrajectoryManager()
    manager.start_recording(inputs={"query": "测试反馈"})
    
    collector = manager.get_collector()
    collector.on_node_start("test_node", StepRole.AGENT.value)
    collector.on_node_end(output={"result": "success"})
    
    request = DFRequest(target="测试反馈")
    state = DFState(request=request)
    
    trajectory = manager.stop_recording(state, "feedback_test")
    
    # 添加反馈
    manager.add_feedback(
        trajectory=trajectory,
        score=5,
        comment="非常好！",
        labels=["accurate", "helpful"]
    )
    
    assert trajectory.feedback is not None
    assert trajectory.feedback.score == 5
    assert "accurate" in trajectory.feedback.labels
    
    log.info(f"✓ 反馈评分: {trajectory.feedback.score}")
    log.info(f"✓ 反馈标签: {trajectory.feedback.labels}")
    
    # 导出包含反馈的轨迹
    filepath = manager.export(trajectory, format="json")
    log.info(f"✓ 已导出包含反馈的轨迹: {filepath}")
    
    return trajectory


async def main():
    """运行所有测试"""
    log.info("\n" + "=" * 60)
    log.info("开始 TRJ 功能测试")
    log.info("=" * 60 + "\n")
    
    try:
        # 测试 1: 基本功能
        await test_basic_trajectory()
        
        # 测试 2: ReAct 模式
        await test_react_mode_trajectory()
        
        # 测试 3: 批量导出
        await test_batch_export()
        
        # 测试 4: 用户反馈
        await test_feedback()
        
        log.info("\n" + "=" * 60)
        log.info("✓ 所有测试通过！")
        log.info("=" * 60)
        
    except Exception as e:
        log.exception(f"✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
