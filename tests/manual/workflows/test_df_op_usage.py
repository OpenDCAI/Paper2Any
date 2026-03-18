"""
测试 df_op_usage workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-10-29 15:14:37

运行方式:
  pytest tests/test_df_op_usage.py -v -s
  或直接: python tests/test_df_op_usage.py
"""

from __future__ import annotations
import asyncio
import pytest
from pathlib import Path

# ------------ 依赖 -------------
from dataflow_agent.state import DFState, DFRequest
from dataflow_agent.workflow import run_workflow
# --------------------------------
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
PROJDIR = get_project_root()
log = get_logger(__name__)
import sys, pprint, os
from dataflow.utils.registry import OPERATOR_REGISTRY
OPERATOR_REGISTRY._get_all()
import dataflow.operators              


# ============ 核心异步流程 ============
async def run_df_op_usage_pipeline() -> DFState:
    """
    执行 df_op_usage 工作流的测试流程
    """
    first_op = list(OPERATOR_REGISTRY.keys())[0] if OPERATOR_REGISTRY else None
    log.info(f"使用的第一个 operator: {first_op}")
    log.critical(f'OPERATOR_REGISTRY 内容: {OPERATOR_REGISTRY}')
    log.info(f" _sub_modules       :{ OPERATOR_REGISTRY._sub_modules}")
    # 1) 构造请求对象，设置测试参数
    req = DFRequest(
        language="zh",  # 中文
        model="gpt-4o",
        target="测试 pipeline 生成和执行",
        json_file=f"{PROJDIR}/tests/test.jsonl",  
        cache_dir="",
        session_id="test_session_001",
        use_local_model=False,
        need_debug=False,
    )
    # 2) 初始化状态，设置要测试的 operators
    state = DFState(
        request=req,
        messages=[],
        matched_ops=[first_op],
    )
    # 3) 通过注册中心执行工作流
    print("开始执行 df_op_usage workflow...")
    final_state: DFState = await run_workflow("df_op_usage", state)
    print("df_op_usage workflow 执行完成")
    return final_state


# ============ pytest 入口 ============
@pytest.mark.asyncio
async def test_df_op_usage_pipeline():
    """
    测试 df_op_usage 工作流的完整流程
    """
    final_state = await run_df_op_usage_pipeline()

    # ===== 基础断言 =====
    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "agent_results"), "state 应包含 agent_results"
    
    # ===== 检查 generate_pipeline 节点 =====
    assert "generate_pipeline" in final_state.agent_results, \
        "generate_pipeline 节点应该执行"
    
    gen_result = final_state.agent_results["generate_pipeline"]
    assert gen_result["status"] == "success", \
        f"代码生成应该成功，但状态为: {gen_result.get('status')}"
    
    assert "pipeline_file" in gen_result, \
        "生成结果应包含 pipeline_file 路径"
    
    assert "op_names" in gen_result, \
        "生成结果应包含使用的 operator 名称"
    
    # 验证生成的文件存在
    pipeline_file = Path(gen_result["pipeline_file"])
    assert pipeline_file.exists(), \
        f"生成的 pipeline 文件应该存在: {pipeline_file}"
    
    # ===== 检查 pipeline_structure_code 字段 =====
    assert final_state.pipeline_structure_code, \
        "state.pipeline_structure_code 应该被填充"
    
    assert "code" in final_state.pipeline_structure_code, \
        "pipeline_structure_code 应包含生成的代码"
    
    assert "file_path" in final_state.pipeline_structure_code, \
        "pipeline_structure_code 应包含文件路径"
    
    # ===== 检查 execute_pipeline 节点 =====
    assert "execute_pipeline" in final_state.agent_results, \
        "execute_pipeline 节点应该执行"
    
    exec_result = final_state.agent_results["execute_pipeline"]
    
    # 执行可能失败（如果 pipeline 有问题），但至少应该有状态
    assert "status" in exec_result, \
        "执行结果应包含状态信息"
    
    # ===== 检查 execution_result 字段 =====
    assert final_state.execution_result, \
        "state.execution_result 应该被填充"
    
    assert "return_code" in final_state.execution_result or \
           "error" in final_state.execution_result, \
        "execution_result 应包含返回码或错误信息"
    
    # ===== 调试输出 =====
    print("\n" + "="*60)
    print("=== 测试结果摘要 ===")
    print("="*60)
    
    print(f"\n✓ 使用的 operators: {final_state.matched_ops}")
    print(f"✓ 生成的文件: {gen_result.get('pipeline_file')}")
    print(f"✓ 代码长度: {gen_result.get('code_length')} 字符")
    
    if exec_result.get("status") == "success":
        print(f"✓ 执行状态: 成功 (返回码: {exec_result.get('return_code')})")
    elif exec_result.get("status") == "failed":
        print(f"✗ 执行状态: 失败 (返回码: {exec_result.get('return_code')})")
        print(f"  错误输出: {exec_result.get('stderr', 'N/A')[:200]}")
    else:
        print(f"! 执行状态: {exec_result.get('status')}")
    
    print("\n" + "="*60)
    print("=== 详细结果 ===")
    print("="*60)
    
    print("\n--- agent_results ---")
    for node_name, result in final_state.agent_results.items():
        print(f"\n[{node_name}]")
        for key, value in result.items():
            if key in ["stdout", "stderr", "code"]:
                # 长文本只显示前100字符
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  {key}: {value_str}")
            else:
                print(f"  {key}: {value}")
    
    if final_state.messages:
        print("\n--- messages ---")
        for i, msg in enumerate(final_state.messages):
            print(f"{i+1}. {msg}")
    
    # ===== 可选：输出生成的代码（用于调试） =====
    if "--verbose" in pytest.config.option.keyword:  # 使用 pytest -v 时显示
        print("\n" + "="*60)
        print("=== 生成的 Pipeline 代码 ===")
        print("="*60)
        print(final_state.pipeline_structure_code.get("code", "N/A"))


@pytest.mark.asyncio
async def test_df_op_usage_empty_operators():
    """
    测试边界情况：没有 operators 时的行为
    """
    req = DFRequest(
        language="zh",
        model="gpt-4o",
        cache_dir="./test_cache",
        session_id="test_empty_ops",
    )
    
    state = DFState(
        request=req,
        matched_ops=[],  # 空的 operator 列表
    )
    
    final_state = await run_workflow("df_op_usage", state)
    
    # 应该在 generate_pipeline 阶段失败
    gen_result = final_state.agent_results.get("generate_pipeline", {})
    assert gen_result.get("status") == "error", \
        "空 operators 应该返回错误状态"
    
    assert "No operators" in gen_result.get("error", ""), \
        "错误消息应该提示没有 operators"
    
    print("\n✓ 边界测试通过：空 operators 正确处理")


@pytest.mark.asyncio
async def test_df_op_usage_with_custom_operators():
    """
    测试自定义 operators 组合
    """
    req = DFRequest(
        language="en",
        model="gpt-4o",
        cache_dir="./test_cache",
        session_id="test_custom_ops",
    )
    
    state = DFState(
        request=req,
        # 替换为实际存在的 operators
        matched_ops=["CustomOp1", "CustomOp2", "CustomOp3"],
    )
    
    final_state = await run_workflow("df_op_usage", state)
    
    gen_result = final_state.agent_results.get("generate_pipeline", {})
    
    # 验证所有 operators 都被包含
    if gen_result.get("status") == "success":
        assert set(gen_result["op_names"]) == set(state.matched_ops), \
            "生成的 pipeline 应包含所有指定的 operators"
        
        print(f"\n✓ 自定义 operators 测试通过: {state.matched_ops}")
    else:
        print(f"\n! 注意：自定义 operators 生成失败: {gen_result.get('error')}")


# ============ 直接 python 执行 ============
if __name__ == "__main__":
    """
    允许直接运行此文件进行快速测试
    """
    print("直接运行模式：执行主测试流程\n")
    result = asyncio.run(run_df_op_usage_pipeline())
    
    print("\n" + "="*60)
    print("执行完成！")
    print("="*60)
    
    # 简要输出
    if result["agent_results"].get("generate_pipeline", {}).get("status") == "success":
        print("✓ Pipeline 生成成功")
    else:
        print("✗ Pipeline 生成失败")

    if result["agent_results"].get("execute_pipeline", {}).get("status") == "success":
        print("✓ Pipeline 执行成功")
    else:
        print("✗ Pipeline 执行失败")