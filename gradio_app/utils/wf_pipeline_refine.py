"""
Pipeline Refine Workflow 工具函数
封装 pipeline 二次优化的 workflow 调用和 JSON/Python 转换逻辑
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_refine import create_pipeline_refine_graph
from dataflow_agent.toolkits.pipetool.pipe_tools import (
    parse_pipeline_file,
    build_pipeline_code_with_full_params,
)
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def python_to_json(python_file_path: str) -> Dict[str, Any]:
    """
    将 Python pipeline 文件解析为 JSON 结构
    
    Args:
        python_file_path: Python 文件路径
    
    Returns:
        {"nodes": [...], "edges": [...]}
    """
    try:
        pipeline_json = parse_pipeline_file(python_file_path)
        log.info(f"[python_to_json] 成功解析 Python 文件: {python_file_path}")
        return pipeline_json
    except Exception as e:
        log.error(f"[python_to_json] 解析失败: {e}")
        raise


def json_to_python_code(
    pipeline_json: Dict[str, Any],
    *,
    cache_dir: str = "./cache_local",
    chat_api_url: str = "",
    model_name: str = "gpt-4o",
    file_path: str = "",
) -> str:
    """
    将 JSON 结构的 pipeline 转换为 Python 代码
    
    Args:
        pipeline_json: {"nodes": [...], "edges": [...]}
        cache_dir: 缓存目录
        chat_api_url: API URL
        model_name: 模型名称
        file_path: 输入文件路径
    
    Returns:
        Python 代码字符串
    """
    try:
        nodes = pipeline_json.get("nodes", [])
        
        # 将 nodes 转换为 opname_and_params 格式
        opname_and_params = []
        for node in nodes:
            config = node.get("config", {})
            opname_and_params.append({
                "op_name": node.get("name", ""),
                "init_params": config.get("init", {}),
                "run_params": config.get("run", {}),
            })
        
        # 调用转换函数
        python_code = build_pipeline_code_with_full_params(
            opname_and_params,
            cache_dir=cache_dir,
            llm_local=False,
            chat_api_url=chat_api_url,
            model_name=model_name,
            file_path=file_path,
        )
        
        log.info(f"[json_to_python_code] 成功转换 {len(nodes)} 个节点为 Python 代码")
        return python_code
        
    except Exception as e:
        log.error(f"[json_to_python_code] 转换失败: {e}")
        raise


async def run_pipeline_refine_workflow(
    refine_target: str,
    pipeline_json: Dict[str, Any],
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: str = "",
    model_name: str = "gpt-4o",
    json_file: str = "",
) -> Dict[str, Any]:
    """
    调用 pipeline refine workflow 进行二次优化
    
    Args:
        refine_target: 用户的优化需求描述
        pipeline_json: 当前 pipeline 的 JSON 结构 {"nodes": [...], "edges": [...]}
        chat_api_url: API URL
        api_key: API Key
        model_name: 模型名称
        json_file: 原始输入文件路径（用于生成代码时的 file_path 参数）
    
    Returns:
        {
            "success": bool,
            "refined_json": {...},  # 优化后的 JSON 结构
            "python_code": "...",   # 优化后的 Python 代码
            "agent_results": {...}, # Agent 执行结果
            "error": "..."          # 错误信息（如果有）
        }
    """
    try:
        # 设置环境变量
        if api_key:
            os.environ["DF_API_KEY"] = api_key
            os.environ["DF_API_URL"] = chat_api_url
        
        # 构造请求
        req = DFRequest(
            language="en",
            chat_api_url=chat_api_url,
            api_key=api_key or os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=refine_target,  # 优化需求作为 target
            json_file=json_file,
        )
        
        # 初始化 state，将当前 pipeline JSON 设置为 pipeline_structure_code
        state = DFState(request=req, messages=[])
        state.pipeline_structure_code = pipeline_json
        
        # 构建并运行 workflow
        log.info(f"[run_pipeline_refine_workflow] 开始优化，目标: {refine_target}")
        graph = create_pipeline_refine_graph().build()
        # 设置 recursion_limit 防止子图无限循环，支持约 5-7 轮工具调用
        final_state = await graph.ainvoke(state, config={"recursion_limit": 25})
        
        # 提取优化后的 JSON
        if isinstance(final_state, dict):
            refined_json = final_state.get("pipeline_structure_code", {})
            agent_results = final_state.get("agent_results", {})
        else:
            refined_json = getattr(final_state, "pipeline_structure_code", {})
            agent_results = getattr(final_state, "agent_results", {})
        
        # 检查 pipeline_refiner 的结果状态
        refiner_result = agent_results.get("pipeline_refiner", {}).get("results", {})
        refiner_status = refiner_result.get("status", "success")
        refiner_message = refiner_result.get("message", "")
        
        # 如果 refiner 返回了 pipeline 字段，优先使用它
        if isinstance(refiner_result.get("pipeline"), dict) and refiner_result["pipeline"].get("nodes"):
            refined_json = refiner_result["pipeline"]
            log.info(f"[run_pipeline_refine_workflow] 使用 refiner 返回的 pipeline")
        
        # 检查是否有错误或部分失败
        warning_message = ""
        if refiner_status in ("partial_failure", "failure", "error"):
            warning_message = refiner_message or f"优化过程遇到问题: {refiner_status}"
            log.warning(f"[run_pipeline_refine_workflow] {warning_message}")
        
        # 检查是否有子图执行错误
        if isinstance(refiner_result, dict) and refiner_result.get("error"):
            error_msg = refiner_result.get("error", "")
            if "Recursion limit" in error_msg:
                warning_message = "优化过程达到最大迭代次数，可能未完全完成优化。" + (f" {refiner_message}" if refiner_message else "")
            else:
                warning_message = f"优化过程出错: {error_msg}"
            log.warning(f"[run_pipeline_refine_workflow] {warning_message}")
        
        # 将优化后的 JSON 转换为 Python 代码
        python_code = ""
        if refined_json and refined_json.get("nodes"):
            try:
                python_code = json_to_python_code(
                    refined_json,
                    cache_dir="./cache_local",
                    chat_api_url=chat_api_url,
                    model_name=model_name,
                    file_path=json_file,
                )
            except Exception as e:
                log.warning(f"[run_pipeline_refine_workflow] JSON 转 Python 失败: {e}")
                python_code = f"# JSON 转 Python 代码失败: {e}"
        
        log.info(f"[run_pipeline_refine_workflow] 优化完成，节点数: {len(refined_json.get('nodes', []))}")
        
        result = {
            "success": True,
            "refined_json": refined_json,
            "python_code": python_code,
            "agent_results": agent_results,
        }
        
        # 添加警告信息（如果有）
        if warning_message:
            result["warning"] = warning_message
            result["refiner_status"] = refiner_status
        
        return result
        
    except Exception as e:
        log.error(f"[run_pipeline_refine_workflow] 优化失败: {e}")
        return {
            "success": False,
            "refined_json": {},
            "python_code": "",
            "agent_results": {},
            "error": str(e),
        }
