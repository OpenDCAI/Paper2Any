"""
df_op_usage workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-10-29 15:14:37

功能：根据推荐的 operators 自动生成 pipeline 代码并执行
"""

from __future__ import annotations
import json
import subprocess
import asyncio
from pathlib import Path
from dataflow_agent.state import DFState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.logger import get_logger
import os

from dataflow_agent.toolkits.pipetool.pipe_tools import build_pipeline_code_with_full_params

log = get_logger(__name__)

@register("df_op_usage")
def create_df_op_usage_graph() -> GenericGraphBuilder:
    """
    Workflow: 自动组装 pipeline 并执行
    """
    builder = GenericGraphBuilder(state_model=DFState, entry_point="generate_pipeline")

    # ==============================================================
    # NODES
    # ==============================================================
    
    async def generate_pipeline(state: DFState) -> DFState:
        """
        根据 state.opname_and_params 生成 pipeline 代码
        """
        log.info("[df_op_usage] Generating pipeline code...")
        
        opname_and_params = state.opname_and_params if state.opname_and_params else []
        
        if not opname_and_params:
            log.warning("[df_op_usage] No operators found in state.opname_and_params")
            state.agent_results["generate_pipeline"] = {
                "status": "error",
                "error": "No operators to generate pipeline"
            }
            return state
        
        # 从 state.request 构建额外参数
        kwargs = {
            "cache_dir": state.request.cache_dir or "./cache_dir",
            "chat_api_url": state.request.chat_api_url or "",
            "model_name": state.request.model or "gpt-4o",
            "file_path": state.request.json_file or "",
        }
        log.critical(opname_and_params)
        # 生成代码
        try:
            pipeline_code = build_pipeline_code_with_full_params(
                opname_and_params=opname_and_params,
                **kwargs
            )
            
            state.temp_data["code"] = pipeline_code
            state.temp_data["output_file"] = f"{state.request.cache_dir}/dataflow_cache_step_step{len(opname_and_params)}.jsonl"
            log.info(f'output_file: {state.temp_data["output_file"]}')
            
            # 保存到指定目录
            output_dir = Path(state.request.cache_dir) / "generated_pipelines"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用 session_id 作为文件名的一部分
            output_file = output_dir / f"pipeline_{state.request.session_id}.py"
            output_file.write_text(pipeline_code, encoding="utf-8")
            
            log.info(f"[df_op_usage] Pipeline code saved to: {output_file}")
            
            # 保存到 state 的标准字段
            state.pipeline_structure_code = {
                "code": pipeline_code,
                "file_path": str(output_file),
                "op_count": len(opname_and_params)
            }
            
            state.agent_results["generate_pipeline"] = {
                "status": "success",
                "pipeline_file": str(output_file),
                "op_count": len(opname_and_params),
                "code_length": len(pipeline_code)
            }
            
        except Exception as e:
            log.error(f"[df_op_usage] Failed to generate pipeline: {e}", exc_info=True)
            state.agent_results["generate_pipeline"] = {
                "status": "error",
                "error": str(e)
            }
        
        return state

    async def execute_pipeline(state: DFState) -> DFState:
        """
        在子进程中执行生成的 pipeline
        """
        log.info("[df_op_usage] Executing generated pipeline...")
        
        gen_result = state.agent_results.get("generate_pipeline", {})
        
        if gen_result.get("status") != "success":
            log.error("[df_op_usage] Cannot execute - generation failed")
            state.execution_result = {
                "status": "skipped",
                "reason": "generation failed"
            }
            state.agent_results["execute_pipeline"] = state.execution_result
            return state
        
        pipeline_file = gen_result.get("pipeline_file")
        
        try:
            # 准备执行环境
            env = os.environ.copy()
            
            # 执行 python 文件
            log.info(f"[df_op_usage] Executing: python {pipeline_file}")
            process = await asyncio.create_subprocess_exec(
                "python", pipeline_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(Path(pipeline_file).parent)
            )
            
            stdout, stderr = await process.communicate()
            
            stdout_text = stdout.decode("utf-8")
            stderr_text = stderr.decode("utf-8")
            
            result = {
                "status": "success" if process.returncode == 0 else "failed",
                "return_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "pipeline_file": pipeline_file
            }
            
            log.info(f"[df_op_usage] Pipeline execution completed with code {process.returncode}")
            
            if process.returncode != 0:
                log.error(f"[df_op_usage] Pipeline execution failed:\n{stderr_text}")
            else:
                log.info(f"[df_op_usage] Pipeline output:\n{stdout_text}")
            
            # 保存到 state 的标准字段
            state.execution_result = result
            state.agent_results["execute_pipeline"] = result
            
        except Exception as e:
            log.error(f"[df_op_usage] Failed to execute pipeline: {e}", exc_info=True)
            result = {
                "status": "error",
                "error": str(e),
                "pipeline_file": pipeline_file
            }
            state.execution_result = result
            state.agent_results["execute_pipeline"] = result
        
        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "generate_pipeline": generate_pipeline,
        "execute_pipeline": execute_pipeline,
    }

    edges = [
        ("generate_pipeline", "execute_pipeline"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder