from __future__ import annotations
import asyncio
from pathlib import Path
from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import create_pipeline_graph

async def run_pipeline_workflow(
    target: str,
    json_file: str,
    need_debug: bool = False,
    session_id: str = "default"
) -> dict:
    """封装你的 workflow，返回结果"""
    
    # 构造请求
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("DF_API_KEY"),
        model="gpt-4o",
        json_file=json_file,
        target=target,
        python_file_path=f"./tmps/{session_id}/pipeline.py",
        need_debug=need_debug,
        session_id=session_id,
    )
    
    # 执行
    state = DFState(request=req, messages=[])
    graph = create_pipeline_graph().build()
    final_state = await graph.ainvoke(state)
    
    return {
        "success": True,
        "python_file": req.python_file_path,
        "execution_result": final_state.get("execution_result", {}),
        "state": final_state
    }