from __future__ import annotations
import asyncio
import base64
import os
from pathlib import Path
from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import create_pipeline_graph
from dataflow_agent.utils import get_project_root

async def run_pipeline_workflow(
    target: str,
    json_file: str,
    need_debug: bool = False,
    session_id: str = "default",
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: str = os.getenv("DF_API_KEY", ""),
    model_name:str = 'gpt-4o',
    max_debug_rounds: int = 2
) -> dict:
    if api_key:
        os.environ["DF_API_KEY"] = api_key
        os.environ["DF_API_URL"] = chat_api_url
        
    """封装 workflow，返回执行结果（包含 agent_results）"""
    # 1. 获取项目根目录
    PROJECT_ROOT: Path = get_project_root()

    # 2. 拼接 tmps 目录
    TMPS_DIR: Path = PROJECT_ROOT / "dataflow_agent" / "tmps"
    session_id = base64.urlsafe_b64encode(session_id.encode()).decode()
    SESSION_DIR: Path = TMPS_DIR / session_id
    SESSION_DIR.mkdir(parents=True, exist_ok=True)  # 3. 如不存在则创建

    python_file_path = SESSION_DIR / "pipeline.py"

    req = DFRequest(
        language="en",
        chat_api_url=chat_api_url,
        api_key=api_key,
        model= model_name,
        json_file=json_file,
        target=target,
        python_file_path=str(python_file_path),
        need_debug=need_debug,
        session_id=session_id,
        max_debug_rounds= max_debug_rounds
    )

    # state 初始化
    state = DFState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = True

    # 执行 workflow
    graph = create_pipeline_graph().build()
    final_state = await graph.ainvoke(state)

    return {
        "success": True,
        "python_file": req.python_file_path,
        "execution_result": final_state.get('debug_history', {}),
        "agent_results":   final_state.get("agent_results", {}),
        "state": final_state
    }