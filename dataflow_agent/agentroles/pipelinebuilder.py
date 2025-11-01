"""
DataPipelineBuilder
~~~~~~~~~~~~~~~~~~~
1) 根据推荐算子列表调用 pipeline_assembler 生成 python 代码
2) 落盘为 .py 文件
3) 启动子进程执行，捕获运行结果

扩展：
    - skip_assemble=True  : 仅执行现有脚本而不重新组装
    - file_path           : 指定要执行的脚本路径
支持调试模式：state.debug_mode = True 时仅取前 10 行数据加速调试。
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.state import DFState
from dataflow_agent.promptstemplates.prompts_repo import NodesExporter
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.toolkits.pipetool.pipe_tools import (
    pipeline_assembler,
    write_pipeline_file,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------- #
#                              工具函数                                   #
# ---------------------------------------------------------------------- #

def export_nodes_with_llm(nodes_info: Any, state: DFState) -> Dict[str, Any]:
    """
    调用 LLM API 处理 nodes_info，自动连接 input_key 和 output_key
    
    Args:
        nodes_info: 节点信息（list 或 dict）
        state: DFState 对象，包含 API 配置信息
    
    Returns:
        Dict[str, Any]: 处理后的节点字典，包含修改后的 input_key/output_key
    """
    # 提取 API 配置
    api_url = state.request.chat_api_url
    api_key = state.request.api_key
    model = state.request.model
    
    log.info(f"[export_nodes_with_llm] 使用模型: {model}, API: {api_url}")
    
    # 准备提示词
#     system_prompt = """
# You are an expert in data processing pipeline node extraction.
# """
    system_prompt = NodesExporter.system_prompt_for_nodes_export
    
    # 将 nodes_info 转换为 JSON 字符串
    if isinstance(nodes_info, str):
        nodes_info = nodes_info
    else:
        nodes_info = json.dumps(nodes_info, indent=2, ensure_ascii=False)
    
    task_prompt = f"""
    我有一个 JSON 格式的 pipeline，包含多个算子节点。每个节点有 "name"（算子名称）基础的包含运行参数（如 input_key、output_key 等）。

    请帮我：
    1. 自动修改每个节点的 input_key 和 output_key，使这些节点从上到下能前后相连
    2. 将数据转换成指定的输出格式

    下面是原始 JSON：
    {nodes_info}

    [处理规则]
    1. 第一个节点的 `input_key` 固定为 "raw_content"
    2. 中间节点的 `output_key` (或 `output_key_*`) 和下一个节点的 `input_key` (或 `input_key_*`) 必须相同，形成数据流连接
    3. 最后一个节点的 `output_key_*` 固定为 "output_final"
    4. 如果某个节点的 `config.run` 中没有 `input_key` 或 `output_key` 相关字段，保持原样不修改
    5. `config.run` 中的 `storage` 字段不要输出到最终结果中

    [输出格式要求]
    返回一个数组，每个元素包含：
    - "op_name": 算子名称（来自原 JSON 的 "name" 字段）
    - "params": 运行参数字典（来自原 JSON 的 "config.run"，但去掉 "storage" 字段）

    [必须遵守: 只返回 JSON 数组，不要任何说明文字、解释或注释！]

    返回格式示例：

    [
    {{
        "op_name": "PromptedFilter",
        "params": {{
            "input_key": "raw_content",
            "output_key": "eval"
        }}
    }},
    {{
        "op_name": "PromptedRefiner",
        "params": {{
            "input_key": "eval",
            "output_question_key": "refined_question"
        }}
    }},
    {{
        "op_name": "AnotherOperator",
        "params": {{
            "input_key": "refined_question",
            "output_key": "output_final"
        }}
    }}
    ]
    """
    log.warning(task_prompt)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ],
        "temperature": 0.0,
    }
    # 调用 API
    try:
        log.info("[export_nodes_with_llm] 开始调用 LLM API...")
        
        response = requests.post(
            f"{api_url}chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        log.info(response)
        
        # 解析响应
        result = response.json()
        log.info(result)
        content = result["choices"][0]["message"]["content"]
        log.info(content)
        
        log.info(f"[export_nodes_with_llm] LLM 返回内容长度: {len(content)}")
        
        # 使用 robust_parse_json 提取 JSON
        from dataflow_agent.utils import robust_parse_json

        parsed_json = robust_parse_json(content)
        return parsed_json
        
    except requests.exceptions.RequestException as e:
        log.error(f"[export_nodes_with_llm] API 请求失败: {str(e)}")
        raise RuntimeError(f"调用 LLM API 失败: {str(e)}")
    except (KeyError, IndexError) as e:
        log.error(f"[export_nodes_with_llm] 响应解析失败: {str(e)}")
        raise RuntimeError(f"解析 LLM 响应失败: {str(e)}")
    except Exception as e:
        log.error(f"[export_nodes_with_llm] 未知错误: {str(e)}")
        raise RuntimeError(f"处理节点导出时出错: {str(e)}")

def _patch_first_entry_file(py_file: str | Path,
                            old_path: str,
                            new_path: str) -> None:
    """
    把脚本中的 first_entry_file_name 由 old_path 替换成 new_path
    """
    py_file = Path(py_file).expanduser().resolve()
    code = py_file.read_text(encoding="utf-8")

    # 既考虑单/双引号，也兼容额外空格
    pattern = (
        r'first_entry_file_name\s*=\s*[\'"]'
        + re.escape(old_path)
        + r'[\'"]'
    )
    replacement = f'first_entry_file_name=\"{new_path}\"'
    new_code, n = re.subn(pattern, replacement, code, count=1)
    if n == 0:
        # 保险：直接字符串替换
        new_code = code.replace(old_path, new_path)

    py_file.write_text(new_code, encoding="utf-8")

def _ensure_py_file(code: str, file_name: str | None = None) -> Path:
    """
    把生成的代码写入文件并返回路径。
    若 file_name 为空，写入系统临时目录。
    """
    if file_name:
        target = Path(file_name).expanduser().resolve()
    else:
        target = Path(tempfile.gettempdir()) / f"recommend_pipeline_{uuid.uuid4().hex}.py"
    target.write_text(textwrap.dedent(code), encoding="utf-8")
    log.warning(f"[pipeline_builder] pipeline code 正在被写入： {target}")
    return target

def _create_debug_sample(src_file: str | Path, sample_lines: int = 10) -> Path:
    """
    从 src_file 抽取前 sample_lines 条记录（不是行），写入临时文件。
    支持 JSON/JSONL/CSV 格式。
    """
    src_path = Path(src_file).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source file not found: {src_path}")

    tmp_path = (
        Path(tempfile.gettempdir())
        / f"{src_path.stem}_sample_{sample_lines}{src_path.suffix}"
    )

    # 判断文件格式
    suffix = src_path.suffix.lower()
    
    if suffix == '.csv':  # CSV 格式
        df = pd.read_csv(src_path)
        sample_df = df.head(sample_lines)
        sample_df.to_csv(tmp_path, index=False, encoding="utf-8")
        log.info(
            f"[pipeline_builder] debug mode: CSV sample written to {tmp_path} "
            f"(first {sample_lines} records)"
        )
    
    elif suffix == '.json':  # JSON 格式
        with src_path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':  # JSON 数组格式
                data = json.load(f)
                sample_data = data[:sample_lines]
                with tmp_path.open("w", encoding="utf-8") as wf:
                    json.dump(sample_data, wf, ensure_ascii=False, indent=2)
            
            else:  # JSONL 格式（每行一个 JSON）
                sample_data = []
                for idx, line in enumerate(f):
                    if idx >= sample_lines:
                        break
                    sample_data.append(json.loads(line.strip()))
                
                with tmp_path.open("w", encoding="utf-8") as wf:
                    for item in sample_data:
                        wf.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        log.info(
            f"[pipeline_builder] debug mode: JSON sample written to {tmp_path} "
            f"(first {sample_lines} records)"
        )
    
    elif suffix == '.jsonl':  # JSONL 格式（明确后缀）
        sample_data = []
        with src_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= sample_lines:
                    break
                sample_data.append(json.loads(line.strip()))
        
        with tmp_path.open("w", encoding="utf-8") as wf:
            for item in sample_data:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        log.info(
            f"[pipeline_builder] debug mode: JSONL sample written to {tmp_path} "
            f"(first {sample_lines} records)"
        )
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Only .json, .jsonl, .csv are supported.")

    return tmp_path

from typing import Callable, List

Condition = Callable[[int, str, str], bool]

# ------ ① 必须正常退出 ------
def _rc_ok(rc: int, *_args) -> bool:
    return rc == 0        # rc==0 代表脚本没有崩

# ------ ② 不得出现关键 Warning ------
_CRITICAL_WARNING_PATTERNS: List[re.Pattern] = [
    re.compile(r"Warning:\s+Unexpected key", re.I),
    # 继续往这里追加 regex
]

def _no_critical_warning(_rc: int, out: str, err: str) -> bool:
    combined = out + "\n" + err
    return not any(p.search(combined) for p in _CRITICAL_WARNING_PATTERNS)


CONDITIONS: List[Condition] = [
    _rc_ok,
    _no_critical_warning,
]

async def _run_py(file_path: Path) -> dict[str, any]:
    """执行 python 文件并根据全局 CONDITIONS 判定 success"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    stdout, stderr = stdout_b.decode(), stderr_b.decode()

    success = all(cond(proc.returncode, stdout, stderr) for cond in CONDITIONS)

    return {
        "success": success,
        "return_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "file_path": str(file_path),
    }

# async def _run_py(file_path: Path) -> Dict[str, Any]:
#     """异步执行 python 文件并捕获输出"""
#     proc = await asyncio.create_subprocess_exec(
#         sys.executable,
#         str(file_path),
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#     )
#     stdout, stderr = await proc.communicate()
#     return {
#         "success": proc.returncode == 0,
#         "return_code": proc.returncode,
#         "stdout": stdout.decode(),
#         "stderr": stderr.decode(),
#         "file_path": str(file_path),
#     }


# ---------------------------------------------------------------------- #
#                                Agent                                    #
# ---------------------------------------------------------------------- #
class DataPipelineBuilder(BaseAgent):
    """把推荐算子列表转换为完整 Pipeline 并立即执行，支持调试与只执行两种模式"""

    # ---------- 基本信息 ----------
    @property
    def role_name(self) -> str:
        return "pipeline_builder"

    # 本 Agent 不调用 LLM，模板仅作占位
    @property
    def system_prompt_template_name(self) -> str:  # noqa: D401
        return "VOID"

    @property
    def task_prompt_template_name(self) -> str:
        return "VOID"

    # ----------- 前置工具默认结果 ------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {"recommendation": []}

    # ----------- 主执行逻辑 ------------------
    async def execute(
        self,
        state: DFState,
        *,
        skip_assemble: bool = False,
        file_path: str | None = None,
        assembler_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> DFState:  # type: ignore[override]
        """
        运行模式说明
        ------------------------------------------------------------------
        1. skip_assemble=False (默认) : 正常「组装→写盘→执行」全流程
        2. skip_assemble=True        : 仅执行已存在的脚本 file_path
                                       (若 file_path 为空则取 self.temp_data['pipeline_file_path'])
        ------------------------------------------------------------------
        """
        # assembler_kwargs = assembler_kwargs or {}
        assembler_kwargs = dict(assembler_kwargs or {})
        try:
            # ---------------- ① 需要重新组装代码 -----------------
            if not skip_assemble:
                # 1) 获取推荐算子
                pre_tool_results = await self.execute_pre_tools(state)
                recommendation: List[str] = (
                    pre_tool_results.get("recommendation")
                    or getattr(state, "recommendation", [])
                )
                if not recommendation:
                    raise ValueError("无可用 recommendation")

                # -------- 调试模式处理 --------
                debug_mode: bool = bool(getattr(state, "debug_mode", False))
                if debug_mode:
                    origin_file: str | None = assembler_kwargs.get("file_path")
                    if not origin_file:
                        raise ValueError(
                            "debug 模式下需要 `assembler_kwargs['file_path']` 指向原始数据文件"
                        )
                    sample_path = _create_debug_sample(origin_file, sample_lines=10)
                    assembler_kwargs["file_path"] = str(sample_path)
                    state.temp_data["debug_sample_file"] = str(sample_path)
                    state.temp_data["origin_file_path"] = origin_file
                    log.info(f"[pipeline_builder] DEBUG mode , sample at {sample_path}")

                # 2) 生成 pipeline 代码字符串
                pipe_obj = pipeline_assembler(recommendation, state, **assembler_kwargs)
                log.info(f"assembler_kwargs : {assembler_kwargs}")
                code_str: str = pipe_obj["pipe_code"]
                # 3) 写临时代码文件
                file_path_obj = _ensure_py_file(code_str, file_name=file_path)
                state.pipeline_file_path = str(file_path_obj)
                file_path = str(file_path_obj)  # 供后续 _run_py 使用
                # 4) LLM重新写code
                import dataflow_agent.toolkits.pipetool.pipe_tools as pt
                graph = pt.parse_pipeline_file(file_path)
                nodes_info  = graph['nodes']
                log.critical(f'测试nodes_info: {nodes_info}')
                op_and_params = export_nodes_with_llm(nodes_info=nodes_info,state=state)
                log.critical(f'测试op_and_params: {op_and_params}')
                from dataflow_agent.toolkits.pipetool.pipe_tools import build_pipeline_code_with_run_params
                pipeline_code = build_pipeline_code_with_run_params(opname_and_params=op_and_params,
                                                                    state=state,
                                                                    cache_dir=state.request.cache_dir,
                                                                    chat_api_url=state.request.chat_api_url,
                                                                    model_name=state.request.model,
                                                                    file_path=assembler_kwargs["file_path"]
                                                                    )
                
                log.critical(f'测试pipeline_code: {pipeline_code}')
                state.pipeline_code = pipeline_code
                state.temp_data["pipeline_code"] = pipeline_code
                file_path_obj = _ensure_py_file(pipeline_code, file_name=file_path)
                state.pipeline_file_path = str(file_path_obj)
                file_path = str(file_path_obj)  # 供后续 _run_py 使用

            # ---------------- ② 仅执行已存在脚本 -----------------
            else:
                file_path = file_path or state.temp_data.get("pipeline_file_path")
                if not file_path:
                    raise ValueError("skip_assemble=True 但未提供 file_path")
                file_path_obj = Path(file_path).expanduser().resolve()
                if not file_path_obj.is_file():
                    raise FileNotFoundError(f"待执行文件不存在: {file_path_obj}")

            # -------------- ③ 真执行 -----------------------
            if state.request.need_debug:
                log.critical("[pipeline_builder] 开始 Debug 执行，need_debug=True")
                exec_result = await _run_py(Path(file_path))
                state.execution_result = exec_result
                log.info(f"[pipeline_builder] run success={exec_result['success']}")
            else:
                log.info("[pipeline_builder] 跳过执行，仅生成代码文件")
                state.execution_result = {
                    "success": None,
                    "skipped": True,
                    "file_path": file_path,
                }

            # 若调试成功，关闭 debug 开关，以便后续跑全量数据
            if getattr(state, "debug_mode", False) and state.execution_result.get("success"):
                state.debug_mode = False
                log.info("[pipeline_builder] debug run passed, state.debug_mode -> False")

                # sample_path: str | None = state.temp_data.pop("debug_sample_file", None)
                sample_path: str | None = state.temp_data.get("debug_sample_file")
                origin_path: str | None = state.temp_data.get("origin_file_path")
                if sample_path and origin_path:
                    _patch_first_entry_file(
                        py_file=state.request.python_file_path,   # 调试期生成的脚本
                        old_path=sample_path,
                        new_path=origin_path,
                    )
                    log.info(f"[pipeline_builder] patched first_entry_file_name -> {origin_path}")
                    # exec_result = await _run_py(Path(state.pipeline_file_path))
                    # state.execution_result = exec_result
                    log.info(f"[pipeline_builder] full run success={exec_result['success']}")

        except Exception as e:
            log.exception("[pipeline_builder] 构建/执行失败")
            state.execution_result = {
                "success": False,
                "stderr": str(e),
                "stdout": "",
                "return_code": -1,
            }
        self.update_state_result(state, state.execution_result, locals().get("pre_tool_results", {}))  # type: ignore[arg-type]
        return state


# ---------------------------------------------------------------------- #
#                    对外统一调用入口                                     #
# ---------------------------------------------------------------------- #
async def data_pipeline_build(
    state: DFState,
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DFState:
    """单步调用：构建并执行推荐管线"""
    builder = DataPipelineBuilder(tool_manager=tool_manager)
    return await builder.execute(state, **kwargs)


def create_pipeline_builder(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DataPipelineBuilder:
    return DataPipelineBuilder(tool_manager=tool_manager, **kwargs)