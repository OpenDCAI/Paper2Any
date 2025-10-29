#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI – DataFlow Operator Pipeline Runner
"""

import json
import inspect
import os
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow import run_workflow
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

import gradio as gr

# -------------------- 准备算子元数据 --------------------
from dataflow.utils.registry import OPERATOR_REGISTRY
import dataflow.operators                  
OPERATOR_REGISTRY._get_all()              
_ALL_OPS = OPERATOR_REGISTRY.get_obj_map()

_CAT2OPS: Dict[str, List[str]] = defaultdict(list)
for op_name, cls in _ALL_OPS.items():
    mod_parts = cls.__module__.split(".")
    if "operators" in mod_parts:
        cat_idx = mod_parts.index("operators") + 1
        category = mod_parts[cat_idx] if cat_idx < len(mod_parts) else "uncategorized"
    else:
        category = "uncategorized"
    _CAT2OPS[category].append(op_name)


# -------------------- 后端执行 --------------------
async def run_df_op_usage_pipeline(               
    matched_ops_with_params: List[Dict[str, Dict[str, Any]]],
    json_file: str,
    chat_api_url: str,
    api_key: str,
    model: str = "gpt-4o",
    language: str = "zh",
    cache_dir: str = f"{get_project_root()}/dataflow_cache",
    session_id: str = "test_session_001",
):
    req = DFRequest(
        language=language,
        model=model,
        target="测试 pipeline 生成和执行",
        json_file=json_file,
        cache_dir=cache_dir,
        session_id=session_id,
        use_local_model=False,
        need_debug=False,
        chat_api_url=chat_api_url,
        api_key=api_key,
    )

    matched_ops_with_params = [
    {"op_name": list(d.keys())[0], "params": list(d.values())[0]}
    for d in matched_ops_with_params
]
    state = DFState(
        request=req,
        messages=[],
        opname_and_params=matched_ops_with_params,
    )

    log.info("开始执行 df_op_usage workflow...")
    final_state = await run_workflow("df_op_usage", state)
    log.info("df_op_usage workflow 执行完成")

    return final_state


# -------------------- Gradio 页面 --------------------
def create_page_dfopuse():
    with gr.Blocks(title="DataFlow-Agent UI") as page:
        gr.Markdown("## 🧩 DataFlow Operator Selector & Pipeline Runner")

        # ========= 0. 顶部 – API / 文件路径 =========
        with gr.Row():
            chat_api_url_tb = gr.Textbox(
                label="Chat API URL",
                value="http://123.129.219.111:3000/v1/",
                scale=3,
            )
            apikey_tb = gr.Textbox(label="API Key", type="password", scale=2)
            jsonl_path_tb = gr.Textbox(
                label="输入 JSONL 文件路径",
                placeholder="/path/to/input.jsonl",
                scale=3,
            )

        gr.Markdown("---")

        # ========= 1. 算子选择 =========
        with gr.Row(equal_height=False):
            # ----- 左列：选择 & 构建 pipeline -----
            with gr.Column(scale=4):
                cat_dd = gr.Dropdown(
                    label="算子分类",
                    choices=sorted(_CAT2OPS.keys()),
                )
                op_dd = gr.Dropdown(label="算子", choices=[])

                param_code = gr.Code(
                    label="run() 参数（JSON 可编辑）",
                    language="json",
                    value="{}",
                    lines=12,
                )

                with gr.Row():
                    add_btn = gr.Button("➕ 添加算子到 Pipeline", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空 Pipeline", variant="secondary")

                pipeline_state = gr.State([])  # List[Dict[op_name -> params]]
                pipeline_json = gr.JSON(label="当前 Pipeline", value=[])

                run_btn = gr.Button("🚀 运行 Pipeline", variant="primary", size="lg")

            # ----- 右列：参数说明 -----
            with gr.Column(scale=6):
                param_md = gr.Markdown("_请选择一个算子_")

        # ========= 2. 结果展示 =========
        gr.Markdown("---")
        gr.Markdown("### 📊 执行结果")

        with gr.Tabs():
            with gr.Tab("生成的代码"):
                code_out = gr.Code(label="生成的 Python 代码", language="python", lines=25)
            with gr.Tab("处理结果数据（前 100 条）"):
                result_out = gr.JSON()
            with gr.Tab("输出文件路径"):
                out_file_tb = gr.Textbox(interactive=False)

        # ========= 3. 交互逻辑 =========
        # --- 选分类 -> 刷新算子 ---
        cat_dd.change(
            lambda cat: gr.Dropdown(choices=sorted(_CAT2OPS.get(cat, []))),
            cat_dd,
            op_dd,
        )

        # --- 选算子 -> 展示参数默认值 & 说明 ---
        def _show_params(op_name: str):
            if not op_name:
                return "_请选择一个算子_", "{}"

            cls = _ALL_OPS[op_name]
            if not hasattr(cls, "run"):
                return f"⚠️ `{op_name}` 没有定义 run() 方法", "{}"

            sig = inspect.signature(cls.run)
            md_lines, defaults = [], {}
            md_lines.append(f"### `{op_name}.run()` 参数说明\n")
            for n, p in sig.parameters.items():
                if n == "self":
                    continue
                ann = (
                    f"`{p.annotation.__name__}`"
                    if p.annotation not in (inspect._empty, None)
                    else ""
                )
                default_val = p.default if p.default is not inspect._empty else ""
                md_lines.append(f"- **{n}** {ann}  \n  默认值：`{default_val}`")
                defaults[n] = default_val

            return "\n".join(md_lines), json.dumps(
                defaults, indent=2, ensure_ascii=False
            )

        op_dd.change(_show_params, op_dd, [param_md, param_code])

        # --- 添加算子到 pipeline ---
        def _add_op(op_name, param_json, pl):
            if not op_name:
                gr.Warning("⚠️ 请选择算子")
                return pl, pl
            try:
                params = json.loads(param_json or "{}")
            except json.JSONDecodeError as e:
                gr.Warning(f"JSON 解析失败：{e}")
                return pl, pl
            pl = list(pl)  # clone
            pl.append({op_name: params})
            gr.Info(f"✅ 已添加算子 `{op_name}`")
            return pl, pl

        add_btn.click(
            _add_op, [op_dd, param_code, pipeline_state], [pipeline_state, pipeline_json]
        )

        # --- 清空 pipeline ---
        clear_btn.click(lambda: ([], []), None, [pipeline_state, pipeline_json])

        # --- 运行 pipeline ---
        async def _run_pipeline(pl, jsonl_path, chat_api_url, apikey):
            if not pl:
                gr.Warning("Pipeline 为空，请先添加算子")
                return "", None, ""
            if not jsonl_path:
                gr.Warning("请输入 jsonl 文件路径")
                return "", None, ""

            # 调用后端
            try:
                final_state = await run_df_op_usage_pipeline(
                    matched_ops_with_params=pl,
                    json_file=jsonl_path,
                    chat_api_url=chat_api_url,
                    api_key=apikey,
                )
            except Exception as e:  # 捕获所有异常，防止 UI 崩溃
                import traceback, io

                buf = io.StringIO()
                traceback.print_exc(file=buf)
                return f"# 执行失败\n\n{buf.getvalue()}", None, ""

            # 1) 代码显示
            code_str = final_state['temp_data'].get("code", "# 未生成 code")

            # 2) 处理结果
            output_file = final_state['temp_data'].get("output_file",'# 未生成 output file')
            data_preview = None
            if output_file and Path(output_file).exists():
                data_preview = []
                with open(output_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        try:
                            data_preview.append(json.loads(line))
                        except Exception:
                            data_preview.append({"raw_line": line.strip()})

            return code_str, data_preview, output_file or ""

        run_btn.click(
            _run_pipeline,
            inputs=[pipeline_state, jsonl_path_tb, chat_api_url_tb, apikey_tb],
            outputs=[code_out, result_out, out_file_tb],
        )

    return page
