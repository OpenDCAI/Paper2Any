#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI – DataFlow Operator Pipeline Runner
"""

import json
import inspect
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


# -------------------- 工具函数 --------------------
def _format_default(val: Any):
    """
    把各种奇怪的默认值转换成 JSON 可以接受的类型
    """
    if val is inspect._empty:
        # 没有默认值 → 前端显示 null
        return None

    # 原生 JSON 类型直接返回
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val

    # 常见但不可 JSON 的类型做特殊处理
    from pathlib import Path
    if isinstance(val, Path):
        return str(val)

    # 其它无法序列化的对象 → 全部转成字符串
    # （用 str 而不是 repr，这样不会再出现额外的单引号）
    try:
        json.dumps(val)          # 能序列化直接用
        return val
    except TypeError:
        return str(val)


# def _format_default(val: Any):
#     if val is inspect._empty:
#         return None        
#     if isinstance(val, str):
#         return val          
#     return val   


def extract_op_params(cls: type) -> tuple:
    """
    提取算子的 __init__ 和 run 参数
    
    Returns:
        (init_kwargs, run_kwargs, has_prompt_template)
    """
    # __init__
    init_kwargs: List[tuple] = []
    has_prompt_template = False
    try:
        init_sig = inspect.signature(cls.__init__)
        for p in list(init_sig.parameters.values())[1:]:  # skip self
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.name == "prompt_template":
                has_prompt_template = True
            init_kwargs.append((p.name, _format_default(p.default)))
    except Exception as e:
        log.warning(f"inspect __init__ of {cls.__name__} failed: {e}")

    # run
    run_kwargs: List[tuple] = []
    if hasattr(cls, "run"):
        try:
            run_sig = inspect.signature(cls.run)
            params = list(run_sig.parameters.values())[1:]  # skip self
            for p in params:
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if p.name in ("storage", "self"):
                    continue
                run_kwargs.append((p.name, _format_default(p.default)))
        except Exception as e:
            log.warning(f"inspect run of {cls.__name__} failed: {e}")

    return init_kwargs, run_kwargs, has_prompt_template


def get_allowed_prompts(op_name: str) -> List[Dict[str, str]]:
    """
    获取算子的 ALLOWED_PROMPTS 列表
    
    Returns:
        [{"label": "PromptClassName", "value": "module.PromptClassName"}, ...]
    """
    cls = _ALL_OPS.get(op_name)
    if not cls:
        return []
    
    allowed_prompts = getattr(cls, "ALLOWED_PROMPTS", None)
    if not allowed_prompts:
        return []
    
    result = []
    for prompt_cls in allowed_prompts:
        result.append({
            "label": prompt_cls.__qualname__,
            "value": f"{prompt_cls.__module__}.{prompt_cls.__qualname__}"
        })
    return result


# -------------------- 后端执行 --------------------
async def run_df_op_usage_pipeline(
    matched_ops_with_params: List[Dict[str, Any]],
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
def create_page_testopuse():
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

                # Prompt Template 选择器（动态显示）
                prompt_dd = gr.Dropdown(
                    label="Prompt Template (可选)",
                    choices=[],
                    visible=False,
                    interactive=True,
                )

                # 分开显示 init 和 run 参数
                init_param_code = gr.Code(
                    label="__init__() 参数（JSON 可编辑）",
                    language="json",
                    value="{}",
                    lines=8,
                )
                
                run_param_code = gr.Code(
                    label="run() 参数（JSON 可编辑）",
                    language="json",
                    value="{}",
                    lines=8,
                )

                with gr.Row():
                    add_btn = gr.Button("➕ 添加算子到 Pipeline", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空 Pipeline", variant="secondary")

                pipeline_state = gr.State([])
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

        # --- 选算子 -> 展示参数 + prompt 选择器 ---
        def _show_params(op_name: str):
            if not op_name:
                return (
                    "_请选择一个算子_",
                    "{}",
                    "{}",
                    gr.Dropdown(visible=False, choices=[])
                )

            cls = _ALL_OPS[op_name]
            if not hasattr(cls, "run"):
                return (
                    f"⚠️ `{op_name}` 没有定义 run() 方法",
                    "{}",
                    "{}",
                    gr.Dropdown(visible=False, choices=[])
                )

            init_kwargs, run_kwargs, has_prompt = extract_op_params(cls)
            
            # 构建文档
            md_lines = [f"### `{op_name}` 参数说明\n"]
            md_lines.append("#### __init__() 参数")
            init_defaults = {}
            for n, default_val in init_kwargs:
                if n == "prompt_template":
                    md_lines.append(f"- **{n}**: 通过下拉框选择")
                else:
                    md_lines.append(f"- **{n}**: 默认值 `{default_val}`")
                # 跳过 llm_serving，不需要用户配置
                if n != "llm_serving":
                    init_defaults[n] = default_val

            md_lines.append("\n#### run() 参数")
            run_defaults = {}
            for n, default_val in run_kwargs:
                md_lines.append(f"- **{n}**: 默认值 `{default_val}`")
                run_defaults[n] = default_val

            # Prompt 下拉框
            if has_prompt:
                allowed_prompts = get_allowed_prompts(op_name)
                prompt_choices = [p["label"] for p in allowed_prompts]
                return (
                    "\n".join(md_lines),
                    json.dumps(init_defaults, indent=2, ensure_ascii=False),
                    json.dumps(run_defaults, indent=2, ensure_ascii=False),
                    gr.Dropdown(
                        visible=True,
                        choices=prompt_choices,
                        value=prompt_choices[0] if prompt_choices else None
                    )
                )
            else:
                return (
                    "\n".join(md_lines),
                    json.dumps(init_defaults, indent=2, ensure_ascii=False),
                    json.dumps(run_defaults, indent=2, ensure_ascii=False),
                    gr.Dropdown(visible=False, choices=[])
                )

        op_dd.change(
            _show_params,
            op_dd,
            [param_md, init_param_code, run_param_code, prompt_dd]
        )

        # --- Prompt 选择器变化 -> 更新 init_param_code ---
        def _update_prompt_in_init(op_name, prompt_label, init_json):
            if not prompt_label or not op_name:
                return init_json
            
            try:
                init_params = json.loads(init_json)
            except:
                init_params = {}
            
            # 获取对应的 value
            allowed_prompts = get_allowed_prompts(op_name)
            for p in allowed_prompts:
                if p["label"] == prompt_label:
                    init_params["prompt_template"] = p["value"]
                    break
            
            return json.dumps(init_params, indent=2, ensure_ascii=False)

        prompt_dd.change(
            _update_prompt_in_init,
            [op_dd, prompt_dd, init_param_code],
            init_param_code
        )

        # --- 添加算子到 pipeline ---
        def _add_op(op_name, init_json, run_json, pl):
            if not op_name:
                gr.Warning("⚠️ 请选择算子")
                return pl, pl
            try:
                init_params = json.loads(init_json or "{}")
                run_params = json.loads(run_json or "{}")
            except json.JSONDecodeError as e:
                gr.Warning(f"JSON 解析失败：{e}")
                return pl, pl
            
            pl = list(pl)
            pl.append({
                "op_name": op_name,
                "init_params": init_params,
                "run_params": run_params
            })
            gr.Info(f"✅ 已添加算子 `{op_name}`")
            return pl, pl

        add_btn.click(
            _add_op,
            [op_dd, init_param_code, run_param_code, pipeline_state],
            [pipeline_state, pipeline_json]
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

            # 调用后端执行
            try:
                final_state = await run_df_op_usage_pipeline(
                    matched_ops_with_params=pl,
                    json_file=jsonl_path,
                    chat_api_url=chat_api_url,
                    api_key=apikey,
                )
            except Exception as e:
                import traceback
                return f"# 执行失败\n\n{traceback.format_exc()}", None, ""

            # 处理结果
            code_str = final_state['temp_data'].get("code", "# 未生成 code")
            output_file = final_state['temp_data'].get("output_file", '')
            
            data_preview = None
            if output_file and Path(output_file).exists():
                data_preview = []
                with open(output_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        try:
                            data_preview.append(json.loads(line))
                        except:
                            data_preview.append({"raw_line": line.strip()})

            return code_str, data_preview, output_file

        run_btn.click(
            _run_pipeline,
            inputs=[pipeline_state, jsonl_path_tb, chat_api_url_tb, apikey_tb],
            outputs=[code_out, result_out, out_file_tb],
        )

    return page


# if __name__ == "__main__":
#     page = create_page_dfopuse()
#     page.launch(server_name="0.0.0.0", server_port=7860, share=False)