
"""
Gradio UI – DataFlow Operator Pipeline Runner
"""

import json
import inspect
import copy
import uuid
import html
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


def _is_empty_value(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, (list, tuple, dict)) and len(val) == 0:
        return True
    return False


def _normalize_param_key(key: str) -> str:
    if not isinstance(key, str):
        return ""
    lowered = key.lower()
    for token in ("input", "output", "key", "keys"):
        lowered = lowered.replace(token, "")
    lowered = lowered.replace("__", "_")
    return lowered.strip("_ ")


def _format_value_for_display(value: Any, max_length: int = 80) -> str:
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _standardize_value(val: Any) -> str:
    if val is None:
        return "__NONE__"
    if isinstance(val, (dict, list, tuple)):
        try:
            return json.dumps(val, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(val)
    return str(val)


def _collect_param_candidates(step: Dict[str, Any], role: str) -> List[Dict[str, Any]]:
    assert role in ("input", "output")
    results = []
    for container in ("init_params", "run_params"):
        params = step.get(container) or {}
        for key, value in params.items():
            if not isinstance(key, str):
                continue
            key_lower = key.lower()
            if role == "output" and "output" not in key_lower:
                continue
            if role == "input" and "input" not in key_lower:
                continue
            results.append({
                "container": container,
                "key": key,
                "value": value,
                "value_std": _standardize_value(value),
                "norm_key": _normalize_param_key(key),
            })
    return results


def _auto_link_pipeline(pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_pipeline = copy.deepcopy(pipeline or [])
    for idx in range(len(new_pipeline) - 1):
        src = new_pipeline[idx]
        tgt = new_pipeline[idx + 1]

        for container_name in ("init_params", "run_params"):
            if src.get(container_name) is None:
                src[container_name] = {}
            if tgt.get(container_name) is None:
                tgt[container_name] = {}

        outputs = _collect_param_candidates(src, "output")
        if not outputs:
            continue

        outputs_by_norm = {}
        for item in outputs:
            outputs_by_norm.setdefault(item["norm_key"], []).append(item)

        outputs_in_order = outputs

        inputs = _collect_param_candidates(tgt, "input")
        for inp in inputs:
            current_val = inp["value"]
            if not _is_empty_value(current_val):
                continue  # 用户已显式设置，尊重用户输入

            candidate_value = None
            if inp["norm_key"] in outputs_by_norm:
                candidate_value = outputs_by_norm[inp["norm_key"]][0]["value"]
            elif outputs_in_order:
                candidate_value = outputs_in_order[0]["value"]

            if candidate_value is not None:
                tgt[inp["container"]][inp["key"]] = candidate_value
    return new_pipeline


def _compute_pipeline_connections(pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    connections = []
    for idx in range(1, len(pipeline)):
        prev_step = pipeline[idx - 1]
        curr_step = pipeline[idx]

        outputs = _collect_param_candidates(prev_step, "output")
        inputs = _collect_param_candidates(curr_step, "input")

        outputs_by_value = {}
        for out in outputs:
            outputs_by_value.setdefault(out["value_std"], []).append(out)

        matches = []
        missing = []
        for inp in inputs:
            if _is_empty_value(inp["value"]):
                missing.append({
                    "input_key": inp["key"],
                    "input_container": inp["container"],
                    "status": "empty",
                    "current_value": inp["value"],
                })
                continue

            value_std = _standardize_value(inp["value"])
            if value_std in outputs_by_value:
                matches.append({
                    "input_key": inp["key"],
                    "input_container": inp["container"],
                    "value": inp["value"],
                    "output_keys": [out["key"] for out in outputs_by_value[value_std]],
                    "output_containers": [out["container"] for out in outputs_by_value[value_std]],
                })
            else:
                missing.append({
                    "input_key": inp["key"],
                    "input_container": inp["container"],
                    "status": "no_match",
                    "current_value": inp["value"],
                })

        connections.append({
            "source_index": idx - 1,
            "target_index": idx,
            "matches": matches,
            "missing": missing,
            "all_outputs": [
                {
                    "key": out["key"],
                    "container": out["container"],
                    "value": out["value"],
                }
                for out in outputs
            ],
        })
    return connections


def _prepare_pipeline_json(pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    display: List[Dict[str, Any]] = []
    connections = _compute_pipeline_connections(pipeline)
    conn_map = {c["target_index"]: c for c in connections}

    for idx, step in enumerate(pipeline):
        item = {
            "op_name": step.get("op_name"),
            "init_params": {k: _format_default(v) for k, v in (step.get("init_params") or {}).items()},
            "run_params": {k: _format_default(v) for k, v in (step.get("run_params") or {}).items()},
        }
        conn = conn_map.get(idx)
        if conn:
            item["_incoming_links"] = [
                {
                    "input_key": link["input_key"],
                    "input_container": link["input_container"],
                    "value": _format_default(link["value"]),
                    "output_keys": link["output_keys"],
                    "output_containers": link["output_containers"],
                }
                for link in conn["matches"]
            ]
            item["_incoming_unlinked_inputs"] = [
                {
                    "input_key": miss["input_key"],
                    "input_container": miss["input_container"],
                    "status": miss["status"],
                    "current_value": _format_default(miss.get("current_value")),
                }
                for miss in conn["missing"]
            ]
        display.append(item)

    return display


def _render_param_section(title: str, params: Dict[str, Any]) -> str:
    title_html = html.escape(title)
    params = params or {}
    if not params:
        return f"""
        <div class="pipeline-param-block">
            <h4>{title_html}</h4>
            <div class="pipeline-param-empty">无</div>
        </div>
        """
    items_html = []
    for idx, (key, value) in enumerate(params.items()):
        if idx >= 5:
            items_html.append("<div class='pipeline-param-more'>…</div>")
            break
        key_html = html.escape(str(key))
        value_html = html.escape(_format_value_for_display(value))
        items_html.append(
            f"<div class='pipeline-param-item'><code>{key_html}</code>: <span>{value_html}</span></div>"
        )
    return f"""
    <div class="pipeline-param-block">
        <h4>{title_html}</h4>
        {''.join(items_html)}
    </div>
    """


def _render_connection_section(step_index: int, connection_map: Dict[int, Dict[str, Any]]) -> str:
    if step_index == 0:
        return "<div class='pipeline-connections pipeline-connections-empty'>起始算子</div>"

    conn = connection_map.get(step_index)
    if not conn:
        return "<div class='pipeline-connections pipeline-connections-empty'>未检测到来自上一步的输出键</div>"

    sections = []

    matches = conn.get("matches") or []
    if matches:
        match_lines = []
        for match in matches:
            outputs_text = "、".join(html.escape(k) for k in match["output_keys"])
            value_text = html.escape(_format_value_for_display(match["value"]))
            match_lines.append(
                f"<div class='pipeline-connection-line'>🔗 <span class='pipeline-output-keys'>{outputs_text}</span> → "
                f"<code>{html.escape(match['input_key'])}</code>"
                f"<span class='pipeline-connection-value'>值：{value_text}</span></div>"
            )
        sections.append(
            "<div class='pipeline-connection-matches'><span class='badge success'>已链接</span>"
            + "".join(match_lines)
            + "</div>"
        )

    missing = conn.get("missing") or []
    if missing:
        missing_lines = []
        for miss in missing:
            reason = "空值" if miss["status"] == "empty" else "未匹配上一步 output"
            current_val = miss.get("current_value")
            if not _is_empty_value(current_val):
                current_text = html.escape(_format_value_for_display(current_val))
                missing_lines.append(
                    f"<div class='pipeline-connection-line'>⚠️ <code>{html.escape(miss['input_key'])}</code> （{reason}）"
                    f" 当前值：{current_text}</div>"
                )
            else:
                missing_lines.append(
                    f"<div class='pipeline-connection-line'>⚠️ <code>{html.escape(miss['input_key'])}</code> （{reason}）</div>"
                )
        sections.append(
            "<div class='pipeline-connection-missing'><span class='badge warn'>待处理</span>"
            + "".join(missing_lines)
            + "</div>"
        )

    if not sections:
        available_outputs = conn.get("all_outputs") or []
        if available_outputs:
            outputs_desc = ", ".join(
                f"{html.escape(item['key'])}={html.escape(_format_value_for_display(item['value']))}"
                for item in available_outputs
            )
            sections.append(
                "<div class='pipeline-connections pipeline-connections-empty'>上一步输出: "
                + outputs_desc
                + "</div>"
            )
        else:
            sections.append("<div class='pipeline-connections pipeline-connections-empty'>未检测到 input/output 字段</div>")

    return "<div class='pipeline-connections'>" + "".join(sections) + "</div>"


def _render_pipeline_html(pipeline: List[Dict[str, Any]]) -> str:
    connections = _compute_pipeline_connections(pipeline)
    connection_map = {c["target_index"]: c for c in connections}

    parts: List[str] = [
        """
        <style>
            .pipeline-wrapper {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 0.75rem;
                background: #fafafa;
            }
            .pipeline-list {
                list-style: none;
                margin: 0;
                padding: 0;
            }
            .pipeline-item {
                background: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                margin-bottom: 0.75rem;
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
                transition: box-shadow 0.2s ease, transform 0.2s ease;
            }
            .pipeline-item:last-child {
                margin-bottom: 0;
            }
            .pipeline-item-header {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.55rem 0.8rem;
                border-bottom: 1px solid #f0f0f0;
                background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(59,130,246,0.02));
                cursor: grab;
            }
            .pipeline-item-header:active {
                cursor: grabbing;
            }
            .pipeline-step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: #2563eb;
                color: #ffffff;
                font-weight: 600;
            }
            .pipeline-op-name {
                font-weight: 600;
                color: #1f2937;
                flex: 1;
            }
            .pipeline-drag-hint {
                font-size: 0.9rem;
                color: #6b7280;
            }
            .pipeline-item-body {
                padding: 0.7rem 0.9rem 0.9rem;
                display: grid;
                gap: 0.6rem;
            }
            .pipeline-param-block h4 {
                margin: 0 0 0.3rem 0;
                font-size: 0.95rem;
                color: #334155;
            }
            .pipeline-param-item {
                font-size: 0.88rem;
                color: #1f2937;
                display: flex;
                gap: 0.35rem;
                align-items: baseline;
                word-break: break-all;
            }
            .pipeline-param-item code {
                background: rgba(37, 99, 235, 0.08);
                padding: 0.05rem 0.35rem;
                border-radius: 4px;
                font-size: 0.82rem;
            }
            .pipeline-param-more {
                font-size: 0.82rem;
                color: #94a3b8;
            }
            .pipeline-param-empty {
                font-size: 0.85rem;
                color: #9ca3af;
            }
            .pipeline-connections {
                border-top: 1px dashed #e5e7eb;
                padding-top: 0.55rem;
                font-size: 0.86rem;
                display: grid;
                gap: 0.35rem;
            }
            .pipeline-connections-empty {
                color: #9ca3af;
            }
            .pipeline-connection-line {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                align-items: baseline;
            }
            .pipeline-output-keys {
                font-weight: 600;
                color: #2563eb;
            }
            .pipeline-connection-value {
                color: #4b5563;
            }
            .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.15rem 0.45rem;
                border-radius: 999px;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.03em;
                font-weight: 600;
            }
            .badge.success {
                background: rgba(22, 163, 74, 0.1);
                color: #15803d;
            }
            .badge.warn {
                background: rgba(234, 179, 8, 0.12);
                color: #ca8a04;
            }
            .pipeline-item-empty {
                padding: 0.8rem;
                text-align: center;
                color: #6b7280;
                background: repeating-linear-gradient(
                    -45deg,
                    #ffffff,
                    #ffffff 12px,
                    #f8fafc 12px,
                    #f8fafc 24px
                );
            }
            .pipeline-wrapper .pipeline-item.sortable-chosen {
                box-shadow: 0 6px 18px rgba(59, 130, 246, 0.18);
                transform: translateY(-3px);
            }
            .pipeline-wrapper .pipeline-item.sortable-ghost {
                opacity: 0.6;
            }
        </style>
        <div class="pipeline-wrapper">
            <ul class="pipeline-list" id="pipeline-list">
        """
    ]

    if not pipeline:
        parts.append("<li class='pipeline-item pipeline-item-empty'>暂无算子，请从左侧选择并添加。</li>")
    else:
        for idx, step in enumerate(pipeline):
            uid_attr = html.escape(step.get("uid", str(uuid.uuid4())))
            op_name = html.escape(step.get("op_name", "未知算子"))
            init_section = _render_param_section("__init__() 参数", step.get("init_params"))
            run_section = _render_param_section("run() 参数", step.get("run_params"))
            connection_section = _render_connection_section(idx, connection_map)

            parts.append(
                f"""
                <li class="pipeline-item" data-uid="{uid_attr}">
                    <div class="pipeline-item-header">
                        <span class="pipeline-step-number">{idx + 1}</span>
                        <span class="pipeline-op-name">{op_name}</span>
                        <span class="pipeline-drag-hint">⇅</span>
                    </div>
                    <div class="pipeline-item-body">
                        {init_section}
                        {run_section}
                        {connection_section}
                    </div>
                </li>
                """
            )

    parts.append(
        """
            </ul>
        </div>
        <script>
        (function() {
            const pipelineRoot = document.getElementById("pipeline-list");
            if (!pipelineRoot) {
                return;
            }

            function ensureSortable(callback) {
                if (window.Sortable) {
                    callback();
                    return;
                }
                if (window.__pipelineSortableLoading) {
                    window.__pipelineSortableLoading.push(callback);
                    return;
                }
                window.__pipelineSortableLoading = [callback];
                const script = document.createElement("script");
                script.src = "https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js";
                script.onload = () => {
                    const cbs = window.__pipelineSortableLoading || [];
                    cbs.forEach(fn => {
                        try { fn(); } catch (err) { console.error(err); }
                    });
                    window.__pipelineSortableLoading = null;
                };
                document.head.appendChild(script);
            }

            ensureSortable(() => {
                if (pipelineRoot.dataset.sortableInit === "true") {
                    return;
                }
                pipelineRoot.dataset.sortableInit = "true";
                Sortable.create(pipelineRoot, {
                    animation: 150,
                    handle: ".pipeline-item-header",
                    onEnd: () => {
                        const items = pipelineRoot.querySelectorAll(".pipeline-item");
                        let visibleIndex = 0;
                        items.forEach(item => {
                            if (!item.dataset.uid) {
                                return;
                            }
                            visibleIndex += 1;
                            const badge = item.querySelector(".pipeline-step-number");
                            if (badge) {
                                badge.textContent = visibleIndex;
                            }
                        });

                        const order = Array.from(items)
                            .map(item => item.dataset.uid)
                            .filter(uid => !!uid);

                        if (order.length === 0) {
                            return;
                        }

                        const holder = document.querySelector('#pipeline-order-holder textarea, #pipeline-order-holder input');
                        if (holder) {
                            holder.value = JSON.stringify(order);
                            holder.dispatchEvent(new Event('input', { bubbles: true }));
                            holder.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                });
            });
        })();
        </script>
        """
    )

    return "\n".join(parts)


def _update_pipeline(pipeline: List[Dict[str, Any]]):
    pipeline = pipeline or []
    linked_pipeline = _auto_link_pipeline(pipeline)
    display_json = _prepare_pipeline_json(linked_pipeline)
    html_view = _render_pipeline_html(linked_pipeline)
    return linked_pipeline, display_json, html_view, ""


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
def create_op_assemble_line():
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
            # 新增模型名称输入框
            model_name_tb = gr.Textbox(
                label="模型名称",
                value="gpt-4o",
                scale=2,
            )
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

                gr.Markdown("#### 当前 Pipeline 序列（支持拖动调整顺序）")
                pipeline_state = gr.State([])
                pipeline_json = gr.JSON(label="当前 Pipeline（含自动链接信息）", value=[])
                pipeline_dnd_html = gr.HTML(value=_render_pipeline_html([]))
                pipeline_order_holder = gr.Textbox(value="", visible=False, elem_id="pipeline-order-holder")

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
                if n != "llm_serving":
                    init_defaults[n] = default_val

            md_lines.append("\n#### run() 参数")
            run_defaults = {}
            for n, default_val in run_kwargs:
                md_lines.append(f"- **{n}**: 默认值 `{default_val}`")
                run_defaults[n] = default_val

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
            pl = list(pl or [])
            if not op_name:
                gr.Warning("⚠️ 请选择算子")
                return _update_pipeline(pl)
            try:
                init_params = json.loads(init_json or "{}")
                run_params = json.loads(run_json or "{}")
            except json.JSONDecodeError as e:
                gr.Warning(f"JSON 解析失败：{e}")
                return _update_pipeline(pl)

            if not isinstance(init_params, dict):
                gr.Warning("`__init__()` 参数必须是 JSON 对象（key-value）")
                return _update_pipeline(pl)

            if not isinstance(run_params, dict):
                gr.Warning("`run()` 参数必须是 JSON 对象（key-value）")
                return _update_pipeline(pl)
            
            pl.append({
                "uid": str(uuid.uuid4()),
                "op_name": op_name,
                "init_params": init_params,
                "run_params": run_params
            })
            gr.Info(f"✅ 已添加算子 `{op_name}`")
            return _update_pipeline(pl)

        add_btn.click(
            _add_op,
            [op_dd, init_param_code, run_param_code, pipeline_state],
            [pipeline_state, pipeline_json, pipeline_dnd_html, pipeline_order_holder]
        )

        # --- 清空 pipeline ---
        def _clear_pipeline():
            gr.Info("🧹 Pipeline 已清空")
            return _update_pipeline([])

        clear_btn.click(
            _clear_pipeline,
            None,
            [pipeline_state, pipeline_json, pipeline_dnd_html, pipeline_order_holder]
        )

        # --- 拖拽排序 ---
        def _apply_new_order(pl, order_json):
            pl = list(pl or [])
            if not order_json:
                return _update_pipeline(pl)
            try:
                uid_order = json.loads(order_json)
                if not isinstance(uid_order, list):
                    raise ValueError("order_json 不是列表")
            except Exception as exc:
                log.warning(f"解析排序结果失败: {exc}")
                return _update_pipeline(pl)

            uid2step = {step.get("uid"): step for step in pl if isinstance(step, dict)}
            new_pl = []
            for uid in uid_order:
                step = uid2step.get(uid)
                if step:
                    new_pl.append(step)
            for step in pl:
                if step.get("uid") not in uid_order:
                    new_pl.append(step)
            return _update_pipeline(new_pl)

        pipeline_order_holder.change(
            _apply_new_order,
            [pipeline_state, pipeline_order_holder],
            [pipeline_state, pipeline_json, pipeline_dnd_html, pipeline_order_holder]
        )

        # --- 运行 pipeline ---
        async def _run_pipeline(pl, jsonl_path, chat_api_url, apikey, model_name):
            if not pl:
                gr.Warning("Pipeline 为空，请先添加算子")
                return "", None, ""
            if not jsonl_path:
                gr.Warning("请输入 jsonl 文件路径")
                return "", None, ""

            prepared_pipeline = _auto_link_pipeline(pl)
            clean_pipeline = [
                {
                    "op_name": step.get("op_name"),
                    "init_params": step.get("init_params") or {},
                    "run_params": step.get("run_params") or {},
                }
                for step in prepared_pipeline
            ]

            try:
                final_state = await run_df_op_usage_pipeline(
                    matched_ops_with_params=clean_pipeline,
                    json_file=jsonl_path,
                    chat_api_url=chat_api_url,
                    api_key=apikey,
                    model= model_name
                )
            except Exception as e:
                import traceback
                return f"# 执行失败\n\n{traceback.format_exc()}", None, ""

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
            inputs=[pipeline_state, jsonl_path_tb, chat_api_url_tb, apikey_tb, model_name_tb],
            outputs=[code_out, result_out, out_file_tb],
        )

    return page

