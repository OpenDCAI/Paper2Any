import json
import inspect
from collections import defaultdict

import gradio as gr

# -------------------- 准备算子元数据 --------------------
from dataflow.utils.registry import OPERATOR_REGISTRY
import dataflow.operators           
OPERATOR_REGISTRY._get_all()         

# name -> class
_ALL_OPS = OPERATOR_REGISTRY.get_obj_map()

# category -> [op_name, ...]
_CAT2OPS = defaultdict(list)
for op_name, cls in _ALL_OPS.items():
    mod_parts = cls.__module__.split(".")
    # dataflow.operators.<category>.*
    if "operators" in mod_parts:
        cat_idx = mod_parts.index("operators") + 1
        category = mod_parts[cat_idx] if cat_idx < len(mod_parts) else "uncategorized"
    else:
        category = "uncategorized"
    _CAT2OPS[category].append(op_name)

# -------------------- Gradio 页面 --------------------
def create_page_dfopuse():
    with gr.Blocks() as page:
        gr.Markdown("## 🧩 DataFlow Operator Selector")

        with gr.Row(equal_height=True):
            # ---------- 左侧：分类 & 算子选择 ----------
            with gr.Column(scale=4):
                cat_dd = gr.Dropdown(
                    label="算子分类",
                    choices=sorted(_CAT2OPS.keys()),
                    value=None
                )
                op_dd = gr.Dropdown(
                    label="算子",
                    choices=[],
                    value=None
                )

                param_json = gr.Code(
                    label="run() 参数（JSON 可编辑）",
                    language="json",
                    value="{}",
                    lines=12
                )
                add_btn = gr.Button("添加算子到 Pipeline", variant="primary")

            # ---------- 右侧：参数说明 ----------
            with gr.Column(scale=6):
                param_md = gr.Markdown("请选择一个算子")

        # ----------- 交互逻辑 -----------
        def _update_op_dd(cat):
            "选中分类时刷新算子下拉选项"
            ops = sorted(_CAT2OPS.get(cat, []))
            # 旧版（错误）：return gr.Dropdown.update(choices=ops, value=ops[0] if ops else None)
            return gr.Dropdown(choices=ops, value=ops[0] if ops else None)

        cat_dd.change(
            _update_op_dd,
            inputs=[cat_dd],
            outputs=[op_dd]
        )

        def _show_params(op_name):
            "选中算子时，解析 run() 签名并生成默认 JSON 与说明"
            if not op_name:
                return (
                    "请选择一个算子",  # 直接返回字符串
                    "{}"
                )

            cls = _ALL_OPS[op_name]
            if not hasattr(cls, "run"):
                return (
                    f"⚠️ `{op_name}` 没有定义 run() 方法",
                    "{}"
                )

            sig = inspect.signature(cls.run)
            md_lines = [f"### `{op_name}.run()` 参数"]
            default_dict = {}
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                default = param.default if param.default is not inspect._empty else ""
                annotation = (
                    f"`{param.annotation.__name__}`"
                    if param.annotation not in (inspect._empty, None)
                    else ""
                )
                md_lines.append(f"- **{name}** {annotation}  默认：`{default}`")
                default_dict[name] = default
            md_text = "\n".join(md_lines) if len(md_lines) > 1 else "_无参数_"

            json_value = json.dumps(default_dict, indent=2, ensure_ascii=False)
            # 直接返回新值，不用 .update()
            return (md_text, json_value)

        op_dd.change(
            _show_params,
            inputs=[op_dd],
            outputs=[param_md, param_json]
        )

        def _add_operator(op_name, param_json_str):
            "点击按钮 -> 返回 (算子名, 参数字典)。这里先简单打印，可接入 generate_pipeline"
            try:
                param_dict = json.loads(param_json_str or "{}")
            except json.JSONDecodeError as e:
                return gr.Info(f"❌ JSON 解析失败：{e}")
            print("📦 已选择算子:", op_name, "参数:", param_dict)
            return gr.Success(f"✅ 已添加算子 `{op_name}`")

        add_btn.click(
            _add_operator,
            inputs=[op_dd, param_json],
            outputs=None        
        )

    return page

# if __name__ == "__main__":
#     demo = create_operator_selector()
#     demo.launch(server_port=7860, share=False)