import gradio as gr
import json
from ..utils.wf_pipeine_rec import run_pipeline_workflow
from ..utils.wf_pipeline_refine import run_pipeline_refine_workflow, python_to_json, json_to_python_code
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


def create_pipeline_rec():
    """子页面：Pipeline 生成（带 Agent 结果展示）+ 二次优化功能"""
    with gr.Blocks(theme=gr.themes.Default()) as page:
        gr.Markdown("# 🚀 DataFlow Pipeline Generator")

        # ==================== State 组件 ====================
        # 存储当前 pipeline 的 JSON 结构（用于二次优化）
        pipeline_json_state = gr.State(value={})
        # 存储优化轮次
        refine_round_state = gr.State(value=0)
        # 存储 API 配置（用于复用）
        api_config_state = gr.State(value={})
        # 存储所有优化轮次的历史记录
        # 结构: [{round: 0, code: "...", json: {...}, log: {...}, requirement: "..."}, ...]
        refine_history_state = gr.State(value=[])
        # 当前查看的轮次索引（0-based）
        current_view_index_state = gr.State(value=-1)

        # ==================== Pipeline 生成区域 ====================
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=2):
                target = gr.Textbox(
                    label="目标描述",
                    placeholder="给我随意符合逻辑的5个算子，过滤，去重！",
                    lines=3
                )
                json_file = gr.Textbox(
                    label="输入 JSONL 文件路径",
                    value=f"{get_project_root()}/tests/test.jsonl"
                )
                session_id = gr.Textbox(
                    label="Session ID",
                    value="default"
                )
                
                # 主要聊天 API 配置
                gr.Markdown("### 主要模型配置")
                chat_api_url = gr.Textbox(
                    label="Chat API URL",
                    value="http://123.129.219.111:3000/v1/"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value="",
                    type="password"
                )
                model_name = gr.Textbox(
                    label="模型名称",
                    placeholder="如：gpt-4o, qwen-max, llama3, etc.",
                    value="gpt-4o"
                )
                
                # 嵌入模型配置
                gr.Markdown("### 嵌入模型配置")
                chat_api_url_for_embeddings = gr.Textbox(
                    label="Embedding API URL",
                    placeholder="留空则使用主要 API URL",
                    value=""
                )
                embedding_model_name = gr.Textbox(
                    label="Embedding 模型名称",
                    placeholder="如：text-embedding-3-small",
                    value="text-embedding-3-small"
                )
                
                # RAG 配置
                gr.Markdown("### RAG 配置")
                update_rag = gr.Checkbox(
                    label="实时更新 RAG 索引（检测到未注册算子时自动重建索引）", 
                    value=True
                )
                
                # 调试配置
                gr.Markdown("### 调试配置")
                debug_mode = gr.Checkbox(label="启用调试模式", value=False)
                debug_times = gr.Dropdown(
                    label="调试模式执行次数",
                    choices=[1, 2, 3, 5, 10],
                    value=2,
                    visible=False
                )
                
                submit_btn = gr.Button("🚀 Generate Pipeline", variant="primary", size="lg")

            # 右侧：输出区（4 个页签）
            with gr.Column(scale=3):
                gr.Markdown("### 📊 Generation Results")
                with gr.Tabs():
                    with gr.Tab("Pipeline Code"):
                        output_code = gr.Code(label="Generated Python Code", language="python", lines=20)
                    with gr.Tab("Execution Log"):
                        output_log = gr.Textbox(label="Execution Log", lines=15)
                    with gr.Tab("Agent Results"):
                        agent_results_json = gr.JSON(label="Agent Results")
                    with gr.Tab("Pipeline JSON"):
                        pipeline_json_display = gr.JSON(label="Pipeline JSON Structure")

        # ==================== Pipeline 优化区域 ====================
        gr.Markdown("---")
        with gr.Row():
            # 左侧：优化输入区
            with gr.Column(scale=2):
                gr.Markdown("### 🔧 Pipeline 二次优化")
                gr.Markdown("*在上方生成 Pipeline 后，可以在此输入优化需求进行二次调整。支持多轮连续优化。*")
                refine_target = gr.Textbox(
                    label="优化需求",
                    placeholder="例如：请将 Pipeline 简化为只包含 3 个节点；添加一个数据过滤步骤；调整节点执行顺序...",
                    lines=3
                )
                refine_btn = gr.Button("🔄 Refine Pipeline", variant="primary", size="lg")
                refine_round_display = gr.Markdown("**当前优化轮次：** 0")
            
            # 右侧：优化结果展示区
            with gr.Column(scale=3):
                gr.Markdown("### 📈 Refinement Results")
                
                # 历史导航栏
                with gr.Row(elem_classes=["history-nav"]):
                    prev_btn = gr.Button("◀ 上一轮", size="sm", scale=1, interactive=False)
                    history_indicator = gr.Markdown("**暂无优化历史**", elem_classes=["history-indicator"])
                    next_btn = gr.Button("下一轮 ▶", size="sm", scale=1, interactive=False)
                
                with gr.Tabs():
                    with gr.Tab("Refined Pipeline Code"):
                        refined_code = gr.Code(label="Refined Python Code", language="python", lines=18)
                    with gr.Tab("Refined Pipeline Json"):
                        refined_json_display = gr.JSON(label="Refined Pipeline JSON Structure")
                    with gr.Tab("Execution Log"):
                        refine_log = gr.JSON(label="Refinement Agent Results")

        # ==================== 事件绑定 ====================
        
        # 调试模式显示下拉
        def toggle_debug_times(is_debug):
            return gr.update(visible=is_debug)

        debug_mode.change(
            toggle_debug_times,
            inputs=debug_mode,
            outputs=debug_times
        )

        # ---------------------- 生成 Pipeline 回调 ----------------------
        async def generate_pipeline(
            target_text, 
            json_path, 
            session_id_val, 
            chat_api_url_val, 
            api_key_val, 
            model_name_val,
            chat_api_url_for_embeddings_val,
            embedding_model_name_val,
            update_rag_val,
            debug,
            max_debug_rounds,
            current_json_state,
            current_round_state,
            current_api_config
        ):
            result = await run_pipeline_workflow(
                target=target_text,
                json_file=json_path,
                need_debug=debug,
                session_id=session_id_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val,
                model_name=model_name_val,
                max_debug_rounds=max_debug_rounds if debug else 2,
                chat_api_url_for_embeddings=chat_api_url_for_embeddings_val,
                embedding_model_name=embedding_model_name_val,
                update_rag_content=update_rag_val
            )

            # 读取生成的 Python 文件
            with open(result["python_file"], "r") as f:
                code = f.read()

            log_result = result["execution_result"]
            agent_results = result.get("agent_results", {})
            
            # 解析 Python 文件为 JSON 结构（用于二次优化）
            pipeline_json = {}
            try:
                pipeline_json = python_to_json(result["python_file"])
                log.info(f"[generate_pipeline] 成功解析 Pipeline JSON，节点数: {len(pipeline_json.get('nodes', []))}")
            except Exception as e:
                log.warning(f"[generate_pipeline] 解析 Pipeline JSON 失败: {e}")
                pipeline_json = {"error": str(e), "nodes": [], "edges": []}
            
            # 保存 API 配置（用于优化时复用）
            api_config = {
                "chat_api_url": chat_api_url_val,
                "api_key": api_key_val,
                "model_name": model_name_val,
                "json_file": json_path,
            }
            
            # 重置优化轮次和历史记录
            new_round = 0
            round_text = f"**当前优化轮次：** {new_round}"
            empty_history = []
            view_index = -1
            history_text = "**暂无优化历史**"
            
            return (
                code,                    # output_code
                log_result,              # output_log
                agent_results,           # agent_results_json
                pipeline_json,           # pipeline_json_display
                pipeline_json,           # pipeline_json_state (更新 State)
                new_round,               # refine_round_state (重置为 0)
                api_config,              # api_config_state (保存配置)
                round_text,              # refine_round_display
                empty_history,           # refine_history_state (清空历史)
                view_index,              # current_view_index_state
                history_text,            # history_indicator
                gr.update(interactive=False),  # prev_btn
                gr.update(interactive=False),  # next_btn
                "",                      # refined_code (清空)
                {},                      # refined_json_display (清空)
                {},                      # refine_log (清空)
            )

        submit_btn.click(
            generate_pipeline,
            inputs=[
                target, 
                json_file, 
                session_id, 
                chat_api_url, 
                api_key, 
                model_name,
                chat_api_url_for_embeddings,
                embedding_model_name,
                update_rag,
                debug_mode, 
                debug_times,
                pipeline_json_state,
                refine_round_state,
                api_config_state,
            ],
            outputs=[
                output_code, 
                output_log, 
                agent_results_json,
                pipeline_json_display,
                pipeline_json_state,
                refine_round_state,
                api_config_state,
                refine_round_display,
                refine_history_state,
                current_view_index_state,
                history_indicator,
                prev_btn,
                next_btn,
                refined_code,
                refined_json_display,
                refine_log,
            ]
        )

        # ---------------------- 优化 Pipeline 回调 ----------------------
        async def refine_pipeline(
            refine_target_text,
            current_json_state,
            current_round_state,
            current_api_config,
            current_history,
            current_view_index,
        ):
            # 检查是否有可优化的 Pipeline
            if not current_json_state or not current_json_state.get("nodes"):
                return (
                    "# 请先生成 Pipeline，然后再进行优化",
                    {"error": "没有可优化的 Pipeline，请先点击 'Generate Pipeline' 按钮"},
                    {"error": "没有可优化的 Pipeline"},
                    current_json_state,
                    current_round_state,
                    f"**当前优化轮次：** {current_round_state}",
                    current_history,
                    current_view_index,
                    "**暂无优化历史**",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            
            # 检查是否输入了优化需求
            if not refine_target_text or not refine_target_text.strip():
                return (
                    "# 请输入优化需求",
                    {"error": "请在 '优化需求' 输入框中描述您的优化需求"},
                    {"error": "优化需求为空"},
                    current_json_state,
                    current_round_state,
                    f"**当前优化轮次：** {current_round_state}",
                    current_history,
                    current_view_index,
                    f"**第 {current_view_index + 1} 轮 / 共 {len(current_history)} 轮**" if current_history else "**暂无优化历史**",
                    gr.update(interactive=current_view_index > 0) if current_history else gr.update(interactive=False),
                    gr.update(interactive=current_view_index < len(current_history) - 1) if current_history else gr.update(interactive=False),
                )
            
            # 调用优化 workflow
            result = await run_pipeline_refine_workflow(
                refine_target=refine_target_text,
                pipeline_json=current_json_state,
                chat_api_url=current_api_config.get("chat_api_url", ""),
                api_key=current_api_config.get("api_key", ""),
                model_name=current_api_config.get("model_name", "gpt-4o"),
                json_file=current_api_config.get("json_file", ""),
            )
            
            if result["success"]:
                new_round = current_round_state + 1
                refined_json = result["refined_json"]
                python_code = result["python_code"]
                agent_results = result["agent_results"]
                
                round_text = f"**当前优化轮次：** {new_round}"
                
                # 保存到历史记录
                new_history_entry = {
                    "round": new_round,
                    "code": python_code,
                    "json": refined_json,
                    "log": agent_results,
                    "requirement": refine_target_text,
                }
                new_history = current_history + [new_history_entry]
                new_view_index = len(new_history) - 1  # 跳转到最新轮次
                
                # 更新历史指示器
                history_text = f"**第 {new_view_index + 1} 轮 / 共 {len(new_history)} 轮**"
                
                # 更新导航按钮状态
                prev_interactive = new_view_index > 0
                next_interactive = False  # 已经是最新的
                
                return (
                    python_code,         # refined_code
                    refined_json,        # refined_json_display
                    agent_results,       # refine_log
                    refined_json,        # pipeline_json_state (更新为优化后的 JSON，支持多轮优化)
                    new_round,           # refine_round_state
                    round_text,          # refine_round_display
                    new_history,         # refine_history_state
                    new_view_index,      # current_view_index_state
                    history_text,        # history_indicator
                    gr.update(interactive=prev_interactive),  # prev_btn
                    gr.update(interactive=next_interactive),  # next_btn
                )
            else:
                error_msg = result.get("error", "未知错误")
                history_text = f"**第 {current_view_index + 1} 轮 / 共 {len(current_history)} 轮**" if current_history else "**暂无优化历史**"
                return (
                    f"# 优化失败: {error_msg}",
                    {"error": error_msg},
                    result.get("agent_results", {"error": error_msg}),
                    current_json_state,  # 保持原 JSON 不变
                    current_round_state, # 轮次不变
                    f"**当前优化轮次：** {current_round_state}（优化失败）",
                    current_history,     # 历史不变
                    current_view_index,  # 视图索引不变
                    history_text,
                    gr.update(interactive=current_view_index > 0) if current_history else gr.update(interactive=False),
                    gr.update(interactive=current_view_index < len(current_history) - 1) if current_history else gr.update(interactive=False),
                )

        refine_btn.click(
            refine_pipeline,
            inputs=[
                refine_target,
                pipeline_json_state,
                refine_round_state,
                api_config_state,
                refine_history_state,
                current_view_index_state,
            ],
            outputs=[
                refined_code,
                refined_json_display,
                refine_log,
                pipeline_json_state,
                refine_round_state,
                refine_round_display,
                refine_history_state,
                current_view_index_state,
                history_indicator,
                prev_btn,
                next_btn,
            ]
        )

        # ---------------------- 历史导航回调 ----------------------
        def navigate_history(direction, current_history, current_view_index):
            """导航到历史记录中的不同轮次"""
            if not current_history:
                return (
                    "",
                    {},
                    {},
                    -1,
                    "**暂无优化历史**",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            
            # 计算新的索引
            if direction == "prev":
                new_index = max(0, current_view_index - 1)
            else:  # next
                new_index = min(len(current_history) - 1, current_view_index + 1)
            
            # 获取对应轮次的数据
            history_entry = current_history[new_index]
            
            # 更新历史指示器
            history_text = f"**第 {new_index + 1} 轮 / 共 {len(current_history)} 轮**"
            
            # 更新导航按钮状态
            prev_interactive = new_index > 0
            next_interactive = new_index < len(current_history) - 1
            
            return (
                history_entry["code"],
                history_entry["json"],
                history_entry["log"],
                new_index,
                history_text,
                gr.update(interactive=prev_interactive),
                gr.update(interactive=next_interactive),
            )

        prev_btn.click(
            lambda h, i: navigate_history("prev", h, i),
            inputs=[refine_history_state, current_view_index_state],
            outputs=[
                refined_code,
                refined_json_display,
                refine_log,
                current_view_index_state,
                history_indicator,
                prev_btn,
                next_btn,
            ]
        )

        next_btn.click(
            lambda h, i: navigate_history("next", h, i),
            inputs=[refine_history_state, current_view_index_state],
            outputs=[
                refined_code,
                refined_json_display,
                refine_log,
                current_view_index_state,
                history_indicator,
                prev_btn,
                next_btn,
            ]
        )

    return page
