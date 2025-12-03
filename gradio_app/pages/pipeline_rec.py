import gradio as gr
from ..utils.wf_pipeine_rec import run_pipeline_workflow
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

def create_pipeline_rec():
    """子页面：Pipeline 生成（带 Agent 结果展示）"""
    with gr.Blocks(theme=gr.themes.Default()) as page:
        gr.Markdown("# 🚀 DataFlow Pipeline Generator")

        with gr.Row():
            # 左侧：输入区
            with gr.Column():
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
                gr.Markdown("### 嵌入模型配置 http://123.129.219.111:3000/v1/embeddings")
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
                
                # 调试配置（暂时禁用）
                # gr.Markdown("### 调试配置")
                # debug_mode = gr.Checkbox(label="启用调试模式", value=False)
                # debug_times = gr.Dropdown(
                #     label="调试模式执行次数",
                #     choices=[1, 2, 3, 5, 10],
                #     value=2,
                #     visible=False
                # )
                
                submit_btn = gr.Button("生成 Pipeline", variant="primary")

            # 右侧：输出区（3 个页签）
            with gr.Column():
                with gr.Tab("Pipeline Code"):
                    output_code = gr.Code(label="生成的 Python 代码", language="python")
                with gr.Tab("Execution Log"):
                    output_log = gr.Textbox(label="执行日志", lines=10)
                with gr.Tab("Agent Results"):
                    agent_results_json = gr.JSON(label="Agent Results")

        # ---------------------- 事件绑定：调试模式显示下拉（暂时禁用） ----------------------
        # def toggle_debug_times(is_debug):
        #     return gr.update(visible=is_debug)

        # debug_mode.change(
        #     toggle_debug_times,
        #     inputs=debug_mode,
        #     outputs=debug_times
        # )

        # ----------------------  后端回调  ----------------------
        async def generate_pipeline(
            target_text, 
            json_path, 
            session_id_val, 
            chat_api_url_val, 
            api_key_val, 
            model_name_val,
            chat_api_url_for_embeddings_val,
            embedding_model_name_val,
            update_rag_val
        ):
            result = await run_pipeline_workflow(
                target=target_text,
                json_file=json_path,
                need_debug=False,
                session_id=session_id_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val,
                model_name=model_name_val,
                max_debug_rounds=2,
                chat_api_url_for_embeddings=chat_api_url_for_embeddings_val,
                embedding_model_name=embedding_model_name_val,
                update_rag_content=update_rag_val
            )

            # 读取生成的 Python 文件
            with open(result["python_file"], "r") as f:
                code = f.read()

            log = result["execution_result"]
            agent_results = result.get("agent_results", {})        
            return code, log, agent_results                        

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
                # debug_mode, 
                # debug_times
            ],
            outputs=[output_code, output_log, agent_results_json]   
        )

    return page
