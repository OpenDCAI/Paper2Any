import gradio as gr
from ..utils.wf_pipeine_rec import run_pipeline_workflow

def create_pipeline_rec():
    """子页面：Pipeline 生成（带 Agent 结果展示）"""
    with gr.Blocks() as page:
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
                    value="/mnt/DataFlow/lz/proj/DataFlow-Agent/tests/test.jsonl"
                )
                session_id = gr.Textbox(
                    label="Session ID",
                    value="default"
                )
                chat_api_url = gr.Textbox(
                    label="Chat API URL",
                    value="http://123.129.219.111:3000/v1/"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value="",  # 或者默认从环境变量读取
                    type="password"
                )
                debug_mode = gr.Checkbox(label="启用调试模式", value=False)
                submit_btn = gr.Button("生成 Pipeline", variant="primary")

            # 右侧：输出区（3 个页签）
            with gr.Column():
                with gr.Tab("Pipeline Code"):
                    output_code = gr.Code(label="生成的 Python 代码", language="python")
                with gr.Tab("Execution Log"):
                    output_log = gr.Textbox(label="执行日志", lines=10)
                with gr.Tab("Agent Results"):
                    agent_results_json = gr.JSON(label="Agent Results")

        # ----------------------  后端回调  ----------------------
        async def generate_pipeline(target_text, json_path, session_id_val, chat_api_url_val, api_key_val, debug):
            result = await run_pipeline_workflow(
                target=target_text,
                json_file=json_path,
                need_debug=debug,
                session_id=session_id_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val
            )

            # 读取生成的 Python 文件
            with open(result["python_file"], "r") as f:
                code = f.read()

            log = result["execution_result"]
            agent_results = result.get("agent_results", {})        
            return code, log, agent_results                        

        submit_btn.click(
            generate_pipeline,
            inputs=[target, json_file, session_id, chat_api_url, api_key, debug_mode],
            outputs=[output_code, output_log, agent_results_json]   
        )

    return page