import gradio as gr
from ..utils.wf_pipeine_rec import run_pipeline_workflow

def create_pipeline_page():
    """子页面：Pipeline 生成"""
    
    with gr.Blocks() as page:
        gr.Markdown("# 🚀 DataFlow Pipeline Generator")
        
        with gr.Row():
            with gr.Column():
                target = gr.Textbox(
                    label="目标描述",
                    placeholder="给我随意符合逻辑的5个算子，过滤，去重！",
                    lines=3
                )
                json_file = gr.Textbox(
                    label="输入 JSONL 文件路径",
                    value="dataflow/example/GeneralTextPipeline/translation.jsonl"
                )
                debug_mode = gr.Checkbox(label="启用调试模式", value=False)
                submit_btn = gr.Button("生成 Pipeline", variant="primary")
            
            with gr.Column():
                output_code = gr.Code(label="生成的 Python 代码", language="python")
                output_log = gr.Textbox(label="执行日志", lines=10)
        
        async def generate_pipeline(target_text, json_path, debug):
            result = await run_pipeline_workflow(
                target=target_text,
                json_file=json_path,
                need_debug=debug
            )
            
            # 读取生成的 Python 文件
            with open(result["python_file"], "r") as f:
                code = f.read()
            
            log = result["execution_result"].get("stdout", "执行完成")
            return code, log
        
        submit_btn.click(
            generate_pipeline,
            inputs=[target, json_file, debug_mode],
            outputs=[output_code, output_log]
        )
    
    return page