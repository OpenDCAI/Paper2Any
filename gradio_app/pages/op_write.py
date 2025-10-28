# import gradio as gr
# from ..utils.wf_operator_write import run_operator_write_workflow

# def create_operator_write():
#     """子页面：Operator Write"""
#     with gr.Blocks() as page:
#         gr.Markdown("# 📝 DataFlow Operator Write")

#         with gr.Row():
#             # 左侧：输入区
#             with gr.Column():
#                 operator_name = gr.Textbox(
#                     label="Operator 名称",
#                     placeholder="如：FilterDuplicates"
#                 )
#                 operator_desc = gr.Textbox(
#                     label="Operator 描述",
#                     placeholder="描述该 Operator 的功能和用途",
#                     lines=3
#                 )
#                 input_schema = gr.Textbox(
#                     label="输入数据格式（JSON Schema）",
#                     placeholder='如：{"type": "object", "properties": {"text": {"type": "string"}}}'
#                 )
#                 output_schema = gr.Textbox(
#                     label="输出数据格式（JSON Schema）",
#                     placeholder='如：{"type": "object", "properties": {"text": {"type": "string"}}}'
#                 )
#                 session_id = gr.Textbox(
#                     label="Session ID",
#                     value="default"
#                 )
#                 chat_api_url = gr.Textbox(
#                     label="Chat API URL",
#                     value="http://123.129.219.111:3000/v1/"
#                 )
#                 api_key = gr.Textbox(
#                     label="API Key",
#                     value="",  # 或者默认从环境变量读取
#                     type="password"
#                 )
#                 debug_mode = gr.Checkbox(label="启用调试模式", value=False)
#                 submit_btn = gr.Button("生成 Operator", variant="primary")

#             # 右侧：输出区
#             with gr.Column():
#                 with gr.Tab("Operator 代码"):
#                     output_code = gr.Code(label="生成的 Python 代码", language="python")
#                 with gr.Tab("日志"):
#                     output_log = gr.Textbox(label="执行日志", lines=10)
#                 with gr.Tab("Agent 结果"):
#                     agent_results_json = gr.JSON(label="Agent Results")

#         # ----------------------  后端回调  ----------------------
#         async def generate_operator(
#             operator_name_val, operator_desc_val, input_schema_val, output_schema_val,
#             session_id_val, chat_api_url_val, api_key_val, debug
#         ):
#             result = await run_operator_write_workflow(
#                 operator_name=operator_name_val,
#                 operator_desc=operator_desc_val,
#                 input_schema=input_schema_val,
#                 output_schema=output_schema_val,
#                 session_id=session_id_val,
#                 chat_api_url=chat_api_url_val,
#                 api_key=api_key_val,
#                 need_debug=debug
#             )

#             # 读取生成的 Python 文件
#             with open(result["python_file"], "r") as f:
#                 code = f.read()

#             log = result["execution_result"].get("stdout", "执行完成")
#             agent_results = result.get("agent_results", {})        
#             return code, log, agent_results                        

#         submit_btn.click(
#             generate_operator,
#             inputs=[
#                 operator_name, operator_desc, input_schema, output_schema,
#                 session_id, chat_api_url, api_key, debug_mode
#             ],
#             outputs=[output_code, output_log, agent_results_json]   
#         )

#     return page