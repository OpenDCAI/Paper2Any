import asyncio
from functools import cache
import os
from typing import List, Tuple

import gradio as gr

from dataflow_agent.state import DFRequest, PromptWritingState
from dataflow_agent.workflow.wf_pipeline_prompt import (
    create_operator_prompt_writing_graph,
)
from dataflow_agent.agentroles.prompt_writer import create_prompt_writer


def _parse_arguments_str(arguments_text: str) -> List[str]:
    if not arguments_text:
        return []
    separators = [",", "\n", " "]
    tokens = [arguments_text]
    for sep in separators:
        new_tokens: List[str] = []
        for t in tokens:
            new_tokens.extend(t.split(sep))
        tokens = new_tokens
    return [t.strip() for t in tokens if t and t.strip()]


async def _run_initial_generation(
    chat_api_url: str,
    api_key: str,
    model: str,
    language: str,
    task_description: str,
    op_name: str,
    output_format: str,
    arguments_text: str,
    file_path: str,
    delete_test_files: bool,
):
    if api_key:
        os.environ["DF_API_KEY"] = api_key
    arguments = _parse_arguments_str(arguments_text)

    req = DFRequest(
        language=language.strip(),
        chat_api_url=chat_api_url.strip(),
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=model.strip(),
        target=task_description,
        cache_dir=file_path or "./pa_cache",
    )
    state = PromptWritingState(request=req, messages=[])
    state.prompt_op_name = op_name.strip()
    state.prompt_args = arguments
    state.prompt_output_format = output_format.strip()
    state.temp_data["pipeline_file_path"] = file_path.strip() if file_path else None
    state.temp_data["delete_test_files"] = delete_test_files

    graph = create_operator_prompt_writing_graph().build()
    invoke_result = await graph.ainvoke(state)
    # 统一转换为 State，便于后续多轮改写使用
    if isinstance(invoke_result, PromptWritingState):
        final_state: PromptWritingState = invoke_result
    else:
        # 保留最初构建的 State（含 prompt_op_name 等），合并图返回的消息与临时数据
        final_state = state
        try:
            msgs = invoke_result.get("messages")
            if msgs is not None:
                final_state.messages = msgs
        except Exception:
            pass
        try:
            td = invoke_result.get("temp_data", {})
            if td:
                final_state.temp_data.update(td)
        except Exception:
            pass

    prompt_code = final_state.temp_data.get("prompt_code", "")
    test_code = final_state.temp_data.get("prompt_test_code", "")
    prompt_file_path = final_state.temp_data.get("prompt_file_path", "")
    test_file_path = final_state.temp_data.get("prompt_test_file_path", "")
    test_data_file = final_state.temp_data.get("test_data_file_path", "")
    # 读取测试数据内容（若存在）
    test_data_preview = ""
    try:
        if test_data_file and os.path.exists(test_data_file):
            with open(test_data_file, "r", encoding="utf-8") as f:
                test_data_preview = f.read()
    except Exception:
        test_data_preview = ""
    # 读取测试结果文件预览（由后端写入）
    result_data_preview = final_state.temp_data.get("prompt_result_preview", "")

    return (
        final_state,
        prompt_code,
        test_code,
        prompt_file_path,
        test_file_path,
        test_data_file,
        test_data_preview,
        result_data_preview,
    )


async def _on_chat_submit(
    user_message: str,
    chat_history: List[Tuple[str, str]],
    state: PromptWritingState,
):
    if state is None or not hasattr(state, "temp_data"):
        chat_history = chat_history + [(user_message, "请先在左侧完成一次初次生成。")]
        return (
            chat_history,
            "",
            "",
            "",
            "",
            "",
            "",
            state,
        )

    agent = create_prompt_writer()
    final_state = await agent.revise_with_feedback(state, user_message)

    prompt_code = final_state.temp_data.get("prompt_code", "")
    test_code = final_state.temp_data.get("prompt_test_code", "")
    prompt_file_path = final_state.temp_data.get("prompt_file_path", "")
    test_file_path = final_state.temp_data.get("prompt_test_file_path", "")
    test_data_file = final_state.temp_data.get("test_data_file_path", "")
    # 读取测试数据内容（若存在）
    test_data_preview = ""
    try:
        if test_data_file and os.path.exists(test_data_file):
            with open(test_data_file, "r", encoding="utf-8") as f:
                test_data_preview = f.read()
    except Exception:
        test_data_preview = ""
    # 读取测试结果文件预览
    result_data_preview = final_state.temp_data.get("prompt_result_preview", "")

    assistant_reply = "已根据反馈更新模板并完成测试。"
    chat_history = chat_history + [(user_message, assistant_reply)]
    return (
        chat_history,
        prompt_code,
        test_code,
        prompt_file_path,
        test_file_path,
        test_data_file,
        test_data_preview,
        result_data_preview,
        final_state,
    )


def create_PA_frontend():
    css = """
#left-pane { max-height: 85vh; overflow: auto; }
#right-pane { max-height: 85vh; overflow: auto; }
#chat-box { max-height: 85vh; overflow: auto; }
"""
    with gr.Blocks(title="DataFlow PromptAgent 前端", css=css, elem_id="page-root-pa") as demo:
        state_holder = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["left-pane-pa"]):
                with gr.Accordion("运行配置", open=True):
                    chat_api_url = gr.Textbox(
                        label="Chat API Base URL",
                        value="http://123.129.219.111:3000/v1/",
                    )
                    api_key = gr.Textbox(label="Chat API Key", placeholder="sk-...", type="password")
                    model = gr.Textbox(label="Model", value="gpt-4o")
                    language = gr.Textbox(label="Language", value="zh")
                    task_description = gr.Textbox(label="任务描述", lines=4)
                    op_name = gr.Textbox(label="算子名称 (op-name)")
                    output_format = gr.Textbox(label="输出格式 (可选)", lines=3)
                    arguments_text = gr.Textbox(
                        label="参数列表 (用逗号/空格/换行分隔)",
                        placeholder="arg1, arg2 或换行分隔",
                        lines=2,
                    )
                    file_path = gr.Textbox(
                        label="文件输出根路径 (可选)",
                        placeholder="默认使用 ./pa_cache",
                    )  
                    delete_test_files = gr.Checkbox(
                        label="生成后删除测试文件(保留路径占位)", value=True
                    )
                    run_btn = gr.Button("生成 Prompt 模板", variant="primary")

                with gr.Accordion("输出与预览", open=True):
                    prompt_file_out = gr.Textbox(label="Prompt 文件路径", interactive=False)
                    test_data_file_out = gr.Textbox(
                        label="测试数据文件路径", interactive=False
                    )
                    test_file_out = gr.Textbox(label="测试代码文件路径", interactive=False)
                    test_data_preview = gr.Code(label="测试数据预览", language="json")
                    result_data_preview = gr.Code(label="测试结果预览", language="json")
                    prompt_code_out = gr.Code(label="Prompt 代码预览", language="python")
                    test_code_out = gr.Code(label="测试代码预览", language="python")

            with gr.Column(scale=7, elem_classes=["right-pane-pa"]):
                with gr.Group(elem_classes=["chat-box-pa"]):
                    chatbot = gr.Chatbot(label="多轮改写对话", height=520)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="对话输入",
                            placeholder="请描述你希望如何修改提示词...",
                            lines=3,
                        )
                    with gr.Row():
                        send_btn = gr.Button("发送改写指令", variant="primary", scale=8)
                        clear_btn = gr.Button("清空会话", scale=4)

        run_btn.click(
            _run_initial_generation,
            inputs=[
                chat_api_url,
                api_key,
                model,
                language,
                task_description,
                op_name,
                output_format,
                arguments_text,
                file_path,
                delete_test_files,
            ],
            outputs=[
                state_holder,
                prompt_code_out,
                test_code_out,
                prompt_file_out,
                test_file_out,
                test_data_file_out,
                test_data_preview,
                result_data_preview,
            ],
        )

        send_btn.click(
            _on_chat_submit,
            inputs=[chat_input, chatbot, state_holder],
            outputs=[
                chatbot,
                prompt_code_out,
                test_code_out,
                prompt_file_out,
                test_file_out,
                test_data_file_out,
                test_data_preview,
                result_data_preview,
                state_holder,
            ],
        )

        clear_btn.click(lambda: [], None, chatbot)

    return demo


if __name__ == "__main__":
    app = create_PA_frontend()
    app.launch(server_port=7890)


