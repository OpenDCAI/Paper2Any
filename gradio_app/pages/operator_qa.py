"""
OperatorQA Gradio Page - 算子问答 Chat UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-12-01

本页面实现了基于 Gradio 的算子问答 Chat UI，支持：
1. 多轮对话（复用 graph 和 state，实现真正的 interactive 模式）
2. 显示相关算子
3. 显示代码片段
4. 对话历史管理
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from dataflow_agent.state import DFRequest, MainState
from dataflow_agent.workflow.wf_operator_qa import create_operator_qa_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


# ==============================================================================
# 会话状态管理
# ==============================================================================
def create_session_state(model: str, api_url: str, api_key: str) -> Dict[str, Any]:
    """
    创建会话状态，包含复用的 graph 和 state
    
    Args:
        model: 模型名称
        api_url: API URL
        api_key: API Key
        
    Returns:
        包含 graph 和 state 的会话状态字典
    """
    actual_api_key = api_key or os.getenv("DF_API_KEY", "")
    
    # 只创建一次 graph（复用 workflow 工厂函数内的共享变量）
    log.info("初始化 workflow graph...")
    graph_builder = create_operator_qa_graph()
    graph = graph_builder.build()
    
    # 创建一次 state，后续复用（messages 会累积）
    req = DFRequest(
        language="zh",
        chat_api_url=api_url,
        api_key=actual_api_key,
        model=model,
        target="",  # 每次查询时更新
    )
    state = MainState(request=req, messages=[])
    
    return {
        "graph": graph,
        "state": state,
        "initialized": True,
    }


# ==============================================================================
# 格式化回复（与命令行格式统一）
# ==============================================================================
def format_bot_response(results: Dict[str, Any]) -> str:
    """
    格式化机器人回复，与命令行格式统一
    
    格式:
        回答内容
        
        📌 信息来源: xxx
        📦 相关算子: xxx
        📄 代码片段: (如果有)
        💡 你可能还想问: (后续建议)
    
    Args:
        results: Agent 返回的结果字典
        
    Returns:
        格式化后的回复字符串
    """
    parts = []
    
    # 主回答
    answer = results.get("answer", "抱歉，无法生成回答")
    parts.append(answer)
    
    # 信息来源
    source = results.get("source_explanation", "")
    if source:
        parts.append(f"\n\n📌 **信息来源:** {source}")
    
    # 相关算子
    related_ops = results.get("related_operators", [])
    if related_ops:
        parts.append(f"\n\n📦 **相关算子:** {', '.join(related_ops)}")
    
    # 代码片段
    code_snippet = results.get("code_snippet", "")
    if code_snippet:
        # 限制代码长度，避免过长
        code_preview = code_snippet[:1000] + "..." if len(code_snippet) > 1000 else code_snippet
        parts.append(f"\n\n📄 **代码片段:**\n```python\n{code_preview}\n```")
    
    # 后续建议
    suggestions = results.get("follow_up_suggestions", [])
    if suggestions:
        suggestion_list = "\n".join([f"   - {s}" for s in suggestions[:3]])
        parts.append(f"\n\n💡 **你可能还想问:**\n{suggestion_list}")
    
    return "".join(parts)


# ==============================================================================
# 核心执行函数
# ==============================================================================
async def execute_operator_qa(
    query: str,
    chat_history: List[Tuple[str, str]],
    session_state: Optional[Dict[str, Any]],
    model: str,
    api_url: str,
    api_key: str,
) -> Tuple[List[Tuple[str, str]], str, str, str, Dict[str, Any]]:
    """
    执行算子问答（多轮对话模式）
    
    通过复用同一个 graph 和 state，实现真正的多轮对话。
    state.messages 会在多轮对话中累积，LLM 能看到完整的对话历史。
    
    Args:
        query: 用户查询
        chat_history: Gradio 格式的对话历史 [(user, bot), ...]
        session_state: 会话状态（包含复用的 graph 和 state）
        model: 模型名称
        api_url: API URL
        api_key: API Key
        
    Returns:
        Tuple of (更新后的对话历史, 相关算子, 代码片段, 状态信息, 更新后的会话状态)
    """
    if not query.strip():
        return chat_history, "", "", "⚠️ 请输入查询内容", session_state
    
    # 初始化或复用会话状态
    if session_state is None or not session_state.get("initialized"):
        session_state = create_session_state(model, api_url, api_key)
        log.info("创建新的会话状态")
    else:
        # 更新 API 配置（用户可能修改了配置）
        state = session_state["state"]
        actual_api_key = api_key or os.getenv("DF_API_KEY", "")
        state.request.chat_api_url = api_url
        state.request.api_key = actual_api_key
        state.request.model = model
    
    graph = session_state["graph"]
    state = session_state["state"]
    
    # 更新查询目标
    state.request.target = query
    
    try:
        log.info(f"执行查询（当前消息历史: {len(state.messages)} 条）: {query}")
        
        # 执行查询（复用同一个 state，messages 会累积）
        final_state_dict = await graph.ainvoke(state)
        
        # 更新 state 的 messages（用于下一轮对话）
        if "messages" in final_state_dict:
            state.messages = final_state_dict["messages"]
            log.debug(f"更新消息历史，现有 {len(state.messages)} 条消息")
        
        # 更新 agent_results
        if "agent_results" in final_state_dict:
            state.agent_results = final_state_dict["agent_results"]
        
        # 提取结果
        agent_result = final_state_dict.get("agent_results", {}).get("operator_qa", {})
        results = agent_result.get("results", {})
        
        # 格式化回复（与命令行格式统一）
        formatted_response = format_bot_response(results)
        
        # 提取用于右侧面板显示的信息
        related_ops = results.get("related_operators", [])
        code_snippet = results.get("code_snippet", "")
        
        # 更新对话历史（用于 Gradio 显示）
        new_history = chat_history + [(query, formatted_response)]
        
        # 格式化相关算子（右侧面板）
        ops_display = ", ".join(related_ops) if related_ops else "无"
        
        return new_history, ops_display, code_snippet, "✅ 查询完成", session_state
        
    except Exception as e:
        log.error(f"执行失败: {e}")
        error_msg = f"❌ 查询失败: {str(e)}"
        new_history = chat_history + [(query, error_msg)]
        return new_history, "", "", error_msg, session_state


# ==============================================================================
# Gradio 页面定义
# ==============================================================================
def create_operator_qa() -> gr.Blocks:
    """
    创建算子问答 Gradio 页面

    Returns:
        gr.Blocks: Gradio 页面对象
    """
    with gr.Blocks(title="DataFlow 算子问答助手") as page:
        gr.Markdown("""
# 🤖 DataFlow 算子问答助手

通过自然语言查询 DataFlow 算子信息，支持：
- 💬 多轮对话
- 🔍 根据需求推荐合适的算子
- 📖 解释算子功能和使用方法
- ⚙️ 说明算子参数含义
- 📄 查看算子源码
        """)
        
        # 会话状态（用于多轮对话，存储复用的 graph 和 state）
        session_state = gr.State(value=None)
        
        with gr.Row():
            # 左侧：对话区域
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="输入你的问题",
                        placeholder="例如：我想过滤掉缺失值用哪个算子？",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清除对话", variant="secondary")
                    history_info = gr.Textbox(
                        label="对话信息",
                        value="对话轮数: 0 轮",
                        interactive=False,
                        scale=1,
                    )
                    status_text = gr.Textbox(
                        label="状态",
                        value="准备就绪",
                        interactive=False,
                        scale=2,
                    )
            
            # 右侧：配置和结果展示
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 配置")
                
                api_url = gr.Textbox(
                    label="Chat API URL",
                    value="http://123.129.219.111:3000/v1/",
                )
                
                api_key = gr.Textbox(
                    label="API Key",
                    placeholder="留空则使用环境变量 DF_API_KEY",
                    type="password",
                )
                
                model_name = gr.Dropdown(
                    label="模型",
                    choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "deepseek-v3", "qwen-max"],
                    value="gpt-4o",
                )
                
                gr.Markdown("### 📦 相关算子")
                related_ops_display = gr.Textbox(
                    label="",
                    value="无",
                    interactive=False,
                    lines=2,
                )
                
                gr.Markdown("### 📄 代码片段")
                code_display = gr.Code(
                    label="",
                    language="python",
                    lines=15,
                )
        
        # 快捷问题按钮
        gr.Markdown("### 💡 快捷问题")
        with gr.Row():
            quick_q1 = gr.Button("过滤缺失值用什么算子？", size="sm")
            quick_q2 = gr.Button("PromptedFilter 怎么用？", size="sm")
            quick_q3 = gr.Button("数据去重用什么算子？", size="sm")
            quick_q4 = gr.Button("如何生成问答对？", size="sm")
        
        # ==================================================================
        # 事件绑定
        # ==================================================================
        
        async def submit_query(query, history, sess_state, model, url, key):
            """提交查询（多轮对话模式）"""
            new_history, ops, code, status, new_sess_state = await execute_operator_qa(
                query, history, sess_state, model, url, key
            )
            # 更新对话轮数（按前端显示的问答轮数统计）
            round_count = len(new_history)
            history_info = f"对话轮数: {round_count} 轮"
            return new_history, ops, code, status, new_sess_state, history_info
        
        def clear_chat(sess_state):
            """清除对话（重置 state.messages 但保留 graph）"""
            if sess_state and sess_state.get("state"):
                sess_state["state"].messages = []
                sess_state["state"].agent_results = {}
                log.info("对话历史已清除（保留 graph）")
            return [], "", "", "✅ 对话已清除", sess_state, "对话轮数: 0 轮"
        
        # 提交按钮
        submit_btn.click(
            fn=submit_query,
            inputs=[query_input, chatbot, session_state, model_name, api_url, api_key],
            outputs=[chatbot, related_ops_display, code_display, status_text, session_state, history_info],
        ).then(
            fn=lambda: "",
            outputs=[query_input],
        )
        
        # 回车提交
        query_input.submit(
            fn=submit_query,
            inputs=[query_input, chatbot, session_state, model_name, api_url, api_key],
            outputs=[chatbot, related_ops_display, code_display, status_text, session_state, history_info],
        ).then(
            fn=lambda: "",
            outputs=[query_input],
        )
        
        # 清除按钮
        clear_btn.click(
            fn=clear_chat,
            inputs=[session_state],
            outputs=[chatbot, related_ops_display, code_display, status_text, session_state, history_info],
        )
        
        # 快捷问题按钮
        quick_q1.click(fn=lambda: "过滤缺失值用什么算子？", outputs=[query_input])
        quick_q2.click(fn=lambda: "PromptedFilter 怎么用？详细解释一下参数", outputs=[query_input])
        quick_q3.click(fn=lambda: "数据去重用什么算子？", outputs=[query_input])
        quick_q4.click(fn=lambda: "如何生成问答对数据？推荐合适的算子", outputs=[query_input])
    
    return page


# ==============================================================================
# 独立运行（用于测试）
# ==============================================================================
if __name__ == "__main__":
    demo = create_operator_qa()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
