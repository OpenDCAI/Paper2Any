"""
算子编写页面 - 用于生成新的 DataFlow 算子
"""

import os
from pathlib import Path
import gradio as gr

# 延迟导入以避免模块初始化时的依赖问题
PROJDIR = None

# ------------------- 数据流工作流执行函数 -------------------
async def run_operator_write_pipeline(
    target: str,
    category: str = "Default",
    json_file: str = "",
    chat_api_url: str = "http://123.129.219.111:3000/v1/",
    api_key: str = "",
    model: str = "gpt-4o",
    language: str = "en",
    need_debug: bool = False,
    max_debug_rounds: int = 3,
    output_path: str = "",
) -> dict:
    # 延迟导入以避免模块初始化时的依赖问题
    global PROJDIR
    from dataflow_agent.state import DFRequest, DFState
    from dataflow_agent.logger import get_logger
    from dataflow_agent.utils import get_project_root
    
    if PROJDIR is None:
        PROJDIR = get_project_root()
    
    log = get_logger(__name__)
    """
    执行算子编写工作流。

    参数说明:
        target (str): 用户需求/新算子的目的（必需）。
        category (str): 算子类别，默认为 'Default'。
        json_file (str): 测试数据文件路径，默认为空（使用默认测试文件）。
        chat_api_url (str): Chat API 的访问地址。
        api_key (str): API Key。
        model (str): 使用的模型名称，默认为 'gpt-4o'。
        language (str): 提示输出语言，默认为 'en'。
        need_debug (bool): 是否启用调试循环，默认为 False。
        max_debug_rounds (int): 最大调试轮次，默认为 3。
        output_path (str): 可选的文件路径，用于保存生成的算子代码。
        
    返回值:
        dict: 包含执行结果的字典。
    """
    # 设置环境变量
    if api_key:
        os.environ["DF_API_KEY"] = api_key
    else:
        api_key = os.getenv("DF_API_KEY", "sk-dummy")

    # 使用默认测试文件路径
    if not json_file:
        json_file = f"{PROJDIR}/tests/test.jsonl"

    # 创建请求对象
    req = DFRequest(
        language=language,
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=model,
        target=target,
        need_debug=need_debug,
        max_debug_rounds=max_debug_rounds,
        json_file=json_file,
    )

    # 创建状态对象
    state = DFState(request=req, messages=[])
    
    # 设置输出路径（如果提供）
    if output_path:
        state.temp_data["pipeline_file_path"] = output_path
    
    # 设置类别
    if category:
        state.temp_data["category"] = category

    # 初始化调试轮次
    state.temp_data["round"] = 0

    # 延迟导入以避免工作流初始化时的依赖问题
    from dataflow_agent.workflow.wf_pipeline_write import create_operator_write_graph
    
    # 构建并执行工作流图
    graph = create_operator_write_graph().build()
    # 计算递归限制：主链 4 步 + 每轮 5 步 * 轮次 + buffer 5
    recursion_limit = 4 + 5 * max_debug_rounds + 5
    final_state: DFState = await graph.ainvoke(
        state, 
        config={"recursion_limit": recursion_limit}
    )

    # 提取结果
    result = {
        "success": True,
        "final_state": final_state,
    }
    
    # 提取匹配的算子
    try:
        if isinstance(final_state, dict):
            matched = final_state.get("matched_ops", [])
            if not matched:
                matched = (
                    final_state.get("agent_results", {})
                    .get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        else:
            matched = getattr(final_state, "matched_ops", [])
            if not matched and hasattr(final_state, "agent_results"):
                matched = (
                    final_state.agent_results.get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        result["matched_ops"] = matched or []
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取匹配算子失败: {e}")
        result["matched_ops"] = []

    # 提取生成的代码
    try:
        if isinstance(final_state, dict):
            temp_data = final_state.get("temp_data", {})
            code_str = temp_data.get("pipeline_code", "") if isinstance(temp_data, dict) else ""
        else:
            temp_data = getattr(final_state, "temp_data", {})
            code_str = temp_data.get("pipeline_code", "") if isinstance(temp_data, dict) else ""
        result["code"] = code_str or ""
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取代码失败: {e}")
        result["code"] = ""

    # 提取执行结果
    try:
        if isinstance(final_state, dict):
            exec_res = final_state.get("execution_result", {}) or {}
            if not exec_res or ("success" not in exec_res):
                exec_res = final_state.get("agent_results", {}).get("operator_executor", {}).get("results", {}) or exec_res
        else:
            exec_res = getattr(final_state, "execution_result", {}) or {}
            if (not exec_res or ("success" not in exec_res)) and hasattr(final_state, "agent_results"):
                exec_res = final_state.agent_results.get("operator_executor", {}).get("results", {}) or exec_res
        result["execution_result"] = exec_res
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取执行结果失败: {e}")
        result["execution_result"] = {}

    # 提取调试运行时信息
    try:
        if isinstance(final_state, dict):
            dbg = (final_state.get("temp_data") or {}).get("debug_runtime")
        else:
            dbg = getattr(final_state, "temp_data", {}).get("debug_runtime")
        result["debug_runtime"] = dbg or {}
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取调试信息失败: {e}")
        result["debug_runtime"] = {}

    # 提取 agent_results
    try:
        if isinstance(final_state, dict):
            agent_results = final_state.get("agent_results", {})
        else:
            agent_results = getattr(final_state, "agent_results", {})
        result["agent_results"] = agent_results
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取 agent_results 失败: {e}")
        result["agent_results"] = {}

    return result

# ------------------- Gradio 页面组件定义 -------------------
def create_operator_write() -> gr.Blocks:
    """
    创建算子编写页面。

    Returns:
        gr.Blocks: Gradio 多组件页面对象。
    """
    # 延迟获取项目根目录
    try:
        from dataflow_agent.utils import get_project_root
        _projdir = get_project_root()
    except Exception:
        _projdir = ""
    
    with gr.Blocks() as page:
        gr.Markdown("# 🛠️ DataFlow 算子编写工具")
        gr.Markdown("根据您的需求自动生成新的 DataFlow 算子代码")

        with gr.Row():
            # 左侧：输入区
            with gr.Column():
                target = gr.Textbox(
                    label="目标描述 *",
                    placeholder="例如：创建一个算子，用于对文本进行情感分析",
                    lines=3,
                    info="描述您想要创建的算子的功能和用途"
                )
                
                category = gr.Textbox(
                    label="算子类别",
                    value="Default",
                    info="算子所属的类别，用于匹配相似算子作为参考"
                )
                
                json_file = gr.Textbox(
                    label="测试数据文件路径（JSONL）",
                    value=f"{_projdir}/tests/test.jsonl" if _projdir else "",
                    info="用于测试和调试的 JSONL 数据文件路径"
                )
                
                with gr.Row():
                    chat_api_url = gr.Textbox(
                        label="Chat API URL",
                        value="http://123.129.219.111:3000/v1/",
                        info="LLM API 服务地址"
                    )
                    model = gr.Textbox(
                        label="模型名称",
                        value="gpt-4o",
                        info="使用的 LLM 模型名称"
                    )
                
                api_key = gr.Textbox(
                    label="API Key",
                    value="",
                    type="password",
                    info="API Key，留空则使用环境变量 DF_API_KEY"
                )
                
                language = gr.Dropdown(
                    label="输出语言",
                    choices=["en", "zh", "zh-CN"],
                    value="en",
                    info="提示词和输出的语言"
                )
                
                with gr.Row():
                    need_debug = gr.Checkbox(
                        label="启用调试模式",
                        value=False,
                        info="启用后会自动执行并修复代码中的错误"
                    )
                    max_debug_rounds = gr.Number(
                        label="最大调试轮次",
                        value=3,
                        minimum=1,
                        maximum=10,
                        precision=0,
                        info="调试模式下的最大重试次数"
                    )
                
                output_path = gr.Textbox(
                    label="输出文件路径（可选）",
                    value="",
                    info="保存生成的算子代码的文件路径，留空则不保存"
                )
                
                submit_btn = gr.Button("生成算子", variant="primary", size="lg")

            # 右侧：输出区
            with gr.Column():
                with gr.Tab("生成的代码"):
                    output_code = gr.Code(
                        label="算子代码",
                        language="python",
                        lines=20
                    )
                
                with gr.Tab("匹配的算子"):
                    matched_ops = gr.JSON(
                        label="参考算子列表"
                    )
                    gr.Markdown("系统根据您的需求匹配到的相似算子")
                
                with gr.Tab("执行结果"):
                    execution_result = gr.JSON(
                        label="执行结果"
                    )
                    gr.Markdown("算子的执行结果和状态")
                
                with gr.Tab("调试信息"):
                    debug_info = gr.JSON(
                        label="调试运行时信息"
                    )
                    gr.Markdown("调试模式下的详细调试信息")
                
                with gr.Tab("Agent 结果"):
                    agent_results_json = gr.JSON(
                        label="Agent 执行结果"
                    )
                    gr.Markdown("各个 Agent 节点的执行结果")
                
                with gr.Tab("执行日志"):
                    output_log = gr.Textbox(
                        label="详细日志",
                        lines=15,
                        info="完整的执行日志信息"
                    )

        # ---------------------- 后端回调 ----------------------
        async def generate_operator(
            target_text,
            category_val,
            json_path,
            chat_api_url_val,
            api_key_val,
            model_val,
            language_val,
            debug,
            max_rounds,
            out_path
):
            """执行算子生成工作流"""
            if not target_text.strip():
                gr.Warning("请输入目标描述")
                return "", [], {}, {}, {}, ""
            
            try:
                # 调用工作流
                result = await run_operator_write_pipeline(
                    target=target_text,
                    category=category_val or "Default",
                    json_file=json_path or "",
                    chat_api_url=chat_api_url_val,
                    api_key=api_key_val,
                    model=model_val,
                    language=language_val,
                    need_debug=bool(debug),
                    max_debug_rounds=int(max_rounds) if max_rounds else 3,
                    output_path=out_path or "",
                )
                
                # 提取结果
                code = result.get("code", "")
                matched = result.get("matched_ops", [])
                exec_res = result.get("execution_result", {})
                debug_runtime = result.get("debug_runtime", {})
                agent_results = result.get("agent_results", {})
                
                # 构建日志信息
                log_lines = []
                log_lines.append("==== 算子编写结果 ====")
                log_lines.append(f"\n匹配到的算子数量: {len(matched)}")
                if matched:
                    log_lines.append(f"匹配的算子: {matched}")
                
                log_lines.append(f"\n生成的代码长度: {len(code)} 字符")
                
                if exec_res:
                    success = exec_res.get("success", False)
                    log_lines.append(f"\n执行成功: {success}")
                    if not success:
                        stderr = exec_res.get("stderr", "") or exec_res.get("traceback", "")
                        if stderr:
                            log_lines.append(f"\n错误信息:\n{stderr[:500]}")
                
                if debug_runtime:
                    log_lines.append("\n==== 调试信息 ====")
                    input_key = debug_runtime.get("input_key")
                    available_keys = debug_runtime.get("available_keys", [])
                    if input_key:
                        log_lines.append(f"选择的输入键: {input_key}")
                    if available_keys:
                        log_lines.append(f"可用键: {available_keys}")
                    stdout = debug_runtime.get("stdout", "")
                    stderr = debug_runtime.get("stderr", "")
                    if stdout:
                        log_lines.append(f"\n标准输出:\n{stdout[:1000]}")
                    if stderr:
                        log_lines.append(f"\n标准错误:\n{stderr[:1000]}")
                
                log_text = "\n".join(log_lines)
                
                return code, matched, exec_res, debug_runtime, agent_results, log_text
                
            except Exception as e:
                import traceback
                error_msg = f"执行失败:\n{traceback.format_exc()}"
                # 在回调函数中无法访问 log，使用 print 代替
                print(f"错误: {error_msg}")
                return "", [], {}, {}, {}, error_msg

        # 绑定按钮点击事件
        submit_btn.click(
            generate_operator,
            inputs=[
                target,
                category,
                json_file,
                chat_api_url,
                api_key,
                model,
                language,
                need_debug,
                max_debug_rounds,
                output_path,
            ],
            outputs=[
                output_code,
                matched_ops,
                execution_result,
                debug_info,
                agent_results_json,
                output_log,
            ],
        )

    return page