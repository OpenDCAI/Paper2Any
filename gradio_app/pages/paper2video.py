"""
Auto-generated on 2025-11-30 19:33:25
本文件由自动化模板生成。你可以在此基础上自定义 Gradio UI 组件与数据流执行函数。
"""
from dataflow_agent.logger import get_logger
from pathlib import Path
import shutil
import os

log = get_logger(__name__)

import gradio as gr

# ------------------- Gradio 页面组件定义 -------------------
def create_paper2video() -> gr.Blocks:
    """
    创建 paper2video 页面，现在使用 gr.File 支持文件上传。

    Returns:
        gr.Blocks: Gradio 多组件页面对象。
    """
    with gr.Blocks(title="Paper2Video — 自动化论文讲解视频生成") as page:
        gr.Markdown("## 🎬 论文转视频生成器 — Paper2Video")

        # API配置区域
        with gr.Accordion("🛠️ API配置", open=True):
            with gr.Row():
                model_input = gr.Textbox(
                    label="🤖 模型名称 (Model Name)",
                    placeholder="例如：gpt-4o",
                    value="gpt-4o"
                )
                chat_api_url_input = gr.Textbox(
                    label="🌐 API地址 (API Endpoint)",
                    placeholder="例如：http://123.129.219.111:3000/v1",
                    value="http://123.129.219.111:3000/v1"
                )
            api_key_input = gr.Textbox(
                label="🔑 API密钥 (API Key)",
                placeholder="输入API密钥",
                value="",
                type="password"
            )

        with gr.Row(visible=True) as file_upload_row:
            # 使用 gr.File 支持 PDF 文件上传，并限制文件类型
            pdf_file_input = gr.File(
                label="📄 上传论文文件", 
                file_types=[".pdf"],
                type="filepath", # 返回文件在服务器上的临时路径
                height=150
            )
            
            # 使用 gr.File 支持图片文件上传，并限制文件类型
            style_image_input = gr.File(
                label="🖼️ 上传自定义图片 (可选)", 
                file_types=[".jpg", ".jpeg", ".png"],
                type="filepath",
                height=150
            )

        gen_btn = gr.Button("🚀 启动视频生成")
        
        # 将输出更改为 File 组件，更贴合输出 PDF/视频文件的语义
        output_file = gr.File(label="📥 生成的演示文稿 (PDF)", type="filepath")

        async def ppt_generate(model, chat_api_url, api_key, pdf_path, image_path):
            """
            执行论文转视频/PPT的核心工作流。
            
            Args:
                pdf_path (str | None): 上传的 PDF 文件的临时路径。
                image_path (str | None): 上传的风格图片的临时路径。
            """
            # 将临时路径中的文件 转移到 当前项目的一个目录中
            if not pdf_path:
                log.error("未上传论文文件。")
                # 返回 None 或抛出异常，Gradio 会显示错误
                raise gr.Error("请先上传一篇 PDF 格式的论文文件。")

            log.info(f"接收到论文文件路径: {pdf_path}")
            log.info(f"接收到自定义图片路径: {image_path}")
            TARGET_DIR = Path("/mnt/DataFlow/lz/proj/agentgroup/ligang/DataFlow-Agent/data")
            TARGET_DIR.mkdir(parents=True, exist_ok=True) # 确保目录存在
            if pdf_path:
                src_pdf_path = Path(pdf_path)
                target_pdf_path = TARGET_DIR / src_pdf_path.name
                shutil.copy2(src_pdf_path, target_pdf_path)
                log.info(f"PDF 文件已经保存到：{target_pdf_path}")

            if image_path:
                src_img_path = Path(image_path)
                target_img_path = TARGET_DIR / src_img_path.name
                shutil.copy2(src_img_path, target_img_path)
                log.info(f"图片已经保存到：{target_img_path}")

            try:
                result = await run_paper2video_pipeline(
                    model,
                    chat_api_url,
                    api_key,
                    str(target_pdf_path),
                    str(target_img_path) if image_path else None
                )
                # 提取结果
                ppt_path = result.get("ppt_path", "")
                
                # 构建日志信息
                if ppt_path and Path(ppt_path).exists():
                    log.info(f"生成的 PPT 文件路径: {ppt_path}")
                    return str(ppt_path)
                else:
                    log.error("未能生成 PPT 文件。")
                    return ""           
            except Exception as e:
                import traceback
                error_msg = f"执行失败:\n{traceback.format_exc()}"
                print(f"错误: {error_msg}")
                return ""


        gen_btn.click(
            ppt_generate, 
            [   model_input, 
                chat_api_url_input, 
                api_key_input, 
                pdf_file_input, 
                style_image_input
            ], 
            [output_file]
        )
        
    return page

# ------------------- 数据流工作流执行函数模板 -------------------
async def run_paper2video_pipeline(
    model: str = "gpt-4o",
    chat_api_url: str = "http://123.129.219.111:3000/v1/", 
    api_key: str = "", 
    pdf_path: str = "", 
    img_path: str = "",
) -> dict :
    """
    执行 DataFlow Paper to Video 工作流。

    参数说明:
        chat_api_url (str): Chat API 的访问地址。
        apikey (str): OpenAI 或自定义大模型接口的 API Key。
        model (str, 可选): 使用的模型名称，默认为 'gpt-4o'。
        pdf_path (str): 输入数据文件路径（pdf 格式）。
        img_path (str, optional): 输入图片文件格式
    返回值:
        Paper2VideoState: 工作流的最终状态对象，包含产出数据与日志信息。
    """
    
    from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
    from dataflow_agent.logger import get_logger
    from dataflow_agent.utils import get_project_root
    
    log = get_logger(__name__)
    # 设置环境变量
    if api_key:
        os.environ["DF_API_KEY"] = api_key
    else:
        api_key = os.getenv("DF_API_KEY", "sk-dummy")

    # 创建请求对象
    req = Paper2VideoRequest(
        chat_api_url=chat_api_url,
        api_key=api_key,
        model=model,
        paper_pdf_path=pdf_path,
        user_imgs_path=img_path,
    )

    # 创建状态对象
    state = Paper2VideoState(request=req, messages=[])

    # 延迟导入以避免工作流初始化时的依赖问题
    from dataflow_agent.workflow.wf_paper2video import create_paper2video_graph
    
    graph = create_paper2video_graph().build()
    final_state: Paper2VideoState = await graph.ainvoke(state)

    # 提取结果
    result = {
        "success": True,
        "final_state": final_state,
    }
    
    # 提取输出的pdf文件
    try:
        if isinstance(final_state, dict):
            ppt_path = final_state.get("ppt_path", [])
        else:
            ppt_path = getattr(final_state, "ppt_path", [])
            
        result["ppt_path"] = ppt_path or []
    except Exception as e:
        if 'log' in locals():
            log.warning(f"提取pdf的ppt失败: {e}")
        result["ppt_path"] = []

    return result