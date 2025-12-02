import os
import asyncio
import gradio as gr

from dataflow_agent.state import MainState, IconGenRequest, IconGenState
from dataflow_agent.workflow.wf_icongen_refine_loop import create_icongen_refine_loop_graph
from dataflow_agent.workflow.wf_icongen import create_icongen_graph
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

# 创建两个独立的graph实例
icon_graph = create_icongen_refine_loop_graph().build()
paper_graph = create_icongen_graph().build()

# 全局状态管理
global_icon_state = IconGenState(request=IconGenRequest(chat_api_url="http://123.129.219.111:3000/v1"))
global_paper_state = IconGenState(request=IconGenRequest(chat_api_url="http://123.129.219.111:3000/v1"))
last_image = None



# 添加模型路径环境变量
os.environ["RM_MODEL_PATH"] = f"{get_project_root()}/dataflow_agent/toolkits/imtool/onnx/model.onnx"

def _merge_state(state, out):
    """合并状态"""
    if isinstance(out, dict):
        req = out.get("request", {}) or {}
        if isinstance(req, IconGenRequest):
            # 如果req已经是IconGenRequest对象，则不作处理或只合并必要的属性
            pass
        elif isinstance(req, dict):
            for k, v in req.items():
                setattr(state.request, k, v)
        state._vars = {**getattr(state, "_vars", {}), **(out.get("_vars") or {})}
        state.agent_results = {**getattr(state, "agent_results", {}), **(out.get("agent_results") or {})}
        return state
    return state


def _get_img_from_icon_graph(state):
    """从图标生成graph中提取图像"""
    return (
        (getattr(state, "_vars", {}) or {}).get("final_img")
        or ((getattr(state, "agent_results", {}) or {}).get("bg_removed") or {}).get("path")
        or ((getattr(state, "agent_results", {}) or {}).get("round2_img") or {}).get("path")
        or ((getattr(state, "agent_results", {}) or {}).get("round1_img") or {}).get("path")
    )


def _get_img_from_paper_graph(state):
    """从论文模型图生成graph中提取图像"""
    return (
        ((getattr(state, "agent_results", {}) or {}).get("bg_removed") or {}).get("path")
        or ((getattr(state, "agent_results", {}) or {}).get("gen_img") or {}).get("path")
    )


async def run_icon_generation(keywords=None, style=None, edit_prompt=None, prev_img=None, 
                              model=None, chat_api_url=None, api_key=None):
    """运行图标生成流程"""
    global global_icon_state
    
    # 更新API配置
    if api_key:
        os.environ["DF_API_KEY"] = api_key
    
    if chat_api_url:
        global_icon_state.request.chat_api_url = chat_api_url
    
    if keywords: 
        global_icon_state.request.keywords= keywords
    if style: 
        global_icon_state.request.style = style
    if prev_img: 
        global_icon_state.request.prev_img = prev_img
    if edit_prompt: 
        global_icon_state.request.edit_prompt = edit_prompt
    if model:
        global_icon_state.request.model = model  # 使用字典访问方式

    out = await icon_graph.ainvoke(global_icon_state)
    global_icon_state = _merge_state(global_icon_state, out)
    return _get_img_from_icon_graph(global_icon_state)


async def run_paper_model_generation(paper_content=None, style=None, edit_prompt=None, prev_img=None,
                                     model=None, chat_api_url=None, api_key=None):
    """运行论文模型图生成流程"""
    global global_paper_state
    
    # 更新API配置 - 确保API密钥优先使用传入的参数
    if api_key:
        os.environ["DF_API_KEY"] = api_key
        # 添加日志验证API密钥是否被正确设置
        log.info(f"API密钥已更新: {api_key[:4]}****")  # 仅显示前4位以保护隐私
    
    if chat_api_url:
        global_paper_state.request.chat_api_url = chat_api_url
        log.info(f"API地址已更新: {chat_api_url}")
    
    # 使用正确的IconGenState和IconGenRequest
    if paper_content:
        global_paper_state.request.keywords = f"论文内容：{paper_content}"
    if style: 
        global_paper_state.request.style = style
    if prev_img: 
        global_paper_state.request.prev_image = prev_img
    if edit_prompt: 
        global_paper_state.request.edit_prompt = edit_prompt
    if model:
        global_paper_state.request.model = model  # IconGenRequest对象属性访问

    out = await paper_graph.ainvoke(global_paper_state)
    global_paper_state = _merge_state(global_paper_state, out)
    return _get_img_from_paper_graph(global_paper_state)


def create_icongen_refine():
    """页面入口函数"""
    with gr.Blocks(title="IconGen Pro — 多模式生成与迭代") as page:
        gr.Markdown("## 🎨 IconGen Pro — 多模式生成与迭代")

        with gr.Row():
            gen_type = gr.Dropdown(
                choices=["图标生成", "论文模型图生成"],
                label="生成类型",
                value="图标生成"
            )

        # API配置区域
        with gr.Accordion("⚙️ API配置", open=False):
            with gr.Row():
                model_input = gr.Textbox(
                    label="🤖 模型名称",
                    placeholder="例如：gemini-2.5-flash-image-preview",
                    value="gemini-2.5-flash-image-preview"
                )
                chat_api_url_input = gr.Textbox(
                    label="🌐 API地址",
                    placeholder="例如：http://123.129.219.111:3000/v1",
                    value="http://123.129.219.111:3000/v1"
                )
            api_key_input = gr.Textbox(
                label="🔑 API密钥",
                placeholder="输入API密钥",
                value="",
                type="password"
            )

        with gr.Row(visible=True) as icon_row:
            keywords = gr.Textbox(label="🎯 关键词", placeholder="例如：小兔子 / 机器人")
            style = gr.Textbox(label="✨ 风格", placeholder="例如：极简 / 扁平 / 赛博朋克")

        with gr.Row(visible=False) as paper_row:
            paper_content = gr.Textbox(
                label="📄 论文内容", 
                placeholder="描述论文模型的核心内容，例如：基于Transformer的文本分类模型",
                lines=3
            )
            paper_style = gr.Textbox(label="✨ 风格", placeholder="例如：学术图表 / 技术图解")

        gen_btn = gr.Button("🚀 生成")
        image_output = gr.Image(label="输出图像", type="filepath")

        edit_prompt = gr.Textbox(label="✏️ 编辑提示词", placeholder="例如：add neon glow / change color scheme")
        refine_btn = gr.Button("🎨 继续 refine")

        def toggle_inputs(gen_type):
            """根据生成类型切换输入框显示"""
            if gen_type == "图标生成":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        gen_type.change(toggle_inputs, [gen_type], [icon_row, paper_row])

        def ui_generate(gen_type, keywords, style, paper_content, paper_style, 
                       model, chat_api_url, api_key):
            """生成按钮点击事件"""
            global last_image
            try:
                # 确保传入的值不为空字符串
                model = model if model and model.strip() else None
                chat_api_url = chat_api_url if chat_api_url and chat_api_url.strip() else None
                api_key = api_key if api_key and api_key.strip() else None
                
                if gen_type == "图标生成":
                    log.info(f"[IconGen] 图标生成 | {keywords=} {style=} {model=}")
                    img = asyncio.run(run_icon_generation(
                        keywords=keywords, 
                        style=style,
                        model=model,
                        chat_api_url=chat_api_url,
                        api_key=api_key
                    ))
                else:
                    log.info(f"[PaperModel] 论文模型图生成 | {paper_content=} {paper_style=} {model=}")
                    img = asyncio.run(run_paper_model_generation(
                        paper_content=paper_content, 
                        style=paper_style,
                        model=model,
                        chat_api_url=chat_api_url,
                        api_key=api_key
                    ))
                
                last_image = img
                return img
            except Exception as e:
                log.error(f"生成失败: {e}", exc_info=True)
                return None

        def ui_refine(gen_type, edit_prompt, model, chat_api_url, api_key):
            """Refine按钮点击事件"""
            global last_image
            if not last_image:
                log.warning("没有可用的上一张图片进行refine")
                return None
            try:
                # 确保传入的值不为空字符串
                model = model if model and model.strip() else None
                chat_api_url = chat_api_url if chat_api_url and chat_api_url.strip() else None
                api_key = api_key if api_key and api_key.strip() else None
                
                if gen_type == "图标生成":
                    log.info(f"[IconGen] refine | {edit_prompt=} {model=}")
                    img = asyncio.run(run_icon_generation(
                        edit_prompt=edit_prompt, 
                        prev_img=last_image,
                        model=model,
                        chat_api_url=chat_api_url,
                        api_key=api_key
                    ))
                else:
                    log.info(f"[PaperModel] refine | {edit_prompt=} {model=}")
                    img = asyncio.run(run_paper_model_generation(
                        edit_prompt=edit_prompt, 
                        prev_img=last_image,
                        model=model,
                        chat_api_url=chat_api_url,
                        api_key=api_key
                    ))
                
                last_image = img
                return img
            except Exception as e:
                log.error(f"Refine失败: {e}", exc_info=True)
                return None

        gen_btn.click(
            ui_generate, 
            [gen_type, keywords, style, paper_content, paper_style, 
             model_input, chat_api_url_input, api_key_input], 
            [image_output]
        )
        refine_btn.click(
            ui_refine, 
            [gen_type, edit_prompt, model_input, chat_api_url_input, api_key_input], 
            [image_output]
        )

    return page