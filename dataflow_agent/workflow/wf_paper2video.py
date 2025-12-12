"""
paper2video workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成时间: 2025-11-26 11:08:03

1. 在 **TOOLS** 区域定义需要暴露给 Prompt 的前置工具
2. 在 **NODES**  区域实现异步节点函数 (await-able)
3. 在 **EDGES**  区域声明有向边
4. 最后返回 builder.compile() 或 GenericGraphBuilder
"""

from __future__ import annotations
import json
from dataclasses import Field
from pydantic import BaseModel
from dataflow_agent.state import Paper2VideoRequest, Paper2VideoState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register

from dataflow_agent.toolkits.tool_manager import get_tool_manager
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dataflow_agent.toolkits.p2vtool.p2v_tool import compile_tex, beamer_code_validator, get_image_paths

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from pathlib import Path
from pdf2image import convert_from_path
from dataflow_agent.toolkits.imtool.mineru_tool import run_mineru_pdf_extract

log = get_logger(__name__)

# @register("paper2video")
def create_paper2video_graph() -> GenericGraphBuilder:
    """
    Workflow factory: dfa run --wf paper2video
    """
    builder = GenericGraphBuilder(state_model=Paper2VideoState,
                                  entry_point="p2v_extract_pdf")  # 自行修改入口

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions)
    # ----------------------------------------------------------------------

    @builder.pre_tool("pdf_markdown", "p2v_extract_pdf")
    def get_markdown(state: Paper2VideoState):
        import subprocess
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        # paper_pdf_path = Path("/mnt/DataFlow/lz/proj/agentgroup/ligang/DataFlow-Agent/data/2510.05096v2.pdf")
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return ""
        paper_pdf_dir = paper_pdf_path.with_suffix('').parent
        if not paper_pdf_path.with_suffix('').exists():
            #fixme: 这里需要修改为部署机器上的mineru
            run_mineru_pdf_extract(str(paper_pdf_path), str(paper_pdf_dir), "modelscope")
            
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        markdown_path = paper_output_dir / "auto" / f"{paper_base_path.name}.md"
        if not markdown_path.exists():
            log.error(f"Markdown 文件不存在: {str(markdown_path)}")
            return ""
        try:
            markdown_content = markdown_path.read_text(encoding='utf-8')
            return markdown_content
        except Exception as e:
            log.error(f'读取 markdown 文件内容失败：{markdown_path}. 错误：{e}')
            return ""
        
    @builder.pre_tool("pdf_images_working_dir", "p2v_extract_pdf")
    def get_images_relative_path(state: Paper2VideoState):
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        # paper_pdf_path = Path("/mnt/DataFlow/lz/proj/agentgroup/ligang/DataFlow-Agent/data/2510.05096v2.pdf")
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return ""
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        images_dir = paper_output_dir/"auto"
        if not images_dir.exists():
            log.error(f"没有生成对应的图片，MinerU 识别图像失败：{images_dir}")
            return ""
        return str(images_dir)
    
    @builder.pre_tool("output_language", "p2v_extract_pdf")
    def get_language(state: Paper2VideoState):
        language_map = {
            'en': "English",
            'zh': "Chinese",
        }
        language = state.request.language
        return language_map.get(language, "English")
        
    @builder.pre_tool("is_beamer_wrong", "p2v_beamer_code_debug")
    def get_is_code_wrong(state: Paper2VideoState):
        return state.is_beamer_wrong

    @builder.pre_tool("is_beamer_warning", "p2v_beamer_code_debug")
    def get_is_code_warning(state: Paper2VideoState):
        return state.is_beamer_warning

    @builder.pre_tool("code_debug_result", "p2v_beamer_code_debug")
    def get_compile_result(state: Paper2VideoState):
        return state.code_debug_result
    
    @builder.pre_tool("beamer_code", "p2v_beamer_code_debug")
    def get_beamer_code(state: Paper2VideoState):
        beamer_code_path = state.beamer_code_path
        beamer_code = Path(beamer_code_path).read_text(encoding='utf-8')
        return beamer_code
    
    @builder.pre_tool("set_subtitle_and_cursor_path", "p2v_subtitle_and_cursor")
    def set_subtitle_and_cursor_path(state: Paper2VideoState):
        # 因为是循环调用VLM，所以这里就只调用一次
        if state.subtitle_and_cursor_path is not "" and state.slide_img_dir is not "":
            return None
        '''处理好slide_img，并且处理好路径，同时将最后输出文档的地址写好'''
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return ""
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        subtitle_and_cursor_path = paper_output_dir/"subtitle_w_cursor.txt"
        state.subtitle_and_cursor_path = str(subtitle_and_cursor_path)

        slide_img_dir = paper_output_dir/"slide_imgs"
        slide_img_dir.mkdir(parents=True, exist_ok=True)
        slide_imgs = convert_from_path(state.ppt_path)
        for i, img in enumerate(slide_imgs):
            img_path = slide_img_dir / f"slide_{i+1:03d}.png"
            img.save(img_path, 'PNG')
        state.slide_img_dir = str(slide_img_dir)
        return None

    # 后置工具就是让agent选择的工具，可以定制多个；
    # class ModuleListInput(BaseModel):
    #     #这里要写好工具的描述，agent会根据实际上下文输入参数：
    #     module_list: list = Field(
    #         description="List of dotted-path python modules or file paths"
    #     )
    # @builder.post_tool("step2")
    # @tool(args_schema=ModuleListInput)
    # def _post_tool1(module_list):
    #     return func(module_list)

    # ----------------------------------------------------------------------

    # ==============================================================
    # NODES
    # ==============================================================
    async def extract_pdf_node(state: Paper2VideoState) -> Paper2VideoState:
        from dataflow_agent.agentroles import create_vlm_agent
        log.info("开始执行extract_pdf_node节点")
        agent = create_vlm_agent(
            name="p2v_extract_pdf",
            vlm_mode="understanding",     # 视觉模式: 'understanding', 'generation', 'edit'
            image_detail="high",          # 图像细节: 'low', 'high', 'auto'
            model_name="gpt-4o-2024-11-20",  # 视觉模型
            temperature=0.1,
            max_image_size=(2048, 2048),  # 最大图像尺寸

            # additional_params={},        # 额外VLM参数，可以存放图片用法为："input_image": image_path
        )
    
        state = await agent.execute(state=state)

        # 可选：处理执行结果
        # agent_result = state.agent_results.get(agent.role_name, {})
        # log.info(f"Agent {agent.role_name} 执行结果: {agent_result}")
        
        return state

    def compile_beamer_node(state: Paper2VideoState) -> Paper2VideoState:
        log.info(f"开始执行compile_beamer_node")
        beamer_code_path = state.beamer_code_path
        state.is_beamer_wrong, state.is_beamer_warning, state.code_debug_result = compile_tex(beamer_code_path)
        if not state.is_beamer_wrong:
            log.info(f"Beamer 代码编译成功，无需调试")
            state.ppt_path = state.beamer_code_path.replace(".tex", ".pdf")
        return state
    
    async def beamer_code_debug_node(state: Paper2VideoState) -> Paper2VideoState:
        from dataflow_agent.agentroles import create_react_agent
        log.info(f"开始执行 p2v_beamer_code_debug node节点")
        agent = create_react_agent(
            name="p2v_beamer_code_debug",
            model_name="gpt-4o-2024-11-20",
            max_retries=10,
            validators=[beamer_code_validator],
        )
        state = await agent.execute(state)
        return state
    
    async def subtitle_and_cursor(state: Paper2VideoState) -> Paper2VideoState:
        log.info(f"开始执行 p2v_subtitle_and_cursor node节点")
        from dataflow_agent.agentroles import create_vlm_agent

        slide_img_dir = state.slide_img_dir
        slide_image_path_list = get_image_paths(slide_img_dir)
        log.info(f"获得了slide_image from {slide_img_dir}, the total images are {len(slide_image_path_list)}")
        for img_path in slide_image_path_list:
            agent = create_vlm_agent(
                name="p2v_subtitle_and_cursor",
                vlm_mode="understanding",     # 视觉模式: 'understanding', 'generation', 'edit'
                image_detail="high",          # 图像细节: 'low', 'high', 'auto'
                model_name="gpt-4o-2024-11-20",  # 视觉模型
                temperature=0.1,
                max_image_size=(2048, 2048),  # 最大图像尺寸

                additional_params={
                    "input_image": img_path,
                },        # 额外VLM参数，可以存放图片用法为："input_image": image_path
            )
            state = await agent.execute(state=state)
        subtitle_and_cursor_info = "\n###\n".join(state.subtitle_and_cursor)
        log.info(f"获取了完整的 Subtitle and Cursor 信息：\n {subtitle_and_cursor_info}")
        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        log.info(f"内容将写入到文件地址 {subtitle_and_cursor_path}中......")
        Path(subtitle_and_cursor_path).write_text(subtitle_and_cursor_info, encoding='utf-8')
        return state
    
    async def compile_beamer_condition(state: Paper2VideoState):
        # todo: 暂时先这样判断
        if state.is_beamer_warning:
            return "p2v_beamer_code_debug"
        else:
            return "__end__"


    async def pdf2ppt_node(state: Paper2VideoState) -> Paper2VideoState:
        
        log.info(f"开始执行 pdf2ppt node节点")
        from dataflow_agent.agentroles import create_simple_agent
        # agent = create_simple_agent(
        #     name=""
        # )
        
        
        return state
    
    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "p2v_extract_pdf": extract_pdf_node,
        "compile_beamer": compile_beamer_node,
        "p2v_beamer_code_debug": beamer_code_debug_node,
        "p2v_subtitle_and_cursor": subtitle_and_cursor,
        "pdf2ppt": pdf2ppt_node,
        '_end_': lambda state: state,  # 终止节点
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("p2v_extract_pdf", "compile_beamer"),
        ("p2v_beamer_code_debug", "__end__")
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("compile_beamer", compile_beamer_condition)
    return builder

if __name__ == "__main__":
    import asyncio
    graph_builder = create_paper2video_graph().build()
    
    p2v_state = Paper2VideoState(request=Paper2VideoRequest(chat_api_url="http://123.129.219.111:3000/v1"))
    out =  asyncio.run(graph_builder.ainvoke(p2v_state))