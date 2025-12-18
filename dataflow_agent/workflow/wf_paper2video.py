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
from dataflow_agent.toolkits.p2vtool.p2v_tool import (
    compile_tex, beamer_code_validator, get_image_paths, parse_script, 
    transcribe_with_whisperx, inference_f5, cursor_infer, get_audio_paths, 
    get_whisperx_model, load_align_whisperx_model, clean_text, timesteps, 
    talking_gen_per_slide, render_video_with_cursor_from_json, add_subtitles,
    parser_beamer_latex, resize_latex_image
)

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
        if state.subtitle_and_cursor_path != "" and state.slide_img_dir != "":
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
        if not state.is_beamer_warning:
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

    async def beamer_code_upgrade_node(state: Paper2VideoState) -> Paper2VideoState:
        log.info(f"开始执行 p2v_beamer_code_debug node节点")
        from dataflow_agent.agentroles import create_vlm_agent
        from tempfile import TemporaryDirectory
        import subprocess
        from pdf2image import convert_from_path

        beamer_code_path = state.beamer_code_path
        old_beamer_code = Path(beamer_code_path).read_text(encoding='utf-8')

        head, frames_code = parser_beamer_latex(old_beamer_code)
        final_frames = []
        doc_header = ["\\documentclass{beamer}", head, "\\begin{document}"]
        doc_footer = ["\\end{document}"]
        
        for frame_code in frames_code:
            current_frame_content = ["\\begin{frame}", frame_code, "\\end{frame}"]
            
            if "includegraphics" not in frame_code:
                final_frames.extend(current_frame_content)
                continue
            
            attempt_code = current_frame_content
            img_size_debug = True

            while img_size_debug:
                with TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    # 在临时目录中创建 .tex 文件
                    tex_path = temp_dir / "input.tex"
                    
                    full_temp_tex = doc_header + attempt_code + doc_footer
                    tex_path.write_text("\n".join(full_temp_tex), encoding='utf-8')
                    try:
                        subprocess.run(
                            ["tectonic", str(tex_path)],
                            check=True, capture_output=True, text=True, cwd=temp_dir
                        )
                        
                        frame_pdf_path = tex_path.with_suffix('.pdf')
                        img_path = tex_path.with_suffix('.png')

                        if frame_pdf_path.exists():
                            images = convert_from_path(str(frame_pdf_path))
                            images[0].save(str(img_path))
                            
                            agent = create_vlm_agent(
                                name="p2v_beamer_code_upgrade",
                                vlm_mode="understanding",
                                model_name="gpt-4o-2024-11-20",
                                additional_params={"input_image": str(img_path)},
                            )
                            
                            state = await agent.execute(state=state)
                            img_size_debug = getattr(state, 'img_size_debug', False)
                            
                            if img_size_debug:
                                log.info(f"当前图片尺寸超出了ppt一页，需要修改：{attempt_code}")
                                attempt_code = resize_latex_image(attempt_code) 
                            else:
                                final_frames.extend(attempt_code)
                        else:
                            log.error("PDF 未生成，跳过调试")
                            final_frames.extend(attempt_code)
                            break
                    except Exception as e:
                        log.error(f"解析单张ppt发生了错误: {e}")
                        final_frames.extend(attempt_code)
                        break
        full_new_code = doc_header + final_frames + doc_footer
        Path(beamer_code_path).write_text("\n".join(full_new_code), encoding='utf-8')
        compile_tex(beamer_code_path)
        state.ppt_path = str(Path(beamer_code_path).with_suffix(".pdf"))
        log.info(f"将更新好的beamer code写回 {beamer_code_path}")

        return state

    
    async def subtitle_and_cursor(state: Paper2VideoState) -> Paper2VideoState:
        log.info(f"开始执行 p2v_subtitle_and_cursor node节点")
        from dataflow_agent.agentroles import create_vlm_agent

        slide_img_dir = state.slide_img_dir
        slide_image_path_list = get_image_paths(slide_img_dir)
        log.info(f"获得了slide_image from {slide_img_dir}, the total images are {len(slide_image_path_list)}, the images path are {"\n".join(slide_image_path_list)}")
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
    
    def generate_speech(state: Paper2VideoState):
        # 先完成pre-tool的工作
        log.info(f"开始执行 p2v_generate_speech node节点")
        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return ""
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        speech_save_dir = paper_output_dir/"audio"
        state.speech_save_dir = str(speech_save_dir)

        speech_save_dir.mkdir(parents=True, exist_ok=True)
        ref_audio_path = state.request.ref_audio_path

        # 1、拿到subtitle的文件，并且读出其中的内容，并解析        
        raw_subtitle_and_cursor_content = Path(subtitle_and_cursor_path).read_text(encoding='utf-8')
        log.info(f"获取到字幕内容：\n{raw_subtitle_and_cursor_content}")
        parsed_subtitle_w_cursor = parse_script(raw_subtitle_and_cursor_content)
        
        # 2、不同的slide分别进行处理
        for slide_idx in range(len(parsed_subtitle_w_cursor)):
            speech_with_cursor = parsed_subtitle_w_cursor[slide_idx]
            subtitle = ""
            for _, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
                if len(subtitle) == 0: 
                    subtitle = prompt
                else: 
                    subtitle = subtitle + "\n\n\n" + prompt
            speech_result_path = speech_save_dir / f"{slide_idx}.wav"
            
            # 3、将每个slide的字幕内容转换为音频，并保存到指定的目录
            ref_text = transcribe_with_whisperx(ref_audio_path)
            inference_f5(subtitle, str(speech_result_path), ref_audio_path, ref_text)
            log.info(f"生成 slide {slide_idx} 的语音，保存到 {speech_result_path}")
        return state
    
    def generate_cursor(state: Paper2VideoState):
        import multiprocessing as mp
        import cv2
        from whisperx import load_audio
        from whisperx.alignment import align
        log.info(f"开始执行 p2v_generate_cursor node节点")
        # 先完成pre-tool的工作
        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        slide_img_dir = state.slide_img_dir
        speech_save_dir = state.speech_save_dir
        
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        cursor_save_path = paper_output_dir/"cursor.json"
        state.cursor_save_path = str(cursor_save_path)

        # 1、获取字幕内容
        raw_subtitle_and_cursor_content = Path(subtitle_and_cursor_path).read_text(encoding='utf-8')
        parsed_subtitle_w_cursor = parse_script(raw_subtitle_and_cursor_content)

        # 2、并行的生成cursor的坐标等信息
        slide_image_path_list = get_image_paths(slide_img_dir)

        task_list = []
        for slide_idx in range(len(parsed_subtitle_w_cursor)):
            slide_image_path = slide_image_path_list[slide_idx]
            speech_with_cursor = parsed_subtitle_w_cursor[slide_idx]
            for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
                task_list.append((slide_idx, sentence_idx, prompt, cursor_prompt, slide_image_path))

        cursor_result = []
        for task_args in task_list:
            result = cursor_infer(task_args)
            cursor_result.append(result)

        slide_w, slide_h = cv2.imread(slide_image_path_list[0]).shape[:2]
        for index in range(len(cursor_result)):
            if cursor_result[index]["cursor_prompt"] == "no":
                cursor_result[index]["cursor"] == (slide_w//2, slide_h//2)
        
        slide_sentence_timesteps = []
        slide_audio_path_list = get_audio_paths(speech_save_dir)

        model = get_whisperx_model("large-v3", device="cuda")
        align_model, metadata = load_align_whisperx_model(language_code="en", device="cuda")
        for idx, slide_audio_path in enumerate(slide_audio_path_list):
            subtitle = []
            cursor = []
            for info in cursor_result: 
                if info["slide"] == idx: 
                    subtitle.append(clean_text(info["speech_text"]))
                    cursor.append(info["cursor"])
            # 转换为模型可处理的语音数据
            audio = load_audio(slide_audio_path)
            # 语音转文字，并生成初步的时间戳
            result = model.transcribe(slide_audio_path, language="en")
            # 语音和文字同步 对齐
            aligned = align(transcript=result["segments"], align_model_metadata=metadata, model=align_model, audio=audio, device="cuda")
            sentence_timesteps = timesteps(subtitle, aligned, slide_audio_path)
            for idx in range(len(sentence_timesteps)): 
                sentence_timesteps[idx]["cursor"] = cursor[idx]
            slide_sentence_timesteps.append(sentence_timesteps)

        # 将相对的、片段的时间 转换成 绝对的、完整的时间
        start_time_now = 0
        new_slide_sentence_timesteps = []
        for sentence_timesteps in slide_sentence_timesteps:
            duration = 0
            for idx in range(len(sentence_timesteps)):
                if sentence_timesteps[idx]["start"] is None: 
                    sentence_timesteps[idx]["start"] = sentence_timesteps[idx-1]["end"]
                if sentence_timesteps[idx]["end"] is None: 
                    sentence_timesteps[idx]["end"] = sentence_timesteps[idx+1]["start"]

            for idx in range(len(sentence_timesteps)):
                sentence_timesteps[idx]["start"] += start_time_now
                sentence_timesteps[idx]["end"] += start_time_now
                duration += sentence_timesteps[idx]["end"] - sentence_timesteps[idx]["start"]
            start_time_now += duration
            new_slide_sentence_timesteps.extend(sentence_timesteps)

        file_name = cursor_save_path.name.replace(".json", "_mid.json")
        cursor_mid_save_path = cursor_save_path.with_name(file_name)
        cursor_mid_save_path.write_text(
            json.dump(cursor_result, indent=2), 
            encoding='utf-8'
        )
        cursor_save_path.write_text(
            json.dump(new_slide_sentence_timesteps, indent=2), 
            encoding='utf-8'
        )
        
        return state

    def generate_talking_video(state: Paper2VideoState):
        log.info(f"开始执行 p2v_generate_taking_video node节点")
        
        # 先完成pre-tool的工作
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        talking_video_save_dir = paper_output_dir/"talking_video"

        state.talking_video_save_dir = str(talking_video_save_dir)
        talking_video_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1、
        talking_inference_input = []
        audio_path_list = get_audio_paths(state.speech_save_dir)
        for audio_path in audio_path_list:
            talking_inference_input.append([state.request.ref_img_path, audio_path])
        talking_gen_per_slide("hallo2", talking_inference_input, paper_output_dir, talking_video_save_dir, "")
        log.info(f"talking-video 的信息已经写入了{talking_video_save_dir}目录中")
        
        return state
    
    def merge_all(state: Paper2VideoState):
        import cv2
        import subprocess
        
        log.info(f"开始执行 p2v_merge_all node节点")
        
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        slide_img_dir = state.slide_img_dir
        talking_save_dir = state.talking_video_save_dir
        ref_img = state.request.ref_img_path
        cursor_save_path = state.cursor_save_path
        # fixme: 这个需要将cursor文件放到对应的地址
        cursor_img_path = paper_base_path / "cursor_image" / "red.png"

        tmp_merage_dir = paper_output_dir / "merage"
        tmp_merage_1 = paper_output_dir / "1_merage.mp4"
        image_size = cv2.imread(Path(slide_img_dir) / '1.png').shape

        size = max(image_size[0]//6, image_size[1]//6)
        width, height = size, size
        num_slide = len(get_image_paths(slide_img_dir))

        # fixme: 这个./1_merge.bash需要处理一下啊
        merage_cmd =  ["./1_merage.bash", slide_img_dir, talking_save_dir, tmp_merage_dir,
                    str(width), str(height), str(num_slide), tmp_merage_1, ref_img.split("/")[-1].replace(".png", "")]
        out = subprocess.run(merage_cmd, text=True)
        # render cursor
        cursor_size = size//6
        tmp_merage_2 = paper_output_dir /  "2_merage.mp4"
        render_video_with_cursor_from_json(video_path=tmp_merage_1, out_video_path=tmp_merage_2, 
                                        json_path=cursor_save_path, cursor_img_path=cursor_img_path, 
                                        transition_duration=0.1, cursor_size=cursor_size)
        # render subtitle
        front_size = size//10
        tmp_merage_3 = paper_output_dir / "3_merage.mp4"
        add_subtitles(tmp_merage_2, tmp_merage_3, size//10)

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
        "p2v_beamer_code_upgrade": beamer_code_upgrade_node,
        "p2v_subtitle_and_cursor": subtitle_and_cursor,
        "p2v_generate_speech": generate_speech,
        "p2v_generate_cursor": generate_cursor,
        "p2v_generate_taking_video": generate_talking_video,
        "p2v_merge": merge_all,  
        "pdf2ppt": pdf2ppt_node,
        
        '_end_': lambda state: state,  # 终止节点
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("p2v_extract_pdf", "compile_beamer"),
        ("p2v_beamer_code_debug", "p2v_beamer_code_upgrade"),
        ("p2v_beamer_code_upgrade", "__end__")
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("compile_beamer", compile_beamer_condition)
    return builder

if __name__ == "__main__":
    import asyncio
    graph_builder = create_paper2video_graph().build()
    
    p2v_state = Paper2VideoState(request=Paper2VideoRequest(chat_api_url="http://123.129.219.111:3000/v1"))
    out =  asyncio.run(graph_builder.ainvoke(p2v_state))