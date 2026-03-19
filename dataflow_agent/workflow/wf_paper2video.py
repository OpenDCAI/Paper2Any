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
    get_image_paths, parse_script_with_cursor,
    transcribe_with_whisperx, cursor_infer, get_audio_paths, get_audio_length,
    clean_text,
    talking_gen_per_slide, render_video_with_cursor_from_json, add_subtitles,
    merge_wav_files, get_mp4_duration_ffprobe,
    speech_task_wrapper_with_cloud_tts,
    speech_task_wrapper_with_f5,
)

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from pathlib import Path
from pdf2image import convert_from_path

log = get_logger(__name__)

# 当 LLM 返回格式不符合 {"subtitle_and_cursor": "..."} 时的最大重试次数
MAX_SUBTITLE_RETRIES = 3

@register("paper2video")
def create_paper2video_graph() -> GenericGraphBuilder:
    """
    Workflow factory: dfa run --wf paper2video
    """
    builder = GenericGraphBuilder(state_model=Paper2VideoState,
                                  entry_point="_start_")  # 自行修改入口

    # ----------------------------------------------------------------------
    # TOOLS (pre_tool definitions，仅 paper2video 相关)
    # ----------------------------------------------------------------------

    @builder.pre_tool("video_language", "p2v_subtitle_and_cursor")
    def get_video_language(state: Paper2VideoState):
        language = "Chinese" if state.request.language == "zh" else "English"
        return language

    @builder.pre_tool("tmp_sentence", "p2v_refine_subtitle_and_cursor")
    def get_tmp_sentence(state: Paper2VideoState):
        return state.tmp_sentence

    @builder.pre_tool("video_language", "p2v_refine_subtitle_and_cursor")
    def get_video_language(state: Paper2VideoState):
        language = "Chinese" if state.request.language == "zh" else "English"
        return language

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
    # NODES (仅 paper2video 相关)
    # ==============================================================

    async def subtitle_and_cursor(state: Paper2VideoState) -> Paper2VideoState:
        '''
        将一个pdf切分成多个image，然后每张图都经过VLM，让其生成对应的内容 句子
        '''
        log.info(f"开始执行 p2v_subtitle_and_cursor node节点")
        from dataflow_agent.agentroles import create_vlm_agent

        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return state
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        subtitle_and_cursor_path = paper_output_dir/"subtitle_w_cursor.txt"
        state.subtitle_and_cursor_path = str(subtitle_and_cursor_path)
        log.info(f"state中的 subtitle_and_cursor_path 设置为了{str(subtitle_and_cursor_path)}")

        slide_img_dir = paper_output_dir/"slide_imgs"
        slide_img_dir.mkdir(parents=True, exist_ok=True)
        slide_imgs = convert_from_path(state.request.get("paper_pdf_path", ""), size=(1920, 1080))
        for i, img in enumerate(slide_imgs):
            img_path = slide_img_dir / f"{i+1}.png"
            img.save(img_path, 'PNG')
        state.slide_img_dir = str(slide_img_dir)
        log.info(f"state中的 slide_img_dir 设置为了{str(slide_img_dir)}")

        slide_image_path_list = get_image_paths(slide_img_dir)
        image_paths = '\n'.join(slide_image_path_list)
        log.info(f"获得了slide_image from {slide_img_dir}, the total images are {len(slide_image_path_list)}, the images path are {image_paths}")
        for img_idx, img_path in enumerate(slide_image_path_list):
            agent = create_vlm_agent(
                name="p2v_subtitle_and_cursor",
                vlm_mode="understanding",
                image_detail="high",
                model_name="gpt-4o-2024-11-20",
                temperature=0.1,
                max_image_size=(2048, 2048),
                additional_params={"input_image": img_path},
            )
            prev_len = len(state.subtitle_and_cursor)
            for attempt in range(MAX_SUBTITLE_RETRIES):
                state = await agent.execute(state=state)
                if len(state.subtitle_and_cursor) > prev_len:
                    break
                log.warning("第 %s 张 slide 返回格式不符合 {\"subtitle_and_cursor\": \"...\"}，第 %s 次重试", img_idx + 1, attempt + 1)
            else:
                state.subtitle_and_cursor.append("")
                log.warning("第 %s 张 slide 重试 %s 次后仍格式错误，使用占位内容", img_idx + 1, MAX_SUBTITLE_RETRIES)
        subtitle_and_cursor_info = "\n###\n".join(state.subtitle_and_cursor)
        log.info(f"获取了完整的 Subtitle and Cursor 信息：\n {subtitle_and_cursor_info}")
        
        # 需要将这些信息写入到那个script_pages中
        script_pages = []
        for page_num, script_text in enumerate(state.subtitle_and_cursor):
            script_pages.append({
                "page_num": page_num,
                "image_path": slide_image_path_list[page_num],
                "script_text": script_text,
            })
        state.script_pages = script_pages
        log.info(f"将script_pages信息写入到state中，script_pages: {script_pages}")
        return state

    async def refine_subtitle_and_cursor(state: Paper2VideoState) -> Paper2VideoState:
        '''
        对subtitle_and_cursor文件中的内容进行细化，使其更加符合实际情况。
        每张 slide 对应 script_pages 中的一页 script_text，经 agent 细化后写入 state.subtitle_and_cursor，
        最后写回 subtitle_and_cursor_path。
        '''
        log.info(f"开始执行 p2v_refine_subtitle_and_cursor node节点")
        from dataflow_agent.agentroles import create_vlm_agent

        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        slide_img_dir = state.slide_img_dir
        script_pages = state.script_pages
        sentences = [script_page["script_text"] for script_page in script_pages]

        slide_image_path_list = get_image_paths(slide_img_dir)

        # 清空后由 refine agent 在 update_state_result 中逐条 append，避免沿用旧内容
        state.subtitle_and_cursor = []

        for i, img_path in enumerate(slide_image_path_list):
            if i >= len(sentences):
                log.warning("slide 数量多于 script_pages，跳过剩余图片")
                break
            state.tmp_sentence = sentences[i]
            agent = create_vlm_agent(
                name="p2v_refine_subtitle_and_cursor",
                vlm_mode="understanding",
                image_detail="high",
                model_name="gpt-4o-2024-11-20",
                temperature=0.1,
                max_image_size=(2048, 2048),
                additional_params={"input_image": img_path},
            )
            prev_len = len(state.subtitle_and_cursor)
            for attempt in range(MAX_SUBTITLE_RETRIES):
                state = await agent.execute(state=state)
                if len(state.subtitle_and_cursor) > prev_len:
                    break
                log.warning("第 %s 张 slide refine 返回格式不符合 {\"subtitle_and_cursor\": \"...\"}，第 %s 次重试", i + 1, attempt + 1)
            else:
                # 重试耗尽则用原句 + "| no" 占位，保证后续流程不中断
                state.subtitle_and_cursor.append((sentences[i] or "").strip() + " | no")
                log.warning("第 %s 张 slide 重试 %s 次后仍格式错误，使用原句占位", i + 1, MAX_SUBTITLE_RETRIES)
        subtitle_and_cursor_info = "\n###\n".join(state.subtitle_and_cursor)
        log.info(f"获取了完整的 Subtitle and Cursor 信息：\n {subtitle_and_cursor_info}")
        Path(subtitle_and_cursor_path).write_text(subtitle_and_cursor_info, encoding="utf-8")
        return state

    def generate_speech(state: Paper2VideoState):
        '''
        从subtitle_and_cursor文件中，读取对应的文本内容，然后逐句生成对应的语音文件，
        之后将per slide中的所有语音句子合并为一个，同时记录每个句子的持续时间，便于后续的时间对齐
        '''
        from concurrent.futures import ThreadPoolExecutor

        log.info(f"开始执行 p2v_generate_speech node节点")
        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        if not paper_pdf_path.exists():
            log.error(f"PDF 文件不存在: {paper_pdf_path}")
            return state
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        speech_save_dir = paper_output_dir/"audio"
        state.speech_save_dir = str(speech_save_dir)
        speech_language = state.request.language
        api_key = state.request.api_key
        tts_model = state.request.tts_model
        tts_voice_name = (getattr(state.request, "tts_voice_name", None) or "").strip()
        chat_api_url = state.request.chat_api_url

        speech_save_dir.mkdir(parents=True, exist_ok=True)
        ref_audio_path = state.request.ref_audio_path
        # 云语音：仅用 tts_voice_name 作为 CosyVoice 的 voice 参数，不涉及参考音频文件。
        # 本地语音（F5-TTS）：ref_audio_path 来自 sys_audio 目录或用户上传，作为参考语音。
        use_specific_sound = ref_audio_path is not None and (ref_audio_path or "").strip() != ""
        ref_text = state.request.ref_text
        script_pages = state.script_pages
        sentences = [script_page["script_text"] for script_page in script_pages]

        # 1、拿到subtitle的文件，并且读出其中的内容，并解析
        log.info(f"开始解析subtitle_and_cursor中的内容，文件路径{subtitle_and_cursor_path}")
        raw_subtitle_and_cursor_content = Path(subtitle_and_cursor_path).read_text(encoding='utf-8')
        parsed_subtitle_w_cursor = parse_script_with_cursor(raw_subtitle_and_cursor_content)

        # 2、不同的slide分别进行处理
        # 记录了所有ppt中所有sentence的持续时间，便于后续确定时间戳
        slide_timesteps = []

        # 2.1 + 2.2: 始终按句生成 wav（F5 或 Gemini 按句），得到 {slide_idx}_{idx}.wav 
        all_tasks = []
        results = []
        organized_results = {}

        if use_specific_sound:
            # 方案一：F5-TTS 按句，多进程 + 多 GPU
            if not (ref_text and ref_text.strip()):
                # fixme: 这里需要修改，硬编码了 4 卡
                ref_text = transcribe_with_whisperx(ref_audio_path, lang=speech_language, device_id=4)
            log.info(f"此时的ref_text 为 {ref_text}")
            gpu_list = [4,5,6]  # 按需修改可用 GPU ID
            num_gpus = len(gpu_list)
            for slide_idx in range(len(parsed_subtitle_w_cursor)):
                speech_with_cursor = parsed_subtitle_w_cursor[slide_idx]
                for idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
                    speech_result_path = speech_save_dir / f"{slide_idx}_{idx}.wav"
                    assigned_gpu = gpu_list[len(all_tasks) % num_gpus]
                    all_tasks.append((
                        slide_idx, idx, prompt, ref_audio_path,
                        ref_text, speech_result_path, assigned_gpu,
                    ))
            log.info(f"开始并行生成语音（F5-TTS 按句），任务总数: {len(all_tasks)}")
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(all_tasks)) as pool:
                results = list(pool.map(speech_task_wrapper_with_f5, all_tasks))
        else:
            # 方案二：云 TTS（CosyVoice）按句，音色仅用 tts_voice_name（如 longanhuan），不传参考音频。
            # paper2video API 统一走阿里云语音，不再在这里回退本地 F5-TTS。
            for slide_idx in range(len(parsed_subtitle_w_cursor)):
                speech_with_cursor = parsed_subtitle_w_cursor[slide_idx]
                for idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
                    speech_result_path = speech_save_dir / f"{slide_idx}_{idx}.wav"
                    all_tasks.append((
                        slide_idx, idx, prompt, speech_result_path,
                        api_key, tts_model, chat_api_url, tts_voice_name,
                    ))
            log.info(f"开始并行生成单句语音（云 TTS 按句，voice_name={tts_voice_name or 'default'}），任务总数: {len(all_tasks)}")
            # 降低 CosyVoice 并发，避免 Throttling.RateQuota（请求过于频繁）
            max_workers = min(3, len(all_tasks)) if all_tasks else 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(speech_task_wrapper_with_cloud_tts, all_tasks))

        # 2.3: 按 slide 归类按句结果，供 slide_timesteps（cursor 用）和后续合并/整段使用
        for s_idx, i_idx, dur, pth in results:
            if s_idx not in organized_results:
                organized_results[s_idx] = {}
            organized_results[s_idx][i_idx] = (dur, pth)

        # 2.4: 按页合并按句 wav 为一个 {slide_idx}.wav；slide_timesteps 来自按句结果
        for slide_idx in range(len(parsed_subtitle_w_cursor)):
            slide_speech_path = speech_save_dir / f"{slide_idx}.wav"
            current_slide_data = sorted(organized_results[slide_idx].items())
            sentence_duration_list = [data[1][0] for data in current_slide_data]
            sentence_speech_path_list = [data[1][1] for data in current_slide_data]
            slide_timesteps.append(sentence_duration_list)
            merge_wav_files(sentence_speech_path_list, str(slide_speech_path))
            for p in sentence_speech_path_list:
                Path(p).unlink(missing_ok=True)
            log.info(f"Slide {slide_idx} 合并完成: {slide_speech_path}")
        
        # 结构化 slide_timesteps中的数据
        formatted_data = []
        for slide_idx, durations in enumerate(slide_timesteps):
            formatted_data.append({
                "slide_id": slide_idx,
                "sentence_duration": [d for d in durations],
            })
        slide_timesteps_file = speech_save_dir / "slide_timesteps.json"
        with open(slide_timesteps_file, "w", encoding="utf-8") as f:
            json.dump(
                formatted_data, f, indent=4, ensure_ascii=False,
            )
        state.slide_timesteps_path = slide_timesteps_file
        log.info(f"生成 slide timesteps 文件，保存到 {slide_timesteps_file}")
        return state

    def generate_talking_video(state: Paper2VideoState):
        '''
        生成 talking head video，当前统一使用 LivePortrait API。
        当 ref_img_path 为空时由条件边跳过，不执行；此处仅做防御性判断。
        '''
        ref_img_path = state.request.ref_img_path or ""
        if not ref_img_path.strip():
            log.info("ref_img_path 为空，跳过 generate_talking_video")
            return state
        log.info(f"开始执行 p2v_generate_taking_video node节点")
        
        # 先完成pre-tool的工作
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        talking_video_save_dir = paper_output_dir/"talking_video"

        state.talking_video_save_dir = str(talking_video_save_dir)
        talking_video_save_dir.mkdir(parents=True, exist_ok=True)
        
        talking_inference_input = []
        audio_path_list = get_audio_paths(state.speech_save_dir)
        for audio_path in audio_path_list:
            talking_inference_input.append([state.request.ref_img_path, audio_path])
        talking_model = getattr(state.request, "talking_model", None) or "liveportrait"
        # LivePortrait Key 仅从环境变量 LIVEPORTRAIT_KEY 读取，不传 request
        talking_gen_per_slide(
            talking_model,
            talking_inference_input,
            paper_output_dir,
            talking_video_save_dir,
            "/root/miniconda3/envs/echomimic/bin/python",
            api_key=None,
        )
        log.info(f"talking-video 的信息已经写入了{talking_video_save_dir}目录中")
            
        return state
    
    def generate_cursor(state: Paper2VideoState):
        '''
        根据之前的cursor信息，让模型给出具体的坐标信息
        同时由于talking head中会导致最后时间 长于 语音文件的时间，而cursor是根据每个句子的持续时间算的
        所以，这里选择微调每个句子的持续时间，从而确保per slide的最终时间是一致的
        '''
        import multiprocessing
        import cv2
        log.info(f"开始执行 p2v_generate_cursor node节点")
        # 先完成pre-tool的工作
        subtitle_and_cursor_path = state.subtitle_and_cursor_path
        slide_img_dir = state.slide_img_dir
        slide_sentence_timesteps_path = state.slide_timesteps_path
        talking_video_save_dir = state.talking_video_save_dir
        
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        cursor_save_path = paper_output_dir/"cursor.json"
        state.cursor_save_path = str(cursor_save_path)

        # 1、获取字幕内容
        raw_subtitle_and_cursor_content = Path(subtitle_and_cursor_path).read_text(encoding='utf-8')
        parsed_subtitle_w_cursor = parse_script_with_cursor(raw_subtitle_and_cursor_content)

        # 2、生成 cursor 坐标（调用 GUI-Plus API，无需本地 GPU，多进程仅用于并行请求）
        slide_image_path_list = get_image_paths(slide_img_dir)

        task_list = []
        for slide_idx in range(len(parsed_subtitle_w_cursor)):
            slide_image_path = slide_image_path_list[slide_idx]
            speech_with_cursor = parsed_subtitle_w_cursor[slide_idx]
            for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
                task_list.append((slide_idx, sentence_idx, prompt, cursor_prompt, slide_image_path))

        num_workers = min(4, len(task_list)) if task_list else 1
        parallel_tasks = [t + (None,) for t in task_list]  # gpu_id 已废弃，传 None
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            cursor_result = pool.map(cursor_infer, parallel_tasks)
        cursor_result.sort(key=lambda x: (x['slide'], x['sentence']))

        slide_h, slide_w= cv2.imread(slide_image_path_list[0]).shape[:2]
        for index in range(len(cursor_result)):
            if cursor_result[index]["cursor_prompt"] == "no":
                cursor_result[index]["cursor"] = (slide_w//2, slide_h//2)
        
        slide_sentence_timesteps_w_cursor = []
        with open(slide_sentence_timesteps_path, 'r', encoding="utf-8") as f:
            slide_sentence_timesteps = json.load(f)

        # 处理 talking video 与 sentence 时间差：仅当有数字人（ref_img_path 非空）时按 talking_video 对齐
        ref_img_path = state.request.ref_img_path or ""
        use_talking_for_align = bool(ref_img_path.strip() and talking_video_save_dir and Path(talking_video_save_dir).exists())
        if use_talking_for_align:
            log.info(f"现在开始进行时间微调（基于 talking_video）")
            subdirs = sorted(
                [p.name for p in Path(talking_video_save_dir).iterdir() if p.is_dir()],
                key=lambda x: int(x) if x.isdigit() else 0
            )
            for subdir in subdirs:
                talking_video_path = Path(talking_video_save_dir)/subdir/"digit_person_withaudio.mp4"
                try:
                    talking_video_duration = get_mp4_duration_ffprobe(talking_video_path)
                except Exception:
                    log.warning(f"读取talking video文件失败 {talking_video_path}")
                    continue
                id = int(subdir)
                if id >= len(slide_sentence_timesteps):
                    log.warning(f"跳过文件夹 {subdir}，索引超出范围")
                    continue
                duration_list = slide_sentence_timesteps[id]["sentence_duration"]
                wav_duration = sum(duration_list)
                num_sentence = len(duration_list)
                bias_us = int(round(talking_video_duration * 1_000_000)) - int(round(wav_duration * 1_000_000))
                per_bias_us = bias_us // num_sentence
                per_bias_duration = per_bias_us / 1_000_000.0
                slide_sentence_timesteps[id]["sentence_duration"] = [
                    max(0, d+per_bias_duration) for d in duration_list
                ]
            with open(slide_sentence_timesteps_path, 'w', encoding="utf-8") as f:
                json.dump(slide_sentence_timesteps, f, indent=4, ensure_ascii=False)
            log.info(f"时间补偿完成（talking_video）")
        else:
            log.info("ref_img_path 为空，不进行 talking_video 时间补偿，直接使用 speech 的 slide_timesteps")

        start_time_now = 0
        cursor_iter = iter(cursor_result)
        for slide_info in slide_sentence_timesteps:
            slide_idx = slide_info["slide_id"]
            duration_list = slide_info["sentence_duration"]
            for sentence_idx, duration in enumerate(duration_list):
                start = start_time_now
                end = start + duration
                cursor_info = next(cursor_iter)
                # 需要确保这个cursor_info中的slide_id和sentence_id是一致的
                slide_sentence_timesteps_w_cursor.append({
                    "slide_id": slide_idx,
                    "start": start,
                    "end": end,
                    "text": clean_text(cursor_info.get("speech_text", "")),
                    "cursor": cursor_info.get("cursor", [slide_w // 2, slide_h // 2]),
                })
                start_time_now = end

        file_name = cursor_save_path.name.replace(".json", "_mid.json")
        cursor_mid_save_path = cursor_save_path.with_name(file_name)
        cursor_mid_save_path.write_text(
            json.dumps(cursor_result, indent=2, ensure_ascii=False), 
            encoding='utf-8'
        )
        cursor_save_path.write_text(
            json.dumps(slide_sentence_timesteps_w_cursor, indent=2, ensure_ascii=False), 
            encoding='utf-8',
        )
        
        return state
    
    def merge_all(state: Paper2VideoState):
        import cv2
        import subprocess
        
        log.info(f"开始执行 p2v_merge_all node节点")
        
        paper_pdf_path = Path(state.request.get("paper_pdf_path", ""))
        paper_base_path = paper_pdf_path.with_suffix('').expanduser().resolve()
        paper_output_dir = paper_base_path
        slide_img_dir = Path(state.slide_img_dir)
        talking_save_dir = state.talking_video_save_dir
        ref_img = (state.request.ref_img_path or "").strip()
        speech_save_dir = Path(state.speech_save_dir)
        cursor_save_path = state.cursor_save_path
        cursor_img_path = state.request.cursor_path

        tmp_merage_dir = paper_output_dir / "merge"
        tmp_merage_1 = paper_output_dir / "1_merge.mp4"
        tmp_merage_dir.mkdir(parents=True, exist_ok=True)
        image_size = cv2.imread(str(slide_img_dir / '1.png')).shape
        size = max(image_size[0]//6, image_size[1]//6)
        width, height = size, size
        num_slide = len(get_image_paths(slide_img_dir))

        if not ref_img:
            # 无数字人头像：用语音文件做 slide + 音频 合并，不做 talking_head
            log.info("ref_img_path 为空，使用 speech 做 slide+音频 合并")
            list_lines = []
            for i in range(num_slide):
                slide_path = slide_img_dir / f"{i+1}.png"
                wav_path = speech_save_dir / f"{i}.wav"
                if not slide_path.exists() or not wav_path.exists():
                    log.warning(f"跳过 page {i}: 缺少 {slide_path} 或 {wav_path}")
                    continue
                duration = get_audio_length(str(wav_path))
                output_name = f"page_{i:03d}.mp4"
                output_path = tmp_merage_dir / output_name
                # 无数字人：整屏就是 slide + 语音，不做 overlay；scale 仅保证宽高为偶（libx264 要求）
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-t", str(duration), "-i", str(slide_path),
                    "-i", str(wav_path),
                    "-filter_complex", "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[v]",
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-c:a", "aac", "-preset", "ultrafast", "-crf", "23",
                    "-shortest", str(output_path),
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                list_lines.append(f"file '{output_path.resolve()}'")
            list_file = tmp_merage_dir / "list.txt"
            list_file.write_text("\n".join(list_lines), encoding="utf-8")
            if list_lines:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(tmp_merage_1)],
                    check=True, capture_output=True, text=True,
                )
                log.info(f"已使用语音合并至 {tmp_merage_1}")
            else:
                raise RuntimeError("无有效 slide/语音片段可供合并")
        else:
            # 有数字人：沿用 1_merge.bash，用 talking_video 合并
            merage_cmd = [
                "/data/users/ligang/Paper2Any/dataflow_agent/toolkits/p2vtool/1_merge.bash",
                str(slide_img_dir), talking_save_dir, str(tmp_merage_dir),
                str(width), str(height), str(num_slide), str(tmp_merage_1),
                ref_img.split("/")[-1].replace(".png", ""),
            ]
            subprocess.run(merage_cmd, text=True, check=True)
        # render cursor
        cursor_size = size//6
        tmp_merage_2 = paper_output_dir /  "2_merge.mp4"
        render_video_with_cursor_from_json(video_path=tmp_merage_1, out_video_path=str(tmp_merage_2), 
                                        json_path=cursor_save_path, cursor_img_path=cursor_img_path, 
                                        transition_duration=0.1, cursor_size=cursor_size)
        # render subtitle
        font_size = size//10
        tmp_merage_3 = paper_output_dir / "video.mp4"
        add_subtitles(str(tmp_merage_2), str(tmp_merage_3), cursor_save_path, font_size)

        state.video_path = str(tmp_merage_3)
        return state

    def _stage_condition(state: Paper2VideoState):
        if state.request.script_stage:
            log.critical("进入subtitle_and_cursor stage")
            return "p2v_subtitle_and_cursor"
        else:
            log.critical("进入generate_speech stage")
            return "p2v_refine_subtitle_and_cursor"

    def _after_speech_condition(state: Paper2VideoState):
        """ref_img_path 非空时走 talking_video，否则跳过直接走 cursor。"""
        ref_img_path = state.request.ref_img_path or ""
        if ref_img_path.strip():
            log.info("ref_img_path 已设置，进入 generate_talking_video")
            return "p2v_generate_talking_video"
        log.info("ref_img_path 为空，跳过 generate_talking_video，直接进入 generate_cursor")
        return "p2v_generate_cursor"

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    nodes = {
        "_start_": lambda state: state,
        "p2v_subtitle_and_cursor": subtitle_and_cursor,
        "p2v_refine_subtitle_and_cursor": refine_subtitle_and_cursor,
        "p2v_generate_speech": generate_speech,
        "p2v_generate_talking_video": generate_talking_video,
        "p2v_generate_cursor": generate_cursor,
        "p2v_merge": merge_all,
        "_end_": lambda state: state,
    }

    # ------------------------------------------------------------------
    # EDGES  (从节点 A 指向节点 B)
    # ------------------------------------------------------------------
    edges = [
        ("p2v_subtitle_and_cursor", "_end_"),
        ("p2v_refine_subtitle_and_cursor", "p2v_generate_speech"),
        ("p2v_generate_talking_video", "p2v_generate_cursor"),
        ("p2v_generate_cursor", "p2v_merge"),
        ("p2v_merge", "_end_"),
    ]
    conditional_edges = {
        "_start_": _stage_condition,
        "p2v_generate_speech": _after_speech_condition,
    }
    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges(conditional_edges)
    return builder

if __name__ == "__main__":
    import asyncio
    graph_builder = create_paper2video_graph().build()
    
    dir_name = "ai"
    p2v_state = Paper2VideoState(
        request=Paper2VideoRequest(
            paper_pdf_path = f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai.pdf",
            ref_audio_path = f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai.wav",
            ref_text = "",
            ref_img_path = f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai.png",
            chat_api_url="http://123.129.219.111:3000/v1",
            language = "en",
        ),
        slide_timesteps_path=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/audio/slide_timesteps.json",
        ppt_path=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai.pdf",
        subtitle_and_cursor_path=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/subtitle_w_cursor.txt",
        slide_img_dir=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/slide_imgs",
        speech_save_dir=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/audio",
        talking_video_save_dir=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/talking_video",
        cursor_save_path=f"/data/users/ligang/DataFlow-Agent/outputs/{dir_name}/ai/cursor.json",
    )
    out =  asyncio.run(graph_builder.ainvoke(p2v_state))
