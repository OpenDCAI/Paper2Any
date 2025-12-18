from __future__ import annotations

from dataflow_agent.logger import get_logger
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import os
import json
from PIL import Image, ImageFont, ImageDraw
import shutil
import multiprocessing
import string
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

log = get_logger(__name__)
import re

def get_image_paths(directory_path: str) -> List[str]:
    """
    遍历指定目录及其子目录，查找所有常见的图片文件，并按照日期排序，返回它们的路径字符串列表。
    """
    # 1. 常用图片文件扩展名列表
    image_extensions = [
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.svg', '*.webp'
    ]
    
    base_path = Path(directory_path)
    if not base_path.is_dir():
        # 如果目录不存在，返回空列表并打印错误
        print(f"Error: Directory not found at {directory_path}")
        return []

    found_image_paths: List[Path] = []
    
    # 2. 递归遍历目录并收集路径
    for ext in image_extensions:
        # rglob(ext) 查找所有匹配该扩展名的文件，无论嵌套多深
        # extend() 将迭代器的所有元素添加到列表中
        found_image_paths.extend(base_path.rglob(ext))

    #3. 对找到的图片路径按照文件名日期进行排序，确保顺序
    def natural_sort_key(path: Path):
        file_name = path.name
        numbers = re.findall(r'(\d+)', file_name)
        return tuple(int(n) for n in numbers)
    
    found_image_paths.sort(key=natural_sort_key)
    return [str(p.resolve()) for p in found_image_paths]

def create_subtitle_image(text, font_size=32, font_path="arial.ttf"):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"[Warning] Failed to load font from '{font_path}': {e}")
        print("Using default font (fixed size, font_size will be ignored!)")
        font = ImageFont.load_default()

    dummy_img = Image.new("RGBA", (70, 70))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    box_w = text_w + 2*padding
    box_h = text_h + 2*padding
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 160))  # semi-transparent black

    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))

    return img

# 根据语音识别结果（带时间戳），生成对应的视频字幕片段
def generate_subtitle_clips(segments, video_w, video_h, font_size):
    clips = []
    for seg in segments:
        # fixme:这里的绝对路径是什么东西？？？？
        img = create_subtitle_image(seg["text"], font_size=font_size, font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
        img_array = np.array(img)
        clip = (ImageClip(img_array, ismask=False)
                .set_duration(seg["end"] - seg["start"])
                .set_start(seg["start"])
                .set_position(("center", video_h - font_size*2)))
        clips.append(clip)
    return clips

def add_subtitles(video_path, output_path, font_size):
    print("[Step 1] Transcribing with Whisper...")
    model = get_whisperx_model("base")
    result = model.transcribe(video_path, language="en")
    segments = result["segments"]

    print("[Step 2] Generating subtitle clips...")
    video = VideoFileClip(video_path)
    subs = generate_subtitle_clips(segments, video.w, video.h, font_size)

    print("[Step 3] Rendering final video...")
    final = CompositeVideoClip([video] + subs)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

def render_cursor_on_video(
    input_video: str,
    output_video: str,
    cursor_points: list,          # list of (time, x, y)
    transition_duration: float = 0.1,
    cursor_size: int = 10,
    cursor_img_path: str = "cursor.png"):

    img = Image.open(cursor_img_path)
    img_resized = img.resize((cursor_size, cursor_size))
    img_resized.save(cursor_img_path)


    def get_video_resolution(path):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]
        return width, height

    w, h = get_video_resolution(input_video)
    print(f"Video resolution: {w}x{h}")

    filters = []

    t_first, _, _ = cursor_points[0]
    if t_first > transition_duration:
        cx = w / 2 - cursor_size / 2
        cy = h / 2 - cursor_size / 2
        global_hold = (
            f"overlay=x={cx}:y={cy-20}:"
            f"enable='between(t,0,{round(t_first - transition_duration, 3)})'"
        )
        filters.append(global_hold)
        
    for i in range(1, len(cursor_points)):
        t0, x0, y0 = cursor_points[i - 1]
        t1, x1, y1 = cursor_points[i]

        hold_start = round(t0, 3)
        hold_end = round(t1 - transition_duration, 3)
        if hold_end > hold_start:
            x_hold = x0 - cursor_size / 2
            y_hold = y0 - cursor_size / 2
            hold_expr = (
                f"overlay=x={x_hold}:y={y_hold}:"
                f"enable='between(t,{hold_start},{hold_end})'"
            )
            filters.append(hold_expr)

        move_start = round(t1 - transition_duration, 3)
        move_end = t1
        dx = x1 - x0
        dy = y1 - y0
        x_expr = f"{x0 - cursor_size/2} + ({dx})*(t-{move_start})/{transition_duration}"
        y_expr = f"{y0 - cursor_size/2} + ({dy})*(t-{move_start})/{transition_duration}"
        move_expr = (
            f"overlay=x={x_expr}:y={y_expr}:"
            f"enable='between(t,{move_start},{move_end})'"
        )
        filters.append(move_expr)

    filter_lines = []
    stream_in = "[0][1]"
    for i, expr in enumerate(filters):
        stream_out = f"[tmp{i}]" if i < len(filters) - 1 else "[vout]"
        filter_lines.append(f"{stream_in} {expr} {stream_out}")
        stream_in = f"{stream_out}[1]"

    filter_complex = "; ".join(filter_lines)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", cursor_img_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"\n✅ Done! Output saved to: {output_video}")


def render_video_with_cursor_from_json(
    video_path,
    out_video_path,
    json_path,
    cursor_img_path,
    transition_duration=0.1,
    cursor_size=16
):
    with open(json_path, "r") as f:
        data = json.load(f)

    cursor_points = []
    for idx, slide in enumerate(data):
        if idx == 0: start_time = slide["start"]
        else: start_time = slide["start"] + 0.5
        x, y = slide["cursor"]
        cursor_points.append((start_time, x, y))
    
    render_cursor_on_video(
        input_video=video_path,
        output_video=out_video_path,
        cursor_points=cursor_points,
        transition_duration=transition_duration,
        cursor_size=cursor_size,
        cursor_img_path=cursor_img_path
    )
'''========================== 解析生成 数字人 相关的函数  =================================='''
def run_hallo2_inference(args):
    source_image, driving_audio, save_video_dir, config_path, script_path, talking_head_env = args

    audio_basename = os.path.splitext(os.path.basename(driving_audio))[0]
    save_path = os.path.join(save_video_dir, f"{audio_basename}")
    config_bak = config_path.replace(".yaml", "_{}.yaml".format(audio_basename))
    shutil.copy(config_path, config_bak)
    
    # 修改原来配置文件中的内容，因为原有文件内容中保存文件的地址不对
    updated_lines = []
    with open(config_bak, "r") as f: 
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("save_path:"): 
            updated_lines.append(f"save_path: {save_path}\n")
        else: 
            updated_lines.append(line)
    with open(config_bak, "w") as f: 
        f.writelines(updated_lines)

    cmd = [
        talking_head_env, script_path,
        "--config", config_bak,
        "--source_image", source_image,
        "--driving_audio", driving_audio,
    ]
    result = subprocess.run(cmd)

    os.remove(config_bak)
    return result

def talking_gen_per_slide(model_name, input_list, project_root, save_dir, env_path):
    save_dir = Path(save_dir)
    # fixme：这个文件究竟需不需要存在呢？？？？？
    config_path = project_root / 'hallo2/configs/inference/long.yaml'
    script_path = project_root / 'hallo2/scripts/inference_long.py'

    task_list = []
    for idx, (ref_img_path, audio_path) in enumerate(input_list):
        ref_img_path = Path(ref_img_path)
        audio_path = Path(audio_path)
        base_name_no_ext = ref_img_path.stem
        target_path = save_dir / str(idx) / base_name_no_ext / "merge_video.mp4"
        if not target_path.exists():
            task_list.append([
                str(ref_img_path), 
                str(audio_path), 
                str(save_dir), 
                str(config_path), 
                str(script_path), 
                env_path
            ])

    results = []
    for task_args in task_list:
        result = run_hallo2_inference(task_args)
        results.append(result)
    return results


def get_audio_paths(slide_audio_dir: Path):
    '''获取 slide_audio_dir 目录下的所有音频文件路径，并按数字顺序排序返回'''
    slide_audio_paths = [
        p for p in slide_audio_dir.iterdir()
        if p.is_file() and re.search(r'\d+', p.name)
    ]

    def get_sort_key(file_path: Path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group()) if match else float('inf')
    
    slide_audio_paths.sort(key=get_sort_key)
    slide_audio_paths = [str(p) for p in slide_audio_paths]
    return slide_audio_paths

def get_whisperx_model(model_name: str, device:str = "cuda"):
    '''根据 model_name 加载 whisperx 中的某个 model '''
    import whisperx
    return whisperx.load_model(model=model_name, device=device)

def load_align_whisperx_model(language_code: str = "en", device: str = "cuda"):
    '''加载 whisperx 中的 align_model '''
    import whisperx
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    return align_model, metadata

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_audio_length(audio_path):
    '''获取音频文件的总时长（秒）'''
    command = ["ffmpeg", "-i", audio_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = map(float, duration_str.split(":"))
            return hours * 3600 + minutes * 60 + seconds
    return 0 

def timesteps(subtitles, aligned_result, audio_path):
    '''为每个字幕句子分配时间戳'''
    aligned_words_in_order = []
    for idx, segment in enumerate(aligned_result["segments"]):
        aligned_words_in_order.extend(segment["words"])
    aligned_words_num = len(aligned_words_in_order) - 1
    
    result = []
    current_idx = 0
    for idx, sentence in enumerate(subtitles):
        words_num = len(re.findall(r'\b\w+\b', sentence.lower()))
        start = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]
        
        current_idx += words_num
        end = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]

        duration = {"start": start, "end": end, "text": sentence}
        result.append(duration)
    
    result[0]["start"] = 0
    result[-1]["end"] = get_audio_length(audio_path)
    return result

'''========================== 解析生成cursor位置信息相关的函数  =================================='''
def _infer_cursor(instruction, image_path):
    from transformers import pipeline
    from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code

    # fixme：修改一下这段代码，最好不要从hf上下载，而是在本地就下载好了
    pipe = pipeline("image-text-to-text", model="ByteDance-Seed/UI-TARS-1.5-7B")
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You must to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction {}".format(instruction)
    messages = [{"role": "user", "content": [{"type": "image", "url": image_path}, {"type": "text", "text": prompt}]},]
    result = pipe(text=messages)[0]
    response = result['generated_text'][1]["content"]
    
    ori_image = cv2.imread(image_path)
    original_image_width, original_image_height = ori_image.shape[:2]
    parsed_dict = parse_action_to_structure_output(
        response,
        factor=1000,
        origin_resized_height=original_image_height,
        origin_resized_width=original_image_width,
        model_type="qwen25vl"
    )

    parsed_pyautogui_code = parsing_response_to_pyautogui_code(
        responses=parsed_dict,
        image_height=original_image_height,
        image_width=original_image_width
    )

    match = re.search(r'pyautogui\.click\(([\d.]+),\s*([\d.]+)', parsed_pyautogui_code)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
    else:
        print(instruction)
    return (x, y)

def cursor_infer(args):
    import torch
    slide_idx, sentence_idx, prompt, cursor_prompt, image_path = args
    point= _infer_cursor(cursor_prompt, image_path)
    torch.cuda.empty_cache()
    result = {
        'slide': slide_idx, 'sentence': sentence_idx, 'speech_text': prompt, 
        'cursor_prompt': cursor_prompt, 'cursor': point,
    }
    return result

'''========================== 解析生成speech相关的函数  =================================='''
def parse_script(script_text):
    '''
    解析脚本的内容，将其分割成（prompt, cursor_prompt）两部分
    '''
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result

# fixme: 这里需要判断device，可能需要多加考虑
def transcribe_with_whisperx(audio_path, lang="en"):
    '''根据ref_audio生成对应的ref_text，从而在后续使用f5模型时，提供对齐文本，更好的提高最后audio的效果'''
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    import whisperx
    log.info(f"transcribe_with_whisperx 使用了 device: {device}")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16" if device == "cuda" else "int8")
    result = model.transcribe(audio_path, language=lang)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    segments = result_aligned["segments"]
    text = " ".join(seg["text"].strip() for seg in segments)
    return text

def inference_f5(text_prompt, save_path, ref_audio, ref_text):
    from f5_tts.api import F5TTS
    f5tts = F5TTS()
    f5tts.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text_prompt, file_wave=save_path, seed=None,)


def extract_beamer_code(text_str):
    match = re.search(r"(\\documentclass(?:\[[^\]]*\])?\{beamer\}.*?\\end\{document\})", text_str, re.DOTALL)
    return match.group(1) if match else None

def compile_tex(beamer_code_path: str):
    tex_path = Path(beamer_code_path).resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"Tex file {tex_path} does not exist.")
    work_dir = tex_path.parent
    try:
        # 会编译.tex文件，然后创建好一个.pdf文件
        result = subprocess.run(
            ["tectonic", str(tex_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        code_debug_result = "\n".join([result.stdout, result.stderr])
        log.info(f"Beamer 编译成功，输出结果：{code_debug_result}")
        is_beamer_warning = False
        if 'warning' in code_debug_result:
            is_beamer_warning = True
            log.info(f"Beamer 代码存在warning，需要更加完善一下")
        is_beamer_wrong = False
        return is_beamer_wrong, is_beamer_warning, code_debug_result
    except subprocess.CalledProcessError as e:
        log.info(f"Beamer 编译失败: {e.stderr}")
        is_beamer_wrong = True
        is_beamer_warning = True
        code_debug_result = e.stderr
        return is_beamer_wrong, is_beamer_warning, code_debug_result

def beamer_code_validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """检查tex是否是正确的"""
    from tempfile import TemporaryDirectory

    # 这里的 dir 具体是什么无所谓，因为我latex code中的图像路径是绝对路径
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        # 在临时目录中创建 .tex 文件
        # todo: 这里可能需要修改一下，因为在临时目录下创建文件还是不太行。
        tex_path = temp_dir / "input.tex" 
        
        raw_beamer_code = parsed_result.get("latex_code", "")
        if not raw_beamer_code:
            log.error(f"The content of beamer code is empty!")
            return False, "The content of beamer code is empty!"
        beamer_code = extract_beamer_code(raw_beamer_code)
        try:
            # 1. 写入内容
            tex_path.write_text(beamer_code, encoding='utf-8')

            result = subprocess.run(
                ["tectonic", str(tex_path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            log.info(f"Beamer代码修改完成，没有出现error")
            code_debug_result = "\n".join([result.stdout, result.stderr])
            return True, None
            
        except subprocess.CalledProcessError as e:
            code_debug_result = f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            return False, code_debug_result


def parser_beamer_latex(code: str):
    # 1. 提取 Head: 从 \documentclass 到 \begin{document} 之间的内容
    head_pattern = r'\\documentclass(?:\[[^\]]*\])?\{beamer\}(.*?)\\begin\{document\}'
    head_match = re.search(head_pattern, code, flags=re.DOTALL)
    head_content = head_match.group(1).strip() if head_match else "未找到导言区"

    # 2. 提取所有 Frame (Slides)
    # 逻辑：匹配 \begin{frame} 和 \end{frame} 之间的所有内容
    # 注意：beamer 的 frame 可能带有参数，如 \begin{frame}{标题} 或 \begin{frame}[fragile]
    frame_pattern = r'\\begin\{frame\}.*?(.*?)\\end\{frame\}'
    frames = re.findall(frame_pattern, code, flags=re.DOTALL)
    
    frames_cleaned = [f.strip() for f in frames]

    return head_content, frames_cleaned

def resize_latex_image(code):
    # 改进正则：
    # 1. 允许 width= 后面有空格
    # 2. 捕获数值后的单位（如 \textwidth, \linewidth, \columnwidth）
    pattern = r'(\\includegraphics\[[^\]]*width\s*=\s*)([\d.]+)\s*(\\[a-z]+|cm|mm|pt|in)?'
    
    def shrink_width_logic(match):
        prefix = match.group(1)
        current_val = float(match.group(2))
        unit = match.group(3) if match.group(3) else "" # 捕获单位
        
        new_val = max(0.1, current_val - 0.2)
        return f"{prefix}{new_val:.1f}{unit}"

    return re.sub(pattern, shrink_width_logic, code)