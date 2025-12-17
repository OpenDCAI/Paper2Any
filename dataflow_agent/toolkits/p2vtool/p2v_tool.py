from __future__ import annotations

from dataflow_agent.logger import get_logger
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import torch

log = get_logger(__name__)
import re

def get_image_paths(directory_path: str) -> List[str]:
    """
    遍历指定目录及其子目录，查找所有常见的图片文件，并返回它们的路径字符串列表。
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

def transcribe_with_whisperx(audio_path, lang="en", device="cuda" if torch.cuda.is_available() else "cpu"):
    '''根据ref_audio生成对应的ref_text，从而在后续使用f5模型时，提供对齐文本，更好的提高最后audio的效果'''
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
