from __future__ import annotations

from dataflow_agent.logger import get_logger
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

log = get_logger(__name__)
import re

def extract_beamer_code(text_str):
    match = re.search(r"(\\documentclass(?:\[[^\]]*\])?\{beamer\}.*?\\end\{document\})", text_str, re.DOTALL)
    return match.group(1) if match else None

def compile_tex(beamer_code_path: str):
    tex_path = Path(beamer_code_path).resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"Tex file {tex_path} does not exist.")
    work_dir = tex_path.parent
    try:
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
