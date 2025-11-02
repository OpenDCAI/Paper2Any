import ast
from json import JSONDecodeError, JSONDecoder
import json
import re
from typing import Any, Dict, Union, List
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def robust_parse_json(
        s: str,
        merge_dicts: bool = False,          # 想合并时显式打开
        strip_double_braces: bool = False   # 可选：把 {{ }} → { }
) -> Union[Dict[str, Any], List[Any]]:
    """
    既能解析普通 JSON，也能从混杂文本中提取多个对象。
    
    - 支持 // 和 /* */ 注释、尾逗号
    - 自动去除 Markdown 代码块 ```json … ```、三引号 ''' … '''
    - 默认返回 dict / list 的原始结构
    - 若提取到多个独立对象，可用 merge_dicts=True 合并
    """

    # ---------- 预处理 ----------
    # 1) 去掉 ```xxx 代码围栏
    s = re.sub(r'```[\w]*\s*', '', s)      # 开始围栏
    s = re.sub(r'```', '', s)              # 结束围栏
    # 2) 去掉成对的三引号 ''' 或 """
    s = re.sub(r"^'''|'''$|^\"\"\"|\"\"\"$", '', s.strip())
    # 3) 可选：{{ }} → { }
    if strip_double_braces:
        s = s.replace('{{', '{').replace('}}', '}')
    # 4) 去注释 + 尾逗号
    s = _strip_json_comments(s)

    # ---------- 步骤 1：整体解析 ----------
    try:
        return json.loads(s)              
    except JSONDecodeError:
        pass                              

    # ---------- 步骤 2：提取多个对象 ----------
    objs: List[Any] = _extract_json_objects(s)
    if not objs:
        raise ValueError("No valid JSON found.")

    # 单个对象
    if len(objs) == 1:
        return objs[0]

    # 多对象：根据参数决定
    if merge_dicts and all(isinstance(o, dict) for o in objs):
        merged: Dict[str, Any] = {}
        for o in objs:
            merged.update(o)
        return merged
    return objs

def _strip_json_comments(s: str) -> str:
    # s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)          # 块注释
    # s = re.sub(r'//.*?$',    '', s, flags=re.M)          # 行注释
    s = re.sub(r',\s*([}\]])', r'\1', s)                 # 尾逗号
    return s

def _extract_json_objects(s: str) -> List[Any]:
    dec = JSONDecoder()
    idx, n = 0, len(s)
    objs = []
    while idx < n:
        # 找下一个 { 或 [
        m = re.search(r'[{\[]', s[idx:])
        if not m:
            break
        idx += m.start()
        try:
            obj, end = dec.raw_decode(s, idx)
            objs.append(obj)
            idx = end
        except JSONDecodeError:
            idx += 1
    return objs
