from json import JSONDecodeError, JSONDecoder
import json
import re
from typing import Any, Dict
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def robust_parse_json(s: str) -> dict:
    """
    Robustly parse one or more JSON objects from a string.
    Merges multiple dicts if multiple JSON objects are found.
    """
    clean = _strip_json_comments(s)
    
    # 1. 先尝试直接解析整个字符串（最常见情况）
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except JSONDecodeError:
        pass
    
    # 2. 失败后再尝试提取多个独立的 JSON 对象
    decoder = JSONDecoder()
    idx = 0
    dicts = []
    length = len(clean)
    
    while True:
        idx = clean.find('{', idx)
        if idx < 0 or idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(clean, idx)
            if isinstance(obj, dict):
                dicts.append(obj)
                idx = end  # 跳到这个对象结束的位置
        except JSONDecodeError:
            idx += 1
    
    if not dicts:
        raise ValueError("No JSON object extracted from the input")
    
    if len(dicts) == 1:
        return dicts[0]
    
    # 3. 只在真正有多个独立对象时才合并
    merged: Dict[str, Any] = {}
    for d in dicts:
        merged.update(d)
    return merged

def _strip_json_comments(s: str) -> str:
        """
        Remove block and line comments, and trailing commas from JSON-like strings.
        """
        # /*  ...  */
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
        # // ...   （仅限行首或前面只有空白）
        s = re.sub(r'^\s*//.*$', '', s, flags=re.MULTILINE)
        # 尾逗号  ,}
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s