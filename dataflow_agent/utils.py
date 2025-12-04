import ast
from json import JSONDecodeError, JSONDecoder
import json
import re
from typing import Any, Dict, Union, List
from pathlib import Path
from dataflow_agent.logger import get_logger
log = get_logger(__name__)
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def robust_parse_json(
    text: str,
    *,
    merge_dicts: bool = False,
    strip_double_braces: bool = False
) -> Union[Dict[str, Any], List[Any]]:
    """
    尽量从 LLM / 日志 / jsonl / Markdown 片段中提取合法 JSON。

    参数
    ----
    text : str
        输入原始文本
    merge_dicts : bool, default False
        提取到多个对象且全部是 dict 时，是否用 dict.update 合并返回
    strip_double_braces : bool, default False
        把 '{{' / '}}' 替换成 '{' / '}'（某些模板语言会加双层花括号）

    返回
    ----
    Dict / List / List[Dict | List]
    """
    s = text.strip()

    # ---------- 预处理：剥去外层包裹 ----------
    s = _remove_markdown_fence(s)          # ```json ... ```
    s = _remove_outer_triple_quotes(s)     # ''' ... ''' / """ ... """
    s = _remove_leading_json_word(s)       # 开头一个 json/JSON 标记

    if strip_double_braces:
        s = s.replace("{{", "{").replace("}}", "}")

    # ---------- 清理注释 & 尾逗号 ----------
    s = _strip_json_comments(s)

    # ---------- 新增：清理非法控制字符 ----------
    # 移除所有 JSON 规范不允许的 ASCII 控制字符。
    # 合法的 \n, \r, \t, 和 \f, \b, \" 都不会被移除，但这里只针对不可打印的控制码。
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

    log.debug(f'清洗完之后内容是： {s}')

    # ---------- Step-1：整体解析 ----------
    # Step-1
    try:
        result = json.loads(s)
        log.info(f"整体解析成功，类型: {type(result)}")
        return result
    except JSONDecodeError as e:
        log.warning(f"整体解析失败: {e}")

    # ---------- Step-2：尝试 JSON Lines ----------
    objs = _parse_json_lines(s)
    if objs is not None:
        return _maybe_merge(objs, merge_dicts)

    # ---------- Step-3：流式提取多个对象 ----------
    objs = _extract_json_objects(s)
    log.warning(f"提取到 {len(objs)} 个对象")
    if not objs:
        raise ValueError("Unable to locate any valid JSON fragment.")

    return _maybe_merge(objs, merge_dicts)


# ======================================================================
#                            工具函数
# ======================================================================

_fence_pat = re.compile(r'```[\w-]*\s*([\s\S]*?)```', re.I)


def _remove_markdown_fence(src: str) -> str:
    """提取 ``` … ``` 内文本；若没找到则原样返回"""
    blocks = _fence_pat.findall(src)
    return "\n".join(blocks).strip() if blocks else src


def _remove_outer_triple_quotes(src: str) -> str:
    if (src.startswith("'''") and src.endswith("'''")) or (
        src.startswith('"""') and src.endswith('"""')
    ):
        return src[3:-3].strip()
    return src


def _remove_leading_json_word(src: str) -> str:
    return src[4:].lstrip() if src.lower().startswith("json") else src


def _strip_json_comments(src: str) -> str:
    # /* ... */  块注释
    src = re.sub(r'/\*[\s\S]*?\*/', '', src)
    # // ...     行注释
    src = re.sub(r'//.*', '', src)
    # 尾逗号 ,}
    src = re.sub(r',\s*([}\]])', r'\1', src)
    return src.strip()


# ----------------  JSON Lines ----------------
def _parse_json_lines(src: str) -> Union[List[Any], None]:
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if len(lines) <= 1:          # 只有 0/1 行就不是 jsonl
        return None

    objs: List[Any] = []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except JSONDecodeError:
            return None  # 某行不是合法 JSON，放弃 jsonl 方案
    return objs


# ------------  多对象提取（改进版） ------------
def _extract_json_objects(src: str) -> List[Any]:
    dec = JSONDecoder()
    idx, n = 0, len(src)
    objs: List[Any] = []

    while idx < n:
        m = re.search(r'[{\[]', src[idx:])
        if not m:
            break
        idx += m.start()
        try:
            obj, end = dec.raw_decode(src, idx)
            # ========== 严格性检查 ==========
            tail = src[end:].lstrip()
            # 允许结束、逗号、换行、右括号、右中括号
            if tail and tail[0] not in ',]}>\n\r':
                idx += 1  # 可能是误判，如  {"a":1  <-- 缺 }
                continue
            objs.append(obj)
            idx = end
        except JSONDecodeError:
            idx += 1
    return objs


def _maybe_merge(objs: List[Any], merge_dicts: bool) -> Union[Any, List[Any]]:
    if len(objs) == 1:
        return objs[0]
    if merge_dicts and all(isinstance(o, dict) for o in objs):
        merged: Dict[str, Any] = {}
        for o in objs:
            merged.update(o)
        return merged
    return objs

# def robust_parse_json(
#         s: str,
#         merge_dicts: bool = False,          # 想合并时显式打开
#         strip_double_braces: bool = False   # 可选：把 {{ }} → { }
# ) -> Union[Dict[str, Any], List[Any]]:
#     """
#     既能解析普通 JSON，也能从混杂文本中提取多个对象。
    
#     - 支持 // 和 /* */ 注释、尾逗号
#     - 自动去除 Markdown 代码块 ```json … ```、三引号 ''' … '''
#     - 默认返回 dict / list 的原始结构
#     - 若提取到多个独立对象，可用 merge_dicts=True 合并
#     """

#     # ---------- 预处理 ----------
#     # 1) 去掉 ```xxx 代码围栏
#     s = re.sub(r'```[\w]*\s*', '', s)      # 开始围栏
#     s = re.sub(r'```', '', s)              # 结束围栏
#     # 2) 去掉成对的三引号 ''' 或 """
#     s = re.sub(r"^'''|'''$|^\"\"\"|\"\"\"$", '', s.strip())
#     # 3) 可选：{{ }} → { }
#     if strip_double_braces:
#         s = s.replace('{{', '{').replace('}}', '}')
#     # 4) 去注释 + 尾逗号
#     s = _strip_json_comments(s)

#     # ---------- 步骤 1：整体解析 ----------
#     try:
#         return json.loads(s)              
#     except JSONDecodeError:
#         pass                              

#     # ---------- 步骤 2：提取多个对象 ----------
#     objs: List[Any] = _extract_json_objects(s)
#     if not objs:
#         raise ValueError("No valid JSON found.")

#     # 单个对象
#     if len(objs) == 1:
#         return objs[0]

#     # 多对象：根据参数决定
#     if merge_dicts and all(isinstance(o, dict) for o in objs):
#         merged: Dict[str, Any] = {}
#         for o in objs:
#             merged.update(o)
#         return merged
#     return objs

# def _strip_json_comments(s: str) -> str:
#     # s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)          # 块注释
#     # s = re.sub(r'//.*?$',    '', s, flags=re.M)          # 行注释
#     s = re.sub(r',\s*([}\]])', r'\1', s)                 # 尾逗号
#     return s

# def _extract_json_objects(s: str) -> List[Any]:
#     dec = JSONDecoder()
#     idx, n = 0, len(s)
#     objs = []
#     while idx < n:
#         # 找下一个 { 或 [
#         m = re.search(r'[{\[]', s[idx:])
#         if not m:
#             break
#         idx += m.start()
#         try:
#             obj, end = dec.raw_decode(s, idx)
#             objs.append(obj)
#             idx = end
#         except JSONDecodeError:
#             idx += 1
#     return objs
