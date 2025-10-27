from json import JSONDecodeError, JSONDecoder
import re
from typing import Any, Dict
from pathlib import Path
import os
import base64
import requests
import random

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def robust_parse_json(s: str) -> dict:
    """
    Robustly parse one or more JSON objects from a string.
    Merges multiple dicts if multiple JSON objects are found.
    """
    clean = _strip_json_comments(s)
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
            idx = end
        except JSONDecodeError:
            idx += 1
    if not dicts:
        raise ValueError("No JSON object extracted from the input")
    if len(dicts) == 1:
        return dicts[0]
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


def call_gemini_image_generation(api_url, api_key, model, prompt, timeout=120):
    url = f"{api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_modalities": ["Image"],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    print("正在生成图像，请稍候...")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def extract_base64(s: str) -> str:
    s = ''.join(s.split())
    b64 = re.findall(r'[A-Za-z0-9+/=]+', s)
    return max(b64, key=len) if b64 else ''

def draw_pic_entry(img_dir, prompt:str):
    API_URL = "http://123.129.219.111:3000/v1"
    API_KEY = os.getenv("TAB_API_KEY")
    MODEL   = "gemini-2.5-flash-image-preview"
    PROMPT  = prompt

    raw = call_gemini_image_generation(API_URL, API_KEY, MODEL, PROMPT)

    if not isinstance(raw, str):
        raise RuntimeError(f"返回内容不是字符串：{type(raw)}\n内容: {raw}")

    base64_data = extract_base64(raw)

    if not base64_data:
        raise RuntimeError("没有在返回内容中找到合法的 Base64 字符串！")

    img_path = os.path.join(img_dir, ''.join(random.sample(prompt, 20)).replace(' ', '_').replace('\n', '') + '.png')

    with open(img_path, "wb") as f:
        f.write(base64.b64decode(base64_data))

    print("保存成功！文件名： ", img_path)

    return img_path