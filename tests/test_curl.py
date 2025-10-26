import os
import re
import base64
import requests

def call_gemini_image_generation(api_url, api_key, model, prompt, timeout=120):
    url = f"{api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_modalities": ["Image"],   # 关键：只要图片
        "temperature": 0.7,
        "max_tokens": 1024
    }
    print("正在生成图像，请稍候...")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]   # 直接返回 content

def extract_base64(s: str) -> str:
    """
    从任意字符串中提取连续的 Base64 字符串（安全写法）
    """
    # 把所有换行、空白干掉
    s = ''.join(s.split())
    # 只保留真正的 Base64 字符
    b64 = re.findall(r'[A-Za-z0-9+/=]+', s)
    # 可能匹配出多段，通常最后一段长度最长就是图像
    return max(b64, key=len) if b64 else ''

if __name__ == "__main__":
    API_URL = "http://123.129.219.111:3000/v1"
    API_KEY = os.getenv("DF_API_KEY")
    MODEL   = "gemini-2.5-flash-image-preview"
    PROMPT  = "给我一个卡通的图标，蓝猫！！"

    raw = call_gemini_image_generation(API_URL, API_KEY, MODEL, PROMPT)

    if not isinstance(raw, str):
        # 万一接口结构变了
        raise RuntimeError(f"返回内容不是字符串：{type(raw)}\n内容: {raw}")

    base64_data = extract_base64(raw)

    if not base64_data:
        raise RuntimeError("没有在返回内容中找到合法的 Base64 字符串！")

    # 解码并保存
    with open("./tests/cat_icon.png", "wb") as f:
        f.write(base64.b64decode(base64_data))

    print("保存成功！文件名：cat_icon.png")