# dataflow_agent/toolkits/imtool/req_img.py
import httpx
import base64
import re

async def call_gemini_image_generation_async(api_url, api_key, model, prompt, timeout=120):
    url = f"{api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # 修改 payload 格式，去掉或调整 response_modalities
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # 尝试这些不同的参数组合：
        "response_format": {"type": "image"},  # 选项1
        # 或者完全去掉 response_modalities，只用 model 控制
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    
    # 打印请求信息用于调试
    print(f"请求 URL: {url}")
    print(f"请求 payload: {payload}")
    
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        http2=False
    ) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            print(f"响应状态码: {resp.status_code}")
            print(f"响应内容: {resp.text[:500]}")  # 打印前500字符
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {e.response.text}")
            raise

def extract_base64(s: str) -> str:
    s = ''.join(s.split())
    matches = re.findall(r'[A-Za-z0-9+/=]+', s)
    return max(matches, key=len) if matches else ''

async def generate_and_save_image_async(
    prompt: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: int = 120
) -> str:
    raw = await call_gemini_image_generation_async(api_url, api_key, model, prompt, timeout)
    b64 = extract_base64(raw)
    if not b64:
        raise RuntimeError(f"未找到 Base64 字符串，原始内容: {raw[:200]}")
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return b64