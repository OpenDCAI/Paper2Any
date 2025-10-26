import os
import requests

def call_gemini_image_generation(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout=120
):
    """
    通过 Gemini 2.5 Flash Image 生成图像
    注意：Gemini 使用 /chat/completions 端点，但返回的是图像数据
    """
    url = f"{api_url}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Gemini 图像生成的关键配置
    payload = {
        "model": model,
        "prompt": prompt,
        "response_modalities": ["Image"],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        print("正在生成图像，请稍候...")
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Gemini 返回格式可能包含 base64 图像数据
        # 需要检查 response 结构
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            
            # 检查是否有图像数据
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                
                # 可能包含 base64 图像数据
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "image" in part:
                            return part["image"]  # base64 数据
                elif isinstance(content, str):
                    return content
                    
        return data  # 返回原始数据供调试
        
    except Exception as e:
        print("调用图像生成失败:", e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("服务器响应:", e.response.json())
            except:
                print("服务器响应文本:", e.response.text)
        return None

if __name__ == "__main__":
    API_URL = "http://123.129.219.111:3000/v1"
    API_KEY = os.getenv("DF_API_KEY")
    MODEL = "gemini-2.5-flash-image-preview"
    
    prompt = "给我一个卡通的图标，蓝猫！！"
    
    result = call_gemini_image_generation(API_URL, API_KEY, MODEL, prompt)
    print("结果:", result)