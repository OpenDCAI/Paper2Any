import os
import base64
import re
from openai import OpenAI
from typing import Tuple


class SimpleGeminiVLM:
    """
    精简版 Gemini VLM 客户端，提取自 APIVLMServing_openai 核心逻辑
    """
    
    def __init__(
        self,
        api_url: str = "http://123.129.219.111:3000/v1/",
        api_key: str = None,
        model_name: str = "gemini-2.5-flash-image-preview",
        send_request_stream: bool = True,
        timeout: int = 1800
    ):
        """
        初始化客户端
        
        :param api_url: API 地址
        :param api_key: API Key (为 None 时从环境变量 DF_API_KEY 读取)
        :param model_name: 模型名称
        :param send_request_stream: 是否启用流式响应（gemini-2.5-flash-image-preview 必须为 True）
        :param timeout: 请求超时时间（秒）
        """
        if api_key is None:
            api_key = os.environ.get("DF_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing API key in environment variable 'DF_API_KEY'")
        
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model_name = model_name
        self.send_request_stream = send_request_stream
        self.timeout = timeout

    def _encode_image_to_base64(self, image_path: str) -> Tuple[str, str]:
        """
        将图像编码为 base64（从原代码提取）
        
        :param image_path: 图像路径
        :return: (base64字符串, 图像格式)
        """
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        ext = image_path.rsplit('.', 1)[-1].lower()

        if ext in ['jpg', 'jpeg']:
            fmt = 'jpeg'
        elif ext == 'png':
            fmt = 'png'
        else:
            raise ValueError(f"Unsupported image format: {ext}")

        return b64, fmt

    def generate(self, image_path: str, prompt: str) -> str:
        """
        发送图像和提示词，获取模型响应
        
        :param image_path: 图像文件路径
        :param prompt: 文本提示词
        :return: 模型生成的文本响应
        """
        # 1. 编码图像（对应原代码 _encode_image_to_base64）
        b64, fmt = self._encode_image_to_base64(image_path)
        
        # 2. 构建消息（对应原代码 _create_messages）
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
        ]
        messages = [{"role": "user", "content": content}]
        
        # 3. 发送请求（对应原代码 _send_chat_request）
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=self.timeout,
            stream=self.send_request_stream
        )

        # print(f"resp: {resp.choices[0]}")
        
        # 4. 处理响应（gemini-2.5-flash-image-preview 的特殊逻辑）
        if self.model_name == "gemini-2.5-flash-image-preview":
            if self.send_request_stream:
                full_content = ""
                for chunk in resp:
                    if chunk.choices[0].delta.content is not None and chunk.choices[0].delta.content != "":
                        # 原代码注释：utilize the final response
                        full_content = chunk.choices[0].delta.content
            else:
                full_content = resp.choices[0].delta.content
            
            print(f"full_content: {full_content}")
            return full_content
        
        # 其他模型的处理
        return resp.choices[0].message.content


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设置 API Key
    # os.environ['DF_API_KEY'] = "your-api-key-here"
    
    # 初始化客户端（默认参数已配置好）
    client = SimpleGeminiVLM()
    
    # 调用
    result = client.generate(
        image_path="/mnt/DataFlow/lz/proj/DataFlow-Agent/tests/cat_icon.png",
        prompt="改成赛博朋克风格！！"
    )
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
    
    base64data = extract_base64(result)
    if not base64data:
        raise RuntimeError("没有在返回内容中找到合法的 Base64 字符串！")

    # 解码并保存
    with open("./cat_icon.png", "wb") as f:
        f.write(base64.b64decode(base64data))

    print("保存成功！文件名：cat_icon.png")
    