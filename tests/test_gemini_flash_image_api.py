#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试 Nano Banana 正式版 (gemini-2.5-flash-image) 图像生成 API

端点：
  https://api.apiyi.com/v1beta/models/gemini-2.5-flash-image:generateContent

运行方式：
  cd /home/ubuntu/liuzhou/myproj/dev/DataFlow-Agent
  python tests/test_gemini_flash_image_api.py
"""

import base64
import os
from datetime import datetime

import requests

# ==========================
# 配置区域
# ==========================

API_KEY = ""
API_URL = "https://api.apiyi.com/v1beta/models/gemini-2.5-flash-image:generateContent"

# 提示词
PROMPT = "一只可爱的小猫坐在花园里，油画风格，高清，细节丰富"

# 宽高比：21:9, 16:9, 4:3, 3:2, 1:1, 9:16, 3:4, 2:3, 5:4, 4:5
ASPECT_RATIO = "16:9"

# 输出目录与文件名
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, f"gemini_flash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
)

# 请求超时时间（秒）
TIMEOUT = 300  # 5 分钟


def build_payload(prompt: str, aspect_ratio: str) -> dict:
    """构造请求 payload（谷歌原生格式）"""
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio
                # 老的 gemini-2.5-flash-image 不一定支持 imageSize，这里不传
            }
        }
    }


def save_base64_image(image_base64: str, output_file: str) -> str:
    """保存 base64 图片到本地，返回最终文件路径"""
    image_bytes = base64.b64decode(image_base64)
    with open(output_file, "wb") as f:
        f.write(image_bytes)
    return output_file


def test_gemini_flash_image_generate() -> None:
    """主测试函数：调用 API 并保存图片"""

    print("=" * 60)
    print("测试 Nano Banana (gemini-2.5-flash-image) 图像生成 API")
    print("=" * 60)
    print(f"API_URL      : {API_URL}")
    print(f"PROMPT       : {PROMPT}")
    print(f"ASPECT_RATIO : {ASPECT_RATIO}")
    print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
    print("-" * 60)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = build_payload(PROMPT, ASPECT_RATIO)

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return

    print(f"HTTP 状态码: {resp.status_code}")
    if resp.status_code != 200:
        print("❌ API 返回非 200 状态码")
        try:
            print("响应内容:", resp.text[:1000])
        except Exception:
            pass
        return

    try:
        data = resp.json()
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}")
        print("原始响应内容（截断）:", resp.text[:1000])
        return

    # 简单打印一下 usage 信息，方便调试
    usage = data.get("usageMetadata") or data.get("usage") or {}
    if usage:
        print("usage:", usage)

    try:
        # 典型返回结构：
        # data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        candidates = data["candidates"]
        if not candidates:
            print("❌ candidates 为空")
            print("完整响应:", data)
            return

        content = candidates[0]["content"]
        parts = content["parts"]
        inline_data = parts[0]["inlineData"]
        image_base64 = inline_data["data"]
    except Exception as e:
        print(f"❌ 解析图片数据失败: {e}")
        print("响应结构可能变化，完整响应如下（截断）:")
        print(str(data)[:2000])
        return

    # 保存图片
    try:
        output_path = save_base64_image(image_base64, OUTPUT_FILE)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"✅ 图片已保存: {output_path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        return

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_gemini_flash_image_generate()
