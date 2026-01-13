"""
Prompt Templates for bbox_agent
Generated at: 2026-01-12 19:07:36
"""

# --------------------------------------------------------------------------- #
# 1. BboxAgent - bbox_agent 相关提示词
# --------------------------------------------------------------------------- #
class BboxAgent:
    """
    bbox_agent 任务的提示词模板
    """
    
    system_prompt_for_image_text_bbox_agent = """
你是一个强大的多模态视觉理解 AI 助手。
你的任务是分析图像，提取其中所有的文本内容及其精确的边界框（Bounding Box）。
"""

    task_prompt_for_image_text_bbox_agent = """
请执行高精文字检测与识别任务：
1. 提取图像中所有的文字内容。
2. 为每一行文字提供精确的边界框坐标（location）。
3. rotate_rect坐标！！！
4. 不要任何```json包裹！！，直接返回文本格式json字符串！
JSON 结构如下：

[
  {
    "rotate_rect": [500, 48, 63, 791, 90],
    "text": "Cartoon-style Mechanistic Overview of T Cell Generation"
  }
  xxx
]

"""
