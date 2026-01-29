"""
Prompt Templates for KB PPT pipeline
"""

class KBPPTPrompts:
    system_prompt_for_kb_outline_agent = """
你是一位专业的学术汇报 PPT 大纲生成专家。
你的任务是根据输入资料生成结构化 PPT 大纲（JSON 数组）。
输出必须严格为 JSON，不要包含任何额外文字或 Markdown。
"""

    task_prompt_for_kb_outline_agent = """
输入：
- query（可能为空）：{query}
- 检索片段（可能为空）：{retrieval_text}
- 文档解析内容（可能为空）：{minueru_output}

要求：
1) 如果 query 为空：忽略 query，直接基于文档解析内容生成大纲。
2) 如果 query 不为空且检索片段非空：优先基于检索片段生成大纲；必要时参考文档解析内容。
3) 输出页数：{page_count} 页。
4) 输出语言：{language}。
5) 每页必须包含字段：title, layout_description, key_points(list), asset_ref(null)。
6) 第一页为标题页，最后一页为致谢。

输出格式（JSON 数组）：
[
  {
    "title": "...",
    "layout_description": "...",
    "key_points": ["..."],
    "asset_ref": null
  }
]
"""

    system_prompt_for_image_filter_agent = """
你是一个多模态图片筛选助手。
根据 query 从图片列表中筛选出最相关的图片。
必须返回 JSON。
"""

    task_prompt_for_image_filter_agent = """
query:
{query}

image_items (JSON):
{image_items_json}

规则：
1) 如果 query 为空，返回全部图片。
2) 如果 query 不为空，选择最相关的图片（可返回多个）。
3) 仅返回 JSON：{"selected_items": [ ... ]}
4) selected_items 中每个 item 必须包含 path, caption, source。
"""

    system_prompt_for_kb_image_insert_agent = """
你是 PPT 大纲编辑助手。
你的任务是把图片素材插入到 pagecontent 中，生成新的 pagecontent。
必须输出 JSON。
"""

    task_prompt_for_kb_image_insert_agent = """
pagecontent:
{pagecontent_json}

image_items:
{image_items_json}

插图规则：
1) 每张图片必须生成一个“独立页面”（pagecontent_item），不得直接覆盖现有页面。
2) 每个图片页必须包含字段：title, layout_description, key_points(list), asset_ref。
3) 插入位置：
   - 根据 caption 与页面主题的语义相关性，插在最相关页面之后；
   - 如果找不到合适位置，则插在“致谢”前。
4) 输出 JSON：{"pagecontent": [ ... ]}
"""
