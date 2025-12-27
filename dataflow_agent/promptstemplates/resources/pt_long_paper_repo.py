"""
Prompt Templates for long_paper
Generated at: 2025-12-27 02:15:52
"""

# --------------------------------------------------------------------------- #
# 1. LongPaperOutlineAgent - 提示词
# --------------------------------------------------------------------------- #
class LongPaperOutlineAgent:
    """
    long_paper_outline_agent 任务的提示词模板
    """
    
    # 系统提示词：与普通 outline_agent 共享或专用
    system_prompt_for_outline_agent = """
你是一位拥有丰富学术汇报经验的PPT设计专家及大纲生成助手。你的核心任务是将一篇学术论文（或长文档的一部分）转化为一份逻辑清晰、视觉布局合理的PPT演示大纲。

请遵循以下严格规则：
1. **深度理解**：仔细阅读用户提供的文本内容，提取核心论点、实验数据和结论。
2. **视觉导向**：在规划每一页PPT时，不仅要生成文字内容，必须明确指出该页是否需要展示特定的插图（Images）或表格（Tables）。
3. **布局建议**：为每一页提供具体的布局指导（例如：左文右图、上标题下表格、两栏对比等）。
4. **格式严格**：输出必须且只能是标准的 JSON 格式数组。严禁包含 markdown 标记（如 ```json）、前言、后语或任何非 JSON 字符。
"""

    # 1. 首页 Prompt (Is First Batch)
    task_prompt_for_long_paper_outline_agent_first = """
这是长文档分批生成 PPT 的**第一批次**。
当前进度：第 {batch_index} 批 / 共 {total_batches} 批。
本批次目标页数：{pages_to_generate} 页（包括封面）。
总目标页数：{page_count} 页。

**输入数据（当前文本片段）：**
{current_chunk}

**任务要求：**
1. **第一页必须是封面**：包含 PPT 主题（Title）和汇报人信息（Presenter）。不需要额外的内容。
2. 后续页面开始进入正文介绍（如背景、引言、核心问题等）。
3.输出内容的语言为 **{language}**。
4. 不需要致谢页（除非文本很短，这是唯一一批）。

**输出格式要求（JSON Array）：**
请返回一个 JSON 数组，数组中每个对象代表一页PPT，结构如下：
- `title`: 该页PPT的标题。
- `layout_description`: 详细的版面布局描述。
- `key_points`: 一个包含多个关键要点的字符串列表（List<String>）。
- `asset_ref`: 如果该页需要展示论文中的原图或表格，请提名或路径取其文件（例如 "Table_2", "images/architecture.png"），并且只能1 个 asset；如果不需要引用原图，请填 null。

示例结构：
[
  {{
    "title": "大语言模型的幻觉问题研究",
    "layout_description": "封面设计，居中放置大号标题，下方为汇报人姓名。",
    "key_points": ["汇报人：DataFlow Agent"],
    "asset_ref": null
  }},
  {{
    "title": "研究背景",
    "layout_description": "左侧文字介绍，右侧配图。",
    "key_points": ["大模型幻觉的定义。", "当前面临的挑战。"],
    "asset_ref": "images/intro.png"
  }}
]
"""

    # 2. 中间页 Prompt (Middle Batch)
    task_prompt_for_long_paper_outline_agent_middle = """
这是长文档分批生成 PPT 的**中间批次**。
当前进度：第 {batch_index} 批 / 共 {total_batches} 批。
本批次目标页数：{pages_to_generate} 页。

**输入数据（当前文本片段）：**
{current_chunk}

**任务要求：**
1. **直接生成正文内容**：不需要封面，也不要致谢。
2. 承接上一批次的内容，继续展开当前的章节。
3. 如果文本包含新的章节标题，请作为新的一页或新章节的开始。
4. 输出内容的语言为 **{language}**。

**输出格式要求（JSON Array）：**
JSON 数组，每个对象代表一页PPT。
结构字段：`title`, `layout_description`, `key_points`, `asset_ref`。
- `asset_ref`: 如果该页需要展示论文中的原图或表格，请提名或路径取其文件（例如 "Table_2", "images/architecture.png"），并且只能1 个 asset；如果不需要引用原图，请填 null。

示例结构：
[
  {
    "title": "Methodology: Overview",
    "layout_description": "Top-down layout: brief textual overview at the top, followed by a large pipeline diagram showing the main components of the approach.",
    "key_points": ["Provide a high-level description of the proposed method.", "List the key components or stages of the approach."],
    "asset_ref": "images/method_pipeline.png"
  },
  {
    "title": "Experimental Setup",
    "layout_description": "Two-column layout: left for text describing datasets, right for a simple table.",
    "key_points": ["Mention the main datasets or benchmarks used.", "Briefly describe the experimental environment."],
    "asset_ref": "Table_1"
  }
]
"""

    # 3. 尾页 Prompt (Is Last Batch)
    task_prompt_for_long_paper_outline_agent_last = """
这是长文档分批生成 PPT 的**最后一批次**。
当前进度：第 {batch_index} 批 / 共 {total_batches} 批。
本批次目标页数：{pages_to_generate} 页（包括致谢）。

**输入数据（当前文本片段）：**
{current_chunk}

**任务要求：**
1. 生成剩余的正文内容（结论、未来展望等）。
2. **最后一页必须是致谢（Thank You）**：简短的结束语。
3.输出内容的语言为 **{language}**。

**输出格式要求（JSON Array）：**
JSON 数组，每个对象代表一页PPT。
结构字段：`title`, `layout_description`, `key_points`, `asset_ref`。
- `asset_ref`: 如果该页需要展示论文中的原图或表格，请提名或路径取其文件（例如 "Table_2", "images/architecture.png"），并且只能1 个 asset；如果不需要引用原图，请填 null。

确保最后一页是致谢页。

示例结构：
[
  {
    "title": "Conclusion",
    "layout_description": "Two-column layout: left column bullet points, right column illustrative figure.",
    "key_points": ["Summarize the main contributions.", "Highlight the effectiveness of the proposed method."],
    "asset_ref": "images/conclusion_chart.png"
  },
  {
    "title": "Thank You",
    "layout_description": "Title centered; minimal content with short closing remark and optional contact/info line.",
    "key_points": ["Thank you for your attention.", "Q&A"],
    "asset_ref": null
  }
]
"""


# --------------------------------------------------------------------------- #
# 2. ContentExpander - 提示词
# --------------------------------------------------------------------------- #
class ContentExpander:
    """
    content_expander 任务的提示词模板
    """

    system_prompt_for_content_expander = """
你是一个专业的学术写作助手和内容扩写专家。你的任务是将输入的简短文本或草稿，扩写成篇幅更长、细节更丰富、逻辑更严密的文章或报告。
你的扩写应保持专业性，增加必要的背景介绍、详细的解释、具体的例子或论证，以满足生成长篇 PPT 的内容需求。
"""

    task_prompt_for_content_expander = """
**当前任务：**
对以下文本进行第 {expansion_round} 轮扩写。

**输入文本：**
{text_content}

**扩写要求：**
1. **大幅增加篇幅**：在保持原意的前提下，通过增加细节、举例、背景分析、优缺点对比等方式，显著增加字数。
2. **结构完整**：如果输入是片段，请将其补全为完整的章节；如果输入是提纲，请将其展开为全文。
3. **保持连贯**：确保扩写后的内容逻辑通顺，段落过渡自然。
4. **输出限制**：直接输出扩写后的完整文本，不要包含任何类似于“好的，这是扩写后的内容”的废话。不要使用 Markdown 代码块包裹。
5. 如果需要表格，必须输出md表格内容，Table_1, xxx
请开始扩写：

"""


# --------------------------------------------------------------------------- #
# 3. TopicWriter - 提示词
# --------------------------------------------------------------------------- #
class TopicWriter:
    """
    topic_writer 任务的提示词模板
    用于根据 Topic 生成长篇研究报告
    """
    
    system_prompt_for_topic_writer = """
你是一位资深的学术研究员和技术写作专家。你的任务是根据给定的主题（Topic），撰写一份详细、专业、结构完整的研究报告或技术文档。

你的写作应该：
1. 内容丰富、逻辑严密、论证充分
2. 包含必要的背景介绍、核心概念、方法论、应用场景等
3. 适合用于生成长篇 PPT 演示文稿
"""
    
    task_prompt_for_topic_writer = """
**任务：** 根据以下主题生成详细的研究报告（第 {generation_round} 轮生成）

**主题：**
{text_content}

**生成要求：**
1. **语言**：使用 {language} 语言撰写
2. **篇幅**：大幅扩展内容，目标字数应达到支持 {target_pages} 页 PPT 的长度
   - 目标字符数：约 {target_chars} 字符
3. **结构**：
   - 包含完整的引言、背景、核心内容、结论等章节
   - 每个章节都要详细展开，提供具体的例子、数据、分析
4. **内容深度**：
   - 如果是第一轮生成，从主题出发构建完整框架
   - 如果是后续轮次，在现有内容基础上继续扩展和深化
   - 不要重复已有内容，而是增加新的维度和细节

**输出格式：**
1.直接输出完整的研究报告文本，不要包含任何说明性文字或 Markdown 代码块标记.
2.如果需要表格，可以输出md表格内容
"""
