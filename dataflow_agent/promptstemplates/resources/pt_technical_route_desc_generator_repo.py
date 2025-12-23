"""
Prompt Templates for technical_route_desc_generator
Generated at: 2025-12-08 01:19:09
"""

# --------------------------------------------------------------------------- #
# 1. TechnicalRouteDescGenerator - technical_route_desc_generator 相关提示词
# --------------------------------------------------------------------------- #
class TechnicalRouteDescGenerator:
    system_prompt_for_figure_desc_generator_free =  """
你是一位世界顶级的 CVPR/NeurIPS 视觉架构师
你的核心能力是将晦涩难懂的论文逻辑，转化为**具体的、画面感极强的视觉描述。
"""

    task_prompt_for_figure_desc_generator_free ="""
下面是一篇论文的核心研究内容（paper_idea）：

{paper_idea}

请根据上述内容，编写一个用于 Text-to-Image 模型的英文提示词（Prompt）。

### 提示词编写策略：
1. 强调 “科研绘图”，用于论文插图；
2. **风格（Style）**：{style}
   - 必须强制包含论文内容的关键词.
3. 白色背景，然后要分成多个panel，就跟论文中的图一样，每个panel都要有自己的标题，标题要放在panel的上方；
4. 信息量要丰富，填满整个画面；

### 最终生成的 fig_desc 必须是一段连贯的英文描述;

# Output Format (The Golden Schema)
请严格遵守以下 JSON 输出要求：

1. 最终响应必须是一个严格合法的 JSON 对象，不能包含任何额外文字、解释或 Markdown 标记。
2. 该 JSON 对象只能包含一个键：fig_desc。
3. fig_desc 的值必须是一个字符串，用于描述整张图的视觉结构和内容。
4. 在 JSON 中：
   - 所有双引号必须写成 \\"；
   - 所有换行必须写成 \\n（不能直接换行输出）；
   - 不要包含制表符或其它控制字符。

示例（仅示意结构，实际内容请根据论文生成）：
{
  "fig_desc": "xxx"
}


"""


    system_prompt_for_figure_desc_generator_mid = """
# Role
你是一位 CVPR/NeurIPS 顶刊的视觉架构师。你的核心能力是将抽象的论文逻辑转化为具体的、结构化的、可直接用于绘图模型的视觉指令。

# Objective
阅读我提供的论文内容，输出一份 [VISUAL SCHEMA]。这份 Schema 将被直接发送给 AI 绘图模型，因此必须使用清晰的物理描述。

# Phase 1: Layout Strategy Selector (关键步骤：布局决策)
在生成 Schema 之前，请先分析论文逻辑，从以下布局原型中选择最合适的一个（或组合）：
1. Linear Pipeline: 左→右流向 (适合 Data Processing, Encoding-Decoding)。
2. Cyclic/Iterative: 中心包含循环箭头 (适合 Optimization, RL, Feedback Loops)。
3. Hierarchical Stack: 上→下或下→上堆叠 (适合 Multiscale features, Tree structures)。
4. Parallel/Dual-Stream: 上下平行的双流结构 (适合 Multi-modal fusion, Contrastive Learning)。
5. Central Hub: 一个核心模块连接四周组件 (适合 Agent-Environment, Knowledge Graphs)。

# Phase 2: Schema Generation Rules
1. Dynamic Zoning: 根据选择的布局，定义 2-5 个物理区域 (Zones)。
2. Internal Visualization: 必须定义每个区域内部的“物体”（图标、网格、树等），禁止仅使用抽象概念。
3. Explicit Connections: 如果是循环过程，必须明确描述 "Curved arrow looping back from Zone X to Zone Y" 之类的连接。

# Output Format (The Golden Schema)
请严格遵守以下 JSON 输出要求：

1. 最终响应必须是一个严格合法的 JSON 对象，不能包含任何额外文字、解释或 Markdown 标记。
2. 该 JSON 对象只能包含一个键：fig_desc。
3. fig_desc 的值必须是一个字符串，用于描述整张图的视觉结构和内容。
4. 在 JSON 中：
   - 所有双引号必须写成 \\"；
   - 所有换行必须写成 \\n（不能直接换行输出）；
   - 不要包含制表符或其它控制字符。

示例（仅示意结构，实际内容请根据论文生成）：
{
  "fig_desc": "[Style & Meta-Instructions] ... \\n[LAYOUT CONFIGURATION] ... \\n[ZONE 1: LOCATION - ...] ... \\n[CONNECTIONS] ..."
}

在 fig_desc 字符串中，建议按照如下区块依次描述：
[Style & Meta-Instructions]
[LAYOUT CONFIGURATION]
[ZONE 1: LOCATION - LABEL]
[ZONE 2: LOCATION - LABEL]
[ZONE 3: LOCATION - LABEL]
[CONNECTIONS]

# Input Data

paper_idea
"""

    task_prompt_for_figure_desc_generator_mid = """
**Style Reference & Execution Instructions:**

1. Art Style (Visio/Illustrator Aesthetic):
   Generate a professional academic architecture diagram suitable for a top-tier computer science paper (CVPR/NeurIPS).
   - Visuals: Flat vector graphics, distinct geometric shapes, clean thin outlines, and soft pastel fills (Azure Blue, Slate Grey, Coral Orange).
   - Layout: Strictly follow the spatial arrangement defined below.
   - Vibe: Technical, precise, clean white background. NOT hand-drawn, NOT photorealistic, NOT 3D render, NO shadows/shading.

2. CRITICAL TEXT CONSTRAINTS (Read Carefully):
   - DO NOT render meta-labels: Do not write words like "ZONE 1", "LAYOUT CONFIGURATION", "Input", "Output", or "Container" inside the image. These are structural instructions for YOU, not text for the image.
   - ONLY render "Key Text Labels": Only text inside double quotes (e.g., "[Text]") listed under "Key Text Labels" should appear in the diagram.
   - Font: Use a clean, bold Sans-Serif font (like Roboto or Helvetica) for all labels.

论文内容（paper_idea）如下：

{paper_idea}

要求提示词一定满足：
1. 信息丰富，信息量大；
2. 科研绘图，白色背景；
3. {style} 风格提示词；
4. 最重要，生成提示词要写入：“生成的文字都要在Icon旁边，不能覆盖Icon！！！！！”

请基于上述论文内容和风格要求，设计对应的视觉架构指令，并按照系统提示中的 JSON 规范，仅输出一个 JSON 对象：
- 该对象只包含一个键：fig_desc；
- fig_desc 的值为完整的视觉描述字符串；
- 保证整个响应是严格合法的 JSON（双引号使用 \\" 转义，换行使用 \\n 转义），不要输出任何多余文本、注释或 Markdown 标记。
"""

    # 用户/任务层提示：描述输入是什么 + 要求生成“复杂、美观、箭头明显”的技术路线图 SVG

    task_prompt_for_technical_route_desc_generator = """
下面是一个论文的研究内容（paper_idea）：

{paper_idea}

请根据该想法设计一份技术路线图，并用 SVG 代码进行表示。

整体要求（重要）：
1. 技术路线图需要包括关键步骤/模块及其先后关系，**建议划分为 3~5 个清晰的阶段**，每个阶段内部可以包含 3~6 个节点，使整体结构**信息量丰富但有条理**。
2. 每个步骤使用风格统一的节点形状（推荐圆角矩形），可以适度使用少量其他形状（如椭圆）突出起点/终点或关键模块，但整体视觉语言要统一。
3. 流程连接必须使用**线条较粗、颜色对比明显的箭头**（可以是直线或略带弧度的 path），箭头头部要清晰可见，确保方向一眼可辨；允许存在分支和汇合。
4. 布局建议采用自左向右或自上而下的多行/多列结构，可以通过阶段分组（背景块或分区标题）表现整体流程的层次感，使图看起来**结构清晰、相对复杂且完整**，但不要杂乱无章。
5. 颜色风格要**美观、现代**：可以区分不同阶段或节点类别，适度使用渐变、阴影或圆角等效果增强观感，但要注意整体协调，避免刺眼的高饱和颜色充斥全图。
6. 整体要在“信息量丰富、结构清晰”和“视觉美观”之间取得平衡，使得技术路线看起来**专业、完整，而不是极简草图**。

关于文字排布（非常重要）：
1. 可以将简短的步骤名称放在节点内部（例如 1~4 个词），也可以在节点外侧（上/下/左/右）放置说明文字；两种方式可以结合使用，但要尽量保持同一层级的节点风格一致。
2. 避免超长句子塞在一个节点中，尽量用简短短语或关键词表达（例如 “数据预处理”“特征工程”“模型训练”“消融实验”等）。
3. 阶段标题可以使用比节点文字略大的字号，放在对应分区上方或左侧，强化层次结构。

SVG 复杂度与风格（非常重要）：
1. 整体元素数量可以相对较多：包含多阶段背景块、若干节点、较多箭头和必要的装饰线条，以呈现出清晰而**相对复杂**的技术流程。
2. 可以适度使用渐变、圆角、阴影、背景分区等视觉元素，使路线图在 PPT 中看起来更加专业、美观。
3. 箭头的线宽应略粗于节点边框线宽，颜色可以采用与背景区分度较高的色彩，以保证“箭头非常明显、方向一眼可见”。
4. 可以使用 <g> 分组对不同阶段、不同类型节点进行逻辑归类，便于整体调整和复用样式。

SVG 技术要求：
- SVG 以 <svg> 根节点开始，并包含必要的 width、height 和 viewBox 属性。
- 整体风格要统一，适合作为论文技术路线图，最终会被插入 PPT 展示。
- 尺寸改成“基于 viewBox 的自适应”，别写死 width/height 像素

风格要求：
- 满足： {style} 风格；

svg代码的text要求： {lang} 语言!!!!

请只根据上述 paper_idea 和要求进行设计，具体 SVG 输出规范见系统提示。
"""

    # 系统层提示：严格约束输出为 {"svg_code": "xxx"}，并强调“复杂、美观、箭头明显”
    system_prompt_for_technical_route_desc_generator = """
你是一个技术路线图设计助手。你的任务是：

1. 从用户提供的论文研究想法（paper_idea）中抽取关键技术步骤、阶段和模块之间的依赖关系。
2. 结合用户在任务提示中提供的整体风格描述（style），设计一个结构清晰、信息量相对丰富、视觉上美观的技术路线图。
3. 使用 SVG 代码来表示该路线图，要求节点层次分明、阶段划分清楚、箭头粗细和颜色足够明显，使流程方向一眼可见，适合直接用于 PPT 展示。

输出格式要求（非常重要）：
- 你必须仅输出一个严格的 JSON 对象，形如：
  {"svg_code": "<svg ...>...</svg>"}
- 不要输出任何额外文字、注释、解释或 markdown 代码块标记。
- JSON 中只能有一个键：svg_code。
- svg_code 的值是完整的 SVG 源代码字符串：
  - 以 <svg ...> 开始，以 </svg> 结束。
  - 包含 width, height, viewBox 等基本属性。
  - 所有双引号必须正确转义，以保证整个 JSON 可被标准 JSON 解析器解析。
  - 换行可以使用 \\n 进行转义。

SVG 内容设计规范（在不影响 JSON 解析的前提下，兼顾复杂度、美观和箭头可读性）：
- 元素与布局：
  - 以圆角矩形等简单图形作为主要步骤节点，可以配合阶段背景块和少量其他形状体现层次结构。
  - 使用 <line> 或 <path> 表示箭头，线条应相对粗一些，并带有清晰的箭头头部（可通过 marker 或简单三角形 path 实现），确保“箭头非常明显”。
  - 支持多阶段（3~5 阶段）布局，节点数量可以适度偏多，但要通过合理的对齐和间距保持整体清晰。

- 颜色与风格：
  - 如果是卡通风格，色系用浅色系列，但是文字要深色；
  - 如果是写实风格，颜色要深一些，以突出重点，多用灰白色；字体黑色；
  - 箭头颜色和节点边框，以及文字 颜色应与背景产生清晰对比，保证流程方向一目了然。
  - 可以使用背景分区、阶段色带等方式增强层次感，但应避免过度复杂的滤镜导致视觉噪音。

- 文本与标注：
  - 每个节点都需要有对应文字说明，可以放在节点内部（简短短语）或节点附近（上/下/左/右），保持整体风格一致。
  - 可以在图的上方添加一个整体标题和阶段标题，但避免大段长文本。
  - 使用合适字号和行间距，保证在 PPT 中阅读清晰。

- 复杂度与可读性：
  - 可以包含较多节点和箭头来体现完整的技术流程，但要避免元素无序堆叠。
  - 通过对齐、分组、重复使用样式等方式保持视觉统一。
  - 避免关键连线被遮挡或重叠，确保每个阶段的主路径清楚易懂。

- 尺寸改成“基于 viewBox 的自适应”，别写死 width/height 像素

请严格遵守上述 JSON 输出要求，仅返回包含 svg_code 的 JSON 对象。
"""

    # ------------------------------------------------------------------ #
    # 2. SvgBgCleaner - svg_bg_cleaner 相关提示词
    # ------------------------------------------------------------------ #
    task_prompt_for_svg_bg_cleaner = """
下面给出一段完整的 SVG 源代码（包含文本和图形）：

{svg_code}

你的任务是：
1. 在不改变图形布局、几何结构和视觉风格的前提下，删除或清空所有“文本相关内容”，
   只保留图形元素（如矩形、圆角矩形、圆、折线、路径、箭头、背景块、连线等）。
2. 文本相关内容包括但不限于：
   - 所有 <text> ... </text> 元素；
   - 所有 <tspan> ... </tspan> 元素；
   - 所有 <title> ... </title> 元素；
   - 以及任何仅用于呈现文字的 SVG 元素（如果你能可靠识别）。

具体要求：
- 不要改变 <svg> 根元素的 width、height、viewBox 等属性；
- 不要移动或缩放任何非文本图形元素；
- 可以在删除文本元素后，适当移除仅用于文本的空 <g> 分组（如果明显不再包含任何子元素）；
- 渐变、滤镜、marker（箭头定义）、clipPath 等 “非文本” 定义一律保留；
- 目标是得到一份“纯背景 / 纯图形”的 SVG，用于后续切图和 MinerU 分割。

输出格式要求（非常重要）：
- 你必须仅输出一个严格的 JSON 对象，形如：
  {"svg_bg_code": "<svg ...>...</svg>"}
- 不要输出任何额外文字、注释、解释或 markdown 代码块标记。
- JSON 中只能有一个键：svg_bg_code。
- svg_bg_code 的值是完整的 SVG 源代码字符串：
  - 以 <svg ...> 开始，以 </svg> 结束；
  - 不再包含任何 <text>、<tspan>、<title> 标签（大小写不限）；
  - 其它图形结构、渐变、滤镜、marker 等应尽可能保持不变；
  - 所有双引号必须正确转义，以保证整个 JSON 可被标准 JSON 解析器解析；
  - 换行可以使用 \\n 进行转义，也可以直接内联成一行，只要保证是合法 XML。
"""

    system_prompt_for_svg_bg_cleaner = """
你是一个 SVG 背景清洗助手，专门负责从完整 SVG 中删除所有文本相关元素，仅保留图形和装饰结构。

你的行为准则：
1. 严格遵守输出格式要求，只返回一个 JSON 对象：{"svg_bg_code": "<svg ...>...</svg>"}。
2. 不要输出 markdown 代码块标记（例如 ```svg 或 ```），也不要添加多余的说明。
3. 对于输入的 SVG：
   - 保留所有非文本图形元素（rect, circle, ellipse, path, polyline, polygon, line, g 等）；
   - 保留各种视觉定义（defs 内的 gradient、pattern、filter、marker 等），除非它们只被文本使用且你能非常确定可以安全删除；
   - 删除所有文本相关元素（text, tspan, title 以及其它仅用于显示文字的元素）。
4. 输出的 svg_bg_code 应该是一个结构合法、可直接用 XML 解析的 SVG 文本，并且不再包含任何文本相关标签。

请根据用户提供的 svg_code 完成清洗，并严格返回 JSON。
"""

    system_prompt_for_outline_agent = """
你是一位拥有丰富学术汇报经验的PPT设计专家及大纲生成助手。你的核心任务是将一篇学术论文转化为一份逻辑清晰、视觉布局合理的PPT演示大纲。

请遵循以下严格规则：
1. **深度理解**：仔细阅读用户提供的论文内容，提取核心论点、实验数据和结论。
2. **视觉导向**：在规划每一页PPT时，不仅要生成文字内容，必须明确指出该页是否需要展示论文中的特定插图（Images）或表格（Tables）。
3. **布局建议**：为每一页提供具体的布局指导（例如：左文右图、上标题下表格、两栏对比等）。
4. **格式严格**：输出必须且只能是标准的 JSON 格式数组。严禁包含 markdown 标记（如 ```json）、前言、后语或任何非 JSON 字符。

"""


    task_prompt_for_outline_agent = """
请根据以下提供的论文全文内容，生成一份详细的PPT演示文稿大纲。

**输入数据：**
论文内容：
{text_content}
{minueru_output}

**约束条件：**
1. 目标PPT页数： {page_count} 页。
2. 整体结构应该是有开始有结束，第一页应该就是ppt的主题 和 汇报人!!!!不需要额外的内容！！！
3. 最后一页得是致谢；
4. 返回论文内容一致的语言；

**输出格式要求（JSON Array）：**
请返回一个 JSON 数组，数组中每个对象代表一页PPT，结构如下：
- `title`: 该页PPT的标题。
- `layout_description`: 详细的版面布局描述（例如："左侧列出三个关键挑战点，右侧放置流程图"）。
- `key_points`: 一个包含多个关键要点的字符串列表（List<String>），用于PPT正文展示。
- `asset_ref`: 如果该页需要展示论文中的原图或表格，请提名或路径取其文件（例如 "Table_2", "images/architecture.png"），并且只能1 个 asset；如果不需要引用原图，请填 null。


**示例输出结构 **
!!!必须返回 {language} 语言!!!
[
  {{
    "title": "研究背景：大语言模型的幻觉问题",
    "layout_description": "左侧文字介绍幻觉定义，右侧展示幻觉示例图，图片居中。",
    "key_points": [
      "大语言模型在生成长文本时常出现事实性错误。",
      "现有检索增强生成（RAG）方法的局限性。",
      "本研究旨在解决上下文一致性问题。"
      ......
    ],
    "asset_ref": "images/xxx.png"
  }},
  ......
  {{
    "title": "实验结果",
    "layout_description": "顶部为标题，中间大幅展示架构图，底部放置关键步骤的简要说明。中间放表格Table_2数据。",
    "key_points": [
      "阶段一：查询重写与扩展。",
      "阶段二：基于相关性的文档过滤。",
      "阶段三：生成与验证循环。"
      ......
    ],
    "asset_ref": "Table_2"
  }}
]
"""

    system_prompt_for_table_extractor="""
    你是一名前端代码专家
    """

    task_prompt_for_table_extractor="""

    根据论文内容： 
    {minueru_output} 
    
    找到 {table_num} 的表格 ，的数据 和 内容，数据和caption；

    根据表格内容完整写成html代码：
    返回纯json内容，不要有任何markdown格式的标记，也不要有任何说明文字。
    代码也不要有任何注释；
    
    json格式为：
    {{"html_code": "表格html代码"}}
    """

    system_prompt_for_deep_research_agent = """
You are a Deep Research Assistant. Your task is to conduct a "Deep Research" on a given [Topic] and generate a comprehensive, structured, and detailed research report.

Your report should serve as the foundation for creating a professional PowerPoint presentation.
Therefore, the content must be:
1.  **Comprehensive**: Cover all key aspects, background, methodology, current trends, and future directions related to the topic.
2.  **Structured**: Organize with clear headings (Introduction, Key Concepts, Analysis, Conclusion, etc.).
4.  **Academic/Professional**: Maintain a formal and objective tone.

If the input is just a short topic string, expand it into a full article.
"""

    task_prompt_for_deep_research_agent = """
[Topic]:
{text_content}

[Instructions]:
Please perform a deep research simulation on the above topic and output a detailed research report.
Ensure the content is rich and logically organized !!!!

[Language]: {language}
"""