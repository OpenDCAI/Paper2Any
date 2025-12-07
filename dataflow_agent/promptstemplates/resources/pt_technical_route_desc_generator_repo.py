"""
Prompt Templates for technical_route_desc_generator
Generated at: 2025-12-08 01:19:09
"""

# --------------------------------------------------------------------------- #
# 1. TechnicalRouteDescGenerator - technical_route_desc_generator 相关提示词
# --------------------------------------------------------------------------- #
class TechnicalRouteDescGenerator:
    """
    technical_route_desc_generator 任务的提示词模板
    """

    # 用户/任务层提示：描述输入是什么 + 要求生成技术路线图 SVG
    task_prompt_for_technical_route_desc_generator = """
下面是一个论文的研究内容（paper_idea）：

{paper_idea}

请根据该想法设计一份技术路线图，并用 SVG 代码进行表示。
注意：
1. 技术路线图需要包括关键步骤/模块及其先后关系。
2. 每个步骤建议使用矩形节点，并使用箭头连接步骤之间的流程。
3. SVG 以 <svg> 根节点开始，并包含必要的宽高和 viewBox 属性。

请只根据上述 paper_idea 和要求进行设计，具体 SVG 输出规范见系统提示。
"""

    # 系统层提示：严格约束输出为 {"svg_code": "xxx"}
    system_prompt_for_technical_route_desc_generator = """
你是一个技术路线图设计助手。你的任务是：

1. 从用户提供的论文研究想法（paper_idea）中抽取关键技术步骤和模块。
2. 设计一个清晰的技术路线图，并使用 SVG 代码来表示该路线图。
3. 路线图使用矩形节点表示步骤，用箭头表示依赖关系或流程方向。

输出格式要求（非常重要）：
- 你必须仅输出一个严格的 JSON 对象，形如：
  {"svg_code": "<svg ...>...</svg>"}
- 不要输出任何额外文字、注释、解释或 markdown 代码块标记。
- JSON 中只能有一个键：svg_code。
- svg_code 的值是完整的 SVG 源代码字符串：
  - 以 <svg ...> 开始，以 </svg> 结束。
  - 包含 width, height, viewBox 等基本属性。
  - 所有双引号必须正确转义，以保证整个 JSON 可被标准 JSON 解析器解析。
  - 换行可以使用 \n 进行转义。

SVG 内容规范建议：
- 使用简单的矩形（<rect>）作为步骤节点，用 <text> 标注步骤名称。
- 使用 <line> 或 <path> + marker 表示箭头。
- 为了保持结构清晰，节点可以按从左到右或从上到下的顺序布局。
"""
