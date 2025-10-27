# -*- coding: utf-8 -*-
"""
定义常见的配色风格和绘图风格选项，用于科研绘图。

1. 配色风格 (Color Palettes)
2. 绘图风格 (Illustration Styles)
"""

# 配色风格定义
COLOR_PALETTES = {
    "macaron": """
    A soft pastel color palette featuring light, fresh, and soothing tones.
    Ideal for presentations where you want a gentle, modern, and aesthetically pleasing appearance.
    Colors: Soft pinks, blues, lavenders, mint greens, and yellows.
    """,

    "monochrome": """
    A color scheme using variations in lightness and saturation of a single color.
    Suitable for more technical or formal illustrations where simplicity and harmony are key.
    Colors: Shades of one color (e.g., various blues or grays).
    """,

    "gradient": """
    A smooth transition between different colors, often used to add depth or a sense of movement.
    Ideal for heatmaps, 3D models, or highlighting trends.
    Colors: Warm-to-cool gradients (e.g., red to yellow to blue, or green to purple).
    """,

    "vibrant": """
    Bold, high-contrast colors that make diagrams and charts pop.
    Perfect for eye-catching presentations, emphasizing specific elements.
    Colors: Bright reds, blues, oranges, greens, yellows, and purples.
    """,

    "earthy": """
    Earth-inspired colors that give a grounded, natural, and warm feel.
    Great for illustrations related to environmental or biological topics.
    Colors: Earthy greens, browns, blues, and soft yellows.
    """,

    "cool_tone": """
    Calm, composed colors, typically with blues, purples, and grays.
    Used for serious, analytical, and scientific presentations.
    Colors: Cool blues, purples, and light grays.
    """,

    "high_contrast": """
    Black and white with sharp contrasts, sometimes with one or two accent colors.
    Ideal for professional or academic publications requiring precision and clarity.
    Colors: Black, white, and a single accent color (e.g., red or blue).
    """
}

# 绘图风格定义
ILLUSTRATION_STYLES = {
    "minimalist": """
    Focuses on simplicity using flat colors and geometric shapes.
    This style is often used in scientific charts, diagrams, and conceptual illustrations.
    Characteristics: Simple, flat shapes, no gradients, clean typography.
    """,

    "tech_modern": """
    A high-tech, futuristic style featuring clean, sharp lines.
    Commonly used for data-driven graphics, circuit diagrams, and technology-related visuals.
    Characteristics: Crisp lines, geometric shapes, metallic gradients, sleek fonts.
    """,

    "watercolor": """
    An artistic, flowing style using textured backgrounds and soft blends of colors.
    Ideal for scientific illustrations related to organic and biological subjects.
    Characteristics: Fluid brush strokes, soft blends, light textures, organic forms.
    """,

    "anime_manga": """
    Popular for visual storytelling, especially for educational illustrations targeting younger audiences.
    Characteristics: Bold lines, exaggerated expressions, vibrant colors, stylized characters.
    """,

    "infographic_vector": """
    Uses clean vector lines, icons, and structured layouts to present complex data in an easy-to-digest format.
    Ideal for statistical and research-driven visuals.
    Characteristics: Simple vector shapes, icons, consistent proportions, modular grids.
    """,

    "sketch_hand_drawn": """
    Mimics hand-drawn illustrations, offering a personalized or informal approach.
    Often used in educational materials or when adding a human touch to scientific subjects.
    Characteristics: Rough lines, textured pencil strokes, imperfections, dynamic composition.
    """,

    "3d_style": """
    Adds depth and dimension to diagrams and models, often used for molecular structures or architectural models.
    Characteristics: Realistic shading, shadows, depth effects, 3D models, perspective.
    """,

    "retro_vintage": """
    Uses retro color schemes and artistic motifs that evoke older graphic design trends.
    Suitable for giving a nostalgic or classical feel to data representation.
    Characteristics: Distressed textures, muted colors, vintage typography, old-school layout.
    """,

    "abstract": """
    Uses unconventional shapes and colors to convey more conceptual or artistic representations of a subject.
    Often used for visualizing abstract data or scientific phenomena.
    Characteristics: Asymmetrical shapes, non-literal representation, freeform designs.
    """
}

# 示例如何使用上述配色风格和绘图风格：
def print_style_example(style_name: str):
    """
    打印指定风格的详细说明。
    :param style_name: 风格名称（例如：'macaron', 'minimalist'）
    """
    if style_name in COLOR_PALETTES:
        print(f"Color Palette - {style_name}:\n{COLOR_PALETTES[style_name]}")
    elif style_name in ILLUSTRATION_STYLES:
        print(f"Illustration Style - {style_name}:\n{ILLUSTRATION_STYLES[style_name]}")
    else:
        print("Invalid style name. Please choose from the available options.")

# 示例调用
# print_style_example("macaron")
