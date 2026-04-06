"""
paper2drawio_sam3 workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use SAM3 HTTP service + VLM OCR (qwen-vl-ocr-latest) to convert a diagram image
(or a PDF first page) into editable draw.io XML.

Pipeline:
1) Load input image (image path or PDF first page)
2) VLM OCR text extraction (qwen-vl-ocr-latest via Comfly)
3) SAM3 text-prompt segmentation via HTTP service (ported from Edit-Banana/sam3_service)
4) Shape/icon classification + color sampling
5) Render draw.io XML
"""

from __future__ import annotations

import base64
import copy
import json
import math
import os
import re
import statistics
import time
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from dataflow_agent.state import Paper2DrawioState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_vlm_agent
from dataflow_agent.logger import get_logger
from dataflow_agent.toolkits.multimodaltool.sam3_tool import (
    Sam3PredictRun,
    decode_sam3_mask,
    dedup_sam3_results_across_groups,
    filter_sam3_items_contained_by_images,
    get_sam3_client,
    run_sam3_predict_runs,
)
from dataflow_agent.toolkits.multimodaltool.ocr_config import get_ocr_api_credentials
from dataflow_agent.toolkits.multimodaltool.ocr_utils import extract_bbox_items
from dataflow_agent.toolkits.drawio_tools import wrap_xml
from dataflow_agent.toolkits.image2drawio import (
    extract_text_color,
    mask_to_bbox,
    normalize_mask,
    sample_fill_stroke,
    save_masked_rgba,
    bbox_iou_px,
)
from dataflow_agent.utils_common import robust_parse_json
from dataflow_agent.workflow.sam3_segment_hint import (
    dedupe_prompts,
    generate_sam3_segment_hints,
    normalize_prompt,
)

log = get_logger(__name__)

# ==================== SAM3 PROMPTS (ported from Edit-Banana/prompts) ====================
SHAPE_PROMPT = [
    "rectangle",
    "rounded rectangle",
    "diamond",
    "ellipse",
    "circle",
    "triangle",
    "hexagon",
]

ARROW_PROMPT = [
    "arrow",
    "line",
    "connector",
]

IMAGE_PROMPT = [
    "icon",
    "symbol",
    "pictogram",
    "glyph",
    "picture",
    "logo",
    "chart",
    "plot",
    "graph",
    "diagram",
    # Scientific / schematic figures often contain semantic icons rather than
    # UI-style pictograms, so keep a broader object vocabulary here.
    "object",
    "illustration",
    "cell",
    "molecule",
    "complex",
    "receptor",
    "particle",
    "node",
    "blob",
]

# 泛化补召回提示词：避免与具体业务词绑定（如 planner/critic/robot）
IMAGE_PROMPT_RECALL = [
    "illustration",
    "object",
    "device",
    "agent",
    "character",
    "avatar",
    "mascot",
    "screenshot",
    "cell",
    "molecule",
    "complex",
    "receptor",
    "node",
    "blob",
]

BACKGROUND_PROMPT = [
    "panel",
    "container",
    "filled region",
    "background",
]

SAM3_GROUPS = {
    "shape": SHAPE_PROMPT,
    "arrow": ARROW_PROMPT,
    "image": IMAGE_PROMPT,
    "background": BACKGROUND_PROMPT,
}

# Thresholds aligned with Edit-Banana config defaults
SAM3_GROUP_CONFIG = {
    "shape": {"score_threshold": 0.5, "min_area": 200, "priority": 3},
    "arrow": {"score_threshold": 0.45, "min_area": 50, "priority": 4},
    "image": {"score_threshold": 0.5, "min_area": 100, "priority": 2},
    "background": {"score_threshold": 0.25, "min_area": 500, "priority": 1},
}

# 第2轮 image 召回配置（低阈值 + 动态最小面积）
SAM3_IMAGE_RECALL_SCORE_THRESHOLD = 0.38
SAM3_IMAGE_RECALL_MIN_AREA_BASE = 40
SAM3_IMAGE_RECALL_MIN_AREA_RATIO = 0.00003
SAM3_IMAGE_RECALL_TRIGGER_MAX_IMAGES = 2

# Dedup params aligned with Edit-Banana defaults
SAM3_DEDUP_IOU = 0.7
SAM3_ARROW_DEDUP_IOU = 0.85
SAM3_SHAPE_IMAGE_IOU = 0.6

MAX_DRAWIO_ELEMENTS = 800
MIN_IMAGE_AREA_RATIO = 0.00001
MAX_IMAGE_BBOX_AREA_RATIO = 0.88
FALLBACK_COARSE_TEXT_MAX_AREA_RATIO = 0.08
FALLBACK_LOCAL_TEXT_MAX_AREA_RATIO = 0.035

# 对低覆盖的大图标做前景细化回退，避免主体缺失且避免纯色背景串入
IMAGE_MASK_REPAIR_LOW_COVERAGE_THRESHOLD = 0.58
IMAGE_MASK_REPAIR_MIN_BBOX_AREA = 5000
IMAGE_MASK_REPAIR_PAD_PX = 12
IMAGE_MASK_REPAIR_MAX_COVERAGE_ON_ORIG_BBOX = 0.74
IMAGE_MASK_REPAIR_MIN_GAIN_RATIO = 0.08
IMAGE_FRAGMENT_SKIP_MAX_AREA = 2500
IMAGE_FRAGMENT_CONTAIN_THRESHOLD = 0.6


# ==================== TEXT VECTORIZATION ====================
@dataclass
class NormalizedCoords:
    x: float
    y: float
    width: float
    height: float
    baseline_y: float
    rotation: float


class CoordProcessor:
    def __init__(self, source_width: int, source_height: int,
                 canvas_width: int = None, canvas_height: int = None):
        self.source_width = source_width
        self.source_height = source_height
        self.canvas_width = canvas_width if canvas_width is not None else source_width
        self.canvas_height = canvas_height if canvas_height is not None else source_height
        self.scale_x = self.canvas_width / source_width
        self.scale_y = self.canvas_height / source_height
        self.uniform_scale = min(self.scale_x, self.scale_y)

    def normalize_polygon(self, polygon: list[tuple[float, float]]) -> NormalizedCoords:
        if len(polygon) < 4:
            return NormalizedCoords(0, 0, 0, 0, 0, 0)

        normalized_points = [
            (p[0] * self.uniform_scale, p[1] * self.uniform_scale)
            for p in polygon
        ]

        p0, p1, p2, p3 = normalized_points[:4]
        rotation = self._calculate_rotation(p0, p1)
        center_x = sum(p[0] for p in normalized_points) / 4
        center_y = sum(p[1] for p in normalized_points) / 4
        edge_top = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        edge_left = math.sqrt((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2)
        width = edge_top
        height = edge_left
        x = center_x - width / 2
        y = center_y - height / 2
        baseline_y = (p2[1] + p3[1]) / 2

        return NormalizedCoords(
            x=x, y=y, width=width, height=height,
            baseline_y=baseline_y, rotation=rotation
        )

    def _calculate_rotation(self, p0: tuple, p1: tuple) -> float:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if abs(angle_deg) < 2:
            return 0.0
        return round(angle_deg, 1)

    def polygon_to_geometry(self, polygon: list[tuple[float, float]]) -> dict:
        coords = self.normalize_polygon(polygon)
        return {
            "x": round(coords.x, 2),
            "y": round(coords.y, 2),
            "width": round(coords.width, 2),
            "height": round(coords.height, 2),
            "baseline_y": round(coords.baseline_y, 2),
            "rotation": coords.rotation
        }


# -- Text processors --
class FontSizeProcessor:
    def __init__(
        self,
        formula_ratio: float = 0.6,
        text_offset: float = 1.0,
        min_font_size: float = 8.0,
        max_font_size: float = 48.0,
        height_ratio: float = 0.85,
        width_safety: float = 1.04,
        drawio_pt_ratio: float = 1.45,
    ):
        self.formula_ratio = formula_ratio
        self.text_offset = text_offset
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.height_ratio = height_ratio
        self.width_safety = width_safety
        # OCR 框常用像素，draw.io fontSize 更接近 pt，需要换算避免视觉放大
        self.drawio_pt_ratio = drawio_pt_ratio

    @staticmethod
    def _visual_text_length(text: str) -> float:
        """估算文本在横向上的视觉长度（以字体 em 为单位）。"""
        if not text:
            return 1.0
        total = 0.0
        for ch in text:
            if ch.isspace():
                total += 0.33
            elif ord(ch) < 128:
                if ch.isalnum():
                    total += 0.56
                else:
                    total += 0.45
            else:
                # CJK 等宽字符通常接近 1em
                total += 1.0
        return max(total, 1.0)

    def _estimate_font_size(self, text: str, geometry: Dict[str, Any], is_latex: bool) -> float:
        """
        智能字号估计：同时受文本框高度和文本长度限制，尽量贴近原图视觉大小。
        """
        width = max(float(geometry.get("width", 12) or 12), 1.0)
        height = max(float(geometry.get("height", 12) or 12), 1.0)

        if is_latex:
            size = height * self.formula_ratio
        else:
            # 约束1：按行高估计
            by_height = height * self.height_ratio
            # 约束2：按文本长度估计（避免短高框导致字号过大）
            vis_len = self._visual_text_length(text)
            by_width = (width / vis_len) * self.width_safety
            size = min(by_height, by_width)
        size = size * self.drawio_pt_ratio
        return max(self.min_font_size, min(size, self.max_font_size))

    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        unify: bool = True,
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        blocks = self.calculate_font_sizes(text_blocks)
        if unify and len(blocks) > 1:
            blocks = self.unify_by_clustering(
                blocks,
                vertical_threshold_ratio,
                font_diff_threshold
            )
        return blocks

    def calculate_font_sizes(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for block in text_blocks:
            block = copy.copy(block)
            geometry = block.get("geometry", {})
            text = block.get("text", "")
            is_latex = block.get("is_latex", False)
            font_size = self._estimate_font_size(text=text, geometry=geometry, is_latex=is_latex)
            block["font_size"] = max(font_size, 6)
            result.append(block)
        return result

    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        if not text_blocks:
            return text_blocks
        n = len(text_blocks)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_group(
                    text_blocks[i], text_blocks[j],
                    vertical_threshold_ratio, font_diff_threshold
                ):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = copy.deepcopy(text_blocks)
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            font_sizes = [result[i].get("font_size", 12) for i in group_indices]
            median_size = statistics.median(font_sizes)
            for idx in group_indices:
                result[idx]["font_size"] = round(median_size, 1)
        return result

    def _should_group(
        self,
        block_a: Dict,
        block_b: Dict,
        vertical_threshold_ratio: float,
        font_diff_threshold: float
    ) -> bool:
        geo_a = block_a.get("geometry", {})
        geo_b = block_b.get("geometry", {})

        x1, y1 = geo_a.get("x", 0), geo_a.get("y", 0)
        w1, h1 = geo_a.get("width", 0), geo_a.get("height", 0)
        x2, y2 = geo_b.get("x", 0), geo_b.get("y", 0)
        w2, h2 = geo_b.get("width", 0), geo_b.get("height", 0)

        font_a = block_a.get("font_size", 12)
        font_b = block_b.get("font_size", 12)

        bottom_a, bottom_b = y1 + h1, y2 + h2
        gap_a_above_b = y2 - bottom_a
        gap_b_above_a = y1 - bottom_b
        if gap_a_above_b < 0 and gap_b_above_a < 0:
            vertical_distance = 0
        else:
            vertical_distance = min(abs(gap_a_above_b), abs(gap_b_above_a))

        min_height = min(h1, h2) if min(h1, h2) > 0 else 1
        vertical_close = vertical_distance < min_height * vertical_threshold_ratio

        right_a, left_b = x1 + w1, x2
        right_b, left_a = x2 + w2, x1
        horizontal_overlap = not (right_a < left_b or right_b < left_a)

        abs_diff = abs(font_a - font_b)
        avg_font = (font_a + font_b) / 2 if (font_a + font_b) > 0 else 1
        rel_diff = abs_diff / avg_font
        font_close = abs_diff < font_diff_threshold or rel_diff < 0.30
        return vertical_close and horizontal_overlap and font_close


class FontFamilyProcessor:
    CODE_KEYWORDS = [
        "id_", "code_", "0x", "struct", "func_", "var_", "ptr_",
        "def ", "class ", "import ", "__", "::", "{}"
    ]

    FONT_MAPPING = {
        "microsoft yahei": "Microsoft YaHei",
        "微软雅黑": "Microsoft YaHei",
        "simhei": "SimHei",
        "黑体": "SimHei",
        "dengxian": "DengXian",
        "等线": "DengXian",
        "arial": "Arial",
        "calibri": "Calibri",
        "verdana": "Verdana",
        "helvetica": "Helvetica",
        "roboto": "Roboto",
        "simsun": "SimSun",
        "宋体": "SimSun",
        "times new roman": "Times New Roman",
        "times": "Times New Roman",
        "georgia": "Georgia",
        "yu mincho": "SimSun",
        "ms mincho": "SimSun",
        "courier new": "Courier New",
        "courier": "Courier New",
        "consolas": "Courier New",
        "monaco": "Courier New",
        "menlo": "Courier New",
    }

    SERIF_KEYWORDS = ["baskerville", "garamond", "palatino", "didot", "bodoni"]
    SANS_KEYWORDS = ["segoe", "tahoma", "trebuchet", "lucida"]
    MONO_KEYWORDS = ["mono", "consolas", "menlo", "monaco", "courier"]

    def __init__(self, default_font: str = "Arial"):
        self.default_font = default_font
        self.font_cache = {}

    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        global_font: str = None,
        unify: bool = True
    ) -> List[Dict[str, Any]]:
        global_font = global_font or self.default_font
        result = []
        for block in text_blocks:
            block = copy.copy(block)
            if block.get("font_family"):
                block["font_family"] = self.standardize(block["font_family"])
            else:
                block["font_family"] = self.infer_from_text(
                    block.get("text", ""),
                    is_bold=block.get("is_bold", False),
                    is_latex=block.get("is_latex", False),
                    default_font=global_font
                )
            result.append(block)
        if unify and len(result) > 1:
            result = self.unify_by_clustering(result)
        return result

    def standardize(self, font_name: str) -> str:
        if not font_name:
            return self.default_font
        original = font_name.strip()
        main_font = original.split(',')[0].strip()
        clean_name = main_font.lower()
        if clean_name in self.FONT_MAPPING:
            return self.FONT_MAPPING[clean_name]
        for key, value in self.FONT_MAPPING.items():
            if key in clean_name:
                return value
        if any(kw in clean_name for kw in self.SERIF_KEYWORDS):
            return "Times New Roman"
        if any(kw in clean_name for kw in self.SANS_KEYWORDS):
            return "Arial"
        if any(kw in clean_name for kw in self.MONO_KEYWORDS):
            return "Courier New"
        return main_font

    def infer_from_text(
        self,
        text: str,
        is_bold: bool = False,
        is_latex: bool = False,
        default_font: str = None
    ) -> str:
        default_font = default_font or self.default_font
        cache_key = f"{text}_{is_bold}_{is_latex}"
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        font = default_font
        if is_latex:
            font = "Times New Roman"
        elif self._is_chinese_text(text):
            font = "SimSun"
        elif self._is_code_text(text):
            font = "Courier New"
        elif self._is_academic_text(text):
            font = "Times New Roman"
        self.font_cache[cache_key] = font
        return font

    def _is_code_text(self, text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            return True
        if "_" in text and len(text.split()) == 1:
            return True
        return False

    def _is_academic_text(self, text: str) -> bool:
        academic_keywords = ['figure', 'table', 'equation', 'result', 'method', 'data', 'analysis']
        if any(kw in text.lower() for kw in academic_keywords) and len(text) > 10:
            return True
        if ' ' in text and len(text) > 15 and any(p in text for p in ['.', ',', ';']):
            return True
        return False

    def _is_chinese_text(self, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def unify_by_clustering(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not text_blocks:
            return text_blocks
        n = len(text_blocks)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(text_blocks[i], text_blocks[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = copy.deepcopy(text_blocks)
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            fonts = [result[i].get("font_family", self.default_font) for i in group_indices]
            most_common = max(set(fonts), key=fonts.count)
            for idx in group_indices:
                result[idx]["font_family"] = most_common
        return result

    def _should_merge(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ga = a.get("geometry", {})
        gb = b.get("geometry", {})
        ax, ay, aw, ah = ga.get("x", 0), ga.get("y", 0), ga.get("width", 0), ga.get("height", 0)
        bx, by, bw, bh = gb.get("x", 0), gb.get("y", 0), gb.get("width", 0), gb.get("height", 0)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return False
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        x_overlap = min(ax2, bx2) - max(ax, bx)
        y_overlap = min(ay2, by2) - max(ay, by)
        if x_overlap <= 0 or y_overlap <= 0:
            return False
        return True


class StyleProcessor:
    def process(
        self,
        text_blocks: List[Dict[str, Any]],
        azure_styles: List[Dict] = None,
        unify: bool = True
    ) -> List[Dict[str, Any]]:
        azure_styles = azure_styles or []
        result = self.extract_styles(text_blocks, azure_styles)
        if unify and len(result) > 1:
            result = self.unify_by_clustering(result)
        return result

    def extract_styles(
        self,
        text_blocks: List[Dict[str, Any]],
        azure_styles: List[Dict]
    ) -> List[Dict[str, Any]]:
        result = []
        for block in text_blocks:
            block = copy.copy(block)
            styles = self._extract_block_styles(block, azure_styles)
            block["font_weight"] = "bold" if styles["is_bold"] else "normal"
            block["font_style"] = "italic" if styles["is_italic"] else "normal"
            block["is_bold"] = styles["is_bold"]
            block["is_italic"] = styles["is_italic"]
            if styles["color"]:
                block["font_color"] = styles["color"]
            if styles["background_color"]:
                block["background_color"] = styles["background_color"]
            result.append(block)
        return result

    def _extract_block_styles(self, block: Dict[str, Any], azure_styles: List[Dict]) -> Dict[str, Any]:
        styles = {
            "is_bold": False,
            "is_italic": False,
            "color": None,
            "background_color": None
        }
        if block.get("font_weight") == "bold" or block.get("is_bold"):
            styles["is_bold"] = True
        if block.get("font_style") == "italic" or block.get("is_italic"):
            styles["is_italic"] = True
        if block.get("font_color"):
            styles["color"] = block["font_color"]
        if block.get("background_color"):
            styles["background_color"] = block["background_color"]
        has_info = styles["is_bold"] or styles["is_italic"] or styles["color"]
        if has_info or not azure_styles:
            return styles
        block_spans = block.get("spans", [])
        if not block_spans:
            return styles
        block_offset = block_spans[0].get("offset", 0) if isinstance(block_spans[0], dict) else 0
        block_length = block_spans[0].get("length", 0) if isinstance(block_spans[0], dict) else 0
        for style in azure_styles:
            style_spans = style.get("spans", [])
            for span in style_spans:
                span_offset = span.get("offset", 0)
                span_length = span.get("length", 0)
                if self._spans_overlap(block_offset, block_length, span_offset, span_length):
                    if style.get("fontWeight") == "bold":
                        styles["is_bold"] = True
                    if style.get("fontStyle") == "italic":
                        styles["is_italic"] = True
                    if style.get("color") and not styles["color"]:
                        styles["color"] = style["color"]
                    if style.get("backgroundColor") and not styles["background_color"]:
                        styles["background_color"] = style["backgroundColor"]
        return styles

    def _spans_overlap(self, offset1: int, length1: int, offset2: int, length2: int) -> bool:
        end1 = offset1 + length1
        end2 = offset2 + length2
        return not (end1 <= offset2 or end2 <= offset1)

    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold: float = 0.8,
        horizontal_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        if not text_blocks:
            return text_blocks
        n = len(text_blocks)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(text_blocks[i], text_blocks[j], vertical_threshold, horizontal_threshold):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = copy.deepcopy(text_blocks)
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            colors = [result[i].get("font_color") for i in group_indices if result[i].get("font_color")]
            if not colors:
                continue
            common = max(set(colors), key=colors.count)
            for idx in group_indices:
                result[idx]["font_color"] = common
        return result

    def _should_merge(self, a: Dict[str, Any], b: Dict[str, Any], vth: float, hth: float) -> bool:
        ga = a.get("geometry", {})
        gb = b.get("geometry", {})
        ax, ay, aw, ah = ga.get("x", 0), ga.get("y", 0), ga.get("width", 0), ga.get("height", 0)
        bx, by, bw, bh = gb.get("x", 0), gb.get("y", 0), gb.get("width", 0), gb.get("height", 0)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return False
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        y_overlap = min(ay2, by2) - max(ay, by)
        x_overlap = min(ax2, bx2) - max(ax, bx)
        if y_overlap <= 0 or x_overlap <= 0:
            return False
        y_min = min(ah, bh)
        x_min = min(aw, bw)
        return (y_overlap / y_min) >= vth and (x_overlap / x_min) >= hth


def _vlm_scale(values: List[float]) -> float:
    max_val = max(abs(v) for v in values) if values else 0.0
    return 1000.0 if max_val > 1.5 else 1.0


def _rotate_rect_to_polygon(rr: List[float], image_w: int, image_h: int) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(rr, list) or len(rr) != 5:
        return None
    # Qwen OCR rotate_rect convention: [cx, cy, width, height, angle]
    cx, cy, rw, rh, angle = [float(v) for v in rr]
    scale = _vlm_scale([cx, cy, rw, rh])
    cx = cx / scale * image_w
    cy = cy / scale * image_h
    rw = rw / scale * image_w
    rh = rh / scale * image_h
    rect = ((cx, cy), (rw, rh), float(angle))
    box = cv2.boxPoints(rect)
    return [(float(p[0]), float(p[1])) for p in box]


def _bbox_to_polygon(bbox: List[float], image_w: int, image_h: int) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    y1, x1, y2, x2 = [float(v) for v in bbox]
    scale = _vlm_scale([x1, y1, x2, y2])
    x1 = x1 / scale * image_w
    x2 = x2 / scale * image_w
    y1 = y1 / scale * image_h
    y2 = y2 / scale * image_h
    if x2 <= x1 or y2 <= y1:
        return None
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _vectorize_text_blocks(
    bbox_res: List[Dict[str, Any]],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    text_blocks: List[Dict[str, Any]] = []
    for it in bbox_res or []:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        polygon = None
        if "rotate_rect" in it:
            polygon = _rotate_rect_to_polygon(it.get("rotate_rect"), image_w, image_h)
        if polygon is None and "bbox" in it:
            polygon = _bbox_to_polygon(it.get("bbox"), image_w, image_h)
        if not polygon:
            continue
        text_blocks.append({
            "text": text,
            "polygon": polygon,
            "is_latex": False,
        })

    if not text_blocks:
        return []

    coord_processor = CoordProcessor(source_width=image_w, source_height=image_h)
    for block in text_blocks:
        polygon = block.get("polygon", [])
        block["geometry"] = coord_processor.polygon_to_geometry(polygon) if polygon else {
            "x": 0, "y": 0, "width": 100, "height": 20, "rotation": 0
        }

    font_size_processor = FontSizeProcessor()
    font_family_processor = FontFamilyProcessor()
    style_processor = StyleProcessor()

    text_blocks = font_size_processor.process(text_blocks)
    text_blocks = font_family_processor.process(text_blocks, global_font="Arial")
    text_blocks = style_processor.process(text_blocks, azure_styles=[])

    return text_blocks


# ==================== DRAWIO HELPERS ====================
TEXT_COLOR = "#111111"
TEXT_FONT_SIZE_DEFAULT = 14
TEXT_FONT_STYLE = 1


def _ensure_result_path(state: Paper2DrawioState) -> str:
    raw = getattr(state, "result_path", None)
    if raw:
        return raw
    ts = int(time.time())
    base_dir = Path(f"outputs/paper2drawio_sam3/{ts}").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    state.result_path = str(base_dir)
    return state.result_path


def _is_image_path(path: str) -> bool:
    if not path:
        return False
    ext = Path(path).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _render_pdf_first_page(pdf_path: str, out_path: str) -> Optional[str]:
    try:
        import fitz
    except Exception as e:
        log.error(f"[paper2drawio_sam3] PyMuPDF not available: {e}")
        return None
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(alpha=False)
        pix.save(out_path)
        return out_path
    except Exception as e:
        log.error(f"[paper2drawio_sam3] PDF render failed: {e}")
        return None


def _extract_paper_content(state: Paper2DrawioState) -> str:
    pdf_path = state.paper_file
    if not pdf_path:
        return state.text_content or ""
    if not str(pdf_path).lower().endswith(".pdf"):
        return state.text_content or ""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_idx in range(min(15, len(doc))):
            page = doc.load_page(page_idx)
            text_parts.append(page.get_text("text") or "")
        return "\n".join(text_parts).strip()
    except Exception as e:
        log.error(f"PDF 解析失败: {e}")
        return state.text_content or ""


def _escape_text(text: str, is_latex: bool = False) -> str:
    escaped = html.escape(text or "")
    if is_latex:
        latex_content = escaped.replace("$", "").strip()
        escaped = f"\\({latex_content}\\)"
    return escaped


def _text_style_from_block(block: Dict[str, Any]) -> str:
    styles = [
        "text", "html=1", "whiteSpace=nowrap", "autosize=1", "resizable=0",
        "align=center", "verticalAlign=middle", "overflow=visible",
    ]
    font_size = block.get("font_size") or TEXT_FONT_SIZE_DEFAULT
    styles.append(f"fontSize={int(font_size)}")

    font_style_value = 0
    if block.get("font_weight") == "bold" or block.get("is_bold"):
        font_style_value += 1
    if block.get("font_style") == "italic" or block.get("is_italic"):
        font_style_value += 2
    if font_style_value > 0:
        styles.append(f"fontStyle={font_style_value}")

    font_color = block.get("font_color")
    if font_color:
        styles.append(f"fontColor={font_color}")

    font_family = block.get("font_family")
    if font_family:
        first_font = font_family.split(",")[0].strip()
        styles.append(f"fontFamily={first_font}")

    rotation = block.get("geometry", {}).get("rotation", 0) or 0
    if rotation:
        styles.append(f"rotation={rotation}")

    return ";".join(styles) + ";"


def _build_mxcell(
    cell_id: str,
    value: str,
    style: str,
    bbox_px: List[float],
    parent: str = "1",
    vertex: bool = True,
    is_latex: bool = False,
) -> str:
    x1, y1, x2, y2 = bbox_px
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    x = int(round(x1))
    y = int(round(y1))
    v_attr = "1" if vertex else "0"
    return (
        f"<mxCell id=\"{cell_id}\" value=\"{_escape_text(value, is_latex=is_latex)}\" style=\"{style}\" "
        f"vertex=\"{v_attr}\" parent=\"{parent}\">"
        f"<mxGeometry x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" as=\"geometry\"/>"
        f"</mxCell>"
    )


def _shape_style(
    shape_type: str,
    fill_hex: str,
    stroke_hex: str,
    font_size: Optional[int] = None,
    font_color: Optional[str] = None,
) -> str:
    st = (shape_type or "").lower()
    if st in {"ellipse", "circle"}:
        base = "shape=ellipse;"
    elif st in {"diamond", "rhombus"}:
        base = "shape=rhombus;"
    elif st in {"triangle"}:
        base = "shape=triangle;"
    elif st in {"hexagon"}:
        base = "shape=hexagon;perimeter=hexagonPerimeter2;fixedSize=1;"
    elif st in {"container", "rounded rectangle", "rounded_rect", "rounded rectangle"}:
        base = "rounded=1;"
    else:
        base = "rounded=1;" if st in {"rounded_rect"} else "rounded=0;"
    fs = int(font_size) if font_size else TEXT_FONT_SIZE_DEFAULT
    fc = font_color or TEXT_COLOR
    return (
        f"{base}whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
        f"fillColor={fill_hex};strokeColor={stroke_hex};"
        f"fontColor={fc};fontStyle={TEXT_FONT_STYLE};fontSize={fs};"
    )


def _text_style(color_hex: str, font_size: Optional[int] = None) -> str:
    fs = int(font_size) if font_size else TEXT_FONT_SIZE_DEFAULT
    return (
        "text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
        f"strokeColor=none;fillColor=none;fontColor={color_hex or TEXT_COLOR};"
        f"fontStyle={TEXT_FONT_STYLE};fontSize={fs};"
    )


def _image_style(data_uri: str) -> str:
    safe_uri = data_uri.replace(";", "%3B")
    return f"shape=image;imageAspect=0;aspect=fixed;image={safe_uri};"


def _bbox_area(b: List[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _sanitize_bbox(bbox_px: Optional[List[int]], target_shape: tuple[int, int, int]) -> Optional[List[int]]:
    if not bbox_px or len(bbox_px) != 4:
        return None
    h, w = target_shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
    except Exception:
        return None
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _text_block_bbox(block: Dict[str, Any], image_w: int, image_h: int) -> Optional[List[int]]:
    geometry = block.get("geometry", {}) or {}
    try:
        x = float(geometry.get("x", 0))
        y = float(geometry.get("y", 0))
        width = float(geometry.get("width", 0))
        height = float(geometry.get("height", 0))
    except Exception:
        return None

    bbox = [
        int(round(x)),
        int(round(y)),
        int(round(x + width)),
        int(round(y + height)),
    ]
    return _sanitize_bbox(bbox, (image_h, image_w, 1))


def _sample_surrounding_color(image_bgr: np.ndarray, bbox: List[int], pad: int = 8) -> Tuple[int, int, int]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(w, x2 + pad)
    oy2 = min(h, y2 + pad)
    if ox2 <= ox1 or oy2 <= oy1:
        return (255, 255, 255)

    outer = image_bgr[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return (255, 255, 255)

    inner_x1 = max(0, x1 - ox1)
    inner_y1 = max(0, y1 - oy1)
    inner_x2 = min(outer.shape[1], x2 - ox1)
    inner_y2 = min(outer.shape[0], y2 - oy1)

    ring_mask = np.ones(outer.shape[:2], dtype=bool)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    ring_pixels = outer[ring_mask]
    if ring_pixels.size == 0:
        return (255, 255, 255)

    median_bgr = np.median(ring_pixels.reshape(-1, 3), axis=0)
    return tuple(int(max(0, min(255, round(v)))) for v in median_bgr.tolist())


def _text_blocks_are_coarse(text_blocks: List[Dict[str, Any]], image_w: int, image_h: int) -> bool:
    if not text_blocks:
        return False

    image_area = float(max(1, image_w * image_h))
    area_ratios: List[float] = []
    for block in text_blocks:
        bbox = _text_block_bbox(block, image_w, image_h)
        if not bbox:
            continue
        area_ratios.append(float(_bbox_area(bbox)) / image_area)

    if not area_ratios:
        return False
    max_ratio = max(area_ratios)
    if max_ratio >= FALLBACK_COARSE_TEXT_MAX_AREA_RATIO:
        return True
    if len(area_ratios) <= 3 and max_ratio >= 0.03:
        return True
    return False


def _build_visual_fallback_elements(
    image_bgr: np.ndarray,
    text_blocks: List[Dict[str, Any]],
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], bool]:
    h, w = image_bgr.shape[:2]
    use_original_with_text = _text_blocks_are_coarse(text_blocks, w, h)

    fallback_dir = out_dir / "visual_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    bg_path = fallback_dir / ("background_original.png" if use_original_with_text else "background_no_text.png")

    bg_image = image_bgr.copy()
    if not use_original_with_text:
        image_area = float(max(1, w * h))
        for block in text_blocks:
            bbox = _text_block_bbox(block, w, h)
            if not bbox:
                continue
            bbox_area_ratio = float(_bbox_area(bbox)) / image_area
            if bbox_area_ratio > FALLBACK_LOCAL_TEXT_MAX_AREA_RATIO:
                continue
            text = (block.get("text") or "").strip()
            if not text:
                continue

            x1, y1, x2, y2 = bbox
            pad = max(2, min(10, int(round(min(x2 - x1, y2 - y1) * 0.15))))
            ex1 = max(0, x1 - pad)
            ey1 = max(0, y1 - pad)
            ex2 = min(w, x2 + pad)
            ey2 = min(h, y2 + pad)
            fill_bgr = _sample_surrounding_color(bg_image, [ex1, ey1, ex2, ey2], pad=pad + 4)
            cv2.rectangle(bg_image, (ex1, ey1), (ex2, ey2), fill_bgr, -1)

    cv2.imwrite(str(bg_path), bg_image)
    elements = [{
        "id": "fallback_background",
        "kind": "image",
        "bbox_px": [0, 0, w, h],
        "image_path": str(bg_path),
        "area": int(w * h),
        "group": "background_fallback",
    }]
    return elements, use_original_with_text


def _shape_type_from_prompt(prompt: str) -> str:
    p = normalize_prompt(prompt)
    if p in {"rounded rectangle", "rounded_rectangle"}:
        return "rounded rectangle"
    if p in {"rectangle", "square", "panel", "background", "filled region", "title bar", "section_panel"}:
        return "rectangle"
    if p in {"container"}:
        return "rounded rectangle"
    if p in {"ellipse", "circle"}:
        return "ellipse"
    if p in {"diamond"}:
        return "diamond"
    if p in {"triangle"}:
        return "triangle"
    if p in {"hexagon"}:
        return "hexagon"
    return p or "rectangle"


def _sam3_predict_groups(
    client: Any,
    image_path: str,
    extra_image_prompts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    image_area: Optional[int] = None
    try:
        with Image.open(image_path) as img:
            image_area = int(img.width * img.height)
    except Exception:
        image_area = None

    merged_image_prompts = dedupe_prompts(IMAGE_PROMPT + (extra_image_prompts or []))
    merged_recall_prompts = dedupe_prompts(IMAGE_PROMPT_RECALL + (extra_image_prompts or []))

    try:
        base_runs: List[Sam3PredictRun] = []
        for group, prompts in SAM3_GROUPS.items():
            cfg = SAM3_GROUP_CONFIG.get(group, {})
            group_prompts = merged_image_prompts if group == "image" else prompts
            base_runs.append(
                Sam3PredictRun(
                    group=group,
                    prompts=group_prompts,
                    score_threshold=cfg.get("score_threshold"),
                    min_area=cfg.get("min_area"),
                )
            )

        all_results = run_sam3_predict_runs(
            client=client,
            image_path=image_path,
            runs=base_runs,
        )
        all_results = dedup_sam3_results_across_groups(
            all_results,
            group_config=SAM3_GROUP_CONFIG,
            dedup_iou=SAM3_DEDUP_IOU,
            arrow_dedup_iou=SAM3_ARROW_DEDUP_IOU,
            shape_image_iou=SAM3_SHAPE_IMAGE_IOU,
        )
        all_results = filter_sam3_items_contained_by_images(
            all_results,
            image_groups=["image"],
            contain_threshold=0.85,
        )

        safe_image_area = image_area if image_area and image_area > 0 else 1
        core_image_hits = 0
        for item in all_results:
            if item.get("group") != "image":
                continue
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bbox_ratio = float(_bbox_area(bbox)) / float(safe_image_area)
            if bbox_ratio <= MAX_IMAGE_BBOX_AREA_RATIO:
                core_image_hits += 1

        should_recall = core_image_hits <= SAM3_IMAGE_RECALL_TRIGGER_MAX_IMAGES
        if should_recall and merged_recall_prompts:
            recall_min_area = SAM3_IMAGE_RECALL_MIN_AREA_BASE
            if image_area and image_area > 0:
                recall_min_area = max(
                    SAM3_IMAGE_RECALL_MIN_AREA_BASE,
                    int(image_area * SAM3_IMAGE_RECALL_MIN_AREA_RATIO),
                )

            recall_results = run_sam3_predict_runs(
                client=client,
                image_path=image_path,
                runs=[
                    Sam3PredictRun(
                        group="image",
                        prompts=merged_recall_prompts,
                        score_threshold=SAM3_IMAGE_RECALL_SCORE_THRESHOLD,
                        min_area=recall_min_area,
                    )
                ],
            )
            if recall_results:
                all_results.extend(recall_results)
                all_results = dedup_sam3_results_across_groups(
                    all_results,
                    group_config=SAM3_GROUP_CONFIG,
                    dedup_iou=SAM3_DEDUP_IOU,
                    arrow_dedup_iou=SAM3_ARROW_DEDUP_IOU,
                    shape_image_iou=SAM3_SHAPE_IMAGE_IOU,
                )
                all_results = filter_sam3_items_contained_by_images(
                    all_results,
                    image_groups=["image"],
                    contain_threshold=0.85,
                )

        return all_results
    except Exception as e:
        log.warning(f"[paper2drawio_sam3] SAM3 grouped predict failed: {e}")
        return []


def _build_elements_from_sam3(
    results: List[Dict[str, Any]],
    image_bgr: np.ndarray,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    image_area = float(h * w)
    shapes: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    icon_dir = out_dir / "sam3_icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    shape_bboxes: List[List[int]] = []
    repaired_image_bboxes: List[List[int]] = []

    def _bbox_contain_ratio(inner: List[int], outer: List[int]) -> float:
        xA = max(inner[0], outer[0])
        yA = max(inner[1], outer[1])
        xB = min(inner[2], outer[2])
        yB = min(inner[3], outer[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area_inner = _bbox_area(inner)
        if area_inner <= 0:
            return 0.0
        return inter / float(area_inner)

    def _refine_low_coverage_image_mask(mask: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, List[int], bool]:
        bbox_area = _bbox_area(bbox)
        if bbox_area < IMAGE_MASK_REPAIR_MIN_BBOX_AREA:
            return mask, (mask_to_bbox(mask) or bbox), False

        mask_area = int(np.count_nonzero(mask))
        if mask_area <= 0:
            return mask, bbox, False

        cover_ratio = float(mask_area) / float(max(1, bbox_area))
        if cover_ratio >= IMAGE_MASK_REPAIR_LOW_COVERAGE_THRESHOLD:
            return mask, (mask_to_bbox(mask) or bbox), False

        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = IMAGE_MASK_REPAIR_PAD_PX
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return mask, (mask_to_bbox(mask) or bbox), False

        mask_u8 = mask.astype(np.uint8)
        seed_dil = cv2.dilate(mask_u8, np.ones((7, 7), np.uint8), iterations=1)
        seed_ero = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)

        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
        gc_mask[seed_dil > 0] = cv2.GC_PR_FGD
        gc_mask[seed_ero > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        except Exception:
            return mask, (mask_to_bbox(mask) or bbox), False

        refined = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        roi = np.zeros_like(refined, dtype=bool)
        roi[y1:y2, x1:x2] = True
        refined = refined & roi
        if int(np.count_nonzero(refined)) <= 0:
            return mask, (mask_to_bbox(mask) or bbox), False

        # 保留与原前景相连的连通域，或面积足够大的连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(refined, dtype=bool)
        min_component_area = max(120, int(mask_area * 0.08))
        for label in range(1, num_labels):
            comp = labels == label
            comp_area = int(stats[label, cv2.CC_STAT_AREA])
            overlap_seed = int(np.count_nonzero(comp & (seed_dil > 0)))
            if overlap_seed > 0 or comp_area >= min_component_area:
                kept |= comp
        if int(np.count_nonzero(kept)) > 0:
            refined = kept

        refined_area = int(np.count_nonzero(refined))
        gain_ratio = float(refined_area - mask_area) / float(max(1, mask_area))
        refined_cover_on_orig_bbox = float(refined_area) / float(max(1, bbox_area))
        if gain_ratio < IMAGE_MASK_REPAIR_MIN_GAIN_RATIO:
            return mask, (mask_to_bbox(mask) or bbox), False
        if refined_cover_on_orig_bbox > IMAGE_MASK_REPAIR_MAX_COVERAGE_ON_ORIG_BBOX:
            return mask, (mask_to_bbox(mask) or bbox), False

        refined_bbox = mask_to_bbox(refined) or bbox
        return refined, refined_bbox, True

    for idx, item in enumerate(results):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        group = item.get("group", "")
        bbox_area = _bbox_area(bbox)
        area = int(item.get("area") or bbox_area)
        area_ratio = float(area) / image_area if image_area > 0 else 0
        if area_ratio > 0.98:
            continue
        bbox_area_ratio = float(bbox_area) / image_area if image_area > 0 else 0
        if group == "image" and bbox_area_ratio > MAX_IMAGE_BBOX_AREA_RATIO:
            continue
        prompt = item.get("prompt", "")

        mask = None
        if item.get("mask"):
            mask = decode_sam3_mask(item.get("mask"))
            if mask is not None:
                mask = normalize_mask(mask.astype(bool), (h, w))

        if group in {"shape", "background"} and mask is not None:
            shape_type = _shape_type_from_prompt(prompt)
            fill_hex, stroke_hex = sample_fill_stroke(image_bgr, mask)
            shapes.append({
                "id": f"s{idx}",
                "kind": "shape",
                "shape_type": shape_type,
                "bbox_px": bbox,
                "fill": fill_hex,
                "stroke": stroke_hex,
                "text": "",
                "text_color": extract_text_color(image_bgr, bbox),
                "font_size": None,
                "area": area,
                "group": group,
                "prompt": prompt,
            })
            shape_bboxes.append(bbox)
            continue

        # 若某大图标已采用 bbox 回退，跳过其附近被拆开的微小碎片，避免重复图元
        if group == "image" and _bbox_area(bbox) <= IMAGE_FRAGMENT_SKIP_MAX_AREA and repaired_image_bboxes:
            should_skip_fragment = False
            for repaired_bbox in repaired_image_bboxes:
                contain = _bbox_contain_ratio(bbox, repaired_bbox)
                if contain >= IMAGE_FRAGMENT_CONTAIN_THRESHOLD or bbox_iou_px(bbox, repaired_bbox) > 0.15:
                    should_skip_fragment = True
                    break
            if should_skip_fragment:
                continue

        if mask is not None:
            work_mask = mask
            if group == "image":
                work_mask, repaired_bbox, is_repaired = _refine_low_coverage_image_mask(mask, bbox)
                if is_repaired:
                    repaired_image_bboxes.append(repaired_bbox)

            out_path = icon_dir / f"sam3_{group}_{idx}.png"
            save_masked_rgba(image_bgr, work_mask, str(out_path), dilate_px=1)
            effective_bbox = mask_to_bbox(work_mask) or bbox

            img_area_ratio = float(_bbox_area(effective_bbox)) / image_area
            if img_area_ratio < MIN_IMAGE_AREA_RATIO:
                continue
            # skip if overlaps a known shape heavily
            skip = False
            for sb in shape_bboxes:
                if bbox_iou_px(effective_bbox, sb) > 0.75:
                    skip = True
                    break
            if skip:
                continue
            images.append({
                "id": f"i{idx}",
                "kind": "image",
                "bbox_px": effective_bbox,
                "image_path": str(out_path),
                "area": area,
                "group": group,
            })
        else:
            out_path = icon_dir / f"sam3_{group}_{idx}.png"
            x1, y1, x2, y2 = bbox
            crop = image_bgr[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                cv2.imwrite(str(out_path), crop)
            images.append({
                "id": f"i{idx}",
                "kind": "image",
                "bbox_px": bbox,
                "image_path": str(out_path),
                "area": area,
                "group": group,
            })

    shapes.sort(key=lambda s: s.get("area", 0), reverse=True)
    images.sort(key=lambda s: s.get("area", 0), reverse=True)
    total = len(shapes) + len(images)
    if total > MAX_DRAWIO_ELEMENTS:
        keep = max(0, MAX_DRAWIO_ELEMENTS - len(shapes))
        if keep < len(images):
            images = images[:keep]
    return shapes + images


# ==================== WORKFLOW ====================
@register("paper2drawio_sam3")
@register("paper2drawio_visual")
def create_paper2drawio_sam3_graph() -> GenericGraphBuilder:
    """
    Paper2Drawio visual workflow: start from an existing image and rebuild
    it into editable draw.io XML with OCR + SAM3 segmentation.
    """
    builder = GenericGraphBuilder(state_model=Paper2DrawioState, entry_point="_start_")

    def _init_node(state: Paper2DrawioState) -> Paper2DrawioState:
        _ensure_result_path(state)
        return state

    def _input_node(state: Paper2DrawioState) -> Paper2DrawioState:
        base_dir = Path(_ensure_result_path(state))
        img_path = state.paper_file or ""
        if img_path and _is_image_path(img_path) and os.path.exists(img_path):
            state.temp_data["input_image_path"] = str(Path(img_path).resolve())
            return state
        if img_path and img_path.lower().endswith(".pdf") and os.path.exists(img_path):
            out_path = base_dir / "input_page_1.png"
            rendered = _render_pdf_first_page(img_path, str(out_path))
            if rendered:
                state.temp_data["input_image_path"] = str(Path(rendered).resolve())
                return state
        # fallback: try text_content as image path
        if state.text_content and _is_image_path(state.text_content) and os.path.exists(state.text_content):
            state.temp_data["input_image_path"] = str(Path(state.text_content).resolve())
            return state
        log.error("[paper2drawio_sam3] No valid image input provided")
        return state

    async def _text_node(state: Paper2DrawioState) -> Paper2DrawioState:
        state.text_content = _extract_paper_content(state)
        img_path = state.temp_data.get("input_image_path")
        if not img_path or not os.path.exists(img_path):
            state.temp_data["text_blocks"] = []
            return state

        ocr_api_url, ocr_api_key = get_ocr_api_credentials()
        api_key = ocr_api_key
        chat_api_url = ocr_api_url
        if not chat_api_url or not api_key:
            log.warning("[paper2drawio_sam3] VLM OCR not configured")
            state.temp_data["text_blocks"] = []
            return state

        try:
            temp_state = copy.copy(state)
            if getattr(temp_state, "request", None):
                temp_state.request = copy.copy(state.request)
                temp_state.request.chat_api_url = chat_api_url
                temp_state.request.api_key = api_key
                temp_state.request.chat_api_key = api_key

            try:
                vlm_timeout = int(os.getenv("VLM_OCR_TIMEOUT", "120"))
            except ValueError:
                vlm_timeout = 120
            agent = create_vlm_agent(
                name="ImageTextBBoxAgent",
                model_name="qwen-vl-ocr-2025-11-20",
                chat_api_url=chat_api_url,
                max_tokens=4096,
                vlm_mode="ocr",
                additional_params={"input_image": img_path, "timeout": vlm_timeout},
            )
            new_state = await agent.execute(temp_state)
            bbox_res = getattr(new_state, "bbox_result", [])
        except Exception as e:
            log.warning(f"[paper2drawio_sam3][VLM] OCR failed: {e}")
            bbox_res = []

        if isinstance(bbox_res, dict) and "raw" in bbox_res:
            bbox_res = bbox_res.get("raw") or ""
        if isinstance(bbox_res, str):
            try:
                bbox_res = robust_parse_json(bbox_res)
            except Exception as e:
                log.warning(f"[paper2drawio_sam3][VLM] parse failed: {e}")
                bbox_res = []

        if not isinstance(bbox_res, list):
            bbox_res = extract_bbox_items(bbox_res)
        if not isinstance(bbox_res, list):
            bbox_res = []

        try:
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size
            text_blocks = _vectorize_text_blocks(bbox_res, w, h)
        except Exception as e:
            log.warning(f"[paper2drawio_sam3][VLM] vectorize failed: {e}")
            text_blocks = []

        log.info(f"[paper2drawio_sam3][VLM] text_blocks={len(text_blocks)} image={img_path}")
        state.temp_data["text_blocks"] = text_blocks
        return state

    async def _segment_hint_node(state: Paper2DrawioState) -> Paper2DrawioState:
        img_path = state.temp_data.get("input_image_path")
        state.temp_data["sam3_segment_hints"] = []
        state.temp_data["sam3_segment_hints_raw"] = {}
        if not img_path or not os.path.exists(img_path):
            return state

        text_blocks = state.temp_data.get("text_blocks", []) or []
        try:
            hints, raw_result, _, _ = await generate_sam3_segment_hints(
                state=state,
                image_path=img_path,
                text_blocks=text_blocks,
                env_prefix="PAPER2DRAWIO_SEGMENT_HINT",
                base_image_prompts=IMAGE_PROMPT,
                base_recall_prompts=IMAGE_PROMPT_RECALL,
                extra_blocked_prompts=SHAPE_PROMPT + ARROW_PROMPT + BACKGROUND_PROMPT,
                log_prefix="[paper2drawio_sam3][segment_hint]",
            )
        except Exception as e:
            log.warning(f"[paper2drawio_sam3][segment_hint] VLM credentials unavailable: {e}")
            return state

        state.temp_data["sam3_segment_hints"] = hints
        state.temp_data["sam3_segment_hints_raw"] = raw_result
        return state

    async def _sam3_node(state: Paper2DrawioState) -> Paper2DrawioState:
        img_path = state.temp_data.get("input_image_path")
        if not img_path:
            return state
        client = get_sam3_client()
        if client is None:
            log.error("[paper2drawio_sam3] SAM3 endpoints not configured")
            state.temp_data["sam3_results"] = []
            return state
        extra_image_prompts = state.temp_data.get("sam3_segment_hints", []) or []
        results = _sam3_predict_groups(client, img_path, extra_image_prompts=extra_image_prompts)
        group_counts: Dict[str, int] = {}
        for item in results:
            group = str(item.get("group") or "unknown")
            group_counts[group] = group_counts.get(group, 0) + 1
        log.info(
            f"[paper2drawio_sam3] sam3_results={len(results)} "
            f"groups={json.dumps(group_counts, ensure_ascii=False, sort_keys=True)} "
            f"extra_image_prompts={json.dumps(extra_image_prompts, ensure_ascii=False)} image={img_path}"
        )
        state.temp_data["sam3_results"] = results
        try:
            base_dir = Path(_ensure_result_path(state))
            debug_path = base_dir / "sam3_results.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_image_path": img_path,
                        "sam3_segment_hints": extra_image_prompts,
                        "sam3_results": results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            log.warning(f"[paper2drawio_sam3_vl] Failed to write sam3_results: {e}")
        return state

    async def _build_elements_node(state: Paper2DrawioState) -> Paper2DrawioState:
        img_path = state.temp_data.get("input_image_path")
        if not img_path:
            return state
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            log.error(f"[paper2drawio_sam3] Failed to read image: {img_path}")
            state.temp_data["drawio_elements"] = []
            return state
        results = state.temp_data.get("sam3_results", []) or []
        base_dir = Path(_ensure_result_path(state))
        elements = _build_elements_from_sam3(results, image_bgr, base_dir)
        fallback_hide_text_blocks = False
        if not elements:
            text_blocks = state.temp_data.get("text_blocks", []) or []
            elements, fallback_hide_text_blocks = _build_visual_fallback_elements(
                image_bgr=image_bgr,
                text_blocks=text_blocks,
                out_dir=base_dir,
            )
            log.warning(
                f"[paper2drawio_sam3] SAM3 produced no drawable elements; "
                f"using visual fallback background (hide_text_blocks={fallback_hide_text_blocks})"
            )
        else:
            shape_count = sum(1 for item in elements if item.get("kind") == "shape")
            image_count = sum(1 for item in elements if item.get("kind") == "image")
            log.info(f"[paper2drawio_sam3] drawio_elements shape={shape_count} image={image_count}")
        state.temp_data["drawio_elements"] = elements
        state.temp_data["fallback_hide_text_blocks"] = fallback_hide_text_blocks
        return state

    async def _render_xml_node(state: Paper2DrawioState) -> Paper2DrawioState:
        img_path = state.temp_data.get("input_image_path")
        if not img_path or not os.path.exists(img_path):
            state.drawio_xml = ""
            return state

        with Image.open(img_path) as img:
            page_width, page_height = img.size

        elements = state.temp_data.get("drawio_elements", []) or []
        text_blocks = state.temp_data.get("text_blocks", []) or []
        if state.temp_data.get("fallback_hide_text_blocks"):
            text_blocks = []

        cells: List[str] = []
        id_counter = 2

        # background shapes first
        bg_shapes = [e for e in elements if e.get("kind") == "shape" and e.get("group") == "background"]
        other_shapes = [e for e in elements if e.get("kind") == "shape" and e.get("group") != "background"]
        images = [e for e in elements if e.get("kind") == "image"]

        for el in bg_shapes + other_shapes:
            style = _shape_style(
                el.get("shape_type", "rect"),
                el.get("fill", "#ffffff"),
                el.get("stroke", "#000000"),
                font_size=el.get("font_size"),
                font_color=el.get("text_color"),
            )
            cells.append(_build_mxcell(str(id_counter), el.get("text", ""), style, el["bbox_px"]))
            id_counter += 1

        for el in images:
            img_path = el.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
            with open(img_path, "rb") as f:
                data = f.read()
            data_uri = "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
            style = _image_style(data_uri)
            cells.append(_build_mxcell(str(id_counter), "", style, el["bbox_px"]))
            id_counter += 1

        for block in text_blocks:
            geo = block.get("geometry", {})
            x = float(geo.get("x", 0))
            y = float(geo.get("y", 0))
            w = float(geo.get("width", 0))
            h = float(geo.get("height", 0))
            bbox = [x, y, x + w, y + h]
            style = _text_style_from_block(block)
            value = block.get("text", "")
            cells.append(_build_mxcell(str(id_counter), value, style, bbox, is_latex=block.get("is_latex", False)))
            id_counter += 1

        xml_cells = "\n".join(cells)
        full_xml = wrap_xml(xml_cells, page_width=page_width, page_height=page_height)

        base_dir = Path(_ensure_result_path(state))
        out_path = base_dir / "paper2drawio_sam3.drawio"
        out_path.write_text(full_xml, encoding="utf-8")

        state.drawio_xml = full_xml
        state.output_xml_path = str(out_path)
        return state

    nodes = {
        "_start_": _init_node,
        "input": _input_node,
        "text_ocr": _text_node,
        "segment_hint": _segment_hint_node,
        "sam3": _sam3_node,
        "build_elements": _build_elements_node,
        "render_xml": _render_xml_node,
        "_end_": lambda s: s,
    }

    edges = [
        ("input", "text_ocr"),
        ("text_ocr", "segment_hint"),
        ("segment_hint", "sam3"),
        ("sam3", "build_elements"),
        ("build_elements", "render_xml"),
        ("render_xml", "_end_"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    builder.add_edge("_start_", "input")
    return builder
