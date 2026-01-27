"""
Draw.io XML 工具函数
提供 XML 包装、提取、验证和编辑功能
"""
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple


# draw.io XML 模板
DRAWIO_WRAPPER_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="{modified}" agent="Paper2Any" version="24.0.0">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1434" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{cells}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''


def wrap_xml(cells_xml: str, modified: str = "") -> str:
    """
    将 mxCell 元素包装为完整的 draw.io XML

    Args:
        cells_xml: 仅包含 mxCell 元素的 XML 字符串
        modified: 修改时间戳

    Returns:
        完整的 draw.io XML 文件内容
    """
    from datetime import datetime
    if not modified:
        modified = datetime.now().isoformat()

    # 确保每行有适当的缩进
    lines = cells_xml.strip().split('\n')
    indented_lines = ['        ' + line.strip() for line in lines if line.strip()]
    indented_cells = '\n'.join(indented_lines)

    return DRAWIO_WRAPPER_TEMPLATE.format(
        modified=modified,
        cells=indented_cells
    )


def extract_cells(full_xml: str) -> str:
    """
    从完整的 draw.io XML 中提取 mxCell 元素

    Args:
        full_xml: 完整的 draw.io XML

    Returns:
        仅包含 mxCell 元素的 XML 字符串（不含 id="0" 和 id="1"）
    """
    try:
        root = ET.fromstring(full_xml)
        cells = []

        # 查找所有 mxCell 元素
        for cell in root.iter('mxCell'):
            cell_id = cell.get('id', '')
            # 跳过根单元格
            if cell_id in ('0', '1'):
                continue
            cells.append(ET.tostring(cell, encoding='unicode'))

        return '\n'.join(cells)
    except ET.ParseError as e:
        # 如果解析失败，尝试用正则提取
        pattern = r'<mxCell[^>]*id="(?!0|1")[^"]*"[^>]*>.*?</mxCell>|<mxCell[^>]*id="(?!0|1")[^"]*"[^/]*/>'
        matches = re.findall(pattern, full_xml, re.DOTALL)
        return '\n'.join(matches)


def validate_xml(cells_xml: str) -> Tuple[bool, List[str]]:
    """
    验证 mxCell XML 的结构

    Args:
        cells_xml: mxCell 元素的 XML 字符串

    Returns:
        (is_valid, errors) 元组
    """
    errors = []

    # 检查是否包含禁止的包装标签
    forbidden_tags = ['<mxfile', '<mxGraphModel', '<root>', '<diagram']
    for tag in forbidden_tags:
        if tag in cells_xml:
            errors.append(f"包含禁止的标签: {tag}")

    # 检查是否包含根单元格
    if re.search(r'<mxCell[^>]*id=["\']0["\']', cells_xml):
        errors.append("包含禁止的根单元格 id='0'")
    if re.search(r'<mxCell[^>]*id=["\']1["\']', cells_xml):
        errors.append("包含禁止的根单元格 id='1'")

    # 尝试解析 XML
    try:
        # 包装后解析以验证结构
        wrapped = f"<root>{cells_xml}</root>"
        root = ET.fromstring(wrapped)

        # 检查 ID 唯一性
        ids = set()
        for cell in root.findall('.//mxCell'):
            cell_id = cell.get('id')
            if cell_id in ids:
                errors.append(f"重复的 ID: {cell_id}")
            ids.add(cell_id)

            # 检查必要属性
            if not cell.get('parent'):
                errors.append(f"单元格 {cell_id} 缺少 parent 属性")

    except ET.ParseError as e:
        errors.append(f"XML 解析错误: {str(e)}")

    return len(errors) == 0, errors


def sanitize_cells_xml(cells_xml: str) -> str:
    """
    Clean mxCell XML output to reduce common rendering failures.
    """
    if not cells_xml:
        return ""

    cleaned = cells_xml.strip()

    # Remove markdown code fences if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

    # Strip XML comments.
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL).strip()

    # Escape bare ampersands (keep valid entities).
    cleaned = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)",
        "&amp;",
        cleaned,
    )

    return cleaned.strip()


def apply_edits(
    current_xml: str,
    operations: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    应用编辑操作到现有 XML

    Args:
        current_xml: 当前的 mxCell XML
        operations: 编辑操作列表
            [{"operation": "update|add|delete", "cell_id": "...", "new_xml": "..."}]

    Returns:
        (new_xml, errors) 元组
    """
    errors = []

    try:
        # 包装后解析
        wrapped = f"<root>{current_xml}</root>"
        root = ET.fromstring(wrapped)

        for op in operations:
            operation = op.get('operation')
            cell_id = op.get('cell_id')
            new_xml = op.get('new_xml', '')

            if operation == 'delete':
                # 删除单元格
                cell = root.find(f".//mxCell[@id='{cell_id}']")
                if cell is not None:
                    parent = root.find(f".//mxCell[@id='{cell_id}']/..")
                    if parent is not None:
                        parent.remove(cell)
                else:
                    errors.append(f"未找到要删除的单元格: {cell_id}")

            elif operation == 'update':
                # 更新单元格
                cell = root.find(f".//mxCell[@id='{cell_id}']")
                if cell is not None:
                    parent = root.find(f".//mxCell[@id='{cell_id}']/..")
                    idx = list(parent).index(cell)
                    parent.remove(cell)
                    new_cell = ET.fromstring(new_xml)
                    parent.insert(idx, new_cell)
                else:
                    errors.append(f"未找到要更新的单元格: {cell_id}")

            elif operation == 'add':
                # 添加新单元格
                new_cell = ET.fromstring(new_xml)
                root.append(new_cell)
            else:
                errors.append(f"未知操作: {operation}")

        # 提取结果
        result_cells = []
        for cell in root.findall('mxCell'):
            result_cells.append(ET.tostring(cell, encoding='unicode'))

        return '\n'.join(result_cells), errors

    except ET.ParseError as e:
        errors.append(f"XML 解析错误: {str(e)}")
        return current_xml, errors


def get_cell_ids(cells_xml: str) -> List[str]:
    """获取所有单元格 ID"""
    ids = []
    try:
        wrapped = f"<root>{cells_xml}</root>"
        root = ET.fromstring(wrapped)
        for cell in root.findall('.//mxCell'):
            cell_id = cell.get('id')
            if cell_id:
                ids.append(cell_id)
    except ET.ParseError:
        # 用正则提取
        pattern = r'<mxCell[^>]*id=["\']([^"\']+)["\']'
        ids = re.findall(pattern, cells_xml)
    return ids


def generate_next_id(cells_xml: str) -> str:
    """生成下一个可用的 ID"""
    ids = get_cell_ids(cells_xml)
    max_num = 1
    for id_str in ids:
        try:
            num = int(id_str)
            max_num = max(max_num, num)
        except ValueError:
            continue
    return str(max_num + 1)
