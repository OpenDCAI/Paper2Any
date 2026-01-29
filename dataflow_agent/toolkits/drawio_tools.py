"""
Draw.io XML 工具函数
提供 XML 包装、提取、验证和编辑功能
"""
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable


# draw.io XML 模板
DRAWIO_WRAPPER_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="{modified}" agent="Paper2Any" version="24.0.0">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1434" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_width}" pageHeight="{page_height}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{cells}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''


def wrap_xml(cells_xml: str, modified: str = "", page_width: int | float = 850, page_height: int | float = 1100) -> str:
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
        cells=indented_cells,
        page_width=int(round(page_width)),
        page_height=int(round(page_height)),
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


def _iter_vertices(root: ET.Element) -> Iterable[ET.Element]:
    for cell in root.findall('.//mxCell'):
        if cell.get('vertex') == '1':
            yield cell


def _get_geometry(cell: ET.Element) -> Tuple[ET.Element, float, float, float, float]:
    geom = cell.find('mxGeometry')
    if geom is None:
        geom = ET.SubElement(cell, 'mxGeometry')
        geom.set('as', 'geometry')
    x = float(geom.get('x', '0') or 0)
    y = float(geom.get('y', '0') or 0)
    w = float(geom.get('width', '120') or 120)
    h = float(geom.get('height', '60') or 60)
    return geom, x, y, w, h


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], padding: float) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + padding <= bx or
        bx + bw + padding <= ax or
        ay + ah + padding <= by or
        by + bh + padding <= ay
    )


def _style_has(cell: ET.Element, token: str) -> bool:
    style = (cell.get('style') or "")
    return token in style


def _set_geometry(geom: ET.Element, x: float, y: float, w: float, h: float) -> None:
    geom.set('x', f"{x:.0f}")
    geom.set('y', f"{y:.0f}")
    geom.set('width', f"{w:.0f}")
    geom.set('height', f"{h:.0f}")


def _center(cell: ET.Element) -> Tuple[float, float]:
    _, x, y, w, h = _get_geometry(cell)
    return x + w / 2.0, y + h / 2.0


def _grid_layout(
    cells: List[ET.Element],
    start_x: float,
    start_y: float,
    max_width: float,
    gap_x: float,
    gap_y: float,
) -> Tuple[float, float]:
    if not cells:
        return start_x, start_y
    cols = max(1, int(max_width // (gap_x + 1)))
    col = 0
    row = 0
    max_x = start_x
    max_y = start_y
    for cell in cells:
        geom, _, _, w, h = _get_geometry(cell)
        x = start_x + col * gap_x
        y = start_y + row * gap_y
        _set_geometry(geom, x, y, w, h)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        col += 1
        if col >= cols:
            col = 0
            row += 1
    return max_x, max_y


def _layout_children_in_container(
    container: ET.Element,
    children: List[ET.Element],
    padding: float,
    gap: float
) -> None:
    if not children:
        return
    geom, x, y, w, h = _get_geometry(container)
    inner_x = x + padding
    inner_y = y + padding
    max_width = max(120.0, w - padding * 2)
    max_x, max_y = _grid_layout(
        children,
        inner_x,
        inner_y,
        max_width=max_width,
        gap_x=max(140.0, gap),
        gap_y=max(90.0, gap),
    )
    needed_w = max(w, (max_x - x) + padding)
    needed_h = max(h, (max_y - y) + padding)
    _set_geometry(geom, x, y, needed_w, needed_h)


def _layout_top_level(
    diagram_type: str,
    vertices: List[ET.Element],
    margin: float,
    canvas_width: float,
    canvas_height: float,
    gap: float
) -> None:
    if not vertices:
        return
    diagram_type = (diagram_type or "auto").lower()
    if diagram_type == "flowchart":
        x = margin
        y = margin
        for cell in vertices:
            geom, _, _, w, h = _get_geometry(cell)
            _set_geometry(geom, x, y, w, h)
            y += h + gap
        return

    if diagram_type == "sequence":
        x = margin
        y = margin
        for cell in vertices:
            geom, _, _, w, h = _get_geometry(cell)
            _set_geometry(geom, x, y, w, h)
            x += max(w + gap, 160)
        return

    if diagram_type == "mindmap":
        root = max(vertices, key=lambda c: (_get_geometry(c)[2] * _get_geometry(c)[3]))
        others = [v for v in vertices if v is not root]
        geom, _, _, w, h = _get_geometry(root)
        center_x = canvas_width / 2.0 - w / 2.0
        center_y = canvas_height / 2.0 - h / 2.0
        _set_geometry(geom, center_x, center_y, w, h)
        if not others:
            return
        import math
        radius = max(200.0, min(canvas_width, canvas_height) / 3.0)
        for idx, cell in enumerate(others):
            angle = (2 * math.pi * idx) / max(1, len(others))
            geom, _, _, cw, ch = _get_geometry(cell)
            x = center_x + w / 2.0 + math.cos(angle) * radius - cw / 2.0
            y = center_y + h / 2.0 + math.sin(angle) * radius - ch / 2.0
            _set_geometry(geom, x, y, cw, ch)
        return

    if diagram_type == "er":
        _grid_layout(vertices, margin, margin, canvas_width - margin * 2, gap_x=220, gap_y=160)
        return

    _grid_layout(vertices, margin, margin, canvas_width - margin * 2, gap_x=220, gap_y=160)


def _add_edge_waypoints(root: ET.Element, cell_by_id: Dict[str, ET.Element]) -> None:
    for cell in root.findall('.//mxCell'):
        if cell.get('edge') != '1':
            continue
        source = cell.get('source')
        target = cell.get('target')
        if not source or not target:
            continue
        s_cell = cell_by_id.get(source)
        t_cell = cell_by_id.get(target)
        if s_cell is None or t_cell is None:
            continue
        geom = cell.find('mxGeometry')
        if geom is None:
            geom = ET.SubElement(cell, 'mxGeometry')
            geom.set('relative', '1')
            geom.set('as', 'geometry')
        # Skip if already has waypoints
        if geom.findall('mxPoint') or geom.find("Array[@as='points']") is not None:
            continue
        sx, sy = _center(s_cell)
        tx, ty = _center(t_cell)
        mid_x = (sx + tx) / 2.0
        # Add two points to encourage orthogonal routing (must be inside Array as="points")
        points = ET.SubElement(geom, 'Array')
        points.set('as', 'points')
        p1 = ET.SubElement(points, 'mxPoint')
        p1.set('x', f"{mid_x:.0f}")
        p1.set('y', f"{sy:.0f}")
        p2 = ET.SubElement(points, 'mxPoint')
        p2.set('x', f"{mid_x:.0f}")
        p2.set('y', f"{ty:.0f}")


def resolve_overlaps(
    cells_xml: str,
    diagram_type: str = "auto",
    canvas_width: float = 800,
    canvas_height: float = 600,
    margin: float = 40,
    gap: float = 60,
    max_attempts: int = 200
) -> str:
    """
    Resolve vertex overlaps by incrementally shifting nodes on a simple grid.

    This is a conservative post-process: it preserves original order and only moves
    nodes when overlaps are detected.
    """
    if not cells_xml:
        return ""

    try:
        wrapped = f"<root>{cells_xml}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return cells_xml

    vertices = list(_iter_vertices(root))
    if not vertices:
        return cells_xml
    cell_by_id: Dict[str, ET.Element] = {cell.get('id'): cell for cell in root.findall('.//mxCell') if cell.get('id')}

    children_by_parent: Dict[str, List[ET.Element]] = {}
    for cell in vertices:
        parent = cell.get('parent') or "1"
        children_by_parent.setdefault(parent, []).append(cell)

    # Layout children inside containers first
    for parent_id, children in list(children_by_parent.items()):
        if parent_id == "1":
            continue
        container = cell_by_id.get(parent_id)
        if container is None:
            continue
        if container.get('vertex') == '1' or _style_has(container, "swimlane"):
            _layout_children_in_container(container, children, padding=20, gap=gap)

    # Layout top-level vertices by diagram type
    top_level = [c for c in vertices if (c.get('parent') or "1") == "1"]
    _layout_top_level(diagram_type, top_level, margin, canvas_width, canvas_height, gap)

    # Final overlap pass (conservative)
    placed: List[Tuple[float, float, float, float]] = []
    for cell in vertices:
        geom, x, y, w, h = _get_geometry(cell)
        attempts = 0
        while any(_overlaps((x, y, w, h), other, gap / 2) for other in placed):
            attempts += 1
            x += gap / 2
            if x + w > canvas_width - margin:
                x = margin
                y += gap / 2
            if attempts >= max_attempts:
                break
        _set_geometry(geom, x, y, w, h)
        placed.append((x, y, w, h))

    _add_edge_waypoints(root, cell_by_id)

    result_cells = [ET.tostring(cell, encoding='unicode') for cell in root.findall('mxCell')]
    return '\n'.join(result_cells)


def export_drawio_png(
    cells_xml: str,
    output_path: str,
    drawio_bin: Optional[str] = None,
    timeout: int = 60,
) -> Tuple[bool, str]:
    """
    Render draw.io XML to PNG using draw.io CLI (server-side renderer).

    Requires draw.io/diagrams.net CLI installed and available in PATH.

    Returns:
        (ok, message) - ok True if PNG created, else False with error message.
    """
    if not cells_xml:
        return False, "empty xml"

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    full_xml = cells_xml if "<mxfile" in cells_xml else wrap_xml(cells_xml)

    # Detect CLI binary
    bin_candidates = [
        drawio_bin,
        os.getenv("DRAWIO_EXPORT_BIN"),
        os.getenv("DRAWIO_BIN"),
        os.getenv("DRAWIO_CLI"),
        "drawio",
        "draw.io",
    ]
    cli = next((b for b in bin_candidates if b and shutil.which(b)), None)
    if not cli:
        return False, "draw.io CLI not found (set DRAWIO_EXPORT_BIN or install draw.io CLI)"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "diagram.drawio"
            tmp_path.write_text(full_xml, encoding="utf-8")

            cmd = [
                cli,
                "--export",
                "--format",
                "png",
                "--output",
                str(output_file),
                str(tmp_path),
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return False, (proc.stderr or proc.stdout or "export failed")

            if not output_file.exists():
                return False, "export finished but output file missing"
    except Exception as e:
        return False, f"export exception: {e}"

    return True, ""


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
