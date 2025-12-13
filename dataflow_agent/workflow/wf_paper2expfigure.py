"""
paper2expfigure workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
从 PDF 论文中提取表格并生成统计图的完整工作流

工作流程：
1. PDF → 图片 (pdf_to_images_node)
2. 图片 → MinerU 识别 (mineru_extract_node)
3. 提取表格数据 (table_extractor_node)
4. 提取论文核心思想 (paper_idea_extractor_node)
5. 智能推荐图表类型和生成代码 (code_executor_node)
   - 调用 chart_type_recommender Agent 推荐图表类型
   - 调用 chart_code_generator Agent 生成 matplotlib 代码
   - 执行代码生成图表
"""

from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dataflow_agent.state import Paper2ExpFigureState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_simple_agent
from dataflow_agent.agentroles.paper2any_agents.chart_type_recommender import create_chart_type_recommender
from dataflow_agent.agentroles.paper2any_agents.chart_code_generator import create_chart_code_generator
from dataflow_agent.toolkits.tool_manager import get_tool_manager
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import (
    pdf_to_pil_images,
    extract_tables_from_mineru_results,
    extract_text_from_mineru_results,
    execute_matplotlib_code,
)
from dataflow_agent.toolkits.imtool.mineru_tool import run_aio_two_step_extract

log = get_logger(__name__)

# ==============================================================
# 辅助方法
# ==============================================================

def _generate_table_understanding(
    table_id: str,
    caption: str,
    headers: List[str],
    rows: List[List[str]]
) -> Dict[str, Any]:
    """
    生成对表格内容的理解和描述。
    
    这是一个简单的启发式分析，用于在没有 LLM 的情况下提供基本的表格理解。
    在实际应用中，可以调用 LLM 来生成更智能的理解。
    """
    # 分析表格结构
    num_rows = len(rows)
    num_cols = len(headers)
    
    # 分析数据类型
    data_types = []
    for col_idx, header in enumerate(headers):
        if col_idx >= num_cols:
            break
        
        # 尝试判断该列的数据类型
        sample_values = [row[col_idx] for row in rows[:min(5, num_rows)] if col_idx < len(row)]
        
        is_numeric = all(
            isinstance(v, (int, float)) or 
            (isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit())
            for v in sample_values if v
        )
        
        data_types.append({
            "column": header,
            "type": "numeric" if is_numeric else "categorical",
            "sample_values": sample_values[:3]  # 保存前3个示例值
        })
    
    # 生成描述性统计
    summary = f"这是一个 {num_rows} 行 x {num_cols} 列的表格。"
    
    if caption:
        summary += f" 表格标题：{caption}。"
    
    # 分析列类型
    numeric_cols = [dt['column'] for dt in data_types if dt['type'] == 'numeric']
    categorical_cols = [dt['column'] for dt in data_types if dt['type'] == 'categorical']
    
    if numeric_cols:
        summary += f" 包含 {len(numeric_cols)} 个数值列：{', '.join(numeric_cols[:3])}等。"
    if categorical_cols:
        summary += f" 包含 {len(categorical_cols)} 个分类列：{', '.join(categorical_cols[:3])}等。"
    
    # 生成内容描述
    content_description = ""
    if headers:
        if num_rows > 0:
            first_col_values = [row[0] for row in rows[:5] if len(row) > 0]
            if first_col_values:
                content_description = f"第一列 '{headers[0]}' 包含值：{', '.join(map(str, first_col_values[:3]))}等。"
    
    # 生成可视化建议
    visualization_suggestions = []
    if len(numeric_cols) >= 2:
        visualization_suggestions.append("可以使用散点图展示数值列之间的关系")
    if categorical_cols and numeric_cols:
        visualization_suggestions.append("可以使用条形图或柱状图比较不同分类的数值")
    if num_rows > 10:
        visualization_suggestions.append("数据量较大，建议选择代表性样本或聚合显示")
    
    return {
        "summary": summary,
        "content_description": content_description,
        "data_structure": {
            "total_rows": num_rows,
            "total_cols": num_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        "data_types": data_types,
        "visualization_suggestions": visualization_suggestions,
        "key_insights": [
            f"表格共有 {num_rows} 个数据项",
            f"包含 {len(numeric_cols)} 个数值指标和 {len(categorical_cols)} 个分类维度",
        ]
    }

@register("paper2expfigure")
def create_paper2expfigure_graph() -> GenericGraphBuilder:
    """
    Paper2ExpFigure Workflow: 从 PDF 提取表格并生成统计图
    
    命令: dfa run --wf paper2expfigure
    """
    builder = GenericGraphBuilder(
        state_model=Paper2ExpFigureState,
        entry_point="_start_"
    )

    # ======================================================================
    # PRE-TOOLS: 为 Agent 提供输入数据
    # ======================================================================
    
    @builder.pre_tool("paper_content", "paper_idea_extractor")
    def _get_paper_content(state: Paper2ExpFigureState) -> str:
        """
        从 MinerU 结果或 PDF 中提取文本内容，供 paper_idea_extractor 使用
        """
        # 优先从 MinerU 结果中提取
        if hasattr(state, 'temp_data') and 'mineru_items' in state.temp_data:
            mineru_items = state.temp_data.get('mineru_items', [])
            if mineru_items:
                text = extract_text_from_mineru_results(mineru_items, max_chars=15000)
                if text:
                    return f"Paper content extracted from PDF:\n\n{text}"
        
        # 如果没有 MinerU 结果，直接从 PDF 读取（回退方案）
        import fitz
        pdf_path = state.paper_file
        if not pdf_path or not os.path.exists(pdf_path):
            log.warning("paper_file 为空或不存在，无法读取 PDF 内容")
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            # 读取前 10 页
            for page_idx in range(min(10, len(doc))):
                page = doc.load_page(page_idx)
                text_parts.append(page.get_text("text") or "")
            doc.close()
            
            content = "\n".join(text_parts).strip()
            log.info(f"[pre_tool] 从 PDF 直接提取了 {len(content)} 字符")
            return f"Paper content from PDF:\n\n{content[:15000]}"
        except Exception as e:
            log.error(f"读取 PDF 失败: {e}")
            return ""

    # ==============================================================
    # NODES: 工作流节点
    # ==============================================================
    
    async def pdf_to_images_node(state: Paper2ExpFigureState) -> Paper2ExpFigureState:
        """
        节点 1: PDF → 图片
        将 PDF 的每一页转换为 PIL Image 对象，保存到临时目录
        """
        pdf_path = Path(state.paper_file)
        if not pdf_path.exists():
            log.error(f"PDF 文件不存在: {pdf_path}")
            return state
        
        log.info(f"[pdf_to_images] 开始转换 PDF: {pdf_path}")
        
        # 转换 PDF 为图片
        images = pdf_to_pil_images(pdf_path, dpi=150)
        
        # 创建临时目录保存图片
        output_dir = state.request.output_dir or f"./outputs/paper2expfigure_{uuid.uuid4().hex[:8]}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 保存图片
        image_paths = []
        for idx, img in enumerate(images):
            img_path = images_dir / f"page_{idx+1}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
            log.info(f"[pdf_to_images] 保存第 {idx+1} 页: {img_path}")
        
        # 存储到 state（使用绝对路径）
        state.temp_data['image_paths'] = image_paths
        state.result_path = str(output_path.absolute())
        
        log.info(f"[pdf_to_images] 完成，共转换 {len(images)} 页")
        return state
    
    async def mineru_extract_node(state: Paper2ExpFigureState) -> Paper2ExpFigureState:
        """
        节点 2: MinerU 识别
        使用 MinerU HTTP API 识别图片中的文本和表格
        """
        image_paths = state.temp_data.get('image_paths', [])
        if not image_paths:
            log.warning("[mineru_extract] 没有图片路径，跳过")
            return state
        
        output_path = Path(state.result_path)
        mineru_dir = output_path / "mineru_results"
        mineru_dir.mkdir(exist_ok=True)
        
        port = state.request.mineru_port
        all_items = []
        
        # 对每一页图片执行 MinerU 识别
        for idx, img_path in enumerate(image_paths, 1):
            log.info(f"[mineru_extract] 处理图片 {idx}/{len(image_paths)}: {img_path}")
            try:
                # 使用 run_aio_two_step_extract 进行识别
                items = await run_aio_two_step_extract(
                    image_path=str(img_path),
                    port=port,
                )
                
                # items 是一个列表，直接扩展到 all_items
                if isinstance(items, list):
                    # 为每个 item 添加页面信息
                    for item in items:
                        item['page_index'] = idx - 1  # 0-based index
                        item['page_number'] = idx      # 1-based number
                    
                    all_items.extend(items)
                    log.info(f"[mineru_extract] 从 page_{idx} 提取了 {len(items)} 个元素")
                    
                    # 保存每页的识别结果为 JSON 文件（便于调试）
                    import json
                    result_file = mineru_dir / f"page_{idx}_result.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    log.debug(f"[mineru_extract] 保存结果到: {result_file}")
                else:
                    log.warning(f"[mineru_extract] 返回结果不是列表: {type(items)}")
                    
            except Exception as e:
                log.error(f"[mineru_extract] MinerU 识别失败 (page_{idx}): {e}")
        
        # 存储结果
        state.temp_data['mineru_items'] = all_items
        
        # 保存所有结果的汇总文件
        if all_items:
            import json
            summary_file = mineru_dir / "all_results.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            log.info(f"[mineru_extract] 保存汇总结果到: {summary_file}")
        
        log.info(f"[mineru_extract] 完成，共提取 {len(all_items)} 个元素")
        return state
    
    async def table_extractor_node(state: Paper2ExpFigureState) -> Paper2ExpFigureState:
        """
        节点 3: 提取表格
        从 MinerU 识别结果中提取表格数据，并保存表格区域图片
        """
        mineru_items = state.temp_data.get('mineru_items', [])
        if not mineru_items:
            log.warning("[table_extractor] 没有 MinerU 结果，跳过")
            return state
        
        log.info("[table_extractor] 开始提取表格...")
        tables = extract_tables_from_mineru_results(mineru_items, min_rows=2, min_cols=2)
        
        state.extracted_tables = tables
        log.info(f"[table_extractor] 提取了 {len(tables)} 个表格")
        
        # 打印表格摘要
        for table in tables:
            log.info(f"  - {table['table_id']}: {len(table['headers'])} 列 x {len(table['rows'])} 行")
        
        # ====================================================================
        # 新增功能：保存表格区域图片
        # ====================================================================
        if tables:
            output_path = Path(state.result_path)
            table_images_dir = output_path / "table_images"
            table_images_dir.mkdir(exist_ok=True)
            
            # 获取原始图片路径
            image_paths = state.temp_data.get('image_paths', [])
            
            # 为每个表格裁剪并保存图片
            saved_count = 0
            for item in mineru_items:
                if item.get('type') != 'table':
                    continue
                
                bbox = item.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # 找到对应的表格（通过 bbox 匹配）
                table_match = None
                for table in tables:
                    if table.get('bbox') == bbox:
                        table_match = table
                        break
                
                if not table_match:
                    continue
                
                # 从 item 中直接获取页面索引（在 mineru_extract_node 中添加的）
                page_idx = item.get('page_index')
                
                if page_idx is not None and page_idx < len(image_paths):
                    try:
                        # 读取原始图片
                        from PIL import Image
                        img_path = image_paths[page_idx]
                        img = Image.open(img_path)
                        
                        # bbox 是归一化坐标 [x0, y0, x1, y1]，范围 0-1
                        img_width, img_height = img.size
                        x0 = int(bbox[0] * img_width)
                        y0 = int(bbox[1] * img_height)
                        x1 = int(bbox[2] * img_width)
                        y1 = int(bbox[3] * img_height)
                        
                        # 裁剪表格区域
                        table_img = img.crop((x0, y0, x1, y1))
                        
                        # 保存图片
                        table_id = table_match['table_id']
                        page_num = item.get('page_number', page_idx + 1)
                        table_img_path = table_images_dir / f"{table_id}_page{page_num}.png"
                        table_img.save(table_img_path)
                        
                        # 将图片路径添加到 table 信息中
                        table_match['image_path'] = str(table_img_path)
                        table_match['page_index'] = page_idx
                        table_match['page_number'] = page_num
                        
                        saved_count += 1
                        log.info(f"[table_extractor] 保存表格图片: {table_img_path}")
                        
                    except Exception as e:
                        log.error(f"[table_extractor] 裁剪表格图片失败 ({table_match.get('table_id', 'unknown')}): {e}")
                        import traceback
                        traceback.print_exc()
            
            log.info(f"[table_extractor] 共保存了 {saved_count} 个表格图片到: {table_images_dir}")
        
        return state
    
    async def paper_idea_extractor_node(state: Paper2ExpFigureState) -> Paper2ExpFigureState:
        """
        节点 4: 提取论文核心思想
        调用 paper_idea_extractor Agent 从论文中提取核心思想
        """
        log.info("[paper_idea_extractor] 开始提取论文核心思想...")
        
        agent = create_simple_agent(
            name="paper_idea_extractor",
            model_name="gpt-4o",
            temperature=0.1,
            max_tokens=4096,
            parser_type="json",
        )
        
        state = await agent.execute(state=state)
        
        paper_idea = state.paper_idea or ""
        log.info(f"[paper_idea_extractor] 提取的核心思想长度: {len(paper_idea)} 字符")
        log.info(f"[paper_idea_extractor] 核心思想预览: {paper_idea[:200]}...")
        
        return state
    
    async def code_executor_node(state: Paper2ExpFigureState) -> Paper2ExpFigureState:
        """
        节点 5: 执行代码生成图表
        调用 chart_type_recommender 和 chart_code_generator Agent 智能生成图表
        """
        tables = state.extracted_tables
        if not tables:
            log.warning("[code_executor] 没有表格数据，跳过")
            return state
        
        output_path = Path(state.result_path)
        charts_dir = output_path / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # 创建中间结果目录
        intermediate_dir = output_path / "chart_intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        generated_charts = []
        
        # 获取论文核心思想
        paper_idea = state.paper_idea or "No paper idea extracted"
        
        for table in tables:
            table_id = table['table_id']
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            caption = table.get('caption', '')
            
            if not headers or not rows:
                log.warning(f"[code_executor] 表格 {table_id} 没有数据，跳过")
                continue
            
            log.info(f"[code_executor] 处理表格: {table_id}")
            
            # ============================================
            # 使用 Agent 进行智能判断和代码生成
            # ============================================
            
            try:
                # 1. 准备表格信息（用于理解和分析）
                table_understanding = _generate_table_understanding(
                    table_id=table_id,
                    caption=caption,
                    headers=headers,
                    rows=rows
                )
                
                # 构建完整的表格信息
                table_info = {
                    "table_id": table_id,
                    "caption": caption,
                    "headers": headers,
                    "rows": rows[:10],  # 只传前10行用于分析
                    "total_rows": len(rows),
                    "total_cols": len(headers),
                    "understanding": table_understanding,
                }
                
                # 2. 调用 chart_type_recommender Agent 推荐图表类型
                log.info(f"[code_executor] 调用 chart_type_recommender Agent...")
                
                # 创建 ToolManager 并注册 pre_tool
                tm = get_tool_manager()
                tm.register_pre_tool(
                    name="paper_idea",
                    role="chart_type_recommender",
                    func=lambda: paper_idea,
                )
                tm.register_pre_tool(
                    name="table_info",
                    role="chart_type_recommender",
                    func=lambda: table_info,
                )
                
                # 调用 Agent
                chart_type_agent = create_chart_type_recommender(
                    tool_manager=tm,
                    model_name="gpt-4o",
                    temperature=0.1,
                    max_tokens=2048,
                )
                state = await chart_type_agent.execute(state=state)
                
                # 获取推荐结果
                if state.chart_configs:
                    chart_config = state.chart_configs[-1]  # 获取刚添加的配置
                    chart_type = chart_config.get('chart_type', 'bar')
                    chart_type_reason = chart_config.get('chart_type_reason', '')
                    is_suitable = chart_config.get('is_suitable_for_chart', True)
                    suitability_reason = chart_config.get('suitability_reason', '')
                    
                    log.info(f"[code_executor] 推荐图表类型: {chart_type}")
                    log.info(f"[code_executor] 推荐理由: {chart_type_reason}")
                    
                    # 检查表格是否适合绘图
                    if not is_suitable or chart_type == "none":
                        log.info(
                            f"[code_executor] 表格 {table_id} 不适合绘图，跳过代码生成。"
                            f"原因: {suitability_reason}"
                        )
                        # 保存中间结果（不含代码）
                        import json
                        intermediate_file = intermediate_dir / f"{table_id}_intermediate.json"
                        intermediate_data = {
                            "table_id": table_id,
                            "timestamp": str(Path(state.result_path).name),
                            "table_data": {
                                "caption": caption,
                                "headers": headers,
                                "rows": rows[:5],
                                "total_rows": len(rows),
                                "total_cols": len(headers),
                            },
                            "table_understanding": table_understanding,
                            "chart_config": chart_config,
                            "skipped": True,
                            "skip_reason": suitability_reason,
                        }
                        with open(intermediate_file, 'w', encoding='utf-8') as f:
                            json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
                        continue  # 跳过此表格
                else:
                    log.warning(f"[code_executor] chart_type_recommender 未返回配置，使用默认")
                    chart_config = {
                        "table_id": table_id,
                        "chart_type": "bar",
                        "chart_type_reason": "默认配置",
                        "data_interpretation": {},
                        "visualization_config": {},
                    }
                    state.chart_configs.append(chart_config)
                
                # 3. 调用 chart_code_generator Agent 生成代码
                log.info(f"[code_executor] 调用 chart_code_generator Agent...")
                
                # 注册新的 pre_tool
                tm.register_pre_tool(
                    name="chart_config",
                    role="chart_code_generator",
                    func=lambda: chart_config,
                )
                tm.register_pre_tool(
                    name="table_headers",
                    role="chart_code_generator",
                    func=lambda: headers,
                )
                tm.register_pre_tool(
                    name="table_rows",
                    role="chart_code_generator",
                    func=lambda: rows[:20],  # 传前20行给代码生成器参考
                )
                
                # 调用 Agent
                chart_code_agent = create_chart_code_generator(
                    tool_manager=tm,
                    model_name="gpt-4o",
                    temperature=0.0,
                    max_tokens=4096,
                )
                state = await chart_code_agent.execute(state=state)
                
                # 获取生成的代码
                if state.generated_codes:
                    code_entry = state.generated_codes[-1]  # 获取刚生成的代码
                    code = code_entry.get('code', '')
                    description = code_entry.get('description', '')
                    log.info(f"[code_executor] 生成代码长度: {len(code)} 字符")
                    log.info(f"[code_executor] 代码描述: {description}")
                else:
                    log.error(f"[code_executor] chart_code_generator 未返回代码")
                    continue
                
                # 4. 保存中间结果
                import json
                intermediate_file = intermediate_dir / f"{table_id}_intermediate.json"
                intermediate_data = {
                    "table_id": table_id,
                    "timestamp": str(Path(state.result_path).name),
                    
                    # 表格数据
                    "table_data": {
                        "caption": caption,
                        "headers": headers,
                        "rows": rows[:5],  # 只保存前5行作为示例
                        "total_rows": len(rows),
                        "total_cols": len(headers),
                    },
                    
                    # 表格理解
                    "table_understanding": table_understanding,
                    
                    # Agent 推荐结果
                    "chart_config": chart_config,
                    
                    # 生成的代码
                    "generated_code": code,
                    "code_description": description,
                }
                
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
                log.info(f"[code_executor] 保存中间结果: {intermediate_file}")
                
                # 保存代码文件（便于查看和调试）
                code_file = intermediate_dir / f"{table_id}_code.py"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                log.debug(f"[code_executor] 保存代码文件: {code_file}")
                
                # 5. 执行代码生成图表
                # 需要将 output_path, headers, rows 注入到代码中
                chart_path = charts_dir / f"{table_id}.png"
                
                # 检查代码是否定义了函数但没有调用
                # 这是一个临时解决方案，用于处理旧的生成代码
                import re
                
                # 查找函数定义（如 def create_bar_chart(...):）
                func_match = re.search(r'def\s+(\w+)\s*\(', code)
                if func_match:
                    func_name = func_match.group(1)
                    # 检查是否已经调用了该函数
                    if f"{func_name}(" not in code.split(func_match.group(0))[-1]:
                        # 函数定义了但没有被调用，添加函数调用
                        log.warning(f"[code_executor] 检测到函数 {func_name} 未被调用，自动添加调用")
                        # 添加函数调用（传入 headers, rows, output_path）
                        code += f"\n\n# Auto-added function call\n{func_name}(headers, rows, output_path)\n"
                
                # 构建完整的可执行代码
                exec_code = f"""
# Auto-generated code execution wrapper
output_path = {repr(str(chart_path))}
headers = {repr(headers)}
rows = {repr(rows)}

# Generated chart code
{code}
"""
                
                result = execute_matplotlib_code(
                    code=exec_code,
                    output_path=chart_path,
                    timeout=30,
                )
                
                if result['success']:
                    generated_charts.append(str(chart_path))
                    log.info(f"[code_executor] 生成图表: {chart_path}")
                    
                    # 更新中间结果，添加执行状态
                    intermediate_data["execution_result"] = {
                        "success": True,
                        "chart_path": str(chart_path),
                        "error": None
                    }
                else:
                    log.error(f"[code_executor] 生成图表失败 ({table_id}): {result['error']}")
                    
                    # 更新中间结果，添加错误信息
                    intermediate_data["execution_result"] = {
                        "success": False,
                        "chart_path": None,
                        "error": result['error']
                    }
                
                # 重新保存中间结果（包含执行结果）
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                log.error(f"[code_executor] 处理表格 {table_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        state.generated_charts = generated_charts
        log.info(f"[code_executor] 完成，共生成 {len(generated_charts)} 个图表")
        log.info(f"[code_executor] 中间结果保存在: {intermediate_dir}")
        
        return state

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    
    # 条件路由函数：根据输入类型决定起点
    def route_from_start(state: Paper2ExpFigureState) -> str:
        """
        路由函数：根据输入类型选择工作流的入口节点
        - input_type == "PDF": 从 pdf_to_images 开始
        - input_type == "TABLE": 直接从 code_executor 开始（使用已有表格数据）
        """
        input_type = state.request.input_type or "PDF"
        if input_type.upper() == "PDF":
            log.info("[route] 输入类型为 PDF，从 pdf_to_images 开始")
            return "pdf_to_images"
        else:
            log.info("[route] 输入类型为 TABLE，直接进入 code_executor")
            return "code_executor"
    
    nodes = {
        "_start_": lambda state: state,  # 起始节点
        "pdf_to_images": pdf_to_images_node,
        "mineru_extract": mineru_extract_node,
        "table_extractor": table_extractor_node,
        "paper_idea_extractor": paper_idea_extractor_node,
        "code_executor": code_executor_node,
        "_end_": lambda state: state,  # 终止节点
    }
    
    # 边定义：PDF 模式的完整流程
    edges = [
        # PDF 流程
        ("pdf_to_images", "mineru_extract"),
        ("mineru_extract", "table_extractor"),
        ("table_extractor", "paper_idea_extractor"),
        ("paper_idea_extractor", "code_executor"),
        
        # 最终节点
        ("code_executor", "_end_"),
    ]
    
    builder.add_nodes(nodes).add_edges(edges).add_conditional_edge("_start_", route_from_start)
    return builder