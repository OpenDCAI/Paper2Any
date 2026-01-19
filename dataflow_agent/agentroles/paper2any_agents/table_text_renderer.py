"""
TableTextRenderer agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
从表格文本生成 matplotlib 代码并渲染为表格图片

支持复杂表格结构：
- 多级表头（跨行/跨列合并）
- LaTeX 表格格式
- Markdown 表格格式
- CSV/TSV 格式

用于 Paper2ExpFigure 工作流的 TEXT 模式
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.cores.registry import register
from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.utils import execute_matplotlib_code

log = get_logger(__name__)


@register("table_text_renderer")
class TableTextRenderer(BaseAgent):
    """从表格文本生成渲染代码的 Agent"""

    @property
    def role_name(self) -> str:
        return "table_text_renderer"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_table_text_renderer"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_table_text_renderer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """从 pre_tool_results 中获取 prompt 参数"""
        return {
            "table_text": pre_tool_results.get("table_text", ""),
            "table_title": pre_tool_results.get("table_title", ""),
            "output_path": pre_tool_results.get("output_path", "table_output.png"),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """默认的 pre_tool_results"""
        return {
            "table_text": "",
            "table_title": "",
            "output_path": "table_output.png",
        }

    def update_state_result(
        self,
        state,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        """将生成的代码和解析结果写入 state"""
        try:
            if isinstance(result, dict):
                code = result.get("code", "")
                table_structure = result.get("table_structure", {})
                
                # 存储到 temp_data
                if not hasattr(state, 'temp_data') or state.temp_data is None:
                    state.temp_data = {}
                
                state.temp_data["table_render_code"] = code
                state.temp_data["table_structure"] = table_structure
                
                log.info(f"[TableTextRenderer] 生成代码长度: {len(code)}")
                log.info(f"[TableTextRenderer] 多级表头: {table_structure.get('has_multi_level_header', False)}")
        except Exception as e:
            log.warning(f"[TableTextRenderer] 更新 state 失败: {e}")

        return super().update_state_result(state, result, pre_tool_results)

    async def execute_pre_tools(self, state) -> Dict[str, Any]:
        """重写 execute_pre_tools，从 state.pre_tool_results 注入参数"""
        results = await super().execute_pre_tools(state)
        
        # 从 state.pre_tool_results 注入参数
        inject_results = getattr(state, 'pre_tool_results', {})
        for key, value in inject_results.items():
            if value:
                results[key] = value
        
        return results


# ----------------------------------------------------------------------
# Helper APIs
# ----------------------------------------------------------------------
def create_table_text_renderer(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> TableTextRenderer:
    """创建 TableTextRenderer 实例"""
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return TableTextRenderer(tool_manager=tool_manager, **kwargs)


async def render_table_from_text(
    table_text: str,
    output_path: Path,
    state,
    title: str = "",
    model_name: str = "gpt-4o",
    tool_manager: Optional[ToolManager] = None,
) -> tuple:
    """
    从表格文本渲染表格图片的完整流程
    
    Args:
        table_text: 原始表格文本（支持 LaTeX/Markdown/CSV 等格式）
        output_path: 输出图片路径
        state: 状态对象
        title: 表格标题
        model_name: 使用的模型名称
        tool_manager: 工具管理器
        
    Returns:
        (success, parsed_data): 是否成功，以及解析出的表格结构数据
    """
    from dataflow_agent.agentroles import create_simple_agent
    
    parsed_data = {
        "headers": [],
        "rows": [],
        "has_multi_level_header": False,
        "header_levels": 1,
    }
    
    try:
        # 创建 agent
        agent = create_simple_agent(
            name="table_text_renderer",
            model_name=model_name,
            temperature=0.1,
            max_tokens=4096,
            tool_manager=tool_manager,
        )
        
        # 注入 pre_tool_results
        state.pre_tool_results = {
            "table_text": table_text,
            "table_title": title,
            "output_path": str(output_path),
        }
        
        # 执行 agent
        state = await agent.execute(state=state, use_agent=False)
        
        # 获取结果
        agent_result = state.agent_results.get("table_text_renderer", {}).get("results", {})
        code = agent_result.get("code", "")
        table_structure = agent_result.get("table_structure", {})
        
        if table_structure:
            parsed_data.update(table_structure)
        
        if not code:
            log.warning("[render_table_from_text] Agent 未返回代码，使用回退方案")
            return _render_table_fallback(table_text, output_path, title), parsed_data
        
        log.info(f"[render_table_from_text] 生成代码长度: {len(code)} 字符")
        
        # 在代码前添加 matplotlib 后端设置
        full_code = f'''
import matplotlib
matplotlib.use('Agg')

{code}
'''
        
        # 执行代码
        result = execute_matplotlib_code(
            code=full_code,
            output_path=output_path,
            timeout=30,
        )
        
        if result['success']:
            log.info(f"[render_table_from_text] 表格图片已生成: {output_path}")
            return True, parsed_data
        else:
            log.warning(f"[render_table_from_text] 代码执行失败: {result['error']}")
            return _render_table_fallback(table_text, output_path, title), parsed_data
            
    except Exception as e:
        log.error(f"[render_table_from_text] 渲染失败: {e}")
        import traceback
        traceback.print_exc()
        return _render_table_fallback(table_text, output_path, title), parsed_data


def _render_table_fallback(
    table_text: str,
    output_path: Path,
    title: str = ""
) -> bool:
    """
    回退方案：简单解析表格文本并用 matplotlib 渲染
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        # 简单解析：按行分割，按常见分隔符分列
        lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
        if not lines:
            return False
        
        # 跳过 LaTeX 命令行和 markdown 分隔行
        filtered_lines = []
        for l in lines:
            # 跳过 LaTeX 命令
            if l.startswith('\\') and not l.startswith('\\hline'):
                continue
            # 跳过 markdown 分隔行
            if all(c in '-|: ' for c in l):
                continue
            # 跳过 \hline
            if l == '\\hline':
                continue
            filtered_lines.append(l)
        
        lines = filtered_lines
        if len(lines) < 2:
            return False
        
        # 检测分隔符
        first_line = lines[0]
        if '&' in first_line:  # LaTeX
            sep = '&'
        elif '|' in first_line:  # Markdown
            sep = '|'
        elif '\t' in first_line:  # TSV
            sep = '\t'
        elif ',' in first_line:  # CSV
            sep = ','
        else:
            sep = None
        
        if sep:
            headers = [c.strip().replace('\\\\', '').strip() for c in first_line.split(sep) if c.strip()]
            rows = []
            for l in lines[1:]:
                row = [c.strip().replace('\\\\', '').strip() for c in l.split(sep) if c.strip()]
                if row:
                    rows.append(row)
        else:
            import re
            headers = re.split(r'\s{2,}', first_line)
            rows = [re.split(r'\s{2,}', l) for l in lines[1:]]
        
        if not headers or not rows:
            return False
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(max(len(headers) * 1.5, 8), max(len(rows) * 0.5 + 1, 4)))
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 确保所有行的列数一致
        max_cols = len(headers)
        normalized_rows = []
        for row in rows:
            if len(row) < max_cols:
                row = row + [''] * (max_cols - len(row))
            elif len(row) > max_cols:
                row = row[:max_cols]
            normalized_rows.append(row)
        
        table = ax.table(
            cellText=normalized_rows,
            colLabels=headers,
            loc='center',
            cellLoc='center',
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(normalized_rows) + 1):
            for j in range(len(headers)):
                try:
                    cell = table[(i, j)]
                    cell.set_facecolor('#D9E2F3' if i % 2 == 0 else 'white')
                except:
                    pass
        
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        log.info(f"[_render_table_fallback] 表格图片已生成: {output_path}")
        return True
        
    except Exception as e:
        log.error(f"[_render_table_fallback] 生成表格图片失败: {e}")
        return False


async def split_tables_from_text(
    text: str,
    state,
    model_name: str = "gpt-4o",
) -> List[Dict[str, str]]:
    """
    使用 LLM 分析文本，识别并分割多个表格
    
    Args:
        text: 包含一个或多个表格的文本
        state: 状态对象
        model_name: 使用的模型名称
        
    Returns:
        [{"text": "表格文本", "caption": "表格标题"}, ...]
    """
    from dataflow_agent.agentroles import create_simple_agent
    
    try:
        # 创建 agent
        agent = create_simple_agent(
            name="table_splitter",
            model_name=model_name,
            temperature=0.1,
            max_tokens=4096,
        )
        
        # 注入 pre_tool_results
        state.pre_tool_results = {
            "input_text": text,
        }
        
        # 执行 agent
        state = await agent.execute(state=state, use_agent=False)
        
        # 获取结果
        agent_result = state.agent_results.get("table_splitter", {}).get("results", {})
        tables = agent_result.get("tables", [])
        
        if tables:
            log.info(f"[split_tables_from_text] 识别到 {len(tables)} 个表格")
            return tables
        else:
            log.warning("[split_tables_from_text] 未识别到表格，返回原文本")
            return [{"text": text, "caption": ""}]
            
    except Exception as e:
        log.error(f"[split_tables_from_text] 分割表格失败: {e}")
        return [{"text": text, "caption": ""}]


@register("table_splitter")
class TableSplitter(BaseAgent):
    """分割文本中多个表格的 Agent"""

    @property
    def role_name(self) -> str:
        return "table_splitter"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_table_splitter"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_table_splitter"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input_text": pre_tool_results.get("input_text", ""),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "input_text": "",
        }

    async def execute_pre_tools(self, state) -> Dict[str, Any]:
        """重写 execute_pre_tools，从 state.pre_tool_results 注入参数"""
        results = await super().execute_pre_tools(state)
        inject_results = getattr(state, 'pre_tool_results', {})
        for key, value in inject_results.items():
            if value:
                results[key] = value
        return results
