#!/usr/bin/env python3
"""
Paper2ExpFigure Workflow 测试脚本
从 PDF 论文中提取表格并生成统计图

用法:
    python run_paper2expfigure.py <pdf_path> [--output_dir <dir>] [--mineru_port <port>]
    
示例:
    python run_paper2expfigure.py data/paper.pdf
    python run_paper2expfigure.py data/paper.pdf --output_dir ./my_outputs
    python run_paper2expfigure.py data/paper.pdf --mineru_port 8002
"""

from __future__ import annotations

import asyncio
import argparse
import os
import time
from pathlib import Path

from dataflow_agent.state import Paper2FigureRequest, Paper2FigureState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow import run_workflow
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Paper2ExpFigure Workflow: 从 PDF 论文中提取表格并生成统计图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s data/paper.pdf
  %(prog)s data/paper.pdf --output_dir ./my_outputs
  %(prog)s data/paper.pdf --mineru_port 8002
        """
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="输入 PDF 文件路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录路径（默认：./outputs/paper2expfigure_TIMESTAMP）"
    )
    
    parser.add_argument(
        "--mineru_port",
        type=int,
        default=8001,
        help="MinerU 服务端口（默认：8001）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM 模型名称（默认：gpt-4o）"
    )
    
    parser.add_argument(
        "--api_url",
        type=str,
        default="https://api.apiyi.com/v1",
        help="LLM API URL（默认：https://api.apiyi.com/v1）"
    )
    
    return parser.parse_args()


async def main() -> None:
    """主函数：运行 paper2expfigure workflow"""
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查 PDF 文件是否存在
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        log.error(f"PDF 文件不存在: {pdf_path}")
        print(f"错误：PDF 文件不存在: {pdf_path}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./outputs/paper2expfigure_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # -------- 构造请求参数 -------- #
    req = Paper2FigureRequest(
        language="en",
        chat_api_url=args.api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target="Extract tables from PDF and generate charts",
        
        # Paper2ExpFigure 特有参数
        input_type="PDF",  # "PDF" 或 "TABLE"
    )
    
    # -------- 初始化 State -------- #
    state = Paper2FigureState(
        request=req, 
        messages=[],
        mineru_port=args.mineru_port,
    )
    
    # 设置输入 PDF 文件
    state.paper_file = str(pdf_path.absolute())
    
    # 初始化 temp_data
    state.temp_data = {}
    
    log.info("=" * 80)
    log.info("Paper2ExpFigure Workflow 开始执行")
    log.info(f"输入文件: {state.paper_file}")
    log.info(f"输出目录: {state.result_path}")
    log.info(f"MinerU 端口: {state.mineru_port}")
    log.info(f"LLM 模型: {req.model}")
    log.info("=" * 80)
    
    # -------- 运行 Workflow -------- #
    try:
        final_state: Paper2ExpFigureState = await run_workflow("paper2expfigure", state)
        
        # -------- 输出结果摘要 -------- #
        log.info("=" * 80)
        log.info("Workflow 执行完成!")
        log.info(f"提取的表格数量: {len(final_state.get('extracted_tables', []))}")
        log.info(f"生成的图表数量: {len(final_state.get('generated_charts', []))}")
        log.info(f"输出目录: {final_state.get('result_path', '')}")
        
        if final_state.get('paper_idea', ''):
            log.info(f"论文核心思想长度: {len(final_state.get('paper_idea', ''))} 字符")
        
        if final_state.get('generated_charts', []):
            log.info("\n生成的图表:")
            for chart_path in final_state.get('generated_charts', []):
                log.info(f"  - {chart_path}")
        
        log.info("=" * 80)
        
        return final_state
        
    except Exception as e:
        log.error(f"Workflow 执行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
