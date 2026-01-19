#!/usr/bin/env python3
"""
Paper2ExpFigure Workflow 测试脚本
从 PDF 论文中提取表格并生成统计图

支持三种输入模式：
- PDF: 从 PDF 论文中提取表格 (默认)
- FIGURE: 直接输入表格图片
- TEXT: 输入表格文本

用法:
    # PDF 模式（默认）
    python script/run_paper2expfigure.py pdf_test.pdf
    python script/run_paper2expfigure.py pdf_test.pdf --output_dir ./my_outputs
    python script/run_paper2expfigure.py pdf_test.pdf --gen_fig_model gemini-3-pro-image-preview
    
    # 图片模式
    python script/run_paper2expfigure.py --input_type FIGURE --input_content table_image_test.png
    
    # 文本模式
    python script/run_paper2expfigure.py --input_type TEXT --input_content "Method,Acc,F1\\nOurs,95.2,94.8"
    python script/run_paper2expfigure.py --input_type TEXT --input_file table.csv
    
    python script/run_paper2expfigure.py --input_type TEXT --input_content "
    Table 1: Performance on COCO
    Method,Acc,F1
    ResNet,76.1,74.2
    ViT,81.3,80.1"
    
    
    latex表格文本示例：
    \begin{tabular}{l|ccc|ccc}
    \hline
    \multirow{2}{*}{Model} & \multicolumn{3}{c|}{English} & \multicolumn{3}{c}{Chinese} \\
    \cline{2-7}
    & P & R & F1 & P & R & F1 \\
    \hline
    BERT & 88.2 & 86.5 & 87.3 & 85.1 & 83.2 & 84.1 \\
    RoBERTa & 89.7 & 88.1 & 88.9 & 86.4 & 85.0 & 85.7 \\
    GPT-4 & 92.1 & 91.3 & 91.7 & 89.8 & 88.5 & 89.1 \\
    Ours & \textbf{94.3} & \textbf{93.8} & \textbf{94.0} & \textbf{91.2} & \textbf{90.7} & \textbf{90.9} \\
    \hline
    \end{tabular}

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
        description="Paper2ExpFigure Workflow: 从 PDF/图片/文本 生成统计图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # PDF 模式（默认）
  %(prog)s data/paper.pdf
  %(prog)s data/paper.pdf --output_dir ./my_outputs
  
  # 图片模式 - 直接输入表格图片
  %(prog)s --input_type FIGURE --input_content table1.png
  %(prog)s --input_type FIGURE --input_content "table1.png,table2.png"
  
  # 文本模式 - 输入表格文本
  %(prog)s --input_type TEXT --input_content "Method,Acc,F1\\nOurs,95.2,94.8"
  %(prog)s --input_type TEXT --input_file table.csv
        """
    )
    
    # 位置参数（PDF 模式的文件路径，可选）
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs='?',
        default=None,
        help="输入 PDF 文件路径（PDF 模式时使用）"
    )
    
    # 输入类型
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["PDF", "FIGURE", "TEXT"],
        default="PDF",
        help="输入类型: PDF（默认）、FIGURE（表格图片）、TEXT（表格文本）"
    )
    
    # 输入内容
    parser.add_argument(
        "--input_content",
        type=str,
        default=None,
        help="输入内容：FIGURE 模式为图片路径（多个用逗号分隔），TEXT 模式为表格文本"
    )
    
    # 从文件读取输入（TEXT 模式）
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="从文件读取输入内容（TEXT 模式时可用，支持 CSV/TSV/Markdown）"
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
        default=8010,
        help="MinerU 服务端口（默认：8010，仅 PDF 模式使用）"
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
    
    parser.add_argument(
        "--style",
        type=str,
        default="kartoon",
        help="图表风格（默认：kartoon）"
    )
    
    parser.add_argument(
        "--gen_fig_model",
        type=str,
        default="gemini-2.5-flash-image-preview",
        help="图表生成模型（默认：gemini-2.5-flash-image-preview）"
    )
    
    return parser.parse_args()


def validate_args(args) -> tuple[str, str]:
    """
    验证参数并返回 (input_type, input_content)
    """
    input_type = args.input_type.upper()
    input_content = ""
    
    if input_type == "PDF":
        # PDF 模式：需要 pdf_path 或 input_content
        if args.pdf_path:
            input_content = args.pdf_path
        elif args.input_content:
            input_content = args.input_content
        else:
            raise ValueError("PDF 模式需要提供 PDF 文件路径（位置参数或 --input_content）")
        
        # 检查文件是否存在
        if not Path(input_content).exists():
            raise ValueError(f"PDF 文件不存在: {input_content}")
    
    elif input_type == "FIGURE":
        # FIGURE 模式：需要图片路径
        if not args.input_content:
            raise ValueError("FIGURE 模式需要提供图片路径（--input_content）")
        
        input_content = args.input_content
        
        # 检查图片文件是否存在
        for img_path in input_content.split(","):
            img_path = img_path.strip()
            if img_path and not Path(img_path).exists():
                raise ValueError(f"图片文件不存在: {img_path}")
    
    elif input_type == "TEXT":
        # TEXT 模式：需要文本内容或文件
        if args.input_file:
            # 从文件读取
            file_path = Path(args.input_file)
            if not file_path.exists():
                raise ValueError(f"输入文件不存在: {file_path}")
            input_content = file_path.read_text(encoding='utf-8')
        elif args.input_content:
            input_content = args.input_content
            # 处理转义的换行符
            input_content = input_content.replace("\\n", "\n")
        else:
            raise ValueError("TEXT 模式需要提供表格文本（--input_content 或 --input_file）")
    
    return input_type, input_content


async def main() -> None:
    """主函数：运行 paper2expfigure workflow"""
    
    # 解析命令行参数
    args = parse_args()
    
    # 验证参数
    try:
        input_type, input_content = validate_args(args)
    except ValueError as e:
        log.error(str(e))
        print(f"错误：{e}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./outputs/paper2expfigure_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # -------- 构造请求参数 -------- #
    req = Paper2FigureRequest(
        language="en",
        gen_fig_model=args.gen_fig_model,
        chat_api_url=args.api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target="Extract tables and generate charts",
        input_type=input_type,
        style=args.style,
    )
    
    # -------- 初始化 State -------- #
    state = Paper2FigureState(
        request=req, 
        messages=[],
        mineru_port=args.mineru_port,
    )
    
    # 根据输入类型设置对应字段
    if input_type == "PDF":
        state.paper_file = str(Path(input_content).absolute())
    elif input_type == "FIGURE":
        state.fig_draft_path = input_content
    elif input_type == "TEXT":
        state.paper_idea = input_content
    
    state.input_type = input_type
    state.result_path = str(Path(output_dir).absolute())
    
    # 初始化 temp_data
    state.temp_data = {}
    
    log.info("=" * 80)
    log.info("Paper2ExpFigure Workflow 开始执行")
    log.info(f"输入类型: {input_type}")
    if input_type == "PDF":
        log.info(f"输入文件: {state.paper_file}")
    elif input_type == "FIGURE":
        log.info(f"输入图片: {state.fig_draft_path}")
    elif input_type == "TEXT":
        log.info(f"输入文本长度: {len(state.paper_idea)} 字符")
    log.info(f"输出目录: {state.result_path}")
    log.info(f"LLM 模型: {req.model}")
    log.info(f"图表风格: {req.style}")
    log.info("=" * 80)
    
    # -------- 运行 Workflow -------- #
    try:
        final_state: Paper2FigureState = await run_workflow("paper2expfigure", state)
        
        # -------- 输出结果摘要 -------- #
        log.info("=" * 80)
        log.info("Workflow 执行完成!")
        log.info(f"提取的表格数量: {len(final_state.get('extracted_tables', []))}")
        log.info(f"生成的图表数量: {len(final_state.get('generated_charts', {}))}")
        log.info(f"输出目录: {final_state.get('result_path', '')}")
        
        if final_state.get('paper_idea', ''):
            log.info(f"论文核心思想长度: {len(final_state.get('paper_idea', ''))} 字符")
        
        if final_state.get('generated_charts', {}):
            log.info("\n生成的图表:")
            for table_id, chart_path in final_state.get('generated_charts', {}).items():
                log.info(f"  - {table_id}: {chart_path}")
        
        if final_state.get('ppt_path', ''):
            log.info(f"\nPPT 文件: {final_state.get('ppt_path', '')}")
        
        log.info("=" * 80)
        
        return final_state
        
    except Exception as e:
        log.error(f"Workflow 执行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
