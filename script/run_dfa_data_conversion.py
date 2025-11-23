from __future__ import annotations
import argparse, asyncio, os
from langgraph.graph import StateGraph, START, END
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path 最前面，优先导入本地修改的版本
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataflow_agent.state import DataCollectionRequest, DataCollectionState
from dataflow_agent.agentroles.dataconvertor import universal_data_conversion
from dataflow_agent.logger import get_logger

log = get_logger()

async def data_conversion_only(
    state: DataCollectionState,
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_sample_length: int = 200,
    num_sample_records: int = 3,
    **kwargs
) -> DataCollectionState:
    """
    【步骤2：数据转换处理】
    从下载目录读取数据，转换并输出为标准格式
    
    Args:
        state: 数据收集状态
        model_name: 模型名称（可选，默认使用环境变量 CHAT_MODEL）
        temperature: 模型温度
        max_tokens: 最大token数
        max_sample_length: 每个字段的最大采样长度（字符数），默认200
        num_sample_records: 采样记录数量，默认3
    
    Returns:
        更新后的 DataCollectionState
    """
    log.info("=" * 60)
    log.info("【步骤2】开始数据转换处理...")
    log.info("=" * 60)
    
    request = state.request
    
    # 检查下载目录是否存在
    if not os.path.exists(request.download_dir):
        log.error(f"下载目录不存在: {request.download_dir}")
        raise ValueError(f"下载目录不存在: {request.download_dir}")
    
    log.info(f"从下载目录读取数据: {request.download_dir}")
    log.info(f"数据类别: {request.category} (PT=预训练, SFT=指令微调)")
    
    # 运行数据转换
    log.info(f"启动 UniversalDataConvertor...")
    converted_state = await universal_data_conversion(
        state,
        model_name=model_name or request.model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_sample_length=max_sample_length,
        num_sample_records=num_sample_records,
        **kwargs
    )
    
    log.info("=" * 60)
    log.info(f"【步骤2完成】数据转换完成")
    log.info(f"  - 处理结果目录: {request.download_dir}/processed_output/")
    log.info(f"  - 标准化数据文件: {request.download_dir}/processed_output/{request.category}.jsonl")
    log.info(f"  - 处理摘要文件: {request.download_dir}/processed_output/summary.txt")
    log.info("=" * 60)
    
    return converted_state


async def main() -> None:
    """主函数：运行数据转换处理流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据转换处理脚本（仅后处理步骤）")
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="下载目录路径（如果未指定，将从环境变量或默认值获取）"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="SFT",
        choices=["PT", "SFT"],
        help="数据类别：PT(预训练) 或 SFT(指令微调)，默认 SFT"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="用户需求描述（用于数据转换时的上下文理解）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称（可选，默认使用环境变量 CHAT_MODEL）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="模型温度，默认 0.0"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="最大token数，默认 4096"
    )
    parser.add_argument(
        "--max-sample-length",
        type=int,
        default=200,
        help="每个字段的最大采样长度（字符数），默认 200"
    )
    parser.add_argument(
        "--num-sample-records",
        type=int,
        default=3,
        help="采样记录数量，默认 3"
    )
    
    args = parser.parse_args()
    
    # 获取下载目录
    # download_dir = args.download_dir or os.getenv("DOWNLOAD_DIR") or "downloaded_data_finally2"
    download_dir ="/mnt/DataFlow/lz/proj/agentgroup/binrui/postprocess_banchmark"  
    # 如果下载目录是相对路径，转换为绝对路径
    if not os.path.isabs(download_dir):
        download_dir = os.path.abspath(download_dir)
    
    # 配置数据收集请求（仅设置必要字段）
    req = DataCollectionRequest(
        target=args.target or "数据转换处理",
        category=args.category,
        download_dir=download_dir,
        
        # API 配置（用于 LLM 调用）
        chat_api_url=os.getenv("CHAT_API_URL"),
        api_key=os.getenv("CHAT_API_KEY"),
        model=args.model or os.getenv("CHAT_MODEL"),
        language="zh",  # 提示词语言
    )
    
    if not req.api_key:
        print("错误: 请设置 CHAT_API_KEY 环境变量！")
        print("示例: export CHAT_API_KEY=your_api_key")
        return
    
    print("\n" + "=" * 60)
    print("数据转换处理流程配置:")
    print(f"  下载目录: {req.download_dir}")
    print(f"  数据类别: {req.category} (PT=预训练, SFT=指令微调)")
    print(f"  用户需求: {req.target}")
    print(f"  模型: {req.model}")
    print(f"  输出目录: {req.download_dir}/processed_output/")
    print(f"  - {req.category}.jsonl (标准化数据)")
    print(f"  - summary.txt (处理摘要)")
    print("=" * 60)
    
    # 检查下载目录是否存在
    if not os.path.exists(download_dir):
        print(f"\n错误: 下载目录不存在: {download_dir}")
        print("请确保已运行数据收集步骤，或指定正确的下载目录。")
        print("使用 --download-dir 参数指定下载目录。")
        return
    
    # 创建初始状态
    state = DataCollectionState(request=req)
    
    # 构建 LangGraph 工作流（只有一个节点）
    graph_builder = StateGraph(DataCollectionState)
    
    # 添加节点
    graph_builder.add_node("data_conversion_only", data_conversion_only)
    
    # 添加边（定义执行顺序）
    graph_builder.add_edge(START, "data_conversion_only")
    graph_builder.add_edge("data_conversion_only", END)
    
    # 编译图
    graph = graph_builder.compile()
    
    # 执行工作流
    print("\n" + "=" * 60)
    print("开始执行数据转换处理流程...")
    print("=" * 60 + "\n")
    
    final_state: DataCollectionState = await graph.ainvoke(
        state,
        config={
            "recursion_limit": 50,  # 设置递归限制
        }
    )
    
    print("\n" + "=" * 60)
    print("流程执行完成！")
    print(f"原始数据: {req.download_dir}")
    print(f"处理结果: {req.download_dir}/processed_output/")
    print(f"  - {req.category}.jsonl (标准化数据)")
    print(f"  - summary.txt (处理摘要)")
    print("=" * 60)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())





