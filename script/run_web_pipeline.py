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
from dataflow_agent.agentroles.datacollector import WebCrawlOrchestrator
from dataflow_agent.agentroles.dataconvertor import universal_data_conversion, UniversalDataConvertor
from dataflow_agent.logger import get_logger

log = get_logger()

async def web_crawl_collection(
    state: DataCollectionState,
    max_crawl_cycles_per_task: int = 10,
    max_crawl_cycles_for_research: int = 15,
    search_engine: str = "tavily",
    use_jina_reader: bool = True,
    enable_rag: bool = True,
    concurrent_pages: int = 5,
    **kwargs
) -> DataCollectionState:
    """
    【步骤1：网页爬取数据收集】
    根据用户需求，自动搜索、分析并下载相关数据文件到指定目录
    
    Args:
        state: 数据收集状态
        max_crawl_cycles_per_task: 下载任务的最大循环次数
        max_crawl_cycles_for_research: research阶段的最大循环次数
        search_engine: 搜索引擎选择 ('tavily', 'duckduckgo', 'jina')
        use_jina_reader: 是否使用 Jina Reader 提取网页结构化内容
        enable_rag: 是否启用 RAG 增强
        concurrent_pages: 并行处理的页面数量
    
    Returns:
        更新后的 DataCollectionState
    """
    log.info("=" * 60)
    log.info("【步骤1】开始网页爬取数据收集...")
    log.info("=" * 60)
    
    request = state.request
    
    # 创建 WebCrawlOrchestrator
    orchestrator = WebCrawlOrchestrator(
        download_dir=request.download_dir,
        dataset_size_categories=[request.dataset_size_category] if request.dataset_size_category else None,
        max_crawl_cycles_per_task=max_crawl_cycles_per_task,
        max_crawl_cycles_for_research=max_crawl_cycles_for_research,
        search_engine=search_engine,
        use_jina_reader=use_jina_reader,
        enable_rag=enable_rag,
        concurrent_pages=concurrent_pages
    )
    
    # 运行网页爬取（自动搜索、分析并下载数据）
    log.info(f"启动 WebCrawlOrchestrator，用户需求: {request.target}")
    web_state = await orchestrator.run(request.target)
    
    # 统计下载结果
    downloaded_count = len([d for d in web_state.crawled_data 
                           if d.get('type') in ['file', 'huggingface_dataset']])
    
    log.info("=" * 60)
    log.info(f"【步骤1完成】网页爬取完成")
    log.info(f"  - 下载文件/数据集数量: {downloaded_count}")
    log.info(f"  - 数据保存目录: {request.download_dir}")
    log.info("=" * 60)
    
    # 更新 state
    # 注意：universal_data_conversion 会直接扫描 download_dir 处理所有下载的数据
    # 它会结合 state.request.target（用户需求）和 state.request.category（PT/SFT）
    # 自动识别数据文件并转换为标准格式，不会进行额外的下载
    state.keywords = ["web_crawl"]  # 设置虚拟关键词用于后续流程
    
    return state


async def main() -> None:
    """主函数：配置并运行完整的网页数据收集与处理流程"""
    
    # 配置数据收集请求
    req = DataCollectionRequest(
        target="我需要一些Python代码的数据集",  # 输入的自然语言指令
        category="SFT",  # 数据类别：PT(预训练) 或 SFT(指令微调)
        dataset_num_limit=5,  # HuggingFace 搜索时每个关键词的数据集数量上限
        dataset_size_category='1K<n<10K',  # HuggingFace 数据集大小范围
        download_dir=r'downloaded_data_finally',  # 下载目录
        
        # API 配置（用于 LLM 调用）
        chat_api_url="http://123.129.219.111:3000/v1",
        api_key=os.getenv("DF_API_KEY"),
        model="gpt-4o",
        language="zh"  # 提示词语言
    )
    
    if not req.api_key:
        print("错误: 请设置 DF_API_KEY 环境变量！")
        print("示例: export DF_API_KEY=your_api_key")
        return
    
    print("\n" + "=" * 60)
    print("网页数据收集与处理流程配置:")
    print(f"  用户需求: {req.target}")
    print(f"  数据类别: {req.category} (PT=预训练, SFT=指令微调)")
    print(f"  下载目录: {req.download_dir}")
    print(f"  HF数据集大小: {req.dataset_size_category}")
    print("=" * 60)
    print("\n流程说明:")
    print("  【步骤1】web_crawl_collection:")
    print("    - 根据用户需求自动搜索相关网页")
    print("    - 智能分析并下载数据文件/数据集")
    print("    - 保存到指定下载目录")
    print("\n  【步骤2】universal_data_conversion:")
    print("    - 扫描下载目录中的所有文件")
    print("    - 结合用户需求理解数据含义")
    print("    - 将数据转换为标准 PT/SFT 格式")
    print("    - 输出到 processed_output 目录")
    print("=" * 60)
    
    # 创建初始状态
    state = DataCollectionState(request=req)
    
    # 构建 LangGraph 工作流
    graph_builder = StateGraph(DataCollectionState)
    
    # 添加节点
    graph_builder.add_node("web_crawl_collection", web_crawl_collection)
    graph_builder.add_node("universal_data_conversion", universal_data_conversion)
    
    # 添加边（定义执行顺序）
    graph_builder.add_edge(START, "web_crawl_collection")
    graph_builder.add_edge("web_crawl_collection", "universal_data_conversion")
    graph_builder.add_edge("universal_data_conversion", END)
    
    # 编译图
    graph = graph_builder.compile()
    
    # 执行工作流
    print("\n" + "=" * 60)
    print("开始执行数据收集与处理流程...")
    print("=" * 60 + "\n")
    
    final_state: DataCollectionState = await graph.ainvoke(state)
    
    print("\n" + "=" * 60)
    print("流程执行完成！")
    print(f"原始数据: {req.download_dir}")
    print(f"处理结果: {req.download_dir}/processed_output/")
    print(f"   - {req.category}.jsonl (标准化数据)")
    print(f"   - summary.txt (处理摘要)")
    print("=" * 60)


if __name__ == "__main__":
    # 设置环境变量（如果未设置）
    os.environ.setdefault("DF_API_URL", "http://123.129.219.111:3000/v1")
    
    # 运行主函数
    asyncio.run(main())

