from __future__ import annotations
import argparse, asyncio, os
from pathlib import Path
import sys

# 将项目根目录添加到 sys.path 最前面，优先导入本地修改的版本
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataflow_agent.state import DataCollectionRequest, DataCollectionState
from dataflow_agent.agentroles.dataconvertor import universal_data_conversion
from dataflow_agent.logger import get_logger

log = get_logger()


async def main(args) -> None:
    """仅执行后处理（universal_data_conversion），扫描并转换下载目录内的数据文件。"""

    # 组装请求
    req = DataCollectionRequest(
        target=args.target,
        category=args.category,
        dataset_num_limit=args.dataset_num_limit,
        dataset_size_category=args.dataset_size_category,
        download_dir=args.download_dir,
        chat_api_url=args.chat_api_url,
        api_key=args.api_key,
        model=args.model,
        language=args.language,
    )

    if not req.api_key:
        print("错误: 请通过 --api_key 或环境变量 DF_API_KEY 设置API Key！")
        return

    print("\n" + "=" * 60)
    print("后处理（统一格式转换）配置:")
    print(f"  用户需求: {req.target}")
    print(f"  数据类别: {req.category} (PT=预训练, SFT=指令微调)")
    print(f"  下载目录: {req.download_dir}")
    print("  将扫描整个目录并识别数据文件，输出到 processed_output/ 下的标准 JSONL")
    print("=" * 60)

    # 初始化状态
    state = DataCollectionState(request=req)

    # 直接执行 universal_data_conversion（不会进行额外下载，只做后处理）
    final_state = await universal_data_conversion(
        state,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_sample_length=args.max_sample_length,
        num_sample_records=args.num_sample_records,
    )

    # 输出结果位置
    print("\n" + "=" * 60)
    print("后处理完成！")
    print(f"原始数据目录: {req.download_dir}")
    print(f"处理结果目录: {req.download_dir}/processed_output/")
    print(f"   - {req.category}.jsonl (标准化数据)")
    print(f"   - summary.txt (处理摘要)")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅执行后处理，扫描下载目录并转换为标准PT/SFT格式")

    # 基本参数
    parser.add_argument("--download_dir", type=str, default="downloaded_data_finally", help="下载数据所在目录")
    parser.add_argument("--category", type=str, choices=["PT", "SFT"], default="SFT", help="数据类别：PT 或 SFT")
    parser.add_argument("--target", type=str, default="对已下载的数据进行统一转换", help="用于帮助LLM理解数据的用户目标/需求")
    parser.add_argument("--language", type=str, default="zh", help="提示词语言：zh/en 等")

    # 可选：与HF搜索相关（虽不下载，但对象需要该字段）
    parser.add_argument("--dataset_num_limit", type=int, default=5, help="占位参数，不影响本脚本")
    parser.add_argument("--dataset_size_category", type=str, default="1K<n<100K", help="占位参数，不影响本脚本")

    # LLM API 配置
    parser.add_argument(
        "--chat_api_url",
        type=str,
        default=os.environ.get("DF_API_URL", "https://api.deepseek.com/v1"),
        help="聊天模型API地址，可用 DF_API_URL 环境变量覆盖",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("DF_API_KEY", ""),
        help="API Key，可用 DF_API_KEY 环境变量提供",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("DF_MODEL", "deepseek-chat"), help="模型名称")

    # 转换器细节参数
    parser.add_argument("--temperature", type=float, default=0.0, help="模型温度")
    parser.add_argument("--max_tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--max_sample_length", type=int, default=200, help="采样字段截断长度")
    parser.add_argument("--num_sample_records", type=int, default=3, help="每个文件采样记录数量")

    return parser.parse_args()


if __name__ == "__main__":
    # 若未显式提供，设置一个默认的 API URL，便于快速启动
    os.environ.setdefault("DF_API_URL", os.environ.get("DF_API_URL", "https://api.deepseek.com/v1"))
    args = parse_args()
    asyncio.run(main(args))




