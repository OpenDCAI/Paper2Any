#!/usr/bin/env python3
"""
脚本：运行 WebStructuredDataExtractionNode，将下载目录下 web_get 中的结构化网页内容
转换为基础的 PT/SFT 数据文件。
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

# 确保项目根目录可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataflow_agent.agentroles.data_agents.webresearch import WebStructuredDataExtractionNode
from dataflow_agent.state import WebCrawlRequest, WebCrawlState


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_state(download_dir: str, category: str) -> WebCrawlState:
    request = WebCrawlRequest()
    request.download_dir = download_dir
    request.category = category
    state = WebCrawlState()
    state.request = request
    state.download_dir = download_dir
    state.initial_request = "Extract structured web QA"
    return state


async def run_extraction(
    download_dir: str,
    category: str,
    max_records: Optional[int],
    output_subdir: str,
    objective: Optional[str],
    concurrency: int,
    api_base_url: Optional[str],
    api_key: Optional[str],
    model_name: Optional[str],
    language: Optional[str],
    max_markdown_chars: Optional[int],
) -> WebCrawlState:
    state = build_state(download_dir, category)
    node_kwargs = {
        "target_category": category,
        "max_records": max_records,
        "output_subdir": output_subdir,
        "default_objective": objective,
        "concurrent_tasks": concurrency,
        "api_base_url": api_base_url,
        "api_key": api_key,
        "model_name": model_name,
        "language": language,
    }
    if max_markdown_chars is not None:
        node_kwargs["max_markdown_chars"] = max_markdown_chars
    node = WebStructuredDataExtractionNode(**node_kwargs)
    return await node.execute(state, user_objective=objective)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 web_get 结构化网页内容提取 PT/SFT 数据集。"
    )
    parser.add_argument(
        "--download-dir",
        default=os.getenv("DF_DOWNLOAD_DIR", os.path.join(PROJECT_ROOT, "downloaded_data")),
        help="下载目录路径，默认读取环境变量 DF_DOWNLOAD_DIR 或项目内 downloaded_data。",
    )
    parser.add_argument(
        "--category",
        default=os.getenv("DF_CATEGORY", "SFT"),
        choices=["PT", "SFT"],
        help="输出数据类别，默认 SFT，可通过 DF_CATEGORY 环境变量覆盖。",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="最大的输出记录数量， 默认不限制，可通过 DF_MAX_RECORDS 环境变量指定。",
    )
    parser.add_argument(
        "--output-subdir",
        default=os.getenv("DF_WEB_GET_OUTPUT_SUBDIR", "web_get_extracted"),
        help="输出子目录名称，默认 web_get_extracted，可通过 DF_WEB_GET_OUTPUT_SUBDIR 环境变量覆盖。",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("DF_WEB_GET_CONCURRENCY", "100")),
        help="并发处理网页的数量，默认 100，可通过 DF_WEB_GET_CONCURRENCY 环境变量设置。",
    )
    parser.add_argument(
        "--objective",
        default=os.getenv("DF_USER_OBJECTIVE"),
        help="用户需求/提炼目标，默认读取 DF_USER_OBJECTIVE 环境变量。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用调试日志。",
    )
    parser.add_argument(
        "--max-markdown-chars",
        type=int,
        default=None,
        help="网页内容最大字符数（超出部分会被截断），默认 9000，可通过 DF_WEB_GET_MAX_MARKDOWN_CHARS 环境变量设置。",
    )
    args = parser.parse_args()

    if args.max_records is not None:
        max_records = args.max_records
    else:
        env_max_records = os.getenv("DF_MAX_RECORDS")
        if env_max_records is None or env_max_records == "":
            max_records = None
        else:
            try:
                max_records = int(env_max_records)
            except ValueError as exc:
                raise ValueError(f"无法解析 DF_MAX_RECORDS 环境变量: {env_max_records}") from exc

    if args.max_markdown_chars is not None:
        max_markdown_chars = args.max_markdown_chars
    else:
        env_max_markdown_chars = os.getenv("DF_WEB_GET_MAX_MARKDOWN_CHARS")
        if env_max_markdown_chars is None or env_max_markdown_chars == "":
            max_markdown_chars = None
        else:
            try:
                max_markdown_chars = int(env_max_markdown_chars)
            except ValueError as exc:
                raise ValueError(f"无法解析 DF_WEB_GET_MAX_MARKDOWN_CHARS 环境变量: {env_max_markdown_chars}") from exc

    configure_logging(verbose=args.verbose)

    logging.info("=== 配置概览 ===")
    logging.info("下载目录: %s", args.download_dir)
    logging.info("目标类别: %s", args.category)
    logging.info("最大记录数: %s", max_records if max_records is not None else "不限制")
    logging.info("输出子目录: %s", args.output_subdir)
    logging.info("用户需求: %s", args.objective or "<未指定>")
    logging.info("并发数量: %s", args.concurrency)
    logging.info("最大网页字符数: %s", max_markdown_chars if max_markdown_chars is not None else "默认(9000)")

    api_base_url = os.getenv("CHAT_API_URL")
    api_key = os.getenv("CHAT_API_KEY")
    chat_model = os.getenv("CHAT_MODEL")
    language = os.getenv("DF_WEB_GET_LANGUAGE")

    # 记录关键环境变量（若存在）
    for key in [
        "CHAT_API_URL",
        "CHAT_API_KEY",
        "HF_ENDPOINT",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "RAG_API_URL",
        "RAG_API_KEY",
        "CHAT_MODEL",
    ]:
        if key in os.environ:
            logging.info("环境变量 %s 已设置。", key)

    result_state = asyncio.run(
        run_extraction(
            download_dir=args.download_dir,
            category=args.category,
            max_records=max_records,
            output_subdir=args.output_subdir,
            objective=args.objective,
            concurrency=args.concurrency,
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=chat_model,
            language=language,
            max_markdown_chars=max_markdown_chars,
        )
    )

    logging.info("=== 执行结果 ===")
    logging.info("记录写入数: %s", result_state.crawled_data[-1]["records_written"] if result_state.crawled_data else 0)
    logging.info("输出文件: %s", result_state.crawled_data[-1]["output_file"] if result_state.crawled_data else "N/A")


if __name__ == "__main__":
    main()

