#!/usr/bin/env python3
"""
实用脚本：加载指定数据文件，复用 UniversalDataConvertor 的采样与提示词构建逻辑，
并打印出采样记录以及发送给 LLM 的提示词与响应，便于人工校对。
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

# 确保可以导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset  # type: ignore

from dataflow_agent.agentroles.dataconvertor import UniversalDataConvertor
from dataflow_agent.state import DataCollectionRequest, DataCollectionState


def infer_builder_type(file_path: str) -> str:
    """根据文件扩展名推断 datasets 的 builder 类型。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext in {".parquet"}:
        return "parquet"
    if ext in {".txt"}:
        return "text"
    raise ValueError(f"无法根据扩展名推断 builder 类型，请使用 --builder-type 参数指定: {file_path}")


def build_state(
    language: str,
    target: str,
    category: str,
    download_dir: Optional[str],
    api_base: str,
    api_key: str,
    model_name: str,
) -> DataCollectionState:
    request = DataCollectionRequest()
    request.language = language
    request.target = target
    request.category = category.upper()
    request.chat_api_url = api_base
    request.api_key = api_key
    request.model = model_name
    if download_dir:
        request.download_dir = download_dir
    state = DataCollectionState()
    state.request = request
    return state


def format_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def parse_llm_response(answer_text: str) -> Optional[dict]:
    if not answer_text:
        return None
    pattern = r"```json([\s\S]*?)```"
    match = re.search(pattern, answer_text)
    content = match.group(1).strip() if match else answer_text
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 UniversalDataConvertor 的采样结果与 LLM 输入输出。")
    parser.add_argument("dataset_path", help="待检查的数据文件绝对路径")
    parser.add_argument(
        "--builder-type",
        dest="builder_type",
        default=None,
        help="datasets.load_dataset 的 builder 类型，如 csv/json/parquet。如未提供将根据扩展名自动推断。",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="需要检查的数据 split 名称，默认 train。",
    )
    parser.add_argument(
        "--category",
        default="PT",
        choices=["PT", "SFT"],
        help="数据类别，影响提示词模板。",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="提示词语言，默认 English，可根据需求调整。",
    )
    parser.add_argument(
        "--target",
        default="",
        help="用户需求描述，将注入提示词模板中。",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="采样记录数量，默认 3。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，提供后可复现采样结果。",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("DF_API_URL", ""),
        help="LLM 接口地址，默认读取环境变量 DF_API_URL。",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DF_API_KEY", ""),
        help="LLM API Key，默认读取环境变量 DF_API_KEY。",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DF_MODEL", "deepseek-v3.1-250821"),
        help="调用的模型名称，默认 deepseek-v3.1-250821，可通过 DF_MODEL 环境变量覆盖。",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据文件不存在: {dataset_path}")

    if not args.api_base:
        raise ValueError("未提供 LLM 接口地址，请通过 --api-base 或环境变量 DF_API_URL 指定。")
    if not args.api_key:
        raise ValueError("未提供 LLM API Key，请通过 --api-key 或环境变量 DF_API_KEY 指定。")

    builder_type = args.builder_type or infer_builder_type(dataset_path)

    if args.seed is not None:
        import random

        random.seed(args.seed)

    dataset_dict = load_dataset(builder_type, data_files=dataset_path)
    if args.split not in dataset_dict:
        raise KeyError(f"数据集中不存在 split '{args.split}'，可用 split: {list(dataset_dict.keys())}")

    dataset = dataset_dict[args.split]
    if len(dataset) == 0:
        raise ValueError(f"数据 split '{args.split}' 为空。")

    convertor = UniversalDataConvertor(model_name=args.model, num_sample_records=args.num_samples)
    state = build_state(
        args.language,
        args.target,
        args.category,
        os.path.dirname(dataset_path),
        args.api_base,
        args.api_key,
        args.model,
    )

    if args.seed is not None:
        import random

        random.seed(args.seed)
    sampled_records = convertor._sample_records(dataset, num_samples=args.num_samples)

    column_names = dataset.column_names
    first_record = dataset[0]

    if args.seed is not None:
        import random

        random.seed(args.seed)
    messages = convertor.build_messages(state, column_names, first_record, dataset=dataset)

    llm = convertor.create_llm(state)
    response = llm.invoke(messages)
    answer_text = response.content.strip() if hasattr(response, "content") else str(response)
    parsed = parse_llm_response(answer_text)

    print("=== 采样元数据 ===")
    print(f"数据文件: {dataset_path}")
    print(f"builder 类型: {builder_type}")
    print(f"split 名称: {args.split}")
    print(f"数据总量: {len(dataset)}")
    print(f"采样数量: {len(sampled_records)}")
    print()

    print("=== 用户需求 ===")
    print(args.target or "<未提供>")
    print()

    print("=== 采样记录 (截断后) ===")
    print(format_json(sampled_records))
    print()

    print("=== LLM 输入消息 ===")
    for idx, msg in enumerate(messages, start=1):
        role = getattr(msg, "type", getattr(msg, "role", "message"))
        print(f"[消息 {idx} | {role}]\n{msg.content}\n")

    print("=== LLM 原始响应 ===")
    print(answer_text or "<空响应>")
    print()

    print("=== 解析后的 JSON（附带用户需求） ===")
    if parsed is not None:
        augmented = dict(parsed)
        augmented["__user_target"] = state.request.target
        print(format_json(augmented))
    else:
        debug_obj = {
            "raw_text": answer_text,
            "__user_target": state.request.target,
        }
        print(format_json(debug_obj))
        print("无法解析为 JSON，请手动检查原始响应。")


if __name__ == "__main__":
    main()
