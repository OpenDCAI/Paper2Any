"""
DataConvertor 集成测试 - 使用真实 API LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

运行方式:
  pytest tests/test_dataconvertor_integration.py -v -s
  或直接: python tests/test_dataconvertor_integration.py

环境变量:
  DF_API_KEY: API密钥（必需）
  DF_API_URL: API地址（可选，默认: http://123.129.219.111:3000/v1）
  DF_MODEL: 模型名称（可选，默认: gpt-4o）
"""

from __future__ import annotations

import pytest
import json
import os
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# 直接导入避免循环导入问题
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow_agent.agentroles.data_convertor import DataConvertor, UniversalDataConvertor
from dataflow_agent.state import DataCollectionState, DataCollectionRequest
from dataflow_agent.schemas.llm_mapping_schema import validate_pt_mapping, validate_sft_mapping, validate_mapping


# ============================================================================
# 测试配置
# ============================================================================

# 从环境变量读取配置
API_KEY = os.getenv("DF_API_KEY")
API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
MODEL = os.getenv("DF_MODEL", "gpt-4o")

# 如果API_KEY未设置，跳过所有测试
pytestmark = pytest.mark.skipif(
    not API_KEY or API_KEY == "test",
    reason="需要设置 DF_API_KEY 环境变量才能运行集成测试"
)


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_real_state(category: str = "PT", download_dir: str = None) -> DataCollectionState:
    """创建使用真实API的DataCollectionState"""
    if download_dir is None:
        download_dir = tempfile.mkdtemp()
    
    request = DataCollectionRequest(
        language="zh",
        chat_api_url=API_URL,
        api_key=API_KEY,
        model=MODEL,
        target="测试数据转换",
        category=category,
        download_dir=download_dir
    )
    
    state = DataCollectionState(
        request=request,
        keywords=["test"],
        sources={},
        downloads={}
    )
    return state


# ============================================================================
# 集成测试1: PT模式 - 真实LLM映射
# ============================================================================

@pytest.mark.asyncio
async def test_pt_mode_real_llm_mapping():
    """测试PT模式：使用真实LLM进行数据映射"""
    convertor = DataConvertor(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096
    )
    state = create_real_state("PT")
    
    # 模拟数据行
    row = {
        "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "dataset_name": "wikipedia",
        "lang": "zh",
        "created_at": "2024-01-01T00:00:00Z",
        "id": "article_001"
    }
    
    column_names = list(row.keys())
    
    # 调用真实LLM进行映射
    print(f"\n=== 测试PT模式LLM映射 ===")
    print(f"数据行: {json.dumps(row, ensure_ascii=False, indent=2)}")
    
    annotation_result = await convertor.invoke(state, column_names, row)
    
    print(f"\nLLM返回的映射结果:")
    print(json.dumps(annotation_result, ensure_ascii=False, indent=2))
    
    # 验证映射结果
    assert annotation_result is not None
    is_valid = validate_pt_mapping(annotation_result)
    assert is_valid, f"PT映射结果验证失败: {annotation_result}"
    
    # 验证必填字段
    assert "text" in annotation_result, "缺少text字段"
    assert annotation_result["text"] is not None, "text字段为null"
    
    # 构建中间格式
    intermediate_record = convertor._build_intermediate_format_pt(
        row, annotation_result, "/test/wikipedia.json", state, 0
    )
    
    print(f"\n构建的中间格式:")
    print(json.dumps(intermediate_record, ensure_ascii=False, indent=2))
    
    # 验证中间格式
    assert intermediate_record is not None
    assert intermediate_record["dataset_type"] == "pretrain"
    assert "id" in intermediate_record
    assert "text" in intermediate_record
    assert intermediate_record["text"] is not None
    assert "meta" in intermediate_record
    
    print(f"\n✓ PT模式测试通过")


# ============================================================================
# 集成测试2: SFT模式 - 真实LLM映射
# ============================================================================

@pytest.mark.asyncio
async def test_sft_mode_real_llm_mapping():
    """测试SFT模式：使用真实LLM进行数据映射"""
    convertor = DataConvertor(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096
    )
    state = create_real_state("SFT")
    
    # 模拟数据行（QA对格式）
    row = {
        "question": "什么是人工智能？",
        "answer": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "dataset_name": "alpaca",
        "lang": "zh"
    }
    
    column_names = list(row.keys())
    
    # 调用真实LLM进行映射
    print(f"\n=== 测试SFT模式LLM映射 ===")
    print(f"数据行: {json.dumps(row, ensure_ascii=False, indent=2)}")
    
    annotation_result = await convertor.invoke(state, column_names, row)
    
    print(f"\nLLM返回的映射结果:")
    print(json.dumps(annotation_result, ensure_ascii=False, indent=2))
    
    # 验证映射结果
    assert annotation_result is not None
    is_valid = validate_sft_mapping(annotation_result)
    assert is_valid, f"SFT映射结果验证失败: {annotation_result}"
    
    # 验证必填字段
    assert "messages" in annotation_result, "缺少messages字段"
    assert annotation_result["messages"] is not None, "messages字段为null"
    assert isinstance(annotation_result["messages"], list), "messages不是列表"
    assert len(annotation_result["messages"]) > 0, "messages列表为空"
    
    # 验证每条消息
    for msg in annotation_result["messages"]:
        assert "role" in msg, "消息缺少role字段"
        assert msg["role"] in ["user", "assistant", "system", "tool"], f"无效的role: {msg['role']}"
        assert "content" in msg, "消息缺少content字段"
    
    # 构建中间格式
    intermediate_record = convertor._build_intermediate_format_sft(
        row, annotation_result, "/test/alpaca.json", state, 0
    )
    
    print(f"\n构建的中间格式:")
    print(json.dumps(intermediate_record, ensure_ascii=False, indent=2))
    
    # 验证中间格式
    assert intermediate_record is not None
    assert intermediate_record["dataset_type"] == "sft"
    assert "id" in intermediate_record
    assert "messages" in intermediate_record
    assert len(intermediate_record["messages"]) > 0
    assert "meta" in intermediate_record
    
    print(f"\n✓ SFT模式测试通过")


# ============================================================================
# 集成测试3: 复杂嵌套结构 - SFT模式
# ============================================================================

@pytest.mark.asyncio
async def test_sft_mode_nested_structure():
    """测试SFT模式：处理嵌套对话结构"""
    convertor = DataConvertor(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096
    )
    state = create_real_state("SFT")
    
    # 模拟嵌套对话数据
    row = {
        "conversation": {
            "turns": [
                {
                    "user_input": "你好",
                    "assistant_response": "你好！有什么我可以帮助你的吗？"
                },
                {
                    "user_input": "介绍一下人工智能",
                    "assistant_response": "人工智能（AI）是计算机科学的一个分支..."
                }
            ]
        },
        "dataset_name": "sharegpt",
        "lang": "zh"
    }
    
    column_names = ["conversation", "dataset_name", "lang"]
    
    # 调用真实LLM进行映射
    print(f"\n=== 测试SFT模式嵌套结构 ===")
    print(f"数据行: {json.dumps(row, ensure_ascii=False, indent=2)}")
    
    annotation_result = await convertor.invoke(state, column_names, row)
    
    print(f"\nLLM返回的映射结果:")
    print(json.dumps(annotation_result, ensure_ascii=False, indent=2))
    
    # 验证映射结果
    assert annotation_result is not None
    is_valid = validate_sft_mapping(annotation_result)
    assert is_valid, f"SFT映射结果验证失败: {annotation_result}"
    
    # 构建中间格式
    intermediate_record = convertor._build_intermediate_format_sft(
        row, annotation_result, "/test/sharegpt.json", state, 0
    )
    
    print(f"\n构建的中间格式:")
    print(json.dumps(intermediate_record, ensure_ascii=False, indent=2))
    
    # 验证中间格式
    assert intermediate_record is not None
    assert len(intermediate_record["messages"]) >= 2, "应该至少包含2条消息"
    
    print(f"\n✓ 嵌套结构测试通过")


# ============================================================================
# 集成测试4: 多字段拼接 - PT模式
# ============================================================================

@pytest.mark.asyncio
async def test_pt_mode_multi_field_concatenation():
    """测试PT模式：多字段拼接"""
    convertor = DataConvertor(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096
    )
    state = create_real_state("PT")
    
    # 模拟需要拼接的数据
    row = {
        "title": "人工智能简介",
        "body": "人工智能（AI）是计算机科学的一个分支...",
        "tags": "科技,AI,计算机",
        "dataset_name": "wikipedia",
        "lang": "zh"
    }
    
    column_names = list(row.keys())
    
    # 调用真实LLM进行映射
    print(f"\n=== 测试PT模式多字段拼接 ===")
    print(f"数据行: {json.dumps(row, ensure_ascii=False, indent=2)}")
    
    annotation_result = await convertor.invoke(state, column_names, row)
    
    print(f"\nLLM返回的映射结果:")
    print(json.dumps(annotation_result, ensure_ascii=False, indent=2))
    
    # 验证映射结果
    assert annotation_result is not None
    is_valid = validate_pt_mapping(annotation_result)
    assert is_valid, f"PT映射结果验证失败: {annotation_result}"
    
    # 构建中间格式
    intermediate_record = convertor._build_intermediate_format_pt(
        row, annotation_result, "/test/wikipedia.json", state, 0
    )
    
    print(f"\n构建的中间格式:")
    print(json.dumps(intermediate_record, ensure_ascii=False, indent=2))
    
    # 验证中间格式
    assert intermediate_record is not None
    assert intermediate_record["text"] is not None
    # 如果LLM返回了多字段拼接，text应该包含多个字段的内容
    if isinstance(annotation_result.get("text"), list):
        assert any(keyword in intermediate_record["text"] for keyword in ["人工智能", "简介", "科技"])
    
    print(f"\n✓ 多字段拼接测试通过")


# ============================================================================
# 集成测试5: Meta字段映射
# ============================================================================

@pytest.mark.asyncio
async def test_meta_fields_mapping():
    """测试Meta字段的完整映射"""
    convertor = DataConvertor(
        model_name=MODEL,
        temperature=0.0,
        max_tokens=4096
    )
    state = create_real_state("PT")
    
    # 模拟包含完整meta信息的数据
    row = {
        "content": "测试内容",
        "source_dataset": "wikipedia",
        "language_code": "zh",
        "created_time": "2024-01-01T00:00:00Z",
        "token_num": 100,
        "quality": 0.95,
        "original_id": "doc_123"
    }
    
    column_names = list(row.keys())
    
    # 调用真实LLM进行映射
    print(f"\n=== 测试Meta字段映射 ===")
    print(f"数据行: {json.dumps(row, ensure_ascii=False, indent=2)}")
    
    annotation_result = await convertor.invoke(state, column_names, row)
    
    print(f"\nLLM返回的映射结果:")
    print(json.dumps(annotation_result, ensure_ascii=False, indent=2))
    
    # 验证映射结果包含meta字段
    if "meta" in annotation_result and annotation_result["meta"]:
        meta = annotation_result["meta"]
        print(f"\nMeta字段映射:")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        
        # 构建中间格式
        intermediate_record = convertor._build_intermediate_format_pt(
            row, annotation_result, "/test/wikipedia.json", state, 0
        )
        
        print(f"\n构建的中间格式（包含meta）:")
        print(json.dumps(intermediate_record, ensure_ascii=False, indent=2))
        
        # 验证meta字段
        assert "meta" in intermediate_record
        assert intermediate_record["meta"] is not None
    
    print(f"\n✓ Meta字段映射测试通过")


# ============================================================================
# 主函数（用于直接运行）
# ============================================================================

if __name__ == "__main__":
    # 检查环境变量
    if not API_KEY or API_KEY == "test":
        print("错误: 需要设置 DF_API_KEY 环境变量")
        print("示例: export DF_API_KEY=your_api_key")
        exit(1)
    
    print(f"使用API配置:")
    print(f"  API URL: {API_URL}")
    print(f"  Model: {MODEL}")
    print(f"  API Key: {'*' * 10}...{API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    print()
    
    # 运行所有测试
    pytest.main([__file__, "-v", "-s"])

