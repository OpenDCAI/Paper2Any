"""
测试 DataConvertor 的数据转换功能
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

运行方式:
  pytest tests/test_dataconvertor.py -v -s
  或直接: python tests/test_dataconvertor.py
"""

from __future__ import annotations

import pytest
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# 直接导入避免循环导入问题
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow_agent.agentroles.data_convertor import DataConvertor, UniversalDataConvertor
from dataflow_agent.state import DataCollectionState, DataCollectionRequest
from dataflow_agent.schemas.llm_mapping_schema import validate_pt_mapping, validate_sft_mapping, validate_mapping


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_state(category: str = "PT", download_dir: str = None) -> DataCollectionState:
    """创建测试用的DataCollectionState"""
    if download_dir is None:
        download_dir = tempfile.mkdtemp()
    
    request = DataCollectionRequest(
        language="zh",
        chat_api_url="http://test",
        api_key="test",
        model="gpt-4o",
        target="测试数据",
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
# 测试1: ID生成工具函数
# ============================================================================

def test_generate_record_id():
    """测试ID生成功能"""
    convertor = DataConvertor()
    
    # 测试基本ID生成
    id1 = convertor._generate_record_id("test content", "/path/to/file", 0)
    assert len(id1) == 32
    assert isinstance(id1, str)
    
    # 测试相同内容生成相同ID
    id2 = convertor._generate_record_id("test content", "/path/to/file", 0)
    assert id1 == id2
    
    # 测试不同内容生成不同ID
    id3 = convertor._generate_record_id("different content", "/path/to/file", 0)
    assert id1 != id3
    
    # 测试不同文件路径生成不同ID
    id4 = convertor._generate_record_id("test content", "/different/path", 0)
    assert id1 != id4
    
    # 测试不同索引生成不同ID
    id5 = convertor._generate_record_id("test content", "/path/to/file", 1)
    assert id1 != id5


# ============================================================================
# 测试2: Meta字段提取
# ============================================================================

def test_extract_meta_fields():
    """测试meta字段提取功能"""
    convertor = DataConvertor()
    state = create_test_state()
    
    # 测试用例1: 完整的meta映射
    row = {
        "dataset_name": "wikipedia",
        "lang": "en",
        "created_at": "2024-01-01",
        "token_count": 1000,
        "quality": 0.85,
        "id": "12345"
    }
    meta_mapping = {
        "source": "dataset_name",
        "language": "lang",
        "timestamp": "created_at",
        "token_count": "token_count",
        "quality_score": "quality",
        "original_id": "id"
    }
    
    meta = convertor._extract_meta_fields(row, meta_mapping, "/test/file.json", state)
    assert meta["source"] == "wikipedia"
    assert meta["language"] == "en"
    assert meta["timestamp"] == "2024-01-01"
    assert meta["token_count"] == 1000
    assert meta["quality_score"] == 0.85
    assert meta["original_id"] == "12345"
    
    # 测试用例2: 直接字符串值
    meta_mapping2 = {
        "source": "wikipedia",  # 直接字符串值
        "language": "zh"  # 直接字符串值
    }
    meta2 = convertor._extract_meta_fields(row, meta_mapping2, "/test/file.json", state)
    assert meta2["source"] == "wikipedia"
    assert meta2["language"] == "zh"
    
    # 测试用例3: 缺失字段，使用默认值
    meta_mapping3 = {}
    # 使用更明确的文件路径，确保stem是wikipedia
    meta3 = convertor._extract_meta_fields(row, meta_mapping3, "/data/wikipedia.json", state)
    assert "source" in meta3
    # 从文件名推断：/data/wikipedia.json -> stem是wikipedia
    # 实际实现：先检查stem，如果stem为空才检查父目录
    assert meta3["source"] == "wikipedia"  # 从文件名推断
    assert meta3["language"] == "zh"  # 从state获取
    
    # 测试用例4: 多字段拼接（任务22）
    row4 = {
        "author": "John",
        "title": "Test Article"
    }
    meta_mapping4 = {
        "source": ["author", "title"]  # 多字段拼接
    }
    meta4 = convertor._extract_meta_fields(row4, meta_mapping4, "/test/file.json", state)
    # 多字段拼接使用下划线连接
    assert "John" in meta4["source"]
    assert "title" in meta4["source"] or "Test" in meta4["source"]  # 可能是title字段的值
    
    # 测试用例5: 数值类型转换
    row5 = {
        "tokens": "1000",  # 字符串格式
        "quality": 0.85  # 浮点数格式
    }
    meta_mapping5 = {
        "token_count": "tokens",
        "quality_score": "quality"
    }
    meta5 = convertor._extract_meta_fields(row5, meta_mapping5, "/test/file.json", state)
    assert isinstance(meta5["token_count"], int)
    assert meta5["token_count"] == 1000
    assert isinstance(meta5["quality_score"], float)
    assert meta5["quality_score"] == 0.85


# ============================================================================
# 测试3: System字段提取
# ============================================================================

def test_extract_system_field():
    """测试system字段提取功能"""
    convertor = DataConvertor()
    
    # 测试用例1: 字段路径映射
    row = {
        "system_prompt": "You are a helpful assistant."
    }
    system_mapping = "system_prompt"
    system = convertor._extract_system_field(row, system_mapping)
    # system_prompt会被_is_field_path识别为字段路径（因为包含下划线且是alnum）
    # 会调用_extract_field_value_raw，使用_traverse_field_tokens提取
    # 对于简单字段名"system_prompt"，tokens=["system_prompt"]
    # _traverse_field_tokens会递归调用，当tokens为空时返回current值
    # 所以应该能正确提取
    assert system == "You are a helpful assistant.", f"Expected 'You are a helpful assistant.', got {system}"
    
    # 测试用例2: 直接字符串值
    system_mapping2 = "You are a helpful assistant."
    system2 = convertor._extract_system_field(row, system_mapping2)
    assert system2 == "You are a helpful assistant."
    
    # 测试用例3: null值
    system3 = convertor._extract_system_field(row, None)
    assert system3 is None
    
    # 测试用例4: 嵌套字段路径
    row4 = {
        "meta": {
            "system": "System instruction"
        }
    }
    system_mapping4 = "meta.system"
    system4 = convertor._extract_system_field(row4, system_mapping4)
    assert system4 == "System instruction"


# ============================================================================
# 测试4: Messages结构提取
# ============================================================================

def test_extract_messages_structure():
    """测试messages结构提取功能"""
    convertor = DataConvertor()
    
    # 测试用例1: 简单的QA对
    row = {
        "question": "What is AI?",
        "answer": "AI is artificial intelligence."
    }
    messages_mapping = [
        {
            "role": "user",
            "content": "question",
            "loss_mask": False
        },
        {
            "role": "assistant",
            "content": "answer",
            "loss_mask": True
        }
    ]
    column_names = ["question", "answer"]
    
    messages = convertor._extract_messages_structure(row, messages_mapping, column_names)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is AI?"
    assert messages[0]["loss_mask"] is False
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "AI is artificial intelligence."
    assert messages[1]["loss_mask"] is True
    
    # 测试用例2: 缺少role字段（应该跳过）
    messages_mapping2 = [
        {
            "content": "question"
            # 缺少role
        }
    ]
    messages2 = convertor._extract_messages_structure(row, messages_mapping2, column_names)
    assert len(messages2) == 0  # 应该被跳过
    
    # 测试用例3: 无效的role（应该跳过）
    messages_mapping3 = [
        {
            "role": "invalid_role",
            "content": "question"
        }
    ]
    messages3 = convertor._extract_messages_structure(row, messages_mapping3, column_names)
    assert len(messages3) == 0  # 应该被跳过
    
    # 测试用例4: 多字段拼接
    row4 = {
        "stem": "以下哪项正确？",
        "options": ["A. 选项1", "B. 选项2"]
    }
    messages_mapping4 = [
        {
            "role": "user",
            "content": ["stem", "options[*]"],
            "loss_mask": False
        }
    ]
    column_names4 = ["stem", "options"]
    messages4 = convertor._extract_messages_structure(row4, messages_mapping4, column_names4)
    assert len(messages4) == 1
    assert "以下哪项正确？" in messages4[0]["content"]
    assert "A. 选项1" in messages4[0]["content"]
    assert "B. 选项2" in messages4[0]["content"]
    
    # 测试用例5: loss_mask默认值
    messages_mapping5 = [
        {
            "role": "user",
            "content": "question"
            # loss_mask未指定
        },
        {
            "role": "assistant",
            "content": "answer"
            # loss_mask未指定
        }
    ]
    messages5 = convertor._extract_messages_structure(row, messages_mapping5, column_names)
    assert len(messages5) == 2
    assert messages5[0]["loss_mask"] is False  # user默认false
    assert messages5[1]["loss_mask"] is True  # assistant默认true
    
    # 测试用例6: 嵌套结构（任务17）
    row6 = {
        "dialogues": [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "How are you?", "assistant": "I'm doing well."}
        ]
    }
    # 注意：对于嵌套结构，LLM应该返回多个消息映射，每个对应一个turn
    # 这里测试单个消息映射包含嵌套路径的情况
    messages_mapping6 = [
        {
            "role": "user",
            "content": "dialogues[*].user"
        },
        {
            "role": "assistant",
            "content": "dialogues[*].assistant"
        }
    ]
    column_names6 = ["dialogues"]
    messages6 = convertor._extract_messages_structure(row6, messages_mapping6, column_names6)
    # 嵌套结构会展开所有值，取第一个
    assert len(messages6) == 2
    assert messages6[0]["content"] == "Hello"
    assert messages6[1]["content"] == "Hi there!"


# ============================================================================
# 测试5: 中间格式构建（PT模式）
# ============================================================================

def test_build_intermediate_format_pt():
    """测试PT模式中间格式构建"""
    convertor = DataConvertor()
    state = create_test_state("PT")
    
    # 测试用例1: 完整的PT映射
    row = {
        "content": "This is a long article about AI...",
        "dataset_name": "wikipedia",
        "lang": "en",
        "id": "12345"
    }
    annotation_result = {
        "text": "content",
        "meta": {
            "source": "dataset_name",
            "language": "lang",
            "original_id": "id"
        }
    }
    
    result = convertor._build_intermediate_format_pt(
        row, annotation_result, "/test/file.json", state, 0
    )
    
    assert result is not None
    assert "id" in result
    assert result["dataset_type"] == "pretrain"
    assert result["text"] == "This is a long article about AI..."
    assert "meta" in result
    assert result["meta"]["source"] == "wikipedia"
    assert result["meta"]["language"] == "en"
    assert result["meta"]["original_id"] == "12345"
    
    # 测试用例2: 多字段拼接的text
    row2 = {
        "title": "Article Title",
        "body": "Article body..."
    }
    annotation_result2 = {
        "text": ["title", "body"],
        "meta": {
            "source": "wikipedia"
        }
    }
    
    result2 = convertor._build_intermediate_format_pt(
        row2, annotation_result2, "/test/file.json", state, 0
    )
    assert result2 is not None
    assert "Article Title" in result2["text"]
    assert "Article body..." in result2["text"]
    
    # 测试用例3: 缺少text字段（应该返回None）
    annotation_result3 = {
        "meta": {
            "source": "wikipedia"
        }
    }
    result3 = convertor._build_intermediate_format_pt(
        row, annotation_result3, "/test/file.json", state, 0
    )
    assert result3 is None
    
    # 测试用例4: 验证必填字段
    row4 = {
        "content": ""  # 空字符串
    }
    annotation_result4 = {
        "text": "content"
    }
    result4 = convertor._build_intermediate_format_pt(
        row4, annotation_result4, "/test/file.json", state, 0
    )
    # 空字符串应该被跳过
    assert result4 is None


# ============================================================================
# 测试6: 中间格式构建（SFT模式）
# ============================================================================

def test_build_intermediate_format_sft():
    """测试SFT模式中间格式构建"""
    convertor = DataConvertor()
    state = create_test_state("SFT")
    
    # 测试用例1: 完整的SFT映射
    row = {
        "question": "What is AI?",
        "answer": "AI is artificial intelligence.",
        "dataset_name": "alpaca",
        "lang": "en"
    }
    annotation_result = {
        "messages": [
            {
                "role": "user",
                "content": "question",
                "loss_mask": False
            },
            {
                "role": "assistant",
                "content": "answer",
                "loss_mask": True
            }
        ],
        "meta": {
            "source": "dataset_name",
            "language": "lang"
        }
    }
    
    result = convertor._build_intermediate_format_sft(
        row, annotation_result, "/test/file.json", state, 0
    )
    
    assert result is not None
    assert "id" in result
    assert result["dataset_type"] == "sft"
    assert "messages" in result
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "What is AI?"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "AI is artificial intelligence."
    assert "meta" in result
    assert result["meta"]["source"] == "alpaca"
    
    # 测试用例2: 包含system字段
    row2 = {
        "system_prompt": "You are helpful.",
        "input": "Hello",
        "output": "Hi!"
    }
    annotation_result2 = {
        "messages": [
            {
                "role": "user",
                "content": "input"
            },
            {
                "role": "assistant",
                "content": "output"
            }
        ],
        "system": "system_prompt",
        "meta": {
            "source": "test"
        }
    }
    
    result2 = convertor._build_intermediate_format_sft(
        row2, annotation_result2, "/test/file.json", state, 0
    )
    assert result2 is not None
    assert result2["system"] == "You are helpful."
    
    # 测试用例3: 缺少messages字段（应该返回None）
    annotation_result3 = {
        "meta": {
            "source": "test"
        }
    }
    result3 = convertor._build_intermediate_format_sft(
        row, annotation_result3, "/test/file.json", state, 0
    )
    assert result3 is None
    
    # 测试用例4: 缺少role的消息（应该返回None）
    annotation_result4 = {
        "messages": [
            {
                "content": "question"
                # 缺少role
            }
        ]
    }
    result4 = convertor._build_intermediate_format_sft(
        row, annotation_result4, "/test/file.json", state, 0
    )
    assert result4 is None  # 因为_extract_messages_structure会跳过缺少role的消息


# ============================================================================
# 测试7: Schema验证
# ============================================================================

def test_validate_pt_mapping():
    """测试PT模式映射验证"""
    # 有效映射
    valid_mapping = {
        "text": "content",
        "meta": {
            "source": "wikipedia"
        }
    }
    assert validate_pt_mapping(valid_mapping) is True
    
    # 无效映射：缺少text
    invalid_mapping1 = {
        "meta": {
            "source": "wikipedia"
        }
    }
    assert validate_pt_mapping(invalid_mapping1) is False
    
    # 无效映射：meta存在但缺少source
    invalid_mapping2 = {
        "text": "content",
        "meta": {
            "language": "en"
        }
    }
    assert validate_pt_mapping(invalid_mapping2) is False
    
    # 有效映射：text为null（数据集不相关）
    valid_mapping2 = {
        "text": None,
        "meta": None
    }
    assert validate_pt_mapping(valid_mapping2) is True  # text字段存在即可


def test_validate_sft_mapping():
    """测试SFT模式映射验证"""
    # 有效映射
    valid_mapping = {
        "messages": [
            {
                "role": "user",
                "content": "question"
            },
            {
                "role": "assistant",
                "content": "answer"
            }
        ],
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_sft_mapping(valid_mapping) is True
    
    # 无效映射：缺少messages
    invalid_mapping1 = {
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_sft_mapping(invalid_mapping1) is False
    
    # 无效映射：messages为空
    invalid_mapping2 = {
        "messages": [],
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_sft_mapping(invalid_mapping2) is False
    
    # 无效映射：消息缺少role
    invalid_mapping3 = {
        "messages": [
            {
                "content": "question"
            }
        ],
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_sft_mapping(invalid_mapping3) is False
    
    # 无效映射：无效的role
    invalid_mapping4 = {
        "messages": [
            {
                "role": "invalid",
                "content": "question"
            }
        ],
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_sft_mapping(invalid_mapping4) is False


def test_validate_mapping():
    """测试通用验证函数"""
    pt_mapping = {
        "text": "content",
        "meta": {
            "source": "wikipedia"
        }
    }
    assert validate_mapping(pt_mapping, "PT") is True
    assert validate_mapping(pt_mapping, "pt") is True  # 大小写不敏感
    
    sft_mapping = {
        "messages": [
            {
                "role": "user",
                "content": "question"
            }
        ],
        "meta": {
            "source": "alpaca"
        }
    }
    assert validate_mapping(sft_mapping, "SFT") is True
    assert validate_mapping(sft_mapping, "sft") is True  # 大小写不敏感


# ============================================================================
# 测试8: 字段提取工具方法
# ============================================================================

def test_extract_field_value_raw():
    """测试原始值提取（支持非字符串类型）"""
    convertor = DataConvertor()
    
    # 测试用例1: 字符串值
    row = {
        "text": "Hello"
    }
    value = convertor._extract_field_value_raw(row, "text")
    assert value == "Hello"
    
    # 测试用例2: 整数值
    row2 = {
        "count": 1000
    }
    value2 = convertor._extract_field_value_raw(row2, "count")
    assert value2 == 1000
    assert isinstance(value2, int)
    
    # 测试用例3: 浮点数值
    row3 = {
        "score": 0.85
    }
    value3 = convertor._extract_field_value_raw(row3, "score")
    assert value3 == 0.85
    assert isinstance(value3, float)
    
    # 测试用例4: 嵌套结构
    row4 = {
        "meta": {
            "timestamp": 1234567890
        }
    }
    value4 = convertor._extract_field_value_raw(row4, "meta.timestamp")
    assert value4 == 1234567890
    
    # 测试用例5: 数组路径（返回拼接后的字符串）
    row5 = {
        "title": "Title",
        "body": "Body"
    }
    value5 = convertor._extract_field_value_raw(row5, ["title", "body"])
    assert isinstance(value5, str)
    assert "Title" in value5
    assert "Body" in value5


# ============================================================================
# 测试9: 边界情况
# ============================================================================

def test_edge_cases():
    """测试边界情况"""
    convertor = DataConvertor()
    state = create_test_state("PT")
    
    # 测试用例1: 空行
    row1 = {}
    annotation_result1 = {
        "text": "content"
    }
    result1 = convertor._build_intermediate_format_pt(
        row1, annotation_result1, "/test/file.json", state, 0
    )
    assert result1 is None  # 无法提取text
    
    # 测试用例2: None值
    row2 = {
        "content": None
    }
    result2 = convertor._build_intermediate_format_pt(
        row2, annotation_result1, "/test/file.json", state, 0
    )
    assert result2 is None
    
    # 测试用例3: 嵌套空值
    row3 = {
        "data": {
            "text": None
        }
    }
    annotation_result3 = {
        "text": "data.text"
    }
    result3 = convertor._build_intermediate_format_pt(
        row3, annotation_result3, "/test/file.json", state, 0
    )
    assert result3 is None
    
    # 测试用例4: 数组路径但数组为空
    row4 = {
        "items": []
    }
    annotation_result4 = {
        "text": "items[*]"
    }
    result4 = convertor._build_intermediate_format_pt(
        row4, annotation_result4, "/test/file.json", state, 0
    )
    assert result4 is None
    
    # 测试用例5: SFT模式中messages为空列表
    row5 = {
        "question": "Test"
    }
    annotation_result5 = {
        "messages": []
    }
    result5 = convertor._build_intermediate_format_sft(
        row5, annotation_result5, "/test/file.json", state, 0
    )
    assert result5 is None


# ============================================================================
# 测试10: 嵌套对话结构
# ============================================================================

def test_nested_conversation_structure():
    """测试嵌套对话结构的处理（任务17）"""
    convertor = DataConvertor()
    state = create_test_state("SFT")
    
    # 测试用例：嵌套对话结构
    # 注意：对于嵌套结构，LLM应该返回多个消息映射，每个对应一个turn
    # 这里测试LLM正确映射的情况
    row = {
        "conversation": {
            "turns": [
                {
                    "user_input": "Hello",
                    "assistant_response": "Hi there!"
                },
                {
                    "user_input": "How are you?",
                    "assistant_response": "I'm doing well."
                }
            ]
        }
    }
    
    # LLM应该为每个turn返回一个消息映射
    annotation_result = {
        "messages": [
            {
                "role": "user",
                "content": "conversation.turns[0].user_input"
            },
            {
                "role": "assistant",
                "content": "conversation.turns[0].assistant_response"
            },
            {
                "role": "user",
                "content": "conversation.turns[1].user_input"
            },
            {
                "role": "assistant",
                "content": "conversation.turns[1].assistant_response"
            }
        ],
        "meta": {
            "source": "test"
        }
    }
    
    result = convertor._build_intermediate_format_sft(
        row, annotation_result, "/test/file.json", state, 0
    )
    
    assert result is not None
    assert len(result["messages"]) == 4
    assert result["messages"][0]["content"] == "Hello"
    assert result["messages"][1]["content"] == "Hi there!"
    assert result["messages"][2]["content"] == "How are you?"
    assert result["messages"][3]["content"] == "I'm doing well."


# ============================================================================
# 测试11: 多字段拼接
# ============================================================================

def test_multi_field_concatenation():
    """测试多字段拼接功能"""
    convertor = DataConvertor()
    state = create_test_state("PT")
    
    # 测试用例1: text字段多字段拼接
    row = {
        "title": "Article Title",
        "body": "Article body content...",
        "tags": ["tag1", "tag2"]
    }
    annotation_result = {
        "text": ["title", "body", "tags[*]"],
        "meta": {
            "source": "wikipedia"
        }
    }
    
    result = convertor._build_intermediate_format_pt(
        row, annotation_result, "/test/file.json", state, 0
    )
    
    assert result is not None
    assert "Article Title" in result["text"]
    assert "Article body content..." in result["text"]
    assert "tag1" in result["text"]
    assert "tag2" in result["text"]
    
    # 测试用例2: meta字段多字段拼接（任务22）
    row2 = {
        "author": "John",
        "title": "Test"
    }
    annotation_result2 = {
        "text": "title",
        "meta": {
            "source": ["author", "title"]  # 多字段拼接
        }
    }
    
    result2 = convertor._build_intermediate_format_pt(
        row2, annotation_result2, "/test/file.json", state, 0
    )
    assert result2 is not None
    # 多字段拼接使用下划线连接
    source_value = result2["meta"]["source"]
    assert "John" in source_value
    assert "Test" in source_value


# ============================================================================
# 测试12: 完整流程测试（模拟）
# ============================================================================

def test_complete_pt_workflow():
    """测试PT模式的完整工作流（不调用真实LLM）"""
    convertor = DataConvertor()
    state = create_test_state("PT")
    
    # 模拟LLM返回的映射结果
    annotation_result = {
        "text": "content",
        "meta": {
            "source": "dataset_name",
            "language": "lang",
            "timestamp": "created_at",
            "original_id": "id"
        }
    }
    
    # 模拟数据行
    row = {
        "content": "This is a test article about machine learning...",
        "dataset_name": "wikipedia",
        "lang": "en",
        "created_at": "2024-01-01T00:00:00Z",
        "id": "article_123"
    }
    
    # 构建中间格式
    result = convertor._build_intermediate_format_pt(
        row, annotation_result, "/test/wikipedia.json", state, 0
    )
    
    # 验证结果
    assert result is not None
    assert result["dataset_type"] == "pretrain"
    assert result["text"] == "This is a test article about machine learning..."
    assert result["meta"]["source"] == "wikipedia"
    assert result["meta"]["language"] == "en"
    assert result["meta"]["timestamp"] == "2024-01-01T00:00:00Z"
    assert result["meta"]["original_id"] == "article_123"
    assert "id" in result
    assert len(result["id"]) == 32


def test_complete_sft_workflow():
    """测试SFT模式的完整工作流（不调用真实LLM）"""
    convertor = DataConvertor()
    state = create_test_state("SFT")
    
    # 模拟LLM返回的映射结果
    annotation_result = {
        "messages": [
            {
                "role": "user",
                "content": "question",
                "loss_mask": False
            },
            {
                "role": "assistant",
                "content": "answer",
                "loss_mask": True
            }
        ],
        "system": "system_prompt",
        "meta": {
            "source": "alpaca",
            "language": "en"
        }
    }
    
    # 模拟数据行
    row = {
        "question": "Explain quantum computing.",
        "answer": "Quantum computing uses quantum mechanical phenomena...",
        "system_prompt": "You are a helpful assistant.",
        "dataset_name": "alpaca",  # 修正：使用dataset_name字段，值为"alpaca"
        "lang": "en"
    }
    
    # 构建中间格式
    result = convertor._build_intermediate_format_sft(
        row, annotation_result, "/test/alpaca.json", state, 0
    )
    
    # 验证结果
    assert result is not None
    assert result["dataset_type"] == "sft"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Explain quantum computing."
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "Quantum computing uses quantum mechanical phenomena..."
    assert result["system"] == "You are a helpful assistant."
    assert result["meta"]["source"] == "alpaca"
    assert "id" in result
    assert len(result["id"]) == 32


# ============================================================================
# 测试13: 字段路径解析
# ============================================================================

def test_field_path_parsing():
    """测试各种字段路径的解析"""
    convertor = DataConvertor()
    
    # 测试用例1: 简单字段
    row = {"text": "Hello"}
    values = convertor._extract_text_values(row, "text")
    assert values == ["Hello"]
    
    # 测试用例2: 嵌套字段
    row2 = {"meta": {"text": "Nested"}}
    values2 = convertor._extract_text_values(row2, "meta.text")
    assert values2 == ["Nested"]
    
    # 测试用例3: 数组索引
    row3 = {"items": ["first", "second", "third"]}
    values3 = convertor._extract_text_values(row3, "items[0]")
    assert values3 == ["first"]
    
    # 测试用例4: 数组通配符
    row4 = {"items": ["first", "second"]}
    values4 = convertor._extract_text_values(row4, "items[*]")
    assert len(values4) == 2
    assert "first" in values4
    assert "second" in values4
    
    # 测试用例5: 复杂嵌套
    row5 = {
        "dialogues": [
            {"turns": [{"text": "Hello"}, {"text": "Hi"}]},
            {"turns": [{"text": "How are you?"}]}
        ]
    }
    values5 = convertor._extract_text_values(row5, "dialogues[*].turns[*].text")
    assert len(values5) >= 1  # 至少提取到一个值


# ============================================================================
# 主函数（用于直接运行）
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

