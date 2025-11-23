"""
Schema定义模块

本模块包含数据转换Agent中使用的各种Schema定义。
"""

from dataflow_agent.schemas.llm_mapping_schema import (
    # 类型别名
    FieldPath,
    FieldPathOrValue,
    MessageRole,
    # PT模式
    PTMetaMapping,
    PTMappingResult,
    # SFT模式
    SFTMessageMapping,
    SFTMetaMapping,
    SFTMappingResult,
    # 统一类型
    LLMMappingResult,
    # 验证函数
    validate_pt_mapping,
    validate_sft_mapping,
    validate_mapping,
    # Schema文档
    PT_SCHEMA_DOC,
    SFT_SCHEMA_DOC,
)

__all__ = [
    "FieldPath",
    "FieldPathOrValue",
    "MessageRole",
    "PTMetaMapping",
    "PTMappingResult",
    "SFTMessageMapping",
    "SFTMetaMapping",
    "SFTMappingResult",
    "LLMMappingResult",
    "validate_pt_mapping",
    "validate_sft_mapping",
    "validate_mapping",
    "PT_SCHEMA_DOC",
    "SFT_SCHEMA_DOC",
]

