"""
LLM 映射输出 JSON Schema 定义

本模块定义了数据转换Agent中LLM应返回的字段映射结构。
用于类型提示、验证和文档生成。
"""

from typing import Dict, List, Optional, Union, Literal, TypedDict, Any


# ============================================================================
# 类型别名
# ============================================================================

# 字段路径可以是单个字符串或数组（用于多字段拼接）
FieldPath = Union[str, List[str]]

# 字段路径或直接值（某些meta字段可以直接返回字符串值）
FieldPathOrValue = Union[str, None]

# Role枚举类型
MessageRole = Literal["user", "assistant", "system", "tool"]


# ============================================================================
# PT (Pretrain) 模式 Schema
# ============================================================================

class PTMetaMapping(TypedDict, total=False):
    """PT模式的meta字段映射"""
    source: Optional[FieldPathOrValue]  # 必须（meta存在时）
    language: Optional[FieldPathOrValue]  # 推荐
    timestamp: Optional[str]  # 可选
    token_count: Optional[str]  # 可选
    quality_score: Optional[str]  # 可选
    original_id: Optional[str]  # 可选


class PTMappingResult(TypedDict, total=False):
    """PT模式的LLM映射结果"""
    text: Optional[FieldPath]  # 必须（如果数据集相关）
    meta: Optional[PTMetaMapping]  # 推荐（至少包含source）


# ============================================================================
# SFT (Supervised Fine-Tuning) 模式 Schema
# ============================================================================

class SFTMessageMapping(TypedDict, total=False):
    """SFT模式的单条消息映射"""
    role: MessageRole  # 必须
    content: Optional[FieldPath]  # 必须（role存在时）
    loss_mask: Optional[bool]  # 可选，默认assistant为true，其他为false


class SFTMetaMapping(TypedDict, total=False):
    """SFT模式的meta字段映射（与PT模式相同）"""
    source: Optional[FieldPathOrValue]  # 必须（meta存在时）
    language: Optional[FieldPathOrValue]  # 推荐
    timestamp: Optional[str]  # 可选
    token_count: Optional[str]  # 可选
    quality_score: Optional[str]  # 可选
    original_id: Optional[str]  # 可选


class SFTMappingResult(TypedDict, total=False):
    """SFT模式的LLM映射结果"""
    messages: Optional[List[SFTMessageMapping]]  # 必须（如果数据集相关），至少1条
    system: Optional[FieldPathOrValue]  # 可选
    meta: Optional[SFTMetaMapping]  # 推荐（至少包含source）


# ============================================================================
# 统一映射结果类型（运行时根据category判断）
# ============================================================================

LLMMappingResult = Union[PTMappingResult, SFTMappingResult]


# ============================================================================
# 验证函数
# ============================================================================

def validate_pt_mapping(mapping: Dict[str, Any]) -> bool:
    """
    验证PT模式的映射结果
    
    Args:
        mapping: LLM返回的映射结果
        
    Returns:
        True if valid, False otherwise
    """
    # text必须存在且不为null（除非数据集不相关）
    if "text" not in mapping:
        return False
    
    # 如果meta存在，source必须存在
    if "meta" in mapping and mapping["meta"] is not None:
        meta = mapping["meta"]
        if not isinstance(meta, dict):
            return False
        if "source" not in meta or meta["source"] is None:
            return False
    
    return True


def validate_sft_mapping(mapping: Dict[str, Any]) -> bool:
    """
    验证SFT模式的映射结果
    
    Args:
        mapping: LLM返回的映射结果
        
    Returns:
        True if valid, False otherwise
    """
    # messages必须存在且为数组，长度至少为1
    if "messages" not in mapping:
        return False
    
    messages = mapping["messages"]
    if messages is None:
        return False  # 数据集不相关的情况
    
    if not isinstance(messages, list):
        return False
    
    if len(messages) == 0:
        return False
    
    # 验证每条消息
    valid_roles = {"user", "assistant", "system", "tool"}
    role_content_pairs = []  # 存储每条消息的 (role, content) 对
    
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or msg["role"] not in valid_roles:
            return False
        if "content" not in msg:
            return False
        
        role = msg["role"]
        content = msg["content"]
        
        # 将content规范化为字符串进行比较（处理列表情况）
        if isinstance(content, list):
            content_str = str(sorted(content))  # 排序后转为字符串
        else:
            content_str = str(content) if content is not None else ""
        
        role_content_pairs.append((role, content_str))
    
    # 检查：不同role的消息必须映射到不同的字段
    # 例外情况：当字段路径是 messages[*].content 时，允许不同role映射到相同路径
    # 因为系统会根据原始数据中每个元素的role字段进行匹配
    # 遍历所有消息对，检查是否有不同role映射到相同content
    for i, (role1, content1) in enumerate(role_content_pairs):
        for j, (role2, content2) in enumerate(role_content_pairs):
            if i != j and role1 != role2 and content1 == content2:
                # 检查是否是特殊情况：messages[*].content 模式
                # 允许 messages[*].content 或 messages[0].content 等模式
                content_path = content1 if isinstance(content1, str) else str(content1)
                if content_path and ("messages[" in content_path and ".content" in content_path):
                    # 这是特殊情况，允许不同role映射到相同路径
                    continue
                return False  # 不同role映射到相同字段，这是不允许的
    
    # 如果meta存在，source必须存在
    if "meta" in mapping and mapping["meta"] is not None:
        meta = mapping["meta"]
        if not isinstance(meta, dict):
            return False
        if "source" not in meta or meta["source"] is None:
            return False
    
    return True


def validate_mapping(mapping: Dict[str, Any], category: str) -> bool:
    """
    根据category验证映射结果
    
    Args:
        mapping: LLM返回的映射结果
        category: 数据类别 ("PT" 或 "SFT")
        
    Returns:
        True if valid, False otherwise
    """
    category_upper = category.upper()
    if category_upper == "PT":
        return validate_pt_mapping(mapping)
    elif category_upper == "SFT":
        return validate_sft_mapping(mapping)
    else:
        return False


# ============================================================================
# Schema文档字符串（用于生成提示词）
# ============================================================================

PT_SCHEMA_DOC = """
PT模式映射结果结构：
{
  "text": "field_path | [field_path, ...] | null",  // 必须（如果数据集相关）
  "meta": {  // 推荐（至少包含source）
    "source": "field_path | string_value | null",  // 必须（meta存在时）
    "language": "field_path | string_value | null",  // 推荐
    "timestamp": "field_path | null",  // 可选
    "token_count": "field_path | null",  // 可选
    "quality_score": "field_path | null",  // 可选
    "original_id": "field_path | null"  // 可选
  }
}
"""

SFT_SCHEMA_DOC = """
SFT模式映射结果结构：
{
  "messages": [  // 必须（如果数据集相关），至少1条
    {
      "role": "user | assistant | system | tool",  // 必须
      "content": "field_path | [field_path, ...] | null",  // 必须（role存在时）
      "loss_mask": true | false | null  // 可选，默认assistant为true，其他为false
    }
  ],
  "system": "field_path | string_value | null",  // 可选
  "meta": {  // 推荐（至少包含source）
    "source": "field_path | string_value | null",  // 必须（meta存在时）
    "language": "field_path | string_value | null",  // 推荐
    "timestamp": "field_path | null",  // 可选
    "token_count": "field_path | null",  // 可选
    "quality_score": "field_path | null",  // 可选
    "original_id": "field_path | null"  // 可选
  }
}
"""

