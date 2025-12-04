# dataflow_agent/agentroles/data_convertor.py

from __future__ import annotations
import asyncio
import os
import re
import json
import zipfile
import tarfile
import gzip
import bz2
import lzma
import shutil
import tempfile
import hashlib
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Type, Tuple, TYPE_CHECKING
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, START, END
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DataCollectionState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow_agent.schemas import validate_mapping
from dataflow_agent.toolkits.datatool.data_convertor_tools import (
    SimpleDataset,
    _build_simple_dataset,
    _ensure_hf_cache_env,
)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

log = get_logger(__name__)

class DataConvertor:

    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 max_sample_length: int = 200,
                 num_sample_records: int = 3):
        """
        初始化Agent
        
        Args:
            model_name: 模型名称
            temperature: 模型温度
            max_tokens: 最大token数
            max_sample_length: 每个字段的最大采样长度（字符数）
            num_sample_records: 采样记录数量
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_sample_length = max_sample_length
        self.num_sample_records = num_sample_records
        self.tool_mode = "auto"  # 默认工具选择模式

    @property
    def role_name(self) -> str:
        return "data_convertor"
    
    # --- 原有的提示词 (用于数据映射) ---
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_conversion"
    
    @property
    def task_prompt_template_name_pt(self) -> str:
        return "task_prompt_for_data_conversion_pt"
    
    @property
    def task_prompt_template_name_sft(self) -> str:
        return "task_prompt_for_data_conversion_sft"

    # --- 新增：用于文件发现的提示词 ---
    @property
    def system_prompt_file_discovery(self) -> str:
        return "system_prompt_for_file_discovery"
    
    @property
    def task_prompt_file_discovery(self) -> str:
        return "task_prompt_for_file_discovery"
    
    def _truncate_value(self, value: Any, max_length: int = None) -> Any:
        """
        截断单个值，防止过长。
        
        Args:
            value: 原始值
            max_length: 最大长度，默认使用 self.max_sample_length
            
        Returns:
            截断后的值
        """
        if max_length is None:
            max_length = self.max_sample_length
            
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length] + "..."
            return value
        elif isinstance(value, (list, tuple)):
            # 对列表/元组，只保留前几个元素
            if len(value) > 3:
                return [self._truncate_value(v, max_length) for v in value[:3]] + ["..."]
            return [self._truncate_value(v, max_length) for v in value]
        elif isinstance(value, dict):
            # 对字典，只保留前几个键
            if len(value) > 3:
                truncated = {k: self._truncate_value(v, max_length) for k, v in list(value.items())[:3]}
                truncated["..."] = "..."
                return truncated
            return {k: self._truncate_value(v, max_length) for k, v in value.items()}
        else:
            return value
    
    def _sample_records(self, dataset: Any, num_samples: int = None) -> List[Dict[str, Any]]:
        """
        从数据集中随机采样指定数量的记录，并截断每个字段的值。
        
        Args:
            dataset: 数据集对象（Dataset）
            num_samples: 采样数量，默认使用 self.num_sample_records
            
        Returns:
            采样并截断后的记录列表
        """
        if num_samples is None:
            num_samples = self.num_sample_records
            
        import random
        
        dataset_size = len(dataset)
        if dataset_size == 0:
            return []
        
        # 确定采样数量（不超过数据集大小）
        actual_samples = min(num_samples, dataset_size)
        
        # 随机采样索引
        if dataset_size <= actual_samples:
            # 如果数据集很小，就全部取
            sample_indices = list(range(dataset_size))
        else:
            # 随机采样
            sample_indices = random.sample(range(dataset_size), actual_samples)
        
        # 获取采样记录并截断
        sampled_records = []
        for idx in sample_indices:
            record = dataset[idx]
            # 截断每个字段的值
            truncated_record = {k: self._truncate_value(v) for k, v in record.items()}
            sampled_records.append(truncated_record)
        
        log.info(f"从数据集中随机采样了 {len(sampled_records)} 条记录（共 {dataset_size} 条）")
        return sampled_records

    FIELD_TOKEN_PATTERN = re.compile(r"([^\[\]]+)(?:\[(.*?)\])?")

    def _try_parse_list_string(self, value: Any) -> Any:
        """
        尝试将字符串解析为列表
        
        Args:
            value: 可能是字符串形式的列表（如 "['a', 'b']" 或 '["a", "b"]'）
            
        Returns:
            解析后的列表，如果无法解析则返回原值
        """
        if not isinstance(value, str):
            return value
        
        # 尝试使用 ast.literal_eval 解析Python字面量（包括列表）
        try:
            import ast
            parsed = ast.literal_eval(value.strip())
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        
        # 尝试使用 json.loads 解析JSON格式的列表
        try:
            parsed = json.loads(value.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        
        return value

    def _field_exists_in_columns(self, field_spec: Optional[Any], column_names: List[str]) -> bool:
        if field_spec is None:
            return False
        if isinstance(field_spec, list):
            if not field_spec:
                return False
            return all(self._field_exists_in_columns(spec, column_names) for spec in field_spec)
        token = field_spec.split(".")[0]
        token = token.split("[")[0]
        return token in column_names

    def _normalize_field_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        # 特殊处理：如果值是列表，将列表中的元素用空格连接
        # 这对于处理像 ["I", "heard", "a", "sentence", ...] 这样的列表字段很重要
        if isinstance(value, (list, tuple)):
            # 过滤掉 None 和空字符串，然后将元素转换为字符串并用空格连接
            parts = []
            for item in value:
                if item is not None:
                    item_str = str(item).strip()
                    if item_str:  # 只添加非空字符串
                        parts.append(item_str)
            if parts:
                return " ".join(parts)
            return None
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _traverse_field_tokens(self, current: Any, tokens: List[str]) -> List[Any]:
        if current is None:
            return []
        if not tokens:
            if isinstance(current, list):
                # 特殊处理：如果列表中的元素都是字符串或数字，直接返回整个列表
                # 这对于处理像 ["I", "heard", "a", "sentence", ...] 这样的列表字段很重要
                if len(current) > 0 and all(isinstance(item, (str, int, float)) for item in current):
                    return [current]  # 返回整个列表，让 _normalize_field_value 处理
                # 否则，遍历列表中的每个元素
                results: List[Any] = []
                for item in current:
                    results.extend(self._traverse_field_tokens(item, []))
                return results
            if isinstance(current, dict):
                return list(current.items())
            return [current]

        token = tokens[0]
        match = self.FIELD_TOKEN_PATTERN.match(token)
        if not match:
            return []
        name, index = match.group(1), match.group(2)

        if isinstance(current, dict):
            next_value = current.get(name)
        else:
            return []

        # 如果指定了索引（如 corrections[0]），尝试将字符串形式的列表解析为真正的列表
        if index is not None and index != "":
            next_value = self._try_parse_list_string(next_value)

        if index is None or index == "":
            return self._traverse_field_tokens(next_value, tokens[1:])

        if not isinstance(next_value, list):
            if isinstance(next_value, dict):
                results: List[Any] = []
                for key, item in next_value.items():
                    child_results = self._traverse_field_tokens(item, tokens[1:])
                    if not child_results:
                        results.append((key, item))
                        continue
                    for child in child_results:
                        results.append((key, child))
                return results
            return []

        results: List[Any] = []
        if index == "*" or index.lower() == "all":
            for item in next_value:
                results.extend(self._traverse_field_tokens(item, tokens[1:]))
        else:
            try:
                idx = int(index)
                if 0 <= idx < len(next_value):
                    results.extend(self._traverse_field_tokens(next_value[idx], tokens[1:]))
            except ValueError:
                return []
        return results

    def _stringify_structure(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, dict):
            parts: List[str] = []
            for key, sub_value in value.items():
                key_str = self._normalize_field_value(key) or str(key)
                sub_str = self._stringify_structure(sub_value)
                if sub_str:
                    parts.append(f"{key_str}. {sub_str}")
                else:
                    parts.append(key_str)
            return "; ".join(part for part in parts if part) if parts else None
        if isinstance(value, (list, tuple, set)):
            parts: List[str] = []
            for item in value:
                sub_str = self._stringify_structure(item)
                if sub_str:
                    parts.append(sub_str)
            return "; ".join(part for part in parts if part) if parts else None
        return self._normalize_field_value(value)

    def _format_mapping_entry(self, key: Any, content: Any) -> Optional[str]:
        key_str = self._normalize_field_value(key) or str(key)
        content_str = self._stringify_structure(content)
        if content_str:
            return f"{key_str}. {content_str}"
        return key_str

    def _extract_field_values(self, row: Dict[str, Any], field_spec: Optional[str]) -> List[str]:
        if not field_spec:
            return []
        field_spec = field_spec.strip()
        if not field_spec:
            return []
        tokens = field_spec.split(".")
        raw_values = self._traverse_field_tokens(row, tokens)
        normalized: List[str] = []
        for value in raw_values:
            if isinstance(value, tuple) and len(value) == 2:
                entry = self._format_mapping_entry(value[0], value[1])
                if entry:
                    normalized.append(entry)
                continue
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    entry = self._format_mapping_entry(key, sub_value)
                    if entry:
                        normalized.append(entry)
                continue
            normalized_value = self._normalize_field_value(value)
            if normalized_value is not None:
                normalized.append(normalized_value)
        return normalized

    def _sanitize_field_spec(self, field_spec: Optional[Any], column_names: List[str]) -> Optional[Any]:
        if field_spec is None:
            return None
        if isinstance(field_spec, list):
            sanitized = [
                spec for spec in field_spec if self._field_exists_in_columns(spec, column_names)
            ]
            return sanitized if sanitized else None
        return field_spec if self._field_exists_in_columns(field_spec, column_names) else None

    def _extract_text_values(self, row: Dict[str, Any], field_spec: Optional[Any]) -> List[str]:
        if field_spec is None:
            return []
        if isinstance(field_spec, list):
            pieces: List[str] = []
            for spec in field_spec:
                values = self._extract_field_values(row, spec)
                if values:
                    pieces.extend(values)
            combined = "\n".join(v for v in pieces if v)
            return [combined] if combined else []
        return self._extract_field_values(row, field_spec)

    # ============================================================================
    # 任务4-10: 中间格式构建相关方法
    # ============================================================================
    
    def _generate_record_id(self, content: Any, file_path: str = "", record_index: int = 0) -> str:
        """
        任务4: 生成全局唯一记录ID
        
        Args:
            content: 记录内容（用于生成hash）
            file_path: 文件路径（用于生成唯一ID）
            record_index: 记录索引
            
        Returns:
            唯一ID字符串（32位hex）
        """
        # 构建ID的组成部分
        parts = []
        if file_path:
            parts.append(str(file_path))
        if record_index is not None:
            parts.append(str(record_index))
        
        # 将content转换为字符串用于hash
        if content is not None:
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, ensure_ascii=False, sort_keys=True)
            else:
                content_str = str(content)
            parts.append(content_str)
        
        # 生成hash
        combined = "|".join(parts)
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()[:32]
    
    def _extract_field_value_raw(self, row: Dict[str, Any], field_spec: Optional[Any]) -> Any:
        """
        任务8: 扩展字段提取方法，支持提取非字符串类型的原始值
        
        Args:
            row: 数据行
            field_spec: 字段规范（字符串路径或数组路径）
            
        Returns:
            原始值（保持类型）或None
        """
        if field_spec is None:
            return None
        
        # 如果是数组路径，拼接后返回字符串
        if isinstance(field_spec, list):
            pieces: List[str] = []
            for spec in field_spec:
                values = self._extract_field_values(row, spec)
                if values:
                    pieces.extend(values)
            combined = "\n".join(v for v in pieces if v)
            return combined if combined else None
        
        # 单个字段路径
        if not isinstance(field_spec, str):
            return None
        
        field_spec = field_spec.strip()
        if not field_spec:
            return None
        
        tokens = field_spec.split(".")
        raw_values = self._traverse_field_tokens(row, tokens)
        
        if not raw_values:
            return None
        
        # 返回第一个原始值（不转换为字符串）
        value = raw_values[0]
        if isinstance(value, tuple) and len(value) == 2:
            # 如果是(key, value)元组，返回value
            return value[1]
        return value
    
    def _extract_meta_fields(self, row: Dict[str, Any], meta_mapping: Optional[Dict[str, Any]], 
                             file_path: str, state: DataCollectionState) -> Dict[str, Any]:
        """
        任务5: 提取meta字段
        
        Args:
            row: 数据行
            meta_mapping: LLM返回的meta字段映射
            file_path: 文件路径（用于推断默认值）
            state: 状态对象
            
        Returns:
            meta字典
        """
        meta = {}
        
        if not meta_mapping:
            # 如果没有映射，使用默认值
            meta["source"] = self._infer_source_from_path(file_path)
            meta["language"] = state.request.language if hasattr(state.request, 'language') else None
            return meta
        
        # 提取source（任务22: 支持多字段拼接）
        source_spec = meta_mapping.get("source")
        if source_spec:
            if isinstance(source_spec, str) and not self._is_field_path(source_spec):
                # 直接字符串值
                meta["source"] = source_spec
            elif isinstance(source_spec, list):
                # 多字段拼接
                pieces: List[str] = []
                for spec in source_spec:
                    value = self._extract_field_value_raw(row, spec)
                    if value is not None:
                        pieces.append(str(value) if not isinstance(value, str) else value)
                meta["source"] = "_".join(pieces) if pieces else self._infer_source_from_path(file_path)
            else:
                # 单个字段路径或可能是直接值
                if isinstance(source_spec, str):
                    # 先尝试作为字段路径提取
                    source_value = self._extract_field_value_raw(row, source_spec)
                    if source_value is not None:
                        meta["source"] = str(source_value) if not isinstance(source_value, str) else source_value
                    else:
                        # 提取失败，检查是否是直接值
                        # 如果包含点号或方括号，是真正的字段路径但字段不存在，使用默认值
                        # 如果不包含点号和方括号，可能是直接值，使用原始值
                        if "." in source_spec or "[" in source_spec:
                            # 真正的字段路径但字段不存在，使用默认值
                            meta["source"] = self._infer_source_from_path(file_path)
                        else:
                            # 不包含点号和方括号，可能是直接值，使用原始值
                            meta["source"] = source_spec
                else:
                    # 非字符串类型，尝试提取
                    source_value = self._extract_field_value_raw(row, source_spec)
                    if source_value is not None:
                        meta["source"] = str(source_value) if not isinstance(source_value, str) else source_value
                    else:
                        meta["source"] = self._infer_source_from_path(file_path)
        else:
            meta["source"] = self._infer_source_from_path(file_path)
        
        # 提取language（任务22: 支持多字段拼接）
        language_spec = meta_mapping.get("language")
        if language_spec:
            if isinstance(language_spec, str) and not self._is_field_path(language_spec):
                # 直接字符串值
                meta["language"] = language_spec
            elif isinstance(language_spec, list):
                # 多字段拼接（较少见，但支持）
                pieces: List[str] = []
                for spec in language_spec:
                    value = self._extract_field_value_raw(row, spec)
                    if value is not None:
                        pieces.append(str(value) if not isinstance(value, str) else value)
                meta["language"] = "_".join(pieces) if pieces else (state.request.language if hasattr(state.request, 'language') else None)
            else:
                # 单个字段路径或可能是直接值
                if isinstance(language_spec, str):
                    # 先尝试作为字段路径提取
                    language_value = self._extract_field_value_raw(row, language_spec)
                    if language_value is not None:
                        meta["language"] = str(language_value) if not isinstance(language_value, str) else language_value
                    else:
                        # 提取失败，检查是否是直接值
                        # 如果包含点号或方括号，是真正的字段路径但字段不存在，使用默认值
                        # 如果不包含点号和方括号，可能是直接值，使用原始值
                        if "." in language_spec or "[" in language_spec:
                            # 真正的字段路径但字段不存在，使用默认值
                            meta["language"] = state.request.language if hasattr(state.request, 'language') else None
                        else:
                            # 不包含点号和方括号，可能是直接值，使用原始值
                            meta["language"] = language_spec
                else:
                    # 非字符串类型，尝试提取
                    language_value = self._extract_field_value_raw(row, language_spec)
                    if language_value is not None:
                        meta["language"] = str(language_value) if not isinstance(language_value, str) else language_value
                    else:
                        meta["language"] = state.request.language if hasattr(state.request, 'language') else None
        else:
            meta["language"] = state.request.language if hasattr(state.request, 'language') else None
        
        # 提取timestamp
        timestamp_spec = meta_mapping.get("timestamp")
        if timestamp_spec:
            timestamp_value = self._extract_field_value_raw(row, timestamp_spec)
            if timestamp_value is not None:
                meta["timestamp"] = str(timestamp_value)
        
        # 提取token_count
        token_count_spec = meta_mapping.get("token_count")
        if token_count_spec:
            token_count_value = self._extract_field_value_raw(row, token_count_spec)
            if token_count_value is not None:
                # 尝试转换为整数
                try:
                    meta["token_count"] = int(token_count_value) if not isinstance(token_count_value, int) else token_count_value
                except (ValueError, TypeError):
                    meta["token_count"] = None
        
        # 提取quality_score
        quality_score_spec = meta_mapping.get("quality_score")
        if quality_score_spec:
            quality_score_value = self._extract_field_value_raw(row, quality_score_spec)
            if quality_score_value is not None:
                # 尝试转换为浮点数
                try:
                    meta["quality_score"] = float(quality_score_value) if not isinstance(quality_score_value, float) else quality_score_value
                except (ValueError, TypeError):
                    meta["quality_score"] = None
        
        # 提取original_id（任务18: 增强默认值推断）
        original_id_spec = meta_mapping.get("original_id")
        if original_id_spec:
            original_id_value = self._extract_field_value_raw(row, original_id_spec)
            if original_id_value is not None:
                meta["original_id"] = str(original_id_value) if not isinstance(original_id_value, str) else original_id_value
            else:
                # 尝试从row中获取id字段
                if "id" in row:
                    meta["original_id"] = str(row["id"])
                # 如果还是没有，可以使用record_index（但需要从外部传入，这里暂时不处理）
        # 如果没有指定original_id_spec，尝试从row中获取id字段作为默认值
        elif "original_id" not in meta and "id" in row:
            meta["original_id"] = str(row["id"])
        
        # 任务18: 增强timestamp默认值（如果未指定且文件存在，使用文件修改时间）
        if "timestamp" not in meta and file_path and os.path.exists(file_path):
            try:
                import time
                mtime = os.path.getmtime(file_path)
                meta["timestamp"] = str(int(mtime))
            except Exception:
                pass  # 忽略错误，保持timestamp为None
        
        return meta
    
    def _extract_system_field(self, row: Dict[str, Any], system_mapping: Optional[Any]) -> Optional[str]:
        """
        任务6: 提取system字段
        
        Args:
            row: 数据行
            system_mapping: LLM返回的system字段映射（字段路径或直接字符串值）
            
        Returns:
            system字符串或None
        """
        if system_mapping is None:
            return None
        
        # 如果是直接字符串值（不是字段路径）
        if isinstance(system_mapping, str) and not self._is_field_path(system_mapping):
            return system_mapping
        
        # 字段路径
        system_value = self._extract_field_value_raw(row, system_mapping)
        if system_value is not None:
            return str(system_value) if not isinstance(system_value, str) else system_value
        
        return None
    
    def _extract_messages_structure(self, row: Dict[str, Any], messages_mapping: Optional[List[Dict[str, Any]]], 
                                    column_names: List[str]) -> List[Dict[str, Any]]:
        """
        任务7: 提取messages结构
        
        Args:
            row: 数据行
            messages_mapping: LLM返回的messages映射列表
            column_names: 列名列表（用于验证字段存在性）
            
        Returns:
            messages列表，每个元素包含role, content, loss_mask
        """
        if not messages_mapping:
            return []
        
        messages = []
        valid_roles = {"user", "assistant", "system", "tool"}
        
        for idx, msg_mapping in enumerate(messages_mapping):
            # 检查 msg_mapping 是否为字典类型
            if not isinstance(msg_mapping, dict):
                log.warning(f"消息 {idx} 不是字典类型（类型: {type(msg_mapping).__name__}），跳过该消息")
                continue
            
            role = msg_mapping.get("role")
            content_spec = msg_mapping.get("content")
            loss_mask = msg_mapping.get("loss_mask")
            
            # 严格要求role必须由LLM明确指定，不进行推断
            if not role:
                log.warning(f"消息 {idx} 缺少role字段，LLM必须明确指定role，跳过该消息")
                continue
            
            # 验证role是否有效
            if role not in valid_roles:
                log.warning(f"消息 {idx} 的role '{role}' 无效，必须是 {valid_roles} 之一，跳过该消息")
                continue
            
            # 提取content（任务17: 支持嵌套对话结构）
            # 注意：完全依赖LLM的映射，不进行规则推断
            # 如果LLM返回嵌套路径（如dialogues[*].turns[*].user_input），
            # _extract_text_values会自动展开并返回所有值
            content = None
            if content_spec:
                if isinstance(content_spec, list):
                    # 多字段拼接
                    pieces: List[str] = []
                    for spec in content_spec:
                        values = self._extract_field_values(row, spec)
                        if values:
                            pieces.extend(values)
                    content = "\n".join(v for v in pieces if v) if pieces else None
                else:
                    # 单个字段路径（支持嵌套结构，如dialogues[*].turns[*].user_input）
                    # 特殊情况：处理 messages[*].content 模式，需要根据role进行匹配
                    content_path = str(content_spec) if content_spec else ""
                    if content_path and "messages[" in content_path and ".content" in content_path:
                        # 检查原始数据中 messages 列表的每个元素是否有 role 字段
                        messages_list = row.get("messages")
                        if isinstance(messages_list, list) and len(messages_list) > 0:
                            # 检查第一个元素是否有 role 字段
                            first_msg = messages_list[0] if messages_list else None
                            if isinstance(first_msg, dict) and "role" in first_msg:
                                # 这是特殊情况：messages 列表中每个元素都有 role 字段
                                # 需要根据映射消息的 role 和原始数据的 role 进行匹配
                                matched_content = self._extract_content_by_role(
                                    messages_list, role, content_spec
                                )
                                if matched_content:
                                    content = matched_content
                                else:
                                    # 如果匹配失败，回退到原来的提取方式
                                    values = self._extract_text_values(row, content_spec)
                                    content = values[0] if values else None
                            else:
                                # 不是特殊情况，使用原来的提取方式
                                values = self._extract_text_values(row, content_spec)
                                content = values[0] if values else None
                        else:
                            # messages 不存在或为空，使用原来的提取方式
                            values = self._extract_text_values(row, content_spec)
                            content = values[0] if values else None
                    else:
                        # 不是 messages[*].content 模式，使用原来的提取方式
                        values = self._extract_text_values(row, content_spec)
                        # 如果提取到多个值（嵌套结构展开），取第一个
                        # 注意：对于嵌套对话，LLM应该返回多个消息映射，每个对应一个turn
                        content = values[0] if values else None
            
            if content is None:
                log.warning(f"消息 {idx} (role={role}) 无法提取content，跳过该消息")
                continue  # 跳过没有content的消息
            
            # 处理loss_mask：如果LLM未指定，使用默认值（assistant为true，其他为false）
            if loss_mask is None:
                loss_mask = (role == "assistant")
            
            messages.append({
                "role": role,
                "content": content,
                "loss_mask": loss_mask
            })
        
        return messages
    
    def _extract_content_by_role(self, messages_list: List[Dict[str, Any]], target_role: str, content_spec: str) -> Optional[str]:
        """
        从 messages 列表中根据 role 匹配提取 content
        
        Args:
            messages_list: messages 列表，每个元素包含 role 和 content 字段
            target_role: 目标 role（"user", "assistant", "system", "tool"）
            content_spec: 字段路径（如 "messages[*].content"）
            
        Returns:
            匹配的 content 字符串，如果未找到则返回 None
        """
        # role 映射：将标准 role 映射到可能的数据中的 role 值
        role_mapping = {
            "user": ["HUMAN", "human", "USER", "user", "User", "Human"],
            "assistant": ["ASSISTANT", "assistant", "Assistant", "GPT", "gpt", "AI", "ai", "Ai"],
            "system": ["SYSTEM", "system", "System"],
            "tool": ["TOOL", "tool", "Tool", "FUNCTION", "function", "Function"]
        }
        
        # 获取目标 role 可能的值
        possible_roles = role_mapping.get(target_role, [target_role])
        
        # 遍历 messages 列表，找到匹配 role 的元素
        matched_contents = []
        for msg in messages_list:
            if not isinstance(msg, dict):
                continue
            
            msg_role = msg.get("role")
            if msg_role and any(msg_role.upper() == r.upper() for r in possible_roles):
                # 找到匹配的 role，提取 content
                msg_content = msg.get("content")
                if msg_content is not None:
                    # 规范化 content 为字符串
                    if isinstance(msg_content, str):
                        matched_contents.append(msg_content)
                    else:
                        matched_contents.append(str(msg_content))
        
        # 如果有多个匹配的内容，用换行符连接（多轮对话场景）
        if matched_contents:
            return "\n".join(matched_contents)
        
        return None
    
    def _is_field_path(self, value: str) -> bool:
        """
        判断字符串是否是字段路径（包含点号或方括号）还是直接值
        
        Args:
            value: 字符串值
            
        Returns:
            True如果是字段路径，False如果是直接值
        """
        if not isinstance(value, str):
            return False
        
        # 如果包含方括号，肯定是字段路径（数组索引）
        if "[" in value:
            return True
        
        # 如果包含点号，需要进一步判断
        # 如果包含空格，很可能是直接值（完整句子），不是字段路径
        if " " in value:
            return False
        
        # 如果包含点号且没有空格，可能是嵌套字段路径（如"meta.field"）
        if "." in value:
            # 检查点号前后的部分是否都像字段名（字母数字下划线）
            parts = value.split(".")
            if all(part.replace("_", "").replace("-", "").isalnum() for part in parts if part):
                return True
            return False
        
        # 如果不包含点号和方括号，检查是否像字段名（字母数字下划线，不包含空格）
        if value.replace("_", "").replace("-", "").isalnum():
            return True
        
        return False
    
    def _infer_source_from_path(self, file_path: str) -> str:
        """
        从文件路径推断source
        
        Args:
            file_path: 文件路径
            
        Returns:
            source字符串
        """
        if not file_path:
            return "unknown"
        
        # 获取文件名（不含扩展名）或目录名
        path_obj = Path(file_path)
        # 尝试从文件名推断
        stem = path_obj.stem
        if stem and stem != "":
            return stem
        
        # 尝试从父目录名推断
        parent = path_obj.parent.name
        if parent and parent not in (".", ""):
            return parent
        
        return "unknown"
    
    def _build_intermediate_format_pt(self, row: Dict[str, Any], annotation_result: Dict[str, Any], 
                                     file_path: str, state: DataCollectionState, record_index: int) -> Optional[Dict[str, Any]]:
        """
        任务9: 构建PT模式的中间格式
        
        Args:
            row: 数据行
            annotation_result: LLM返回的映射结果
            file_path: 文件路径
            state: 状态对象
            record_index: 记录索引
            
        Returns:
            中间格式字典或None（如果数据无效）
        """
        # 任务19: 数据验证
        text_field_spec = annotation_result.get("text")
        if not text_field_spec:
            log.warning(f"PT模式：未找到text字段映射，跳过记录")
            return None
        
        text_values = self._extract_text_values(row, text_field_spec)
        if not text_values:
            log.warning(f"PT模式：无法从记录中提取text值，跳过")
            return None
        
        # 构建结果（可能有多条记录，如果text_values有多个值）
        results = []
        for text in text_values:
            if not text:
                continue
            
            # 生成ID
            record_id = self._generate_record_id(text, file_path, record_index)
            
            # 提取meta字段
            meta_mapping = annotation_result.get("meta")
            meta = self._extract_meta_fields(row, meta_mapping, file_path, state)
            
            # 构建中间格式
            intermediate_record = {
                "id": record_id,
                "dataset_type": "pretrain",
                "text": text,
                "meta": meta
            }
            
            # 任务19: 验证必填字段
            if not intermediate_record.get("id"):
                log.warning(f"PT模式：记录缺少id字段，跳过")
                continue
            if not intermediate_record.get("text"):
                log.warning(f"PT模式：记录缺少text字段，跳过")
                continue
            if intermediate_record.get("dataset_type") != "pretrain":
                log.warning(f"PT模式：dataset_type不正确，跳过")
                continue
            
            results.append(intermediate_record)
        
        # 如果只有一条记录，直接返回；否则返回第一条（后续可能需要处理多条记录的情况）
        return results[0] if results else None
    
    def _build_intermediate_format_sft(self, row: Dict[str, Any], annotation_result: Dict[str, Any], 
                                      file_path: str, state: DataCollectionState, record_index: int) -> Optional[Dict[str, Any]]:
        """
        任务10: 构建SFT模式的中间格式
        
        Args:
            row: 数据行
            annotation_result: LLM返回的映射结果
            file_path: 文件路径
            state: 状态对象
            record_index: 记录索引
            
        Returns:
            中间格式字典或None（如果数据无效）
        """
        # 提取messages
        messages_mapping = annotation_result.get("messages")
        if not messages_mapping:
            # log.warning(f"SFT模式：未找到messages字段映射，跳过记录")
            return None
        
        column_names = list(row.keys()) if isinstance(row, dict) else []
        messages = self._extract_messages_structure(row, messages_mapping, column_names)
        
        if not messages:
            log.warning(f"SFT模式：无法从记录中提取messages，跳过")
            return None
        
        # 任务19: 验证messages结构
        valid_roles = {"user", "assistant", "system", "tool"}
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                log.warning(f"SFT模式：消息 {idx} 不是字典类型，跳过记录")
                return None
            if "role" not in msg or msg["role"] not in valid_roles:
                log.warning(f"SFT模式：消息 {idx} 的role无效，跳过记录")
                return None
            if "content" not in msg or not msg["content"]:
                log.warning(f"SFT模式：消息 {idx} 缺少content或content为空，跳过记录")
                return None
        
        # 提取system字段
        system_mapping = annotation_result.get("system")
        system = self._extract_system_field(row, system_mapping)
        
        # 提取meta字段
        meta_mapping = annotation_result.get("meta")
        meta = self._extract_meta_fields(row, meta_mapping, file_path, state)
        
        # 生成ID（基于messages内容）
        content_for_id = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        record_id = self._generate_record_id(content_for_id, file_path, record_index)
        
        # 构建中间格式
        intermediate_record = {
            "id": record_id,
            "dataset_type": "sft",
            "messages": messages
        }
        
        if system:
            intermediate_record["system"] = system
        
        intermediate_record["meta"] = meta
        
        # 任务19: 验证必填字段
        if not intermediate_record.get("id"):
            log.warning(f"SFT模式：记录缺少id字段，跳过")
            return None
        if not intermediate_record.get("messages"):
            log.warning(f"SFT模式：记录缺少messages字段，跳过")
            return None
        if intermediate_record.get("dataset_type") != "sft":
            log.warning(f"SFT模式：dataset_type不正确，跳过")
            return None
        
        return intermediate_record

    def build_messages(self, state: DataCollectionState, column_names: List[str], sample_record: Dict[str, Any], dataset: Any = None) -> List[BaseMessage]:
        """(数据映射) 构建消息列表"""
        log.info("构建(数据映射)提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        # 如果提供了 dataset，使用采样功能；否则使用单条记录
        if dataset is not None:
            sampled_records = self._sample_records(dataset)
            # 对于模板，我们仍然传 first_row，但现在是采样列表的第一个
            # 格式化额外的示例记录用于提示词
            sample_rows_info = ""
            if len(sampled_records) > 1:
                import json
                additional_samples = sampled_records[1:min(3, len(sampled_records))]
                sample_rows_info = "\nAdditional Sample Records:\n"
                for idx, row in enumerate(additional_samples, start=2):
                    sample_rows_info += f"- Record {idx}: {json.dumps(row, ensure_ascii=False)}\n"
            
            task_params = {
                'column_names': column_names, 
                'first_row': sampled_records[0] if sampled_records else sample_record,
                'sample_rows_info': sample_rows_info,
                'user_target': state.request.target  # 添加用户需求
            }
        else:
            # 如果没有提供 dataset，截断单条记录
            truncated_record = {k: self._truncate_value(v) for k, v in sample_record.items()}
            task_params = {
                'column_names': column_names, 
                'first_row': truncated_record,
                'sample_rows_info': "",  # 没有额外示例
                'user_target': state.request.target  # 添加用户需求
            }
        
        task_prompt = ptg.render(eval(f"self.task_prompt_template_name_{state.request.category.lower()}"), **task_params)
        
        log.info(f"系统提示词: {sys_prompt}")
        log.debug(f"任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def create_llm(self, state: DataCollectionState) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return llm
    
    async def invoke(self, state: DataCollectionState, column_names: List[str], sample_record: Dict[str, Any], dataset: Any = None) -> Dict[str, Any]:
        """(数据映射) 调用LLM并处理响应"""
        log.info(f"{self.role_name} 调用LLM(数据映射)并处理响应...")
        
        # 记录输入
        inputs = {
            "role_name": self.role_name,
            "column_names": column_names,
            "sample_record_keys": list(sample_record.keys()) if sample_record else [],
            "dataset_size": len(dataset) if dataset is not None else None,
            "category": state.request.category
        }
        log.info(f"[Agent Input] {self.role_name}.invoke: {json.dumps(inputs, indent=2, ensure_ascii=False)}")
        
        messages = self.build_messages(state, column_names, sample_record, dataset)
        llm = self.create_llm(state)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            log.info(f'LLM(数据映射)调用成功并返回结果: {answer_text}')

            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, answer_text).group(1).strip() if re.search(pattern, answer_text) else answer_text

            try:
                annotation_result = json.loads(match)
                
                # 任务11: 验证映射结果
                category = state.request.category.upper()
                is_valid = validate_mapping(annotation_result, category)
                
                if not is_valid:
                    log.warning(f"LLM返回的映射结果验证失败 (category={category})")
                    log.warning(f"映射结果: {json.dumps(annotation_result, indent=2, ensure_ascii=False)}")
                    
                    # 根据category给出具体的验证错误信息
                    if category == "PT":
                        if "text" not in annotation_result or annotation_result.get("text") is None:
                            log.warning("PT模式：缺少text字段或text为null")
                        if "meta" in annotation_result and annotation_result["meta"]:
                            if "source" not in annotation_result["meta"] or annotation_result["meta"].get("source") is None:
                                log.warning("PT模式：meta存在但缺少source字段或source为null")
                    elif category == "SFT":
                        if "messages" not in annotation_result or not annotation_result.get("messages"):
                            log.warning("SFT模式：缺少messages字段或messages为空")
                        else:
                            messages = annotation_result.get("messages", [])
                            for idx, msg in enumerate(messages):
                                if not isinstance(msg, dict):
                                    log.warning(f"SFT模式：消息 {idx} 不是字典类型")
                                else:
                                    if "role" not in msg:
                                        log.warning(f"SFT模式：消息 {idx} 缺少role字段")
                                    elif msg.get("role") not in {"user", "assistant", "system", "tool"}:
                                        log.warning(f"SFT模式：消息 {idx} 的role '{msg.get('role')}' 无效")
                                    if "content" not in msg:
                                        log.warning(f"SFT模式：消息 {idx} 缺少content字段")
                            
                            # 检查是否有不同role映射到相同字段的问题
                            role_content_pairs = []
                            for idx, msg in enumerate(messages):
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    role = msg.get("role")
                                    content = msg.get("content")
                                    # 规范化content为字符串
                                    if isinstance(content, list):
                                        content_str = str(sorted(content))
                                    else:
                                        content_str = str(content) if content is not None else ""
                                    role_content_pairs.append((idx, role, content_str))
                            
                            # 检查不同role是否映射到相同字段
                            for i, (idx1, role1, content1) in enumerate(role_content_pairs):
                                for j, (idx2, role2, content2) in enumerate(role_content_pairs):
                                    if i != j and role1 != role2 and content1 == content2:
                                        # 检查是否是特殊情况：messages[*].content 模式
                                        content_path = content1 if isinstance(content1, str) else str(content1)
                                        if content_path and ("messages[" in content_path and ".content" in content_path):
                                            # 这是特殊情况，允许不同role映射到相同路径
                                            log.debug(f"SFT模式：消息 {idx1} (role={role1}) 和消息 {idx2} (role={role2}) 映射到了相同的字段 '{content1}'，这是允许的（messages列表带role字段的特殊情况）")
                                        else:
                                            log.warning(f"SFT模式：消息 {idx1} (role={role1}) 和消息 {idx2} (role={role2}) 映射到了相同的字段 '{content1}'，不同role必须映射到不同的字段")
                    
                    # 不抛出异常，允许继续处理（可能部分字段可用）
                    log.warning("继续处理，但可能产生不完整的数据")
                
                # 任务21: 更新日志输出，展示完整的映射结构
                log.info(f"[Agent Output] {self.role_name}.invoke: {json.dumps(annotation_result, indent=2, ensure_ascii=False)}")
                if category == "PT":
                    log.debug(f"PT模式映射 - text: {annotation_result.get('text')}, meta: {annotation_result.get('meta')}")
                elif category == "SFT":
                    messages_count = len(annotation_result.get('messages', [])) if annotation_result.get('messages') else 0
                    log.debug(f"SFT模式映射 - messages数量: {messages_count}, system: {annotation_result.get('system')}, meta: {annotation_result.get('meta')}")
                
                return annotation_result
            except json.JSONDecodeError as e:
                log.exception(f"解析GPT(数据映射)响应为JSON失败: 内容为{match}")
                raise ValueError(f"Failed to parse GPT response as JSON: {e}")
            
        except Exception as e:
            log.exception("hf数据集(数据映射)标注失败: %s", e)
            log.error(f"[Agent Output] {self.role_name}.invoke: Error - {str(e)}")
            raise Exception(f"Error during dataset annotation: {e}")

    
    def record_summary(self, state, output_dir=None):
        """(通用) 汇总报告"""
        info = ""

        if state.request.category not in ['PT', 'SFT']:
            info += f"Unsupported data category '{state.request.category}'. Only 'PT' or 'SFT' are supported.\n"
            return info

        if not state.keywords:
            info += "Sorry, I couldn't extract any valid keywords from your request."
            return info
        info += f"Extracted keywords: {', '.join(state.keywords)}\n\n"

        # 检查 'datasets'（用于旧版）或 'downloads'（用于新版）
        if not state.datasets and not state.downloads:
             info += "No datasets were found matching your keywords."
             return info
        
        # --- 原始逻辑 (适用于旧的 execute) ---
        if state.datasets and all(len(lst) == 0 for lst in state.datasets.values()):
            info += "No datasets were found matching your keywords."
            return info
        
        if state.downloads:
            info += "Datasets found:\n"
            for keyword, dataset_infos in state.datasets.items():
                if not dataset_infos:
                    info += f"- No datasets found for keyword: {keyword}\n"
                    continue
                # 确保 downloads 字典中有这个 key
                if keyword not in state.downloads:
                    info += f"- Keyword {keyword} found but no download info available.\n"
                    continue
                
                download_infos = state.downloads[keyword]
                info += f"- {len(download_infos)} datasets found for keyword: {keyword}\n"
                for download_info in download_infos:
                    status = "Download succeeded" if download_info['success'] else "Download failed"
                    info += f"  - {download_info['dataset_id']}: {status}\n"
            info += "\n"
        # --- 原始逻辑结束 ---

        info += "Post-processing summary:\n"
        info += "Data category: " + state.request.category + "\n"
        category_key = state.request.category.upper() # 'PT' or 'SFT'

        for keyword, sources in state.sources.items():
            info += f"- Keyword: {keyword}\n"
            
            # 兼容新旧两种 sources 结构
            source_list = []
            if category_key in sources:
                source_list = sources[category_key] # 新结构: {'PT': [...]}
            elif state.request.category in sources:
                source_list = sources[state.request.category] # 旧结构: {'pt': [...]}
            
            if not source_list:
                info += "-- No datasets were successfully post-processed.\n"
                continue

            info += f"-- Total count: \t{sum(item[1] for item in source_list)}\n"
            info += "-- Source details:\n"
            for dataset_id, record_count in source_list:
                info += f"{dataset_id}: {record_count} records\t"
            info += "\n"
        
        info = info.strip()
        log.info("处理结果汇总:\n" + info)
        
        # 确定输出目录
        if output_dir is None:
            output_dir = state.request.download_dir
        
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(info)
        
        log.info(f"摘要报告已保存: {os.path.abspath(summary_path)}")

    async def execute(self, state: DataCollectionState, **kwargs) -> DataCollectionState:
        """
        (原始 execute)
        执行入口 - 原始版本，依赖于 'HuggingFaceDatasetManager' 的下载结构。
        它处理已下载并整理到 'tmp' 目录的 HF 数据集。
        """
        log.info(f"{self.role_name} (Base) 开始执行...")

        _ensure_hf_cache_env(state.request.download_dir)
        from datasets import load_from_disk

        # 记录输入
        inputs = {
            "role_name": self.role_name,
            "category": state.request.category,
            "keywords": state.keywords,
            "downloads_count": sum(len(v) for v in state.downloads.values()) if state.downloads else 0,
            "download_dir": state.request.download_dir
        }
        log.info(f"[Agent Input] {self.role_name}.execute: {json.dumps(inputs, indent=2, ensure_ascii=False)}")

        category = state.request.category.upper() # PT/SFT

        # Step 1: Convert datasets
        for keyword in state.keywords:
            if keyword not in state.downloads.keys() or not Counter([res['success'] for res in state.downloads[keyword]])[True]:
                state.sources[keyword] = {category: []}
                continue
            
            data_sources = {category: []}

            data_dir = os.path.join(state.request.download_dir, keyword.replace(" ", "_"))
            for dataset in state.downloads[keyword]:
                if not dataset['success']:
                    continue
                
                dataset_id = dataset['dataset_id']
                # 原始逻辑：从 'tmp' 目录加载
                dataset_path = os.path.join(data_dir, 'tmp', dataset_id.replace("/", "_"))
                
                if not os.path.exists(dataset_path):
                    log.warning(f"数据集 {dataset_path} 不存在，跳过。")
                    continue
                    
                try:
                    # 使用 load_from_disk 因为 'tmp' 目录是 save_to_disk 的产物
                    data = load_from_disk(dataset_path) 
                    
                    # 确保 data 是一个 dict (DatasetDict)
                    if not isinstance(data, dict):
                        # 如果是单个 Dataset，包装成 dict
                        data = {"train": data}

                    for split, data_content in data.items():
                        if len(data_content) == 0:
                            log.info(f"Split '{split}' (from {dataset_id}) 为空，跳过。")
                            continue
                            
                        # 调用 LLM (数据映射)，传入完整 dataset 用于采样
                        annotation_result = await self.invoke(state, data_content.column_names, data_content[0], dataset=data_content)

                        if category == 'PT':
                            # 使用新的中间格式构建器
                            data_file = os.path.join(data_dir, 'PT.jsonl')
                            count = 0
                            file_path = os.path.join(data_dir, 'tmp', dataset_id.replace("/", "_"))
                            
                            with open(data_file, 'a', encoding='utf-8') as f:
                                for record_index, row in enumerate(data_content):
                                    intermediate_record = self._build_intermediate_format_pt(
                                        row, annotation_result, file_path, state, record_index
                                    )
                                    if intermediate_record:
                                        f.write(json.dumps(intermediate_record, ensure_ascii=False) + '\n')
                                        count += 1
                            
                            data_sources['PT'].append((f'{dataset_id}_({split})', count))
                            log.info(f"从数据集 {dataset_id}, split {split} 中提取了 {count} 条 PT 样本（中间格式：包含id, dataset_type, text, meta）。")

                        elif category == 'SFT':
                            # 使用新的中间格式构建器
                            data_file = os.path.join(data_dir, 'SFT.jsonl')
                            count = 0
                            file_path = os.path.join(data_dir, 'tmp', dataset_id.replace("/", "_"))
                            
                            with open(data_file, 'a', encoding='utf-8') as f:
                                for record_index, row in enumerate(data_content):
                                    intermediate_record = self._build_intermediate_format_sft(
                                        row, annotation_result, file_path, state, record_index
                                    )
                                    if intermediate_record:
                                        f.write(json.dumps(intermediate_record, ensure_ascii=False) + '\n')
                                        count += 1
                            
                            data_sources['SFT'].append((f'{dataset_id}_({split})', count))
                            log.info(f"从数据集 {dataset_id}, split {split} 中提取了 {count} 条 SFT 样本（中间格式：包含id, dataset_type, messages, system?, meta）。")
                            
                except Exception as e:
                    log.error(f"处理数据集 {dataset_id} 时出错: {e}, 跳过该数据集")
                    continue

            state.sources[keyword] = data_sources
        
        # Step 2: Record summary
        self.record_summary(state)

        # 记录输出
        outputs = {
            "sources": {k: {cat: len(v) for cat, v in sources.items()} for k, sources in state.sources.items()},
            "total_keywords_processed": len(state.keywords)
        }
        log.info(f"[Agent Output] {self.role_name}.execute: {json.dumps(outputs, indent=2, ensure_ascii=False)}")
        log.info(f"{self.role_name} (Base) 执行完成")
        return state

class UniversalDataConvertor(DataConvertor):
    """
    通用数据转换器。
    继承自 DataConvertor，重写 execute 逻辑。
    它会扫描文件夹，用 LLM 找出数据文件，然后用 LLM 映射数据格式。
    """

    @property
    def role_name(self) -> str:
        return "universal_data_convertor"

    def __init__(self, *args, max_concurrent_discovery: int = 7, max_concurrent_mapping: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        # 临时目录列表，用于清理
        self._temp_dirs = []
        # 并发控制参数
        self.max_concurrent_discovery = max_concurrent_discovery  # 文件发现并发数
        self.max_concurrent_mapping = max_concurrent_mapping  # 文件映射并发数
        # 文件写入锁，保证并发写入时的线程安全
        self._write_lock = asyncio.Lock()

    def _is_compressed_file(self, file_path: str) -> bool:
        """
        判断文件是否为压缩文件。
        """
        compressed_extensions = [
            '.zip', '.tar', '.tar.gz', '.tgz', 
            '.tar.bz2', '.tbz2', '.tar.xz', '.txz',
            '.gz', '.bz2', '.xz', '.7z', '.rar'
        ]
        path_lower = file_path.lower()
        return any(path_lower.endswith(ext) for ext in compressed_extensions)

    def _extract_compressed_file(self, compressed_path: str) -> Optional[str]:
        """
        解压缩文件到临时目录。
        
        Args:
            compressed_path: 压缩文件的完整路径
            
        Returns:
            解压后的目录路径，如果解压失败则返回 None
        """
        if not os.path.exists(compressed_path):
            log.error(f"压缩文件不存在: {compressed_path}")
            return None
            
        # 创建可控位置的临时目录，优先使用 DF_TEMP_DIR 或下载目录下的 .tmp
        temp_base_dir = os.getenv("DF_TEMP_DIR") or None
        if temp_base_dir is None:
            # 尝试从压缩文件所在目录的上级派生一个 .tmp
            parent_dir = os.path.dirname(os.path.abspath(compressed_path))
            tmp_candidate = os.path.join(parent_dir, ".tmp")
            try:
                os.makedirs(tmp_candidate, exist_ok=True)
                temp_base_dir = tmp_candidate
            except Exception:
                temp_base_dir = None
        temp_dir = tempfile.mkdtemp(prefix="dataflow_extract_", dir=temp_base_dir)
        self._temp_dirs.append(temp_dir)
        log.info(f"正在解压 {compressed_path} 到 {temp_dir}")
        
        try:
            path_lower = compressed_path.lower()
            
            # ZIP 文件
            if path_lower.endswith('.zip'):
                with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                log.info(f"成功解压 ZIP 文件")
                return temp_dir
            
            # TAR 文件（包括 .tar.gz, .tar.bz2, .tar.xz 等）
            elif '.tar' in path_lower or path_lower.endswith(('.tgz', '.tbz2', '.txz')):
                with tarfile.open(compressed_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_dir)
                log.info(f"成功解压 TAR 文件")
                return temp_dir
            
            # GZIP 文件（单文件压缩）
            elif path_lower.endswith('.gz') and not '.tar' in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                log.info(f"成功解压 GZIP 文件")
                return temp_dir
            
            # BZIP2 文件（单文件压缩）
            elif path_lower.endswith('.bz2') and not '.tar' in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with bz2.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                log.info(f"成功解压 BZIP2 文件")
                return temp_dir
            
            # XZ/LZMA 文件（单文件压缩）
            elif path_lower.endswith('.xz') and not '.tar' in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with lzma.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                log.info(f"成功解压 XZ 文件")
                return temp_dir
            
            else:
                log.warning(f"不支持的压缩格式: {compressed_path}")
                return None
                
        except Exception as e:
            log.error(f"解压文件失败 {compressed_path}: {e}")
            return None

    def _cleanup_temp_dirs(self):
        """
        清理所有临时目录。
        """
        for temp_dir in self._temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    log.info(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                log.warning(f"清理临时目录失败 {temp_dir}: {e}")
        self._temp_dirs.clear()
    
    def _cleanup_download_dir_cache_files(self, download_dir: str):
        """
        清理下载目录中的残留缓存文件（如 .conda 文件等）。
        
        Args:
            download_dir: 下载目录路径
        """
        if not os.path.exists(download_dir):
            return
        
        cleaned_count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(download_dir):
            # 跳过临时目录和输出目录
            dirs[:] = [d for d in dirs if d not in ('.tmp', 'processed_output', '.cache', 'rag_db', 'web_get')]
            
            for f in files:
                # 清理 conda 包缓存文件
                if f.endswith('.conda'):
                    file_path = os.path.join(root, f)
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_count += 1
                        total_size += file_size
                        log.info(f"已清理 conda 缓存文件: {file_path} ({file_size / (1024*1024):.2f} MB)")
                    except Exception as e:
                        log.warning(f"清理 conda 缓存文件失败 {file_path}: {e}")
        
        if cleaned_count > 0:
            log.info(f"共清理 {cleaned_count} 个 conda 缓存文件，释放空间 {total_size / (1024*1024):.2f} MB")

    @staticmethod
    def _flatten_column_name(column: Any) -> str:
        if isinstance(column, tuple):
            return "__".join(str(part) for part in column if part not in (None, ""))
        return str(column)

    def _normalize_value(self, value: Any) -> Any:
        import pandas as pd
        import numpy as np

        if value is None:
            return None

        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._normalize_value(v) for v in list(value)]

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, pd.Timedelta):
            return value.isoformat()

        if isinstance(value, pd.Period):
            return value.to_timestamp().isoformat()

        if isinstance(value, np.ndarray):
            return [self._normalize_value(v) for v in value.tolist()]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            if np.isfinite(value):
                return float(value)
            return None

        if isinstance(value, (np.bool_,)):
            return bool(value)

        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return self._normalize_value(value.item())
            except Exception:
                pass

        return value

    def _dataframe_to_simple_dataset(self, df: "pd.DataFrame") -> Optional[Dict[str, SimpleDataset]]:
        import pandas as pd

        if df is None or len(df) == 0:
            log.warning("pandas DataFrame 为空，无法构建 SimpleDataset")
            return None

        df = df.copy()
        df.columns = [self._flatten_column_name(col) for col in df.columns]

        records = df.to_dict(orient="records")
        normalized_records: List[Dict[str, Any]] = []
        for record in records:
            normalized_record = {str(k): self._normalize_value(v) for k, v in record.items()}
            normalized_records.append(normalized_record)

        dataset = _build_simple_dataset(normalized_records)
        if dataset:
            sample_columns = dataset["train"].column_names[:5]
            log.info(
                "使用 pandas 构建 SimpleDataset 成功，共 %d 条记录，示例列: %s",
                len(dataset["train"]),
                sample_columns,
            )
        return dataset

    def _get_file_list_string(self, root_path: str, exclude_files: List[str] = None) -> str:
        """
        遍历目录，生成所有文件的相对路径列表字符串。
        
        Args:
            root_path: 根目录路径
            exclude_files: 需要排除的文件名列表（如输出文件）
        """
        if exclude_files is None:
            exclude_files = []
        
        file_list = []
        for root, dirs, files in os.walk(root_path, topdown=True):
            # 忽略临时/缓存/输出目录，避免把临时文件当作数据处理
            dirs[:] = [
                d for d in dirs
                if not d.startswith(('.', '__'))
                and d not in ('.cache', 'processed_output', '.tmp', 'rag_db', 'web_get')
                and not d.startswith(('datasets_cache_', 'dataflow_extract_', 'hf_cache_', 'kaggle_cache_'))
            ]
            files = [f for f in files if not f.startswith(('.', '__'))]
            
            for f in files:
                # 忽略排除列表中的文件
                if f in exclude_files:
                    continue
                # 忽略 conda 包缓存文件
                if f.endswith('.conda'):
                    continue
                full_path = os.path.join(root, f)
                relative_path = os.path.relpath(full_path, root_path)
                file_list.append(relative_path.replace(os.sep, '/'))
        
        if not file_list:
            return "This directory is empty."
        
        return "File list:\n" + "\n".join(sorted(file_list))

    def _chunk_file_list_for_llm(
        self,
        file_list_str: str,
        max_chars: int = 8000,
        max_lines: int = 200,
    ) -> List[str]:
        """
        将文件列表字符串按块拆分，防止单次请求超出上下文长度限制。
        """
        if not file_list_str:
            return []

        lines = file_list_str.splitlines()
        if not lines:
            return []

        header = None
        if lines[0].strip().lower().startswith("file list"):
            header = lines[0]
            content_lines = lines[1:]
        else:
            content_lines = lines

        if not content_lines:
            return [file_list_str]

        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_char_len = 0

        for line in content_lines:
            line_len = len(line) + 1  # 估算换行符
            if current_chunk and (
                current_char_len + line_len > max_chars
                or len(current_chunk) >= max_lines
            ):
                chunks.append(current_chunk)
                current_chunk = [line]
                current_char_len = line_len
            else:
                current_chunk.append(line)
                current_char_len += line_len

        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) <= 1:
            return [file_list_str]

        total_chunks = len(chunks)
        result_chunks: List[str] = []
        for idx, chunk_lines in enumerate(chunks, start=1):
            if header:
                chunk_header = f"{header} (chunk {idx}/{total_chunks})"
            else:
                chunk_header = f"File list (chunk {idx}/{total_chunks})"
            chunk_str = chunk_header + "\n" + "\n".join(chunk_lines)
            result_chunks.append(chunk_str)

        return result_chunks

    def _get_builder_type(self, file_path: str) -> Optional[str]:
        """
        根据文件路径猜测 'load_dataset' 需要的 builder_type。
        'load_dataset' 会自动处理压缩。
        """
        path_lower = file_path.lower()
        if '.jsonl' in path_lower or '.json' in path_lower:
            return 'json'
        if '.csv' in path_lower:
            return 'csv'
        if '.parquet' in path_lower:
            return 'parquet'
        if '.arrow' in path_lower:
            return 'arrow'
        if '.txt' in path_lower or '.md' in path_lower:
            return 'text'
        
        log.warning(f"无法确定文件 '{file_path}' 的 builder type")
        return None
    
    async def _manual_load_json(self, file_path: str, max_file_size_mb: int = 5000) -> Optional[Any]:
        """
        手动加载 JSON 文件的备用方法。
        
        """
        try:
            # 检查文件大小，避免加载超大文件导致内存问题
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                log.warning(f"文件过大 ({file_size_mb:.2f} MB > {max_file_size_mb} MB)，跳过手动加载: {file_path}")
                return None
            
            log.info(f"尝试使用 pandas 手动读取 JSON 文件 ({file_size_mb:.2f} MB): {file_path}")

            import pandas as pd

            # 优先尝试按行读取（适用于 JSONL）
            try:
                df = pd.read_json(file_path, lines=True)
                if len(df) > 0:
                    log.info(f"pandas 按行读取 JSON 成功，获得 {len(df)} 条记录")
                    return self._dataframe_to_simple_dataset(df)
                log.info("pandas 按行读取 JSON 返回空 DataFrame，尝试其他策略")
            except ValueError as err:
                log.info(f"pandas 按行读取 JSON 失败: {err}")

            # 回退到手动解析
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                log.warning(f"文件为空: {file_path}")
                return None

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                log.error(f"JSON 解析失败: {e}")
                return None

            candidate_records: List[Any] = []
            if isinstance(parsed, list):
                candidate_records = parsed
            elif isinstance(parsed, dict):
                key_candidates = ['data', 'items', 'records', 'examples', 'train', 'test', 'validation', 'val']
                for key in key_candidates:
                    if key in parsed and isinstance(parsed[key], list) and parsed[key]:
                        candidate_records = parsed[key]
                        log.info(f"在 JSON 字典的 '{key}' 字段中找到 {len(parsed[key])} 条记录")
                        break
                if not candidate_records:
                    candidate_records = [parsed]
            else:
                log.warning(f"不支持的 JSON 顶层类型: {type(parsed)}，跳过 {file_path}")
                return None

            if not candidate_records:
                log.warning(f"JSON 内容未包含可用记录: {file_path}")
                return None

            normalized_records: List[Dict[str, Any]] = []
            for item in candidate_records:
                if isinstance(item, dict):
                    normalized_records.append(item)
                else:
                    normalized_records.append({"value": item})

            df = pd.json_normalize(normalized_records)
            if len(df) == 0:
                log.warning(f"pd.json_normalize 结果为空: {file_path}")
                return None

            return self._dataframe_to_simple_dataset(df)

        except Exception as e:
            log.error(f"手动加载 JSON 文件时出错: {e}")
            return None

    async def _manual_load_parquet(self, file_path: str, max_file_size_mb: int = 5000) -> Optional[Any]:
        """
        手动加载 Parquet 文件，避免缓存问题
        """
        try:
            # 检查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                log.warning(f"Parquet文件过大 ({file_size_mb:.2f} MB > {max_file_size_mb} MB)，跳过: {file_path}")
                return None
            
            log.info(f"尝试手动读取 Parquet 文件 ({file_size_mb:.2f} MB): {file_path}")
            
            # 使用pandas读取parquet文件
            import pandas as pd
            
            # 尝试多种读取策略
            strategies = [
                # 策略1: 使用pyarrow直接读取
                {"name": "pyarrow直接读取", "func": self._read_parquet_pyarrow},
                # 策略2: 使用pandas + pyarrow引擎
                {"name": "pandas+pyarrow", "func": self._read_parquet_pandas_pyarrow},
                # 策略3: 使用pandas + fastparquet引擎
                {"name": "pandas+fastparquet", "func": self._read_parquet_pandas_fastparquet},
                # 策略4: 使用pandas默认引擎
                {"name": "pandas默认", "func": self._read_parquet_pandas_default},
            ]
            
            df = None
            successful_strategy = None
            
            for strategy in strategies:
                try:
                    log.info(f"尝试Parquet读取策略: {strategy['name']}")
                    df = await strategy['func'](file_path)
                    if df is not None and len(df) > 0:
                        successful_strategy = strategy['name']
                        log.info(f"Parquet读取策略 '{strategy['name']}' 成功!")
                        break
                except Exception as e:
                    log.warning(f"Parquet读取策略 '{strategy['name']}' 失败: {e}")
                    continue
            
            if df is None:
                log.error(f"所有Parquet读取策略都失败: {file_path}")
                return None
            
            if len(df) == 0:
                log.warning(f"Parquet文件为空: {file_path}")
                return None

            return self._dataframe_to_simple_dataset(df)
                
        except Exception as e:
            log.error(f"手动加载 Parquet 文件时出错: {e}")
            return None

    async def _read_parquet_pyarrow(self, file_path: str):
        """使用pyarrow直接读取parquet文件"""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            return table.to_pandas()
        except Exception as e:
            log.warning(f"pyarrow直接读取失败: {e}")
            return None

    async def _read_parquet_pandas_pyarrow(self, file_path: str):
        """使用pandas + pyarrow引擎读取parquet文件"""
        try:
            import pandas as pd
            return pd.read_parquet(file_path, engine='pyarrow')
        except Exception as e:
            log.warning(f"pandas+pyarrow读取失败: {e}")
            return None

    async def _read_parquet_pandas_fastparquet(self, file_path: str):
        """使用pandas + fastparquet引擎读取parquet文件"""
        try:
            import pandas as pd
            return pd.read_parquet(file_path, engine='fastparquet')
        except Exception as e:
            log.warning(f"pandas+fastparquet读取失败: {e}")
            return None

    async def _read_parquet_pandas_default(self, file_path: str):
        """使用pandas默认引擎读取parquet文件"""
        try:
            import pandas as pd
            return pd.read_parquet(file_path)
        except Exception as e:
            log.warning(f"pandas默认引擎读取失败: {e}")
            return None

    async def _manual_load_generic(self, file_path: str, max_file_size_mb: int = 5000) -> Optional[Any]:
        """
        通用文件加载方法，用于处理其他类型的文件
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                log.warning(f"文件过大 ({file_size_mb:.2f} MB > {max_file_size_mb} MB)，跳过: {file_path}")
                return None
            
            log.info(f"尝试通用方法读取文件 ({file_size_mb:.2f} MB): {file_path}")
            
            # 根据文件扩展名选择读取方法
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return self._dataframe_to_simple_dataset(df)
            
            elif file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 将文本内容包装为单条记录
                records = [{"text": content}]
                dataset = _build_simple_dataset(records)
                if dataset:
                    log.info("成功从文本文件加载内容")
                return dataset
            
            else:
                log.warning(f"不支持的文件类型: {file_ext}")
                return None
                
        except Exception as e:
            log.error(f"通用文件加载方法出错: {e}")
            return None

    async def _load_with_datasets(self, builder_type: str, file_path: str) -> Optional[Any]:
        """使用datasets库的load_dataset方法加载文件"""
        try:
            # 为 datasets 创建可控位置的临时缓存目录，避免写入系统 /tmp
            temp_base_dir = os.getenv("DF_TEMP_DIR") or None
            if temp_base_dir is None:
                parent_dir = os.path.dirname(os.path.abspath(file_path))
                tmp_candidate = os.path.join(parent_dir, ".tmp")
                try:
                    os.makedirs(tmp_candidate, exist_ok=True)
                    temp_base_dir = tmp_candidate
                except Exception:
                    temp_base_dir = None
            temp_cache_dir = tempfile.mkdtemp(prefix="datasets_cache_", dir=temp_base_dir)
            self._temp_dirs.append(temp_cache_dir)

            # 明确指定使用内存与临时缓存目录，避免默认 .cache 目录
            from datasets import DownloadConfig, load_dataset
            dl_config = DownloadConfig(cache_dir=temp_cache_dir)

            # 尝试多种 load_dataset 参数组合（全部使用临时缓存目录 + 内存优先）
            strategies = [
                {
                    "name": "临时缓存+内存优先",
                    "params": {
                        "path": builder_type,
                        "data_files": file_path,
                        "cache_dir": temp_cache_dir,
                        "keep_in_memory": True,
                        "download_config": dl_config,
                    },
                },
                {
                    "name": "临时缓存+强制重建",
                    "params": {
                        "path": builder_type,
                        "data_files": file_path,
                        "cache_dir": temp_cache_dir,
                        "keep_in_memory": True,
                        "download_config": dl_config,
                        "download_mode": "force_redownload",
                    },
                },
            ]
            
            for strategy in strategies:
                try:
                    log.info(f"尝试datasets策略: {strategy['name']}")
                    data = load_dataset(**strategy['params'])
                    log.info(f"datasets策略 '{strategy['name']}' 成功!")
                    return data
                except Exception as e:
                    log.warning(f"datasets策略 '{strategy['name']}' 失败: {e}")
                    continue
            
            return None
        except Exception as e:
            log.error(f"datasets加载方法出错: {e}")
            return None

    async def _load_with_fallback(self, builder_type: str, file_path: str) -> Optional[Any]:
        """使用备用方法加载文件"""
        try:
            if builder_type == 'parquet':
                return await self._manual_load_parquet(file_path)
            if builder_type == 'json':
                return await self._manual_load_json(file_path)
            if builder_type == 'csv':
                return await self._manual_load_generic(file_path)
            return await self._manual_load_generic(file_path)
        except Exception as e:
            log.error(f"备用加载方法出错: {e}")
            return None

    async def _process_dataset(
        self, 
        data: Any, 
        file_path: str, 
        state: DataCollectionState, 
        category: str,
        output_jsonl_prefix: str,
        processed_sources_list: List[Tuple[str, int]]
    ) -> int:
        """
        处理单个数据集，提取数据并写入输出文件。
        
        Args:
            data: load_dataset 返回的数据集对象
            file_path: 数据文件路径（用于日志）
            state: 状态对象
            category: 数据类别 ('PT' 或 'SFT')
            output_jsonl_path: 输出文件路径
            processed_sources_list: 用于记录处理结果的列表
            
        Returns:
            处理的记录总数
        """
        total_count = 0
        file_name = os.path.basename(file_path)
        
        # 'data' 是一个 DatasetDict, e.g., {'train': Dataset, 'test': Dataset}
        # 准备所有需要处理的 split
        splits_to_process = []
        for split_name, data_content in data.items():
            if len(data_content) == 0:
                log.info(f"Split '{split_name}' 为空，跳过。")
                continue
            splits_to_process.append((split_name, data_content))
        
        if not splits_to_process:
            return 0
        
        # 并发处理所有 split 的映射（使用 Semaphore 控制并发数）
        semaphore_mapping = asyncio.Semaphore(self.max_concurrent_mapping)
        mapping_results = {}  # {split_name: (annotation_result, error)}
        
        async def process_split_mapping(split_name: str, data_content: Any):
            """处理单个 split 的映射"""
            async with semaphore_mapping:
                log.info(f"--- 正在处理 Split 映射: '{split_name}' (来自 {file_name}) ---")
                column_names = data_content.column_names
                sample_record = data_content[0]
                
                try:
                    # 显式调用父类方法，避免在嵌套函数中使用 super() 的问题
                    annotation_result = await DataConvertor.invoke(self, state, column_names, sample_record, dataset=data_content)
                    log.info(f"Split '{split_name}' LLM 映射成功")
                    return (split_name, annotation_result, None)
                except Exception as e:
                    log.error(f"LLM 数据映射失败，跳过 Split '{split_name}': {e}")
                    return (split_name, None, e)
        
        # 并发执行所有 split 的映射
        if len(splits_to_process) > 1:
            log.info(f"并发处理 {len(splits_to_process)} 个 Split 的映射（最大并发数: {self.max_concurrent_mapping}）")
            tasks = [process_split_mapping(split_name, data_content) 
                     for split_name, data_content in splits_to_process]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    log.error(f"Split 映射任务异常: {result}")
                    continue
                split_name, annotation_result, error = result
                mapping_results[split_name] = (annotation_result, error)
        else:
            # 只有一个 split，直接处理
            split_name, data_content = splits_to_process[0]
            _, annotation_result, error = await process_split_mapping(split_name, data_content)
            mapping_results[split_name] = (annotation_result, error)
        
        # 串行写入文件（保证写入顺序和线程安全）
        for split_name, data_content in splits_to_process:
            if split_name not in mapping_results:
                continue
                
            annotation_result, error = mapping_results[split_name]
            if error or annotation_result is None:
                log.error(f"跳过 Split '{split_name}' 的文件写入（映射失败）")
                continue
            
            log.info(f"--- 正在写入 Split: '{split_name}' (来自 {file_name}) ---")
                
            # 格式化并写入统一的 jsonl 文件
            split_record_count = 0
            # 分片写入：每 10000 条切换到下一个文件
            chunk_size = 10000
            current_chunk_index = 1
            current_chunk_count = 0

            def _open_chunk_file(index: int):
                chunk_path = f"{output_jsonl_prefix}_{index:05d}.jsonl"
                return open(chunk_path, 'a', encoding='utf-8')

            # 使用锁保护文件写入（并发写入时需要）
            async with self._write_lock:
                f_out = _open_chunk_file(current_chunk_index)
                try:
                    if category == 'PT':
                        # 使用新的中间格式构建器
                        for record_index, row in enumerate(data_content):
                            intermediate_record = self._build_intermediate_format_pt(
                                row, annotation_result, file_path, state, record_index
                            )
                            if intermediate_record:
                                json.dump(intermediate_record, f_out, ensure_ascii=False)
                                f_out.write('\n')
                                split_record_count += 1
                                current_chunk_count += 1
                                if current_chunk_count >= chunk_size:
                                    f_out.close()
                                    current_chunk_index += 1
                                    current_chunk_count = 0
                                    f_out = _open_chunk_file(current_chunk_index)
                                
                    elif category == 'SFT':
                        # 使用新的中间格式构建器
                        for record_index, row in enumerate(data_content):
                            intermediate_record = self._build_intermediate_format_sft(
                                row, annotation_result, file_path, state, record_index
                            )
                            if intermediate_record:
                                json.dump(intermediate_record, f_out, ensure_ascii=False)
                                f_out.write('\n')
                                split_record_count += 1
                                current_chunk_count += 1
                                if current_chunk_count >= chunk_size:
                                    f_out.close()
                                    current_chunk_index += 1
                                    current_chunk_count = 0
                                    f_out = _open_chunk_file(current_chunk_index)
                finally:
                    try:
                        f_out.close()
                    except Exception:
                        pass
            
            if split_record_count > 0:
                log.info(f"从 {file_name} ({split_name}) 提取了 {split_record_count} 条记录（中间格式）。")
                # 任务21: 记录中间格式详细信息
                if category == 'PT':
                    log.debug(f"PT模式中间格式包含: id, dataset_type='pretrain', text, meta={{source, language, ...}}")
                elif category == 'SFT':
                    log.debug(f"SFT模式中间格式包含: id, dataset_type='sft', messages=[{{role, content, loss_mask}}], system?, meta={{source, language, ...}}")
                processed_sources_list.append((f"{file_name}_({split_name})", split_record_count))
                total_count += split_record_count
        
        return total_count

    def _build_file_discovery_messages(self, state: DataCollectionState, file_list_str: str) -> List[BaseMessage]:
        """(文件发现) 为文件发现任务构建消息"""
        log.info("构建(文件发现)提示词消息...")
        ptg = PromptsTemplateGenerator(state.request.language)
        
        # 使用在父类中定义的 prompt name
        sys_prompt = ptg.render(self.system_prompt_file_discovery)
        task_params = {'file_list': file_list_str}
        task_prompt = ptg.render(self.task_prompt_file_discovery, **task_params)
        
        return [SystemMessage(content=sys_prompt), HumanMessage(content=task_prompt)]

    async def _invoke_file_discovery(self, state: DataCollectionState, file_list_str: str) -> List[str]:
        """
        (文件发现) 调用 LLM 以识别哪些文件是数据文件。
        """
        log.info("调用LLM(文件发现)并处理响应...")
        messages = self._build_file_discovery_messages(state, file_list_str)
        llm = self.create_llm(state) # 复用父类的 create_llm
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            log.info(f'LLM(文件发现)调用成功并返回结果: {answer_text}')

            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, answer_text).group(1).strip() if re.search(pattern, answer_text) else answer_text
            
            result = json.loads(match)
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
            else:
                log.error(f"LLM(文件发现)未返回字符串列表: {result}")
                raise ValueError("LLM did not return a JSON list of strings.")
        
        except Exception as e:
            log.exception(f"解析LLM(文件发现)响应失败: {e}")
            raise

    async def execute(self, state: DataCollectionState, **kwargs) -> DataCollectionState:
        """
        新的执行入口：
        1. 直接扫描整个下载文件夹结构。
        2. 调用 LLM 识别数据文件。
        3. 遍历识别出的文件：
            a. 用 'load_dataset' 加载。
            b. 调用父类的 'invoke' (LLM) 进行数据映射。
            c. 将数据写入统一的 jsonl 文件。
        4. 记录总结。
        """
        log.info(f"{self.role_name} (Universal) 开始执行...")
        _ensure_hf_cache_env(state.request.download_dir)
        category = state.request.category.upper() # 'PT' or 'SFT'
        
        # 记录输入
        inputs = {
            "role_name": self.role_name,
            "category": category,
            "download_dir": state.request.download_dir
        }
        log.info(f"[Agent Input] {self.role_name}.execute: {json.dumps(inputs, indent=2, ensure_ascii=False)}")
        
        if category not in ['PT', 'SFT']:
            log.error(f"不支持的数据类别: {category}")
            return state

        # 直接处理整个下载目录
        data_root = state.request.download_dir
        # 设置全局 TMPDIR，确保标准库和第三方库的临时文件落在受控目录
        try:
            controlled_tmp = os.getenv("DF_TEMP_DIR") or os.path.join(data_root, ".tmp")
            os.makedirs(controlled_tmp, exist_ok=True)
            os.environ.setdefault("TMPDIR", controlled_tmp)
        except Exception as e:
            log.warning(f"设置受控临时目录失败（可忽略）: {e}")

        if not os.path.exists(data_root):
            log.error(f"下载目录 {data_root} 不存在")
            return state

        # === 步骤 1: 文件发现 (LLM-driven) ===
        log.info(f"正在扫描整个下载目录: {data_root}")
        # 排除输出文件和摘要文件和rag文件，避免重复处理
        exclude_files = ['PT.jsonl', 'SFT.jsonl', 'summary.txt','chroma.sqlite3','data_level0.bin','header.bin','length.bin','link_lists.bin']
        file_list_str = self._get_file_list_string(data_root, exclude_files=exclude_files)
        
        if file_list_str == "This directory is empty.":
            log.warning(f"目录 {data_root} 为空，无文件可处理。")
            return state
        log.debug(f"文件列表:\n{file_list_str}")

        chunked_file_lists = self._chunk_file_list_for_llm(file_list_str)
        total_chunks = len(chunked_file_lists)
        log.info(f"文件列表将拆分为 {total_chunks} 个分块提交给 LLM 进行文件发现。")

        data_file_list: List[str] = []
        seen_paths: Set[str] = set()
        failed_chunks = 0

        for idx, chunk_str in enumerate(chunked_file_lists, start=1):
            log.info(
                f"正在处理文件发现分块 {idx}/{total_chunks}，字符数约 {len(chunk_str)}。"
            )
            try:
                chunk_result = await self._invoke_file_discovery(state, chunk_str)
                log.info(
                    f"分块 {idx}/{total_chunks} 返回 {len(chunk_result)} 个候选文件。"
                )
                for candidate in chunk_result:
                    if isinstance(candidate, str) and candidate not in seen_paths:
                        seen_paths.add(candidate)
                        data_file_list.append(candidate)
            except Exception as e:
                failed_chunks += 1
                log.error(f"LLM 文件发现分块 {idx}/{total_chunks} 失败: {e}")

        if not data_file_list:
            if failed_chunks == total_chunks:
                log.error("所有文件发现分块均失败，无法继续执行。")
            else:
                log.warning(f"LLM 未在 {data_root} 中找到任何数据文件。")
            return state

        if failed_chunks:
            log.warning(
                f"文件发现过程中有 {failed_chunks}/{total_chunks} 个分块失败，结果可能不完整。"
            )
        log.info(f"LLM 识别出 {len(data_file_list)} 个数据文件: {data_file_list}")

        # === 步骤 2 & 3: 数据转换与合并 ===
        
        # 创建专门的输出目录
        output_dir = os.path.join(data_root, "processed_output")
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"输出目录: {os.path.abspath(output_dir)}")
        
        # 统一输出文件前缀（放在输出目录中），每 10000 条切分一个文件
        output_jsonl_prefix = os.path.join(output_dir, f"{category.upper()}")
        log.info(f"========================================")
        log.info(f"输出文件前缀（绝对路径），将每10000条切分一个文件:")
        log.info(f"   {os.path.abspath(output_jsonl_prefix)}_00001.jsonl ...")
        log.info(f"========================================")
        processed_sources_list = [] 
        # 不提前创建文件，由写入过程按需创建各分片
            
        for relative_file_path in data_file_list:
            absolute_file_path = os.path.join(data_root, relative_file_path)
            
            if not os.path.exists(absolute_file_path):
                log.warning(f"LLM 返回了不存在的文件路径 '{relative_file_path}'，跳过。")
                continue
            
            log.info(f"--- 正在处理文件: {absolute_file_path} ---")
            
            files_to_process = []
            
            if self._is_compressed_file(absolute_file_path):
                log.info(f"检测到压缩文件: {absolute_file_path}")
                extracted_dir = self._extract_compressed_file(absolute_file_path)
                
                if not extracted_dir:
                    log.error(f"解压失败，跳过文件: {absolute_file_path}")
                    continue
                
                for root, dirs, files in os.walk(extracted_dir):
                    for f in files:
                        full_path = os.path.join(root, f)
                        if any(full_path.lower().endswith(ext) for ext in 
                               ['.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt']):
                            files_to_process.append(full_path)
                
                if not files_to_process:
                    log.warning(f"解压后未找到数据文件: {absolute_file_path}")
                    continue
                
                log.info(f"解压后找到 {len(files_to_process)} 个数据文件")
            else:
                files_to_process = [absolute_file_path]

            for file_path in files_to_process:
                log.info(f"--- 正在处理数据文件: {file_path} ---")
                builder_type = self._get_builder_type(file_path)
                if not builder_type:
                    log.warning(f"无法确定 builder type，跳过文件: {file_path}")
                    continue
                
                # 尝试多种加载策略
                data = None
                load_strategies = [
                    # 策略1: 使用load_dataset（原始方法）
                    {"name": "load_dataset", "func": self._load_with_datasets},
                    # 策略2: 根据文件类型使用专门的备用方法
                    {"name": "备用方法", "func": self._load_with_fallback},
                ]
                
                for strategy in load_strategies:
                    try:
                        log.info(f"尝试加载策略: {strategy['name']}")
                        data = await strategy['func'](builder_type, file_path)
                        if data is not None:
                            log.info(f"加载策略 '{strategy['name']}' 成功!")
                            break
                    except Exception as e:
                        log.warning(f"加载策略 '{strategy['name']}' 失败: {e}")
                        continue
                
                if data is None:
                    log.error(f"所有加载策略都失败，跳过文件: {file_path}")
                    continue

                await self._process_dataset(
                    data, file_path, state, category, 
                    output_jsonl_prefix, processed_sources_list
                )

        # --- 文件处理循环结束 ---
        total_records_processed = sum(count for _, count in processed_sources_list)
        log.info(f"整个下载目录处理完毕。总计提取 {total_records_processed} 条记录。")
        
        # 输出文件位置信息
        if total_records_processed > 0:
            log.info(f"========================================")
            log.info(f"数据已成功写入多个分片文件，前缀如下:")
            log.info(f"前缀路径: {os.path.abspath(output_jsonl_prefix)}_*.jsonl")
            log.info(f"记录总数: {total_records_processed}")
            log.info(f"========================================")
        else:
            log.warning(f"未提取到任何有效记录，输出文件可能为空或不存在。")
        
        # 将结果存入 state，以便 record_summary 使用
        # 使用 "all" 作为虚拟关键词
        state.sources["all"] = {category: processed_sources_list}
        
        # 如果 state.keywords 为空，添加 "all"
        if not state.keywords:
            state.keywords = ["all"]

        # 步骤 4: 清理临时目录
        log.info("正在清理临时解压目录...")
        self._cleanup_temp_dirs()
        
        # 步骤 4.5: 清理下载目录中的残留缓存文件（如 .conda 文件）
        log.info("正在清理下载目录中的残留缓存文件...")
        self._cleanup_download_dir_cache_files(data_root)

        # 步骤 5: 记录总结 (调用父类方法，传递输出目录)
        log.info("所有文件处理完毕，正在生成总结报告...")
        super().record_summary(state, output_dir=output_dir) 
        
        # 记录输出
        outputs = {
            "total_records_processed": total_records_processed,
            "processed_sources_count": len(processed_sources_list),
            "output_dir": os.path.abspath(output_dir)
        }
        log.info(f"[Agent Output] {self.role_name}.execute: {json.dumps(outputs, indent=2, ensure_ascii=False)}")
        log.info(f"========================================")
        log.info(f"{self.role_name} (Universal) 执行完成")
        log.info(f"所有输出文件位于: {os.path.abspath(output_dir)}")
        log.info(f"========================================")
        return state

    async def execute_with_langgraph(self, state: DataCollectionState, **kwargs) -> DataCollectionState:
        """
        使用 LangGraph 实现的数据转换流程（功能与 execute() 完全一致）
        """
        log.info(f"{self.role_name} (Universal, LangGraph版本) 开始执行...")
        _ensure_hf_cache_env(state.request.download_dir)
        category = state.request.category.upper()
        
        inputs = {
            "role_name": self.role_name,
            "category": category,
            "download_dir": state.request.download_dir
        }
        log.info(f"[Agent Input] {self.role_name}.execute_with_langgraph: {json.dumps(inputs, indent=2, ensure_ascii=False)}")
        
        if category not in ['PT', 'SFT']:
            log.error(f"不支持的数据类别: {category}")
            return state

        data_root = state.request.download_dir
        try:
            controlled_tmp = os.getenv("DF_TEMP_DIR") or os.path.join(data_root, ".tmp")
            os.makedirs(controlled_tmp, exist_ok=True)
            os.environ.setdefault("TMPDIR", controlled_tmp)
        except Exception as e:
            log.warning(f"设置受控临时目录失败（可忽略）: {e}")

        if not os.path.exists(data_root):
            log.error(f"下载目录 {data_root} 不存在")
            return state

        # 使用闭包变量存储临时数据，而不是 state 的临时字段
        data_file_list: List[str] = []
        processed_sources_list: List[Tuple[str, int]] = []
        output_jsonl_prefix: str = ""

        async def file_discovery_node(state: DataCollectionState) -> DataCollectionState:
            """文件发现节点"""
            nonlocal data_file_list, output_jsonl_prefix
            
            log.info("=== [LangGraph] 文件发现节点 ===")
            log.info(f"正在扫描整个下载目录: {data_root}")
            exclude_files = ['PT.jsonl', 'SFT.jsonl', 'summary.txt','chroma.sqlite3','data_level0.bin','header.bin','length.bin','link_lists.bin']
            file_list_str = self._get_file_list_string(data_root, exclude_files=exclude_files)
            
            if file_list_str == "This directory is empty.":
                log.warning(f"目录 {data_root} 为空，无文件可处理。")
                data_file_list = []
                return state
            
            log.debug(f"文件列表:\n{file_list_str}")
            chunked_file_lists = self._chunk_file_list_for_llm(file_list_str)
            total_chunks = len(chunked_file_lists)
            log.info(f"文件列表将拆分为 {total_chunks} 个分块提交给 LLM 进行文件发现。")

            discovered_files: List[str] = []
            seen_paths: Set[str] = set()
            failed_chunks = 0

            # 并发处理文件发现分块
            semaphore = asyncio.Semaphore(self.max_concurrent_discovery)
            
            async def process_chunk(idx: int, chunk_str: str):
                """处理单个文件发现分块"""
                async with semaphore:
                    log.info(f"正在处理文件发现分块 {idx}/{total_chunks}，字符数约 {len(chunk_str)}。")
                    try:
                        chunk_result = await self._invoke_file_discovery(state, chunk_str)
                        log.info(f"分块 {idx}/{total_chunks} 返回 {len(chunk_result)} 个候选文件。")
                        return (idx, chunk_result, None)
                    except Exception as e:
                        log.error(f"LLM 文件发现分块 {idx}/{total_chunks} 失败: {e}")
                        return (idx, [], e)
            
            # 并发执行所有分块
            tasks = [process_chunk(idx, chunk_str) for idx, chunk_str in enumerate(chunked_file_lists, start=1)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集结果（按顺序处理，避免重复）
            for result in results:
                if isinstance(result, Exception):
                    failed_chunks += 1
                    continue
                idx, chunk_result, error = result
                if error:
                    failed_chunks += 1
                else:
                    for candidate in chunk_result:
                        if isinstance(candidate, str) and candidate not in seen_paths:
                            seen_paths.add(candidate)
                            discovered_files.append(candidate)

            if not discovered_files:
                if failed_chunks == total_chunks:
                    log.error("所有文件发现分块均失败，无法继续执行。")
                else:
                    log.warning(f"LLM 未在 {data_root} 中找到任何数据文件。")
                data_file_list.clear()
                return state

            if failed_chunks:
                log.warning(f"文件发现过程中有 {failed_chunks}/{total_chunks} 个分块失败，结果可能不完整。")
            
            log.info(f"LLM 识别出 {len(discovered_files)} 个数据文件: {discovered_files}")
            # 更新闭包变量
            data_file_list.clear()
            data_file_list.extend(discovered_files)

            # 创建输出目录
            output_dir = os.path.join(data_root, "processed_output")
            os.makedirs(output_dir, exist_ok=True)
            log.info(f"输出目录: {os.path.abspath(output_dir)}")
            output_jsonl_prefix = os.path.join(output_dir, f"{category.upper()}")
            log.info(f"========================================")
            log.info(f"输出文件前缀（绝对路径），将每10000条切分一个文件:")
            log.info(f"   {os.path.abspath(output_jsonl_prefix)}_00001.jsonl ...")
            log.info(f"========================================")

            return state

        async def process_all_files_node(state: DataCollectionState) -> DataCollectionState:
            """批量处理所有文件节点（一次性处理整个文件列表）"""
            nonlocal data_file_list, data_root, category, output_jsonl_prefix, processed_sources_list
            
            log.info("=== [LangGraph] 批量处理所有文件节点 ===")
            log.info(f"开始批量处理 {len(data_file_list)} 个文件...")
            
            if not data_file_list:
                log.warning("文件列表为空，跳过处理")
                return state

            # 遍历所有文件并处理
            for relative_file_path in data_file_list:
                absolute_file_path = os.path.join(data_root, relative_file_path)
                
                if not os.path.exists(absolute_file_path):
                    log.warning(f"LLM 返回了不存在的文件路径 '{relative_file_path}'，跳过。")
                    continue

                log.info(f"--- 正在处理文件: {absolute_file_path} ---")
                files_to_process = []

                if self._is_compressed_file(absolute_file_path):
                    log.info(f"检测到压缩文件: {absolute_file_path}")
                    extracted_dir = self._extract_compressed_file(absolute_file_path)
                    
                    if not extracted_dir:
                        log.error(f"解压失败，跳过文件: {absolute_file_path}")
                        continue
                    
                    for root, dirs, files in os.walk(extracted_dir):
                        for f in files:
                            full_path = os.path.join(root, f)
                            if any(full_path.lower().endswith(ext) for ext in 
                                   ['.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt']):
                                files_to_process.append(full_path)
                    
                    if not files_to_process:
                        log.warning(f"解压后未找到数据文件: {absolute_file_path}")
                        continue
                    
                    log.info(f"解压后找到 {len(files_to_process)} 个数据文件")
                else:
                    files_to_process = [absolute_file_path]

                for file_path in files_to_process:
                    log.info(f"--- 正在处理数据文件: {file_path} ---")
                    builder_type = self._get_builder_type(file_path)
                    if not builder_type:
                        log.warning(f"无法确定 builder type，跳过文件: {file_path}")
                        continue
                    
                    data = None
                    load_strategies = [
                        {"name": "load_dataset", "func": self._load_with_datasets},
                        {"name": "备用方法", "func": self._load_with_fallback},
                    ]
                    
                    for strategy in load_strategies:
                        try:
                            log.info(f"尝试加载策略: {strategy['name']}")
                            data = await strategy['func'](builder_type, file_path)
                            if data is not None:
                                log.info(f"加载策略 '{strategy['name']}' 成功!")
                                break
                        except Exception as e:
                            log.warning(f"加载策略 '{strategy['name']}' 失败: {e}")
                            continue
                    
                    if data is None:
                        log.error(f"所有加载策略都失败，跳过文件: {file_path}")
                        continue

                    await self._process_dataset(
                        data, file_path, state, category, 
                        output_jsonl_prefix, processed_sources_list
                    )
            
            log.info(f"批量处理完成，共处理 {len(data_file_list)} 个文件")
            return state

        async def finalize_node(state: DataCollectionState) -> DataCollectionState:
            """最终化节点"""
            nonlocal processed_sources_list, output_jsonl_prefix, category, data_root
            
            log.info("=== [LangGraph] 最终化节点 ===")
            total_records_processed = sum(count for _, count in processed_sources_list)
            log.info(f"整个下载目录处理完毕。总计提取 {total_records_processed} 条记录。")
            
            if total_records_processed > 0:
                log.info(f"========================================")
                log.info(f"数据已成功写入多个分片文件，前缀如下:")
                log.info(f"前缀路径: {os.path.abspath(output_jsonl_prefix)}_*.jsonl")
                log.info(f"记录总数: {total_records_processed}")
                log.info(f"========================================")
            else:
                log.warning(f"未提取到任何有效记录，输出文件可能为空或不存在。")
            
            state.sources["all"] = {category: processed_sources_list}
            if not state.keywords:
                state.keywords = ["all"]

            output_dir = os.path.join(data_root, "processed_output")
            log.info("正在清理临时解压目录...")
            self._cleanup_temp_dirs()
            log.info("正在清理下载目录中的残留缓存文件...")
            self._cleanup_download_dir_cache_files(data_root)
            log.info("所有文件处理完毕，正在生成总结报告...")
            # 直接调用父类方法，因为 super() 在闭包中可能无法正确工作
            DataConvertor.record_summary(self, state, output_dir=output_dir)
            
            outputs = {
                "total_records_processed": total_records_processed,
                "processed_sources_count": len(processed_sources_list),
                "output_dir": os.path.abspath(output_dir)
            }
            log.info(f"[Agent Output] {self.role_name}.execute_with_langgraph: {json.dumps(outputs, indent=2, ensure_ascii=False)}")
            log.info(f"========================================")
            log.info(f"{self.role_name} (Universal, LangGraph版本) 执行完成")
            log.info(f"所有输出文件位于: {os.path.abspath(output_dir)}")
            log.info(f"========================================")

            return state

        # 构建 LangGraph 工作流（简化结构：文件发现 -> 批量处理 -> 最终化）
        graph_builder = StateGraph(DataCollectionState)
        graph_builder.add_node("file_discovery", file_discovery_node)
        graph_builder.add_node("process_all_files", process_all_files_node)
        graph_builder.add_node("finalize", finalize_node)

        # 简化的图结构：文件发现 -> 批量处理 -> 最终化
        graph_builder.add_edge(START, "file_discovery")
        graph_builder.add_edge("file_discovery", "process_all_files")
        graph_builder.add_edge("process_all_files", "finalize")
        graph_builder.add_edge("finalize", END)

        graph = graph_builder.compile()
        state = await graph.ainvoke(state)
        return state