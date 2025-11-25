# dataflow_agent/workflow/wf_data_convertor.py

from __future__ import annotations
from typing import Optional
from dataflow_agent.state import DataCollectionState
from dataflow_agent.agentroles.data_convertor import DataConvertor, UniversalDataConvertor
from dataflow_agent.workflow.registry import register

async def data_conversion(
    state: DataCollectionState,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_sample_length: int = 200,
    num_sample_records: int = 3,
    **kwargs,
) -> DataCollectionState:
    """
    调用原始的 DataConvertor，处理 'tmp' 目录下的 HF 数据集。
    
    Args:
        state: 数据收集状态
        model_name: 模型名称
        temperature: 模型温度
        max_tokens: 最大token数
        max_sample_length: 每个字段的最大采样长度（字符数），默认200
        num_sample_records: 采样记录数量，默认3
    """
    data_collector = DataConvertor(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_sample_length=max_sample_length,
        num_sample_records=num_sample_records,
    )
    return await data_collector.execute(state, **kwargs)

async def universal_data_conversion(
    state: DataCollectionState,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_sample_length: int = 200,
    num_sample_records: int = 3,
    **kwargs,
) -> DataCollectionState:
    """
    调用新的 UniversalDataConvertor，处理原始下载目录。
    使用 LangGraph 版本实现。
    
    Args:
        state: 数据收集状态
        model_name: 模型名称
        temperature: 模型温度
        max_tokens: 最大token数
        max_sample_length: 每个字段的最大采样长度（字符数），默认200
        num_sample_records: 采样记录数量，默认3
    """
    data_collector = UniversalDataConvertor(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_sample_length=max_sample_length,
        num_sample_records=num_sample_records,
    )
    # 使用 LangGraph 版本
    return await data_collector.execute_with_langgraph(state, **kwargs)