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
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Type, Tuple
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DataCollectionState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

log = get_logger()


# =================================================================
# 1. 原始 DataConvertor (已添加新方法）
# =================================================================

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

    def build_messages(self, state: DataCollectionState, column_names: List[str], sample_record: Dict[str, Any], dataset: Any = None) -> List[BaseMessage]:
        """(数据映射) 构建消息列表"""
        log.info("构建(数据映射)提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        # 如果提供了 dataset，使用采样功能；否则使用单条记录
        if dataset is not None:
            sampled_records = self._sample_records(dataset)
            # 对于模板，我们仍然传 first_row，但现在是采样列表的第一个
            task_params = {
                'column_names': column_names, 
                'first_row': sampled_records[0] if sampled_records else sample_record,
                'sample_rows': sampled_records,  # 额外提供采样列表
                'user_target': state.request.target  # 添加用户需求
            }
        else:
            # 如果没有提供 dataset，截断单条记录
            truncated_record = {k: self._truncate_value(v) for k, v in sample_record.items()}
            task_params = {
                'column_names': column_names, 
                'first_row': truncated_record,
                'sample_rows': [truncated_record],
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
                return annotation_result
            except json.JSONDecodeError as e:
                log.exception(f"解析GPT(数据映射)响应为JSON失败: 内容为{match}")
                raise ValueError(f"Failed to parse GPT response as JSON: {e}")
            
        except Exception as e:
            log.exception("hf数据集(数据映射)标注失败: %s", e)
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
                            text_field = annotation_result.get('text', None)
                            if text_field is None or text_field not in data_content.column_names:
                                log.info(f"数据集 {dataset_id}_{split} 标注结果中未包含有效的 'text' 字段，跳过该数据集")
                                continue

                            data_file = os.path.join(data_dir, 'PT.jsonl')
                            count = 0
                            with open(data_file, 'a', encoding='utf-8') as f:
                                for row in data_content:
                                    text = row.get(text_field)
                                    if text and isinstance(text, str):
                                        json_obj = {'text': text}
                                        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                                        count += 1
                            data_sources['PT'].append((f'{dataset_id}_({split})', count))
                            log.info(f"从数据集 {dataset_id}, split {split} 中提取了 {count} 条 PT 样本。")

                        elif category == 'SFT':
                            question_field = annotation_result.get('question', None)
                            answer_field = annotation_result.get('answer', None)

                            if question_field is None or question_field not in data_content.column_names or answer_field is None or answer_field not in data_content.column_names:
                                log.info(f"数据集 {dataset_id}_{split} 标注结果中未包含有效的 'question'/'answer' 字段，跳过该数据集")
                                continue

                            data_file = os.path.join(data_dir, 'SFT.jsonl')
                            count = 0
                            with open(data_file, 'a', encoding='utf-8') as f:
                                for row in data_content:
                                    question = row.get(question_field)
                                    answer = row.get(answer_field)
                                    if question and isinstance(question, str) and answer and isinstance(answer, str):
                                        json_obj = {
                                            'question': question,
                                            'answer': answer
                                        }
                                        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                                        count += 1
                            data_sources['SFT'].append((f'{dataset_id}_({split})', count))
                            log.info(f"从数据集 {dataset_id}, split {split} 中提取了 {count} 条 SFT 样本。")
                            
                except Exception as e:
                    log.error(f"处理数据集 {dataset_id} 时出错: {e}, 跳过该数据集")
                    continue

            state.sources[keyword] = data_sources
        
        # Step 2: Record summary
        self.record_summary(state)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 临时目录列表，用于清理
        self._temp_dirs = []

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
            
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="dataflow_extract_")
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
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__')) and d not in ('.cache', 'processed_output')]
            files = [f for f in files if not f.startswith(('.', '__'))]
            
            for f in files:
                if f in exclude_files:
                    continue                
                full_path = os.path.join(root, f)
                relative_path = os.path.relpath(full_path, root_path)
                file_list.append(relative_path.replace(os.sep, '/'))
        
        if not file_list:
            return "This directory is empty."
        
        return "File list:\n" + "\n".join(sorted(file_list))

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
        from datasets import Dataset, DatasetDict
        
        try:
            # 检查文件大小，避免加载超大文件导致内存问题
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                log.warning(f"文件过大 ({file_size_mb:.2f} MB > {max_file_size_mb} MB)，跳过手动加载: {file_path}")
                return None
            
            log.info(f"尝试手动读取 JSON 文件 ({file_size_mb:.2f} MB): {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                log.warning(f"文件为空: {file_path}")
                return None
            
            records = []
            
            # 尝试1: JSONL 格式（每行一个 JSON）
            try:
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                
                if records:
                    log.info(f"成功以 JSONL 格式加载 {len(records)} 条记录")
                    dataset = Dataset.from_list(records)
                    return DatasetDict({"train": dataset})
            except:
                pass
            
            # 尝试2: JSON 数组格式
            try:
                data = json.loads(content)
                
                # 如果是列表，直接使用
                if isinstance(data, list):
                    if data:
                        log.info(f"成功以 JSON 数组格式加载 {len(data)} 条记录")
                        dataset = Dataset.from_list(data)
                        return DatasetDict({"train": dataset})
                    else:
                        log.warning(f"JSON 数组为空: {file_path}")
                        return None
                
                # 如果是字典，检查是否包含数据列表
                elif isinstance(data, dict):
                    # 尝试查找可能的数据字段
                    for key in ['data', 'items', 'records', 'examples', 'train', 'test']:
                        if key in data and isinstance(data[key], list) and data[key]:
                            log.info(f"在字典的 '{key}' 字段中找到 {len(data[key])} 条记录")
                            dataset = Dataset.from_list(data[key])
                            return DatasetDict({"train": dataset})
                    
                    # 如果字典本身就是一条记录，包装成列表
                    log.info(f"将单个 JSON 对象包装为数据集")
                    dataset = Dataset.from_list([data])
                    return DatasetDict({"train": dataset})
                
                else:
                    log.warning(f"不支持的 JSON 类型: {type(data)}")
                    return None
                    
            except json.JSONDecodeError as e:
                log.error(f"JSON 解析失败: {e}")
                return None
        
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
            
            # 转换为Dataset格式
            if len(df) > 0:
                # 将DataFrame转换为字典列表
                records = df.to_dict('records')
                log.info(f"成功从Parquet文件加载 {len(records)} 条记录")
                
                # 创建Dataset
                dataset = Dataset.from_list(records)
                return DatasetDict({"train": dataset})
            else:
                log.warning(f"Parquet文件为空: {file_path}")
                return None
                
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
                records = df.to_dict('records')
                dataset = Dataset.from_list(records)
                log.info(f"成功从CSV文件加载 {len(records)} 条记录")
                return DatasetDict({"train": dataset})
            
            elif file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 将文本内容包装为单条记录
                records = [{"text": content}]
                dataset = Dataset.from_list(records)
                log.info(f"成功从文本文件加载内容")
                return DatasetDict({"train": dataset})
            
            else:
                log.warning(f"不支持的文件类型: {file_ext}")
                return None
                
        except Exception as e:
            log.error(f"通用文件加载方法出错: {e}")
            return None

    async def _load_with_datasets(self, builder_type: str, file_path: str) -> Optional[Any]:
        """使用datasets库的load_dataset方法加载文件"""
        try:
            # 尝试多种load_dataset参数组合
            strategies = [
                # 策略1: 基本参数
                {"name": "基本参数", "params": {"path": builder_type, "data_files": file_path}},
                # 策略2: 禁用缓存
                {"name": "禁用缓存", "params": {"path": builder_type, "data_files": file_path, "cache_dir": None}},
                # 策略3: 强制重新下载
                {"name": "强制重新下载", "params": {"path": builder_type, "data_files": file_path, "download_mode": "force_redownload"}},
                # 策略4: 使用临时缓存
                {"name": "临时缓存", "params": {"path": builder_type, "data_files": file_path, "cache_dir": "/tmp/datasets_cache"}},
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
            elif builder_type in ['json', 'csv']:
                return await self._manual_load_json(file_path)
            else:
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
        output_jsonl_path: str,
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
        for split_name, data_content in data.items():
            log.info(f"--- 正在处理 Split: '{split_name}' (来自 {file_name}) ---")
            
            if len(data_content) == 0:
                log.info(f"Split '{split_name}' 为空，跳过。")
                continue
                
            # 获取示例，调用 LLM (父类方法) 进行数据映射
            column_names = data_content.column_names
            sample_record = data_content[0]
            
            try:
                # 调用 DataConvertor.invoke()，传入完整 dataset 用于采样
                annotation_result = await super().invoke(state, column_names, sample_record, dataset=data_content)
                log.info(f"LLM 映射结果: {annotation_result}")
            except Exception as e:
                log.error(f"LLM 数据映射失败，跳过 Split '{split_name}': {e}")
                continue # 跳过这个 split
                
            # 格式化并写入统一的 jsonl 文件
            split_record_count = 0
            with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
                if category == 'PT':
                    text_field = annotation_result.get('text')
                    if not text_field or text_field not in column_names:
                        log.warning(f"未在 {file_name} ({split_name}) 中找到有效的 'text' 字段 (来自 LLM: {text_field})，跳过。")
                        continue
                    
                    for row in data_content:
                        text = row.get(text_field)
                        if text and isinstance(text, str):
                            json.dump({'text': text}, f_out, ensure_ascii=False)
                            f_out.write('\n')
                            split_record_count += 1
                            
                elif category == 'SFT':
                    q_field = annotation_result.get('question')
                    a_field = annotation_result.get('answer')
                    
                    if not q_field or q_field not in column_names or not a_field or a_field not in column_names:
                        log.warning(f"未在 {file_name} ({split_name}) 中找到有效的 'question'/'answer' 字段 (来自 LLM: {q_field}, {a_field})，跳过。")
                        continue
                    
                    for row in data_content:
                        question = row.get(q_field)
                        answer = row.get(a_field)
                        if question and isinstance(question, str) and answer and isinstance(answer, str):
                            json.dump({'question': question, 'answer': answer}, f_out, ensure_ascii=False)
                            f_out.write('\n')
                            split_record_count += 1
            
            if split_record_count > 0:
                log.info(f"从 {file_name} ({split_name}) 提取了 {split_record_count} 条记录。")
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
        category = state.request.category.upper() # 'PT' or 'SFT'
        
        if category not in ['PT', 'SFT']:
            log.error(f"不支持的数据类别: {category}")
            return state

        # 直接处理整个下载目录
        data_root = state.request.download_dir

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
        
        try:
            # 调用 LLM 筛选数据文件
            data_file_list = await self._invoke_file_discovery(state, file_list_str)
            log.info(f"LLM 识别出 {len(data_file_list)} 个数据文件: {data_file_list}")
        except Exception as e:
            log.error(f"LLM 文件发现失败: {e}")
            return state
        
        if not data_file_list:
            log.warning(f"LLM 未在 {data_root} 中找到任何数据文件。")
            return state

        # === 步骤 2 & 3: 数据转换与合并 ===
        
        # 创建专门的输出目录
        output_dir = os.path.join(data_root, "processed_output")
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"输出目录: {os.path.abspath(output_dir)}")
        
        # 统一输出文件（放在输出目录中）
        output_jsonl_path = os.path.join(output_dir, f"{category.upper()}.jsonl")
        log.info(f"========================================")
        log.info(f"输出文件路径（绝对路径）:")
        log.info(f"   {os.path.abspath(output_jsonl_path)}")
        log.info(f"========================================")
        processed_sources_list = [] 
        
        if os.path.exists(output_jsonl_path):
            log.info(f"输出文件已存在")
        else:
            log.info(f"将创建新的输出文件")
            
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
                    output_jsonl_path, processed_sources_list
                )

        # --- 文件处理循环结束 ---
        total_records_processed = sum(count for _, count in processed_sources_list)
        log.info(f"整个下载目录处理完毕。总计提取 {total_records_processed} 条记录。")
        
        # 输出文件位置信息
        if total_records_processed > 0:
            log.info(f"========================================")
            log.info(f"数据已成功写入文件:")
            log.info(f"文件路径: {os.path.abspath(output_jsonl_path)}")
            log.info(f"记录总数: {total_records_processed}")
            log.info(f"文件大小: {os.path.getsize(output_jsonl_path) / (1024*1024):.2f} MB" if os.path.exists(output_jsonl_path) else "   文件大小: 未知")
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

        # 步骤 5: 记录总结 (调用父类方法，传递输出目录)
        log.info("所有文件处理完毕，正在生成总结报告...")
        super().record_summary(state, output_dir=output_dir) 
        
        log.info(f"========================================")
        log.info(f"{self.role_name} (Universal) 执行完成")
        log.info(f"所有输出文件位于: {os.path.abspath(output_dir)}")
        log.info(f"========================================")
        return state


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
    return await data_collector.execute(state, **kwargs)