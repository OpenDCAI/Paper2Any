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
from typing import Any, Dict, List, Optional, Set, Type, Tuple, TYPE_CHECKING
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DataCollectionState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

log = get_logger()


class SimpleDataset:
    """A lightweight dataset container that mimics the interface used from HuggingFace datasets."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records
        self._column_names = list(records[0].keys()) if records else []

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._records[index]

    def __iter__(self):  # type: ignore[override]
        return iter(self._records)

    @property
    def column_names(self) -> List[str]:
        return self._column_names


def _build_simple_dataset(records: List[Dict[str, Any]]) -> Optional[Dict[str, SimpleDataset]]:
    if not records:
        return None
    return {"train": SimpleDataset(records)}


def _ensure_hf_cache_env(download_dir: Optional[str]) -> None:
    """确保 HuggingFace 相关环境变量指向下载目录。"""
    if not download_dir:
        return

    base_dir = os.path.abspath(download_dir)
    hf_cache_root = os.path.join(base_dir, ".cache", "hf")
    hub_dir = os.path.join(hf_cache_root, "hub")
    datasets_dir = os.path.join(hf_cache_root, "datasets")
    transformers_dir = os.path.join(hf_cache_root, "transformers")

    for path in (hf_cache_root, hub_dir, datasets_dir, transformers_dir):
        os.makedirs(path, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", transformers_dir)


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
                # 记录输出
                log.info(f"[Agent Output] {self.role_name}.invoke: {json.dumps(annotation_result, indent=2, ensure_ascii=False)}")
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
            dirs[:] = [d for d in dirs if d not in ('.tmp', 'processed_output', '.cache', 'rag_db')]
            
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
                and d not in ('.cache', 'processed_output', '.tmp', 'rag_db')
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
            # 分片写入：每 10000 条切换到下一个文件
            chunk_size = 10000
            current_chunk_index = 1
            current_chunk_count = 0

            def _open_chunk_file(index: int):
                chunk_path = f"{output_jsonl_prefix}_{index:05d}.jsonl"
                return open(chunk_path, 'a', encoding='utf-8')

            f_out = _open_chunk_file(current_chunk_index)
            try:
                if category == 'PT':
                    text_field = annotation_result.get('text') if annotation_result else None
                    if not text_field or text_field not in column_names:
                        log.warning(f"未在 {file_name} ({split_name}) 中找到有效的 'text' 字段 (来自 LLM: {annotation_result})，跳过。")
                        continue
                    
                    for row in data_content:
                        text = row.get(text_field)
                        if text and isinstance(text, str):
                            json.dump({'text': text}, f_out, ensure_ascii=False)
                            f_out.write('\n')
                            split_record_count += 1
                            current_chunk_count += 1
                            if current_chunk_count >= chunk_size:
                                f_out.close()
                                current_chunk_index += 1
                                current_chunk_count = 0
                                f_out = _open_chunk_file(current_chunk_index)
                            
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