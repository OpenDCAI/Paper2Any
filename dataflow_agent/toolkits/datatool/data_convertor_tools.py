# dataflow_agent/toolkits/datatool/data_convertor_tools.py

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from dataflow_agent.logger import get_logger

log = get_logger(__name__)

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