from __future__ import annotations
import asyncio
import inspect
import sys
import os
from pydantic import BaseModel
import httpx
import json
import uuid
from typing import List, Dict, Sequence, Any, Union, Optional, Iterable, Mapping, Set, Callable
from pathlib import Path
 
from functools import lru_cache
import yaml
# from clickhouse_connect import get_client
import subprocess
from collections import defaultdict, deque
from dataflow.utils.storage import FileStorage
# from dataflow_agent.logger import get_logger
# logger = get_logger()
from dataflow_agent.storage.storage_service import SampleFileStorage
from dataflow_agent.state import DFState,DFRequest

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataflow.utils.registry import OPERATOR_REGISTRY
from langchain_core.tools import tool
from dataflow_agent.logger import get_logger

log = get_logger(__name__)
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
OPS_JSON_PATH = RESOURCE_DIR / "ops.json"

def local_tool_for_get_purpose(req: DFRequest) -> str:
    return req.target or ""

# ===================================================================更新算子库部分代码：
def _safe_json_val(val: Any) -> Any:
    """
    把任意 Python 对象转换成 JSON 可序列化的值。
    规则：
    1. 基本类型（None / bool / int / float / str）直接返回；
    2. enum/类对象 → 返回 'module.qualname'；
    3. 其它复杂对象 → 返回 str(val)；
    """
    # 空值直接交给 _param_to_dict 去处理
    if val is inspect.Parameter.empty:
        return None

    # 基本可 JSON 类型
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val

    # 类、函数、枚举等 → module.qualname
    if isinstance(val, type):
        return f"{val.__module__}.{val.__qualname__}"

    # Python3.10+ 的 A | B 产生的 UnionType
    if getattr(val, "__origin__", None) is None and val.__class__.__name__ == "UnionType":
        return str(val)          # e.g. "A | B | C"

    # 尝试直接 dump
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)

# 工具函数：安全调用带 @staticmethod 的 get_desc(lang)
def _call_get_desc_static(cls, lang: str = "zh") -> str | None:
    """
    仅当类的 get_desc 被显式声明为 @staticmethod 时才调用。
    兼容两种签名: (lang) 或 (self, lang)。
    返回 None 表示跳过此算子。
    """
    func_obj = cls.__dict__.get("get_desc")
    if not isinstance(func_obj, staticmethod):
        return None

    fn = func_obj.__func__
    params = list(inspect.signature(fn).parameters)
    try:
        if params == ["lang"]:
            return fn(lang)
        if params == ["self", "lang"]:
            return fn(None, lang)
    except Exception as e:
        log.warning(f"调用 {cls.__name__}.get_desc 失败: {e}")
    return None


# ---------------------------------------------------------------------------
def _param_to_dict(p: inspect.Parameter) -> Dict[str, Any]:
    """把 inspect.Parameter 转成 JSON 可序列化的字典（参考 MCP func 定义）"""
    return {
        "name": p.name,
        # "default": None if p.default is inspect.Parameter.empty else p.default,
        "default": _safe_json_val(p.default),
        "kind": p.kind.name,  # POSITIONAL_OR_KEYWORD / VAR_POSITIONAL / ...
    }


def _get_method_params(
    method: Any, skip_first_self: bool = False
) -> List[Dict[str, Any]]:
    """
    提取方法形参，转换为列表。
    skip_first_self=True 时会丢掉第一个 self 参数。
    """
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if skip_first_self and params and params[0].name == "self":
            params = params[1:]
        return [_param_to_dict(p) for p in params]
    except Exception as e:
        log.warning(f"获取方法参数出错: {e}")
        return []


def _gather_single_operator(
    op_name: str, cls: type, node_index: int
) -> Tuple[str, Dict[str, Any]]:
    """
    收集单个算子的全部信息，返回 (category, info_dict)
    """
    # 1) 分类：dataflow.operators.<category>.xxx
    category = "unknown"
    if hasattr(cls, "__module__"):
        parts = cls.__module__.split(".")
        if len(parts) >= 3 and parts[0] == "dataflow" and parts[1] == "operators":
            category = parts[2]

    # 2) 描述
    description = _call_get_desc_static(cls, lang="zh") or ""

    # 3) command 形参
    init_params = _get_method_params(cls.__init__, skip_first_self=True)
    run_params = _get_method_params(getattr(cls, "run", None), skip_first_self=True)

    info = {
        "node": node_index,
        "name": op_name,
        "description": description,
        "parameter": {
            "init": init_params,
            "run": run_params,
        },
        # 下面三项暂时留空，后续有需要再填
        "required": "",
        "depends_on": [],
        "mode": "",
    }
    return category, info


def _dump_all_ops_to_file() -> Dict[str, List[Dict[str, Any]]]:
    """
    遍历 OPERATOR_REGISTRY，构建完整字典并写入 ops.json。
    额外添加 "Default" → 所有算子全集。
    """
    log.info("开始扫描 OPERATOR_REGISTRY，生成 ops.json ...")

    if hasattr(OPERATOR_REGISTRY, "_init_loaders"):
        OPERATOR_REGISTRY._init_loaders()
    if hasattr(OPERATOR_REGISTRY, "_get_all"):
        OPERATOR_REGISTRY._get_all()

    all_ops: Dict[str, List[Dict[str, Any]]] = {}
    default_bucket: List[Dict[str, Any]] = []

    idx = 1
    for op_name, cls in OPERATOR_REGISTRY:
        category, info = _gather_single_operator(op_name, cls, idx)
        all_ops.setdefault(category, []).append(info)   
        default_bucket.append(info)
        idx += 1

    all_ops["Default"] = default_bucket

    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(OPS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_ops, f, ensure_ascii=False, indent=2)
        log.info(f"算子信息已写入 {OPS_JSON_PATH}")
    except Exception as e:
        log.warning(f"写入 {OPS_JSON_PATH} 失败: {e}")

    return all_ops

def _ensure_ops_cache() -> Dict[str, List[Dict[str, Any]]]:
    """
    若 ops.json 不存在或为空，则重新生成。
    返回文件中的全部数据。
    """
    if OPS_JSON_PATH.exists():
        try:
            with open(OPS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:  # 非空文件
                return data
        except Exception as e:
            log.warning(f"读取 {OPS_JSON_PATH} 失败，将重新生成: {e}")
    return _dump_all_ops_to_file()


# 供 LangChain Tool 调用的主函数
def get_operator_content(data_type: str) -> str:
    """
    根据传入的 `data_type`（即算子类别，如 "text2sql", "rag" …）
    返回该类别下所有算子的 JSON 字符串。

    如果该类别不存在，返回 "[]"
    """
    # all_ops = _ensure_ops_cache()
    all_ops = _dump_all_ops_to_file()

    import copy

    if data_type in all_ops:
        content = copy.deepcopy(all_ops[data_type])
    else:
        content = []

    # 作为字符串返回，方便 LLM 直接嵌入提示词
    return json.dumps(content, ensure_ascii=False, indent=2)


def get_operator_content_str(data_type: str) -> str:
    """
    返回该类别下所有算子的 “name:描述” 长字符串，用分号分隔。
    """
    all_ops = _dump_all_ops_to_file()  # 或 _ensure_ops_cache()
    raw_items = all_ops.get(data_type, [])

    # 用英文引号，如果有需要可用中文引号
    lines = [
        f'"{item.get("name", "")}":"{item.get("description", "")}"'
        for item in raw_items
    ]
    return "\n".join(lines)

def get_prompt_sources_of_operator(op_name: str) -> Dict[str, str]:
    """
    获取 operator 的 prompt_templates 的源码，并随机获取2个示例
    """
    import random
    cls = OPERATOR_REGISTRY.get(op_name)
    if cls is None:
        raise KeyError(f"Operator {op_name} not found in registry")
    log.info(f"Getting prompt_sources of {op_name}")
    
    # 获取 prompt_templates，如果没有则抛出异常
    if getattr(cls, "ALLOWED_PROMPTS", None):
        prompt_classes = cls.ALLOWED_PROMPTS
    else:
        raise ValueError(f"Operator {op_name} has no ALLOWED_PROMPTS")
    
    # 如果 prompt_templates 为空，则抛出异常，若只有一个，则直接使用，否则随机采样2个示例
    if len(prompt_classes) == 0:
        raise ValueError(f"Operator {op_name} has no prompt_templates")
    if len(prompt_classes) == 1:
        sample_classes = prompt_classes
    else:
        sample_classes = random.sample(prompt_classes, 2)
        
    # 获取源码
    out = {}
    for c in sample_classes:
        try:
            out[c.__name__] = inspect.getsource(c)
        except OSError:
            out[c.__name__] = "# 源码不可用（可能是C扩展/找不到源码/zip导入）"
    return out

def get_operators_info_by_names(operator_names: List[str]) -> str:
    """
    根据算子名称列表获取基本信息（node, name, description, category）。
    
    Args:
        operator_names: 算子名称列表，如 ['ExtractSmilesFromText', 'LLMLanguageFilter', ...]
        
    Returns:
        包含所有指定算子基本信息的JSON字符串。
        如果某个算子不存在，会在结果中标注 "error" 字段。
    """
    # 初始化 OPERATOR_REGISTRY
    if hasattr(OPERATOR_REGISTRY, "_init_loaders"):
        OPERATOR_REGISTRY._init_loaders()
    if hasattr(OPERATOR_REGISTRY, "_get_all"):
        OPERATOR_REGISTRY._get_all()
    
    # 构建名称到类的映射
    name_to_cls = {name: cls for name, cls in OPERATOR_REGISTRY}
    
    # 收集结果
    results = []
    idx = 1
    
    for op_name in operator_names:
        cls = name_to_cls.get(op_name)
        if cls is None:
            # 算子不存在
            results.append({
                "node": idx,
                "name": op_name,
                "error": f"算子 '{op_name}' 未在 OPERATOR_REGISTRY 中注册"
            })
        else:
            # 获取分类
            category = "unknown"
            if hasattr(cls, "__module__"):
                parts = cls.__module__.split(".")
                if len(parts) >= 3 and parts[0] == "dataflow" and parts[1] == "operators":
                    category = parts[2]
            
            # 获取描述
            description = _call_get_desc_static(cls, lang="zh") or ""
            
            # 只返回基本信息
            results.append({
                "node": idx,
                "name": op_name,
                "description": description,
                "category": category
            })
        idx += 1
    
    # 返回 JSON 字符串
    return json.dumps(results, ensure_ascii=False, indent=2)

def get_operator_source_by_name(operator_name: str) -> str:
    """
    根据算子名称获取算子的源码。
    参数:
        operator_name: 算子名称（注册在 OPERATOR_REGISTRY 中）
    返回:
        源码字符串或错误提示信息
    """
    try:
        # 初始化 OPERATOR_REGISTRY（如果需要）
        if hasattr(OPERATOR_REGISTRY, "_init_loaders"):
            OPERATOR_REGISTRY._init_loaders()
        if hasattr(OPERATOR_REGISTRY, "_get_all"):
            OPERATOR_REGISTRY._get_all()
        
        # 遍历注册的算子，找到匹配的名称
        for name, cls in OPERATOR_REGISTRY:
            if name == operator_name:
                # 获取源码
                try:
                    source_code = inspect.getsource(cls)
                    return source_code
                except Exception as e:
                    return f"# 无法获取源码: {e}"
        
        # 如果未找到对应的算子名称
        return f"# 未找到算子 '{operator_name}'，请检查名称是否正确。"
    
    except Exception as e:
        return f"# 获取算子源码时发生错误: {e}"

def get_prompt_sources_of_operator(op_name: str) -> Dict[str, str]:
    """
    获取 operator 的 prompt_templates 的源码，并随机获取2个示例
    """
    import random
    cls = OPERATOR_REGISTRY.get(op_name)
    if cls is None:
        raise KeyError(f"Operator {op_name} not found in registry")
    log.info(f"Getting prompt_sources of {op_name}")
    
    # 获取 prompt_templates，如果没有则抛出异常
    if getattr(cls, "ALLOWED_PROMPTS", None):
        prompt_classes = cls.ALLOWED_PROMPTS
    else:
        raise ValueError(f"Operator {op_name} has no ALLOWED_PROMPTS")
    
    # 如果 prompt_templates 为空，则抛出异常，若只有一个，则直接使用，否则随机采样2个示例
    if len(prompt_classes) == 0:
        raise ValueError(f"Operator {op_name} has no prompt_templates")
    if len(prompt_classes) == 1:
        sample_classes = prompt_classes
    else:
        sample_classes = random.sample(prompt_classes, 2)
        
    # 获取源码
    out = {}
    for c in sample_classes:
        try:
            out[c.__name__] = inspect.getsource(c)
        except OSError:
            out[c.__name__] = "# 源码不可用（可能是C扩展/找不到源码/zip导入）"
    return out

def post_process_combine_pipeline_result(results: Dict) -> str:

    return "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"


# if __name__ == "__main__":
#     log.info(get_operator_content("text2sql"))


# =================================================================== 算子RAG部分代码：
import os
import json
import pickle
import httpx
import numpy as np
import faiss
from typing import List, Dict, Union, Optional

import dataflow_agent.utils as utils

def _call_openai_embedding_api(
    texts: List[str],
    model_name: str = "text-embedding-ada-002",
    base_url: str = "https://api.openai.com/v1/embeddings",
    api_key: str | None = None,
    timeout: float = 120.0,
) -> np.ndarray:
    """调用OpenAI API获取文本向量"""
    if api_key is None:
        api_key = os.getenv("DF_API_KEY")
    if not api_key:
        raise RuntimeError("必须提供 OpenAI API-Key，可通过参数或环境变量 DF_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    vecs: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for t in texts:
            resp = client.post(
                base_url,
                headers=headers,
                json={"model": model_name, "input": t},
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"调用 OpenAI embedding 失败: {e}\n{resp.text}") from e

            try:
                data = resp.json()
                vec = data["data"][0]["embedding"]
            except Exception as e:
                raise RuntimeError(f"解析返回 JSON 失败: {resp.text}") from e

            vecs.append(vec)

    arr = np.asarray(vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


class RAGOperatorSearch:
    """RAG 算子检索类，支持向量持久化和批量查询"""
    
    def __init__(
        self,
        ops_json_path: str,
        category: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        base_url: str = "https://api.openai.com/v1/embeddings",
        api_key: Optional[str] = None,
    ):
        """
        初始化 RAG 检索器
        
        Args:
            ops_json_path: 算子JSON文件路径
            category: 算子类别，如果为None则读取全部
            faiss_index_path: FAISS索引文件路径，如果存在则复用，否则生成并保存
            model_name: embedding模型名称
            base_url: API base URL
            api_key: OpenAI API key
        """
        self.ops_json_path = ops_json_path
        self.category = category
        self.faiss_index_path = faiss_index_path
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        
        self.index = None
        self.ops_list = []
        
        self._load_or_build_index()
    
    def _load_operators(self) -> List[Dict]:
        """加载算子数据"""
        with open(self.ops_json_path, "r", encoding="utf-8") as f:
            all_ops = json.load(f)
        
        if self.category:
            # 指定类别
            ops = all_ops.get(self.category, [])
            log.info(f"✓ 加载类别 '{self.category}' 的算子: {len(ops)} 个")
        else:
            # 读取全部类别 - 直接使用 "Default" 避免重复加载
            ops = all_ops.get("Default", [])
            log.info(f"✓ 加载全部算子: {len(ops)} 个")
        
        return ops
    
    def _load_or_build_index(self):
        """加载或构建FAISS索引"""
        # 检查是否可以复用索引
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            meta_path = self.faiss_index_path + ".meta"
            if os.path.exists(meta_path):
                log.info(f"✓ 从 {self.faiss_index_path} 加载已有索引...")
                self.index = faiss.read_index(self.faiss_index_path)
                with open(meta_path, "rb") as f:
                    self.ops_list = pickle.load(f)
                log.info(f"✓ 索引加载成功，包含 {len(self.ops_list)} 个算子")
                return

        # 先用最新的 OPERATOR_REGISTRY 刷新 ops.json 快照
        log.info("⚙ 正在刷新 ops.json 算子快照...")
        _dump_all_ops_to_file()

        # 重新构建索引
        log.info("⚙ 开始构建新的向量索引...")
        self.ops_list = self._load_operators()
        
        if not self.ops_list:
            raise ValueError("没有找到任何算子数据！")
        
        # 生成文本描述
        texts = [f"{op['name']} {op.get('description', '')}" for op in self.ops_list]
        
        # 调用API获取向量
        log.info(f"⚙ 正在获取 {len(texts)} 个算子的 embedding...")
        embeddings = _call_openai_embedding_api(
            texts,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # 构建FAISS索引
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        log.info(f"✓ 索引构建完成，维度: {dim}")
        
        # 保存索引（如果指定了路径）
        if self.faiss_index_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.faiss_index_path) or ".", exist_ok=True)
            log.info(f"⚙ 保存索引到 {self.faiss_index_path}...")
            faiss.write_index(self.index, self.faiss_index_path)
            with open(self.faiss_index_path + ".meta", "wb") as f:
                pickle.dump(self.ops_list, f)
            log.info("✓ 索引保存成功")
    
    def search(
        self,
        queries: Union[str, List[str]],
        top_k: int = 5,
        return_scores: bool = False
    ) -> Union[List[str], List[List[str]], List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        检索最相关的算子
        
        Args:
            queries: 单个查询字符串或查询列表
            top_k: 返回top-k个结果
            return_scores: 是否返回相似度分数
        
        Returns:
            如果 return_scores=False:
                如果输入是字符串，返回List[str]
                如果输入是列表，返回List[List[str]]
            如果 return_scores=True:
                如果输入是字符串，返回List[Dict]，每个Dict包含 name, description, similarity_score
                如果输入是列表，返回List[List[Dict]]
        """
        # 统一处理为列表
        is_single = isinstance(queries, str)
        if is_single:
            queries = [queries]
        
        # 批量获取query向量
        query_vecs = _call_openai_embedding_api(
            queries,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # 检索
        D, I = self.index.search(query_vecs, top_k)
        
        # 组织结果
        results = []
        for i, (indices, scores) in enumerate(zip(I, D)):
            if return_scores:
                # 返回包含分数的详细信息
                matched_ops = []
                for idx, score in zip(indices, scores):
                    op_info = self.ops_list[idx]
                    matched_ops.append({
                        "name": op_info["name"],
                        "description": op_info.get("description", ""),
                        "similarity_score": float(score)  # FAISS cosine similarity score
                    })
                results.append(matched_ops)
                log.info(f"Query {i+1}: '{queries[i][:50]}...' -> {[(op['name'], round(op['similarity_score'], 3)) for op in matched_ops]}")
            else:
                # 原有逻辑，只返回名称列表
                matched_ops = [self.ops_list[idx]["name"] for idx in indices]
                results.append(matched_ops)
                log.info(f"Query {i+1}: '{queries[i][:50]}...' -> {matched_ops}")
        
        # 如果是单查询，返回单个列表
        return results[0] if is_single else results


def get_operators_by_rag(
    search_queries: Union[str, List[str]],
    category: Optional[str] = None,
    top_k: int = 4,
    ops_json_path: str = utils.get_project_root() / "dataflow_agent/toolkits/resources/ops.json",
    faiss_index_path: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    base_url: str = "http://123.129.219.111:3000/v1/embeddings",
    api_key: str = os.getenv("DF_API_KEY"),
) -> Union[List[str], List[List[str]]]:
    """
    通过RAG检索算子
    
    Args:
        search_queries: 单个查询字符串 或 查询列表 ['xxx1', 'xxx2']
        category: 算子类别，None表示读取全部
        top_k: 每个查询返回top-k结果
        ops_json_path: 算子JSON文件路径
        faiss_index_path: FAISS索引文件路径，如果存在则复用，否则重新生成
        model_name: embedding模型
        base_url: API地址
        api_key: API密钥
    
    Returns:
        单查询返回List[str]，多查询返回List[List[str]]
    
    Examples:
        # 单个查询
        result = get_operators_by_rag("将自然语言转换为SQL")
        # 返回: ['op1', 'op2', 'op3', 'op4']
        
        # 批量查询
        results = get_operators_by_rag(['query1', 'query2'])
        # 返回: [['op1', 'op2'], ['op3', 'op4']]
    """
    searcher = RAGOperatorSearch(
        ops_json_path=ops_json_path,
        category=category,
        faiss_index_path=faiss_index_path,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    
    return searcher.search(search_queries, top_k=top_k)


def local_tool_for_get_match_operator_code(pre_task_result):
    import time
    import sys
    import inspect
    from dataflow.utils.registry import OPERATOR_REGISTRY

    start_time = time.time()
    if not pre_task_result or not isinstance(pre_task_result, dict):
        return "# ❗ pre_task_result is empty, cannot extract operator names"

    _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}

    blocks = []
    for op_name in pre_task_result.get("match_operators", [])[:2]:
        cls = _NAME2CLS.get(op_name)
        if cls is None:
            blocks.append(f"# --- {op_name} is not registered in OPERATOR_REGISTRY ---")
            continue
        try:
            cls_src = inspect.getsource(cls)
            module_src = inspect.getsource(sys.modules[cls.__module__])
            import_lines = [
                l for l in module_src.splitlines()
                if l.strip().startswith(("import ", "from "))
            ]
            import_block = "\n".join(import_lines)
            src_block = f"# === Source of {op_name} ===\n{import_block}\n\n{cls_src}"
            blocks.append(src_block)
        except (OSError, TypeError) as e:
            blocks.append(f"# --- Failed to get the source code of {op_name}: {e} ---")
    
    elapsed = time.time() - start_time
    log.info(f"[local_tool_for_get_match_operator_code] Time used: {elapsed:.4f} seconds")
    return "\n\n".join(blocks)


# =================================================================== LangChain Tool 封装的 RAG 工具：

# 匹配质量阈值定义
MATCH_QUALITY_THRESHOLDS = {
    "high": 0.5,      # >= 0.5 为高度匹配
    "medium": 0.3,    # >= 0.3 为中等匹配
    # < 0.3 为低匹配
}


# 默认 FAISS 索引缓存路径
DEFAULT_FAISS_INDEX_PATH = str(utils.get_project_root() / "dataflow_agent/resources/faiss_cache/all_ops.index")


def _get_operators_by_rag_with_scores(
    search_query: str,
    top_k: int = 4,
    ops_json_path: str = None,
    faiss_index_path: str = None,
    model_name: str = "text-embedding-3-small",
    base_url: str = "http://123.129.219.111:3000/v1/embeddings",
    api_key: str = None,
) -> List[Dict[str, Any]]:
    """
    通过RAG检索算子，返回包含相似度分数的详细结果
    
    Args:
        search_query: 搜索查询
        top_k: 返回top-k结果
        ops_json_path: 算子JSON文件路径
        faiss_index_path: FAISS索引文件路径，如果存在则复用，否则生成并保存
        model_name: embedding模型
        base_url: API地址
        api_key: API密钥
    
    Returns:
        List[Dict]，每个Dict包含 name, description, similarity_score
    """
    if ops_json_path is None:
        ops_json_path = utils.get_project_root() / "dataflow_agent/toolkits/resources/ops.json"
    if faiss_index_path is None:
        faiss_index_path = DEFAULT_FAISS_INDEX_PATH
    if api_key is None:
        api_key = os.getenv("DF_API_KEY")
    
    searcher = RAGOperatorSearch(
        ops_json_path=str(ops_json_path),
        category=None,
        faiss_index_path=faiss_index_path,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    
    return searcher.search(search_query, top_k=top_k, return_scores=True)


def _determine_match_quality(max_score: float) -> str:
    """根据最高相似度分数判断匹配质量"""
    if max_score >= MATCH_QUALITY_THRESHOLDS["high"]:
        return "high"
    elif max_score >= MATCH_QUALITY_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def _generate_match_warning(query: str, max_score: float, match_quality: str) -> Optional[str]:
    """根据匹配质量生成警告信息"""
    if match_quality == "high":
        return None
    elif match_quality == "medium":
        return (
            f"提示：与'{query}'相关的算子匹配度为中等(最高相似度: {max_score:.3f})。"
            f"请仔细阅读算子描述，确认是否满足您的需求。"
        )
    else:  # low
        return (
            f"警告：未找到与'{query}'高度匹配的算子。最高相似度仅为{max_score:.3f}，"
            f"低于推荐阈值{MATCH_QUALITY_THRESHOLDS['medium']}。"
            f"当前返回的算子可能无法满足您的需求。如果没有合适的算子，"
            f"请在回复中说明'未能找到满足{query}需求的算子'。"
        )


@tool
def search_operator_by_description(query: str, top_k: int = 4) -> str:
    """
    根据功能描述搜索最匹配的数据处理算子。
    
    当需要在 pipeline 中添加新算子时，必须先调用此工具搜索真实存在的算子。
    禁止使用此工具返回结果之外的算子名称。
    
    **重要**：该工具会返回匹配质量评估(match_quality)：
    - "high": 高度匹配(相似度>=0.5)，可以放心使用
    - "medium": 中等匹配(相似度0.3-0.5)，请仔细确认是否满足需求
    - "low": 低匹配(相似度<0.3)，可能无法满足需求，请考虑说明"未能找到满足需求的算子"
    
    Args:
        query: 算子功能描述，例如 "情感分析"、"数据清洗"、"文本分类"、"去重"、"数据增强" 等
        top_k: 返回的候选算子数量，默认为4
    
    Returns:
        JSON 格式的搜索结果，包含匹配的算子名称、描述、相似度分数和匹配质量评估
    
    Examples:
        >>> search_operator_by_description("情感分析")
        >>> search_operator_by_description("数据去重", top_k=3)
    """
    try:
        # 调用 RAG 检索（返回包含分数的详细结果）
        matched_operators = _get_operators_by_rag_with_scores(query, top_k=top_k)
        
        # 计算最高相似度分数
        max_score = 0.0
        if matched_operators:
            max_score = max(op.get("similarity_score", 0.0) for op in matched_operators)
        
        # 判断匹配质量
        match_quality = _determine_match_quality(max_score)
        
        # 生成警告信息
        warning = _generate_match_warning(query, max_score, match_quality)
        
        # 构建返回结果
        result = {
            "query": query,
            "matched_operators": matched_operators,
            "max_similarity_score": round(max_score, 4),
            "match_quality": match_quality,
        }
        
        # 添加警告信息（如果有）
        if warning:
            result["warning"] = warning
        
        # 根据匹配质量生成不同的指导说明
        if match_quality == "high":
            result["instruction"] = (
                "请从 matched_operators 中选择最合适的算子名称(name字段)。"
                "匹配质量高，可以放心使用。"
            )
        elif match_quality == "medium":
            result["instruction"] = (
                "请从 matched_operators 中选择最合适的算子名称(name字段)。"
                "注意：匹配质量为中等，请仔细阅读算子描述(description)确认是否满足需求。"
            )
        else:  # low
            result["instruction"] = (
                "注意：当前匹配质量较低！请仔细评估 matched_operators 中的算子是否能满足需求。"
                f"如果没有合适的算子，请在回复中明确说明'未能找到满足「{query}」需求的算子'，"
                "并给出建议（如：建议用户自定义算子，或使用其他方式实现该功能）。"
            )
        
        log.info(
            f"[search_operator_by_description] 查询: '{query}' -> "
            f"匹配到 {len(matched_operators)} 个算子, "
            f"最高相似度: {max_score:.3f}, 匹配质量: {match_quality}"
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log.error(f"[search_operator_by_description] 搜索失败: {e}")
        return json.dumps({
            "error": str(e),
            "query": query,
            "matched_operators": [],
            "match_quality": "error"
        }, ensure_ascii=False)


@tool
def get_operator_code_by_name(operator_name: str) -> str:
    """
    根据算子名称获取算子的源代码。
    
    在选择了要使用的算子后，可以调用此工具获取算子的源代码，
    以便了解算子的 init 参数和 run 参数的具体用法。
    
    Args:
        operator_name: 算子名称，必须是 search_operator_by_description 返回的算子名称
    
    Returns:
        算子的源代码字符串
    """
    try:
        code = get_operator_source_by_name(operator_name)
        log.info(f"[get_operator_code_by_name] 获取算子 '{operator_name}' 的源代码成功")
        return code
    except Exception as e:
        log.error(f"[get_operator_code_by_name] 获取失败: {e}")
        return f"# 获取算子 '{operator_name}' 源代码失败: {e}"


if __name__ == "__main__":
    # ============ 示例1: 单个查询 + 指定category + 持久化索引 ============
    # log.info("\n" + "="*70)
    # log.info("示例1: 单个查询 + 指定category + 持久化索引")
    # log.info("="*70)
    # result1 = get_operators_by_rag(
    #     search_queries="将自然语言转换为SQL查询语句",
    #     category="text2sql",
    #     top_k=3,
    #     faiss_index_path="./faiss_cache/text2sql.index"  # 第一次生成，后续复用
    # )
    # log.info(f"\n返回结果: {result1}\n")
    
    # ============ 示例2: 批量查询 + 读取全部category ============
    log.info("\n" + "="*70)
    log.info("示例2: 批量查询 + 读取全部category")
    log.info("="*70)
    queries = [
        "数据清洗和预处理",
        "文本分类任务",
        "生成SQL语句"
    ]
    result2 = get_operators_by_rag(
        search_queries=queries,
        category=None,  # 不指定category，读取全部
        top_k=4,
        faiss_index_path=""
    )
    log.info(f"\n返回结果: {result2}\n")
    
    # ============ 示例3: 不持久化，每次重新生成 ============
    # log.info("\n" + "="*70)
    # log.info("示例3: 不持久化索引，每次重新生成")
    # log.info("="*70)
    # result3 = get_operators_by_rag(
    #     search_queries=["数据可视化", "模型训练"],
    #     category="text2sql",
    #     top_k=3,
    #     faiss_index_path=None  # 不指定路径，不持久化
    # )
    # log.info(f"\n返回结果: {result3}\n")
    
    # ============ 示例4: 自定义top_k ============
    # log.info("\n" + "="*70)
    # log.info("示例4: 自定义top_k=5")
    # log.info("="*70)
    # result4 = get_operators_by_rag(
    #     search_queries="数据库查询",
    #     top_k=5,
    #     faiss_index_path="./faiss_cache/all_ops.index"
    # )
    # log.info(f"\n返回结果: {result4}\n")
