# dataflow_agent/toolkits/datatool/dataset_managers.py

from __future__ import annotations
import os
import asyncio
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from playwright.async_api import Page
import tenacity
import requests.exceptions

from dataflow_agent.logger import get_logger
from dataflow_agent.agentroles.data_agents.webresearch import ToolManager

# 在导入 huggingface_hub / datasets 之前，优先设置 HF_ENDPOINT
_df_hf_endpoint = os.getenv("DF_HF_ENDPOINT")
if _df_hf_endpoint and not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = _df_hf_endpoint

log = get_logger(__name__)

class HuggingFaceDatasetManager:

    
    def __init__(self, max_retries: int = 2, retry_delay: int = 5, size_categories: List[str] = None, cache_dir: str = None, disable_cache: bool = False, temp_base_dir: str = None):
        self.hf_endpoint = 'https://hf-mirror.com'
        self.max_retries = max_retries
        self.retry_delay = retry_delay # seconds
        self.size_categories = size_categories  # e.g., ["n<1K", "1K<n<10K", "10K<n<100K"]
        self.disable_cache = disable_cache
        os.environ['HF_ENDPOINT'] = self.hf_endpoint
        
        # 允许通过 DF_TEMP_DIR 或传参指定临时目录基准，避免写入系统 /tmp
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir
        if self.temp_base_dir:
            os.makedirs(self.temp_base_dir, exist_ok=True)

        # 如果禁用缓存，使用可控的临时目录并在下载后清理
        if disable_cache:
            import tempfile
            temp_cache = tempfile.mkdtemp(prefix="hf_cache_", dir=self.temp_base_dir)
            os.environ['HF_HOME'] = temp_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(temp_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(temp_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(temp_cache, "transformers")
            self._temp_cache_dir = temp_cache
            log.info(f"[HuggingFace] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
        elif cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            # 设置 HuggingFace 相关的缓存环境变量
            hf_cache = os.path.join(cache_dir, "hf_cache")
            os.makedirs(hf_cache, exist_ok=True)
            os.environ['HF_HOME'] = hf_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(hf_cache, "transformers")
            self._temp_cache_dir = None
            log.info(f"[HuggingFace] 缓存目录已设置为: {hf_cache} (避免占用系统盘)")
        else:
            # 如果未指定，使用默认的项目目录
            default_cache = os.path.join(os.getcwd(), ".cache", "hf")
            os.makedirs(default_cache, exist_ok=True)
            os.environ['HF_HOME'] = default_cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(default_cache, "hub")
            os.environ['HF_DATASETS_CACHE'] = os.path.join(default_cache, "datasets")
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(default_cache, "transformers")
            self._temp_cache_dir = None
            log.info(f"[HuggingFace] 使用默认缓存目录: {default_cache}")
        
        log.info(f"[HuggingFace] 初始化，最大重试次数: {self.max_retries}, 延迟: {self.retry_delay}s (线性增长), 数据集大小类别: {self.size_categories if self.size_categories else '不限制'}")

        # 延迟导入 HuggingFace 依赖，确保上方已正确设置缓存相关环境变量
        from huggingface_hub import HfApi, snapshot_download
        from datasets import get_dataset_config_names

        self.hf_api = HfApi(endpoint=self.hf_endpoint)
        self._snapshot_download = snapshot_download
        self._get_dataset_config_names = get_dataset_config_names

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:

        if isinstance(e, (
            ConnectionResetError, 
            ConnectionRefusedError, 
            requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError
        )):
            return True

        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [502, 503, 504]:
            return True
            
        error_str = str(e).lower()
        if any(err_msg in error_str for err_msg in [
            "10054", 
            "connection reset by peer",
            "timeout", 
            "serviceunavailable" 
        ]):
             return True
             
        return False

    async def _retry_async_thread(self, func, *args, **kwargs):

        def log_retry_attempt(retry_state: tenacity.RetryCallState):
            attempt = retry_state.attempt_number
            exception = retry_state.outcome.exception()
            log.info(f"[HuggingFace] 发生可重试网络错误 (Attempt {attempt}/{self.max_retries}): {exception}")

        retryer = tenacity.AsyncRetrying(

            stop=tenacity.stop_after_attempt(self.max_retries),
        
            wait=tenacity.wait_incrementing(start=self.retry_delay, increment=self.retry_delay),
            retry=tenacity.retry_if_exception(self._is_retryable_error),
            before_sleep=log_retry_attempt,
            reraise=True  
        )

        async def func_to_retry():
            return await asyncio.to_thread(func, *args, **kwargs)

        try:
            return await retryer(func_to_retry)
        
        except tenacity.RetryError as e:
            log.info(f"[HuggingFace] 所有 {self.max_retries} 次重试均失败。")
            if e.last_attempt and e.last_attempt.failed:
                raise e.last_attempt.exception
            else:
                raise Exception(f"HuggingFace操作失败 ({func.__name__})，但未捕获到特定异常。")
        
        except Exception as e:
            log.info(f"[HuggingFace] 发生不可重试错误: {e}")
            raise e 

    
    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> Dict[str, List[Dict]]:
        results = {}
        
        for keyword in keywords:
            try:
                log.info(f"[HuggingFace] 搜索关键词: '{keyword}'")

                datasets = await self._retry_async_thread(
                    self.hf_api.list_datasets, 
                    search=keyword, 
                    limit=max_results,
                    # size_categories=self.size_categories
                )
                
                results[keyword] = []
                for dataset in datasets:
                    # 尝试获取数据集大小信息
                    dataset_size = None
                    try:
                        # 尝试从dataset对象获取大小信息
                        if hasattr(dataset, 'siblings'):
                            # 计算所有文件的总大小
                            total_size = 0
                            for sibling in getattr(dataset, 'siblings', []):
                                if hasattr(sibling, 'size') and sibling.size:
                                    total_size += sibling.size
                            if total_size > 0:
                                dataset_size = total_size
                        # 如果siblings中没有，尝试从其他属性获取
                        if not dataset_size and hasattr(dataset, 'size'):
                            dataset_size = getattr(dataset, 'size', None)
                    except Exception as e:
                        # 如果获取大小失败，继续使用None
                        pass
                    
                    results[keyword].append({
                        "id": dataset.id,
                        "title": getattr(dataset, 'title', dataset.id),
                        "description": getattr(dataset, 'description', ''),
                        "downloads": getattr(dataset, 'downloads', 0),
                        "tags": getattr(dataset, 'tags', []),
                        "size": dataset_size  # 数据集大小（字节），可能为None
                    })
                
                log.info(f"[HuggingFace] 找到 {len(results[keyword])} 个数据集")
                
            except Exception as e:
                log.info(f"[HuggingFace] 搜索关键词 '{keyword}' 时出错 (经过重试后): {e}")
                results[keyword] = []
        
        return results
    
    # -----------------------------------------------------------------
    # vvvvvvvvvvvv   修改后的 download_dataset 方法   vvvvvvvvvvvv
    # -----------------------------------------------------------------
    async def download_dataset(self, dataset_id: str, save_dir: str) -> str | None:
        try:
            log.info(f"[HuggingFace] 开始下载数据集: {dataset_id}")
            dataset_dir = os.path.join(save_dir, dataset_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            config_to_load = None
            try:
                log.info(f"[HuggingFace] 正在检查 {dataset_id} 的配置...")
                
                configs = await self._retry_async_thread(
                    self._get_dataset_config_names,
                    path=dataset_id,
                    # base_url=self.hf_endpoint,  # 显式传入镜像端点
                    # token=self.hf_api.token 
                )
                
                if configs:
                    config_to_load = configs[0] 
                    log.info(f"[HuggingFace] 数据集 {dataset_id} 有 {len(configs)} 个配置. 自动选择第一个: {config_to_load}")
                else:
                    log.info(f"[HuggingFace] 数据集 {dataset_id} 没有特定的配置.")
            
            except Exception as e:
                log.info(f"[HuggingFace] 检查配置时出错 (将跳过配置检查，直接下载): {e}")
                config_to_load = None
            
            # --- 核心修改：使用 snapshot_download 替换 load_dataset ---
            log.info(f"[HuggingFace] 开始下载 {dataset_id} 的所有文件...")
            
            returned_path = await self._retry_async_thread(
                self._snapshot_download, 
                repo_id=dataset_id,
                local_dir=dataset_dir,
                repo_type="dataset",             # 明确告知是数据集
                force_download=True,           # 相当于 download_mode="force_redownload"
                # local_dir_use_symlinks=False,  # 推荐设置，避免Windows或跨设备问题
                endpoint=self.hf_endpoint      # 显式传入镜像端点，确保重试时使用镜像
                # token=self.hf_api.token      # 如果需要私有库，可以传入
            )
            # --- 修改结束 ---
            
            # 如果禁用了缓存，下载完成后清理临时缓存目录
            if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
                try:
                    if os.path.exists(self._temp_cache_dir):
                        shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                        log.info(f"[HuggingFace] 已清理临时缓存目录: {self._temp_cache_dir}")
                except Exception as e:
                    log.info(f"[HuggingFace] 清理临时缓存目录时出错: {e} (可忽略)")
            
            config_str = f"(配置: {config_to_load})" if config_to_load else "(默认配置)"
            log.info(f"[HuggingFace] 数据集 {dataset_id} {config_str} *文件*下载成功，保存至 {returned_path}")
            return returned_path
            
        except Exception as e:
            log.info(f"[HuggingFace] 下载数据集 {dataset_id} 失败 (经过重试后): {e}")
            return None

class KaggleDatasetManager:

    def __init__(self, search_engine: str = "tavily", cache_dir: str = None, disable_cache: bool = False, temp_base_dir: str = None):
        self.search_engine = search_engine
        self.api = None
        self.disable_cache = disable_cache
        self.temp_base_dir = os.getenv("DF_TEMP_DIR") or temp_base_dir
        if self.temp_base_dir:
            os.makedirs(self.temp_base_dir, exist_ok=True)
        
        # 如果禁用缓存，使用可控的临时目录并在下载后清理
        if disable_cache:
            import tempfile
            temp_cache = tempfile.mkdtemp(prefix="kaggle_cache_", dir=self.temp_base_dir)
            os.environ['KAGGLE_HUB_CACHE'] = temp_cache
            kaggle_config = os.path.join(temp_cache, "config")
            os.makedirs(kaggle_config, exist_ok=True)
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = temp_cache
            log.info(f"[Kaggle] 缓存已禁用，使用临时目录: {temp_cache} (下载后将自动清理)")
        elif cache_dir:
            cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            kaggle_cache = os.path.join(cache_dir, "kaggle_cache")
            os.makedirs(kaggle_cache, exist_ok=True)
            # 设置 kagglehub 缓存目录（通过环境变量）
            os.environ['KAGGLE_HUB_CACHE'] = kaggle_cache
            # Kaggle API 默认下载到指定目录，但可能还会有元数据缓存
            # 设置 KAGGLE_CONFIG_DIR 可以控制配置和缓存位置
            kaggle_config = os.path.join(kaggle_cache, "config")
            os.makedirs(kaggle_config, exist_ok=True)
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = None
            log.info(f"[Kaggle] 缓存目录已设置为: {kaggle_cache} (避免占用系统盘)")
        else:
            # 如果未指定，使用默认的项目目录
            default_cache = os.path.join(os.getcwd(), ".cache", "kaggle")
            os.makedirs(default_cache, exist_ok=True)
            os.environ['KAGGLE_HUB_CACHE'] = default_cache
            if 'KAGGLE_CONFIG_DIR' not in os.environ:
                kaggle_config = os.path.join(default_cache, "config")
                os.makedirs(kaggle_config, exist_ok=True)
                os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config
            self._temp_cache_dir = None
            log.info(f"[Kaggle] 使用默认缓存目录: {default_cache}")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
            self.api = KaggleApi()
            self.api.authenticate()
            log.info("[Kaggle] 已使用 KaggleApi 进行认证。")
        except Exception as e:
            log.info(f"[Kaggle] KaggleApi 初始化/认证失败: {e}. 请配置 ~/.kaggle/kaggle.json 或设置 KAGGLE_USERNAME/KAGGLE_KEY。将无法使用 Kaggle API。")

    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> Dict[str, List[Dict]]:
        """搜索Kaggle数据集，返回详细信息（类似HF格式）"""
        if not self.api:
            log.info("[Kaggle] 未初始化 KaggleApi，跳过 Kaggle 搜索。")
            return {}
        results = {}
        try:
            # KaggleApi 不支持并发调用，这里串行合并结果
            for kw in keywords:
                try:
                    # 设置 60 秒超时
                    items = await asyncio.wait_for(
                        asyncio.to_thread(self.api.dataset_list, search=kw),
                        timeout=60.0
                    )
                    results[kw] = []
                    for it in (items or [])[:max_results]:
                        # it.ref 格式如 owner/slug
                        ref = getattr(it, 'ref', None) or f"{getattr(it, 'ownerSlug', '')}/{getattr(it, 'datasetSlug', '')}"
                        if ref and '/' in ref:
                            total_size = getattr(it, 'totalBytes', 0) or getattr(it, 'total_bytes', 0)
                            if not total_size and self.api:
                                try:
                                    files_resp = await asyncio.wait_for(
                                        asyncio.to_thread(self.api.dataset_list_files, ref),
                                        timeout=30.0
                                    )
                                    if files_resp:
                                        files = getattr(files_resp, 'files', None) or []
                                        size_acc = 0
                                        for f in files:
                                            size_acc += getattr(f, 'totalBytes', 0) or getattr(f, 'fileSize', 0) or getattr(f, 'size', 0)
                                        if size_acc > 0:
                                            total_size = size_acc
                                except asyncio.TimeoutError:
                                    log.info(f"[Kaggle] 获取文件大小超时: {ref}")
                                except Exception as size_err:
                                    log.info(f"[Kaggle] 获取 {ref} 文件大小失败: {size_err}")
                            # 提取详细信息，并将复杂对象转换为字符串
                            raw_tags = getattr(it, 'tags', [])
                            try:
                                tags_list = [getattr(t, 'name', str(t)) for t in (raw_tags or [])]
                            except Exception:
                                tags_list = []
                            dataset_info = {
                                "id": ref,
                                "title": getattr(it, 'title', ref),
                                "description": getattr(it, 'description', ''),
                                "downloads": getattr(it, 'usabilityRating', 0),
                                "size": total_size,
                                "tags": tags_list,
                                "owner": getattr(it, 'ownerSlug', ''),
                                "url": f"https://www.kaggle.com/datasets/{ref}"
                            }
                            results[kw].append(dataset_info)
                except asyncio.TimeoutError:
                    log.info(f"[Kaggle] 搜索 '{kw}' 超时（60秒），跳过")
                    results[kw] = []
                except Exception as e:
                    log.info(f"[Kaggle] 搜索 '{kw}' 出错: {e}")
                    results[kw] = []
        except Exception as e:
            log.info(f"[Kaggle] 搜索失败: {e}")
            return {}
        
        log.info(f"[Kaggle] API 搜索汇总结果: {sum(len(v) for v in results.values())} 个候选")
        return results

    @staticmethod
    def _to_ref(s: str) -> str | None:
        # 支持 URL 或者 owner/slug 形式
        s = (s or '').strip()
        if not s:
            return None
        if 'kaggle.com/datasets/' in s:
            m = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?#]+)", s)
            if not m:
                return None
            return f"{m.group(1)}/{m.group(2)}"
        # 直接是 ref
        if '/' in s and len(s.split('/')) == 2:
            return s
        return None

    async def try_download(self, page: Page, dataset_identifier: str, save_dir: str) -> str | None:
        os.makedirs(save_dir, exist_ok=True)
        ref = self._to_ref(dataset_identifier)
        if not ref:
            log.info(f"[Kaggle] 无法解析数据集标识: {dataset_identifier}")
            return None
        
        # 优先使用 kagglehub
        # 注意：kagglehub 可能会在缓存目录留下文件，但我们已经设置了环境变量
        try:
            import kagglehub  # type: ignore
            log.info(f"[Kaggle] 优先使用 kagglehub 下载: {ref}")
            path = await asyncio.to_thread(kagglehub.dataset_download, ref)
            if path and os.path.exists(path):
                log.info(f"[Kaggle] kagglehub 下载完成: {path}")
                # 如果 kagglehub 下载的路径不在 save_dir 中，尝试将文件移动到指定目录
                # 这样可以避免在缓存目录留下文件
                if os.path.abspath(path) != os.path.abspath(save_dir):
                    try:
                        # 如果是文件，移动到 save_dir；如果是目录，复制内容
                        if os.path.isfile(path):
                            dest_path = os.path.join(save_dir, os.path.basename(path))
                            shutil.move(path, dest_path)
                            log.info(f"[Kaggle] 已移动文件到指定目录: {dest_path}")
                            return dest_path
                        elif os.path.isdir(path):
                            # 如果是目录，复制内容到 save_dir
                            for item in os.listdir(path):
                                src_item = os.path.join(path, item)
                                dst_item = os.path.join(save_dir, item)
                                if os.path.isdir(src_item):
                                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src_item, dst_item)
                            log.info(f"[Kaggle] 已复制内容到指定目录: {save_dir}")
                            # 如果禁用了缓存，删除原始缓存目录
                            if self.disable_cache:
                                try:
                                    shutil.rmtree(path, ignore_errors=True)
                                    log.info(f"[Kaggle] 已清理 kagglehub 缓存目录: {path}")
                                except Exception as e:
                                    log.info(f"[Kaggle] 清理缓存目录时出错: {e} (可忽略)")
                            return save_dir
                    except Exception as move_e:
                        log.info(f"[Kaggle] 移动/复制文件时出错: {move_e}，返回原始路径")
                return path
            log.info("[Kaggle] kagglehub 返回无效路径。")
        except Exception as e:
            log.info(f"[Kaggle] kagglehub 失败或未安装: {e}，尝试使用 KaggleApi。")
        
        # 如果 kagglehub 失败或未安装，尝试使用 KaggleApi（直接下载到指定目录，避免系统盘缓存）
        if self.api:
            try:
                log.info(f"[Kaggle] 使用 KaggleApi 下载: {ref} (直接下载到 {save_dir}，避免系统盘缓存)")
                # 设置 60 秒超时
                await asyncio.wait_for(
                    asyncio.to_thread(self.api.dataset_download_files, ref, path=save_dir, unzip=True, quiet=False),
                    timeout=60.0
                )
                log.info(f"[Kaggle] 下载完成并解压至: {save_dir}")
                return save_dir
            except asyncio.TimeoutError:
                log.info(f"[Kaggle] API 下载超时（60秒），失败")
            except Exception as e:
                log.info(f"[Kaggle] API 下载失败: {e}")
        else:
            log.info("[Kaggle] 未初始化 KaggleApi。")
        
        # 如果禁用了缓存，清理临时缓存目录
        if self.disable_cache and hasattr(self, '_temp_cache_dir') and self._temp_cache_dir:
            try:
                if os.path.exists(self._temp_cache_dir):
                    shutil.rmtree(self._temp_cache_dir, ignore_errors=True)
                    log.info(f"[Kaggle] 已清理临时缓存目录: {self._temp_cache_dir}")
            except Exception as e:
                log.info(f"[Kaggle] 清理临时缓存目录时出错: {e} (可忽略)")
        
        # 最后兜底失败
        return None

class PaddleDatasetManager:

    def __init__(self, search_engine: str = "tavily"):
        self.search_engine = search_engine
        log.info(f"[Paddle] 初始化 (search_engine={self.search_engine})")

    async def search_datasets(self, keywords: List[str], max_results: int = 5) -> List[str]:
        urls: List[str] = []
        for kw in keywords:
            try:
                query = f"site:paddlepaddle.org.cn {kw} 数据集"
                text = await ToolManager.search_web(query, search_engine=self.search_engine)
                found = ToolManager._extract_urls_from_markdown(text)
                # 只保留 Paddle 官方域名上的链接
                filtered = [u for u in found if "paddlepaddle.org.cn" in u]
                urls.extend(filtered)
            except Exception as e:
                log.info(f"[Paddle] 搜索 '{kw}' 出错: {e}")
        dedup = list(dict.fromkeys(urls))[:max_results]
        log.info(f"[Paddle] 搜索汇总结果: {len(dedup)} 个候选")
        return dedup

    async def try_download(self, page: Page, dataset_page_url: str, save_dir: str) -> str | None:
        os.makedirs(save_dir, exist_ok=True)
        try:
            content = await ToolManager._read_with_jina_reader(dataset_page_url)
            urls = content.get("urls", []) if content else []
            candidates = [u for u in urls if any(u.lower().endswith(ext) for ext in [".zip", ".csv", ".tar", ".gz", ".parquet"])]
            candidates = list(dict.fromkeys(candidates))
            log.info(f"[Paddle] 页面解析得到 {len(candidates)} 个下载候选链接")
            for u in candidates:
                path = await ToolManager.download_file(page, u, save_dir)
                if path:
                    return path
        except Exception as e:
            log.info(f"[Paddle] 解析页面失败: {e}")
        return None