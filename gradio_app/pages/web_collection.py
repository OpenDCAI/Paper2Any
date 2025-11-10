import os
import asyncio
import logging
from typing import Optional
import gradio as gr
from langgraph.graph import StateGraph, START, END

from dataflow_agent.state import DataCollectionRequest, DataCollectionState
from dataflow_agent.agentroles.dataconvertor import universal_data_conversion
from script.run_dfa_web_collection import web_crawl_collection


def create_web_collection():
    """子页面：网页数据采集与转换（基于 run_web_pipeline 工作流）"""
    with gr.Blocks() as page:
        gr.Markdown("# 🌐 网页数据采集与转换")

        with gr.Row():
            # 左侧：输入区域
            with gr.Column():
                gr.Markdown("### 采集配置")
                target = gr.Textbox(
                    label="目标描述",
                    placeholder="例如：收集 Python 代码示例的数据集",
                    lines=3
                )
                category = gr.Dropdown(
                    label="数据类别",
                    choices=["PT", "SFT"],
                    value="SFT"
                )
                dataset_num_limit = gr.Slider(
                    label="数据集数量上限（每关键词，仅用于参考）",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=5
                )
                dataset_size_category = gr.Dropdown(
                    label="数据集大小范围",
                    choices=["n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "n>1M"],
                    value="1K<n<10K"
                )
                max_download_subtasks = gr.Number(
                    label="下载子任务上限",
                    value=None,
                    precision=0,
                    minimum=0,
                    info="限制最终执行的下载子任务数量，留空表示不限制"
                )
                with gr.Row():
                    max_dataset_size_value = gr.Number(
                        label="最大数据集大小",
                        value=None,
                        precision=0,
                        minimum=0,
                        info="可留空；输入数值后选择单位"
                    )
                    max_dataset_size_unit = gr.Dropdown(
                        label="单位",
                        choices=["B", "KB", "MB", "GB", "TB"],
                        value="GB"
                    )
                download_dir = gr.Textbox(
                    label="下载目录",
                    value="downloaded_data",
                )
                language = gr.Dropdown(
                    label="提示词语言",
                    choices=["zh", "en"],
                    value="zh"
                )

                gr.Markdown("### LLM 配置")
                chat_api_url = gr.Textbox(
                    label="CHAT_API_URL",
                    value=os.getenv("CHAT_API_URL", "http://123.129.219.111:3000/v1/chat/completions")
                )
                api_key = gr.Textbox(
                    label="CHAT_API_KEY",
                    value=os.getenv("CHAT_API_KEY", ""),
                    type="password"
                )
                model = gr.Textbox(
                    label="CHAT_MODEL",
                    value=os.getenv("CHAT_MODEL", "deepseek-chat")
                )

                gr.Markdown("### 其他环境配置")
                hf_endpoint = gr.Textbox(
                    label="HF_ENDPOINT",
                    value=os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
                )
                kaggle_username = gr.Textbox(
                    label="KAGGLE_USERNAME",
                    value=os.getenv("KAGGLE_USERNAME", "")
                )
                kaggle_key = gr.Textbox(
                    label="KAGGLE_KEY",
                    value=os.getenv("KAGGLE_KEY", ""),
                    type="password"
                )
                tavily_api_key = gr.Textbox(
                    label="TAVILY_API_KEY",
                    value=os.getenv("TAVILY_API_KEY", ""),
                    type="password"
                )

                gr.Markdown("### RAG 配置")
                rag_ebd_model = gr.Textbox(
                    label="RAG_EBD_MODEL",
                    value=os.getenv("RAG_EBD_MODEL", "text-embedding-3-large")
                )
                rag_api_url = gr.Textbox(
                    label="RAG_API_URL",
                    value=os.getenv("RAG_API_URL", "http://123.129.219.111:3000/v1/chat/completions")
                )
                rag_api_key = gr.Textbox(
                    label="RAG_API_KEY",
                    value=os.getenv("RAG_API_KEY", ""),
                    type="password"
                )

                # 高级配置区域（可折叠）
                with gr.Accordion("⚙️ 高级配置", open=False):
                    gr.Markdown("### 网页采集高级配置")
                    max_crawl_cycles_per_task = gr.Slider(
                        label="下载任务最大循环次数",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=10,
                        info="控制每个下载任务的最大重试循环次数"
                    )
                    max_crawl_cycles_for_research = gr.Slider(
                        label="研究阶段最大循环次数",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=15,
                        info="research阶段的最大循环次数，允许访问更多网站"
                    )
                    search_engine = gr.Dropdown(
                        label="搜索引擎",
                        choices=["tavily", "duckduckgo", "jina"],
                        value="tavily",
                        info="选择用于搜索的引擎"
                    )
                    use_jina_reader = gr.Checkbox(
                        label="使用 Jina Reader",
                        value=True,
                        info="是否使用 Jina Reader 提取网页结构化内容（Markdown格式，快速）"
                    )
                    enable_rag = gr.Checkbox(
                        label="启用 RAG 增强",
                        value=True,
                        info="是否启用 RAG 增强（无论使用哪种解析方法，都用 RAG 精炼内容）"
                    )
                    concurrent_pages = gr.Slider(
                        label="并行处理页面数",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                        info="并行处理的页面数量，可根据网络和机器性能调整（建议3-10）"
                    )
                    disable_cache = gr.Checkbox(
                        label="禁用缓存",
                        value=True,
                        info="如果启用，将完全禁用 HuggingFace 和 Kaggle 的缓存，使用临时目录并在下载后自动清理"
                    )
                    temp_base_dir = gr.Textbox(
                        label="临时目录（可选）",
                        value="",
                        placeholder="留空则使用默认临时目录",
                        info="自定义临时目录路径，用于缓存和临时文件"
                    )

                    gr.Markdown("### 数据转换高级配置")
                    conversion_temperature = gr.Slider(
                        label="转换模型温度",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=0.0,
                        info="数据转换时使用的模型温度参数"
                    )
                    conversion_max_tokens = gr.Slider(
                        label="转换最大 Token 数",
                        minimum=512,
                        maximum=8192,
                        step=256,
                        value=4096,
                        info="数据转换时的最大 token 数"
                    )
                    conversion_max_sample_length = gr.Slider(
                        label="最大采样长度（字符）",
                        minimum=50,
                        maximum=1000,
                        step=50,
                        value=200,
                        info="每个字段的最大采样长度（字符数）"
                    )
                    conversion_num_sample_records = gr.Slider(
                        label="采样记录数量",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=3,
                        info="用于分析的采样记录数量"
                    )

                submit_btn = gr.Button("开始网页采集与转换", variant="primary")

            # 右侧：输出区域
            with gr.Column():
                with gr.Tab("执行日志"):
                    output_log = gr.Textbox(label="日志", lines=18)
                with gr.Tab("结果摘要"):
                    output_json = gr.JSON(label="执行结果")

        async def run_pipeline(
            target_text: str,
            category_val: str,
            dataset_num_limit_val: int,
            dataset_size_category_val: str,
            max_download_subtasks_val: float | None,
            max_dataset_size_value_val: float | None,
            max_dataset_size_unit_val: str,
            download_dir_val: str,
            language_val: str,
            chat_api_url_val: str,
            api_key_val: str,
            model_val: str,
            hf_endpoint_val: str,
            kaggle_username_val: str,
            kaggle_key_val: str,
            rag_ebd_model_val: str,
            rag_api_url_val: str,
            rag_api_key_val: str,
            tavily_api_key_val: str,
            # 高级配置参数
            max_crawl_cycles_per_task_val: int,
            max_crawl_cycles_for_research_val: int,
            search_engine_val: str,
            use_jina_reader_val: bool,
            enable_rag_val: bool,
            concurrent_pages_val: int,
            disable_cache_val: bool,
            temp_base_dir_val: str,
            conversion_temperature_val: float,
            conversion_max_tokens_val: int,
            conversion_max_sample_length_val: int,
            conversion_num_sample_records_val: int,
        ):
            # 注入/覆盖运行所需的环境变量
            os.environ["CHAT_API_URL"] = chat_api_url_val or ""
            os.environ["CHAT_API_KEY"] = api_key_val or ""
            os.environ["CHAT_MODEL"] = model_val or ""
            os.environ["HF_ENDPOINT"] = hf_endpoint_val or ""
            os.environ["KAGGLE_USERNAME"] = kaggle_username_val or ""
            os.environ["KAGGLE_KEY"] = kaggle_key_val or ""
            os.environ["RAG_EBD_MODEL"] = rag_ebd_model_val or ""
            os.environ["RAG_API_URL"] = rag_api_url_val or ""
            os.environ["RAG_API_KEY"] = rag_api_key_val or ""
            if tavily_api_key_val:
                os.environ["TAVILY_API_KEY"] = tavily_api_key_val
            else:
                os.environ.pop("TAVILY_API_KEY", None)

            # 设置高级配置相关环境变量
            if disable_cache_val:
                os.environ["DF_DISABLE_CACHE"] = "true"
            else:
                os.environ.pop("DF_DISABLE_CACHE", None)

            if temp_base_dir_val:
                os.environ["DF_TEMP_DIR"] = temp_base_dir_val
            else:
                os.environ.pop("DF_TEMP_DIR", None)

            # 组装请求
            def _convert_size_to_bytes(value: float | None, unit: str) -> Optional[int]:
                if value is None:
                    return None
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                if numeric <= 0:
                    return None
                unit = (unit or "B").upper()
                multipliers = {
                    "B": 1,
                    "KB": 1024,
                    "MB": 1024 ** 2,
                    "GB": 1024 ** 3,
                    "TB": 1024 ** 4,
                }
                multiplier = multipliers.get(unit, 1)
                return int(numeric * multiplier)

            max_dataset_size_bytes = _convert_size_to_bytes(max_dataset_size_value_val, max_dataset_size_unit_val)

            def _normalize_download_limit(value: float | None) -> Optional[int]:
                if value is None:
                    return None
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return None
                if numeric <= 0:
                    return None
                return numeric

            max_download_subtasks_int = _normalize_download_limit(max_download_subtasks_val)

            req = DataCollectionRequest(
                target=target_text,
                category=category_val,
                dataset_num_limit=int(dataset_num_limit_val),
                dataset_size_category=dataset_size_category_val,
                max_dataset_size=max_dataset_size_bytes,
                max_download_subtasks=max_download_subtasks_int,
                download_dir=download_dir_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val,
                model=model_val,
                language=language_val,
                tavily_api_key=tavily_api_key_val or None,
            )

            # 构建工作流
            state = DataCollectionState(request=req)

            # 创建包装函数以传递高级配置参数
            async def web_crawl_collection_wrapper(state: DataCollectionState) -> DataCollectionState:
                return await web_crawl_collection(
                    state,
                    max_crawl_cycles_per_task=int(max_crawl_cycles_per_task_val),
                    max_crawl_cycles_for_research=int(max_crawl_cycles_for_research_val),
                    search_engine=search_engine_val,
                    use_jina_reader=use_jina_reader_val,
                    enable_rag=enable_rag_val,
                    concurrent_pages=int(concurrent_pages_val),
                    disable_cache=bool(disable_cache_val),
                    temp_base_dir=(temp_base_dir_val.strip() or None) if isinstance(temp_base_dir_val, str) else None,
                    max_download_subtasks=max_download_subtasks_int,
                )

            async def universal_data_conversion_wrapper(state: DataCollectionState) -> DataCollectionState:
                return await universal_data_conversion(
                    state,
                    model_name=model_val or None,
                    temperature=float(conversion_temperature_val),
                    max_tokens=int(conversion_max_tokens_val),
                    max_sample_length=int(conversion_max_sample_length_val),
                    num_sample_records=int(conversion_num_sample_records_val),
                )

            graph_builder = StateGraph(DataCollectionState)
            graph_builder.add_node("web_crawl_collection", web_crawl_collection_wrapper)
            graph_builder.add_node("universal_data_conversion", universal_data_conversion_wrapper)
            graph_builder.add_edge(START, "web_crawl_collection")
            graph_builder.add_edge("web_crawl_collection", "universal_data_conversion")
            graph_builder.add_edge("universal_data_conversion", END)
            graph = graph_builder.compile()

            header_lines = [
                "=" * 60,
                "开始执行网页采集与转换工作流",
                "=" * 60,
                f"目标: {req.target}",
                f"类别: {req.category}",
                f"下载目录: {req.download_dir}",
                "\n【网页采集配置】",
                f"  - 搜索引擎: {search_engine_val}",
                f"  - 下载子任务上限: {max_download_subtasks_int if max_download_subtasks_int is not None else '不限制'}",
                f"  - 任务最大循环次数: {max_crawl_cycles_per_task_val}",
                f"  - 研究阶段最大循环次数: {max_crawl_cycles_for_research_val}",
                f"  - 使用 Jina Reader: {'是' if use_jina_reader_val else '否'}",
                f"  - 启用 RAG: {'是' if enable_rag_val else '否'}",
                f"  - 并行页面数: {concurrent_pages_val}",
                f"  - 禁用缓存: {'是' if disable_cache_val else '否'}",
                "\n【数据转换配置】",
                f"  - 模型温度: {conversion_temperature_val}",
                f"  - 最大 Token 数: {conversion_max_tokens_val}",
                f"  - 最大采样长度: {conversion_max_sample_length_val}",
                f"  - 采样记录数: {conversion_num_sample_records_val}",
                f"\n数据集大小限制: {max_dataset_size_bytes if max_dataset_size_bytes else '不限制'}",
                "=" * 60,
            ]

            log_lines: list[str] = header_lines.copy()
            log_queue: asyncio.Queue = asyncio.Queue()

            class DataflowLogFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                    return record.name.startswith("dataflow_agent") or record.name.startswith("script")

            class GradioLogHandler(logging.Handler):
                def __init__(self, queue: asyncio.Queue[str]):
                    super().__init__(level=logging.INFO)
                    self.queue = queue
                    self.addFilter(DataflowLogFilter())
                    self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

                def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                    try:
                        message = self.format(record)
                        loop = asyncio.get_running_loop()
                        loop.call_soon_threadsafe(self.queue.put_nowait, message)
                    except RuntimeError:
                        try:
                            self.queue.put_nowait(self.format(record))
                        except asyncio.QueueFull:
                            pass
                    except Exception:
                        pass

            handler = GradioLogHandler(log_queue)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            original_level = root_logger.level
            if original_level == 0 or original_level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            attached_loggers: set[logging.Logger] = {root_logger}

            def _attach_to_existing_loggers() -> None:
                logger_dict = logging.root.manager.loggerDict  # type: ignore[attr-defined]
                for name in list(logger_dict.keys()):
                    if isinstance(name, str) and (name.startswith("dataflow_agent") or name.startswith("script")):
                        logger_obj = logging.getLogger(name)
                        logger_obj.addHandler(handler)
                        attached_loggers.add(logger_obj)

                for name in ("dataflow_agent", "script"):
                    logger_obj = logging.getLogger(name)
                    logger_obj.addHandler(handler)
                    attached_loggers.add(logger_obj)

            _attach_to_existing_loggers()

            # 初始输出
            yield "\n".join(log_lines), gr.update(value=None)

            async def run_workflow() -> DataCollectionState:
                return await graph.ainvoke(state)

            workflow_task = asyncio.create_task(run_workflow())
            result_payload: Optional[dict] = None

            try:
                while True:
                    try:
                        message = await asyncio.wait_for(log_queue.get(), timeout=0.3)
                        log_lines.append(message)
                        yield "\n".join(log_lines), gr.update(value=result_payload)
                    except asyncio.TimeoutError:
                        if workflow_task.done():
                            break

                await workflow_task

                # 清空剩余日志
                while True:
                    try:
                        pending = log_queue.get_nowait()
                        log_lines.append(pending)
                    except asyncio.QueueEmpty:
                        break

                log_lines.append("流程执行完成！")

                result_payload = {
                    "download_dir": req.download_dir,
                    "processed_output": os.path.join(req.download_dir, "processed_output"),
                    "category": req.category,
                    "language": req.language,
                    "chat_model": req.model,
                    "max_download_subtasks": req.max_download_subtasks,
                    "max_dataset_size_bytes": req.max_dataset_size,
                    "max_dataset_size_unit": max_dataset_size_unit_val if req.max_dataset_size else None,
                    "max_dataset_size_value": max_dataset_size_value_val if req.max_dataset_size else None,
                }

                yield "\n".join(log_lines), result_payload

            except Exception as exc:
                error_message = f"流程执行失败: {exc}"
                log_lines.append(error_message)
                while True:
                    try:
                        pending = log_queue.get_nowait()
                        log_lines.append(pending)
                    except asyncio.QueueEmpty:
                        break
                result_payload = {"error": str(exc)}
                yield "\n".join(log_lines), result_payload
                raise
            finally:
                for logger_obj in attached_loggers:
                    logger_obj.removeHandler(handler)
                handler.close()
                root_logger.setLevel(original_level)

        submit_btn.click(
            run_pipeline,
            inputs=[
                target,
                category,
                dataset_num_limit,
                dataset_size_category,
                max_download_subtasks,
                max_dataset_size_value,
                max_dataset_size_unit,
                download_dir,
                language,
                chat_api_url,
                api_key,
                model,
                hf_endpoint,
                kaggle_username,
                kaggle_key,
                rag_ebd_model,
                rag_api_url,
                rag_api_key,
                tavily_api_key,
                # 高级配置参数
                max_crawl_cycles_per_task,
                max_crawl_cycles_for_research,
                search_engine,
                use_jina_reader,
                enable_rag,
                concurrent_pages,
                disable_cache,
                temp_base_dir,
                conversion_temperature,
                conversion_max_tokens,
                conversion_max_sample_length,
                conversion_num_sample_records,
            ],
            outputs=[output_log, output_json],
        )

    return page


