import os
import gradio as gr
from langgraph.graph import StateGraph, START, END

from dataflow_agent.state import DataCollectionRequest, DataCollectionState
from dataflow_agent.agentroles.dataconvertor import universal_data_conversion
from script.run_web_pipeline import web_crawl_collection


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
            
            # 设置高级配置相关环境变量
            if disable_cache_val:
                os.environ["DF_DISABLE_CACHE"] = "true"
            else:
                os.environ.pop("DF_DISABLE_CACHE", None)
            
            if temp_base_dir_val:
                os.environ["DF_TEMP_DIR"] = temp_base_dir_val

            # 组装请求
            req = DataCollectionRequest(
                target=target_text,
                category=category_val,
                dataset_num_limit=int(dataset_num_limit_val),
                dataset_size_category=dataset_size_category_val,
                download_dir=download_dir_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val,
                model=model_val,
                language=language_val,
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

            # 执行
            log_lines = []
            log_lines.append("=" * 60)
            log_lines.append("开始执行网页采集与转换工作流")
            log_lines.append("=" * 60)
            log_lines.append(f"目标: {req.target}")
            log_lines.append(f"类别: {req.category}")
            log_lines.append(f"下载目录: {req.download_dir}")
            log_lines.append("\n【网页采集配置】")
            log_lines.append(f"  - 搜索引擎: {search_engine_val}")
            log_lines.append(f"  - 任务最大循环次数: {max_crawl_cycles_per_task_val}")
            log_lines.append(f"  - 研究阶段最大循环次数: {max_crawl_cycles_for_research_val}")
            log_lines.append(f"  - 使用 Jina Reader: {'是' if use_jina_reader_val else '否'}")
            log_lines.append(f"  - 启用 RAG: {'是' if enable_rag_val else '否'}")
            log_lines.append(f"  - 并行页面数: {concurrent_pages_val}")
            log_lines.append(f"  - 禁用缓存: {'是' if disable_cache_val else '否'}")
            log_lines.append("\n【数据转换配置】")
            log_lines.append(f"  - 模型温度: {conversion_temperature_val}")
            log_lines.append(f"  - 最大 Token 数: {conversion_max_tokens_val}")
            log_lines.append(f"  - 最大采样长度: {conversion_max_sample_length_val}")
            log_lines.append(f"  - 采样记录数: {conversion_num_sample_records_val}")
            log_lines.append("=" * 60)

            final_state: DataCollectionState = await graph.ainvoke(state)

            log_lines.append("流程执行完成！")

            result = {
                "download_dir": req.download_dir,
                "processed_output": os.path.join(req.download_dir, "processed_output"),
                "category": req.category,
                "language": req.language,
                "chat_model": req.model,
            }

            return "\n".join(log_lines), result

        submit_btn.click(
            run_pipeline,
            inputs=[
                target,
                category,
                dataset_num_limit,
                dataset_size_category,
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


