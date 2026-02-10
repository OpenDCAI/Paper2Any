"""
运行 websearch_knowledge_store 工作流的脚本
===========================================================================

■ 功能简介
-----------
本脚本用于启动 "websearch_knowledge_store" 工作流。
该工作流以一组种子 URL 为起点，通过 **多 Agent 协作** 完成：
  1. 种子页面的正文提取与初步分析
  2. 自动规划「深度调研路线」
  3. 联网搜索、逐条执行调研计划
  4. 对全部原始数据进行清洗，产出结构化知识总结

最终产物存放于 raw_data_store/ 目录，包括 Markdown 知识文件与多媒体资源。


■ 工作流节点 & 流程
---------------------
整体流程由 LangGraph 条件边驱动，以 planner 为"中枢"：

  ┌─────────────────────────────────────────────────────┐
  │                       START                         │
  └────────────────────────┬────────────────────────────┘
                           ▼
                    ┌─────────────┐
              ┌────▶│   planner   │◀──────────────────┐
              │     │             │                    │
              │     └──────┬──────┘                    │
              │            │ 根据当前状态做条件路由      │
              │            ├───────────┐               │
              │            │           │               │
              │            ▼           ▼               │
              │  ┌──────────────┐ ┌──────────────┐     │
              │  │  initial_    │ │  web_        │     │
              │  │  analyzer    │ │  researcher  │     │
              │  │ (初始分析师) │ │ (webagent) │     │
              │  └──────┬───────┘ └──────┬───────┘     │
              │         │                │             │
              │         └───────┬────────┘             │
              │                 ▼                      │
              │          (回到 planner)                 │
              │                                        │
              │  当调研计划全部完成且数据尚未清洗 ──────▶│
              │                                        │
              │         ┌──────────────┐               │
              └─────────│ chief_curator│───────────────┘
                        │ (数据整理)    │
                        └──────┬───────┘
                               │
                               ▼
                          ┌─────────┐
                          │  _end_  │
                          └─────────┘

节点说明:
  1. planner
     - 管理任务队列 research_routes
     - 处理 Researcher 产出并入库 raw_data_store
     - 设定下一个执行任务 current_task
     - 根据状态做如下条件路由:
       A) research_routes 为空 & input_urls 有值 & 尚未分析  → initial_analyzer
       B) research_routes 非空（有未执行计划）                → web_researcher
       C) 计划全部完成 & raw_data_store 有数据但未清洗        → chief_curator
       D) 计划全部完成 & knowledge_base_summary 已生成        → _end_

  2. initial_analyzer (初始分析师)
     - 访问种子 URL，提取正文内容并保存 DOM 快照
     - 分析页面内容，规划出一系列深度调研路线 (research_routes)

  3. web_researcher (web_agent)
     - 依据 current_task 进行联网搜索探索
     - 抓取新网页内容，保存 DOM 与提取文本

  4. chief_curator (数据整理)
     - 读取 raw_data_store 全量数据
     - 对每条调研路线分别生成结构化 Markdown 知识文件
     - 汇总生成 knowledge_base_summary


■ 全局状态字段 (WebsearchKnowledgeState)
------------------------------------------
  - input_urls              : 用户初始输入的种子 URL 列表
  - research_routes         : 研究计划队列（会被 planner 逐步弹出）
  - original_research_routes: 原始研究路线（不变，供 curator 参照）
  - current_task            : 当前由 planner 分配给 researcher 的任务
  - raw_data_store          : 追加型列表，存储所有阶段抓取到的原始数据
  - knowledge_base_summary  : 最终清洗后的结构化知识总结


■ 可配置参数 (见下方 "可配置参数" 区域)
------------------------------------------
  INPUT_URLS   : 种子 URL 列表，可以是一篇技术博客、论文页面、文档等
  LANGUAGE     : 输出语言偏好，"zh" 为中文、"en" 为英文


■ 环境变量 (在 MainRequest 基类中读取)
------------------------------------------
  DF_API_URL   : LLM Chat API 的基础地址  (默认 "test")
  DF_API_KEY   : LLM API Key              (默认 "test")

  也可在下方 WebsearchKnowledgeRequest 实例中手动覆盖:
    req.chat_api_url = "https://..."
    req.api_key      = "sk-..."
    req.model        = "gpt-4o"


■ 依赖安装
-----------
  1. 安装项目基础依赖（在项目根目录下执行）:
       pip install -r requirements-base.txt

     其中与本工作流直接相关的关键依赖包括:
       - langgraph / langchain 系列  : 工作流引擎 & LLM 调用
       - playwright                  : 无头浏览器，用于抓取网页 DOM
       - beautifulsoup4              : HTML 解析与正文清洗
       - httpx                       : 异步 HTTP 客户端（调用 MineruHTML API 等）
       - trafilatura                 : 备选正文提取
       - openai                      : OpenAI 兼容 Chat API 调用

  2. 安装 Playwright 浏览器内核（首次需要）:
       playwright install chromium

     如果系统缺少依赖库，可追加:
       playwright install --with-deps chromium


■ MineruHTML 部署
------------------
  本工作流的 initial_analyzer 和 chief_curator 均依赖 MineruHTML
  （一个基于 LLM 微调的 HTML 正文提取服务）来从网页 DOM 中提取有效正文。

  默认 API 地址: http://localhost:7771  （可通过环境变量 MINERUHTML_API_URL 覆盖）
  API 端点:      POST /extract
  请求体:        { "html": "<完整 HTML 字符串>" }
  响应体:        { "main_html": "<提取出的正文 HTML>" }

  部署步骤:
    1) 克隆仓库:
         git clone https://github.com/opendatalab/MinerU-HTML.git
         cd MinerU-HTML

    2) 安装依赖与模型:
         pip install .

    3) 启动服务（默认监听 7771 端口）:
         python -m dripper.server \
    --model_path /path/to/your/model \
    --port 7771

    4) 验证服务是否就绪:
         curl -X POST http://localhost:7771/extract \
              -H "Content-Type: application/json" \
              -d '{"html": "<html><body><p>hello</p></body></html>"}'

  如果 MineruHTML 部署在其他机器或端口，通过环境变量指定:
    export MINERUHTML_API_URL="http://<host>:<port>"


■ 使用方法
-----------
  1. 激活 conda 环境:
       conda activate <your_env>

  2. 配置必需的环境变量（也可写入 .env 文件）:
       export DF_API_URL="https://api.openai.com/v1"   # LLM Chat API 地址
       export DF_API_KEY="sk-..."                       # LLM API Key
       # 可选:
       export THIRD_PARTY_MODEL="gpt-4o"                # 模型名称（默认 gpt-4o）
       export MINERUHTML_API_URL="http://localhost:7771" # MineruHTML 服务地址
       export HEADLESS="true"                            # 浏览器是否无头模式（默认 true）
       export HTTP_PROXY="http://127.0.0.1:7890"         # 代理（可选，最好设置代理）
       export HTTPS_PROXY="http://127.0.0.1:7890"

  3. 在项目根目录下运行:
       python script/run_websearch_knowledge_store.py

  4. 运行结束后，在 raw_data_store/ 目录下查看产出的知识文件。
"""

from __future__ import annotations

import asyncio

from dataflow_agent.state import WebsearchKnowledgeRequest, WebsearchKnowledgeState
from dataflow_agent.workflow import run_workflow


# ================== 可配置参数 ==================
# 种子 URL 列表 —— 工作流将从这些页面出发进行深度调研
# 可以替换为你感兴趣的任意网页 URL（支持多个）
INPUT_URLS: list[str] = [
    "https://zhuanlan.zhihu.com/p/624221952",
]

# 输出语言偏好: "zh" (中文) | "en" (英文) | 其他 BCP-47 语言代码
LANGUAGE: str = "zh"
# =================================================


async def run_websearch_knowledge_store_pipeline():
    """
    构造 WebsearchKnowledgeState，并运行 websearch_knowledge_store 工作流。

    流程:
      1) 根据上方可配置参数构造 Request 对象
      2) 用 Request 初始化全局 State
      3) 调用 run_workflow 启动 LangGraph 工作流
      4) 返回最终 State（包含 knowledge_base_summary 等产物）
    """
    # 1) 构造 Request —— 携带用户输入与 LLM 配置
    req = WebsearchKnowledgeRequest(
        language=LANGUAGE,
        input_urls=INPUT_URLS,
    )

    # 2) 初始化 State —— 将 input_urls 同步写入顶层字段
    state = WebsearchKnowledgeState(
        request=req,
        input_urls=req.input_urls,
    )

    # 3) 运行工作流 —— 名称需与 @register("websearch_knowledge_store") 一致
    final_state = await run_workflow(
        "websearch_knowledge_store", state
    )
    return final_state


def main() -> None:
    """
    同步入口: 运行异步主流程并打印关键结果。
    """
    final_state = asyncio.run(run_websearch_knowledge_store_pipeline())

    # ---------- 打印关键信息，便于快速查看结果 ----------
    print("\n=== WebsearchKnowledgeState ===")

    # 兼容处理: run_workflow 可能返回 dict 或 dataclass 对象
    if isinstance(final_state, dict):
        input_urls = final_state.get("input_urls", [])
        research_routes = final_state.get("research_routes", [])
        raw_data_store = final_state.get("raw_data_store", [])
        knowledge_base_summary = final_state.get("knowledge_base_summary", "")
    else:
        input_urls = getattr(final_state, "input_urls", [])
        research_routes = getattr(final_state, "research_routes", [])
        raw_data_store = getattr(final_state, "raw_data_store", [])
        knowledge_base_summary = getattr(final_state, "knowledge_base_summary", "")

    print(f"input_urls          : {input_urls}")
    print(f"research_routes     : {research_routes}")
    print(f"raw_data_store size : {len(raw_data_store) if raw_data_store else 0}")
    print("knowledge_base_summary (截取前 500 字):")
    if knowledge_base_summary:
        print(knowledge_base_summary[:500])
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
