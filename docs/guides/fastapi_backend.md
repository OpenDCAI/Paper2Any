# FastAPI 后端与 Workflow 适配说明

本篇文档介绍 `fastapi_app` 包的整体设计、执行逻辑，以及 **如何将 `dataflow_agent.workflow.*` 中的工作流（Workflow）封装为 HTTP API**，并给出扩展与贡献方式。

适合读者：

- 已读过[项目架构概览](architecture.md)，想了解后端 API 部分
- 想把新的 Workflow 暴露为 HTTP 接口
- 想从 Gradio 前端迁移到统一的 RESTful 接口

---

## 1. 总体架构概览

后端总体分层可以简化为：

```text
HTTP 请求
   ↓
FastAPI 路由 (fastapi_app.routers.*)
   ↓
Workflow 适配层 (fastapi_app.workflow_adapters)
   ↓
核心 Workflow (dataflow_agent.workflow.*)
   ↓
DFRequest / DFState / Graph 执行
```

其中：

- **FastAPI 路由层**：负责 HTTP 协议、URL 路由、请求体解析、响应序列化。
- **Workflow 适配层 (`workflow_adapters.py`)**：负责“HTTP 模型 ↔ 内部 Workflow 状态”的转换。
- **Workflow 本体 (`dataflow_agent.workflow.*`)**：实现具体的业务逻辑和算法流程。
- **Pydantic 模型 (`schemas.py`)**：定义请求和响应的数据结构，供 FastAPI 自动校验和文档生成。

`fastapi_app` 的目标是：**不修改核心 Workflow，实现“即插即用”的 API 封装**。

---

## 2. FastAPI 应用与路由挂载（`main.py`）

### 2.1 应用创建

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_app.routers import operator_write, pipeline_rec, workflows


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例。

    这里只做基础框架搭建：
    - CORS 配置
    - 路由挂载
    """
    app = FastAPI(
        title="DataFlow Agent FastAPI Backend",
        version="0.1.0",
        description="HTTP API wrapper for dataflow_agent.workflow.* pipelines",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由挂载
    app.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
    app.include_router(operator_write.router, prefix="/operator", tags=["operator_write"])
    app.include_router(pipeline_rec.router, prefix="/pipeline", tags=["pipeline_recommend"])

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


# 供 uvicorn 使用：uvicorn fastapi_app.main:app --reload
app = create_app()
```

核心要点：

- `create_app()` 只负责：
  - 初始化 `FastAPI` 应用（标题、版本、描述）
  - 配置 CORS（允许任意来源、方法和头，方便前端跨域访问）
  - 挂载子路由模块（`/workflows`、`/operator`、`/pipeline`）
  - 提供 `/health` 健康检查接口
- 通过全局变量 `app = create_app()`，可以直接使用：

```bash
uvicorn fastapi_app.main:app --reload
```

启动后端服务。

---

## 3. Pydantic 模型定义（`schemas.py`）

`schemas.py` 通过 Pydantic 的 `BaseModel` 定义 HTTP 请求/响应结构，FastAPI 会据此自动：

- 解析请求体 JSON → Python 对象
- 校验字段类型/默认值
- 生成 OpenAPI / Swagger 文档

### 3.1 通用错误模型

```python
class APIError(BaseModel):
    code: str
    message: str
```

用于统一返回错误信息（目前代码中未强制使用，未来可扩展）。

### 3.2 算子编写相关模型

#### `OperatorWriteRequest`

```python
class OperatorWriteRequest(BaseModel):
    """
    对标 gradio_app.pages.operator_write 中 generate_operator 的输入参数。
    """
    target: str
    category: str = "Default"
    json_file: Optional[str] = None

    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    api_key: str = ""
    model: str = "gpt-4o"
    language: str = "en"

    need_debug: bool = False
    max_debug_rounds: int = 3

    # 若提供，则用于保存生成的算子代码
    output_path: Optional[str] = None
```

字段含义：

- `target`：算子需求/功能描述（自然语言）。
- `category`：算子类别（如 `"Default"`、`"filter"` 等），用于 workflow 内部分类处理。
- `json_file`：测试数据文件（JSONL），为空时会使用默认路径。
- `chat_api_url` / `api_key` / `model`：LLM 调用配置。
- `language`：工作语言（如 `en` / `zh`）。
- `need_debug` / `max_debug_rounds`：是否需要自动调试及最大调试轮数。
- `output_path`：若指定，则将生成的算子代码写入该文件路径。

> 此模型与 Gradio 前端的 `generate_operator` 参数保持一致，便于功能迁移。

#### `OperatorWriteResponse`

```python
class OperatorWriteResponse(BaseModel):
    success: bool
    code: str
    matched_ops: List[str]
    execution_result: Dict[str, Any]
    debug_runtime: Dict[str, Any]
    agent_results: Dict[str, Any]
    log: str
```

- `success`：整体调用是否成功（当前实现始终为 True，可根据需求增强）。
- `code`：生成的算子代码字符串。
- `matched_ops`：workflow 匹配到的相关算子名称列表。
- `execution_result`：执行/调试算子时的执行结果（如 `success`、`stderr` 等）。
- `debug_runtime`：调试过程中的中间信息（输入 key、stdout/stderr 等）。
- `agent_results`：workflow 内部各 Agent 的原始结果。
- `log`：人类可读的多行文本日志，方便前端直接展示。

### 3.3 流水线推荐 / 导出相关模型

#### `PipelineRecommendRequest`

```python
class PipelineRecommendRequest(BaseModel):
    """
    对标 gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow 的参数。
    """
    target: str
    json_file: str

    need_debug: bool = False
    session_id: str = "default"

    chat_api_url: str = "http://123.129.219.111:3000/v1/"
    api_key: str = ""
    model_name: str = "gpt-4o"
    max_debug_rounds: int = 2

    chat_api_url_for_embeddings: str = ""
    embedding_model_name: str = "text-embedding-3-small"
    update_rag_content: bool = True
```

- `target`：希望自动构建的 Pipeline 目标描述。
- `json_file`：用于测试的输入数据文件。
- `session_id`：会话标识，用于区分不同用户/任务。
- `chat_api_url` / `api_key` / `model_name`：主 LLM 调用配置。
- `chat_api_url_for_embeddings` / `embedding_model_name`：向量模型配置。
- `update_rag_content`：是否更新 RAG 索引内容。
- `need_debug` / `max_debug_rounds`：调试模式与轮次数量。

#### `PipelineRecommendResponse`

```python
class PipelineRecommendResponse(BaseModel):
    success: bool
    python_file: str
    execution_result: Dict[str, Any]
    agent_results: Dict[str, Any]
```

- `success`：是否成功生成推荐 Pipeline。
- `python_file`：生成的 `pipeline.py` 文件路径。
- `execution_result`：这里与原实现对齐，使用 workflow 的 `debug_history`。
- `agent_results`：workflow 内部各 Agent 的执行结果。

---

## 4. Workflow 适配层（`workflow_adapters.py`）

这是“**Workflow → 后端 API**”的关键桥接代码，主要参考了 Gradio 版本的封装：

- `gradio_app.pages.operator_write.run_operator_write_pipeline`
- `gradio_app.utils.wf_pipeine_rec.run_pipeline_workflow`

职责可以概括为：

1. 将 HTTP 请求模型转换为内部的 `DFRequest` / `DFState`
2. 调用对应的 Workflow 图构建函数（`create_*_graph`）
3. 执行图（`graph.ainvoke(...)`）
4. 从最终状态中提取结果字段
5. 封装为 Pydantic 响应模型返回

### 4.1 依赖与公共工具

```python
import base64
import os
from pathlib import Path
from typing import Any, Dict, List

from dataflow_agent.logger import get_logger
from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.utils import get_project_root
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import (
    create_pipeline_graph,
)
from dataflow_agent.workflow.wf_pipeline_write import create_operator_write_graph

from .schemas import (
    OperatorWriteRequest,
    OperatorWriteResponse,
    PipelineRecommendRequest,
    PipelineRecommendResponse,
)

log = get_logger(__name__)
```

- `DFRequest`：一次请求的配置与输入。
- `DFState`：Workflow 运行时状态，包含 `request`、`messages`、`temp_data` 等。
- `create_operator_write_graph` / `create_pipeline_graph`：构建不同业务场景的 Workflow 图。
- `get_project_root()`：获取项目根目录，用于拼接基本路径。

---

### 4.2 算子编写工作流封装：`run_operator_write_pipeline_api`

函数签名：

```python
async def run_operator_write_pipeline_api(
    req: OperatorWriteRequest,
) -> OperatorWriteResponse:
    ...
```

#### 4.2.1 环境变量与默认值处理

```python
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
    else:
        # 若未显式提供，则回落到环境变量或一个 dummy key
        req.api_key = os.getenv("DF_API_KEY", "sk-dummy")
```

- 若请求中提供了 `api_key`，则写入环境变量 `DF_API_KEY`，供下游 LLM 调用使用。
- 否则从环境中读取 `DF_API_KEY`，找不到则使用 `"sk-dummy"` 作为兜底。

```python
    projdir = get_project_root()
    json_file = req.json_file or f"{projdir}/tests/test.jsonl"
```

- `json_file` 未指定时，默认使用项目根目录下 `tests/test.jsonl`。

#### 4.2.2 构造 `DFRequest` 和初始 `DFState`

```python
    df_req = DFRequest(
        language=req.language,
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model,
        target=req.target,
        need_debug=req.need_debug,
        max_debug_rounds=req.max_debug_rounds,
        json_file=json_file,
    )
    state = DFState(request=df_req, messages=[])
```

- `DFRequest` 把所有必要的 HTTP 参数（LLM 配置、业务输入、调试设置、数据文件路径）统一包装。
- `DFState` 初始状态携带 `request` 和空的 `messages` 列表。

#### 4.2.3 通过 `temp_data` 注入额外上下文

```python
    # 设置输出路径（如果提供）
    if req.output_path:
        state.temp_data["pipeline_file_path"] = req.output_path

    # 设置类别
    if req.category:
        state.temp_data["category"] = req.category

    # 初始化调试轮次
    state.temp_data["round"] = 0
```

- `pipeline_file_path`：用于指定生成算子代码的输出文件路径。
- `category`：算子类别，影响 workflow 的分支逻辑。
- `round`：调试轮次计数，从 0 开始。

> `DFState.temp_data` 是一个自由扩展的 dict，用于携带 workflow 内部使用的临时字段。

#### 4.2.4 构建并执行 Workflow 图

```python
    graph = create_operator_write_graph().build()
    # 递归限制与 Gradio 版本保持一致：主链 4 步 + 每轮 5 步 * 轮次 + buffer 5
    recursion_limit = 4 + 5 * req.max_debug_rounds + 5
    final_state = await graph.ainvoke(
        state,
        config={"recursion_limit": recursion_limit},
    )
```

- `create_operator_write_graph()`：返回一个图构建器对象。
- `.build()`：构建出可执行图。
- `recursion_limit`：控制图执行的最大步数（防止死循环），与 Gradio 版本保持一致：
  - 主流程 4 步
  - 每轮调试估算 5 步
  - 预留 buffer 5 步
- `graph.ainvoke(state, config=...)`：异步执行，返回 `final_state`（可能是 `DFState` 或 `dict`）。

#### 4.2.5 从 `final_state` 抽取结果字段

代码对多种返回形式做了兼容和降级，例如提取 `matched_ops`：

```python
    try:
        if isinstance(final_state, dict):
            matched = final_state.get("matched_ops", [])
            if not matched:
                matched = (
                    final_state.get("agent_results", {})
                    .get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        else:
            matched = getattr(final_state, "matched_ops", [])
            if not matched and hasattr(final_state, "agent_results"):
                matched = (
                    final_state.agent_results.get("match_operator", {})
                    .get("results", {})
                    .get("match_operators", [])
                )
        matched_ops = list(matched or [])
    except Exception as e:  # pragma: no cover - 仅日志
        log.warning(f"[operator_write] 提取匹配算子失败: {e}")
        matched_ops = []
```

- 先检查 `final_state` 是否为 `dict`，否则当作对象处理（用 `getattr`）。
- 当顶层字段没有时，尝试从 `agent_results["match_operator"]["results"]["match_operators"]` 中兜底。
- 任一环节出错只打 warning，不中断接口。

类似逻辑用于提取：

- 生成代码 `pipeline_code`（在 `temp_data` 中）
- 执行结果 `execution_result`
- 调试运行信息 `debug_runtime`
- `agent_results` 整体

#### 4.2.6 构造可读日志

```python
    log_lines: List[str] = []
    log_lines.append("==== 算子编写结果 ====")
    log_lines.append(f"\n匹配到的算子数量: {len(matched_ops)}")
    if matched_ops:
        log_lines.append(f"匹配的算子: {matched_ops}")

    log_lines.append(f"\n生成的代码长度: {len(code_str)} 字符")

    if execution_result:
        success_flag = execution_result.get("success", False)
        log_lines.append(f"\n执行成功: {success_flag}")
        ...
```

- 将关键结果（匹配算子数量、代码长度、执行是否成功、stderr/stdout 片段、调试信息）串成多行文本 `log_text`。
- 前端可以直接以日志组件/文本展示。

#### 4.2.7 返回响应模型

```python
    return OperatorWriteResponse(
        success=True,
        code=code_str or "",
        matched_ops=matched_ops,
        execution_result=execution_result,
        debug_runtime=debug_runtime,
        agent_results=agent_results,
        log=log_text,
    )
```

- 将所有提取结果打包进 `OperatorWriteResponse`，由 FastAPI 完成 JSON 序列化。

---

### 4.3 流水线推荐工作流封装：`run_pipeline_recommend_api`

函数签名：

```python
async def run_pipeline_recommend_api(
    req: PipelineRecommendRequest,
) -> PipelineRecommendResponse:
    ...
```

#### 4.3.1 环境变量与会话目录

```python
    if req.api_key:
        os.environ["DF_API_KEY"] = req.api_key
        os.environ["DF_API_URL"] = req.chat_api_url
```

- 设置 LLM 相关环境变量。

```python
    project_root: Path = get_project_root()
    tmps_dir: Path = project_root / "dataflow_agent" / "tmps"

    # 对 session_id 做一次 URL-safe 的 base64 编码，确保目录名安全
    session_id_encoded = base64.urlsafe_b64encode(req.session_id.encode()).decode()
    session_dir: Path = tmps_dir / session_id_encoded
    session_dir.mkdir(parents=True, exist_ok=True)

    python_file_path = session_dir / "pipeline.py"
```

- 为每个 `session_id` 建立独立目录，名称采用 URL-safe Base64 编码，避免非法字符。
- Workflow 生成的 `pipeline.py` 会写入该目录。

#### 4.3.2 构造 `DFRequest` 和状态

```python
    df_req = DFRequest(
        language="en",
        chat_api_url=req.chat_api_url,
        api_key=req.api_key,
        model=req.model_name,
        json_file=req.json_file,
        target=req.target,
        python_file_path=str(python_file_path),
        need_debug=req.need_debug,
        session_id=session_id_encoded,
        max_debug_rounds=req.max_debug_rounds,
        chat_api_url_for_embeddings=req.chat_api_url_for_embeddings,
        embedding_model_name=req.embedding_model_name,
        update_rag_content=req.update_rag_content,
    )

    state = DFState(request=df_req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = True
```

- 传入 workflow 所需的所有字段，包括 Python 文件路径、RAG 和嵌入模型配置等。
- `state.debug_mode = True`：启用 workflow 内部的 debug 记录逻辑（如 `debug_history`）。

#### 4.3.3 构建并执行 Workflow 图

```python
    graph = create_pipeline_graph().build()
    final_state = await graph.ainvoke(state)
```

- 与算子编写类似，只是不设置 `recursion_limit`（默认配置足够）。

#### 4.3.4 提取 `debug_history` 与 `agent_results`

```python
    if isinstance(final_state, dict):
        debug_history = dict(final_state.get("debug_history", {}) or {})
        agent_results = dict(final_state.get("agent_results", {}) or {})
    else:
        debug_history = dict(getattr(final_state, "debug_history", {}) or {})
        agent_results = dict(getattr(final_state, "agent_results", {}) or {})
```

- 将 workflow 的调试历史作为 `execution_result` 返回。

```python
    return PipelineRecommendResponse(
        success=True,
        python_file=str(df_req.python_file_path),
        execution_result=debug_history,
        agent_results=agent_results,
    )
```

- 最终响应模型中：
  - `python_file`：可直接读取生成的 `pipeline.py`。
  - `execution_result`：流程执行历史，便于前端展示。

---

## 5. 从 Workflow 到后端 API 的通用模式

总结一下，将 `dataflow_agent.workflow.*` 中的任意 workflow 暴露为 HTTP API，基本遵循下面 5 步：

1. **选定 workflow 构建函数**
   - 例如：
     - 算子编写：`wf_pipeline_write.create_operator_write_graph`
     - 流水线推荐：`wf_pipeline_recommend_extract_json.create_pipeline_graph`
   - 这些函数通常返回一个“图构建器”，需要 `.build()` 得到可执行图。

2. **梳理 workflow 输入与输出**
   - 查看 workflow 对 `DFRequest` 的依赖字段：
     - 如 `target`、`json_file`、`model`、`python_file_path`、`session_id` 等。
   - 查看 `DFState` 中需要/产生的字段：
     - 如 `temp_data["pipeline_code"]`、`debug_history`、`agent_results` 等。
   - 明确：
     - 哪些是 HTTP **请求参数**
     - 哪些是 HTTP **响应字段**

3. **在 `schemas.py` 中定义 Pydantic 模型**
   - Request 模型：对应 workflow 输入配置。
   - Response 模型：只暴露对调用方有价值的结果（可包含日志）。

4. **在 `workflow_adapters.py` 中写适配函数**
   - 典型模板：
     1. 接收 Request 模型实例
     2. 设置必要环境变量（如 `DF_API_KEY`）
     3. 构造 `DFRequest`、`DFState`，初始化 `temp_data` / `debug_mode`
     4. 调用 `create_xxx_graph().build().ainvoke(...)`
     5. 兼容 dict/对象两种返回形式，提取所需字段
     6. 构造日志字符串 `log_text`（可选）
     7. 返回 Response 模型

5. **在 `fastapi_app.routers` 中增加相应路由**
   - 例如新增 `new_feature`：

     ```python
     # fastapi_app/routers/new_feature.py
     from fastapi import APIRouter
     from fastapi_app.schemas import NewFeatureRequest, NewFeatureResponse
     from fastapi_app.workflow_adapters import run_new_feature_api

     router = APIRouter()

     @router.post("/run", response_model=NewFeatureResponse)
     async def run_new_feature(req: NewFeatureRequest):
         return await run_new_feature_api(req)
     ```

   - 再在 `main.py` 中挂载：

     ```python
     from fastapi_app.routers import new_feature

     app.include_router(new_feature.router, prefix="/new_feature", tags=["new_feature"])
     ```

> 这样即可在不改动 workflow 本体的情况下，对外新增一个 REST API。

---

## 6. 如何在后端扩展与贡献

### 6.1 新增基于 Workflow 的接口

假设你编写了一个新 workflow：`dataflow_agent.workflow.wf_my_feature.create_my_feature_graph`，想暴露为 HTTP API，可以按下面步骤：

1. **阅读 workflow 实现**
   - 找出使用到的 `DFRequest` 字段和 `DFState.temp_data`、`agent_results`、`debug_history` 等。
   - 可以参考现有两个 adapter 的写法。

2. **在 `schemas.py` 中新增请求/响应模型**

   ```python
   class MyFeatureRequest(BaseModel):
       target: str
       json_file: str
       # ... 其他 workflow 所需字段

   class MyFeatureResponse(BaseModel):
       success: bool
       result: Dict[str, Any]
       log: str
   ```

3. **在 `workflow_adapters.py` 中新增适配函数**

   ```python
   async def run_my_feature_api(req: MyFeatureRequest) -> MyFeatureResponse:
       # 1. 环境变量
       if req.api_key:
           os.environ["DF_API_KEY"] = req.api_key

       # 2. DFRequest / DFState
       df_req = DFRequest(
           target=req.target,
           json_file=req.json_file,
           # ...
       )
       state = DFState(request=df_req, messages=[])
       state.temp_data["round"] = 0

       # 3. 构建并执行图
       graph = create_my_feature_graph().build()
       final_state = await graph.ainvoke(state)

       # 4. 提取结果（兼容 dict / 对象）
       if isinstance(final_state, dict):
           result = dict(final_state.get("result", {}) or {})
       else:
           result = dict(getattr(final_state, "result", {}) or {})

       # 5. 构造日志
       log_text = "..."  # 根据需要组织

       return MyFeatureResponse(
           success=True,
           result=result,
           log=log_text,
       )
   ```

4. **在 `fastapi_app/routers` 新建或扩展路由文件**
   - 将该适配函数暴露为 HTTP API，挂载到合适的前缀路径和 tag 下。

5. **编写/更新测试与文档**
   - 使用 FastAPI `TestClient` 或 `httpx` 写集成测试。
   - 在 `docs/guides` 下补充对应功能说明，或在本篇文档基础上添加章节。

### 6.2 改进现有接口的常见方向

- **增强错误处理**
  - 当前 `success` 字段多为固定 True，可根据 `execution_result["success"]` 等实际结果设置。
  - 统一使用 `APIError` 结构返回错误，或抛出 `HTTPException`。
- **统一日志构造**
  - 将构建 `log_text` 的逻辑抽取为工具函数，避免重复代码。
- **更强类型约束**
  - 对 `agent_results` / `debug_history` 等结构设计更细的 Pydantic 模型，减少 `Dict[str, Any]` 的使用。
- **增加权限与限流**
  - 在 FastAPI 层增加认证、鉴权或限流中间件，根据实际部署环境配置。

---

## 7. 与 Gradio 前端的关系

目前仓库中 Web 交互主要有两条路径：

1. **Gradio 前端（`gradio_app`）**
   - 直接在 Python 内部调用 workflow/adapter 函数，主要为互动式开发与调试设计。
   - 文档可参考：[算子编写指南](operator_write.md)。

2. **FastAPI 后端（`fastapi_app`）**
   - 把同样的 workflow 能力封装为 HTTP API，适合：
     - 与其他服务对接（微服务架构）
     - 提供统一的 REST 接口给前端/脚本调用
     - 融入现有后端系统

两者共享同一套核心：`dataflow_agent.workflow.*`。后续如果新增/修改 workflow，建议同时评估：

- 是否需要在 Gradio 前端暴露交互页面？
- 是否需要在 FastAPI 后端暴露对应 REST API？

---

## 8. 小结

- `fastapi_app` 的核心职责是：**用统一模式将 `dataflow_agent.workflow.*` 暴露为 HTTP API**。
- `schemas.py`：定义 HTTP 层的请求/响应结构。
- `workflow_adapters.py`：负责 HTTP 模型与内部 Workflow 状态的互相转换。
- `main.py` 与 `fastapi_app.routers.*`：构成标准 FastAPI 路由层，对外提供 REST 接口。
- 若要贡献新的后端接口，建议：
  1. 先设计/实现新的 workflow；
  2. 在 `schemas.py` 中设计 Pydantic 模型；
  3. 在 `workflow_adapters.py` 中编写适配函数；
  4. 在 `routers` 中挂载路由，并补充测试与文档。

通过复用本篇中两个现有适配函数的模式，可以在不侵入核心逻辑的前提下，快速扩展整个系统的 HTTP 能力。
