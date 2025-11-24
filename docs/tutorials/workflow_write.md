# 📋 Workflow 编写教程

本教程将指导你如何在 DataFlow-Agent 项目中创建和编写 Workflow（工作流）。

## 目录

1. [准备工作](#1-准备工作)
2. [使用 CLI 快速创建](#2-使用-cli-快速创建)
3. [Workflow 基础结构](#3-workflow-基础结构)
4. [State 和 Request 定义](#4-state-和-request-定义)
5. [Agent 创建方式](#5-agent-创建方式)
6. [工具系统](#6-工具系统)
7. [节点编写](#7-节点编写)
8. [图构建](#8-图构建)
9. [完整示例](#9-完整示例)
10. [调试和测试](#10-调试和测试)
11. [最佳实践](#11-最佳实践)

---

## 1. 准备工作

### 1.1 了解项目结构

```
dataflow_agent/
├── workflow/          # Workflow 定义目录
│   ├── wf_*.py       # Workflow 文件（必须以 wf_ 开头）
│   └── registry.py   # Workflow 注册中心
├── agentroles/       # Agent 角色定义
├── state.py          # State 和 Request 定义
├── graphbuilder/     # 图构建器
├── templates/        # 代码生成模板
└── cli.py           # CLI 命令工具
```

### 1.2 核心概念

- **Workflow**: 由多个节点（Node）和边（Edge）组成的有向图
- **State**: 在 Workflow 中流转的状态对象，包含所有上下文信息
- **Request**: State 中的请求对象，包含用户输入和配置
- **Node**: Workflow 中的处理单元，通常调用 Agent 执行任务
- **Agent**: 具有特定角色的 LLM 调用封装，负责完成具体任务
- **Tool**: Agent 可以使用的工具函数（前置工具和后置工具）

### 1.3 命名规范

- **Workflow 文件**: 必须以 `wf_` 开头，如 `wf_my_workflow.py`
- **注册名称**: 去掉 `wf_` 前缀，如 `wf_pipeline_write.py` 注册为 `"pipeline_write"`
- **State 类**: 以 `State` 结尾，如 `MyWorkflowState`
- **Request 类**: 以 `Request` 结尾，如 `MyWorkflowRequest`

---

## 2. 使用 CLI 快速创建

DataFlow-Agent 提供了 `dfa create` 命令来快速生成各种组件。

### 2.1 创建 Workflow

```bash
# 创建一个新的 workflow
dfa create --wf_name my_workflow

# 这将生成：
# - dataflow_agent/workflow/wf_my_workflow.py  (workflow 源码)
# - tests/test_my_workflow.py                  (测试文件)
```

### 2.2 创建 Agent

```bash
# 创建一个新的 agent
dfa create --agent_name my_agent

# 这将生成：
# - dataflow_agent/agentroles/my_agent_agent.py
```

### 2.3 创建 State

```bash
# 创建自定义 State 和 Request
dfa create --state_name my_workflow

# 这将生成：
# - dataflow_agent/states/my_workflow_state.py
```

### 2.4 创建 Prompt Template

```bash
# 创建 prompt 模板
dfa create --prompt_name my_agent

# 这将生成：
# - dataflow_agent/promptstemplates/resources/pt_my_agent_repo.py
```

### 2.5 其他创建选项

```bash
# 创建 Gradio 页面
dfa create --gradio_name my_page

# 创建 Agent-as-Tool
dfa create --agent_as_tool_name my_tool_agent
```

---

## 3. Workflow 基础结构

### 3.1 基本模板

```python
"""
my_workflow workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能描述：简要说明工作流功能
"""

from __future__ import annotations
from dataflow_agent.state import MainState  # 或你的自定义 State
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("my_workflow")  # 注册名称（不含 wf_ 前缀）
def create_my_workflow_graph() -> GenericGraphBuilder:
    """
    Workflow 工厂函数
    
    通过 dfa run --wf my_workflow 调用
    """
    builder = GenericGraphBuilder(
        state_model=MainState,      # State 类型
        entry_point="step1"         # 入口节点名称
    )
    
    # ========== 工具定义 ==========
    # 在这里定义前置工具和后置工具
    
    # ========== 节点定义 ==========
    # 在这里定义节点函数
    
    # ========== 图构建 ==========
    # 在这里添加节点和边
    
    return builder
```

### 3.2 注册机制

使用 `@register` 装饰器将 Workflow 注册到全局注册中心：

```python
from dataflow_agent.workflow.registry import register

@register("my_workflow")  # 注册名称
def create_my_workflow_graph() -> GenericGraphBuilder:
    # ...
    pass
```

注册后可以通过以下方式调用：

```python
from dataflow_agent.workflow import run_workflow

# 运行 workflow
result = await run_workflow("my_workflow", state)
```

---

## 4. State 和 Request 定义

### 4.1 继承体系

所有 State 和 Request 都应该继承自基类：

```python
from dataclasses import dataclass, field
from dataflow_agent.state import MainState, MainRequest

# ==================== Request ====================
@dataclass
class MyWorkflowRequest(MainRequest):
    """自定义请求参数"""
    # 继承 MainRequest 的所有字段：
    # - language: str = "en"
    # - chat_api_url: str
    # - api_key: str
    # - model: str = "gpt-4o"
    # - target: str = ""
    
    # 添加自定义字段
    input_data: str = ""
    config_param: str = "default"
    max_retries: int = 3

# ==================== State ====================
@dataclass
class MyWorkflowState(MainState):
    """自定义状态"""
    # 继承 MainState 的所有字段：
    # - request: MainRequest
    # - messages: List[BaseMessage]
    # - agent_results: Dict[str, Any]
    # - temp_data: Dict[str, Any]
    
    # 重写 request 类型
    request: MyWorkflowRequest = field(default_factory=MyWorkflowRequest)
    
    # 添加自定义字段
    processing_result: dict = field(default_factory=dict)
    current_step: str = "start"
    intermediate_data: list = field(default_factory=list)
```

### 4.2 使用 CLI 创建

```bash
# 快速创建 State 和 Request
dfa create --state_name my_workflow

# 生成的文件会包含基本结构，你只需添加自定义字段
```

### 4.3 State 字段说明

**MainRequest 核心字段**:
- `language`: 用户偏好语言（"en" | "zh"）
- `chat_api_url`: LLM API 端点
- `api_key`: API 密钥
- `model`: 模型名称（如 "gpt-4o"）
- `target`: 任务描述

**MainState 核心字段**:
- `request`: 请求对象
- `messages`: 消息历史（LangChain 格式）
- `agent_results`: Agent 执行结果字典
- `temp_data`: 临时数据存储

---

## 5. Agent 创建方式

### 5.1 新的策略模式

项目引入了策略模式，提供多种便捷的 Agent 创建函数：

```python
from dataflow_agent.agentroles import (
    create_simple_agent,   # 简单模式
    create_react_agent,    # ReAct 模式（带验证）
    create_graph_agent,    # 图模式（带工具调用）
    create_vlm_agent,      # 视觉语言模型模式
)
```

### 5.2 简单模式 Agent

适用于单次 LLM 调用的场景：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """使用简单模式创建 Agent"""
    agent = create_simple_agent(
        name="my_agent",           # Agent 名称（必须已注册）
        model_name="gpt-4o",       # 模型名称
        temperature=0.7,           # 采样温度 (0.0-1.0)
        max_tokens=4096,           # 最大 token 数
        parser_type="json",        # 解析器类型: "json" | "xml" | "text"
    )
    
    # 执行 Agent
    state = await agent.execute(state)
    
    # 获取结果
    result = state.agent_results.get("my_agent", {}).get("results", {})
    
    return state
```

### 5.3 ReAct 模式 Agent

适用于需要验证和重试的场景：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """使用 ReAct 模式创建 Agent"""
    agent = create_react_agent(
        name="my_agent",
        model_name="gpt-4o",
        temperature=0.1,
        max_retries=3,             # 最大重试次数
        parser_type="json",
        # validators=[...],        # 可选：自定义验证器
    )
    
    state = await agent.execute(state)
    return state
```

### 5.4 图模式 Agent

适用于需要调用工具的场景：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """使用图模式创建 Agent（支持工具调用）"""
    from dataflow_agent.toolkits.tool_manager import get_tool_manager
    
    agent = create_graph_agent(
        name="my_agent",
        model_name="gpt-4o",
        temperature=0.2,
        tool_mode="auto",          # 工具调用模式: "auto" | "none" | "required"
    )
    
    state = await agent.execute(state)
    return state
```

### 5.5 VLM 模式 Agent

适用于处理图像的场景：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """使用 VLM 模式创建 Agent"""
    agent = create_vlm_agent(
        name="my_agent",
        model_name="gpt-4-vision-preview",
        temperature=0.1,
        vlm_mode="understanding",      # 'understanding' | 'generation' | 'edit'
        image_detail="high",           # 'low' | 'high' | 'auto'
        max_image_size=(2048, 2048),
    )
    
    state = await agent.execute(state)
    return state
```

### 5.6 并行模式 Agent

适用于需要批量处理多条数据的场景：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """使用并行模式创建 Agent"""
    agent = create_parallel_agent(
        name="my_agent",
        model_name="gpt-4o",
        temperature=0.3,
        concurrency_limit=3,           # 并行度限制，默认5
        parser_type="json",
    )
    
    state = await agent.execute(state)
    return state
```

#### 并行模式的数据准备

并行模式会自动检测前置工具结果中的列表数据，支持三种方式：

**方式 1: 前置工具直接返回列表**

```python
@builder.pre_tool("items", "my_agent")
def get_items(state: MyWorkflowState):
    """返回需要并行处理的数据列表"""
    return [
        {"text": "第一条数据", "id": 1},
        {"text": "第二条数据", "id": 2},
        {"text": "第三条数据", "id": 3},
    ]
```

**方式 2: 使用 parallel_items 字段**

```python
@builder.pre_tool("data", "my_agent")
def get_data(state: MyWorkflowState):
    """返回包含 parallel_items 的字典"""
    return {
        "parallel_items": [
            {"text": "数据1"},
            {"text": "数据2"},
        ],
        "context": "共享上下文信息"  # 会被所有并行任务共享
    }
```

**方式 3: 任意列表字段（自动检测）**

```python
@builder.pre_tool("batch_data", "my_agent")
def get_batch_data(state: MyWorkflowState):
    """返回包含列表字段的字典"""
    return {
        "items": [  # 会被自动检测为并行数据
            {"name": "item1"},
            {"name": "item2"},
        ],
        "config": {"mode": "fast"}  # 会被所有并行任务共享
    }
```

#### 并行结果处理

并行模式执行后，结果会包含在 `parallel_results` 字段中：

```python
async def process_parallel_results(state: MyWorkflowState) -> MyWorkflowState:
    """处理并行执行结果"""
    result = state.agent_results.get("my_agent", {}).get("results", {})
    
    # 获取所有并行结果
    parallel_results = result.get("parallel_results", [])
    total_processed = result.get("total_processed", 0)
    
    log.info(f"共处理 {total_processed} 条数据")
    
    # 聚合结果
    aggregated = {
        "success_count": sum(1 for r in parallel_results if not r.get("error")),
        "error_count": sum(1 for r in parallel_results if r.get("error")),
        "results": parallel_results
    }
    
    state.temp_data["aggregated_results"] = aggregated
    return state
```

### 5.6 使用配置对象

也可以使用配置对象创建 Agent：

```python
from dataflow_agent.agentroles import create_agent
from dataflow_agent.agentroles.cores.configs import SimpleConfig, ReactConfig

# 方式 1: 使用 SimpleConfig
config = SimpleConfig(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    parser_type="json",
)
agent = create_agent(name="my_agent", config=config)

# 方式 2: 使用 ReactConfig
config = ReactConfig(
    model_name="gpt-4o",
    temperature=0.1,
    max_retries=3,
    parser_type="json",
)
agent = create_agent(name="my_agent", config=config)
```

---

## 6. 工具系统

> ⚠️ **重要提示**：`@builder.pre_tool()` 和 `@builder.post_tool()` 装饰器的第二个参数（role）应该填写 **Agent 的 role_name**，而不是 workflow 中的节点名称！这是因为 Agent 在执行时会通过 `self.role_name` 从 ToolManager 中获取对应的工具。

### 6.1 前置工具（Pre-Tool）

前置工具在 Agent 执行前运行，用于收集上下文信息并注入到 prompt 中。

#### 定义前置工具

```python
@builder.pre_tool("placeholder_name", "agent_role_name")  # ← 第二个参数是 Agent 的 role_name
def my_pre_tool(state: MyWorkflowState):
    """
    前置工具函数
    
    Args:
        state: 当前状态对象
        
    Returns:
        任何可序列化的数据（字符串、列表、字典等）
    """
    return state.request.input_data
```

#### 参数说明

- **第一个参数**: Prompt 模板中的占位符名称
- **第二个参数**: **Agent 的 role_name**（不是节点名称！）

#### 正确示例 ✅

```python
# 假设你的 Agent 的 role_name 是 "processor"
async def process_node(state: MyWorkflowState) -> MyWorkflowState:
    agent = create_simple_agent(
        name="processor",  # ← 这是 Agent 的 role_name
        model_name="gpt-4o",
    )
    state = await agent.execute(state)
    return state

# 前置工具应该绑定到 Agent 的 role_name
@builder.pre_tool("user_input", "processor")  # ✅ 使用 Agent 的 role_name
def get_user_input(state: MyWorkflowState):
    return state.request.target

@builder.pre_tool("context", "processor")  # ✅ 使用 Agent 的 role_name
def get_context(state: MyWorkflowState):
    return {
        "history": state.messages,
        "previous_results": state.agent_results
    }
```

#### 错误示例 ❌

```python
# ❌ 错误：绑定到节点名称而不是 Agent 的 role_name
@builder.pre_tool("user_input", "process_node")  # ❌ 这是节点名称，不是 role_name
def get_user_input(state: MyWorkflowState):
    return state.request.target

# 这样会导致工具无法被 Agent 获取到！
```

#### 工作原理说明

```python
# 1. 在 workflow 中定义工具时，绑定到 Agent 的 role_name
@builder.pre_tool("input", "my_agent")  # "my_agent" 是 Agent 的 role_name

# 2. GenericGraphBuilder 将工具注册到 ToolManager
# ToolManager.register_pre_tool(name="input", role="my_agent", func=...)

# 3. Agent 执行时，通过自己的 role_name 获取工具
class MyAgent(BaseAgent):
    @property
    def role_name(self):
        return "my_agent"  # ← 必须匹配
    
    async def execute_pre_tools(self, state):
        # 使用 self.role_name 获取工具
        results = await self.tool_manager.execute_pre_tools(self.role_name)
        # 这里会获取到所有绑定到 "my_agent" 的前置工具
```

在 Prompt 模板中使用：

```jinja2
# task_prompt_for_my_agent.jinja
用户输入: {{ user_input }}

上下文信息:
{{ context }}

请根据以上信息完成任务。
```

### 6.2 后置工具（Post-Tool）

后置工具是 Agent 在执行过程中可以调用的工具函数。

> ⚠️ **同样重要**：`@builder.post_tool()` 的参数也应该是 **Agent 的 role_name**，而不是节点名称！

#### 定义后置工具

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class MyToolInput(BaseModel):
    """工具输入参数定义"""
    param1: str = Field(description="参数1的描述")
    param2: int = Field(default=10, description="参数2的描述")

@builder.post_tool("agent_role_name")  # ← 使用 Agent 的 role_name
@tool(args_schema=MyToolInput)
def my_post_tool(param1: str, param2: int = 10):
    """
    后置工具函数
    
    这个工具的描述会被 LLM 看到，用于决定是否调用此工具。
    
    Args:
        param1: 参数1
        param2: 参数2
        
    Returns:
        工具执行结果
    """
    # 工具逻辑
    result = f"处理 {param1}，参数2={param2}"
    return result
```

#### 示例：文件读取工具

```python
from pydantic import BaseModel, Field
from langchain.tools import tool
from pathlib import Path

class ReadFileInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径")

@builder.post_tool("file_processor")
@tool(args_schema=ReadFileInput)
def read_file_tool(file_path: str):
    """读取指定文件的内容"""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 6.3 工具使用流程

```
1. 定义前置工具 (@builder.pre_tool)
   ↓
2. 前置工具执行，收集上下文
   ↓
3. 上下文注入到 Prompt
   ↓
4. Agent 调用 LLM
   ↓
5. LLM 决定是否调用后置工具
   ↓
6. 执行后置工具（如果需要）
   ↓
7. 返回最终结果
```

---

## 7. 节点编写

### 7.1 节点函数签名

节点函数必须是异步函数，接收 State 并返回 State：

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """
    节点函数
    
    Args:
        state: 当前状态对象
        
    Returns:
        更新后的状态对象
    """
    # 节点逻辑
    return state
```

### 7.2 调用 Agent 的节点

```python
async def process_node(state: MyWorkflowState) -> MyWorkflowState:
    """调用 Agent 处理数据"""
    from dataflow_agent.agentroles import create_simple_agent
    
    # 创建 Agent
    agent = create_simple_agent(
        name="processor",
        model_name="gpt-4o",
        temperature=0.5,
    )
    
    # 执行 Agent
    state = await agent.execute(state)
    
    # 获取结果
    result = state.agent_results.get("processor", {}).get("results", {})
    log.info(f"处理结果: {result}")
    
    return state
```

### 7.3 数据处理节点

```python
async def transform_node(state: MyWorkflowState) -> MyWorkflowState:
    """数据转换节点"""
    # 从前一个节点获取结果
    previous_result = state.agent_results.get("processor", {}).get("results", {})
    
    # 进行数据转换
    transformed = {
        "data": previous_result.get("raw_data", []),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": "processor"
        }
    }
    
    # 存储到 temp_data
    state.temp_data["transformed_data"] = transformed
    
    return state
```

### 7.4 条件判断节点

```python
def decision_node(state: MyWorkflowState) -> str:
    """
    条件判断节点（返回下一个节点名称）
    
    注意：条件节点通常是同步函数，返回字符串
    """
    result = state.agent_results.get("processor", {}).get("results", {})
    
    if result.get("success"):
        return "success_node"
    else:
        return "error_node"
```

### 7.5 终止节点

```python
async def end_node(state: MyWorkflowState) -> MyWorkflowState:
    """终止节点"""
    log.info("Workflow 执行完成")
    return state

# 或者使用 lambda
nodes = {
    "_end_": lambda state: state,
}
```

---

## 8. 图构建

### 8.1 添加节点

```python
# 方式 1: 使用字典批量添加
nodes = {
    "step1": step1_node,
    "step2": step2_node,
    "step3": step3_node,
    "_end_": lambda state: state,
}
builder.add_nodes(nodes)

# 方式 2: 单个添加
builder.add_node("step1", step1_node)
builder.add_node("step2", step2_node)
```

### 8.2 添加边（普通边）

```python
# 方式 1: 使用列表批量添加
edges = [
    ("step1", "step2"),
    ("step2", "step3"),
    ("step3", "_end_"),
]
builder.add_edges(edges)

# 方式 2: 单个添加
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
```

### 8.3 添加条件边

条件边根据函数返回值决定下一个节点：

```python
def condition_func(state: MyWorkflowState) -> str:
    """
    条件函数
    
    Returns:
        下一个节点的名称
    """
    if state.temp_data.get("success"):
        return "success_node"
    elif state.temp_data.get("retry_count", 0) < 3:
        return "retry_node"
    else:
        return "_end_"

# 添加条件边
builder.add_conditional_edges({
    "decision_node": condition_func
})
```

### 8.4 链式调用

GenericGraphBuilder 支持链式调用：

```python
builder = (GenericGraphBuilder(state_model=MyWorkflowState, entry_point="start")
    .add_nodes(nodes)
    .add_edges(edges)
    .add_conditional_edges(conditional_edges)
)
```

### 8.5 完整图构建示例

```python
@register("my_workflow")
def create_my_workflow_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(
        state_model=MyWorkflowState,
        entry_point="start"
    )
    
    # 定义节点
    nodes = {
        "start": start_node,
        "process": process_node,
        "decision": decision_node,
        "success": success_node,
        "retry": retry_node,
        "_end_": lambda state: state,
    }
    
    # 定义普通边
    edges = [
        ("start", "process"),
        ("process", "decision"),
        ("success", "_end_"),
        ("retry", "process"),
    ]
    
    # 定义条件边
    def decision_condition(state: MyWorkflowState) -> str:
        if state.temp_data.get("success"):
            return "success"
        elif state.temp_data.get("retry_count", 0) < 3:
            return "retry"
        else:
            return "_end_"
    
    conditional_edges = {
        "decision": decision_condition
    }
    
    # 构建图
    return (builder
        .add_nodes(nodes)
        .add_edges(edges)
        .add_conditional_edges(conditional_edges)
    )
```

---

## 9. 完整示例

### 9.1 场景：文本分析 Workflow

创建一个分析文本情感和关键词的 Workflow。

#### Step 1: 创建 State 和 Request

```python
# dataflow_agent/states/text_analysis_state.py
from dataclasses import dataclass, field
from dataflow_agent.state import MainState, MainRequest

@dataclass
class TextAnalysisRequest(MainRequest):
    """文本分析请求"""
    text: str = ""
    analysis_type: str = "sentiment"  # "sentiment" | "keywords" | "both"

@dataclass
class TextAnalysisState(MainState):
    """文本分析状态"""
    request: TextAnalysisRequest = field(default_factory=TextAnalysisRequest)
    
    # 分析结果
    sentiment_result: dict = field(default_factory=dict)
    keywords_result: dict = field(default_factory=dict)
    final_report: str = ""
```

#### Step 2: 创建 Workflow

```python
# dataflow_agent/workflow/wf_text_analysis.py
"""
text_analysis workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能：分析文本的情感和关键词
"""

from __future__ import annotations
from dataflow_agent.states.text_analysis_state import TextAnalysisState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_simple_agent
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("text_analysis")
def create_text_analysis_graph() -> GenericGraphBuilder:
    """创建文本分析工作流"""
    builder = GenericGraphBuilder(
        state_model=TextAnalysisState,
        entry_point="start"
    )
    
    # ========== 前置工具 ==========
    @builder.pre_tool("text", "sentiment_analysis")
    def get_text_for_sentiment(state: TextAnalysisState):
        return state.request.text
    
    @builder.pre_tool("text", "keyword_extraction")
    def get_text_for_keywords(state: TextAnalysisState):
        return state.request.text
    
    @builder.pre_tool("sentiment", "report_generation")
    def get_sentiment(state: TextAnalysisState):
        return state.sentiment_result
    
    @builder.pre_tool("keywords", "report_generation")
    def get_keywords(state: TextAnalysisState):
        return state.keywords_result
    
    # ========== 节点定义 ==========
    async def start_node(state: TextAnalysisState) -> TextAnalysisState:
        """起始节点"""
        log.info(f"开始分析文本，类型: {state.request.analysis_type}")
        return state
    
    async def sentiment_analysis_node(state: TextAnalysisState) -> TextAnalysisState:
        """情感分析节点"""
        agent = create_simple_agent(
            name="sentiment_analyzer",
            model_name="gpt-4o",
            temperature=0.3,
            parser_type="json",
        )
        
        state = await agent.execute(state)
        
        # 保存结果
        result = state.agent_results.get("sentiment_analyzer", {}).get("results", {})
        state.sentiment_result = result
        
        log.info(f"情感分析完成: {result}")
        return state
    
    async def keyword_extraction_node(state: TextAnalysisState) -> TextAnalysisState:
        """关键词提取节点"""
        agent = create_simple_agent(
            name="keyword_extractor",
            model_name="gpt-4o",
            temperature=0.3,
            parser_type="json",
        )
        
        state = await agent.execute(state)
        
        # 保存结果
        result = state.agent_results.get("keyword_extractor", {}).get("results", {})
        state.keywords_result = result
        
        log.info(f"关键词提取完成: {result}")
        return state
    
    async def report_generation_node(state: TextAnalysisState) -> TextAnalysisState:
        """报告生成节点"""
        agent = create_simple_agent(
            name="report_generator",
            model_name="gpt-4o",
            temperature=0.7,
            parser_type="text",
        )
        
        state = await agent.execute(state)
        
        # 保存报告
        result = state.agent_results.get("report_generator", {}).get("results", {})
        state.final_report = result.get("raw", "")
        
        log.info("报告生成完成")
        return state
    
    # ========== 条件判断 ==========
    def route_after_start(state: TextAnalysisState) -> str:
        """根据分析类型路由"""
        analysis_type = state.request.analysis_type
        
        if analysis_type == "sentiment":
            return "sentiment_analysis"
        elif analysis_type == "keywords":
            return "keyword_extraction"
        else:  # "both"
            return "sentiment_analysis"
    
    def route_after_sentiment(state: TextAnalysisState) -> str:
        """情感分析后的路由"""
        if state.request.analysis_type == "both":
            return "keyword_extraction"
        else:
            return "report_generation"
    
    # ========== 图构建 ==========
    nodes = {
        "start": start_node,
        "sentiment_analysis": sentiment_analysis_node,
        "keyword_extraction": keyword_extraction_node,
        "report_generation": report_generation_node,
        "_end_": lambda state: state,
    }
    
    edges = [
        ("keyword_extraction", "report_generation"),
        ("report_generation", "_end_"),
    ]
    
    conditional_edges = {
        "start": route_after_start,
        "sentiment_analysis": route_after_sentiment,
    }
    
    return (builder
        .add_nodes(nodes)
        .add_edges(edges)
        .add_conditional_edges(conditional_edges)
    )
```

#### Step 3: 创建测试文件

```python
# tests/test_text_analysis.py
"""
测试 text_analysis workflow
"""

import asyncio
import pytest
from dataflow_agent.states.text_analysis_state import TextAnalysisState, TextAnalysisRequest
from dataflow_agent.workflow import run_workflow

async def run_text_analysis_pipeline():
    """执行文本分析工作流"""
    # 构造请求
    request = TextAnalysisRequest(
        text="这是一个非常棒的产品，我很喜欢！",
        analysis_type="both",
        language="zh",
        model="gpt-4o",
    )
    
    # 初始化状态
    state = TextAnalysisState(request=request)
    
    # 运行 workflow
    final_state = await run_workflow("text_analysis", state)
    
    return final_state

@pytest.mark.asyncio
async def test_text_analysis_pipeline():
    """测试文本分析工作流"""
    final_state = await run_text_analysis_pipeline()
    
    # 断言
    assert final_state is not None
    assert final_state.sentiment_result
    assert final_state.keywords_result
    assert final_state.final_report
    
    # 打印结果
    print("\n=== 情感分析结果 ===")
    print(final_state.sentiment_result)
    
    print("\n=== 关键词提取结果 ===")
    print(final_state.keywords_result)
    
    print("\n=== 最终报告 ===")
    print(final_state.final_report)

if __name__ == "__main__":
    asyncio.run(run_text_analysis_pipeline())
```

### 9.2 场景：批量数据处理 Workflow（并行模式）

创建一个批量处理多条数据的 Workflow，展示并行模式的使用。

#### Step 1: 创建 State 和 Request

```python
# dataflow_agent/states/batch_process_state.py
from dataclasses import dataclass, field
from typing import List
from dataflow_agent.state import MainState, MainRequest

@dataclass
class BatchProcessRequest(MainRequest):
    """批量处理请求"""
    items: List[dict] = field(default_factory=list)  # 待处理的数据列表
    process_type: str = "summarize"  # 处理类型

@dataclass
class BatchProcessState(MainState):
    """批量处理状态"""
    request: BatchProcessRequest = field(default_factory=BatchProcessRequest)
    
    # 处理结果
    processed_items: List[dict] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    summary: str = ""
```

#### Step 2: 创建 Workflow

```python
# dataflow_agent/workflow/wf_batch_process.py
"""
batch_process workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能：批量处理多条数据，使用并行模式提高效率
"""

from __future__ import annotations
from dataflow_agent.states.batch_process_state import BatchProcessState
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.agentroles import create_parallel_agent, create_simple_agent
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("batch_process")
def create_batch_process_graph() -> GenericGraphBuilder:
    """创建批量处理工作流"""
    builder = GenericGraphBuilder(
        state_model=BatchProcessState,
        entry_point="prepare"
    )
    
    # ========== 前置工具 ==========
    
    # 为并行处理 Agent 准备数据
    @builder.pre_tool("items", "batch_processor")
    def get_items_for_parallel(state: BatchProcessState):
        """返回需要并行处理的数据列表"""
        # 将每个 item 包装成包含必要上下文的字典
        return [
            {
                "item": item,
                "process_type": state.request.process_type,
                "index": idx
            }
            for idx, item in enumerate(state.request.items)
        ]
    
    # 为汇总 Agent 准备数据
    @builder.pre_tool("results", "summarizer")
    def get_results_for_summary(state: BatchProcessState):
        return state.processed_items
    
    @builder.pre_tool("stats", "summarizer")
    def get_stats_for_summary(state: BatchProcessState):
        return {
            "total": len(state.request.items),
            "success": state.success_count,
            "error": state.error_count
        }
    
    # ========== 节点定义 ==========
    
    async def prepare_node(state: BatchProcessState) -> BatchProcessState:
        """准备节点：验证输入数据"""
        items = state.request.items
        
        if not items:
            log.warning("没有待处理的数据")
            state.temp_data["skip_processing"] = True
        else:
            log.info(f"准备处理 {len(items)} 条数据")
            state.temp_data["skip_processing"] = False
        
        return state
    
    async def parallel_process_node(state: BatchProcessState) -> BatchProcessState:
        """并行处理节点：使用并行模式处理所有数据"""
        
        # 创建并行模式 Agent
        agent = create_parallel_agent(
            name="batch_processor",
            model_name="gpt-4o",
            temperature=0.3,
            concurrency_limit=5,  # 同时处理5条数据
            parser_type="json",
        )
        
        # 执行并行处理
        state = await agent.execute(state)
        
        # 提取并行结果
        result = state.agent_results.get("batch_processor", {}).get("results", {})
        parallel_results = result.get("parallel_results", [])
        
        # 统计结果
        success_count = 0
        error_count = 0
        processed_items = []
        
        for idx, item_result in enumerate(parallel_results):
            if item_result.get("error"):
                error_count += 1
                processed_items.append({
                    "index": idx,
                    "status": "error",
                    "error": item_result.get("error")
                })
            else:
                success_count += 1
                processed_items.append({
                    "index": idx,
                    "status": "success",
                    "result": item_result
                })
        
        # 更新状态
        state.processed_items = processed_items
        state.success_count = success_count
        state.error_count = error_count
        
        log.info(f"并行处理完成: 成功 {success_count}, 失败 {error_count}")
        return state
    
    async def summarize_node(state: BatchProcessState) -> BatchProcessState:
        """汇总节点：生成处理报告"""
        
        agent = create_simple_agent(
            name="summarizer",
            model_name="gpt-4o",
            temperature=0.5,
            parser_type="text",
        )
        
        state = await agent.execute(state)
        
        # 保存汇总结果
        result = state.agent_results.get("summarizer", {}).get("results", {})
        state.summary = result.get("raw", "")
        
        log.info("汇总报告生成完成")
        return state
    
    # ========== 条件判断 ==========
    
    def check_skip(state: BatchProcessState) -> str:
        """检查是否跳过处理"""
        if state.temp_data.get("skip_processing"):
            return "_end_"
        return "parallel_process"
    
    # ========== 图构建 ==========
    
    nodes = {
        "prepare": prepare_node,
        "parallel_process": parallel_process_node,
        "summarize": summarize_node,
        "_end_": lambda state: state,
    }
    
    edges = [
        ("parallel_process", "summarize"),
        ("summarize", "_end_"),
    ]
    
    conditional_edges = {
        "prepare": check_skip,
    }
    
    return (builder
        .add_nodes(nodes)
        .add_edges(edges)
        .add_conditional_edges(conditional_edges)
    )
```

#### Step 3: 创建测试文件

```python
# tests/test_batch_process.py
"""
测试 batch_process workflow
"""

import asyncio
import pytest
from dataflow_agent.states.batch_process_state import BatchProcessState, BatchProcessRequest
from dataflow_agent.workflow import run_workflow

async def run_batch_process_pipeline():
    """执行批量处理工作流"""
    # 构造请求
    request = BatchProcessRequest(
        items=[
            {"title": "文章1", "content": "这是第一篇文章的内容..."},
            {"title": "文章2", "content": "这是第二篇文章的内容..."},
            {"title": "文章3", "content": "这是第三篇文章的内容..."},
            {"title": "文章4", "content": "这是第四篇文章的内容..."},
            {"title": "文章5", "content": "这是第五篇文章的内容..."},
        ],
        process_type="summarize",
        language="zh",
        model="gpt-4o",
    )
    
    # 初始化状态
    state = BatchProcessState(request=request)
    
    # 运行 workflow
    final_state = await run_workflow("batch_process", state)
    
    return final_state

@pytest.mark.asyncio
async def test_batch_process_pipeline():
    """测试批量处理工作流"""
    final_state = await run_batch_process_pipeline()
    
    # 断言
    assert final_state is not None
    assert len(final_state.processed_items) == 5
    assert final_state.success_count + final_state.error_count == 5
    
    # 打印结果
    print(f"\n=== 处理统计 ===")
    print(f"成功: {final_state.success_count}")
    print(f"失败: {final_state.error_count}")
    
    print(f"\n=== 处理详情 ===")
    for item in final_state.processed_items:
        print(f"  [{item['index']}] {item['status']}")
    
    print(f"\n=== 汇总报告 ===")
    print(final_state.summary)

if __name__ == "__main__":
    asyncio.run(run_batch_process_pipeline())
```

---

## 10. 调试和测试

### 10.1 添加日志

```python
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    log.info("开始处理节点")
    log.debug(f"当前状态: {state}")
    
    try:
        # 处理逻辑
        result = await process_data(state)
        log.info(f"处理成功: {result}")
    except Exception as e:
        log.error(f"处理失败: {e}")
        log.exception("详细错误信息")
    
    return state
```

### 10.2 运行测试

```bash
# 运行单个测试文件
pytest tests/test_my_workflow.py -v -s

# 运行所有测试
pytest tests/ -v

# 直接运行测试文件
python tests/test_my_workflow.py
```

### 10.3 调试技巧

#### 1. 打印中间状态

```python
async def debug_node(state: MyWorkflowState) -> MyWorkflowState:
    """调试节点"""
    log.critical("=== 调试信息 ===")
    log.critical(f"Request: {state.request}")
    log.critical(f"Agent Results: {state.agent_results}")
    log.critical(f"Temp Data: {state.temp_data}")
    return state
```

#### 2. 检查 Agent 结果

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    state = await agent.execute(state)
    
    # 检查结果
    result = state.agent_results.get("my_agent", {})
    log.info(f"前置工具结果: {result.get('pre_tool_results')}")
    log.info(f"后置工具: {result.get('post_tools')}")
    log.info(f"执行结果: {result.get('results')}")
    
    return state
```

#### 3. 使用断点

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    # 在需要的地方添加断点
    import pdb; pdb.set_trace()
    
    # 或使用 breakpoint()（Python 3.7+）
    breakpoint()
    
    return state
```

### 10.4 常见问题排查

#### 问题 1: Agent 未注册

```
错误: KeyError: 'my_agent'
解决: 确保 Agent 类使用了 @register 装饰器或正确继承了 BaseAgent
```

#### 问题 2: 前置工具未执行

```
错误: 前置工具结果为空
解决: 检查 @builder.pre_tool 的第二个参数是否与 Agent 的 role_name 匹配（不是节点名称！）
```

#### 问题 3: State 字段缺失

```
错误: AttributeError: 'MyWorkflowState' object has no attribute 'xxx'
解决: 在 State 类中添加缺失的字段定义
```

#### 问题 4: 工具绑定到错误的角色

```
错误: Agent 执行时获取不到前置工具结果
原因: @builder.pre_tool("xxx", "node_name") 使用了节点名称而不是 Agent 的 role_name
解决: 将第二个参数改为 Agent 的 role_name
示例: @builder.pre_tool("xxx", "my_agent")  # my_agent 是 Agent 的 role_name
```

---

## 11. 最佳实践

### 11.1 模块化设计

**原则**: 每个节点功能单一明确

```python
# ✅ 好的做法
async def fetch_data_node(state):
    """只负责获取数据"""
    state.raw_data = await fetch_from_api()
    return state

async def process_data_node(state):
    """只负责处理数据"""
    state.processed_data = process(state.raw_data)
    return state

# ❌ 不好的做法
async def fetch_and_process_node(state):
    """一个节点做太多事情"""
    state.raw_data = await fetch_from_api()
    state.processed_data = process(state.raw_data)
    state.validated_data = validate(state.processed_data)
    state.final_result = transform(state.validated_data)
    return state
```

### 11.2 错误处理

**原则**: 在节点中添加异常捕获

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """带错误处理的节点"""
    try:
        # 主要逻辑
        result = await agent.execute(state)
        state.success = True
        return result
    except Exception as e:
        log.exception(f"节点执行失败: {e}")
        
        # 记录错误
        state.temp_data["error"] = str(e)
        state.success = False
        
        # 可以选择继续或终止
        return state
```

### 11.3 状态管理

**原则**: 合理设计 State 类的字段

```python
# ✅ 好的做法：清晰的字段分类
@dataclass
class MyWorkflowState(MainState):
    # 输入数据
    request: MyWorkflowRequest = field(default_factory=MyWorkflowRequest)
    
    # 中间结果
    intermediate_data: dict = field(default_factory=dict)
    processing_status: str = "pending"
    
    # 最终输出
    final_result: dict = field(default_factory=dict)
    
    # 元数据
    start_time: str = ""
    end_time: str = ""
    duration: float = 0.0

# ❌ 不好的做法：字段混乱
@dataclass
class MyWorkflowState(MainState):
    data1: dict = field(default_factory=dict)
    data2: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    temp: dict = field(default_factory=dict)
```

### 11.4 工具复用

**原则**: 充分利用现有的工具函数

```python
# 复用项目中的工具
from dataflow_agent.toolkits.basetool.file_tools import local_tool_for_sample
from dataflow_agent.toolkits.optool.op_tools import get_operator_content_str

@builder.pre_tool("sample_data", "my_node")
def get_sample(state: MyWorkflowState):
    """复用现有工具"""
    from types import SimpleNamespace
    req = SimpleNamespace(json_file=state.request.data_file)
    return local_tool_for_sample(req, sample_size=5)
```

### 11.5 文档注释

**原则**: 为每个节点和工具添加详细注释

```python
async def my_node(state: MyWorkflowState) -> MyWorkflowState:
    """
    节点功能描述
    
    此节点负责：
    1. 从 API 获取数据
    2. 验证数据格式
    3. 存储到 state.raw_data
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态对象
        
    Raises:
        ValueError: 当数据格式不正确时
        ConnectionError: 当 API 连接失败时
    """
    # 实现
    pass
```

### 11.6 并行模式最佳实践

**原则**: 合理使用并行模式提高效率

#### 何时使用并行模式

✅ **适合使用并行模式的场景**：
- 需要处理多条独立的数据（如批量文本分析、图像处理）
- 每条数据的处理逻辑相同
- 数据之间没有依赖关系
- 处理时间较长，并行可以显著提升效率

❌ **不适合使用并行模式的场景**：
- 数据之间有依赖关系（需要按顺序处理）
- 数据量很小（并行开销大于收益）
- 需要共享状态或累积结果
- 处理逻辑复杂且需要大量上下文

#### 并发度设置建议

```python
# 根据任务特点设置合适的并发度
agent = create_parallel_agent(
    name="processor",
    concurrency_limit=5,  # 建议值：3-10
)
```

**设置原则**：
- **API 限流考虑**：如果 LLM API 有速率限制，设置较低的并发度（3-5）
- **数据量考虑**：数据量大时可以适当提高并发度（5-10）
- **资源限制**：考虑内存和网络带宽，避免过高并发导致系统负载过大
- **成本控制**：并发度越高，API 调用成本越高

#### 数据准备注意事项

```python
# ✅ 好的做法：为每个并行项提供完整上下文
@builder.pre_tool("items", "batch_processor")
def prepare_parallel_data(state: MyWorkflowState):
    return [
        {
            "item": item,
            "context": state.request.context,  # 共享上下文
            "config": state.request.config,    # 共享配置
            "index": idx                       # 用于追踪
        }
        for idx, item in enumerate(state.request.items)
    ]

# ❌ 不好的做法：只传递原始数据
@builder.pre_tool("items", "batch_processor")
def prepare_parallel_data(state: MyWorkflowState):
    return state.request.items  # 缺少必要的上下文信息
```

#### 错误处理建议

```python
async def parallel_process_node(state: MyWorkflowState) -> MyWorkflowState:
    """并行处理节点（带完善的错误处理）"""
    
    agent = create_parallel_agent(
        name="batch_processor",
        concurrency_limit=5,
    )
    
    state = await agent.execute(state)
    
    # 提取结果
    result = state.agent_results.get("batch_processor", {}).get("results", {})
    parallel_results = result.get("parallel_results", [])
    
    # 分类处理结果
    success_items = []
    failed_items = []
    
    for idx, item_result in enumerate(parallel_results):
        if item_result.get("error"):
            # 记录失败项
            failed_items.append({
                "index": idx,
                "error": item_result.get("error"),
                "original_data": state.request.items[idx]
            })
            log.warning(f"项 {idx} 处理失败: {item_result.get('error')}")
        else:
            success_items.append({
                "index": idx,
                "result": item_result
            })
    
    # 保存结果
    state.temp_data["success_items"] = success_items
    state.temp_data["failed_items"] = failed_items
    
    # 决定是否需要重试失败项
    if failed_items and len(failed_items) < len(parallel_results) * 0.3:
        # 如果失败率低于30%，可以考虑重试
        log.info(f"将重试 {len(failed_items)} 个失败项")
        state.temp_data["need_retry"] = True
    else:
        state.temp_data["need_retry"] = False
    
    return state
```

#### 结果聚合模式

```python
async def aggregate_results_node(state: MyWorkflowState) -> MyWorkflowState:
    """聚合并行处理结果"""
    
    success_items = state.temp_data.get("success_items", [])
    failed_items = state.temp_data.get("failed_items", [])
    
    # 模式 1: 统计聚合
    stats = {
        "total": len(success_items) + len(failed_items),
        "success": len(success_items),
        "failed": len(failed_items),
        "success_rate": len(success_items) / (len(success_items) + len(failed_items))
    }
    
    # 模式 2: 数据聚合
    aggregated_data = {
        "all_results": [item["result"] for item in success_items],
        "summary": {
            "key_metrics": calculate_metrics(success_items),
            "common_patterns": find_patterns(success_items)
        }
    }
    
    # 模式 3: 分组聚合
    grouped_results = {}
    for item in success_items:
        category = item["result"].get("category", "unknown")
        if category not in grouped_results:
            grouped_results[category] = []
        grouped_results[category].append(item)
    
    state.temp_data["aggregated"] = {
        "stats": stats,
        "data": aggregated_data,
        "grouped": grouped_results
    }
    
    return state
```

#### 性能优化建议

```python
# 1. 使用合适的数据结构
@builder.pre_tool("items", "batch_processor")
def prepare_data(state: MyWorkflowState):
    # ✅ 好：预处理数据，减少每个并行任务的工作量
    return [
        {
            "text": item["text"].strip(),  # 预处理
            "metadata": extract_metadata(item),  # 提前提取
            "index": idx
        }
        for idx, item in enumerate(state.request.items)
    ]

# 2. 批量大小控制
def split_into_batches(items: list, batch_size: int = 50):
    """将大量数据分批处理"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def process_large_dataset(state: MyWorkflowState):
    """处理大数据集"""
    all_results = []
    
    for batch in split_into_batches(state.request.items, batch_size=50):
        # 每批使用并行模式处理
        batch_state = create_batch_state(batch)
        result = await process_batch(batch_state)
        all_results.extend(result)
    
    return all_results

# 3. 监控和日志
async def parallel_process_with_monitoring(state: MyWorkflowState):
    """带监控的并行处理"""
    import time
    
    start_time = time.time()
    
    agent = create_parallel_agent(
        name="batch_processor",
        concurrency_limit=5,
    )
    
    state = await agent.execute(state)
    
    elapsed = time.time() - start_time
    
    # 记录性能指标
    log.info(f"并行处理完成:")
    log.info(f"  - 总数: {len(state.request.items)}")
    log.info(f"  - 耗时: {elapsed:.2f}秒")
    log.info(f"  - 平均: {elapsed/len(state.request.items):.2f}秒/项")
    log.info(f"  - 吞吐: {len(state.request.items)/elapsed:.2f}项/秒")
    
    return state
```

#### 常见陷阱

❌ **陷阱 1：忽略共享状态**
```python
# 错误：并行任务之间共享可变状态
shared_counter = {"count": 0}

@builder.pre_tool("items", "processor")
def prepare_data(state):
    return [{"data": item, "counter": shared_counter} for item in items]
    # 问题：并发修改 shared_counter 会导致竞态条件
```

✅ **正确做法：每个任务独立**
```python
@builder.pre_tool("items", "processor")
def prepare_data(state):
    return [{"data": item, "index": idx} for idx, item in enumerate(items)]
    # 每个任务有独立的 index，不共享可变状态
```

❌ **陷阱 2：过度并行**
```python
# 错误：对少量数据使用高并发
agent = create_parallel_agent(
    name="processor",
    concurrency_limit=20,  # 只有5条数据却设置20并发
)
```

✅ **正确做法：根据数据量调整**
```python
# 根据数据量动态调整并发度
data_count = len(state.request.items)
concurrency = min(data_count, 5)  # 最多5并发

agent = create_parallel_agent(
    name="processor",
    concurrency_limit=concurrency,
)
```

---

## 总结

本教程涵盖了 DataFlow-Agent 项目中 Workflow 编写的完整流程：

1. ✅ 使用 CLI 快速创建组件
2. ✅ 定义 State 和 Request
3. ✅ 使用新的 Agent 创建方式
4. ✅ 配置前置工具和后置工具
5. ✅ 编写节点函数
6. ✅ 构建工作流图
7. ✅ 调试和测试
8. ✅ 遵循最佳实践

### 快速开始检查清单

- [ ] 使用 `dfa create --wf_name xxx` 创建 Workflow
- [ ] 定义或复用 State 和 Request
- [ ] 创建必要的 Agent（如果不存在）
- [ ] 定义前置工具（@builder.pre_tool）
- [ ] 定义后置工具（@builder.post_tool，如果需要）
- [ ] 实现节点函数
- [ ] 配置节点、边和条件边
- [ ] 编写测试文件
- [ ] 运行测试验证
- [ ] 添加文档注释

### 参考资源

- [项目架构文档](../guides/architecture.md)
- [Agent 开发指南](../guides/agent_development.md)
- [工具系统文档](../guides/tool_system.md)
- [示例 Workflow](../../dataflow_agent/workflow/)

---

**祝你编写出高效、可维护的 Workflow！** 🚀
