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
