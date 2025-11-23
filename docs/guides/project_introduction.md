# DataFlow-Agent 项目介绍

## 1. 项目定位

**DataFlow-Agent** 是一个围绕「数据流 / 工作流」构建的智能 Agent 框架，目标是：

- 把复杂的自然语言任务拆分为一系列可组合的 **Operator / Tool / Workflow**；
- 通过统一的 `BaseAgent` 抽象与多种执行模式（Simple / ReAct / Graph / VLM），让 Agent 能够在不同场景下稳定地执行任务；
- 支撑上层的 Gradio 前端、流水线编排、图式工作流等多种使用方式。

---

## 2. 整体架构

整体架构围绕以下几个核心层次展开：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gradio 前端 / CLI                          │
│                    (gradio_app/ / script/)                      │
├─────────────────────────────────────────────────────────────────┤
│                      Workflow 工作流层                           │
│                   (dataflow_agent.workflow)                     │
├─────────────────────────────────────────────────────────────────┤
│                       Agent 角色层                               │
│                  (dataflow_agent.agentroles)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              执行核心 (cores/)                           │   │
│  │   configs.py (配置) + strategies.py (策略)              │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    LLM & Parser 层                              │
│        (dataflow_agent.llm_callers / parsers)                  │
├─────────────────────────────────────────────────────────────────┤
│                    Tool & State 层                              │
│        (dataflow_agent.toolkits / state)                       │
├─────────────────────────────────────────────────────────────────┤
│                    Prompt 模板层                                │
│              (dataflow_agent.promptstemplates)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 各层职责

| 层级 | 模块路径 | 职责 |
|------|----------|------|
| **Agent 层** | `dataflow_agent.agentroles` | 具体的智能角色，如 `Classifier`、`PipelineBuilder`、`Writer`、`Recommender` 等，每个角色都是 `BaseAgent` 的子类 |
| **执行核心层** | `dataflow_agent.agentroles.cores` | 统一的 Agent 配置与执行策略，提供 Simple / ReAct / Graph / VLM 多种执行模式 |
| **LLM & Parser 层** | `dataflow_agent.llm_callers`, `dataflow_agent.parsers` | 屏蔽具体模型与返回格式差异，支持文本与多模态调用，通过统一解析器解析 LLM 输出 |
| **Tool & Workflow 层** | `dataflow_agent.toolkits`, `dataflow_agent.workflow` | 定义各类工具和流水线工作流，将复杂任务拆解成可复用组件 |
| **State & Prompt 层** | `dataflow_agent.state`, `dataflow_agent.promptstemplates` | 统一请求/中间状态表示，通过模板化 prompt 生成系统提示与任务提示 |

---

## 3. 核心模块详解

### 3.1 `dataflow_agent/__init__.py`：包入口与兼容性

`dataflow_agent/__init__.py` 中保留了一段注释掉的"兼容性 shim"代码，功能是：

- 在运行时创建一个 `dataflow.dataflowagent` 的模块代理；
- 自动把 `dataflow_agent` 下的子模块映射到 `dataflow.dataflowagent.*` 的命名空间。

用途是在项目从旧包名迁移时，保证外部依赖仍能通过旧路径导入。当前版本中这段代码被注释掉，说明项目包名已统一为 `dataflow_agent`。

---

### 3.2 Agent 执行核心：`dataflow_agent.agentroles.cores`

这是整个 Agent 框架的核心，包含两个关键文件：

#### 3.2.1 `configs.py` - 执行模式配置

定义了 Agent 的执行模式枚举和对应的配置类：

```python
class ExecutionMode(Enum):
    """执行模式枚举"""
    SIMPLE = "simple"    # 简单模式：单次LLM调用
    REACT = "react"      # ReAct模式：带验证的循环
    GRAPH = "graph"      # 图模式：子图+工具调用
    VLM = "vlm"          # 视觉语言模型模式
    CUSTOM = "custom"    # 自定义模式（预留）
```

**配置类继承体系：**

```
BaseAgentConfig (基础配置)
    ├── SimpleConfig   (简单模式)
    ├── ReactConfig    (ReAct模式，含 max_retries、validators)
    ├── GraphConfig    (图模式，含 enable_react_validation)
    └── VLMConfig      (视觉语言模型，含 vlm_mode、image_detail)
```

**BaseAgentConfig 核心参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_name` | `str` | LLM 模型名称 |
| `chat_api_url` | `str` | API 接口地址 |
| `temperature` | `float` | 生成温度 |
| `max_tokens` | `int` | 最大 token 数 |
| `tool_mode` | `str` | 工具调用模式 |
| `parser_type` | `str` | 解析器类型 (json/xml/text) |
| `ignore_history` | `bool` | 是否忽略历史消息 |

#### 3.2.2 `strategies.py` - 执行策略

采用**策略模式**，将不同执行模式的逻辑封装为独立的策略类：

```python
ExecutionStrategy (抽象基类)
    ├── SimpleStrategy   # 简单模式：单次 LLM 调用
    ├── ReactStrategy    # ReAct模式：循环调用 + 验证
    ├── GraphStrategy    # 图模式：构建子图执行
    └── VLMStrategy      # VLM模式：多模态处理
```

**策略工厂 `StrategyFactory`：**

```python
# 创建策略
strategy = StrategyFactory.create("react", agent, config)

# 注册自定义策略
StrategyFactory.register("my_mode", MyCustomStrategy)
```

---

### 3.3 `BaseAgent` - Agent 基类

`BaseAgent` 是所有 Agent 的抽象基类，定义了统一的执行框架和扩展点。

#### 3.3.1 核心属性（子类必须实现）

```python
@property
@abstractmethod
def role_name(self) -> str:
    """角色名称"""
    pass

@property
@abstractmethod
def system_prompt_template_name(self) -> str:
    """系统提示词模板名称"""
    pass

@property
@abstractmethod
def task_prompt_template_name(self) -> str:
    """任务提示词模板名称"""
    pass
```

#### 3.3.2 执行流程

```
execute()
    │
    ├─[策略模式]─→ _execution_strategy.execute()
    │
    └─[传统模式]─┬─→ execute_pre_tools()     # 执行前置工具
                 │
                 ├─→ [VLM模式] _execute_vlm()
                 │
                 ├─→ [图模式+后置工具] _execute_react_graph()
                 │
                 ├─→ [ReAct模式] process_react_mode()
                 │
                 └─→ [简单模式] process_simple_mode()
```

#### 3.3.3 关键方法

| 方法 | 说明 |
|------|------|
| `execute()` | 统一执行入口 |
| `execute_pre_tools()` | 执行前置工具 |
| `build_messages()` | 构建 LLM 消息列表 |
| `create_llm()` | 创建 LLM 实例 |
| `parse_result()` | 解析 LLM 输出 |
| `process_simple_mode()` | 简单模式处理 |
| `process_react_mode()` | ReAct 模式处理（带验证循环） |
| `_execute_react_graph()` | 图模式处理（构建子图） |
| `_execute_vlm()` | VLM 模式处理 |
| `update_state_result()` | 更新状态结果 |

#### 3.3.4 Agent-as-Tool 功能

`BaseAgent` 支持将自身包装为可被其他 Agent 调用的工具：

```python
# 将 Agent 包装为工具
tool = my_agent.as_tool(state)

# 工具相关方法
get_tool_name()              # 工具名称
get_tool_description()       # 工具描述
get_tool_args_schema()       # 参数模式
prepare_tool_execution_params()  # 准备执行参数
extract_tool_result()        # 提取执行结果
```

#### 3.3.5 ReAct 验证机制

ReAct 模式支持自定义验证器：

```python
def get_react_validators(self) -> List[ValidatorFunc]:
    """返回验证器列表"""
    return [
        self._default_json_validator,
        self._check_required_fields,  # 自定义验证器
    ]

# 验证器签名
def validator(content: str, parsed_result: Dict) -> Tuple[bool, Optional[str]]:
    """返回 (是否通过, 错误信息)"""
    pass
```

#### 3.3.6 自动注册机制

子类在定义时会自动注册到 `AgentRegistry`：

```python
class MyAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "MyAgent"
    # ...

# 自动注册后可通过名称获取
agent_cls = AgentRegistry.get("myagent")
```

---

### 3.4 State 状态管理

#### 3.4.1 状态类继承体系

```
MainState (基础状态)
    ├── DFState              (主流程状态)
    ├── DataCollectionState  (数据采集状态)
    ├── IconGenState         (图标生成状态)
    ├── WebCrawlState        (网页爬取状态)
    └── PromptWritingState   (提示词生成状态)
```

#### 3.4.2 MainState 核心字段

```python
@dataclass
class MainState:
    request: MainRequest                    # 请求配置
    messages: List[BaseMessage]             # 消息历史
    agent_results: Dict[str, Any]           # Agent 执行结果
    temp_data: Dict[str, Any]               # 临时数据
```

#### 3.4.3 MainRequest 核心字段

```python
@dataclass
class MainRequest:
    language: str = "en"                    # 语言偏好
    chat_api_url: str = "..."               # LLM API 地址
    api_key: str = "..."                    # API 密钥
    model: str = "gpt-4o"                   # 模型名称
    target: str = ""                        # 需求描述
```

---

## 4. 使用示例

### 4.1 创建简单 Agent

```python
from dataflow_agent.agentroles.base_agent import BaseAgent

class MySimpleAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "MySimpleAgent"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "my_system_prompt"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "my_task_prompt"

# 使用
agent = MySimpleAgent()
result = await agent.execute(state)
```

### 4.2 使用策略模式

```python
from dataflow_agent.agentroles.cores.configs import ReactConfig

# 创建 ReAct 配置
config = ReactConfig(
    model_name="gpt-4o",
    max_retries=5,
    validators=[my_custom_validator]
)

# 创建 Agent 并注入配置
agent = MyAgent(execution_config=config)
result = await agent.execute(state)
```

### 4.3 Agent-as-Tool

```python
# 将 Agent 包装为工具
recommender_tool = recommender_agent.as_tool(state)

# 在 ToolManager 中注册
tool_manager.register_post_tool("PipelineBuilder", recommender_tool)
```

---

## 5. 目录结构

```
dataflow_agent/
├── __init__.py              # 包入口
├── cli.py                   # 命令行接口
├── logger.py                # 日志配置
├── state.py                 # 状态定义
├── utils.py                 # 工具函数
│
├── agentroles/              # Agent 角色
│   ├── base_agent.py        # Agent 基类
│   ├── registry.py          # Agent 注册表
│   ├── cores/               # 执行核心
│   │   ├── configs.py       # 配置类
│   │   └── strategies.py    # 策略类
│   ├── classifier.py        # 分类器 Agent
│   ├── recommender.py       # 推荐器 Agent
│   ├── writer.py            # 写作 Agent
│   └── ...                  # 其他 Agent
│
├── llm_callers/             # LLM 调用器
│   ├── base.py              # 基类
│   ├── text.py              # 文本 LLM
│   └── image.py             # 视觉 LLM
│
├── parsers/                 # 解析器
│   └── parsers.py           # JSON/XML/Text 解析器
│
├── toolkits/                # 工具集
│   ├── tool_manager.py      # 工具管理器
│   ├── basetool/            # 基础工具
│   ├── optool/              # 操作工具
│   └── ...                  # 其他工具
│
├── workflow/                # 工作流
│   ├── base.py              # 工作流基类
│   ├── registry.py          # 工作流注册表
│   └── wf_*.py              # 具体工作流
│
├── promptstemplates/        # 提示词模板
│   ├── prompt_template.py   # 模板生成器
│   └── prompts_repo.py      # 模板仓库
│
├── graphbuilder/            # 图构建器
│   ├── graph_builder.py     # 图构建
│   └── message_history.py   # 消息历史
│
└── storage/                 # 存储服务
    └── storage_service.py
```

---

## 6. 设计亮点

1. **策略模式**：通过 `ExecutionStrategy` 将不同执行模式解耦，易于扩展新模式
2. **自动注册**：Agent 子类自动注册到 Registry，支持按名称动态获取
3. **Agent-as-Tool**：Agent 可以被包装为工具，支持 Agent 间的嵌套调用
4. **ReAct 验证**：内置验证循环机制，支持自定义验证器
5. **多模态支持**：统一的 VLM 执行路径，支持图像理解/生成/编辑
6. **状态继承**：通过 State 继承体系，不同任务类型可以复用基础字段并扩展特有字段

---

## 7. 扩展指南

### 7.1 添加新的执行模式

```python
# 1. 在 configs.py 中添加配置类
@dataclass
class MyModeConfig(BaseAgentConfig):
    mode: ExecutionMode = field(default=ExecutionMode.CUSTOM, init=False)
    my_param: str = "default"

# 2. 在 strategies.py 中添加策略类
class MyModeStrategy(ExecutionStrategy):
    async def execute(self, state, **kwargs):
        # 实现执行逻辑
        pass

# 3. 注册策略
StrategyFactory.register("my_mode", MyModeStrategy)
```

### 7.2 添加新的 Agent

```python
from dataflow_agent.agentroles.base_agent import BaseAgent

class NewAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "NewAgent"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "new_agent_system"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "new_agent_task"
    
    def get_react_validators(self):
        """添加自定义验证器"""
        return super().get_react_validators() + [
            self._my_custom_validator
        ]
```

---

## 8. 总结

DataFlow-Agent 通过清晰的分层架构和灵活的策略模式，提供了一个可扩展的智能 Agent 框架。核心设计理念是：

- **统一抽象**：所有 Agent 继承自 `BaseAgent`，共享执行框架
- **模式分离**：通过策略模式分离不同执行逻辑
- **可组合性**：Agent 可以作为工具被其他 Agent 调用
- **易扩展性**：通过继承和注册机制，轻松添加新的 Agent 和执行模式
