# 🛠️ Paper2Any CLI 脚手架使用说明

Paper2Any 内置了一套基于 Jinja2 模板的 CLI 代码生成工具（来自 DataFlow-Agent 框架），可以快速生成 **Agent / Workflow / Gradio 页面 / Prompt 模板 / State / Agent-as-Tool** 等标准化代码文件，极大提升开发效率。

> CLI 可执行入口通常为 `dfa`（或等价的 Python entrypoint），下文统一使用：
>
> ```bash
> dfa create ...
> ```

---

## 功能总览

CLI 提供以下代码模板类型：

| 命令参数                | 功能说明           | 生成文件示例                                  | 自动集成能力               |
|-------------------------|--------------------|-----------------------------------------------|----------------------------|
| `--agent_name`          | 创建 Agent 角色    | `agentroles/{name}_agent.py`                  | ✅ `@register` 自动注册    |
| `--wf_name`             | 创建 Workflow      | `workflow/wf_{name}.py` + `tests/test_{name}.py` | ✅ Workflow 自动注册       |
| `--gradio_name`         | 创建 Gradio 页面   | `gradio_app/pages/page_{name}.py`             | ✅ 页面自动发现            |
| `--prompt_name`         | 创建 Prompt 模板库 | `promptstemplates/resources/pt_{name}_repo.py` | 手动在 Agent 中引用       |
| `--state_name`          | 创建自定义 State   | `states/{name}_state.py`                      | 手动在 Workflow / Agent 中使用 |
| `--agent_as_tool_name`  | 创建 Agent 工具    | `agentroles/{name}_agent.py`                  | ✅ `@register` + Tool 集成 |

---

## 基本用法

```bash
# 查看帮助（如 CLI 支持）
dfa --help
dfa create --help

# 典型使用：根据不同参数生成对应模板
dfa create --agent_name my_agent
dfa create --wf_name text_pipeline
dfa create --gradio_name paper2figure
dfa create --prompt_name code_review
dfa create --state_name image_processing
dfa create --agent_as_tool_name text_summarizer
```

---

## 1. 创建 Agent 角色

### 命令示例

```bash
dfa create --agent_name sentiment_analyzer
```

### 生成内容

- 文件路径（示例）：  
  `dataflow_agent/agentroles/common_agents/sentiment_analyzer_agent.py`

### 主要特性

- ✅ 自动注册到 Agent 注册中心（`@register("sentiment_analyzer")`）
- ✅ 预置 `BaseAgent` 继承结构
- ✅ 预留 prompt 配置接口
- ✅ 支持 Simple / ReAct / Graph / VLM 等多种执行策略扩展
- ✅ 提供异步执行入口函数和工厂方法

### 典型代码结构（示意）

```python
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register
from dataflow_agent.states import MainState

@register("sentiment_analyzer")
class SentimentAnalyzer(BaseAgent):
    """情感分析 Agent 示例"""

    @property
    def system_prompt_template_name(self) -> str:
        # 返回系统 Prompt 名称（在 promptstemplates 中定义）
        return "system_prompt_for_sentiment_analyzer"

    def get_task_prompt_params(self, pre_tool_results) -> dict:
        # TODO: 自定义参数映射逻辑
        return {}

    @classmethod
    def create(cls, tool_manager=None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)

    async def execute(self, state: MainState) -> MainState:
        # TODO: 实现核心业务逻辑
        return state


# 便捷调用函数
async def sentiment_analyzer(state: MainState, **kwargs) -> MainState:
    agent = SentimentAnalyzer.create(**kwargs)
    return await agent.execute(state)
```

---

## 2. 创建 Workflow（工作流）

### 命令示例

```bash
dfa create --wf_name text_processing
```

### 生成内容

- Workflow 定义：`dataflow_agent/workflow/wf_text_processing.py`
- 测试用例：`tests/test_text_processing.py`

### 主要特性

- ✅ 自动注册到 Workflow 注册中心（如 `@register("text_processing")`）
- ✅ 基于 StateGraph 的节点/边定义框架
- ✅ 预置 pre_tool / post_tool 装饰器示例
- ✅ 包含完整单元测试模板
- ✅ 支持多种 Agent 创建与组合模式

### Workflow 代码结构（示意）

```python
from dataflow_agent.workflow import GenericGraphBuilder, register_workflow
from dataflow_agent.states import MainState

@register_workflow("text_processing")
def create_text_processing_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=MainState, entry_point="step1")

    # 前置工具（可选）
    @builder.pre_tool("purpose", "step1")
    def _purpose(state: MainState):
        return "工具描述或提示词参数"

    # 节点定义
    async def step1(state: MainState) -> MainState:
        # 可以在这里创建/调用 Agent
        # agent = create_simple_agent(name="your_agent", ...)
        # return await agent.execute(state)
        return state

    # 注册节点与边
    builder.add_nodes({
        "step1": step1,
    }).add_edges([
        ("step1", "_end_"),
    ])

    return builder
```

### 测试示例（简化）

```python
from dataflow_agent.workflow import run_workflow
from dataflow_agent.states import MainState

async def test_text_processing():
    state = MainState(messages=["hello"])
    result = await run_workflow("text_processing", state=state)
    assert isinstance(result, MainState)
```

运行测试：

```bash
pytest tests/test_text_processing.py -v -s
```

---

## 3. 创建 Gradio 页面

### 命令示例

```bash
dfa create --gradio_name model_hub
```

### 生成内容

- 文件：`gradio_app/pages/page_model_hub.py`

### 主要特性

- ✅ 自动被 `gradio_app/app.py` 发现和注册到 Tab
- ✅ 统一命名：`create_{page_name}` 函数
- ✅ 内置 Gradio UI 组件示例
- ✅ 预留调用 Workflow / Agent 的逻辑框架

### 页面结构示例

```python
import gradio as gr

def create_model_hub() -> gr.Blocks:
    with gr.Blocks() as page:
        gr.Markdown("## Model Hub")
        # TODO: 添加输入输出组件、按钮、回调等
    return page

# 示例：调用 workflow 的占位函数
async def run_xxx_pipeline(...):
    # TODO: 调用 dataflow_agent.workflow.run_workflow(...)
    # state = await run_workflow("wf_xxx", state)
    # return state
    return ...
```

> 重启 `python gradio_app/app.py` 后，新页面会自动出现在 Web 界面的 Tab 中。

---

## 4. 创建 Prompt 模板库

### 命令示例

```bash
dfa create --prompt_name code_review
```

### 生成内容

- 文件：`dataflow_agent/promptstemplates/resources/pt_code_review_repo.py`

### 代码结构示例

```python
class CodeReview:
    task_prompt_for_example = """
    Your task description here.
    Input: {input_data}
    """

    system_prompt_for_example = """
    You are an AI assistant for code review tasks.
    """
```

### 使用方式（在 Agent 中调用）

```python
from dataflow_agent.promptstemplates.resources.pt_code_review_repo import CodeReview

# 在 Agent 中指定模板名
@property
def task_prompt_template_name(self) -> str:
    return "task_prompt_for_example"
```

---

## 5. 创建自定义 State

### 命令示例

```bash
dfa create --state_name image_processing
```

### 生成内容

- 文件：`dataflow_agent/states/image_processing_state.py`

### 代码结构示例

```python
from dataclasses import dataclass, field
from dataflow_agent.states import MainRequest, MainState

@dataclass
class ImageProcessingRequest(MainRequest):
    """自定义请求参数"""
    # TODO: 在这里增加你的字段
    pass

@dataclass
class ImageProcessingState(MainState):
    """自定义状态对象"""
    request: ImageProcessingRequest = field(default_factory=ImageProcessingRequest)
```

### 使用方式

```python
from dataflow_agent.states.image_processing_state import ImageProcessingState

state = ImageProcessingState(messages=[])
```

---

## 6. 创建 Agent-as-Tool（可作为 Tool 被调用的 Agent）

### 命令示例

```bash
dfa create --agent_as_tool_name text_summarizer
```

### 生成内容

- 文件：`dataflow_agent/agentroles/text_summarizer_agent.py`

### 主要特性

- ✅ 既可作为普通 Agent 使用
- ✅ 又可作为 Tool 提供给其他 Agent / Workflow 调用
- ✅ 支持自定义工具描述和参数 Schema
- ✅ 自动完成参数解析与映射

### 代码结构示例

```python
from pydantic import BaseModel, Field
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("text_summarizer")
class TextSummarizer(BaseAgent):
    """文本总结 Agent / Tool"""

    def get_tool_description(self) -> str:
        return "用于总结文本内容"

    def get_tool_args_schema(self) -> type[BaseModel]:
        class SummarizerArgs(BaseModel):
            content: str = Field(description="要总结的内容")
            max_length: int = Field(default=500, description="摘要最大长度")
        return SummarizerArgs

    async def execute(self, state):
        # TODO: 实现核心 summarization 逻辑
        return state
```

### 作为 Tool 使用（示意）

```python
# 在其他 Agent 或 Workflow 中
# 例如在 ReAct / Graph Agent 中启用工具模式
# text_summarizer 会自动出现在可用工具列表中
```

---

## 7. 模板通用特性

- 🕐 **时间戳**：生成文件通常包含创建时间注释，方便追踪。
- 🔤 **智能命名**：自动处理 snake_case / CamelCase 转换。
- 📝 **TODO 标记**：关键位置预留 `TODO` 注释，指引你补充业务逻辑。
- 🎯 **最佳实践**：遵循项目内部约定的编码风格与结构。
- 🔗 **自动集成**：
  - Agent / Workflow 自动注册；
  - Gradio 页面自动发现；
  - Prompt / State 模板方便在 Agent / Workflow 中复用。

---

## 8. 命名规范与自动转换

CLI 会对输入名称进行规范化处理，保证**文件名、类名、注册名**统一：

```bash
# 以下三种写法等价
dfa create --agent_name "My Data Processor"
dfa create --agent_name "my-data-processor"
dfa create --agent_name "my_data_processor"

# 统一转换为：
# - 文件名: my_data_processor_agent.py
# - 类名: MyDataProcessor
# - 注册名: "my_data_processor"
```

> 建议尽量使用语义清晰的英文名称，方便在大型 Workflow 中组织与检索。

---

以上即为 Paper2Any / DataFlow-Agent CLI 脚手架的整理版说明，建议你在本项目中创建 Agent / Workflow / 页面时优先使用 CLI，以保持代码风格一致并提升开发效率。
