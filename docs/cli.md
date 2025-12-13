### 🛠️ CLI脚手架

DataFlow-Agent提供强大的代码生成工具，基于Jinja2模板自动创建标准化代码文件。

#### 支持的模板类型

| 命令参数 | 功能说明 | 生成文件 | 自动集成 |
|---------|---------|---------|---------|
| `--agent_name` | 创建Agent角色 | `agentroles/{name}_agent.py` | ✅ @register装饰器 |
| `--wf_name` | 创建Workflow | `workflow/wf_{name}.py` + `tests/test_{name}.py` | ✅ @register装饰器 |
| `--gradio_name` | 创建Gradio页面 | `gradio_app/pages/page_{name}.py` | ✅ 自动发现 |
| `--prompt_name` | 创建Prompt模板 | `promptstemplates/resources/pt_{name}_repo.py` | 手动引用 |
| `--state_name` | 创建自定义State | `states/{name}_state.py` | 手动引用 |
| `--agent_as_tool_name` | 创建Agent工具 | `agentroles/{name}_agent.py` | ✅ @register + as_tool |

#### 快速开始

```bash
# 1. 创建一个数据清洗Agent
dfa create --agent_name data_cleaner

# 2. 创建对应的Workflow（自动生成测试文件）
dfa create --wf_name data_cleaning_pipeline

# 3. 创建Web界面页面
dfa create --gradio_name data_cleaner_ui

# 4. 创建Prompt模板库
dfa create --prompt_name data_cleaning_prompts

# 5. 创建自定义State对象
dfa create --state_name data_cleaning_state

# 6. 创建可作为工具调用的Agent
dfa create --agent_as_tool_name text_analyzer
```

#### 详细示例

<details>
<summary><b>📝 创建Agent</b></summary>

```bash
dfa create --agent_name sentiment_analyzer
```

**生成文件**: `dataflow_agent/agentroles/common_agents/sentiment_analyzer_agent.py`

**核心特性**:
- ✅ 自动注册到Agent注册中心（`@register("sentiment_analyzer")`）
- ✅ 包含完整的BaseAgent实现框架
- ✅ 预置prompt模板配置接口
- ✅ 支持多种执行策略（Simple/ReAct/Graph/VLM）
- ✅ 提供异步执行函数和工厂函数

**生成的代码结构**:
```python
@register("sentiment_analyzer")
class SentimentAnalyzer(BaseAgent):
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_sentiment_analyzer"
    
    def get_task_prompt_params(self, pre_tool_results) -> Dict:
        # TODO: 自定义参数映射
        return {}

# 便捷调用函数
async def sentiment_analyzer(state, **kwargs) -> MainState:
    agent = SentimentAnalyzer.create(**kwargs)
    return await agent.execute(state)
```

</details>

<details>
<summary><b>🔄 创建Workflow</b></summary>

```bash
dfa create --wf_name text_processing
```

**生成文件**: 
- `dataflow_agent/workflow/wf_text_processing.py` - 工作流定义
- `tests/test_text_processing.py` - 单元测试

**核心特性**:
- ✅ 自动注册到Workflow注册中心（`@register("text_processing")`）
- ✅ 基于StateGraph的节点和边定义框架
- ✅ 预置pre_tool和post_tool装饰器示例
- ✅ 包含完整的测试用例模板
- ✅ 支持多种Agent创建策略示例

**生成的代码结构**:
```python
@register("text_processing")
def create_text_processing_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(state_model=xxState, entry_point="step1")
    
    # 定义前置工具
    @builder.pre_tool("purpose", "step1")
    def _purpose(state):
        return "工具描述"
    
    # 定义节点
    async def step1(state):
        agent = create_simple_agent(name="your_agent", ...)
        return await agent.execute(state)
    
    # 注册节点和边
    builder.add_nodes({"step1": step1}).add_edges([("step1", "_end_")])
    return builder
```

**运行测试**:
```bash
pytest tests/test_text_processing.py -v -s
```

</details>

<details>
<summary><b>🎨 创建Gradio页面</b></summary>

```bash
dfa create --gradio_name model_hub
```

**生成文件**: `gradio_app/pages/page_model_hub.py`

**核心特性**:
- ✅ 自动被`gradio_app/app.py`发现并加载
- ✅ 函数名遵循`create_{page_name}`规范
- ✅ 包含Gradio组件示例和工作流调用模板
- ✅ 预置异步执行函数框架

**生成的代码结构**:
```python
def create_model_hub() -> gr.Blocks:
    with gr.Blocks() as page:
        gr.Markdown("## Model Hub")
        # TODO: 添加组件
    return page

async def run_xxx_pipeline(...):
    # TODO: 调用workflow
    state = await run_workflow("wf_xxx", state)
    return state
```

**自动集成**: 重启`python gradio_app/app.py`后，新页面自动出现在Tab栏

</details>

<details>
<summary><b>💬 创建Prompt模板</b></summary>

```bash
dfa create --prompt_name code_review
```

**生成文件**: `dataflow_agent/promptstemplates/resources/pt_code_review_repo.py`

**生成的代码结构**:
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

**使用方式**:
```python
from dataflow_agent.promptstemplates.resources.pt_code_review_repo import CodeReview

# 在Agent中引用
@property
def task_prompt_template_name(self) -> str:
    return "task_prompt_for_example"
```

</details>

<details>
<summary><b>📦 创建自定义State</b></summary>

```bash
dfa create --state_name image_processing
```

**生成文件**: `dataflow_agent/states/image_processing_state.py`

**生成的代码结构**:
```python
@dataclass
class ImageProcessingRequest(MainRequest):
    """自定义请求参数"""
    pass

@dataclass
class ImageProcessingState(MainState):
    """自定义状态对象"""
    request: ImageProcessingRequest = field(default_factory=ImageProcessingRequest)
```

**使用方式**:
```python
from dataflow_agent.states.image_processing_state import ImageProcessingState

state = ImageProcessingState(messages=[])
```

</details>

<details>
<summary><b>🔧 创建Agent-as-Tool</b></summary>

```bash
dfa create --agent_as_tool_name text_summarizer
```

**生成文件**: `dataflow_agent/agentroles/text_summarizer_agent.py`

**核心特性**:
- ✅ 可作为普通Agent使用
- ✅ 可作为Tool被其他Agent调用
- ✅ 支持自定义工具描述和参数Schema
- ✅ 自动参数转换和映射

**生成的代码结构**:
```python
@register("text_summarizer")
class TextSummarizer(BaseAgent):
    # 可重写以下方法自定义工具行为
    def get_tool_description(self) -> str:
        return "用于总结文本内容"
    
    def get_tool_args_schema(self) -> type[BaseModel]:
        class SummarizerArgs(BaseModel]:
            content: str = Field(description="要总结的内容")
            max_length: int = Field(default=500)
        return SummarizerArgs
```

**作为工具使用**:
```python
# 在其他Agent的Workflow中
agent = create_graph_agent(name="orchestrator", tool_mode="auto")
# text_summarizer会自动作为可用工具
```

</details>

#### 模板特性

- 🕐 **时间戳**: 每个生成文件包含创建时间
- 🔤 **智能命名**: 自动转换snake_case/CamelCase
- 📝 **TODO标记**: 关键位置预留TODO注释
- 🎯 **最佳实践**: 遵循项目编码规范
- 🔗 **自动集成**: Agent/Workflow自动注册，Gradio页面自动发现

#### 命名规范

CLI工具会自动处理命名转换：

```bash
# 输入任意格式
dfa create --agent_name "My Data Processor"
dfa create --agent_name "my-data-processor"
dfa create --agent_name "my_data_processor"

# 统一转换为
# - 文件名: my_data_processor_agent.py
# - 类名: MyDataProcessor
# - 注册名: "my_data_processor"
```
把这些内容整理一下 整成一篇单独的readme 叫做cli.md