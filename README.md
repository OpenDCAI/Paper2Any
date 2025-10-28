# DataFlow-Agent 目录结构说明

下面对本仓库的核心目录 / 文件做简要中文说明，帮助新同事快速了解各模块用途及放置内容。  
（括号内为常见文件类型，仅作示例）

| 级别 | 路径 | 主要内容 | 作用 |
| ---- | ---- | -------- | ---- |
| 根 | `LICENSE` | - | 开源协议（Apache-2.0）。 |
| 根 | `README.md` | - | 项目总览与快速上手。 |
| 根 | `pyproject.toml` | - | Python 包元数据、入口脚本、依赖声明。 |
| 根 | `requirements.txt` | txt | 运行时依赖列表。 |
| 根 | `requirements-dev.txt` | txt | 开发 / 测试 / 格式化工具依赖。 |
| 根 | `docs/` | md, png | MkDocs/Sphinx 源文件，存放详细文档。 |
| 根 | `static/` | png, gif | Logo、流程图、演示 GIF 等静态资源。 |
| 根 | `gradio_app/` | py, css | Gradio Web UI（`dataflow_agent webui`）相关代码。 |
| 根 | `script/` | py, sh | 常用启动脚本、批处理脚本、Docker 等。 |
| 根 | `tests/` | py | PyTest 单元 / 集成测试。 |
| 包 | `dataflow_agent/` | 见下表 | Python 主包，所有业务代码。 |

## 向 DataFlow-Agent 增加一个新页面，只需 3 步 (还在更新)


> 目录结构（节选）
> ```
> gradio_app/
> ├── app.py                  # 主程序，上文的代码
> └── pages/
>     ├── __init__.py
>     ├── pipeline_rec.py
>     └── 🔥 你的新文件.py      ← 只要放在这里
> ```

### 1. 取一个文件名

* 放在 `gradio_app/pages/`
* 不能以下划线 `_` 开头  
  例：`hello_world.py`

### 2. 按约定写一个函数

函数 **必须** 叫  
`create_<文件名去掉扩展名>`  
并返回一个 `gr.Blocks`（或 `gr.Row`/`gr.Column` 等可渲染组件）。

> 文件：`gradio_app/pages/hello_world.py`

```python
import gradio as gr

def create_hello_world():
    """
    示例页面：Hello World
    只要返回一个 Gradio 组件即可
    """
    with gr.Blocks() as page:
        gr.Markdown("## 🌟 Hello World!")
        name = gr.Textbox(label="Your Name")
        btn  = gr.Button("Say Hi")
        out  = gr.Textbox(label="Output")

        def say_hi(n):          # 普通同步函数就行
            return f"Hi, {n} 👋"

        btn.click(say_hi, name, out)

    return page
```

命名小贴士  
如果文件叫 `model_hub.py` ⇒ 函数得叫 `create_model_hub()`；  
如果叫 `chat.py` ⇒ 函数叫 `create_chat()`。

### 3. 运行主程序，自动上 Tab

主程序的 `load_pages()` 会：

1. 扫描 `pages/` 目录
2. 用文件名拼出函数名 `create_xxx`
3. 调用函数并把返回的页面塞到 Tab 里

因此**无需**修改任何其他代码；保存文件后重启，就能在 UI 里看到 “Hello World” 这个 Tab。

---


这样别人也可以写：

```python
def create_page():
    ...
```

---

就这么简单——照着示例模版写个 `create_xxx()`，提交 PR，界面里立刻多一个 Tab 🤝

### 4. 指令说明

使用以下指令可以在 `gradio_app` 目录下创建一个名为 `page_test` 的页面：

```bash
dfa create --gradio_name <test>
```

# DFA 命令行工具使用指南

## 简介

`dfa` 是 DataFlow-Agent 项目的脚手架工具，用于快速生成 Workflow 和 Agent 模板代码。

## 安装

```bash
# 开发模式安装
pip install -e .
```

## 使用方法

### 创建 Workflow

```bash
dfa create --wf_name <workflow名称>
```

**示例：**
```bash
dfa create --wf_name my_refine
```

**生成位置：** `dataflow_agent/workflow/my_refine.py`

**生成内容：**
- 基于 StateGraph 的 workflow 框架
- 预置的节点定义和路由逻辑
- 标准的构建和运行方法

---

### 创建 Agent

```bash
dfa create --agent_name <agent名称>
```

**示例：**
```bash
dfa create --agent_name iconagent
```

**生成位置：** `dataflow_agent/agentroles/iconagent.py`

**生成内容：**
- 带 `@register` 装饰器的 Agent 类
- 自动继承自 `BaseAgent`
- prompt 参数构造方法占位
- 状态更新方法占位
- 工厂方法和辅助函数

---

## 注意事项

1. **互斥参数**：`--wf_name` 和 `--agent_name` 只能选择其一
2. **命名规范**：工具会自动处理 snake_case 和 CamelCase 转换
3. **防止覆盖**：如果目标文件已存在，会跳过生成并提示
4. **唯一性**：确保每个 agent 的注册名称在项目中唯一

---

## 后续步骤

生成模板后，需要手动补充：

**Workflow：**
- 实现具体的节点逻辑
- 配置路由条件

**Agent：**
- 指定 prompt 模板名称
- 实现 `get_task_prompt_params` 方法
- 实现 `update_state_result` 方法

## Agent 注册与调用机制

### `agentroles/` 注册流程

```python
# 1. Agent 定义时通过 @register 装饰器自动注册
@register("icon_editor")
class IconEditor(BaseAgent):
    ...

# 2. 包初始化时自动发现并导入所有 Agent
# dataflow_agent/agentroles/__init__.py 会扫描所有 .py 文件并导入

# 3. 使用时通过注册中心获取
from dataflow_agent.agentroles import get_agent_cls, create_agent

# 方式1：获取类后手动实例化
AgentCls = get_agent_cls("icon_editor")
agent = AgentCls(tool_manager=tm)

# 方式2：通过工厂方法创建（推荐）
agent = create_agent("icon_editor", tool_manager=tm, temperature=0.7)
```

### ReAct 模式说明（基于 BaseAgent 代码）

- `react_mode=True` 时，Agent 调用 LLM 后，会自动校验输出格式/内容。
- 如果输出未通过验证（如不是合格 JSON、缺字段等），Agent 会自动将错误反馈追加到对话消息，要求 LLM 修正并重试。
- 这一过程会循环进行，直到 LLM 输出通过所有验证器或达到最大重试次数。
- 注意：**此 ReAct 模式并没有实现经典的“Thought-Action-Observation”多轮推理与工具调用流程**，仅用于自动格式纠错和结果自我修正。

#### 主要流程：
1. 构建初始对话消息，调用 LLM 生成输出。
2. 校验输出（格式、内容等）。
3. 未通过则将错误作为人类反馈追加，要求 LLM 重新生成。
4. 重复以上步骤，直到通过或达到最大重试次数。


```python
exporter = create_exporter(
    tool_manager=get_tool_manager(),    # 工具管理器实例
    react_mode=True,                   # 启用 ReAct 模式
    react_max_retries=3                # 最多自动纠错（重试）3次
)
```

### Agent-as-Tool 说明


> 一句话结论：  
> *register_agent_as_tool* 只是把 **Agent 包装成 LangChain Tool 并注册为「后置工具（post-tool）」**。  
> 之后能否被调用，取决于：  
> 1. 执行父 Agent 时用 `use_agent=True`，  
> 2. 其 `ToolManager` 中确实包含该 Tool，  
> 3. LLM 在生成回答时主动选择调用该 Tool。

---

#### 1. 注册流程

```python
tool_manager = get_tool_manager()

# ① 创建要被包装的 Agent 实例（必须提前把同一个 tool_manager 传进去）
icon_editor = IconEditor.create(tool_manager=tool_manager)

# ② 把它注册成工具；本质上会放进 role_post_tools 或 global_post_tools
tool_manager.register_agent_as_tool(icon_editor, state, role="parent_agent_role")
```

源码要点（tool_manager.py）  

```python
def register_agent_as_tool(self, agent, state, role=None):
    tool = agent.as_tool(state)          # <─ 把 Agent 包成 LangChain Tool
    self.register_post_tool(tool, role)  # <─ 存到“后置工具”列表
```

> • “后置工具”= 只有在 **父 Agent 使用 _graph/agent 模式_（`use_agent=True`）** 时，  
>   `create_llm(..., bind_post_tools=True)` 才会把这些 Tool 绑定给 LLM。  
>
> • 如果用普通 `react_mode` / `simple_mode`，因为 `bind_post_tools=False`，LLM 根本看不到这些工具。

---

#### 2. 调用方式

1. **由 LLM 自动调用（推荐）**

   ```python
   # 让父 Agent 进入图模式
   await parent_agent.execute(state, use_agent=True)
   ```
   - `execute()` 检测到 `use_agent=True` 且存在后置工具 → 进入 **graph 模式**  
   - 生成的 LLM 被 `bind_tools(...)`，可以在回答中产生 `tool_calls`。  
   - 如果模型选择调用 `icon_editor`，LangChain 会自动触发  
     `icon_editor._execute_as_tool(state, **tool_kwargs)`，再递归执行子 Agent。

2. **直接在 Python 调用（调试或脚本化使用）**

   源码里没有 `call_tool` 方法；若想手动触发，可用下面两种做法：

   ```python
   # 方法 A：用 Tool 对象
   tool = icon_editor.as_tool(state)
   result = await tool.coroutine(task_description="...", additional_params={...})

   # 方法 B：用封装好的内部方法
   result = await icon_editor._execute_as_tool(state,
                                               task_description="...",
                                               additional_params={...})
   ```

## Workflow 注册与调用机制

### `workflow/` 注册流程

```python
# 1. 工作流定义时通过 @register 装饰器注册
# dataflow_agent/workflow/wf_pipeline_recommend.py
from dataflow_agent.workflow.registry import register

@register("pipeline_recommend")
def create_pipeline_recommend_graph():
    """创建 Pipeline 推荐工作流图"""
    builder = GraphBuilder()
    # ... 构建图逻辑
    return builder

# 2. 包初始化时自动发现 wf_*.py 并注册
# dataflow_agent/workflow/__init__.py 会扫描所有 wf_*.py 文件并导入

# 3. 使用时通过统一接口调用
from dataflow_agent.workflow import get_workflow, run_workflow, list_workflows

# 方式1：获取工厂并手动构建
factory = get_workflow("pipeline_recommend")
graph_builder = factory()
graph = graph_builder.compile()
result = await graph.ainvoke(state)

# 方式2：直接运行（推荐）
result = await run_workflow("pipeline_recommend", state)

# 查看所有可用工作流
all_workflows = list_workflows()  # 返回 {name: factory} 字典
```

### 工作流命名规范

| 文件名模式 | 注册名示例 | 用途 |
| ---------- | ---------- | ---- |
| `wf_pipeline_recommend.py` | `"pipeline_recommend"` | Pipeline 推荐工作流 |
| `wf_operator_write.py` | `"operator_write"` | Operator 生成工作流 |
| `wf_pipeline_refine.py` | `"pipeline_refine"` | Pipeline 精修工作流 |

---

## 新增模块指南

### 添加新 Agent

1. 在 `dataflow_agent/agentroles/` 下创建文件（如 `my_agent.py`）
2. 继承 `BaseAgent` 并使用 `@register` 装饰器：
```python
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("my_agent")
class MyAgent(BaseAgent):
    """我的自定义 Agent"""
    
    @classmethod
    def create(cls, tool_manager=None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)
    
    async def execute(self, state, use_agent=False, **kwargs):
        # 实现执行逻辑
        pass
```
3. Agent 会自动注册，无需手动导入

### 添加新 Workflow

1. 在 `dataflow_agent/workflow/` 下创建文件（如 `wf_my_workflow.py`）
2. 使用 `@register` 装饰器注册工厂函数：
```python
from dataflow_agent.workflow.registry import register
from dataflow_agent.graghbuilder import GraphBuilder

@register("my_workflow")
def create_my_workflow_graph():
    """创建我的工作流图"""
    builder = GraphBuilder()
    # 添加节点和边
    builder.add_node("start", my_start_func)
    builder.add_node("process", my_process_func)
    builder.add_edge("start", "process")
    return builder
```
3. Workflow 会自动注册，可通过 `run_workflow("my_workflow", state)` 调用

### 实践

- 保持包结构扁平且语义清晰
- Agent 和 Workflow 使用注册机制，避免循环导入
- 新增功能后补充单元测试与文档
- 工具函数优先放在 `utils.py`，避免创建过多小文件

---