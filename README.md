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

### ReAct 模式说明

ReAct（Reasoning + Acting）模式允许 Agent 在执行过程中进行推理-行动循环：

```python
# 开启 ReAct 模式
state = await agent.execute(state, use_agent=True)

# 执行流程：
# 1. Thought: 分析当前状态，确定下一步行动
# 2. Action: 调用工具执行操作
# 3. Observation: 观察执行结果
# 4. 重复 1-3 直到任务完成
```

### Agent-as-Tool 说明

Agent 可以被其他 Agent 作为工具调用，实现多 Agent 协作：

```python
# 在 tool_manager 中注册 Agent 作为工具
tool_manager.register_agent_as_tool("icon_editor", IconEditor)

# 其他 Agent 可以调用
result = await parent_agent.call_tool("icon_editor", state=state)
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