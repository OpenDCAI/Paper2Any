## 📋 Workflow编写教程大纲

### 1. 准备工作
- 了解DataFlow-Agent项目结构
- 熟悉State和Request类的定义模式
- 掌握GraphBuilder的基本概念

### 2. Workflow文件命名规范
- 文件名必须以`wf_`开头，如`wf_my_workflow.py`
- 注册名去掉前缀，如`wf_pipeline_write.py`注册为`"pipeline_write"`

### 3. 基础Workflow结构
```python
from dataflow_agent.workflow.registry import register
from dataflow_agent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow_agent.state import YourStateClass

@register("your_workflow_name")
def create_your_workflow_graph() -> GenericGraphBuilder:
    """创建你的工作流图"""
    builder = GenericGraphBuilder(
        state_model=YourStateClass,  # 你的State类
        entry_point="start_node"     # 入口节点名
    )
    
    # 节点定义
    # 边定义
    
    return builder
```

### 4. 定义State和Request类
在`state.py`中添加对应的数据类：
```python
from dataclasses import dataclass
from dataflow_agent.state import MainRequest, MainState

@dataclass
class YourWorkflowRequest(MainRequest):
    """你的工作流请求参数"""
    input_data: str = ""
    config_param: str = "default"

@dataclass  
class YourWorkflowState(MainState):
    """你的工作流状态"""
    request: YourWorkflowRequest = None
    processing_result: dict = None
    current_step: str = "start"
```

### 5. 节点(Node)编写
```python
async def start_node(state: YourWorkflowState) -> YourWorkflowState:
    """起始节点"""
    # 处理逻辑
    state.current_step = "processing"
    return state

async def processing_node(state: YourWorkflowState) -> YourWorkflowState:
    """处理节点"""
    # 调用Agent或工具
    from dataflow_agent.agentroles import create_agent
    agent = create_agent("your_agent_role")
    state = await agent.execute(state, use_agent=True)
    return state
```

### 6. 工具绑定（前置/后置工具）
```python
# 前置工具
@builder.pre_tool("tool_name", "node_name")
@builder.desc("参数描述字符串")
def pre_tool_function(state: YourWorkflowState):
    return state.some_data

# 后置工具（简化版）
@builder.post_tool("tool_name", "node_name")  
def post_tool_function(module_list):
    “”“
    
    Args:
        module_list: xxx

    ”“”
    return result
```

### 7. 图构建流程
```python
# 定义节点字典
nodes = {
    "start": start_node,
    "process": processing_node,
    "end": lambda state: state,
}

# 定义边（节点流向）
edges = [
    ("start", "process"),
    ("process", "end"),
]

# 注册到builder
builder.add_nodes(nodes).add_edges(edges)
```

### 8. 运行Workflow
```python
from dataflow_agent.workflow import run_workflow

# 创建初始状态
state = YourWorkflowState(request=YourWorkflowRequest(...))

# 运行工作流
result = await run_workflow("your_workflow_name", state)
```

### 9. 调试和测试
- 使用`get_logger(__name__)`添加日志
- 在开发环境中测试单个节点
- 检查状态流转是否正确

### 10. 最佳实践
1. **模块化设计**：每个节点功能单一明确
2. **错误处理**：在节点中添加异常捕获
3. **状态管理**：合理设计State类的字段
4. **工具复用**：充分利用现有的工具函数
5. **文档注释**：为每个节点和工具添加详细注释

### 11. 完整示例模板
```python
"""
your_workflow workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能描述：简要说明工作流功能
"""

from dataflow_agent.workflow.registry import register
from dataflow_agent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow_agent.state import YourWorkflowState
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

@register("your_workflow")
def create_your_workflow_graph() -> GenericGraphBuilder:
    builder = GenericGraphBuilder(
        state_model=YourWorkflowState,
        entry_point="start"
    )
    
    # 工具定义
    @builder.pre_tool("input_data", "start")
    @builder.desc("获取输入数据")
    def get_input_data(state: YourWorkflowState):
        return state.request.input_data
    
    # 节点定义
    async def start_node(state: YourWorkflowState) -> YourWorkflowState:
        log.info("开始处理")
        return state
    
    # 图构建
    nodes = {"start": start_node, "end": lambda state: state}
    edges = [("start", "end")]
    
    return builder.add_nodes(nodes).add_edges(edges)
```

这个教程涵盖了从基础概念到实际开发的完整流程，你可以根据具体需求调整每个步骤的详细内容。需要我详细解释某个特定步骤吗？
        