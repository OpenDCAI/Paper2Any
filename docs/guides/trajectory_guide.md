# DFA Trajectory (TRJ) 使用指南

## 概述

DFA Trajectory (TRJ) 是 DataFlow-Agent 的执行轨迹导出功能，用于捕获、构建和导出 Workflow 执行过程数据。

### 核心功能

1. **实时捕获**：在 Workflow 执行过程中实时记录所有关键信息
2. **模式自适应**：自动识别 ReAct 和 Workflow 模式
3. **多格式导出**：支持 JSON、JSONL、SFT、DPO 等格式
4. **训练就绪**：导出的数据可直接用于模型训练

## 快速开始

### 基本使用

```python
from dataflow_agent.trajectory import TrajectoryManager
from dataflow_agent.state import DFState, DFRequest

# 1. 创建管理器
trj_manager = TrajectoryManager()

# 2. 开始记录
trj_manager.start_recording(inputs={"query": "用户的问题"})

# 3. 执行 workflow（需要集成 collector）
# ... workflow 执行 ...

# 4. 停止记录并生成轨迹
trajectory = trj_manager.stop_recording(
    state=final_state,
    workflow_name="my_workflow"
)

# 5. 导出
filepath = trj_manager.export(trajectory, format="json")
print(f"轨迹已导出: {filepath}")
```

## 核心组件

### 1. TrajectoryCollector（收集器）

负责在执行过程中实时捕获数据。

```python
from dataflow_agent.trajectory import TrajectoryCollector, StepRole

collector = TrajectoryCollector()

# 开始记录
collector.start(inputs={"query": "..."})

# 记录节点执行
collector.on_node_start("classifier", StepRole.AGENT.value)
collector.on_llm_call(
    model="gpt-4o",
    messages=[...],
    response="...",
    duration_ms=1200
)
collector.on_node_end(output={"category": "text"})

# 完成记录
steps = collector.finish()
```

### 2. TrajectoryBuilder（构建器）

将收集的原始数据转换为标准 TRJ 格式。

```python
from dataflow_agent.trajectory import TrajectoryBuilder

builder = TrajectoryBuilder()

# 从 State 和 Collector 构建
trajectory = builder.build_from_state(
    state=final_state,
    collector=collector,
    workflow_name="my_workflow"
)

# 或直接从步骤构建
trajectory = builder.build_from_steps(
    steps=steps,
    workflow_name="my_workflow",
    inputs={"query": "..."},
    final_output={"result": "..."}
)
```

### 3. TrajectoryExporter（导出器）

将轨迹导出为不同格式。

```python
from dataflow_agent.trajectory import TrajectoryExporter

exporter = TrajectoryExporter()

# 导出单个轨迹为 JSON
exporter.export_to_json(trajectory)

# 批量导出为 JSONL
exporter.export_to_jsonl(trajectories, mode="raw")

# 导出 SFT 训练数据
exporter.export_sft_dataset(trajectories, filter_success=True)

# 导出 DPO 训练数据
exporter.export_dpo_dataset(
    chosen_trajectories=success_list,
    rejected_trajectories=failed_list
)
```

### 4. TrajectoryManager（管理器）

统一的管理入口，整合了上述所有组件。

```python
from dataflow_agent.trajectory import TrajectoryManager

manager = TrajectoryManager()

# 完整流程
manager.start_recording(inputs={"query": "..."})
# ... workflow 执行 ...
trajectory = manager.stop_recording(state, "workflow_name")
filepath = manager.export(trajectory, format="json")
```

## 数据模型

### Trajectory 结构

```python
{
  "trace_id": "trj_20251124_abc123",
  "workflow_name": "pipeline_recommend",
  "timestamp": "2025-11-24T12:00:00Z",
  "status": "success",  # success | failed | partial
  "mode": "workflow",   # react | workflow | hybrid
  
  # 输入
  "inputs": {
    "query": "用户问题",
    "model": "gpt-4o"
  },
  
  # 执行步骤
  "steps": [
    {
      "step_index": 0,
      "node_name": "classifier",
      "role": "agent",
      "timestamp": "...",
      "input_context": {...},
      "llm_calls": [...],
      "tool_calls": [...],
      "node_output": {...}
    },
    ...
  ],
  
  # 最终输出
  "final_output": {...},
  
  # 统计信息
  "statistics": {
    "total_steps": 5,
    "total_llm_calls": 3,
    "total_tool_calls": 2,
    "total_duration_ms": 5000
  }
}
```

### Step 结构

每个步骤包含：

- **基本信息**：step_index, node_name, role, timestamp
- **输入上下文**：input_context
- **ReAct 字段**：thought, action_type, action_payload, observation
- **输出**：node_output
- **详细记录**：llm_calls, tool_calls
- **多模态**：multimodal_input, multimodal_output
- **错误信息**：error

## 使用场景

### 场景 1：单次执行导出

```python
# 在 workflow 执行后导出轨迹
manager = TrajectoryManager()
manager.start_recording(inputs={"query": "..."})

# 执行 workflow
final_state = await workflow.run(...)

# 导出
trajectory = manager.stop_recording(final_state, "my_workflow")
manager.export(trajectory, format="json")
```

### 场景 2：批量导出训练数据

```python
# 收集多次执行的轨迹
trajectories = []

for task in tasks:
    manager = TrajectoryManager()
    manager.start_recording(inputs=task)
    
    final_state = await workflow.run(task)
    trajectory = manager.stop_recording(final_state, "workflow")
    trajectories.append(trajectory)

# 批量导出为 SFT 格式
exporter = TrajectoryExporter()
exporter.export_sft_dataset(
    trajectories,
    filepath="training_data.jsonl",
    filter_success=True
)
```

### 场景 3：DPO 数据准备

```python
# 分离成功和失败的轨迹
success_trajectories = [t for t in all_trajectories if t.status == "success"]
failed_trajectories = [t for t in all_trajectories if t.status == "failed"]

# 导出 DPO 数据集
exporter = TrajectoryExporter()
exporter.export_dpo_dataset(
    chosen_trajectories=success_trajectories,
    rejected_trajectories=failed_trajectories,
    filepath="dpo_dataset.jsonl"
)
```

### 场景 4：添加用户反馈

```python
# 执行并导出
trajectory = manager.stop_recording(final_state, "workflow")

# 用户评价
manager.add_feedback(
    trajectory=trajectory,
    score=5,
    comment="结果很准确",
    labels=["accurate", "helpful"]
)

# 导出包含反馈的轨迹
manager.export(trajectory, format="json")
```

## 集成到 Workflow

### 方法 1：手动集成

在 workflow 的每个节点中手动调用 collector：

```python
async def my_node(state: DFState):
    # 获取 collector
    collector = state.temp_data.get('trajectory_collector')
    
    if collector:
        # 记录节点开始
        collector.on_node_start("my_node", StepRole.AGENT.value)
        
        # 执行逻辑
        result = await do_something()
        
        # 记录节点结束
        collector.on_node_end(output=result)
    
    return state
```

### 方法 2：装饰器集成（推荐）

```python
from dataflow_agent.trajectory import get_trajectory_manager

def with_trajectory_recording(workflow_name: str):
    """装饰器：自动记录轨迹"""
    def decorator(func):
        async def wrapper(state, *args, **kwargs):
            manager = get_trajectory_manager()
            
            # 开始记录
            inputs = {"query": state.request.target}
            manager.start_recording(inputs=inputs)
            
            # 将 collector 注入到 state
            state.temp_data['trajectory_collector'] = manager.get_collector()
            
            # 执行 workflow
            final_state = await func(state, *args, **kwargs)
            
            # 停止记录并导出
            trajectory = manager.stop_recording(final_state, workflow_name)
            manager.export(trajectory, format="json")
            
            return final_state
        return wrapper
    return decorator

# 使用
@with_trajectory_recording("my_workflow")
async def my_workflow(state: DFState):
    # workflow 逻辑
    return state
```

## 导出格式

### JSON 格式

完整的轨迹数据，适合单次查看和分析。

```bash
outputs/trajectories/trj_20251124_abc123.json
```

### JSONL 格式

每行一个 JSON 对象，适合批量处理。

```bash
outputs/trajectories/trajectories_raw_20251124_120000.jsonl
```

### SFT 格式

OpenAI messages 格式，可直接用于 SFT 训练。

```json
{
  "trace_id": "trj_xxx",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "metadata": {
    "workflow": "pipeline_recommend",
    "status": "success"
  }
}
```

### DPO 格式

成对数据，用于 DPO 训练。

```json
{
  "prompt": "用户问题",
  "chosen": [...],  # 成功的轨迹
  "rejected": [...],  # 失败的轨迹
  "metadata": {
    "chosen_trace_id": "trj_xxx",
    "rejected_trace_id": "trj_yyy"
  }
}
```

## 最佳实践

### 1. 合理使用记录粒度

- **粗粒度**：只记录关键节点（适合生产环境）
- **细粒度**：记录所有 LLM/工具调用（适合调试和训练）

### 2. 及时导出

```python
# 在 workflow 执行后立即导出
trajectory = manager.stop_recording(state, "workflow")
manager.export(trajectory, format="json")
```

### 3. 定期清理

```python
# 定期清理旧的轨迹文件
import os
from pathlib import Path
from datetime import datetime, timedelta

output_dir = Path("outputs/trajectories")
cutoff_date = datetime.now() - timedelta(days=30)

for file in output_dir.glob("*.json"):
    if file.stat().st_mtime < cutoff_date.timestamp():
        file.unlink()
```

### 4. 使用反馈提升质量

```python
# 收集用户反馈
if user_satisfied:
    manager.add_feedback(
        trajectory=trajectory,
        score=5,
        labels=["good"]
    )
else:
    manager.add_feedback(
        trajectory=trajectory,
        score=1,
        labels=["bad"],
        edited_response=user_corrected_answer
    )
```

## 常见问题

### Q: 如何减小轨迹文件大小？

A: 
1. 不记录 base64 编码的多模态数据，只记录路径
2. 使用 JSONL 格式而不是 JSON
3. 定期清理旧文件

### Q: 如何处理敏感信息？

A:
```python
# 在导出前过滤敏感信息
def sanitize_trajectory(trajectory):
    for step in trajectory.steps:
        if 'password' in step.input_context:
            step.input_context['password'] = '***'
    return trajectory

trajectory = sanitize_trajectory(trajectory)
manager.export(trajectory)
```

### Q: 如何与现有 workflow 集成？

A: 参考"集成到 Workflow"章节，推荐使用装饰器方式。

## 下一步

- 查看 `tests/test_trajectory.py` 了解更多示例
- 阅读 API 文档了解详细参数
- 参考需求文档了解训练数据格式转换
