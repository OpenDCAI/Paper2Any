# 快速开始

## 环境要求

- **Python**: 3.11 或更高版本
- **操作系统**: Windows / macOS / Linux

## 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/OpenDCAI/DataFlow-Agent
cd DataFlow-Agent
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 或使用 conda
conda create -n dataflow python=3.11
conda activate dataflow
```

### 3. 安装依赖
```bash
pip install -r requirements-dev.txt
pip install -e .
```

## 启动应用

### 方式一：Web界面（推荐）
```bash
python gradio_app/app.py
```
访问 **http://127.0.0.1:7860** 使用可视化界面。

### 方式二：命令行工具
```bash
python script/xxx.py
```

## 基础使用示例

### 运行预置工作流
```python
from dataflow_agent.workflow import run_workflow

# 执行管线推荐工作流
result = await run_workflow("pipeline_recommend", state={
    "task": "数据清洗和分析"
})
```

### 创建自定义 Agent
```python
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("my_agent")
class MyAgent(BaseAgent):
    """自定义 Agent"""
    
    async def execute(self, state):
        # 实现你的业务逻辑
        state["result"] = "处理完成"
        return state
```

## 下一步

- 查看 [功能特性](../index.md#✨-功能特性) 了解完整功能
- 学习 [CLI 工具使用](guides/cli-tool.md) 提升开发效率
- 探索 [项目架构](../index.md#🏗️-项目架构) 深入理解设计