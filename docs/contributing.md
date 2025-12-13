## 🤝 贡献指南

### 开发流程

```bash
# 1. Fork并克隆
git clone https://github.com/<your-username>/DataFlow-Agent.git
cd DataFlow-Agent

# 2. 安装开发依赖
pip install -r requirements-dev.txt
pip install -e .

# 3. 创建分支
git checkout -b feature/your-feature

# 4. 运行测试
pytest

# 5. 提交PR
git push origin feature/your-feature
```

### 添加新Agent

```python
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("my_agent")  # 自动注册
class MyAgent(BaseAgent):
    @classmethod
    def create(cls, tool_manager=None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)
```

### 添加新Workflow

```python
# 文件: dataflow_agent/workflow/wf_my_workflow.py
from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder import GraphBuilder

@register("my_workflow")  # 注册名 = 文件名去掉wf_前缀
def create_my_workflow_graph():
    builder = GraphBuilder()
    # 定义节点和边...
    return builder
```

### 添加Gradio页面

```python
# 文件: gradio_app/pages/my_page.py
import gradio as gr

def create_my_page():  # 函数名 = create_ + 文件名
    with gr.Blocks() as page:
        gr.Markdown("## 我的页面")
        # 添加组件...
    return page
```

### 文档贡献

```bash
# 本地预览
pip install mkdocs-material
mkdocs serve  # 访问 http://127.0.0.1:8000

# 添加新页面
# 1. 在docs/对应目录创建.md文件
# 2. 在mkdocs.yml的nav中添加链接
# 3. 提交PR
```