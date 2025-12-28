# DataFlow-Agent项目文档主页

<div align="center">

<!-- ![DataFlow-Agent Logo](static/LogoDataFlow_Agentlogo_image_1.png) -->

智能化数据流处理框架 · 模块化 Agent 编排系统

<!-- [[License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[[Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[[Documentation](https://img.shields.io/badge/docs-mkdocs-green.svg)](https://)
[[GitHub Stars](https://img.shields.io/github/stars/your-org/DataFlow-Agent.svg)](https://github.com/your-org/DataFlow-Agent)

[快速开始](#快速开始) · [功能特性](#功能特性) · [文档](guides/cli-tool.md) · [贡献指南](#贡献指南) -->

</div>

---

## 💡 项目简介

**DataFlow-Agent** 是一个基于 Python 的智能化数据流处理框架，提供模块化的 Agent 编排、可视化工作流设计和强大的工具管理能力。通过插件式架构和 CLI 脚手架，让开发者能够快速构建、部署和管理复杂的数据处理任务。

### 核心优势

- 🎯 **开箱即用**：预置多种 Agent 和 Workflow 模板，零配置快速启动
- 🔌 **插件化架构**：Agent、Workflow、Tool 自动注册，解耦灵活
- 🎨 **可视化操作**：基于 Gradio 的 Web 界面，拖拽式流程设计
- ⚡ **高效开发**：CLI 工具一键生成模板代码，大幅提升开发效率
- 🔄 **灵活编排**：基于 StateGraph 的工作流引擎，支持复杂业务逻辑

---

## ✨ 功能特性

### 🤖 Agent 系统
- **自动注册机制**：通过 `@register` 装饰器实现 Agent 的自动发现和注册
- **角色化设计**：支持数据清洗、分析、验证等多种预定义角色
- **灵活扩展**：继承 `BaseAgent` 快速创建自定义 Agent

### 🔄 Workflow 编排
- **状态图引擎**：基于 StateGraph 的流程控制，支持条件分支和循环
- **可视化设计**：通过 Gradio 界面拖拽式创建工作流
- **命名规范**：`wf_*.py` 文件自动识别为 Workflow 模块

### 🛠️ 工具管理
- **统一注册**：工具函数集中管理，统一调用接口
- **类型安全**：完善的类型提示和参数验证
- **易于集成**：支持第三方工具库快速接入

### 🎨 Web 界面
- **响应式设计**：适配桌面和移动端设备
- **页面自动发现**：`gradio_app/pages/` 下的页面自动加载
- **实时交互**：热重载支持，修改代码即时生效

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.10 或更高版本（[下载 Python](https://www.python.org/downloads/)）
- **操作系统**: Windows / macOS / Linux
- **依赖管理**: pip 或 conda

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/OpenDCAI/Paper2Any
cd Paper2Any
```

#### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 或使用 conda
conda create -n dataflow python=3.10
conda activate dataflow
```

#### 3. 安装依赖

```bash
pip install -r requirements-dev.txt
pip install -e .
```

#### 4. 启动应用

```bash
# 启动 Web 界面
python gradio_app/app.py
```

访问 **http://127.0.0.1:7860** 即可使用可视化界面。

---

## 📚 使用示例

### 创建第一个 Agent

使用 CLI 工具快速生成 Agent 模板：

```bash
dfa create --agent_name my_first_agent
```

生成的代码位于 `dataflow_agent/agentroles/common_agents/my_first_agent_agent.py`：

```python
from dataflow_agent.agentroles.base_agent import BaseAgent
from dataflow_agent.agentroles.registry import register

@register("my_first_agent")
class MyFirstAgent(BaseAgent):
    """我的第一个 Agent"""
    
    @classmethod
    def create(cls, tool_manager=None, **kwargs):
        return cls(tool_manager=tool_manager, **kwargs)
    
    async def execute(self, state):
        # 实现你的业务逻辑
        return state
```

### 运行 Workflow

```python
from dataflow_agent.workflow import run_workflow

# 执行预定义的数据验证流程
result = await run_workflow("data_validation", state={
    "data": your_data,
    "config": validation_config
})
```

### 添加自定义 Gradio 页面

```bash
dfa create --gradio_name analytics_dashboard
```

CLI 会在 `gradio_app/pages/page_analytics_dashboard.py` 中生成脚手架文件，
你可以在其中实现页面逻辑，重启应用后会自动出现在 Tab 栏。

---

## 📖 文档导航

<!-- - **[CLI 工具使用指南](guides/cli-tool.md)** - 学习如何使用命令行工具快速开发
- **[Agent 开发教程](guides/agent-development.md)** - 深入了解 Agent 的设计与实现
- **[Workflow 编排指南](guides/workflow-orchestration.md)** - 掌握工作流的构建技巧
- **[API 参考手册](api-reference/agent-api.md)** - 完整的 API 文档
- **[常见问题解答](faq.md)** - 快速解决常见问题 -->

---

## 🏗️ 项目架构

```
DataFlow-Agent/
├── dataflow_agent/          # 核心业务模块
│   ├── agentroles/          # Agent 角色定义（自动注册）
│   ├── workflow/            # Workflow 流程定义（wf_*.py）
│   ├── promptstemplates/    # 提示词模板库（基于 jinja 的 prompt）
│   ├── templates/           # CLI 脚手架 jinja 模板（由 dfa create 使用）
│   ├── toolkits/            # 工具集（文件/算子等工具）
│   ├── state.py             # State / Request 定义
│   ├── utils.py             # 通用工具函数
│   └── ...                  # 其他模块（graphbuilder / llm_callers / parsers / trajectory / resources 等）
├── gradio_app/             # Gradio Web 应用
│   ├── app.py             # 主应用入口
│   └── pages/             # 页面模块（自动发现）
├── docs/                   # MkDocs 文档源文件
├── tests/                  # 单元测试与集成测试
└── script/                 # 开发脚本工具
```

---

## 🤝 贡献指南

我们欢迎任何形式的贡献！无论是提交 Bug、提出新功能建议，还是改进文档。

### 贡献流程

1. **Fork 本仓库**并克隆到本地
2. **创建功能分支**: `git checkout -b feature/amazing-feature`
3. **提交代码**: `git commit -m 'Add amazing feature'`
4. **推送到分支**: `git push origin feature/amazing-feature`
5. **提交 Pull Request**

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 为新功能添加单元测试
- 更新相关文档（包括 docstring 和 MkDocs 文档）
- 提交信息清晰描述变更内容

<!-- 详见 [贡献者指南](CONTRIBUTING.md)。 -->

---

## 🎯 路线图

- [x] 基础 Agent 注册机制
- [x] Workflow 编排引擎
- [x] Gradio Web 界面
- [x] CLI 脚手架工具
- [ ] 多模态支持
- [ ] NL2workflow

<!-- 查看完整 [项目路线图](https://github.com/your-org/DataFlow-Agent/projects)。 -->

---

## 📄 开源协议

本项目采用 **Apache License 2.0** 开源协议。详情请查看 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和使用者！

特别鸣谢：
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排灵感来源
- [Gradio](https://gradio.app/) - 出色的 Web 界面框架
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) - 精美的文档主题

---

## 📞 联系我们

- **问题反馈**: [GitHub Issues](https://github.com/OpenDCAI/Paper2Any/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/OpenDCAI/Paper2Any/discussions)
<!-- - **邮件联系**: contact@dataflow-agent.com -->

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

Made with ❤️ by DataFlow-Agent Team

</div>
