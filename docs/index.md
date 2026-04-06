# Paper2Any 项目文档

<div align="center">

**从论文到多模态输出的智能化工作流平台**

</div>

---

## 💡 项目简介

**Paper2Any** 是一个基于深度学习的智能化工作流平台，专注于将学术论文转换为多种形式的输出，包括示意图、PPT、视频、技术报告等。通过集成最新的多模态大模型和计算机视觉技术，Paper2Any 能够自动解析论文内容并生成高质量的视觉和文本输出。

### 核心优势

- 🎯 **多模态输出**：支持从论文生成示意图(Figure)、PPT、视频(Video)、技术报告(Technical Report)等多种格式
- 🔌 **模块化设计**：基于 DataFlow-Agent 框架，工作流可灵活组合和扩展
- 🎨 **高质量生成**：集成前沿的视觉生成模型和文本生成模型，确保输出质量
- ⚡ **高效处理**：支持批量处理和并行计算，快速处理大量论文
- 🔄 **灵活部署**：提供 Docker 容器化部署和本地部署选项

---

## ✨ 核心功能

### 📊 Paper2Figure
从论文中提取关键信息，自动生成高质量的示意图和图表，支持学术演示和论文插图需求。

### 📽️ Paper2PPT
基于论文内容自动生成结构化的 PowerPoint 演示文稿，包括封面、目录、内容页和参考文献页。

### 🎬 Paper2Video
将论文内容转换为讲解视频，自动生成脚本、配音和视觉内容，适合快速了解论文核心思想。

### 📝 Paper2Technical
提取论文的技术细节，生成详细的技术报告、方法描述和实现指南。

### 🔧 其他功能
- **PDF2PPT**：将现有的PDF文件转换为可编辑的PPT演示文稿
- **Paper2ExpFigure**：为论文生成实验数据图表
- **Paper2PageContent**：提取论文页面内容，用于知识库构建

---

## 🚀 新用户先看这里

这个项目目前已经不是“只有一个 Python 环境、一个 `.env`、一个启动命令”的结构。

如果你是从 GitHub clone 下来准备自己部署，请按这个顺序阅读：

1. [开源部署与配置总指南](guides/open_source_deployment.md)
2. [快速开始](quickstart.md)
3. [安装与环境准备](installation.md)
4. [配置文件参考](guides/configuration.md)

先给出最短事实：

- 前端是 `Node.js/Vite`
- 后端是 `Python/FastAPI`
- 模型服务通常也是 `Python`，可与后端共用，也可拆分
- 核心配置分为 `fastapi_app/.env`、`frontend-workflow/.env`、`deploy/profiles/*.env`

最小可运行链路通常是：

```bash
cp fastapi_app/.env.example fastapi_app/.env
cp frontend-workflow/.env.example frontend-workflow/.env

bash deploy/start.sh
bash deploy/start_frontend.sh
```

默认访问地址：

- 前端：`http://127.0.0.1:3000`
- 后端：`http://127.0.0.1:8000`

如果你还想启用本地 SAM3 / OCR / 视频 worker，再继续看 [开源部署与配置总指南](guides/open_source_deployment.md) 里的模型服务章节。

---

## 📖 文档导航

- **[开源部署与配置总指南](guides/open_source_deployment.md)** - 当前仓库最重要的部署入口文档
- **[快速开始](quickstart.md)** - 最短跑通路径
- **[安装与环境准备](installation.md)** - 环境与依赖准备
- **[配置文件参考](guides/configuration.md)** - 三类 `.env` 的职责边界
- **[功能指南总览](guides/index.md)** - 各 workflow 的用途、输入输出与阅读入口
  - [Paper2Figure](guides/paper2figure.md)
  - [Paper2PPT](guides/paper2ppt.md)
  - [Paper2Video](guides/paper2video.md)
  - [Paper2Technical](guides/paper2technical.md)
- **[CLI工具](cli.md)** - 命令行工具使用说明
- **[常见问题解答](faq.md)** - 常见问题解决方法
- **[贡献指南](contributing.md)** - 参与项目开发的指南
- **[更新日志](changelog.md)** - 版本更新记录

---

## 🏗️ 系统架构

```
Paper2Any/
├── dataflow_agent/          # 工作流引擎、agent、toolkits、底层 workflow
├── fastapi_app/             # FastAPI 后端服务与业务配置
├── frontend-workflow/       # 前端界面 (Vite + React + TypeScript)
├── deploy/                  # 前后端启动、停止、profile、日志脚本
├── script/                  # 模型服务、CLI、准备脚本
├── docs/                    # MkDocs 文档
├── tests/                   # 测试与手工验证样例
├── models/                  # 本地模型目录（SAM3、RMBG 等）
└── outputs/                 # 生成结果输出目录
```


---

## 🤝 参与贡献

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

详见 [贡献指南](contributing.md)。

---

## 📄 开源协议

本项目采用 **Apache License 2.0** 开源协议。详情请查看仓库根目录的 `LICENSE` 文件。

---

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和使用者！

特别鸣谢：
- DataFlow-Agent - 底层工作流框架
- React / Vite - 前端界面与构建工具
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 API 框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排灵感来源

---

## 📞 联系我们

- **问题反馈**: [GitHub Issues](https://github.com/OpenDCAI/Paper2Any/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/OpenDCAI/Paper2Any/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

Made with ❤️ by Paper2Any Team

</div>
