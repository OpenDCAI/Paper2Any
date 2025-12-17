<div align="center">

<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/new_logo_bgrm.png" alt="DataFlow-Agent Logo" width="200"/><br>

# DataFlow-Agent

**AI-Powered Data & Paper Workflow Orchestration Platform**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FDataFlow--Agent-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow-Agent)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)

<a href="https://dcai-paper2any.cpolar.top/" target="_blank">
  <img alt="Try Online" src="https://img.shields.io/badge/Try%20Online-Paper2Figure-56CCF2?style=flat-square&logo=googlechrome&logoColor=white" />
</a>
<a href="docs/" target="_blank">
  <img alt="Docs" src="https://img.shields.io/badge/Docs-Read%20the%20Docs-2D9CDB?style=flat-square&logo=readthedocs&logoColor=white" />
</a>
<a href="docs/contributing.md" target="_blank">
  <img alt="Contributing" src="https://img.shields.io/badge/Contributing-Guide-27AE60?style=flat-square&logo=github&logoColor=white" />
</a>

*基于DFA的AI Agent多功能平台*

中文 | [English](README_EN.md)

[快速开始](#-快速开始) • <a href="https://dcai-paper2any.cpolar.top/" target="_blank">在线体验</a> • <a href="docs/" target="_blank">文档</a> • <a href="docs/contributing.md" target="_blank">贡献指南</a>

</div>

---

## 📑 目录

- [🔥 News](#-news)
- [📐 项目架构](#-项目架构)
- [✨ 核心应用](#-核心应用)
  - [Paper2Any - 论文多模态工作流](#1️⃣-paper2any---论文多模态工作流)
  - [Easy-DataFlow - 数据治理管线](#2️⃣-easy-dataflow---数据治理管线)
  - [DataFlow-Table - 多源数据分析](#3️⃣-dataflow-table---多源数据分析)
- [🚀 快速开始](#-快速开始)
- [📂 项目结构](#-项目结构)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 贡献](#-贡献)

---

## 🔥 News

<table>
<tr>
<td width="120"><strong>2025.12.12</strong></td>
<td>
🎉 <strong>Paper2Figure 网页端内测上线</strong>（需邀请码）<br>
支持一键生成多种<strong>可编辑</strong>科研绘图，包括模型架构图、技术路线图、实验数据图<br>
📧 <strong>邮箱联系获取体验邀请码</strong>（仅限前50名）：<a href="mailto:liuzhou225919@163.com">liuzhou225919@163.com</a><br>
👉 访问地址：<a href="https://dcai-paper2any.cpolar.top/">https://dcai-paper2any.cpolar.top/</a>
</td>
</tr>
<tr>
<td><strong>2024.09.01</strong></td>
<td>
🚀 发布 <code>0.1.0</code> 首个版本（详见 <a href="docs/changelog.md">changelog</a>）
</td>
</tr>
</table>

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_1.png" alt="Paper2Figure Web UI" width="49%"/>
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_2.png" alt="Paper2Figure Web UI (2)" width="49%"/>
</div>

---

## 📐 项目架构

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/projs_dist.png" alt="项目架构图" width="800"/>
<br><sub>DataFlow-Agent 延伸的三个核心应用：Paper2Any（论文多模态工作流）、Easy-DataFlow（数据治理管线）、DataFlow-Table（多源数据分析）</sub>
</div>

---

## ✨ 核心应用

### 1️⃣ Paper2Any - 论文多模态工作流

> 从论文 PDF / 图片 / 文本出发，一键生成**可编辑**的科研绘图、演示文稿、视频脚本、学术海报等多模态内容。

#### 🎯 核心功能

Paper2Any 当前包含以下几个子能力：

<table>
<tr>
<td width="50%" valign="top">

**📊 Paper2Figure - 可编辑科研绘图**
- ✅ 模型架构图生成
- ✅ 技术路线图生成（PPT + SVG）
- 🚧 实验数据图生成 (优化中)
- ✅ 支持 PDF / 图片 / 文本输入
- ✅ 输出可编辑 PPTX 格式

</td>
<td width="50%" valign="top">

**🎬 Paper2PPT - 可编辑演示文稿**
- ✅ Beamer 版本 PPT 生成
- 🚧 开放式可编辑 PPT 生成（开发中）

</td>
</tr>
<tr>
<td valign="top">

**🎬 Paper2Video - 论文讲解视频**
- 🚧 自动生成视频脚本
- 🚧 分镜描述与时间轴
- 🚧 配图素材推荐
- 🚧 视频自动合成（开发中）

</td>
<td valign="top">

**📌 Paper2Poster - 可编辑学术海报**
- 🚧 版式自动设计
- 🚧 要点提炼
- 🚧 视觉优化

</td>
</tr>
</table>

---

#### 📸 ShowCase - Paper2Figure

##### 模型架构图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">生成图</th>
<th width="33%">PPTX 截图</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 论文 PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_3.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>绘图难度：简单</strong> - 基础模型结构，清晰的模块划分
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_1.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 论文PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_3.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>绘图难度：中等</strong> - 包含多层次结构和数据流
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_1.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 输入核心段落</sub>
</td>
<td align="center">
<img src="static/paper2any_imgs/p2f/p2f_paper_hard_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_3.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>绘图难度：困难</strong> - 复杂的多模块交互和详细注释
</td>
</tr>
</table>

<div align="center">

上传论文 PDF 文件，根据选择的**绘图难度**（简单/中等/困难），自动提取模型架构信息，生成对应复杂度的**可编辑 PPTX 格式**模型架构图。

</div>

---

##### 技术路线图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">生成图（SVG）</th>
<th width="33%">PPTX 截图</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/paper1.png" alt="输入：论文文本（中文）" width="100%"/>
<br><sub>📝 论文方法部分（中文）</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/cn_img_1.png" alt="技术路线图 SVG" width="100%"/>
<br><sub>🗺️ 技术路线图 SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/cn_img_2.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>语言：中文</strong> - 中文技术路线图，适合国内学术交流
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/paper2.png" alt="输入：论文文本（英文）" width="100%"/>
<br><sub>📝 论文方法部分（英文）</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/en_img_1.png" alt="技术路线图 SVG" width="100%"/>
<br><sub>🗺️ 技术路线图 SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/en_img_2.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>语言：英文</strong> - 英文技术路线图，适合国际学术发表
</td>
</tr>
</table>

<div align="center">

粘贴论文方法部分文本，选择**语言**（中文/英文），自动梳理技术路线与模块依赖关系，生成**清晰的技术路线图 PPTX 与可编辑 SVG**。

</div>

---

##### 实验数据图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">常规实验图</th>
<th width="33%">精美实验图</th>
</tr>
<tr>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1.png" alt="输入：实验结果截图" width="100%"/>
  <br><sub>📄 输入：论文 PDF / 实验结果截图</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1_2.png" alt="输出：实验数据图（基础样式）" width="100%"/>
  <br><sub>📈 输出：常规 Python 风格实验数据图</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1_3.png" alt="输出：实验数据图（精美样式）" width="100%"/>
  <br><sub>🎨 输出：精美排版的实验数据图</sub>
</td>
</tr>
</table>

<div align="center">

上传实验结果截图或表格，自动抽取关键数据并生成**可编辑的实验数据图 PPTX**，同时提供常规和精美两种风格，便于论文和汇报复用。

</div>

---

#### 🖥️ 使用方式

**方式一：Web 前端（推荐）**

(目前在线版只支持邀请用户体验)访问在线体验地址：[https://dcai-paper2any.cpolar.top/](https://dcai-paper2any.cpolar.top/)

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend.png" alt="前端界面" width="100%"/>
</div>

**特点**：
- 🎨 现代化 UI 设计
- 📤 支持拖拽上传
- ⚙️ 可视化参数配置
- 📊 实时进度展示
- 📥 一键下载结果

<!-- 
**方式二：Gradio 界面**

```bash
python gradio_app/app.py
```

访问 `http://127.0.0.1:7860`

**特点**：
- 🚀 快速部署
- 🔧 灵活配置
- 📝 支持批量处理
-->

---

### 2️⃣ Easy-DataFlow - 数据治理管线

> 从任务描述到可执行数据处理管线，AI 驱动的数据治理全流程

#### 🎯 核心功能

| 功能模块 | 说明 | 状态 |
|---------|------|------|
| 📊 **管线推荐** | 从任务描述自动生成可执行 Python 管线代码 | ✅ |
| ✍️ **算子编写** | AI 辅助编写自定义数据处理算子 | ✅ |
| 🎨 **可视化编排** | 拖拽式构建数据处理流程 | ✅ |
| 🔄 **Prompt 优化** | 自动优化算子提示词，提升效果 | ✅ |
| 🌐 **Web 采集** | 自动化网页数据采集与结构化 | ✅ |

---

#### 📸 功能展示

**管线推荐：从任务到代码**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/imag_piperec.png" alt="管线推荐" width="50%"/>
<br><sub>💻 智能分析任务需求，自动推荐最优算子组合，生成可执行的 Python 管线代码</sub>
</div>

---

**算子编写：AI 辅助开发**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/image_opwrite.png" alt="算子编写" width="50%"/>
<br><sub>⚙️ 使用 LLM 辅助从功能描述自动生成算子代码，并在同一界面内完成测试与调试</sub>
</div>

---

**可视化编排：拖拽式构建**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/image.png" alt="可视化编排" width="50%"/>
<br><sub>🎨 通过可视化界面拖拽组合算子，自由搭建数据处理流程，所见即所得</sub>
</div>

---

**Prompt 优化：自动调优**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/promptagent.png" alt="Prompt 优化" width="50%"/>
<br><sub>✨ 复用现有算子，自动书写 DataFlow 的算子 Prompt Template，智能优化提示词</sub>
</div>

---

**Web 采集：网页到数据**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/web_collection.png" alt="Web 采集" width="50%"/>
<br><sub>📊 自动化网页数据采集与结构化转换，直接输出 DataFlow-ready 数据</sub>
</div>

---

### 3️⃣ DataFlow-Table - 多源数据分析

> 一站式接入多源数据，自动化分析与洞察生成

#### 🚧 正在开发中

DataFlow-Table 正在积极开发中，敬请期待！

**计划功能**：
- 📥 多数据源接入（数据库 / 文件 / Web / API）
- 🧹 智能清洗与标准化
- 📊 基于 AI 的自动分析
- 📝 自然语言分析报告生成
- 📈 交互式图表与报表

---

## 🚀 快速开始

### 环境要求

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

### 安装步骤

> 建议使用 Conda 创建隔离环境（推荐 Python 3.12）。

```bash
# 0. 创建并激活 conda 环境
conda create -n dataflow-agent python=3.12 -y
conda activate dataflow-agent

# 1. 克隆仓库
git clone https://github.com/OpenDCAI/DataFlow-Agent.git
cd DataFlow-Agent

# 2. 安装依赖（基础）
pip install -r requirements.txt

# 3. 安装包（开发模式）
pip install -e .
```

#### Paper2Any 额外依赖（可选，但推荐）

Paper2Any 需要额外依赖（见 `requirements-paper.txt`），以及一些系统/conda 工具用于渲染与矢量图处理：

```bash
# 安装 Paper2Any 依赖
pip install -r requirements-paper.txt

# tectonic：建议用 conda 单独安装（LaTeX 引擎）
conda install -c conda-forge tectonic -y

# inkscape：用于 SVG/矢量图相关处理（Linux）
sudo apt-get update
sudo apt-get install -y inkscape
```

### 配置环境

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx 
# 如果需要使用第三方API中转站
```
第三方API中转站：

[https://api.apiyi.com/](https://api.apiyi.com/)

[http://123.129.219.111:3000/](http://123.129.219.111:3000/)

---

### 启动应用

> [!NOTE]
> **Paper2Any**：从论文 PDF / 图片 / 文本一键生成可编辑的科研绘图、技术路线图、实验数据图和演示文稿。

#### 🎨 Paper2Any - 论文工作流

**Web 前端（推荐）**

```bash
# 启动后端 API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# 启动前端（新终端）
cd frontend-workflow
npm install
npm run dev

# 配置dev/DataFlow-Agent/frontend-workflow/vite.config.ts
# 修改 server.proxy 为：
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',  // FastAPI 后端地址
        changeOrigin: true,
      },
    },
  },
})
```

访问 `http://localhost:3000`

> [!TIP]
> **Paper2Figure 网页端内测说明**
> - 当你部署了前端，还需要**手动新建**一个 `invite_codes.txt` 文件，并写入你的邀请码（例如：`ABCDEFG123456`）。
> - 然后再启动后端；
> - 如果暂时不想部署前后端，可以先通过本地脚本体验 Paper2Any 的核心能力：
>   - `python tests/test_paper2figure_with_sam.py`：模型架构图生成
>   - `python tests/test_paper2technical.py`：技术路线图生成
>   - `python tests/test_paper2video.py`：Beamer PPT 生成

**特点**：
- ✨ 现代化 UI 设计
- 🎯 可视化参数配置
- 📊 实时进度展示
- 📥 一键下载结果

---

> [!NOTE]
> **Easy-DataFlow**：从自然语言任务描述出发，自动推荐算子与管线结构，生成可执行的数据处理管线。

#### 📊 Easy-DataFlow - 数据治理

**Gradio Web 界面**

```bash
python gradio_app/app.py
```

访问 `http://127.0.0.1:7860`

**特点**：
- 🚀 快速部署
- 🔧 灵活配置
- 📝 支持批量处理
---

> [!NOTE]
> **DataFlow-Table**：面向多源数据接入与探索式分析，目前仍在开发中。

#### 🔍 DataFlow-Table - 数据分析

🚧 **正在开发中，敬请期待！**

---

## 📂 项目结构

```
DataFlow-Agent/
├── dataflow_agent/          # 核心框架代码
│   ├── agentroles/         # Agent 定义（@register 自动注册）
│   ├── workflow/           # Workflow 定义（wf_*.py）
│   ├── promptstemplates/   # Prompt 模板库
│   ├── toolkits/           # 工具集（LLM/Docker/Image 等）
│   ├── graphbuilder/       # StateGraph 构建器
│   └── states/             # 状态管理
├── gradio_app/             # Gradio Web 界面
│   ├── app.py             # 主程序
│   └── pages/             # 页面模块（自动发现）
├── fastapi_app/            # FastAPI 后端服务
│   ├── main.py            # API 入口
│   └── routers/           # 路由模块
├── frontend-workflow/      # 前端工作流编辑器
│   ├── src/               # 源代码
│   └── public/            # 静态资源
├── docs/                   # 文档
├── static/                 # 静态资源（图片等）
├── script/                 # 脚本工具
└── tests/                  # 测试用例
```

---

## 🗺️ Roadmap

### 🎓 Paper 系列

<table>
<tr>
<th width="35%">功能</th>
<th width="15%">状态</th>
<th width="50%">子功能</th>
</tr>
<tr>
<td><strong>📊 Paper2Figure</strong><br><sub>可编辑科研绘图</sub></td>
<td><img src="https://img.shields.io/badge/进度-75%25-blue?style=flat-square&logo=progress" alt="75%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-模型架构图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-技术路线图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/⚠-实验数据图-yellow?style=flat-square" alt="进行中"/><br>
<img src="https://img.shields.io/badge/✓-Web_前端-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2Video</strong><br><sub>论文讲解视频</sub></td>
<td><img src="https://img.shields.io/badge/进度-25%25-orange?style=flat-square&logo=progress" alt="25%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-视频脚本生成-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/○-分镜描述-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-配图素材-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-视频合成-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2PPT</strong><br><sub>可编辑演示文稿</sub></td>
<td><img src="https://img.shields.io/badge/进度-50%25-yellow?style=flat-square&logo=progress" alt="50%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Beamer_样式-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/⚠-可编辑_PPTX-yellow?style=flat-square" alt="进行中"/>
</td>
</tr>
<tr>
<td><strong>📌 Paper2Poster</strong><br><sub>可编辑学术海报</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-版式设计-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-要点提炼-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-视觉优化-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
<tr>
<td><strong>🧪 Paper2Exp</strong><br><sub>自动实验运行器</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-代码生成-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-环境部署-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-自动执行-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
<tr>
<td><strong>📚 PaperCiter</strong><br><sub>智能文献引用</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-文献检索-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-自动格式化-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
</table>

---

### 📊 Data 系列

<table>
<tr>
<th width="35%">功能</th>
<th width="15%">状态</th>
<th width="50%">子功能</th>
</tr>
<tr>
<td><strong>🔄 Easy-DataFlow</strong><br><sub>数据治理管线</sub></td>
<td><img src="https://img.shields.io/badge/进度-100%25-success?style=flat-square&logo=progress" alt="100%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-管线推荐-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-算子编写-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-可视化编排-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-Prompt_优化-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-Web_采集-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>📊 DataFlow-Table</strong><br><sub>多源数据分析</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-多源接入-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-智能取数-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-血缘追踪-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-高级可视化-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
</table>

---

### 🛠️ 工具增强

<table>
<tr>
<th width="35%">功能</th>
<th width="15%">状态</th>
<th width="50%">子功能</th>
</tr>
<tr>
<td><strong>🎨 Workflow 可视化编辑器</strong><br><sub>拖拽式工作流构建器</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-拖拽界面-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-5_种_Agent_模式-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-20+_预设节点-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
<tr>
<td><strong>💾 轨迹数据导出</strong><br><sub>训练数据导出</sub></td>
<td><img src="https://img.shields.io/badge/状态-开发中-lightgrey?style=flat-square" alt="开发中"/></td>
<td>
<img src="https://img.shields.io/badge/○-JSON/JSONL_格式-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-SFT_格式-lightgrey?style=flat-square" alt="开发中"/><br>
<img src="https://img.shields.io/badge/○-DPO_格式-lightgrey?style=flat-square" alt="开发中"/>
</td>
</tr>
</table>

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/dfa_fronted.png" width="800" alt="Workflow Editor"/>
<br><sub>🎨 Workflow 可视化编辑器预览</sub>
</div>

---

## 🤝 贡献

我们欢迎所有形式的贡献！

[![Issues](https://img.shields.io/badge/Issues-提交_Bug-red?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/issues)
[![Discussions](https://img.shields.io/badge/Discussions-功能建议-blue?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/discussions)
[![PR](https://img.shields.io/badge/PR-提交代码-green?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/pulls)

详细贡献指南：[docs/contributing.md](docs/contributing.md)

---

## 📄 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

本项目采用 [Apache License 2.0](LICENSE) 开源协议

---

## 🙏 致谢

感谢所有贡献者！特别感谢 [OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow) 上游项目

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=social)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/DataFlow-Agent?style=social)](https://github.com/OpenDCAI/DataFlow-Agent/network/members)

[提交 Issue](https://github.com/OpenDCAI/DataFlow-Agent/issues) • [查看文档](docs/) • [加入讨论](https://github.com/OpenDCAI/DataFlow-Agent/discussions)

Made with ❤️ by OpenDCAI Team

</div>

---

## 🌐 加入社区

加入 DataFlow 开源社区，一起交流想法、反馈问题、共建生态！

- 📮 **GitHub Issues**：用于反馈 Bug 或提交功能建议  
  👉 https://github.com/OpenDCAI/DataFlow-Agent/issues
- 🔧 **GitHub Pull Requests**：提交代码改进与文档优化  
  👉 https://github.com/OpenDCAI/DataFlow-Agent/pulls
- 💬 **社区交流群**：与开发者和贡献者实时交流

<div align="center">
  <img src="static/team_wechat.png" alt="DataFlow-Agent 社区微信群" width="560"/>
  <br>
  <sub>扫码加入 DataFlow-Agent 社区微信群</sub>
</div>
