<div align="center">

<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/new_logo_bgrm.png" alt="DataFlow-Agent Logo" width="180"/>

# DataFlow-Agent

[![DataFlow](https://img.shields.io/badge/DataFlow-OpenDCAI%2FDataFlow-0F9D58?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow)

<!-- **从论文与原始数据，到图表、PPT 和数据管线的一站式 AI Orchestrator** -->

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FDataFlow--Agent-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow-Agent)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)

中文 | [English](README_EN.md)

<a href="https://github.com/OpenDCAI/DataFlow-Agent#-快速开始" target="_blank">
  <img alt="Quickstart" src="https://img.shields.io/badge/🚀-快速开始-2F80ED?style=for-the-badge" />
</a>
<a href="http://dcai-paper2any.nas.cpolar.cn/" target="_blank">
  <img alt="Online Demo" src="https://img.shields.io/badge/🌐-在线体验_Paper2Any-56CCF2?style=for-the-badge" />
</a>
<a href="docs/" target="_blank">
  <img alt="Docs" src="https://img.shields.io/badge/📚-文档-2D9CDB?style=for-the-badge" />
</a>
<a href="docs/contributing.md" target="_blank">
  <img alt="Contributing" src="https://img.shields.io/badge/🤝-参与贡献-27AE60?style=for-the-badge" />
</a>

*基于 LangGraph 的多智能体工作流平台：专注论文多模态工作流，并结合 DataFlow 能力扩展到数据治理场景*

</div>

<div align="center">
  <img src="static/frontend_pages/paper2figure-1.png" alt="Paper2Figure" width="45%"/>
  <span>&nbsp;|&nbsp;</span>
  <img src="static/frontend_pages/paper2ppt-1.png" alt="Paper2PPT" width="45%"/>
</div>

---

## 📑 目录

- [🔥 News](#-news)
- [🧠 平台概览](#-平台概览)
- [✨ 核心应用](#-核心应用)
- [🚀 快速开始](#-快速开始)
- [📂 项目结构](#-项目结构)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 贡献](#-贡献)

---

## 🔥 News

> [!TIP]
> 🆕 <strong>2025-12-12 · Paper2Figure 网页端公测上线</strong><br>
> 支持一键生成多种<strong>可编辑</strong>科研绘图（模型架构图 / 技术路线图 / 实验数据图）<br>
> 🌐 在线体验：<a href="http://dcai-paper2any.nas.cpolar.cn/">http://dcai-paper2any.nas.cpolar.cn/</a>

- 2024-09-01 · 发布 <code>0.1.0</code> 首个版本（详见 <a href="docs/changelog.md">changelog</a>）

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_1.png" alt="Paper2Figure Web UI" width="49%"/>
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_2.png" alt="Paper2Figure Web UI (2)" width="49%"/>
</div>

---

## 🧠 平台概览

DataFlow-Agent 基于 LangGraph 搭建多智能体工作流，当前主要聚焦以下典型场景：

- 🎓 <strong>科研工作流（Paper2Any）</strong>：从论文 PDF / 截图 / 文本，一键生成模型示意图、技术路线图、实验图和演示文稿
- 📊 <strong>数据治理（Easy-DataFlow）</strong>：结合 <a href="https://github.com/OpenDCAI/DataFlow">OpenDCAI/DataFlow</a> 平台，从自然语言任务描述，到可执行的数据处理管线与可视化编排界面

当前主要提供两大核心应用：

- <strong>Paper2Any</strong>：论文多模态工作流（图表 / PPT / 视频脚本 / 海报）
- <strong>Easy-DataFlow</strong>：数据治理工作流与可视化管线

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
- ✅实验数据图生成 (优化中)
- ✅ 支持 PDF / 图片 / 文本输入
- ✅ 输出可编辑 PPTX 格式

</td>
<td width="50%" valign="top">

**🎬 Paper2PPT - 可编辑演示文稿**
- ✅ Beamer 版本 PPT 生成
- ✅ 开放式可编辑 PPT 生成
- ✅ PDF2PPT 转换，背景保留+可编辑

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

#### 📸 ShowCase - Paper2PPT

##### 论文 PDF 转 PPT

<table>
<tr>
<th width="25%">输入</th>
<th width="25%">输出</th>
<th width="25%">输入</th>
<th width="25%">输出</th>
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/input_1.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 论文 PDF</sub>
</td>
<td align="center">
<img src="static/paper2ppt/ouput_1.png" alt="输出：生成 PPT" width="100%"/>
<br><sub>📊 生成的 PPT</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_3.png" alt="输入：论文内容" width="100%"/>
<br><sub>📝 论文内容</sub>
</td>
<td align="center">
<img src="static/paper2ppt/ouput_3.png" alt="输出：生成 PPT" width="100%"/>
<br><sub>📊 生成的 PPT</sub>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<strong>PPT 生成</strong> - 上传论文 PDF，自动提取关键信息，生成结构化的学术汇报 PPT。
</td>
<td colspan="2" align="center">
<strong>PPT 生成</strong> - 智能分析论文内容，可以自动插入论文内部表格和插图到PPT。
</td>
</tr>
</table>

---

#### 📸 ShowCase - PDF2PPT

<!-- 占位：PDF2PPT 示例，后续可补充具体图片路径与说明 -->

<!-- <table>
<tr>
<th width="25%">输入</th>
<th width="25%">输出</th>
<th width="25%">输入</th>
<th width="25%">输出</th>
</tr>
<tr>
<td align="center">
<img src="" alt="输入：PDF 页面" width="100%"/>
<br><sub>📄 PDF 页面</sub>
</td>
<td align="center">
<img src="" alt="输出：生成 PPT 页面" width="100%"/>
<br><sub>📊 生成的 PPT 页面</sub>
</td>
<td align="center">
<img src="" alt="输入：PDF 页面" width="100%"/>
<br><sub>📄 PDF 页面</sub>
</td>
<td align="center">
<img src="" alt="输出：生成 PPT 页面" width="100%"/>
<br><sub>📊 生成的 PPT 页面</sub>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<strong>PDF2PPT</strong> - 保留原始版式和背景，将 PDF 页面转换为可编辑的 PPT 页面。
</td>
<td colspan="2" align="center">
<strong>PDF2PPT</strong> - 适合会议汇报、课堂讲解等场景，一键从论文 PDF 生成演示文稿。
</td>
</tr>
</table> -->

---

#### 📸 ShowCase - PPT Polish（PPT 智能美化）

<p><sub>🎨 <b>PPT 增色美化</b> — 基于原有 PPT 内容，智能调整风格、配色与视觉层次</sub></p>

<table>
<tr>
<th width="25%">原始 PPT</th>
<th width="25%">增色后</th>
<th width="25%">原始 PPT</th>
<th width="25%">增色后</th>
</tr>
<tr>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_orgin_1.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_polish_1.png" alt="增色后PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_orgin_2.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_polish_2.png" alt="增色后PPT" width="100%"/>
</td>
</tr>
</table>

<p><sub>✍️ <b>PPT 润色拓展</b> — 将纯文字或简易空白 PPT 智能润色，自动生成精美排版与视觉元素</sub></p>

<table>
<tr>
<th width="25%">原始 PPT</th>
<th width="25%">润色后</th>
<th width="25%">原始 PPT</th>
<th width="25%">润色后</th>
</tr>
<tr>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/orgin_3.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/polish_3.png" alt="润色后PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/orgin_4.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/polish_4.png" alt="润色后PPT" width="100%"/>
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
<th width="33%">精美实验图（可选不同风格）</th>
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
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@lz/dev/static/paper2any_imgs/p2e/paper_1_3.png" alt="输出：实验数据图（手绘风格）" width="100%"/>
  <br><sub>🎨 输出：手绘风格的实验数据图</sub>
</td>
</tr>

<tr>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2.png" alt="输入：实验结果截图" width="100%"/>
  <br><sub>📄 输入：论文 PDF / 实验结果截图</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2_2.png" alt="输出：实验数据图（基础样式）" width="100%"/>
  <br><sub>📈 输出：常规 Python 风格实验数据图</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2_3.png" alt="输出：实验数据图（卡通风格）" width="100%"/>
  <br><sub>🎨 输出：卡通风格的实验数据图</sub>
</td>
</tr>

<tr>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3.png" alt="输入：实验结果截图" width="100%"/>
  <br><sub>📄 输入：论文 PDF / 实验结果截图</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3_2.png" alt="输出：实验数据图（基础样式）" width="100%"/>
  <br><sub>📈 输出：常规 Python 风格实验数据图</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3_3.png" alt="输出：实验数据图（多边形风格）" width="100%"/>
  <br><sub>🎨 输出：多边形风格的实验数据图</sub>
</td>
</tr>

</table>

<div align="center">

上传实验结果截图或表格，自动抽取关键数据并生成**可编辑的实验数据图 PPTX**，同时提供常规和精美两种风格，便于论文和汇报复用。

</div>

---

#### 🖥️ 使用方式

**方式一：Web 前端（推荐）**

(目前在线版只支持邀请用户体验)访问在线体验地址：[http://dcai-paper2any.nas.cpolar.cn/](http://dcai-paper2any.nas.cpolar.cn/)

<div align="center">
  <img src="static/frontend_pages/paper2figure-1.png" alt="Web UI - Paper2Figure" width="48%"/>
  <span>&nbsp;|&nbsp;</span>
  <img src="static/frontend_pages/paper2ppt-1.png" alt="Web UI - Paper2PPT" width="48%"/>
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


## 🚀 快速开始

### 环境要求

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

---

### 🐧 Linux 安装

> 建议使用 Conda 创建隔离环境（推荐 Python 3.11）。  
> 下述命令以 Ubuntu 为例，其他发行版请参考对应包管理器命令。

#### 1. 创建环境并安装基础依赖

```bash
# 0. 创建并激活 conda 环境
conda create -n dataflow-agent python=3.11 -y
conda activate dataflow-agent

# 1. 克隆仓库
git clone https://github.com/OpenDCAI/DataFlow-Agent.git
cd DataFlow-Agent

# 2. 安装基础依赖
pip install -r requirements-base.txt

# 3. 开发模式安装
pip install -e .
```

#### 2. 安装 Paper2Any 相关依赖（可选但推荐）

Paper2Any 涉及 LaTeX 渲染与矢量图处理，需要额外依赖（见 `requirements-paper.txt`）：

```bash
# Python 依赖
pip install -r requirements-paper.txt

# tectonic：LaTeX 引擎（推荐用 conda 安装）
conda install -c conda-forge tectonic -y

# inkscape：用于 SVG / 矢量图处理（Ubuntu 示例）
sudo apt-get update
sudo apt-get install -y inkscape
```

##### 2.1 PPT / PDF 相关系统依赖（Paper2PPT 与 PPT 美化推荐安装）

如果你需要使用 **Paper2PPT / PPT 智能美化 / PDF2PPT** 等功能，建议在 Linux 下额外安装以下系统依赖（以 Ubuntu 为例）：

```bash
sudo apt-get update
sudo apt-get install -y libreoffice        # 用于 PPT 打开 / 转换等操作
sudo apt-get install -y poppler-utils      # 提供 pdftoppm / pdftocairo 等 PDF 工具
sudo apt-get install -y wkhtmltopdf        # HTML 转 PDF，部分版式转换场景会用到
```

#### 3. 配置环境变量

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx  # 可选：如需使用第三方 API 中转站
```

第三方 API 中转示例：

- https://api.apiyi.com/
- http://123.119.219.111:3000/

---

### 🪟 Windows 安装（预留）

> [!NOTE]
> Windows 安装说明正在整理中，后续将在此补充。  
> 目前推荐优先在 Linux / WSL 环境下体验 DataFlow-Agent。

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
>   - `python script/run_paper2figure.py`：模型架构图生成
>   - `python script/run_paper2expfigure.py`：实验数据图生成
>   - `python script/run_paper2technical.py`：技术路线图生成
>   - `python script/run_paper2ppt.py`：论文内容生成可编辑 PPT
>   - `python script/run_pdf2ppt_with_paddle_sam_mineru.py`：PDF2PPT（保留版式 + 可编辑内容）

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

##  项目结构

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

## 📐 项目架构

<div align="center">
<img src="static/projs_dist.png" alt="项目架构图" width="800"/>
<br><sub>DataFlow-Agent 延伸的核心应用：Paper2Any（论文多模态工作流）、Easy-DataFlow（数据治理管线）</sub>
</div>

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
