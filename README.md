<div align="center">

<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/new_logo_bgrm.png" alt="Paper2Any Logo" width="180"/>

# Paper2Any

<!-- **从论文与原始数据，到图表、PPT 和数据管线的一站式 AI Orchestrator** -->

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FPaper2Any-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/Paper2Any)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/Paper2Any/stargazers)

中文 | [English](README_EN.md)

<a href="#-快速开始" target="_self">
  <img alt="Quickstart" src="https://img.shields.io/badge/🚀-快速开始-2F80ED?style=for-the-badge" />
</a>
<a href="http://dcai-paper2any.nas.cpolar.cn/" target="_blank">
  <img alt="Online Demo" src="https://img.shields.io/badge/🌐-在线体验-56CCF2?style=for-the-badge" />
</a>
<a href="docs/" target="_blank">
  <img alt="Docs" src="https://img.shields.io/badge/📚-文档-2D9CDB?style=for-the-badge" />
</a>
<a href="docs/contributing.md" target="_blank">
  <img alt="Contributing" src="https://img.shields.io/badge/🤝-参与贡献-27AE60?style=for-the-badge" />
</a>

*专注论文多模态工作流：从论文 PDF/截图/文本，一键生成模型示意图、技术路线图、实验图和演示文稿*

</div>

<div align="center">
  <img src="static/frontend_pages/paper2figure-1.png" alt="Paper2Figure" width="45%"/>
  <span>&nbsp;|&nbsp;</span>
  <img src="static/frontend_pages/paper2ppt-1.png" alt="Paper2PPT" width="45%"/>
</div>

---

## 📢 Roadmap & 拆分公告

> [!IMPORTANT]
> **本项目正在进行架构拆分，以提供更专注的功能体验。**

- **[Paper2Any](https://github.com/OpenDCAI/Paper2Any)** (本仓库)：
  - 专注于论文多模态工作流（Paper2Figure, Paper2PPT, Paper2Video 等）。
  - 为科研人员提供一键式绘图、PPT 生成和视频脚本辅助工具。

- **[DataFlow-Agent](https://github.com/OpenDCAI/DataFlow-Agent)** (新仓库)：
  - 专注于 DataFlow 算子编排和编写。
  - 提供通用的多智能体数据流处理框架和算子开发工具。

---

## 📑 目录

- [🔥 News](#-news)
- [✨ 核心功能](#-核心功能)
- [📸 功能展示](#-功能展示)
- [🚀 快速开始](#-快速开始)
- [📂 项目结构](#-项目结构)
- [🗺️ 开发计划](#️-开发计划)
- [🤝 贡献](#-贡献)

---

## 🔥 News

> [!TIP]
> 🆕 <strong>2025-12-12 · Paper2Figure 网页端公测上线</strong><br>
> 支持一键生成多种<strong>可编辑</strong>科研绘图（模型架构图 / 技术路线图 / 实验数据图）<br>
> 🌐 在线体验：<a href="http://dcai-paper2any.nas.cpolar.cn/">http://dcai-paper2any.nas.cpolar.cn/</a>

- 2025-10-01 · 发布 <code>0.1.0</code> 首个版本

---

## ✨ 核心功能

> 从论文 PDF / 图片 / 文本出发，一键生成**可编辑**的科研绘图、演示文稿、视频脚本、学术海报等多模态内容。

Paper2Any 当前包含以下几个子能力：

<table>
<tr>
<td width="50%" valign="top">

**📊 Paper2Figure - 可编辑科研绘图**
- ✅ 模型架构图生成
- ✅ 技术路线图生成（PPT + SVG）
- ✅ 实验数据图生成 (优化中)
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

## 📸 功能展示

### 1. Paper2PPT - 论文转演示文稿

#### 基础生成 (Paper / Text / Topic → PPT)

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
<img src="static/paper2ppt/output_1.png" alt="输出：生成 PPT" width="100%"/>
<br><sub>📊 生成的 PPT</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_3.png" alt="输入：论文内容" width="100%"/>
<br><sub>📝 论文内容</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_3.png" alt="输出：生成 PPT" width="100%"/>
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
<tr>
<td align="center">
<img src="static/paper2ppt/input_2-1.png" alt="输入：文本 1" width="100%"/>
<br><sub>📄 输入文本 1</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_2-2.png" alt="输入：文本 2" width="100%"/>
<br><sub>📄 逐页面编辑</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_2-3.png" alt="输入：文本 3" width="100%"/>
<br><sub>📄 生成PPT页面</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_2.png" alt="输出：生成 PPT" width="100%"/>
<br><sub>📊 生成的 PPT</sub>
</td>
</tr>
<tr>
<td colspan="4" align="center">
<strong>Text2PPT</strong> - 输入长文本/大纲，自动生成结构化的 PPT。
</td>
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/input_4-1.png" alt="输入：主题 1" width="100%"/>
<br><sub>📄 输入主题 1</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_4-2.png" alt="输入：主题 2" width="100%"/>
<br><sub>📄 逐页面编辑</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_4-3.png" alt="输入：主题 3" width="100%"/>
<br><sub>📄 生成PPT页面</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_4.png" alt="输出：生成 PPT" width="100%"/>
<br><sub>📊 生成的 PPT</sub>
</td>
</tr>
<tr>
<td colspan="4" align="center">
<strong>Topic2PPT</strong> - 输入简短主题，自动扩充内容并生成 PPT。
</td>
</tr>
</table>

#### 🚀 超长文档生成 (40+ 页 PPT)

> 针对整本书籍、长篇综述或超长技术文档，支持自动分章节处理，生成 40~100 页的完整演示文稿。

<table>
<tr>
<th width="25%">输入：长篇论文/书籍</th>
<th width="25%">PPT生成（红色风格）</th>
<th width="25%">PPT生成（紫色风格）</th>
<!-- <th width="25%">最终 PPT (40+页)</th> -->
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/long_paper/input_0.png" alt="输入：长文档" width="100%"/>
<br><sub>📚 输入：研究主题，选择长文模式</sub>
</td>
<td align="center">
<img src="static/paper2ppt/long_paper/output_1.png" alt="PPT生成（红色风格）" width="100%"/>
<br><sub>📝 PPT生成（红色风格）</sub>
</td>
<td align="center">
<img src="static/paper2ppt/long_paper/output_2.png" alt="PPT生成（紫色风格）" width="100%"/>
<br><sub>🔄 PPT生成（紫色风格）</sub>
</td>
</tr>
</table>

---

#### PDF2PPT - PDF 转可编辑 PPT

<table>
<tr>
<th width="25%">输入</th>
<th width="25%">输出</th>
<th width="25%">输入</th>
<th width="25%">输出</th>
</tr>
<tr>
<td align="center">
<img src="static/pdf2ppt/input_1.png" alt="输入：PDF 页面" width="100%"/>
<br><sub>📄 PDF 页面</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/output_1.png" alt="输出：生成 PPT 页面" width="100%"/>
<br><sub>📊 生成的 PPT (白底)</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/input_2.png" alt="输入：PDF 页面" width="100%"/>
<br><sub>📄 PDF 页面</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/output_2.png" alt="输出：生成 PPT 页面" width="100%"/>
<br><sub>📊 生成的 PPT (AI重绘)</sub>
</td>
</tr>
</table>

#### PPT Polish - 智能美化

<table>
<tr>
<th width="25%">原始 PPT</th>
<th width="25%">增色后</th>
<th width="25%">原始 PPT</th>
<th width="25%">润色后</th>
</tr>
<tr>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_orgin_1.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_polish_1.png" alt="增色后PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/orgin_3.png" alt="原始PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/polish_3.png" alt="润色后PPT" width="100%"/>
</td>
</tr>
</table>

---

### 2. Paper2Figure - 科研绘图生成

#### 模型架构图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">生成图</th>
<th width="33%">PPTX 截图</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 论文 PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_3.png" alt="PPTX 截图" width="100%"/>
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
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_1.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 论文PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_3.png" alt="PPTX 截图" width="100%"/>
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
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_1.png" alt="输入：论文 PDF" width="100%"/>
<br><sub>📄 输入核心段落</sub>
</td>
<td align="center">
<img src="static/paper2any_imgs/p2f/p2f_paper_hard_img_2.png" alt="生成的模型图" width="100%"/>
<br><sub>🎨 生成的模型架构图</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_3.png" alt="PPTX 截图" width="100%"/>
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
上传论文 PDF，根据选择的难度（简单/中等/困难），自动生成<strong>可编辑 PPTX 格式</strong>模型架构图。
</div>

#### 技术路线图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">生成图（SVG）</th>
<th width="33%">PPTX 截图</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/paper1.png" alt="输入：论文文本" width="100%"/>
<br><sub>📝 方法部分（中文）</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/cn_img_1.png" alt="技术路线图 SVG" width="100%"/>
<br><sub>🗺️ 技术路线图 SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/cn_img_2.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>语言：中文</strong> - 适合国内学术交流
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/paper2.png" alt="输入：论文文本（英文）" width="100%"/>
<br><sub>📝 方法部分（英文）</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/en_img_1.png" alt="技术路线图 SVG" width="100%"/>
<br><sub>🗺️ 技术路线图 SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/en_img_2.png" alt="PPTX 截图" width="100%"/>
<br><sub>📊 可编辑 PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>语言：英文</strong> - 适合国际学术发表
</td>
</tr>
</table>

#### 实验数据图生成

<table>
<tr>
<th width="33%">输入</th>
<th width="33%">常规风格</th>
<th width="33%">手绘风格</th>
</tr>
<tr>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2e/paper_1.png" alt="输入：实验结果" width="100%"/>
  <br><sub>📄 实验结果截图</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2e/paper_1_2.png" alt="输出：标准样式" width="100%"/>
  <br><sub>📈 常规 Python 风格</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@lz/dev/static/paper2any_imgs/p2e/paper_1_3.png" alt="输出：手绘风格" width="100%"/>
  <br><sub>🎨 手绘风格</sub>
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
conda create -n paper2any python=3.11 -y
conda activate paper2any

# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. 安装基础依赖
pip install -r requirements-base.txt

# 3. 开发模式安装
pip install -e .
```

#### 2. 安装 Paper2Any 相关依赖（必须）

Paper2Any 涉及 LaTeX 渲染、矢量图处理以及 PPT/PDF 转换，需要额外依赖：

```bash
# 1. Python 依赖
# (如果 requirements-paper.txt 安装失败，可尝试 requirements-paper-backup.txt)
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt

# 2. LaTeX 引擎 (tectonic) - 推荐用 conda 安装
conda install -c conda-forge tectonic -y

# 3. 解决 doclayout_yolo 依赖冲突（重要）
# 由于 doclayout_yolo 可能与 paddleocr 存在依赖冲突（albumentations 版本不一致），建议忽略依赖检查单独安装：
pip install doclayout_yolo --no-deps

# 4. 系统依赖 (Ubuntu 示例)
# 包含：
# - inkscape: SVG / 矢量图处理
# - libreoffice: PPT 打开 / 转换
# - poppler-utils: PDF 工具 (pdftoppm / pdftocairo)
# - wkhtmltopdf: HTML 转 PDF
sudo apt-get update
sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
```

#### 3. 配置环境变量

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx  # 可选：如需使用第三方 API 中转站

# [可选] 配置 MinerU PDF 解析任务的 GPU 资源池（负载均衡）
# 指定一组可用 GPU ID（逗号分隔），PDF 解析任务会自动随机选择一张卡运行，避免拥堵。
# 默认值：5,6,7
# 这个主要用于 paper2ppt 场景下，MinerU 的解析服务
export MINERU_DEVICES="0,1,2,3"
```

第三方 API 中转示例：

- https://api.apiyi.com/
- http://123.119.219.111:3000/

<details>
<summary><b>🔧 高级配置：本地模型服务负载均衡（可选）</b></summary>

<br>

如果是本地部署高并发环境，可以使用 `script/start_model_servers.sh` 启动本地模型服务集群（MinerU / SAM / OCR），以提升 Paper2Any 在 PDF 解析和图像处理场景下的吞吐能力。

**脚本位置**：`script/start_model_servers.sh`

**主要配置项说明**（与 Paper2Any 强相关的部分）：

*   **MinerU (PDF 解析)**
    *   `MINERU_MODEL_PATH`: 模型路径 (默认 `models/MinerU2.5-2509-1.2B`)
    *   `MINERU_GPU_UTIL`: 显存占用比例 (默认 `0.2`)
    *   **实例配置**: 脚本默认在多张 GPU 上启动若干实例，端口范围通常为 `8011-8018`。
    *   **Load Balancer**: 端口 `8010`，自动分发请求。

*   **SAM (Segment Anything Model，用于图像分割)**
    *   **实例配置**: 默认在多张 GPU 上各启动 1 个实例，端口范围 `8021-8022`。
    *   **Load Balancer**: 端口 `8020`。

*   **OCR (PaddleOCR)**
    *   **配置**: 通常运行在 CPU 上，使用 uvicorn 的 worker 机制 (默认 4 workers)。
    *   **端口**: `8003`。

使用前请根据实际 GPU 数量和显存情况修改脚本中的 `gpu_id` 和实例数量。

</details>

---

### 🪟 Windows 安装

> [!NOTE]  
> 目前推荐优先在 Linux / WSL 环境下体验 Paper2Any。  
> 若你需要在原生 Windows 上部署，请按以下步骤操作。

#### 1. 创建环境并安装基础依赖

```bash
# 0. 创建并激活 conda 环境
conda create -n paper2any python=3.12 -y
conda activate paper2any

# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. 安装基础依赖
pip install -r requirements-win-base.txt

# 3. 开发模式安装
pip install -e .
```

#### 2. 安装 Paper2Any 相关依赖（必须）

Paper2Any 涉及 LaTeX 渲染与矢量图处理，需要额外依赖（见 `requirements-paper.txt`）：

```bash
# Python 依赖
pip install -r requirements-paper.txt

# tectonic：LaTeX 引擎（推荐用 conda 安装）
conda install -c conda-forge tectonic -y
```

🎨 安装 Inkscape（SVG/矢量图处理｜推荐/必装）

- 下载并安装（Windows 64-bit MSI）：  
  https://inkscape.org/release/inkscape-1.4.2/windows/64-bit/msi/?redirected=1  
  选择 **Windows Installer Package（msi）**

- 将 Inkscape 可执行文件目录加入系统环境变量 `Path`（示例）：
  - `C:\Program Files\Inkscape\bin\`

> [!TIP]  
> 配置 `Path` 后建议重新打开终端（或重启 VS Code / PowerShell），确保环境变量生效。

⚡ 安装 Windows 编译版 vLLM（可选｜用于本地 MinerU 推理加速）

- 发布页参考：https://github.com/SystemPanic/vllm-windows/releases  
- 示例版本：**0.11.0**（whl 文件名示例）

```bash
pip install vllm-0.11.0+cu124-cp312-cp312-win_amd64.whl
```

> [!IMPORTANT]  
> 请确保 `.whl` 与当前环境匹配：  
> - Python：`cp312`（Python 3.12）  
> - 平台：`win_amd64`  
> - CUDA：`cu124`（需与你本机 CUDA/驱动适配）

---

### 启动应用

> [!NOTE]
> **Paper2Any**：从论文 PDF / 图片 / 文本一键生成可编辑的科研绘图、技术路线图、实验数据图和演示文稿。

#### 🎨 Web 前端（推荐）

```bash
# 启动后端 API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# 启动前端（新终端）
cd frontend-workflow
npm install
npm run dev

# 如需要自定义前端端口或反向代理，可在 vite.config.ts 中配置 server.proxy：
# 示例：
# export default defineConfig({
#   plugins: [react()],
#   server: {
#     port: 3000,
#     open: true,
#     allowedHosts: true,
#     proxy: {
#       '/api': {
#         target: 'http://127.0.0.1:8000',  // FastAPI 后端地址
#         changeOrigin: true,
#       },
#     },
#   },
# })
```

访问 `http://localhost:3000`。

> [!TIP]
> **Paper2Figure 网页端内测说明**
> - 部署了前端之后，需要**手动新建**一个 `invite_codes.txt` 文件，并写入你的邀请码（例如：`ABCDEFG123456`）。
> - 然后再启动后端；
> - 如果暂时不想部署前后端，可以通过本地脚本直接体验 Paper2Any 的核心能力：
>   - `python script/run_paper2figure.py`：模型架构图生成
>   - `python script/run_paper2expfigure.py`：实验数据图生成
>   - `python script/run_paper2technical.py`：技术路线图生成
>   - `python script/run_paper2ppt.py`：论文内容生成可编辑 PPT
>   - `python script/run_pdf2ppt_with_paddle_sam_mineru.py`：PDF2PPT（保留版式 + 可编辑内容）

#### 🪟 Windows 下加载 MinerU 预训练模型（可选）

如果你在 Windows + GPU 环境中使用 vLLM 部署 MinerU 模型，可以参考下述命令（PowerShell 环境）：

```bash
# 加载 MinerU 预训练模型
vllm serve opendatalab/MinerU2.5-2509-1.2B `
  --host 127.0.0.1 `
  --port 8010 `
  --logits-processors mineru_vl_utils:MinerULogitsProcessor `
  --gpu-memory-utilization 0.6 `
  --trust-remote-code `
  --enforce-eager
```

> [!TIP]
> - 启动成功后，将 MinerU 推理服务地址配置到对应的环境变量或配置文件中即可被 Paper2Any 使用。
> - 端口号和 GPU 占用比例可根据实际资源进行调整。

> [!TIP]
> 如果暂时不想部署前后端，可以通过本地脚本体验核心功能：
> - `python script/run_paper2figure.py`：模型架构图生成
> - `python script/run_paper2ppt.py`：论文生成 PPT
> - `python script/run_pdf2ppt_with_paddle_sam_mineru.py`：PDF 转 PPT

---

## 📂 项目结构

```
Paper2Any/
├── dataflow_agent/          # 核心代码库
│   ├── agentroles/         # Agent 定义
│   │   └── paper2any_agents/ # Paper2Any 专用 Agent
│   ├── workflow/           # Workflow 定义
│   ├── promptstemplates/   # Prompt 模板
│   └── toolkits/           # 工具集（绘图、PPT生成等）
├── fastapi_app/            # 后端 API 服务
├── frontend-workflow/      # 前端 Web 界面
├── static/                 # 静态资源
├── script/                 # 脚本工具
└── tests/                  # 测试用例
```

---

## 🗺️ 开发计划

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
</table>

---

## 🤝 贡献

我们欢迎所有形式的贡献！

[![Issues](https://img.shields.io/badge/Issues-提交_Bug-red?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/issues)
[![Discussions](https://img.shields.io/badge/Discussions-功能建议-blue?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/discussions)
[![PR](https://img.shields.io/badge/PR-提交代码-green?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/pulls)

---

## 📄 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

本项目采用 [Apache License 2.0](LICENSE) 开源协议

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/network/members)

[提交 Issue](https://github.com/OpenDCAI/Paper2Any/issues) • [加入讨论](https://github.com/OpenDCAI/Paper2Any/discussions)

Made with ❤️ by OpenDCAI Team

</div>

---

## 🌐 加入社区

- 📮 **GitHub Issues**：用于反馈 Bug 或提交功能建议  
  👉 https://github.com/OpenDCAI/Paper2Any/issues
- 💬 **社区交流群**：与开发者和贡献者实时交流

<div align="center">
  <img src="static/team_wechat.png" alt="DataFlow-Agent 社区微信群" width="560"/>
  <br>
  <sub>扫码加入社区微信群</sub>
</div>
