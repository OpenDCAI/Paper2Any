<div align="center">

<img src="static/new_readme/logo图.png" alt="Paper2Any Logo" width="200"/>

# Paper2Any

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FPaper2Any-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/Paper2Any)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/Paper2Any/stargazers)

English | [中文](README.md)

✨ **Focus on Paper Multimodal Workflow: One-click generation of model diagrams, technical roadmaps, experimental plots, and presentations from paper PDFs/screenshots/text** ✨

| 📄 **Universal File Support** &nbsp;|&nbsp; 🎯 **AI-Powered Generation** &nbsp;|&nbsp; 🎨 **Custom Styling** &nbsp;|&nbsp; ⚡ **Lightning Speed** |

<br>

<a href="#-quick-start" target="_self">
  <img alt="Quickstart" src="https://img.shields.io/badge/🚀-Quick_Start-2F80ED?style=for-the-badge" />
</a>
<a href="http://dcai-paper2any.nas.cpolar.cn/" target="_blank">
  <img alt="Online Demo" src="https://img.shields.io/badge/🌐-Online_Demo-56CCF2?style=for-the-badge" />
</a>
<a href="docs/" target="_blank">
  <img alt="Docs" src="https://img.shields.io/badge/📚-Docs-2D9CDB?style=for-the-badge" />
</a>
<a href="docs/contributing.md" target="_blank">
  <img alt="Contributing" src="https://img.shields.io/badge/🤝-Contributing-27AE60?style=for-the-badge" />
</a>

<br>
<br>

<img src="static/new_readme/前端页面-01.png" alt="Paper2Any Web Interface" width="100%"/>

</div>

---

## 📢 Roadmap & Announcement

> [!IMPORTANT]
> **This project is undergoing an architectural split to provide a more focused experience.**

- **[Paper2Any](https://github.com/OpenDCAI/Paper2Any)** (Current Repository):
  - Focuses on paper multimodal workflows (Paper2Figure, Paper2PPT, Paper2Video, etc.).
  - Provides researchers with one-click tools for plotting, PPT generation, and video scripting.

- **[DataFlow-Agent](https://github.com/OpenDCAI/DataFlow-Agent)** (New Repository):
  - Focuses on DataFlow operator orchestration and authoring.
  - Provides a general-purpose multi-agent dataflow processing framework and operator development tools.

---

## 📑 Table of Contents

- [🔥 News](#-news)
- [✨ Core Features](#-core-features)
- [📸 Showcase](#-showcase)
- [🚀 Quick Start](#-quick-start)
- [📂 Project Structure](#-project-structure)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)

---

## 🔥 News

> [!TIP]
> 🆕 <strong>2025-12-12 · Paper2Figure Web public beta is live</strong><br>
> One-click generation of multiple <strong>editable</strong> scientific figures (Model Architecture / Technical Roadmap / Experimental Plots)<br>
> 🌐 Online Demo: <a href="http://dcai-paper2any.nas.cpolar.cn/">http://dcai-paper2any.nas.cpolar.cn/</a>

- 2025-10-01 · Released <code>0.1.0</code> first version

---

## ✨ Core Features

> From paper PDFs / images / text to **editable** scientific figures, slide decks, video scripts, posters and more in one click.

Paper2Any currently includes the following sub-capabilities:

- **📊 Paper2Figure - Editable Scientific Figures**: One-click generation of model architecture diagrams, technical roadmaps (PPT + SVG), and experimental plots. Supports various input sources and outputs editable PPTX.
- **🎬 Paper2PPT - Editable Slide Decks**: Generate Beamer-style or open-format editable PPTs. Supports long document processing, with built-in table extraction and figure parsing capabilities.
- **🖼️ PDF2PPT - Layout Preserved Conversion**: Intelligent cutout and layout analysis to accurately convert PDFs into editable PPTX.
- **🎨 PPT Smart Beautification**: AI-based PPT layout optimization and style transfer.

---

## 📸 Showcase

### 📊 Paper2Figure: Scientific Figure Generation

<div align="center">

<br>
<img src="static/new_readme/科研绘图-01.png" width="90%"/>
<br><sub>✨ Model Architecture Diagram Generation</sub>

<br><br>
<img src="static/new_readme/技术路线图.png" width="90%"/>
<br><sub>✨ Technical Roadmap Generation</sub>

<br><br>
<img src="static/new_readme/实验数据图.png" width="90%"/>
<br><sub>✨ Experimental Plot Generation (Multiple Styles)</sub>

</div>

---

### 🎬 Paper2PPT: Paper to Presentation

<div align="center">

<br>
<img src="static/new_readme/paper2ppt案例-1.png" width="90%"/>
<br><sub>✨ Paper / Text / Topic → PPT</sub>

<br><br>
<img src="static/new_readme/paper2ppt-长文长ppt.png" width="90%"/>
<br><sub>✨ Long Document Support (40+ Slides)</sub>

<br><br>
<img src="static/new_readme/paper2ppt-表格提取功能.png" width="90%"/>
<br><sub>✨ Intelligent Table Extraction & Insertion</sub>

</div>

---

### 🖼️ PDF2PPT: Layout Preserved Conversion

<div align="center">

<br>
<img src="static/new_readme/pdf2ppt抠图.png" width="90%"/>
<br><sub>✨ Intelligent Cutout & Layout Preservation</sub>

</div>

---

### 🎨 PPT Smart Beautification

<div align="center">

<br>
<img src="static/new_readme/ppt美化-1.png" width="90%"/>
<br><sub>✨ AI-based Layout Optimization & Style Transfer</sub>

</div>

---

## 🚀 Quick Start

### Requirements

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

### 🐧 Linux Installation

> We recommend using Conda to create an isolated environment (Python 3.11).  

#### 1. Create Environment & Install Base Dependencies

```bash
# 0. Create and activate a conda environment
conda create -n paper2any python=3.11 -y
conda activate paper2any

# 1. Clone repository
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. Install base dependencies
pip install -r requirements-base.txt

# 3. Install in editable (dev) mode
pip install -e .
```

#### 2. Install Paper2Any-specific Dependencies (Required)

Paper2Any involves LaTeX rendering, vector graphics processing and PPT/PDF conversion, which require extra dependencies:

```bash
# 1. Python dependencies
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt

# 2. LaTeX engine (tectonic) - recommended via conda
conda install -c conda-forge tectonic -y

# 3. Resolve doclayout_yolo dependency conflicts (Important)
pip install doclayout_yolo --no-deps

# 4. System dependencies (Ubuntu example)
sudo apt-get update
sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
```

#### 3. Environment Configuration

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx  # Optional: if using a third-party API gateway
export MINERU_DEVICES="0,1,2,3" # Optional: MinerU task GPU resource pool
```

### 🪟 Windows Installation

> [!NOTE]  
> We recommend Linux / WSL for best experience.  
> If you need to deploy on native Windows, please refer to the [Windows Installation](#-windows-installation) section or `README_EN.md`.

---

### Launch Applications

#### 🎨 Web Frontend (Recommended)

```bash
# Start backend API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd frontend-workflow
npm install
npm run dev
```

Visit `http://localhost:3000`.

> [!TIP]
> If you don't want to deploy frontend/backend for now, you can try core features via local scripts:
> - `python script/run_paper2figure.py`: model architecture diagram generation
> - `python script/run_paper2ppt.py`: PPT generation from content
> - `python script/run_pdf2ppt_with_paddle_sam_mineru.py`: PDF2PPT

---

## 📂 Project Structure

```
Paper2Any/
├── dataflow_agent/          # Core framework code
│   ├── agentroles/         # Agent definitions
│   │   └── paper2any_agents/ # Agents specific to Paper2Any
│   ├── workflow/           # Workflow definitions
│   ├── promptstemplates/   # Prompt template library
│   └── toolkits/           # Toolkits (Figure gen, PPT gen, etc.)
├── fastapi_app/            # FastAPI backend service
├── frontend-workflow/      # Frontend workflow editor
├── static/                 # Static resources
├── script/                 # Script tools
└── tests/                  # Test cases
```

---

## 🗺️ Roadmap

<table>
<tr>
<th width="35%">Feature</th>
<th width="15%">Status</th>
<th width="50%">Sub-features</th>
</tr>
<tr>
<td><strong>📊 Paper2Figure</strong><br><sub>Editable Scientific Figures</sub></td>
<td><img src="https://img.shields.io/badge/Progress-80%25-blue?style=flat-square&logo=progress" alt="80%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Model_Architecture-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Technical_Roadmap-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/⚠-Experimental_Plots-yellow?style=flat-square" alt="WIP"/><br>
<img src="https://img.shields.io/badge/✓-Web_Frontend-success?style=flat-square" alt="Done"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2PPT</strong><br><sub>Editable Slide Decks</sub></td>
<td><img src="https://img.shields.io/badge/Progress-60%25-yellow?style=flat-square&logo=progress" alt="60%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Beamer_Style-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Long_Doc_PPT-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Table_Extraction-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Figure_Extraction-success?style=flat-square" alt="Done"/>
</td>
</tr>
<tr>
<td><strong>🖼️ PDF2PPT</strong><br><sub>Layout Preserved Conversion</sub></td>
<td><img src="https://img.shields.io/badge/Progress-90%25-green?style=flat-square&logo=progress" alt="90%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Smart_Cutout-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Layout_Preservation-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Editable_PPTX-success?style=flat-square" alt="Done"/>
</td>
</tr>
<tr>
<td><strong>🎨 PPT Beautification</strong><br><sub>Smart Layout Optimization</sub></td>
<td><img src="https://img.shields.io/badge/Progress-50%25-yellow?style=flat-square&logo=progress" alt="50%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Style_Transfer-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/⚠-Layout_Optimization-yellow?style=flat-square" alt="WIP"/>
</td>
</tr>
</table>

---

## 🤝 Contributing

We welcome all forms of contributions!

[![Issues](https://img.shields.io/badge/Issues-Submit_Bug-red?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/issues)
[![Discussions](https://img.shields.io/badge/Discussions-Feature_Request-blue?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/discussions)
[![PR](https://img.shields.io/badge/PR-Submit_Code-green?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/pulls)

---

## 📄 License

This project is licensed under [Apache License 2.0](LICENSE).

---

<div align="center">

**If this project helps you, please give us a ⭐️ Star!**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/network/members)

<br>

<img src="static/team_wechat.png" alt="DataFlow-Agent WeChat Community" width="800"/>
<br>
<sub>Scan to join the community group</sub>

<p align="center"> 
  <em> ❤️ Made with by OpenDCAI Team</em>
</p>

</div>
