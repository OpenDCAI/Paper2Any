<div align="center">

<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/new_logo_bgrm.png" alt="DataFlow-Agent Logo" width="200"/><br>

# DataFlow-Agent

**AI-Powered Data & Paper Workflow Orchestration Platform**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FDataFlow--Agent-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow-Agent)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)

<a href="http://dcai-paper2figure.cpolar.top/" target="_blank">
  <img alt="Try Online" src="https://img.shields.io/badge/Try%20Online-Paper2Figure-56CCF2?style=flat-square&logo=googlechrome&logoColor=white" />
</a>
<a href="docs/" target="_blank">
  <img alt="Docs" src="https://img.shields.io/badge/Docs-Read%20the%20Docs-2D9CDB?style=flat-square&logo=readthedocs&logoColor=white" />
</a>
<a href="docs/contributing.md" target="_blank">
  <img alt="Contributing" src="https://img.shields.io/badge/Contributing-Guide-27AE60?style=flat-square&logo=github&logoColor=white" />
</a>

*A multi-purpose AI agent platform based on DFA*

English | [中文](README.md)

[Quick Start](#-quick-start) • <a href="http://dcai-paper2figure.cpolar.top/" target="_blank">Try Online</a> • <a href="docs/" target="_blank">Docs</a> • <a href="docs/contributing.md" target="_blank">Contributing</a>

</div>

---

## 📑 Table of Contents

- [🔥 News](#-news)
- [📐 Architecture](#-architecture)
- [✨ Core Applications](#-core-applications)
  - [Paper2Any - Paper Multimodal Workflow](#1%EF%B8%8F%E2%83%A3-paper2any---paper-multimodal-workflow)
  - [Easy-DataFlow - Data Governance Pipeline](#2%EF%B8%8F%E2%83%A3-easy-dataflow---data-governance-pipeline)
  - [DataFlow-Table - Multi-source Data Analysis](#3%EF%B8%8F%E2%83%A3-dataflow-table---multi-source-data-analysis)
- [🚀 Quick Start](#-quick-start)
- [📂 Project Structure](#-project-structure)
- [🗺️ Roadmap](#%EF%B8%8F-roadmap)
- [🤝 Contributing](#-contributing)

---

## 🔥 News

<table>
<tr>
<td width="120"><strong>2025.12.12</strong></td>
<td>
🎉 <strong>Paper2Figure Web beta is live</strong> (invitation code required)<br>
One-click generation of multiple <strong>editable</strong> scientific figures, including model architecture diagrams, technical roadmap diagrams, and experimental plots.<br>
📧 <strong>Contact via email for invitation code</strong> (limited to first 50 users): <a href="mailto:liuzhou225919@163.com">liuzhou225919@163.com</a><br>
👉 URL: <a href="http://dcai-paper2figure.cpolar.top/">http://dcai-paper2figure.cpolar.top/</a>
</td>
</tr>
<tr>
<td><strong>2024.09.01</strong></td>
<td>
🚀 Released <code>0.1.0</code> (see <a href="docs/changelog.md">changelog</a>)
</td>
</tr>
</table>

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_1.png" alt="Paper2Figure Web UI" width="49%"/>
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend_2.png" alt="Paper2Figure Web UI (2)" width="49%"/>
</div>

---

## 📐 Architecture

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/projs_dist.png" alt="Project Architecture" width="800"/>
<br><sub>Three core applications extended from DataFlow-Agent: Paper2Any (Paper Multimodal Workflow), Easy-DataFlow (Data Governance Pipeline), DataFlow-Table (Multi-source Data Analysis)</sub>
</div>

---

## ✨ Core Applications

### 1️⃣ Paper2Any - Paper Multimodal Workflow

> Starting from a paper PDF / image / text, generate **editable** multimodal outputs such as scientific figures, slide decks, video scripts, academic posters, and more.

#### 🎯 Key Capabilities

Paper2Any currently includes the following sub-modules:

<table>
<tr>
<td width="50%" valign="top">

**📊 Paper2Figure - Editable Scientific Figures**
- ✅ Model architecture diagram generation
- ✅ Technical roadmap diagram generation (PPT + SVG)
- 🚧 Experimental plot generation (improving)
- ✅ Supports PDF / image / text inputs
- ✅ Editable PPTX output

</td>
<td width="50%" valign="top">

**🎬 Paper2PPT - Editable Slide Decks**
- ✅ Beamer-based slide deck generation
- 🚧 Fully editable slide deck generation (in progress)

</td>
</tr>
<tr>
<td valign="top">

**🎬 Paper2Video - Paper Explanation Videos**
- 🚧 Script generation
- 🚧 Storyboard descriptions & timeline
- 🚧 Visual material recommendations
- 🚧 Video auto composition (in progress)

</td>
<td valign="top">

**📌 Paper2Poster - Editable Academic Posters**
- 🚧 Layout auto-design
- 🚧 Key point summarization
- 🚧 Visual refinement

</td>
</tr>
</table>

---

#### 📸 Showcase - Paper2Figure

##### Model Architecture Diagram Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Generated Figure</th>
<th width="33%">PPTX Screenshot</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img.png" alt="Input: paper PDF" width="100%"/>
<br><sub>📄 Paper PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_3.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Difficulty: Easy</strong> - Clean modular structure
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_1.png" alt="Input: paper PDF" width="100%"/>
<br><sub>📄 Paper PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_3.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Difficulty: Medium</strong> - Multi-level structure and data flows
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_1.png" alt="Input: key paragraphs" width="100%"/>
<br><sub>📄 Input key paragraphs</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_3.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Difficulty: Hard</strong> - Complex interactions and detailed annotations
</td>
</tr>
</table>

<div align="center">

Upload a paper PDF and choose the diagram difficulty (Easy/Medium/Hard). The system extracts architecture information and generates an **editable PPTX** diagram at the selected complexity.

</div>

---

##### Technical Roadmap Diagram Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Generated Figure (SVG)</th>
<th width="33%">PPTX Screenshot</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/paper1.png" alt="Input: paper text (Chinese)" width="100%"/>
<br><sub>📝 Method section (Chinese)</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/cn_img_1.png" alt="Roadmap diagram SVG" width="100%"/>
<br><sub>🗺️ Roadmap diagram SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/cn_img_2.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Language: Chinese</strong> - Ideal for Chinese academic communications
</td>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/paper2.png" alt="Input: paper text (English)" width="100%"/>
<br><sub>📝 Method section (English)</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/en_img_1.png" alt="Roadmap diagram SVG" width="100%"/>
<br><sub>🗺️ Roadmap diagram SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2t/en_img_2.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Language: English</strong> - Ideal for international publications
</td>
</tr>
</table>

<div align="center">

Paste the method section and select the language (Chinese/English). The system organizes modules and dependencies and generates a clean **PPTX roadmap** plus an **editable SVG**.

</div>

---

##### Experimental Plot Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Standard Style</th>
<th width="33%">Polished Style</th>
</tr>
<tr>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1.png" alt="Input: experimental results screenshot" width="100%"/>
  <br><sub>📄 Input: paper PDF / results screenshot</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1_2.png" alt="Output: standard plot" width="100%"/>
  <br><sub>📈 Output: standard Python-style plot</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/paper2any_imgs/p2e/paper_1_3.png" alt="Output: polished plot" width="100%"/>
  <br><sub>🎨 Output: publication-ready styled plot</sub>
</td>
</tr>
</table>

<div align="center">

Upload an experimental results screenshot/table. The system extracts key numbers and generates an **editable PPTX plot**, offering both standard and polished styles for papers and presentations.

</div>

---

#### 🖥️ How to Use

**Option 1: Web Frontend (Recommended)**

The online version is currently available for invited users only:  
[http://dcai-paper2figure.cpolar.top/](http://dcai-paper2figure.cpolar.top/)

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/p2f_frontend.png" alt="Web UI" width="100%"/>
</div>

**Highlights**:
- 🎨 Modern UI
- 📤 Drag & drop upload
- ⚙️ Visual parameter configuration
- 📊 Real-time progress
- 📥 One-click download

<!--
**Option 2: Gradio UI**

```bash
python gradio_app/app.py
```

Open `http://127.0.0.1:7860`

**Highlights**:
- 🚀 Fast deployment
- 🔧 Flexible configuration
- 📝 Batch processing
-->

---

### 2️⃣ Easy-DataFlow - Data Governance Pipeline

> From task description to executable pipelines: an AI-powered end-to-end data governance workflow.

#### 🎯 Key Features

| Module | Description | Status |
|-------|-------------|--------|
| 📊 **Pipeline Recommendation** | Generate executable Python pipeline code from task description | ✅ |
| ✍️ **Operator Authoring** | AI-assisted development of custom data operators | ✅ |
| 🎨 **Visual Orchestration** | Drag-and-drop pipeline composition | ✅ |
| 🔄 **Prompt Optimization** | Automatically refine prompts to improve operator performance | ✅ |
| 🌐 **Web Collection** | Automated web data collection and structuring | ✅ |

---

#### 📸 Feature Demos

**Pipeline Recommendation: From task to code**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/imag_piperec.png" alt="Pipeline recommendation" width="50%"/>
<br><sub>💻 Analyze requirements and generate an optimal operator chain with runnable Python pipeline code</sub>
</div>

---

**Operator Authoring: AI-assisted development**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/image_opwrite.png" alt="Operator authoring" width="50%"/>
<br><sub>⚙️ Generate operator code from functional descriptions and test/debug in the same UI</sub>
</div>

---

**Visual Orchestration: Drag-and-drop**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/image.png" alt="Visual orchestration" width="50%"/>
<br><sub>🎨 Build pipelines visually by composing operators with a WYSIWYG interface</sub>
</div>

---

**Prompt Optimization: Automatic tuning**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/promptagent.png" alt="Prompt optimization" width="50%"/>
<br><sub>✨ Reuse existing operators to auto-generate DataFlow prompt templates and optimize prompts</sub>
</div>

---

**Web Collection: Web to data**

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/web_collection.png" alt="Web collection" width="50%"/>
<br><sub>📊 Automate web collection & structuring into DataFlow-ready datasets</sub>
</div>

---

### 3️⃣ DataFlow-Table - Multi-source Data Analysis

> Connect to multiple data sources and generate automated analysis and insights.

#### 🚧 Work in Progress

DataFlow-Table is under active development. Stay tuned!

**Working features**:
- 📥 Multi-source ingestion (DB / files / web / API)
- 🧹 Intelligent cleaning & normalization
- 📊 AI-driven automated analysis
- 📝 Natural-language reports
- 📈 Interactive charts & dashboards

---

## 🚀 Quick Start

### Requirements

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

### Installation

> We recommend using Conda to create an isolated environment (Python 3.12+).

```bash
# 0. Create and activate a conda environment
conda create -n dataflow-agent python=3.12 -y
conda activate dataflow-agent

# 1. Clone repository
git clone https://github.com/OpenDCAI/DataFlow-Agent.git
cd DataFlow-Agent

# 2. Install dependencies (base)
pip install -r requirements.txt

# 3. Install package (editable / dev mode)
pip install -e .
```

#### Paper2Any extra dependencies (optional but recommended)

Paper2Any requires extra Python dependencies (see `requirements-paper.txt`) and a few system/conda tools for rendering and vector graphics processing:

```bash
# Install Paper2Any dependencies
pip install -r requirements-paper.txt

# tectonic: recommended to install via conda (LaTeX engine)
conda install -c conda-forge tectonic -y

# inkscape: for SVG/vector graphics processing (Linux)
sudo apt-get update
sudo apt-get install -y inkscape
```

### Environment Configuration

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx 
# If using third-party API gateway
```

Third-party API gateways:

[https://api.apiyi.com/](https://api.apiyi.com/)

[http://123.129.219.111:3000/](http://123.129.219.111:3000/)

---

### Launch Applications

> [!NOTE]
> **Paper2Any**: Generate editable scientific figures, technical roadmaps, experimental plots, and presentations from paper PDFs / images / text.

#### 🎨 Paper2Any - Paper Workflow

**Web Frontend (Recommended)**

```bash
# Start backend API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd frontend-workflow
npm install
npm run dev

# Configure dev/DataFlow-Agent/frontend-workflow/vite.config.ts
# Modify server.proxy to:
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',  // FastAPI backend address
        changeOrigin: true,
      },
    },
  },
})
```

Visit `http://localhost:3000`

**Features**:
- ✨ Modern UI design
- 🎯 Visual parameter configuration
- 📊 Real-time progress tracking
- 📥 One-click download

---

> [!NOTE]
> **Easy-DataFlow**: From natural language task descriptions, automatically recommend operators and pipeline structures, generating executable data processing pipelines.

#### 📊 Easy-DataFlow - Data Governance

**Gradio Web Interface**

```bash
python gradio_app/app.py
```

Visit `http://127.0.0.1:7860`

**Features**:
- 🚀 Fast deployment
- 🔧 Flexible configuration
- 📝 Batch processing support

---

> [!NOTE]
> **DataFlow-Table**: For multi-source data ingestion and exploratory analysis, currently under development.

#### 🔍 DataFlow-Table - Data Analysis

🚧 **Under development, stay tuned!**

---

## 📂 Project Structure

```
DataFlow-Agent/
├── dataflow_agent/          # Core framework code
│   ├── agentroles/         # Agent definitions (@register auto-registration)
│   ├── workflow/           # Workflow definitions (wf_*.py)
│   ├── promptstemplates/   # Prompt template library
│   ├── toolkits/           # Toolkits (LLM/Docker/Image, etc.)
│   ├── graphbuilder/       # StateGraph builder
│   └── states/             # State management
├── gradio_app/             # Gradio Web interface
│   ├── app.py             # Main program
│   └── pages/             # Page modules (auto-discovery)
├── fastapi_app/            # FastAPI backend service
│   ├── main.py            # API entry point
│   └── routers/           # Router modules
├── frontend-workflow/      # Frontend workflow editor
│   ├── src/               # Source code
│   └── public/            # Static assets
├── docs/                   # Documentation
├── static/                 # Static resources (images, etc.)
├── script/                 # Script tools
└── tests/                  # Test cases
```

---

## 🗺️ Roadmap

### 🎓 Paper Series

<table>
<tr>
<th width="35%">Feature</th>
<th width="15%">Status</th>
<th width="50%">Sub-features</th>
</tr>
<tr>
<td><strong>📊 Paper2Figure</strong><br><sub>Editable Scientific Figures</sub></td>
<td><img src="https://img.shields.io/badge/Progress-75%25-blue?style=flat-square&logo=progress" alt="75%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Model_Architecture-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Technical_Roadmap-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/⚠-Experimental_Plots-yellow?style=flat-square" alt="WIP"/><br>
<img src="https://img.shields.io/badge/✓-Web_Frontend-success?style=flat-square" alt="Done"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2Video</strong><br><sub>Paper Explanation Videos</sub></td>
<td><img src="https://img.shields.io/badge/Progress-25%25-orange?style=flat-square&logo=progress" alt="25%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Script_Generation-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/○-Storyboard-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Visual_Materials-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Auto_Composition-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2PPT</strong><br><sub>Editable Slide Decks</sub></td>
<td><img src="https://img.shields.io/badge/Progress-50%25-yellow?style=flat-square&logo=progress" alt="50%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Beamer_Style-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/⚠-Editable_PPTX-yellow?style=flat-square" alt="WIP"/>
</td>
</tr>
<tr>
<td><strong>📌 Paper2Poster</strong><br><sub>Editable Academic Posters</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-Layout_Design-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Key_Points-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Visual_Refinement-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
<tr>
<td><strong>🧪 Paper2Exp</strong><br><sub>Auto Experiment Runner</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-Code_Generation-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Environment_Setup-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Auto_Execution-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
<tr>
<td><strong>📚 PaperCiter</strong><br><sub>Smart Citation Assistant</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-Citation_Search-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Auto_Formatting-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
</table>

---

### 📊 Data Series

<table>
<tr>
<th width="35%">Feature</th>
<th width="15%">Status</th>
<th width="50%">Sub-features</th>
</tr>
<tr>
<td><strong>🔄 Easy-DataFlow</strong><br><sub>Data Governance Pipeline</sub></td>
<td><img src="https://img.shields.io/badge/Progress-100%25-success?style=flat-square&logo=progress" alt="100%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Pipeline_Recommendation-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Operator_Authoring-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Visual_Orchestration-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Prompt_Optimization-success?style=flat-square" alt="Done"/><br>
<img src="https://img.shields.io/badge/✓-Web_Collection-success?style=flat-square" alt="Done"/>
</td>
</tr>
<tr>
<td><strong>📊 DataFlow-Table</strong><br><sub>Multi-source Data Analysis</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-Multi--source_Ingestion-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Smart_Retrieval-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Lineage_Tracking-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-Advanced_Visualization-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
</table>

---

### 🛠️ Tool Enhancements

<table>
<tr>
<th width="35%">Feature</th>
<th width="15%">Status</th>
<th width="50%">Sub-features</th>
</tr>
<tr>
<td><strong>🎨 Workflow Visual Editor</strong><br><sub>Drag-and-drop Workflow Builder</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-Drag_&_Drop_Interface-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-5_Agent_Modes-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-20+_Preset_Nodes-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
<tr>
<td><strong>💾 Trajectory Export</strong><br><sub>Training Data Export</sub></td>
<td><img src="https://img.shields.io/badge/Status-Working-lightgrey?style=flat-square" alt="Working"/></td>
<td>
<img src="https://img.shields.io/badge/○-JSON/JSONL_Format-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-SFT_Format-lightgrey?style=flat-square" alt="Working"/><br>
<img src="https://img.shields.io/badge/○-DPO_Format-lightgrey?style=flat-square" alt="Working"/>
</td>
</tr>
</table>

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/DataFlow-Agent@main/static/dfa_fronted.png" width="800" alt="Workflow Editor"/>
<br><sub>🎨 Workflow Visual Editor Preview</sub>
</div>

---

## 🤝 Contributing

We welcome all forms of contributions!

[![Issues](https://img.shields.io/badge/Issues-Submit_Bug-red?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/issues)
[![Discussions](https://img.shields.io/badge/Discussions-Feature_Request-blue?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/discussions)
[![PR](https://img.shields.io/badge/PR-Submit_Code-green?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/DataFlow-Agent/pulls)

Detailed contribution guide: [docs/contributing.md](docs/contributing.md)

---

## 📄 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

This project is licensed under [Apache License 2.0](LICENSE)

---

## 🙏 Acknowledgments

Thanks to all contributors! Special thanks to the upstream project [OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow)

---

<div align="center">

**If this project helps you, please give us a ⭐️ Star!**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=social)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/DataFlow-Agent?style=social)](https://github.com/OpenDCAI/DataFlow-Agent/network/members)

[Submit Issue](https://github.com/OpenDCAI/DataFlow-Agent/issues) • [View Docs](docs/) • [Join Discussion](https://github.com/OpenDCAI/DataFlow-Agent/discussions)

Made with ❤️ by OpenDCAI Team

</div>
