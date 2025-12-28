<div align="center">

<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/new_logo_bgrm.png" alt="Paper2Any Logo" width="180"/><br>

# Paper2Any

<!-- **From Papers & Raw Data to Charts, PPTs and Data Pipelines — an All-in-One AI Orchestrator** -->

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FPaper2Any-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/Paper2Any)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/Paper2Any/stargazers)

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

*Focus on Paper Multimodal Workflow: One-click generation of model diagrams, technical roadmaps, experimental plots, and presentations from paper PDFs/screenshots/text.*

English | [中文](README.md)

</div>

<div align="center">
  <img src="static/frontend_pages/paper2figure-1.png" alt="Web UI - Paper2Figure" width="48%"/>
  <span>&nbsp;|&nbsp;</span>
  <img src="static/frontend_pages/paper2ppt-1.png" alt="Web UI - Paper2PPT" width="48%"/>
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

- 2024-09-01 · Released <code>0.1.0</code> first version

---

## ✨ Core Features

> From paper PDFs / images / text to **editable** scientific figures, slide decks, video scripts, posters and more in one click.

Paper2Any currently includes the following sub-capabilities:

<table>
<tr>
<td width="50%" valign="top">

**📊 Paper2Figure - Editable Scientific Figures**
- ✅ Model architecture diagram generation
- ✅ Technical roadmap diagram generation (PPT + SVG)
- ✅ Experimental plot generation (under optimization)
- ✅ Supports PDF / image / text inputs
- ✅ Editable PPTX output

</td>
<td width="50%" valign="top">

**🎬 Paper2PPT - Editable Slide Decks**
- ✅ Beamer slide generation
- ✅ Open, fully editable PPT generation
- ✅ PDF2PPT conversion with background preserved & editable content

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

## 📸 Showcase

### 1. Paper2PPT - Paper to Presentation

#### Basic Generation (Paper / Text / Topic → PPT)

<table>
<tr>
<th width="25%">Input</th>
<th width="25%">Output</th>
<th width="25%">Input</th>
<th width="25%">Output</th>
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/input_1.png" alt="Input: paper PDF" width="100%"/>
<br><sub>📄 Paper PDF</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_1.png" alt="Output: generated PPT" width="100%"/>
<br><sub>📊 Generated PPT</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_3.png" alt="Input: paper content" width="100%"/>
<br><sub>📝 Paper content</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_3.png" alt="Output: generated PPT" width="100%"/>
<br><sub>📊 Generated PPT</sub>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<strong>PPT Generation</strong> - Upload a paper PDF, automatically extract key information and generate a structured academic presentation.
</td>
<td colspan="2" align="center">
<strong>PPT Generation</strong> - Intelligently analyze paper content and automatically insert internal tables and figures into the slides.
</td>
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/input_2-1.png" alt="Input: text 1" width="100%"/>
<br><sub>📄 Input text 1</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_2-2.png" alt="Input: text 2" width="100%"/>
<br><sub>📄 Input text 2</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_2-3.png" alt="Input: text 3" width="100%"/>
<br><sub>📄 Input text 3</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_2.png" alt="Output: generated PPT" width="100%"/>
<br><sub>📊 Generated PPT</sub>
</td>
</tr>
<tr>
<td colspan="4" align="center">
<strong>Text2PPT</strong> - Input long text/outline, automatically generate structured PPT.
</td>
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/input_4-1.png" alt="Input: topic 1" width="100%"/>
<br><sub>📄 Input topic 1</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_4-2.png" alt="Input: topic 2" width="100%"/>
<br><sub>📄 Input topic 2</sub>
</td>
<td align="center">
<img src="static/paper2ppt/input_4-3.png" alt="Input: topic 3" width="100%"/>
<br><sub>📄 Input topic 3</sub>
</td>
<td align="center">
<img src="static/paper2ppt/output_4.png" alt="Output: generated PPT" width="100%"/>
<br><sub>📊 Generated PPT</sub>
</td>
</tr>
<tr>
<td colspan="4" align="center">
<strong>Topic2PPT</strong> - Input brief topic, automatically expand content and generate PPT.
</td>
</tr>
</table>

#### 🚀 Long Document Generation (40+ Slides)

> Supports entire books, long reviews, or lengthy technical documents. Automatically processes by chapter to generate comprehensive 40-100 slide presentations.

<table>
<tr>
<th width="25%">Input: Long Paper/Book</th>
<th width="25%">Outline Generation</th>
<th width="25%">Content Filling</th>
<!-- <th width="25%">Final PPT (40+ Slides)</th> -->
</tr>
<tr>
<td align="center">
<img src="static/paper2ppt/long_paper/input_0.png" alt="Input: Long Doc" width="100%"/>
<br><sub>📚 Input: Full Book / Long Review</sub>
</td>
<td align="center">
<img src="static/paper2ppt/long_paper/output_1.png" alt="Outline Generation" width="100%"/>
<br><sub>📝 Auto Multi-level Outline</sub>
</td>
<td align="center">
<img src="static/paper2ppt/long_paper/output_2.png" alt="Content Filling" width="100%"/>
<br><sub>🔄 Parallel Chapter Generation</sub>
</td>
</tr>
</table>


---

#### PDF2PPT - PDF to Editable PPT

<table>
<tr>
<th width="25%">Input</th>
<th width="25%">Output</th>
<th width="25%">Input</th>
<th width="25%">Output</th>
</tr>
<tr>
<td align="center">
<img src="static/pdf2ppt/input_1.png" alt="Input: PDF page" width="100%"/>
<br><sub>📄 PDF page</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/output_1.png" alt="Output: generated PPT page" width="100%"/>
<br><sub>📊 Generated PPT page (White BG)</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/input_2.png" alt="Input: PDF page" width="100%"/>
<br><sub>📄 PDF page</sub>
</td>
<td align="center">
<img src="static/pdf2ppt/output_2.png" alt="Output: generated PPT page" width="100%"/>
<br><sub>📊 Generated PPT page (AI Redraw)</sub>
</td>
</tr>
</table>

#### PPT Polish - Smart Enhancement

<table>
<tr>
<th width="25%">Original PPT</th>
<th width="25%">Enhanced</th>
<th width="25%">Original PPT</th>
<th width="25%">Polished</th>
</tr>
<tr>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_orgin_1.png" alt="Original PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/paper2ppt_polish_1.png" alt="Enhanced PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/orgin_3.png" alt="Original PPT" width="100%"/>
</td>
<td align="center">
<img src="frontend-workflow/public/ppt2polish/polish_3.png" alt="Polished PPT" width="100%"/>
</td>
</tr>
</table>

---

### 2. Paper2Figure - Scientific Figure Generation

#### Model Architecture Diagram Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Generated Figure</th>
<th width="33%">PPTX Screenshot</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img.png" alt="Input: paper PDF" width="100%"/>
<br><sub>📄 Paper PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_pdf_img_3.png" alt="PPTX screenshot" width="100%"/>
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
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_1.png" alt="Input: paper PDF" width="100%"/>
<br><sub>📄 Paper PDF</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_mid_img_3.png" alt="PPTX screenshot" width="100%"/>
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
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_1.png" alt="Input: key paragraphs" width="100%"/>
<br><sub>📄 Input key paragraphs</sub>
</td>
<td align="center">
<img src="static/paper2any_imgs/p2f/p2f_paper_hard_img_2.png" alt="Generated model diagram" width="100%"/>
<br><sub>🎨 Generated model architecture</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2f/p2f_paper_hard_img_3.png" alt="PPTX screenshot" width="100%"/>
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

#### Technical Roadmap Diagram Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Generated Figure (SVG)</th>
<th width="33%">PPTX Screenshot</th>
</tr>
<tr>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/paper1.png" alt="Input: paper text (Chinese)" width="100%"/>
<br><sub>📝 Method section (Chinese)</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/cn_img_1.png" alt="Roadmap diagram SVG" width="100%"/>
<br><sub>🗺️ Roadmap diagram SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/cn_img_2.png" alt="PPTX screenshot" width="100%"/>
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
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/paper2.png" alt="Input: paper text (English)" width="100%"/>
<br><sub>📝 Method section (English)</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/en_img_1.png" alt="Roadmap diagram SVG" width="100%"/>
<br><sub>🗺️ Roadmap diagram SVG</sub>
</td>
<td align="center">
<img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2t/en_img_2.png" alt="PPTX screenshot" width="100%"/>
<br><sub>📊 Editable PPTX</sub>
</td>
</tr>
<tr>
<td colspan="3" align="center">
<strong>Language: English</strong> - Ideal for international publications
</td>
</tr>
</table>

#### Experimental Plot Generation

<table>
<tr>
<th width="33%">Input</th>
<th width="33%">Standard Style</th>
<th width="33%">Hand-drawn Style</th>
</tr>
<tr>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2e/paper_1.png" alt="Input: experimental results" width="100%"/>
  <br><sub>📄 Experimental results screenshot</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@main/static/paper2any_imgs/p2e/paper_1_2.png" alt="Output: standard style" width="100%"/>
  <br><sub>📈 Standard Python style</sub>
</td>
<td align="center">
  <img src="https://cdn.jsdelivr.net/gh/OpenDCAI/Paper2Any@lz/dev/static/paper2any_imgs/p2e/paper_1_3.png" alt="Output: hand-drawn style" width="100%"/>
  <br><sub>🎨 Hand-drawn style</sub>
</td>
</tr>
<tr>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2.png" alt="Input: experimental results screenshot" width="100%"/>
  <br><sub>📄 Input: paper PDF / results screenshot</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2_2.png" alt="Output: plot (standard)" width="100%"/>
  <br><sub>📈 Output: standard Python style plot</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_2_3.png" alt="Output: plot (cartoon style)" width="100%"/>
  <br><sub>🎨 Output: cartoon style experimental plot</sub>
</td>
</tr>
<tr>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3.png" alt="Input: experimental results screenshot" width="100%"/>
  <br><sub>📄 Input: paper PDF / results screenshot</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3_2.png" alt="Output: plot (standard)" width="100%"/>
  <br><sub>📈 Output: standard Python style plot</sub>
</td>
<td align="center">
  <img src="static/paper2any_imgs/p2e/paper_3_3.png" alt="Output: plot (polygon style)" width="100%"/>
  <br><sub>🎨 Output: polygon style experimental plot</sub>
</td>
</tr>
</table>

---

## 🚀 Quick Start

### Requirements

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

### Installation

> We recommend using Conda to create an isolated environment (Python 3.11+).

```bash
# 0. Create and activate a conda environment
conda create -n paper2any python=3.11 -y
conda activate paper2any

# 1. Clone repository
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. Install dependencies (base)
pip install -r requirements-base.txt

# 3. Install package (editable / dev mode)
pip install -e .
```

#### Paper2Any Extra Dependencies (Required)

Paper2Any involves LaTeX rendering, vector graphics processing, and PPT/PDF conversion, which require additional dependencies:

```bash
# 1. Python Dependencies
# (If requirements-paper.txt fails, try requirements-paper-backup.txt)
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt

# 2. LaTeX Engine (tectonic) - Recommended via conda
conda install -c conda-forge tectonic -y

# 3. Resolve doclayout_yolo dependency conflict (Important)
# Due to a conflict between doclayout_yolo and paddleocr, install it separately:
pip install doclayout_yolo --no-deps

# 4. System Dependencies (Ubuntu example)
# Includes:
# - inkscape: SVG / Vector graphics processing
# - libreoffice: PPT operations / conversion
# - poppler-utils: PDF utilities
# - wkhtmltopdf: HTML to PDF
sudo apt-get update
sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
```

### Environment Configuration

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx 
# If using third-party API gateway

# [Optional] Configure GPU resource pool for MinerU PDF parsing
export MINERU_DEVICES="0,1,2,3"
```

---

### Launch Applications

**Web Frontend (Recommended)**

```bash
# 1. Start backend API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Start frontend (new terminal)
cd frontend-workflow
npm install
npm run dev
```

Visit `http://localhost:3000`

> [!TIP]
> If you don't want to deploy the frontend/backend for now, you can try core features locally via scripts:
> - `python script/run_paper2figure.py`: model architecture diagram generation
> - `python script/run_paper2ppt.py`: content-based PPT generation
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
</table>

---

## 🤝 Contributing

We welcome all forms of contributions!

[![Issues](https://img.shields.io/badge/Issues-Submit_Bug-red?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/issues)
[![Discussions](https://img.shields.io/badge/Discussions-Feature_Request-blue?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/discussions)
[![PR](https://img.shields.io/badge/PR-Submit_Code-green?style=for-the-badge&logo=github)](https://github.com/OpenDCAI/Paper2Any/pulls)

---

## 📄 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

This project is licensed under [Apache License 2.0](LICENSE)

---

<div align="center">

**If this project helps you, please give us a ⭐️ Star!**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/network/members)

[Submit Issue](https://github.com/OpenDCAI/Paper2Any/issues) • [Join Discussion](https://github.com/OpenDCAI/Paper2Any/discussions)

Made with ❤️ by OpenDCAI Team

</div>

---

## 🌐 Join the Community

- 📮 **GitHub Issues**: Report bugs or suggest new features  
  👉 https://github.com/OpenDCAI/Paper2Any/issues
- 💬 **Community Group**: Connect with maintainers and other contributors

<div align="center">
  <img src="static/team_wechat.png" alt="DataFlow-Agent WeChat Community" width="560"/>
  <br>
  <sub>Scan to join the community group</sub>
</div>
