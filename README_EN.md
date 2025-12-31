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
<img src="static/new_readme/2figure.gif" width="90%"/>
<br><sub>✨ Model Architecture Diagram Generation</sub>

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
<img src="static/new_readme/paper2ppt操作.gif" width="85%"/>
<br><sub>✨ PPT Generation Demo</sub>

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
<img src="static/new_readme/polish.gif" width="90%"/>
<br><sub>✨ AI-based Layout Optimization</sub>

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

#### 4. Configure Supabase (Required for Frontend/Backend)

Create a `.env` file in the `frontend-workflow` directory and fill in the following configuration:

```bash
# frontend-workflow/.env

VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Backend
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_JWT_SECRET=your_jwt_secret

# Application Settings
DAILY_WORKFLOW_LIMIT=10
```

<details>
<summary><strong>Advanced Configuration: Local Model Service Load Balancing</strong></summary>

If deploying in a high-concurrency environment locally, you can use `script/start_model_servers.sh` to start a local model service cluster (MinerU / SAM / OCR).

Script location: `/DataFlow-Agent/script/start_model_servers.sh`

**Main Configuration Items:**

- **MinerU (PDF Parsing)**
  - `MINERU_MODEL_PATH`: Model path (default `models/MinerU2.5-2509-1.2B`)
  - `MINERU_GPU_UTIL`: GPU memory utilization (default 0.2)
  - **Instance Config**: Default starts 4 instances each on GPU 0 and GPU 4 (total 8), ports 8011-8018.
  - **Load Balancer**: Port 8010, automatically distributes requests.

- **SAM (Segment Anything Model)**
  - **Instance Config**: Default starts 1 instance each on GPU 2 and GPU 3, ports 8021-8022.
  - **Load Balancer**: Port 8020.

- **OCR (PaddleOCR)**
  - **Config**: Runs on CPU, uses uvicorn worker mechanism (default 4 workers).
  - **Port**: 8003.

> Please modify `gpu_id` and instance count in the script according to your actual GPU quantity and memory before use.

</details>

### 🪟 Windows Installation

> [!NOTE]
> Currently, we recommend experiencing Paper2Any in a Linux / WSL environment. If you need to deploy on native Windows, please follow the steps below.

#### 1. Create Environment & Install Base Dependencies

```bash
# 0. Create and activate a conda environment
conda create -n paper2any python=3.12 -y
conda activate paper2any

# 1. Clone repository
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. Install base dependencies
pip install -r requirements-win-base.txt

# 3. Install in editable (dev) mode
pip install -e .
```

#### 2. Install Paper2Any-specific Dependencies (Recommended)

Paper2Any involves LaTeX rendering and vector graphics processing, requiring extra dependencies (see requirements-paper.txt):

```bash
# Python dependencies
pip install -r requirements-paper.txt

# tectonic: LaTeX engine (recommended via conda)
conda install -c conda-forge tectonic -y
```

**🎨 Install Inkscape (SVG/Vector Graphics Processing | Recommended/Required)**

1. Download and install (Windows 64-bit MSI): [Inkscape Download](https://inkscape.org/release/inkscape-1.4.2/windows/64-bit/msi/?redirected=1)
2. Add the Inkscape executable directory to the system environment variable Path (Example): `C:\Program Files\Inkscape\bin\`

> [!TIP]
> After configuring Path, it is recommended to reopen the terminal (or restart VS Code / PowerShell) to ensure the environment variables take effect.

#### ⚡ Install Windows Compiled vLLM (Optional | For Local Inference Acceleration)

Release page reference: [vllm-windows releases](https://github.com/SystemPanic/vllm-windows/releases)
Recommended version: 0.11.0

```bash
pip install vllm-0.11.0+cu124-cp312-cp312-win_amd64.whl
```

> [!IMPORTANT]
> Please ensure `.whl` matches your current environment:
> - Python: cp312 (Python 3.12)
> - Platform: win_amd64
> - CUDA: cu124 (Must match your local CUDA/driver)

#### Launch Applications

**Paper2Any - Paper Workflow Web Frontend (Recommended)**

```bash
# Start backend API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd frontend-workflow
npm install
npm run dev
```

**Configure Frontend Proxy**

Modify `server.proxy` in `frontend-workflow/vite.config.ts`:

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',  // FastAPI Backend Address
        changeOrigin: true,
      },
    },
  },
})
```
Visit `http://localhost:3000`

**Windows Load MinerU Pre-trained Model**

```powershell
# Start in PowerShell environment
vllm serve opendatalab/MinerU2.5-2509-1.2B `
  --host 127.0.0.1 `
  --port 8010 `
  --logits-processors mineru_vl_utils:MinerULogitsProcessor `
  --gpu-memory-utilization 0.6 `
  --trust-remote-code `
  --enforce-eager
```

> [!TIP]
> **Paper2Figure Web Beta Instructions**
> 
> When you deploy the frontend, you also need to manually create a `invite_codes.txt` file and write your invitation code (e.g., `ABCDEFG123456`).
> Then start the backend.
> 
> If you don't want to deploy frontend/backend for now, you can try core features via local scripts first:
> - `python script/run_paper2figure.py`: Model architecture diagram generation
> - `python script/run_paper2expfigure.py`: Experimental plot generation
> - `python script/run_paper2technical.py`: Technical roadmap generation
> - `python script/run_paper2ppt.py`: Paper content to editable PPT
> - `python script/run_pdf2ppt_with_paddle_sam_mineru.py`: PDF2PPT (Layout preservation + Editable content)

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
