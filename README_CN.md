<div align="center">

<img src="static/new_readme/branding/logo.png" alt="Paper2Any Logo" width="200"/>

# Paper2Any

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FPaper2Any-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/Paper2Any)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/Paper2Any/stargazers)

中文 | [English](README.md)

<a href="https://trendshift.io/repositories/17634" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17634" alt="OpenDCAI%2FPaper2Any | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

✨ **专注论文多模态工作流：从论文 PDF/截图/文本，一键生成模型示意图、技术路线图、实验图和演示文稿** ✨

| 📄 **Universal File Support** &nbsp;|&nbsp; 🎯 **AI-Powered Generation** &nbsp;|&nbsp; 🎨 **Custom Styling** &nbsp;|&nbsp; ⚡ **Lightning Speed** |

<br>

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
<a href="#wechat-group" target="_self">
  <img alt="WeChat" src="https://img.shields.io/badge/💬-微信群-07C160?style=for-the-badge" />
</a>

<br>
<br>

<img src="static/new_readme/ui/home.png" alt="Paper2Any Web Interface" width="80%"/>

</div>


## 📑 目录

- [🔥 News](#-news)
- [✨ 核心功能](#-核心功能)
- [📸 功能展示](#-功能展示)
- [🧩 Drawio](#-drawio)
- [🚀 快速开始](#-快速开始)
- [📂 项目结构](#-项目结构)
- [🗺️ 开发计划](#️-开发计划)
- [🤝 贡献](#-贡献)

---

## 🔥 News

> [!TIP]
> 🆕 <strong>2026-02-02 · Paper2Rebuttal 更新</strong><br>
> 新增反驳意见草拟与修改建议，支持结构化回复与图文要点对齐。

> [!TIP]
> 🆕 <strong>2026-01-28 · Drawio 更新</strong><br>
> 新增 Drawio 支持，用于可视化图示的快速创作与展示输出。<br>
> KB 一句话概括：支持多文件 PPT 生成（文档转换/合并 + 图片注入 + 向量检索增强）。

> [!TIP]
> 🆕 <strong>2026-01-20 · Bug 修复</strong><br>
> 修复了实验数据图生成的图片和文本 bug，并解决了历史文件缺失的问题。<br>
> 🌐 在线体验：<a href="http://dcai-paper2any.nas.cpolar.cn/">http://dcai-paper2any.nas.cpolar.cn/</a>

> [!TIP]
> 🆕 <strong>2025-12-12 · Paper2Figure 网页端公测上线</strong><br>
> 支持一键生成多种<strong>可编辑</strong>科研绘图（模型架构图 / 技术路线图 / 实验数据图）<br>
> 🌐 在线体验：<a href="http://dcai-paper2any.nas.cpolar.cn/">http://dcai-paper2any.nas.cpolar.cn/</a>

- 2025-10-01 · 发布 <code>0.1.0</code> 首个版本

---

## ✨ 核心功能

> 从论文 PDF / 图片 / 文本出发，一键生成**可编辑**的科研绘图、演示文稿、视频脚本、学术海报等多模态内容。

Paper2Any 当前包含以下几个子能力：

- **📊 Paper2Figure - 可编辑科研绘图**：模型架构图、技术路线图（PPT + SVG）与实验数据图，输出可编辑 PPTX。
- **🧩 Paper2Diagram / Image2Drawio - 可编辑流程图**：从论文/文本或图片生成 Drawio 图，支持 drawio/png/svg 导出与对话式编辑。
- **🎬 Paper2PPT - 可编辑演示文稿**：论文/文本/主题一键生成，支持超长文档与表格/图表抽取。
- **📝 Paper2Rebuttal**：自动生成结构化反驳草稿与修改建议，辅助审稿意见回复。
- **🖼️ PDF2PPT - 版式保留转换**：精准保留版式的 PDF → 可编辑 PPTX。
- **🖼️ Image2PPT - 图片转 PPT**：将图片或截图快速转换为结构化幻灯片。
- **🎨 PPTPolish 智能美化**：基于 AI 的排版优化与风格迁移。
- **🎬 Paper2Video**：生成讲解视频脚本与配音素材。
- **📚 知识库（KB）**：文件入库/向量化、语义检索，以及 KB 驱动的 PPT/播客/思维导图生成。

---

## 📸 功能展示

### 🧩 Drawio

<div align="center">

<br>
<img src="static/new_readme/drawio/drawio-1.gif" width="90%"/>
<br><sub>✨ Diagram 生成（思维导图 / 流程图 / ER 图 ......）</sub>

<br><br>
<img src="static/new_readme/drawio/drawio-2.gif" width="90%"/>
<br><sub>✨ 模型图生成：上传 PDF 或输入文本，自动生成科研绘图</sub>

<br><br>
<img src="static/new_readme/drawio/image-to-editable.png" width="90%"/>
<br><sub>✨ 图片转可编辑 DrawIO 图</sub>

</div>

---

### 📝 Paper2Rebuttal：审稿回复

<div align="center">

<br>
<img src="static/new_readme/paper2rebuttal/rebuttal.png" width="90%"/>
<br><sub>✨ 审稿回复草拟与修改建议</sub>

</div>

---

### 📊 Paper2Figure: 科研绘图生成

<div align="center">

<br>
<img src="static/new_readme/paper2figure/model-arch-demo.gif" width="90%"/>
<br><sub>✨ 模型架构图生成</sub>

<br>
<img src="static/new_readme/paper2figure/model-arch-1.png" width="90%"/>
<br><sub>✨ 模型架构图生成</sub>

<br><br>
<img src="static/new_readme/paper2figure/technical-roadmap-1.png" width="90%"/>
<br><br>
<img src="static/new_readme/paper2figure/technical-roadmap-2.png" width="90%"/>
<br><sub>✨ 技术路线图生成</sub>

<br><br>
<img src="static/new_readme/paper2figure/experimental-plot.png" width="90%"/>
<br><sub>✨ 实验数据图生成 (多种风格)</sub>

</div>

---

### 🎬 Paper2PPT: 论文转演示文稿

<div align="center">

<br>
<img src="static/new_readme/paper2ppt/demo.gif" width="85%"/>
<br><sub>✨ PPT生成案例</sub>

<br>
<img src="static/new_readme/paper2ppt/case-1.png" width="90%"/>
<br><sub>✨ Paper / Text / Topic → PPT</sub>

<br><br>
<img src="static/new_readme/paper2ppt/long-doc.png" width="85%"/>
<br><sub>✨ 超长文档支持 (40+ 页 PPT)</sub>

<br><br>
<img src="static/new_readme/paper2ppt/table-extraction.png" width="90%"/>
<br><sub>✨ 智能表格提取与插入</sub>

<br><br>
<img src="static/new_readme/paper2ppt/outline-edit.png" width="90%"/>
<br><sub>✨ AI 辅助编辑大纲</sub>

<br><br>
<img src="static/new_readme/paper2ppt/version-history.png" width="90%"/>
<br><sub>✨ 历史版本管理</sub>

</div>

---

### 🎨 PPT 智能美化

<div align="center">

<br>
<img src="static/new_readme/pptpolish/polish-demo.gif" width="90%"/>
<br><sub>✨ 基于 AI 的排版优化</sub>

<br>
<img src="static/new_readme/pptpolish/style-transfer-1.png" width="90%"/>
<br><sub>✨ 基于 AI 的排版优化与风格迁移</sub>

</div>

---

### 🖼️ PDF2PPT: 版式保留转换

<div align="center">

<br>
<img src="static/new_readme/pdf2ppt/cutout.png" width="90%"/>
<br><sub>✨ 智能抠图 & 版式保留</sub>

<br>
<img src="static/new_readme/image2ppt/image2ppt.png" width="93%"/>
<br><sub>✨ 图片转 PPT</sub>

</div>

---

## 🚀 快速开始

### 环境要求

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-3776AB?style=flat-square&logo=pypi&logoColor=white)

<details>
<summary><strong>🐳 Docker 快速启动（推荐）— 部署与更新</strong></summary>

```bash
# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. 配置环境变量
cp fastapi_app/.env.example fastapi_app/.env
cp frontend-workflow/.env.example frontend-workflow/.env
```

**必须修改的配置项：**

`fastapi_app/.env`（后端）：
```bash
# 必填：你的 LLM API 地址（替换为你自己的）
DEFAULT_LLM_API_URL=https://api.openai.com/v1/
# 可选：Supabase（不填则跳过用户认证，核心功能不受影响）
# SUPABASE_URL=https://your-project-id.supabase.co
# SUPABASE_ANON_KEY=your_supabase_anon_key
```

`frontend-workflow/.env`（前端）：
```bash
# 必填：前端可用的 LLM API 地址（逗号分隔，显示在 UI 下拉菜单中）
VITE_DEFAULT_LLM_API_URL=https://api.openai.com/v1
VITE_LLM_API_URLS=https://api.openai.com/v1
# 可选：Supabase（与后端保持一致）
# VITE_SUPABASE_URL=https://your-project-id.supabase.co
# VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

```bash
# 3. 构建并启动
docker compose up -d --build
```

访问地址：
- 前端：http://localhost:3000
- 后端健康检查：http://localhost:8000/health

> **GPU 服务说明：** Docker 默认启动的是前后端服务，不包含 GPU 模型服务。
> - Paper2PPT、Paper2Figure、知识库等功能仅依赖 LLM API，Docker 启动后即可使用。
> - **PDF2PPT、Image2PPT、Image2Drawio** 依赖 SAM3 图像分割服务（需要 GPU），需额外部署：
>   ```bash
>   # 在有 GPU 的机器上启动 SAM3 服务
>   python -m dataflow_agent.toolkits.model_servers.sam3_server \
>       --port 8001 --checkpoint models/sam3/sam3.pt \
>       --bpe models/sam3/bpe_simple_vocab_16e6.txt.gz --device cuda
>   ```
>   然后在 `fastapi_app/.env` 中添加：`SAM3_SERVER_URLS=http://GPU机器IP:8001`
>
> 详见下方「高级配置：本地模型服务负载均衡」部分。

修改与更新：
- 代码或 `.env` 变更后重新构建：`docker compose up -d --build`
- 拉取最新代码并重建：
  - `git pull`
  - `docker compose up -d --build`

常用命令：
- 查看日志：`docker compose logs -f`
- 停止服务：`docker compose down`

说明：
- 首次构建会比较慢（系统依赖 + Python 依赖）。
- 前端配置在构建期生效（compose build args），修改后需重新 `docker compose up -d --build`。
- 输出和模型目录会挂载到宿主机（`./outputs`、`./models`），数据不会丢。

</details>

### 🐧 Linux 安装

> 建议使用 Conda 创建隔离环境（推荐 Python 3.11）。  

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
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt

# 2. LaTeX 引擎 (tectonic) - 推荐用 conda 安装
conda install -c conda-forge tectonic -y

# 3. 解决 doclayout_yolo 依赖冲突（重要）
pip install doclayout_yolo --no-deps

# 4. 系统依赖 (Ubuntu 示例)
sudo apt-get update
sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
```

#### 3. 配置环境变量

```bash
export DF_API_KEY=your_api_key_here
export DF_API_URL=xxx  # 可选：如需使用第三方 API 中转站
export MINERU_DEVICES="0,1,2,3" # 可选：MinerU 任务 GPU 资源池
```

#### 4. 配置环境文件（可选）

<details>
<summary><strong>📝 点击展开：详细的 .env 配置指南</strong></summary>

Paper2Any 使用两个 `.env` 文件进行配置。**两者都是可选的** - 您可以使用默认设置运行应用程序。

##### 步骤 1：复制示例文件

```bash
# 复制后端环境文件
cp fastapi_app/.env.example fastapi_app/.env

# 复制前端环境文件
cp frontend-workflow/.env.example frontend-workflow/.env
```

##### 步骤 2：后端配置（`fastapi_app/.env`）

**Supabase（可选）** - 仅在需要用户认证和云存储时配置：
```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
```

**模型配置** - 自定义不同工作流使用的模型：
```bash
# 默认 LLM API 地址
DEFAULT_LLM_API_URL=http://123.129.219.111:3000/v1/

# 工作流级别默认值
PAPER2PPT_DEFAULT_MODEL=gpt-5.1
PAPER2PPT_DEFAULT_IMAGE_MODEL=gemini-3-pro-image-preview
PDF2PPT_DEFAULT_MODEL=gpt-4o
# ... 完整列表请查看 .env.example
```

##### 步骤 3：前端配置（`frontend-workflow/.env`）

**LLM 提供商配置** - 控制 UI 中的 API 端点下拉菜单：
```bash
# UI 中显示的默认 API 地址
VITE_DEFAULT_LLM_API_URL=https://api.apiyi.com/v1

# 下拉菜单中的可用 API 地址（逗号分隔）
VITE_LLM_API_URLS=https://api.apiyi.com/v1,http://b.apiyi.com:16888/v1,http://123.129.219.111:3000/v1
```

**修改 `VITE_LLM_API_URLS` 后的效果：**
- 前端会显示一个**下拉菜单**，包含您指定的所有 URL
- 用户可以选择不同的 API 端点，无需手动输入 URL
- 适用于在 OpenAI、本地模型或自定义 API 网关之间切换

**Supabase（可选）** - 如需用户认证，取消注释这些行：
```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret
```

##### 不配置 Supabase 的情况

如果跳过 Supabase 配置：
- ✅ 所有核心功能正常工作
- ✅ CLI 脚本无需任何配置即可使用
- ❌ 无用户认证或配额限制
- ❌ 无云文件存储

</details>

> [!NOTE]
> **快速开始：** 您可以完全跳过 `.env` 配置，直接使用 CLI 脚本并通过 `--api-key` 参数传递密钥。详见下方 [CLI 脚本](#️-cli-脚本命令行界面) 部分。

---

<details>
<summary><strong>高级配置：本地模型服务负载均衡</strong></summary>

如果是本地部署高并发环境，可以使用 `script/start_model_servers.sh` 启动本地模型服务集群（MinerU / SAM / OCR）。

脚本位置：`/DataFlow-Agent/script/start_model_servers.sh`

**主要配置项说明：**

- **MinerU (PDF 解析)**
  - `MINERU_MODEL_PATH`: 模型路径 (默认 `models/MinerU2.5-2509-1.2B`)
  - `MINERU_GPU_UTIL`: 显存占用比例 (默认 0.85)
  - **实例配置**: 脚本默认在每个配置 GPU 上各启动 1 个实例，端口范围 8011-8013。
  - **Load Balancer**: 端口 8010，自动分发请求。

- **SAM3 (Segment Anything Model 3)**
  - **实例配置**: 默认每个配置 GPU 启动 1 个实例，端口范围 8021-8022。
  - **模型路径**: 默认使用 `./models/sam3/sam3.pt` 与 `./models/sam3/bpe_simple_vocab_16e6.txt.gz`。
  - **Load Balancer**: 端口 8020。

- **OCR (PaddleOCR)**
  - **配置**: 运行在 CPU 上，使用 uvicorn 的 worker 机制 (默认 4 workers)。
  - **端口**: 8003。

> 使用前请根据实际 GPU 数量和显存情况修改脚本中的 `gpu_id` 和实例数量。

如果你要在单张 GPU 上一条命令联调（SAM3 + 后端 + 前端），可执行：

```bash
bash script/start_local_sam3_dev.sh
```

</details>

---

### 🪟 Windows 安装

> [!NOTE]
> 目前推荐优先在 Linux / WSL 环境下体验 Paper2Any。 若你需要在 原生 Windows 上部署，请按以下步骤操作。

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

#### 2. 安装 Paper2Any 相关依赖（推荐）

Paper2Any 涉及 LaTeX 渲染与矢量图处理，需要额外依赖（见 requirements-paper.txt）：

```bash
# Python 依赖
pip install -r requirements-paper.txt

# tectonic：LaTeX 引擎（推荐用 conda 安装）
conda install -c conda-forge tectonic -y
```

**🎨 安装 Inkscape（SVG/矢量图处理｜推荐/必装）**

1. 下载并安装（Windows 64-bit MSI）：[Inkscape Download](https://inkscape.org/release/inkscape-1.4.2/windows/64-bit/msi/?redirected=1)
2. 将 Inkscape 可执行文件目录加入系统环境变量 Path（示例）：`C:\Program Files\Inkscape\bin\`

> [!TIP]
> 配置 Path 后建议重新打开终端（或重启 VS Code / PowerShell），确保环境变量生效。

#### ⚡ 安装 Windows 编译版 vLLM（可选｜用于本地推理加速）

发布页参考：[vllm-windows releases](https://github.com/SystemPanic/vllm-windows/releases)
推荐版本：0.11.0

```bash
pip install vllm-0.11.0+cu124-cp312-cp312-win_amd64.whl
```

> [!IMPORTANT]
> 请确保 `.whl` 与当前环境匹配：
> - Python：cp312（Python 3.12）
> - 平台：win_amd64
> - CUDA：cu124（需与你本机 CUDA/驱动适配）

#### 启动应用

**Paper2Any - 论文工作流 Web 前端（推荐）**

```bash
# 启动后端 API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# 启动前端（新终端）
cd frontend-workflow
npm install
npm run dev
```

**配置前端代理**

修改 `frontend-workflow/vite.config.ts` 中的 `server.proxy`：

```typescript
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

**Windows 加载 MinerU 预训练模型**

```powershell
# PowerShell环境下启动
vllm serve opendatalab/MinerU2.5-2509-1.2B `
  --host 127.0.0.1 `
  --port 8010 `
  --logits-processors mineru_vl_utils:MinerULogitsProcessor `
  --gpu-memory-utilization 0.6 `
  --trust-remote-code `
  --enforce-eager
```

---

### 启动应用

#### 🎨 Web 前端（推荐）

```bash
# 启动后端 API
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000

# 启动前端（新终端）
cd frontend-workflow
npm install
npm run dev
```

访问 `http://localhost:3000`。

---

### 🖥️ CLI 脚本（命令行界面）

Paper2Any 提供独立的 CLI 脚本，支持命令行参数输入，可直接执行工作流，无需启动 Web 前后端。

#### 环境变量配置

通过环境变量配置 API 访问（可选）：

```bash
export DF_API_URL=https://api.openai.com/v1  # LLM API 地址
export DF_API_KEY=sk-xxx                      # API 密钥
export DF_MODEL=gpt-4o                        # 默认模型
```

#### 可用的 CLI 脚本

**1. Paper2Figure CLI** - 生成科学图表（3种类型）

```bash
# 从 PDF 生成模型架构图
python script/run_paper2figure_cli.py \
  --input paper.pdf \
  --graph-type model_arch \
  --api-key sk-xxx

# 从文本生成技术路线图
python script/run_paper2figure_cli.py \
  --input "Transformer 架构与注意力机制" \
  --input-type TEXT \
  --graph-type tech_route

# 生成实验数据可视化图表
python script/run_paper2figure_cli.py \
  --input paper.pdf \
  --graph-type exp_data
```

**图表类型：** `model_arch`（模型架构图）、`tech_route`（技术路线图）、`exp_data`（实验数据图）

**2. Paper2PPT CLI** - 将论文转换为 PPT 演示文稿

```bash
# 基础用法
python script/run_paper2ppt_cli.py \
  --input paper.pdf \
  --api-key sk-xxx \
  --page-count 15

# 自定义风格
python script/run_paper2ppt_cli.py \
  --input paper.pdf \
  --style "学术风格；中文；现代设计" \
  --language zh
```

**3. PDF2PPT CLI** - 一键将 PDF 转换为可编辑 PPT

```bash
# 基础转换（无 AI 增强）
python script/run_pdf2ppt_cli.py --input slides.pdf

# 启用 AI 增强
python script/run_pdf2ppt_cli.py \
  --input slides.pdf \
  --use-ai-edit \
  --api-key sk-xxx
```

**4. Image2PPT CLI** - 将图片转换为可编辑 PPT

```bash
# 基础转换
python script/run_image2ppt_cli.py --input screenshot.png

# 启用 AI 增强
python script/run_image2ppt_cli.py \
  --input diagram.jpg \
  --use-ai-edit \
  --api-key sk-xxx
```

**5. PPT2Polish CLI** - 美化现有 PPT 文件

```bash
# 基础美化
python script/run_ppt2polish_cli.py \
  --input old_presentation.pptx \
  --style "学术风格，简洁大方" \
  --api-key sk-xxx

# 使用参考图片保持风格一致性
python script/run_ppt2polish_cli.py \
  --input old_presentation.pptx \
  --style "现代简约风格" \
  --ref-img reference_style.png \
  --api-key sk-xxx
```

> [!NOTE]
> **PPT2Polish 系统要求：**
> - LibreOffice: `sudo apt-get install libreoffice`（Ubuntu/Debian）
> - pdf2image: `pip install pdf2image`
> - poppler-utils: `sudo apt-get install poppler-utils`

#### 通用选项

所有 CLI 脚本都支持以下通用选项：

- `--api-url URL` - LLM API 地址（默认：从 `DF_API_URL` 环境变量读取）
- `--api-key KEY` - API 密钥（默认：从 `DF_API_KEY` 环境变量读取）
- `--model NAME` - 文本模型名称（默认：各脚本不同）
- `--output-dir DIR` - 自定义输出目录（默认：`outputs/cli/{脚本名称}/{时间戳}`）
- `--help` - 显示详细帮助信息

查看完整参数文档，可运行任意脚本并添加 `--help` 参数：

```bash
python script/run_paper2figure_cli.py --help
```

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
<td><img src="https://img.shields.io/badge/进度-85%25-blue?style=flat-square&logo=progress" alt="85%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-模型架构图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-技术路线图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-实验数据图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-Web_前端-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🧩 Paper2Diagram</strong><br><sub>Drawio 绘图</sub></td>
<td><img src="https://img.shields.io/badge/进度-80%25-blue?style=flat-square&logo=progress" alt="80%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-论文或文本生成-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-图片转_Drawio-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-对话式编辑-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-Drawio_PNG_SVG_导出-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2PPT</strong><br><sub>可编辑演示文稿</sub></td>
<td><img src="https://img.shields.io/badge/进度-70%25-yellow?style=flat-square&logo=progress" alt="70%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-Beamer_样式-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-长文_PPT-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-参考模版PPT生成-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-基于知识库的PPT生成-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-表格提取-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-配图提取-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🖼️ PDF2PPT</strong><br><sub>版式保留转换</sub></td>
<td><img src="https://img.shields.io/badge/进度-90%25-green?style=flat-square&logo=progress" alt="90%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-智能抠图-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-版式保留-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-可编辑_PPTX-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🖼️ Image2PPT</strong><br><sub>图片转 PPT</sub></td>
<td><img src="https://img.shields.io/badge/进度-85%25-blue?style=flat-square&logo=progress" alt="85%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-单图与多图输入-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-版式生成-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🎨 PPTPolish</strong><br><sub>智能美化</sub></td>
<td><img src="https://img.shields.io/badge/进度-60%25-yellow?style=flat-square&logo=progress" alt="60%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-样式迁移-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/⚠-布局优化-yellow?style=flat-square" alt="进行中"/><br>
<img src="https://img.shields.io/badge/⚠-参考图美化-yellow?style=flat-square" alt="进行中"/>
</td>
</tr>
<tr>
<td><strong>📚 知识库（KB）</strong><br><sub>KB 工作流</sub></td>
<td><img src="https://img.shields.io/badge/进度-75%25-blue?style=flat-square&logo=progress" alt="75%"/></td>
<td>
<img src="https://img.shields.io/badge/✓-文件入库与向量化-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-语义检索-success?style=flat-square" alt="完成"/><br>
<img src="https://img.shields.io/badge/✓-KB_PPT_播客_思维导图-success?style=flat-square" alt="完成"/>
</td>
</tr>
<tr>
<td><strong>🎬 Paper2Video</strong><br><sub>视频脚本生成</sub></td>
<td><img src="https://img.shields.io/badge/进度-40%25-yellow?style=flat-square&logo=progress" alt="40%"/></td>
<td>
<img src="https://img.shields.io/badge/⚠-脚本与配音-yellow?style=flat-square" alt="进行中"/><br>
<img src="https://img.shields.io/badge/⚠-分镜与素材-yellow?style=flat-square" alt="进行中"/>
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

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

<!-- --- -->

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenDCAI/Paper2Any&type=Date)](https://star-history.com/#OpenDCAI/Paper2Any&Date) -->

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenDCAI/Paper2Any?style=social)](https://github.com/OpenDCAI/Paper2Any/network/members)

<br>

<a name="wechat-group"></a>
<img src="frontend-workflow/public/wechat.png" alt="DataFlow-Agent 社区微信群" width="200"/>
<br>
<sub>扫码加入社区微信群</sub>

<p align="center"> 
  <em> ❤️ Made with by OpenDCAI Team</em>
</p>

</div>
