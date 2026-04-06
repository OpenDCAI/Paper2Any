# 安装与环境准备

这页专门回答两个问题：

1. 这个项目到底需要哪些运行环境和系统依赖？
2. 前端、后端、模型服务是否必须共用一个环境？

如果你还没开始配置仓库，先读：

- [开源部署与配置总指南](guides/open_source_deployment.md)

## 1. 环境关系

结论很明确：通常不是一个环境。

| 组件 | 推荐运行时 | 备注 |
| --- | --- | --- |
| 前端 | Node.js 20 | `frontend-workflow/`，开发态默认用 Vite |
| 后端 | Python 3.11 | `fastapi_app/`，FastAPI API 层 |
| 模型服务 | Python 3.11 | 可以与后端共用，也可以拆出独立 Python |
| 视频子 worker | 单独 Python（可选） | 例如 UI-TARS cursor、本地 TTS、本地 talking worker |

推荐理解方式：

- 前端：独立 Node 环境
- 后端：主 Python 环境
- 模型服务：按机器资源决定是否和后端共用

## 2. 推荐版本

| 组件 | 推荐版本 |
| --- | --- |
| Python | 3.11 |
| Node.js | 20 |
| npm | 与 Node 20 配套 |
| OS | Linux |

## 3. 系统依赖

从当前 `Dockerfile` 可以看到，Linux 环境通常建议准备这些系统包：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  curl \
  git \
  inkscape \
  libreoffice \
  poppler-utils \
  wget \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libsndfile1
```

说明：

- `Inkscape`、`LibreOffice`、`poppler-utils` 对图形导出和文档转换链路很常见
- 部分功能还会用到 `wkhtmltopdf`

## 4. 后端安装

推荐：

```bash
conda create -n paper2any python=3.11 -y
conda activate paper2any

pip install --upgrade pip
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt
```

可选：

```bash
pip install -e .
```

## 5. 前端安装

```bash
cd frontend-workflow
npm ci
cd ..
```

## 6. 本地模型相关说明

如果你需要本地 SAM3 / OCR / 视频 worker：

- 先安装后端 Python 环境
- 再准备模型目录
- 再配置 `deploy/profiles/*.env`

请注意：

- `script/prepare_local_models.sh` 只会帮你整理目录、下载 `RMBG-2.0`
- 它不会自动下载完整 SAM3 checkpoint 和源码

## 7. 下一步

安装完环境后，继续看：

- [快速开始](quickstart.md)
- [配置文件参考](guides/configuration.md)
- [开源部署与配置总指南](guides/open_source_deployment.md)
