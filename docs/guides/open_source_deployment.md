# 开源部署与配置总指南

这份文档面向从 GitHub clone 本项目后，准备在本地或自有机器上部署、调试、使用 `Paper2Any` 的开发者与使用者。

目标不是讲“理论上怎么部署”，而是把当前仓库里真实存在的启动链路、配置文件职责、环境边界、功能依赖和常见坑一次讲清楚。

## 1. 先读结论

`Paper2Any` 不是“clone 后执行一个命令就能完整运行全部工作流”的单体项目。当前仓库至少涉及下面几层运行时：

| 层 | 运行时 | 主要目录 | 作用 |
| --- | --- | --- | --- |
| 前端 | Node.js / npm | `frontend-workflow/` | Web UI，开发态默认用 Vite |
| 后端 | Python | `fastapi_app/` | FastAPI API 层，挂接全部 workflow |
| 模型服务 | Python | `script/` + `dataflow_agent/toolkits/model_servers/` | SAM3、OCR 等本地模型服务 |
| 第三方服务 | 外部 API | `.env` 中配置 | LLM、DashScope、MinerU、Supabase 等 |

直接说最重要的几个事实：

1. 前端和后端不是一个环境。
2. 模型服务通常也是 Python，但可以和后端共用，也可以按功能拆成单独 Python。
3. 配置文件不是只有一个 `.env`，而是分成三类：
   - `fastapi_app/.env`
   - `frontend-workflow/.env`
   - `deploy/profiles/*.env`
4. `docker-compose up` 能拉起前后端容器，但不等于所有 workflow 都能完整可用。
5. 如果你只想先把系统跑起来，先做“远端 API + 本地前后端”的最小方案，不要一开始就追求本地全链路。

## 2. 当前项目的真实启动拓扑

### 2.1 标准一键启动链路

项目推荐的一键启动脚本是：

- Muxi 机器：`bash deploy/start_muxi.sh`
- NVIDIA 机器：`bash deploy/start_nv.sh`

这两个脚本的执行顺序一致：

1. `script/prepare_local_models.sh`
2. `script/start_model_servers.sh`
3. `deploy/start.sh`
4. `deploy/start_frontend.sh`

也就是说，完整链路不是“只起前后端”，而是先准备本地模型目录，再起模型服务，再起后端，再起前端。

### 2.2 后端如何读取配置

后端启动入口是 `fastapi_app/main.py`。

关键点：

- `fastapi_app/.env` 会在应用启动时被自动加载，不需要额外 `export`
- 后端统一监听在 `deploy/app_config.sh` 里的 `APP_PORT`，默认 `8000`
- 所有 workflow 路由挂在 `/api/v1/*`
- 生成结果目录通过 `/outputs` 暴露静态文件

健康检查地址：

```text
http://127.0.0.1:8000/health
```

### 2.3 前端如何连接后端

前端开发态默认使用 Vite：

- 启动命令来自 `frontend-workflow/package.json` 的 `npm run dev`
- 默认端口是 `3000`
- 在本地 dev 模式下，`vite.config.ts` 会把 `/api` 和 `/outputs` 代理到 `http://localhost:8000`

这意味着：

- 本地 `npm run dev` 时，`VITE_API_BASE_URL` 通常留空即可
- 如果你以后改成静态构建或跨域部署，就必须显式配置 `VITE_API_BASE_URL`

### 2.4 模型服务和后端是不是一个 Python

不一定。

项目当前设计是：

- 后端主进程优先使用 `APP_PYTHON`
- 模型服务优先使用 `PAPER2ANY_PYTHON`
- 若未单独指定，模型服务可以回退到 `APP_PYTHON`
- 个别 workflow 子能力还能进一步拆环境，例如：
  - `PAPER2VIDEO_CURSOR_LOCAL_PYTHON`
  - `PAPER2VIDEO_LOCAL_TTS_PYTHON`
  - `PAPER2VIDEO_TALKING_LOCAL_PYTHON`

所以推荐理解方式是：

- 前端：单独 Node 环境
- 后端：主 Python 环境
- 模型服务：可共用主 Python，也可拆分
- 某些视频子模块：可以再额外单独建 Python 环境

## 3. 先选你的部署目标

不要在没有目标的情况下配一堆环境变量。建议先在下面三种路线里选一种。

### 路线 A：最小可运行

适合：

- 先验证前后端能通
- 先体验核心页面
- 你准备依赖远端 LLM / OCR / MinerU / Supabase

特点：

- 起前端 + 后端即可
- 本地模型服务不是必选
- 最快上手

### 路线 B：本地完整链路

适合：

- 有 Linux + GPU 机器
- 需要把依赖尽量落到自己机器上
- 想跑 SAM3、本地 OCR、本地视频子 worker

特点：

- 需要准备模型目录
- 需要配置 `deploy/profiles/*.env`
- 依赖更多，排障成本也更高

### 路线 C：Docker 前后端容器

适合：

- 想快速容器化 API 和前端页面
- 已经有外部模型服务
- 不打算把 SAM3/OCR 一起塞进 `docker-compose`

特点：

- `docker-compose.yml` 当前只覆盖前后端容器
- 不是完整工作流基础设施

## 4. 环境准备

## 4.1 操作系统建议

当前仓库里的部署脚本以 Bash 为主，最稳的环境是 Linux。

推荐：

- Ubuntu 22.04+
- Debian 12+
- 带 NVIDIA GPU 的 Linux 主机

Windows/macOS 不是当前 `deploy/*.sh` 的首选目标环境。如果你用的是这些系统，建议先只跑前后端开发态，或自己改写启动脚本。

## 4.2 推荐版本

| 组件 | 推荐 |
| --- | --- |
| Python | 3.11 |
| Node.js | 20 |
| npm | 与 Node 20 配套版本 |
| Shell | bash |

## 4.3 系统包

从当前 `Dockerfile` 看，项目在 Linux 上常用到这些系统依赖：

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

额外说明：

- `wkhtmltopdf` 在 `Dockerfile` 里也被安装，和部分导出链路相关
- `LibreOffice` 常用于 Office/PDF 相关转换
- `Inkscape` 常用于图形导出

## 5. 克隆仓库

```bash
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any
```

## 6. 安装运行环境

## 6.1 后端 Python 环境

建议使用 `conda` 或者独立虚拟环境。推荐 Python 3.11。

```bash
conda create -n paper2any python=3.11 -y
conda activate paper2any

pip install --upgrade pip
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt
```

如果你需要本地包开发模式：

```bash
pip install -e .
```

说明：

- `requirements-paper.txt` 已经包含 `requirements-base.txt`
- 当前部署脚本会检查后端环境里是否至少有这些运行时依赖：`cv2`、`cairosvg`、`fastapi`、`moviepy`、`supabase`、`torch`、`uvicorn`

## 6.2 前端 Node 环境

```bash
cd frontend-workflow
npm ci
cd ..
```

## 6.3 可选的子环境

如果你打算启用视频相关的本地 worker，可能还需要额外准备单独 Python 环境，然后把解释器路径写进 `fastapi_app/.env`：

- `PAPER2VIDEO_CURSOR_LOCAL_PYTHON`
- `PAPER2VIDEO_LOCAL_TTS_PYTHON`
- `PAPER2VIDEO_TALKING_LOCAL_PYTHON`

这不是起系统必需条件，只是某些高级功能的隔离运行方式。

## 7. 理解四类关键配置文件

当前项目最容易让新用户混乱的点，就是“到底该改哪个 `.env`”。

请按下面这个分工理解。

| 文件 | 作用 | 典型内容 |
| --- | --- | --- |
| `fastapi_app/.env` | 后端业务配置 | 后端 API key、LLM 网关、Supabase、视频/Drawio/MinerU 业务配置 |
| `frontend-workflow/.env` | 前端默认展示配置 | `VITE_*` 变量、前端 API key、默认模型和地址 |
| `deploy/profiles/*.env` | 机器部署参数 | Python 路径、端口、GPU、SAM3、本地模型布局 |
| `logs/model_servers.env` | 运行时自动生成 | SAM3 地址、SAM3 路径等，通常不要手改 |

## 7.1 复制模板

最常见的复制动作是：

```bash
cp fastapi_app/.env.example fastapi_app/.env
cp frontend-workflow/.env.example frontend-workflow/.env
cp deploy/profiles/nv.env.example deploy/profiles/nv.env
```

如果你是 Muxi 机器：

```bash
cp deploy/profiles/muxi.env.example deploy/profiles/muxi.env
```

## 7.2 先记住三条铁律

1. `frontend-workflow/.env` 里的 `VITE_API_KEY` 必须和 `fastapi_app/.env` 里的 `BACKEND_API_KEY` 完全一致。
2. 不要把第三方业务密钥重复写到 `deploy/profiles/*.env`。
3. 不要手工维护 `logs/model_servers.env`，它是模型服务脚本自动生成给后端用的。

## 8. 后端配置：`fastapi_app/.env`

## 8.1 必填项

先把下面这几个概念配对清楚。

### `BACKEND_API_KEY`

这是“前端调用本项目后端”的内部鉴权 key，不是 OpenAI、DashScope、Supabase 的 key。

```bash
BACKEND_API_KEY=your-backend-api-key
```

前端必须配置同样的值：

```bash
VITE_API_KEY=your-backend-api-key
```

### `APP_BILLING_MODE`

这是项目当前非常关键的运行模式开关。

```bash
APP_BILLING_MODE=paid
```

可选值：

- `paid`
  - 用户在前端页面手动填写业务 API URL / API Key
  - 平台默认不扣点
- `free`
  - 业务模型由后端 `.env` 托管
  - 前端很多页面会隐藏手填 API 区域
  - 配额/点数由后端 runtime config 控制

### `DF_API_URL` / `DF_API_KEY`

这是通用回退 LLM 网关。

```bash
DF_API_URL=https://your-llm-gateway/v1
DF_API_KEY=your-llm-api-key
```

在 `free` 模式下：

- 如果某个 workflow 没有单独配置 `PAPER2*_MANAGED_API_*`
- 就会回退到这组 `DF_*`

### 每个 workflow 的托管 API

推荐做法是：你打算开放哪些 workflow，就显式配置哪些 `*_MANAGED_API_URL` / `*_MANAGED_API_KEY`。

例如：

```bash
PAPER2PPT_MANAGED_API_URL=https://your-llm-gateway/v1
PAPER2PPT_MANAGED_API_KEY=your-paper2ppt-key

PDF2PPT_MANAGED_API_URL=https://your-llm-gateway/v1
PDF2PPT_MANAGED_API_KEY=your-pdf2ppt-key
```

这样做的好处是：

- 配置意图清晰
- 不会所有 workflow 都隐式共用 `DF_*`
- 后续切不同模型网关更容易

## 8.2 功能相关配置

### Drawio / OCR

`Paper2Drawio`、`Image2Drawio` 相关链路通常需要 OCR / VLM 配置：

```bash
PAPER2DRAWIO_OCR_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PAPER2DRAWIO_OCR_API_KEY=your_dashscope_key
```

### MinerU

PDF 解析相关能力常会依赖 MinerU：

```bash
MINERU_API_BASE_URL=https://mineru.net/api/v4
MINERU_API_KEY=your_mineru_api_key
```

### 视频

`Paper2Video` 相关配置项很多，文档里不建议一开始全开。通常你只需要先明确自己走哪条链路：

- 云 CosyVoice / LivePortrait
- 本地 cursor worker
- 本地 TTS
- 本地 talking-head

最常见的是先只配云端：

```bash
COSYVOICE_KEY=your_cosyvoice_key
LIVEPORTRAIT_KEY=your_liveportrait_key
```

## 8.3 Supabase：可选但很重要

如果你需要下面这些功能，就必须补全 Supabase：

- 登录 / 注册 / 匿名登录
- 账户页
- 点数 / 邀请码
- 历史文件
- 云端账号级配额

后端至少需要：

```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

如果你不配 Supabase：

- 系统依然可以做匿名或本地测试
- 但账户、登录、历史、邀请码、账号点数这些能力会不完整或不可用

## 9. 前端配置：`frontend-workflow/.env`

## 9.1 必填项

### `VITE_API_KEY`

必须和后端一致：

```bash
VITE_API_KEY=your-backend-api-key
```

### `VITE_API_BASE_URL`

在不同部署模式下含义不同：

- Vite dev 模式：通常留空
- 静态部署 / Nginx / CDN：必须写成完整后端地址

示例：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 9.2 只是前端默认值，不是后端真实业务配置

这些变量只影响前端默认展示，不一定是最终生效值：

- `VITE_DEFAULT_LLM_API_URL`
- `VITE_LLM_API_URLS`
- `VITE_DEFAULT_LLM_MODEL`
- 各工作流的 `VITE_*MODEL*`

尤其在 `APP_BILLING_MODE=free` 时：

- 真正生效的业务 API URL / API Key 在后端
- 前端的这些值更多只是默认展示和 UI 初始值

## 9.3 前端 Supabase

如果你启用登录：

```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

## 10. 机器部署配置：`deploy/profiles/*.env`

这是最容易被误填的一类文件。

这里应该放的是：

- Python 路径
- 端口
- GPU 布局
- SAM3 配置
- 本地模型路径

不应该放的是：

- `DF_API_URL`
- `DF_API_KEY`
- `SUPABASE_*`
- 各类业务 API key

### 10.1 典型字段

```bash
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=1
APP_PYTHON=/path/to/python

FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=3000
FRONTEND_NPM=/path/to/npm

PAPER2ANY_PYTHON=/path/to/python

SAM3_ENABLED=1
SAM3_GPU_MODE=auto
SAM3_START_PORT=8021
```

### 10.2 `APP_PYTHON` 和 `PAPER2ANY_PYTHON` 的区别

- `APP_PYTHON`：后端 `uvicorn` 主进程使用
- `PAPER2ANY_PYTHON`：模型服务使用

如果你不拆环境，两个都可以指向同一个 Python。

## 11. 本地模型目录和模型服务

## 11.1 `prepare_local_models.sh` 做什么

这个脚本的职责是“整理本地模型目录”。

它会做的事：

- 把 legacy 目录里的 `sam3.pt` / `bpe` / `sam3` 源码拷到当前仓库标准目录
- 下载 `RMBG-2.0`

它不会自动帮你下载完整 SAM3 资产。

## 11.2 如果你要启本地 SAM3，必须先准备这些路径

至少要满足下面这些文件存在：

```text
models/sam3/sam3.pt
models/sam3/bpe_simple_vocab_16e6.txt.gz
models/sam3-official/sam3/
```

或者你提供一个 legacy 资产根目录给：

```bash
PAPER2ANY_ASSET_ROOT=/your/asset/root
```

然后由 `prepare_local_models.sh` 复制到仓库标准目录。

## 11.3 `start_model_servers.sh` 做什么

这个脚本会按 profile 配置启动：

- SAM3 服务
- OCR 服务（如果开启）

并把运行时结果写到：

```text
logs/model_servers.env
```

这个文件随后会被 `deploy/start.sh` 自动读取。

## 11.4 什么时候可以跳过本地模型服务

如果你满足以下任一条件，可以先不启本地模型服务：

- 你只验证前后端链路
- 你暂时不使用依赖 SAM3 的工作流
- 你已经有外部可达的 `SAM3_SERVER_URLS`
- 你计划把 OCR / MinerU 完全交给远端 API

## 12. 推荐启动方式

## 12.1 NVIDIA 机器：推荐

```bash
cp deploy/profiles/nv.env.example deploy/profiles/nv.env

# 编辑 deploy/profiles/nv.env
# 编辑 fastapi_app/.env
# 编辑 frontend-workflow/.env

bash deploy/start_nv.sh
```

## 12.2 Muxi 机器：推荐

```bash
cp deploy/profiles/muxi.env.example deploy/profiles/muxi.env

bash deploy/start_muxi.sh
```

## 12.3 手动分步启动

如果你不想直接跑 one-click，可以分开启动。

### 方式 A：只起前后端

```bash
bash deploy/start.sh
bash deploy/start_frontend.sh
```

适合：

- 先看界面
- 只用远端模型服务
- 不需要本地 SAM3

### 方式 B：先 source profile，再手动分步

```bash
set -a
source deploy/profiles/nv.env
set +a

bash script/prepare_local_models.sh
bash script/start_model_servers.sh
bash deploy/start.sh
bash deploy/start_frontend.sh
```

### 方式 C：单 GPU 本地开发

仓库还提供了单机开发脚本：

```bash
bash script/start_local_sam3_dev.sh
```

它会起：

- 单个 SAM3 服务
- 后端
- 前端

更适合本地开发调试，而不是正式部署。

## 12.4 直接裸命令开发

如果你只调前后端代码，不想走部署脚本，也可以直接起：

后端：

```bash
uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd frontend-workflow
npm run dev -- --host 0.0.0.0 --port 3000
```

注意：

- 裸命令模式下，模型服务不会自动启动
- 你自己需要保证 `.env` 和依赖已准备好

## 13. 启动后如何验证

### 13.1 验证后端健康

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok"}
```

### 13.2 验证运行时配置

```bash
curl -H "X-API-Key: your-backend-api-key" \
  http://127.0.0.1:8000/api/v1/account/runtime-config
```

### 13.3 验证前端

打开：

```text
http://127.0.0.1:3000
```

### 13.4 验证模型服务

如果启用了本地 SAM3，可以检查：

```text
logs/model_servers.env
```

里面是否写出了 `SAM3_SERVER_URLS`。

## 14. 使用方式

## 14.1 浏览器入口

默认访问：

```text
http://127.0.0.1:3000
```

## 14.2 `free` 模式和 `paid` 模式的差别

### `APP_BILLING_MODE=free`

- 前端很多页面不再要求用户手填业务 API
- workflow 请求会走后端托管模型凭据
- 页面右上角显示配额/点数

### `APP_BILLING_MODE=paid`

- 用户在前端填写业务 API URL / API Key
- 平台默认不扣点
- 后端更像 workflow orchestrator

## 14.3 不配 Supabase 时能不能用

可以做本地体验，但要有预期：

- 匿名访问通常还能工作
- 后端 runtime-config / guest quota 还能工作
- 登录、账户、邀请码、历史文件、账号积分会不完整

## 15. 日志、停止与重启

### 15.1 查看日志

后端日志：

```bash
bash deploy/logs.sh app
```

也可以直接看：

```text
logs/app.log
logs/frontend.log
logs/sam3*.log
```

### 15.2 停止整套服务

```bash
bash deploy/stop_stack.sh
```

### 15.3 分别停止

```bash
bash deploy/stop.sh
bash deploy/stop_frontend.sh
bash script/stop_model_servers.sh
```

## 16. Docker 方案说明

项目提供了：

- `Dockerfile`：后端镜像
- `frontend-workflow/Dockerfile`：前端静态构建镜像
- `docker-compose.yml`：前后端组合

但要注意当前边界：

1. `docker-compose.yml` 主要是前后端容器编排。
2. 它不会自动起完整本地 SAM3/OCR 集群。
3. 如果你走前端静态构建，`VITE_API_BASE_URL` 需要在构建时确定。

简单来说：

- `docker-compose up` 适合“把 UI + API 包起来”
- 不等于“本地全能力工作流平台一键完备”

## 17. 常见问题与排障

## 17.1 前端能打开，但所有 API 都 401

优先检查：

- `fastapi_app/.env` 里的 `BACKEND_API_KEY`
- `frontend-workflow/.env` 里的 `VITE_API_KEY`

这两个值必须完全一致。

## 17.2 前端能打开，但请求打不到后端

看你是哪种模式：

- Vite dev 模式：`VITE_API_BASE_URL` 一般留空，依赖本地代理
- 静态部署模式：必须显式写完整后端地址

## 17.3 `deploy/start.sh` 启不来

优先检查：

- `APP_PYTHON` 是否可执行
- 当前 Python 是否安装了后端依赖
- `logs/app.log` 是否有缺包或导入错误

## 17.4 `docker-compose` 起起来了，但 Drawio / PDF2PPT / 视频不可用

这是预期内常见问题。

原因通常不是前后端本身，而是：

- SAM3 没起
- OCR 没配
- MinerU 没配
- 视频相关第三方服务没配

## 17.5 账户、登录、点数页面异常

先检查：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

如果没配 Supabase，不要把“匿名模式可用”误解成“账号系统也可用”。

## 17.6 本地 SAM3 起不来

先检查这三个路径是否真的存在：

```text
models/sam3/sam3.pt
models/sam3/bpe_simple_vocab_16e6.txt.gz
models/sam3-official/sam3/
```

`prepare_local_models.sh` 不会替你下载完整 SAM3。

## 18. 推荐阅读顺序

如果你是新用户，建议按这个顺序读：

1. 本页：[开源部署与配置总指南](open_source_deployment.md)
2. [快速开始](../quickstart.md)
3. [安装与环境准备](../installation.md)
4. [配置文件参考](configuration.md)
5. 对应 workflow 的功能指南

如果你只想定位某个工作流的输入输出和页面行为，再去看对应功能文档：

- [Paper2Figure](paper2figure.md)
- [Paper2PPT](paper2ppt.md)
- [Paper2Video](paper2video.md)
- [Paper2Technical](paper2technical.md)
- [多模态 API 开发](multimodal_api.md)
