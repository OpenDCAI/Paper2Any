# 快速开始

如果你是第一次从 GitHub clone 这个项目，先读这两份文档：

- [开源部署与配置总指南](guides/open_source_deployment.md)
- [安装与环境准备](installation.md)

这页只保留一条“尽快跑起来”的最短路径，适合本地先验证前后端链路。

## 最短路径

### 1. 克隆仓库

```bash
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any
```

### 2. 准备后端 Python 环境

推荐 Python 3.11。

```bash
conda create -n paper2any python=3.11 -y
conda activate paper2any

pip install --upgrade pip
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt
```

### 3. 准备前端 Node 环境

推荐 Node.js 20。

```bash
cd frontend-workflow
npm ci
cd ..
```

### 4. 复制配置模板

```bash
cp fastapi_app/.env.example fastapi_app/.env
cp frontend-workflow/.env.example frontend-workflow/.env
cp deploy/profiles/nv.env.example deploy/profiles/nv.env
```

### 5. 至少填这几项

`fastapi_app/.env`

```bash
BACKEND_API_KEY=your-backend-api-key
APP_BILLING_MODE=free
DF_API_URL=https://your-llm-gateway/v1
DF_API_KEY=your-llm-api-key
```

`frontend-workflow/.env`

```bash
VITE_API_KEY=your-backend-api-key
VITE_API_BASE_URL=
```

说明：

- `VITE_API_KEY` 必须和 `BACKEND_API_KEY` 完全一致。
- 本地 `npm run dev` + Vite 代理模式下，`VITE_API_BASE_URL` 通常留空。
- 如果你准备启用登录、账户点数、历史文件，需要继续补 `SUPABASE_*`。详见 [开源部署与配置总指南](guides/open_source_deployment.md)。

### 6. 启动后端

```bash
bash deploy/start.sh
```

### 7. 启动前端

```bash
bash deploy/start_frontend.sh
```

### 8. 验证

- 前端：`http://127.0.0.1:3000`
- 后端健康检查：`http://127.0.0.1:8000/health`

可以额外验证运行时配置：

```bash
curl -H "X-API-Key: your-backend-api-key" \
  http://127.0.0.1:8000/api/v1/account/runtime-config
```

## 这条路径能跑到什么程度

这套最短路径优先解决“前端能打开、后端能响应、基础 workflow 配置不报错”。

如果你还想把以下能力也稳定跑起来，需要继续补配置或模型服务：

- `Paper2Drawio` / `Image2Drawio`
- `PDF2PPT` 的高保真分割与抠图链路
- `Paper2Video`
- `Supabase` 登录、账户点数、历史文件
- 本地 `SAM3` / `OCR` / `MinerU`

这些都在 [开源部署与配置总指南](guides/open_source_deployment.md) 里有详细说明。
