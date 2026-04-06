# 配置文件参考

这页只做一件事：把当前仓库里最重要的配置文件边界说明清楚。

完整启动链路、部署命令、功能依赖和排障，请看：

- [开源部署与配置总指南](open_source_deployment.md)

## 1. 四类配置文件

| 文件 | 作用 | 不该放什么 |
| --- | --- | --- |
| `fastapi_app/.env` | 后端业务配置 | 不要放机器部署参数 |
| `frontend-workflow/.env` | 前端默认展示配置 | 不要放后端第三方业务密钥 |
| `deploy/profiles/*.env` | 机器部署参数 | 不要放 `DF_API_URL`、`SUPABASE_*` 等业务配置 |
| `logs/model_servers.env` | 模型服务运行时自动生成 | 不建议手工维护 |

## 2. `fastapi_app/.env`

这是后端业务配置中心。

典型内容：

- `BACKEND_API_KEY`
- `APP_BILLING_MODE`
- `DF_API_URL` / `DF_API_KEY`
- `PAPER2*_MANAGED_API_*`
- `KB*_MANAGED_API_*`
- `PAPER2DRAWIO_OCR_*`
- `MINERU_API_*`
- `PAPER2VIDEO_*`
- `SUPABASE_*`
- workflow 默认模型

最关键的两个约束：

1. `BACKEND_API_KEY` 必须与前端 `VITE_API_KEY` 一致。
2. `fastapi_app/.env` 会被后端自动加载，不需要手动 `export`。

## 3. `frontend-workflow/.env`

这是前端默认展示配置中心。

典型内容：

- `VITE_API_KEY`
- `VITE_API_BASE_URL`
- `VITE_DEFAULT_LLM_API_URL`
- `VITE_LLM_API_URLS`
- 各 workflow 的默认模型展示值
- `VITE_SUPABASE_*`

最容易误解的点：

- 这些 `VITE_*` 变量并不等于后端真实业务配置
- 在 `APP_BILLING_MODE=free` 时，真正的业务 API URL / API Key 由后端控制

## 4. `deploy/profiles/*.env`

这是部署脚本的机器参数层。

典型内容：

- `APP_PYTHON`
- `APP_PORT`
- `FRONTEND_PORT`
- `PAPER2ANY_PYTHON`
- `SAM3_*`
- `OCR_*`
- `RMBG_MODEL_PATH`

不要放这些：

- `DF_API_URL`
- `DF_API_KEY`
- `SUPABASE_*`
- `PAPER2DRAWIO_OCR_API_KEY`
- 其他第三方业务密钥

## 5. `logs/model_servers.env`

这个文件由模型服务脚本自动写入，通常包含：

- `SAM3_SERVER_URLS`
- `SAM3_HOME`
- `SAM3_CHECKPOINT_PATH`
- `SAM3_BPE_PATH`

后端启动脚本会自动 source 它，所以通常不建议手改。

## 6. 推荐的配置顺序

1. 先复制三个模板：

```bash
cp fastapi_app/.env.example fastapi_app/.env
cp frontend-workflow/.env.example frontend-workflow/.env
cp deploy/profiles/nv.env.example deploy/profiles/nv.env
```

2. 先配通 `BACKEND_API_KEY` / `VITE_API_KEY`
3. 再决定 `APP_BILLING_MODE=free` 还是 `paid`
4. 再补 `DF_API_*` 或每个 workflow 的托管 API
5. 最后才配 `deploy/profiles/*.env` 的 Python、端口和 GPU

## 7. 你最可能先改的字段

### 后端

```bash
BACKEND_API_KEY=
APP_BILLING_MODE=
DF_API_URL=
DF_API_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
```

### 前端

```bash
VITE_API_KEY=
VITE_API_BASE_URL=
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

### 部署 profile

```bash
APP_PYTHON=
PAPER2ANY_PYTHON=
APP_PORT=8000
FRONTEND_PORT=3000
SAM3_ENABLED=1
SAM3_GPUS=
```
