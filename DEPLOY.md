# Paper2Any Deployment

## 1. 配置文件职责

这个项目现在只保留三类配置文件，各管各的，不要重复写同一套 URL / key。

### `fastapi_app/.env`

用途：后端业务配置。

这里放：

- `BACKEND_API_KEY`
- `DF_API_URL` / `DF_API_KEY`
- `LLM_VERIFY_TIMEOUT_SECONDS`
- `LLM_VERIFY_MAX_TOKENS`
- `PAPER2DRAWIO_OCR_*`
- `PAPER2DRAWIO_SEGMENT_HINT_*`
- `PAPER2PPT_SEGMENT_HINT_*`
- `PAPER2VIDEO_*`
- `PAPER2FIGURE_TO_PPT_FORCE_AI_EDIT`
- `MINERU_API_*`
- `SUPABASE_*`
- workflow 默认模型参数
- 后端回退用的 `SAM3_SERVER_URLS`

不要放：

- 前后端主进程启动用的 Python 路径
- GPU 布局
- 端口分配策略
- 机器部署参数

补充：

- `PAPER2VIDEO_CURSOR_LOCAL_PYTHON` / `PAPER2VIDEO_LOCAL_TTS_PYTHON` / `PAPER2VIDEO_TALKING_LOCAL_PYTHON` 这类“某个 workflow 的可选隔离子环境解释器”放在这里。
- `APP_PYTHON` / `PAPER2ANY_PYTHON` 这类“整套服务如何启动”的解释器路径放在 `deploy/profiles/*.env`。

### `frontend-workflow/.env`

用途：前端默认展示配置。

这里放：

- `VITE_API_KEY`
- `VITE_API_BASE_URL`
- `VITE_DEFAULT_LLM_API_URL`
- `VITE_LLM_API_URLS`
- `VITE_DEFAULT_LLM_MODEL`
- `VITE_LLM_VERIFY_TIMEOUT_MS`
- `VITE_PAPER2DRAWIO_MODEL`
- 其他 `VITE_*` 默认模型项

说明：

- `VITE_API_KEY` 必须和 `fastapi_app/.env` 里的 `BACKEND_API_KEY` 一致。
- `VITE_API_BASE_URL` 在本地 `vite` 代理模式下可以留空。

### `deploy/profiles/muxi.env` / `deploy/profiles/nv.env`

用途：机器部署参数。

这里放：

- Python 路径
- 前后端监听端口
- GPU 类型 / GPU 列表
- SAM3 实例数
- 本地模型路径
- 是否启用本地 OCR / 本地 MinerU

不要放：

- `DF_API_URL`
- `DF_API_KEY`
- `VITE_DEFAULT_LLM_API_URL`
- `VITE_DEFAULT_LLM_MODEL`
- `SUPABASE_*`
- 其他业务 URL / key

### `logs/model_servers.env`

用途：运行时自动生成。

这里会写：

- `SAM3_SERVER_URLS`
- `SAM3_HOME`
- `SAM3_CHECKPOINT_PATH`
- `SAM3_BPE_PATH`

这个文件不要手改。

## 2. `deploy/profiles/muxi.env` 常见参数说明

### 后端启动参数

- `APP_HOST`
  后端监听地址。`0.0.0.0` 表示外部机器也能访问。
- `APP_PORT`
  后端端口，当前是 `8000`。
- `APP_WORKERS`
  `uvicorn` worker 数量。
- `APP_PYTHON`
  启动后端时优先使用的 Python 解释器。
- `APP_FALLBACK_PYTHON`
  `APP_PYTHON` 不可用时的回退解释器。

### 前端启动参数

- `FRONTEND_HOST`
  前端 dev server 监听地址。
- `FRONTEND_PORT`
  前端端口，当前是 `3000`。

### 模型服务启动参数

- `PAPER2ANY_PYTHON`
  启动模型服务时使用的 Python。
- `MODEL_SERVER_ENV_FILE`
  模型服务启动后写回运行时环境变量的文件，默认是 `logs/model_servers.env`。

### SAM3 相关参数

- `SAM3_ENABLED`
  是否启动本地 SAM3。
- `SAM3_GPU_MODE`
  GPU 选择模式，`manual` 表示使用 `SAM3_GPUS`。
- `GPU_QUERY_TOOL`
  查 GPU 的工具。Muxi 机器用 `mx-smi`，NVIDIA 机器通常用 `nvidia-smi`。
- `SAM3_GPUS`
  实例分布。比如 `0,0,1,1,1,1,1,1` 表示 8 个实例，前 2 个绑 0 卡，后 6 个绑 1 卡。
- `SAM3_INSTANCES_PER_GPU`
  自动模式下每张卡起几个实例。
- `SAM3_MAX_INSTANCES`
  最大实例数。
- `SAM3_START_PORT`
  SAM3 起始端口，后续实例按顺序递增。
- `SAM3_HOST`
  SAM3 监听地址。
- `SAM3_STARTUP_STRATEGY`
  `sequential` 表示逐个健康检查后再起下一个，比较稳。
- `SAM3_STARTUP_STAGGER_SEC`
  启动实例之间的间隔秒数。
- `SAM3_INSTANCE_HEALTH_TIMEOUT`
  单实例健康检查超时时间。

### 其他本地服务开关

- `OCR_ENABLED`
  是否启动本地 OCR 服务。当前项目在 muxi 上一般设为 `0`，走远端 API。
- `MINERU_LOCAL_ENABLED`
  是否启动本地 MinerU。当前一般设为 `0`，走远端 API。
- `DRIPPER_AUTOSTOP`
  是否自动停止 dripper 相关本地服务。

## 3. 推荐部署方式

### Muxi 机器

1. 准备后端配置。

```bash
cp fastapi_app/.env.example fastapi_app/.env
```

2. 准备前端配置。

```bash
cp frontend-workflow/.env.example frontend-workflow/.env
```

3. 准备机器部署配置。

```bash
cp deploy/profiles/muxi.env.example deploy/profiles/muxi.env
```

4. 只在 `fastapi_app/.env` 填后端业务 key。

例如：

- `DF_API_URL`
- `DF_API_KEY`
- `LLM_VERIFY_TIMEOUT_SECONDS`
- `LLM_VERIFY_MAX_TOKENS`
- `PAPER2DRAWIO_OCR_API_KEY`
- `PAPER2DRAWIO_SEGMENT_HINT_*`
- `PAPER2PPT_SEGMENT_HINT_*`
- `PAPER2VIDEO_CURSOR_*`
- `PAPER2VIDEO_ENABLE_LOCAL_TTS`
- `PAPER2VIDEO_SUBTITLE_FONT_PATH`
- `PAPER2VIDEO_VIDEO_RENDER_THREADS`
- `PAPER2VIDEO_TALKING_*`
- `PAPER2FIGURE_TO_PPT_FORCE_AI_EDIT`
- `MINERU_API_KEY`
- `SUPABASE_*`

补充说明：

- muxi 当前部署的默认视频链路是：`CosyVoice API + LivePortrait API`。
- 所以 `PAPER2VIDEO_ENABLE_LOCAL_TTS=false` 即可，`F5-TTS` / `WhisperX` 不是默认部署必需依赖。
- 若要启用本地 UI-TARS cursor，不建议把 `ui-tars` 装进后端 base 环境；推荐单独建环境，并在 `fastapi_app/.env` 里设置：
  - `PAPER2VIDEO_CURSOR_LOCAL_ENABLED=auto`
  - `PAPER2VIDEO_CURSOR_LOCAL_MODEL_PATH=/your/UI-TARS-1.5-7B`
  - `PAPER2VIDEO_CURSOR_LOCAL_PYTHON=/your/conda/env/bin/python`
  - 按需再配 `PAPER2VIDEO_CURSOR_LOCAL_EXTRA_PYTHONPATH`
- 这套 UI-TARS 是按次拉起的子进程 worker，不常驻内存。

5. 只在 `frontend-workflow/.env` 填前端默认值。

例如：

- `VITE_API_KEY`
- `VITE_DEFAULT_LLM_API_URL`
- `VITE_DEFAULT_LLM_MODEL`
- `VITE_LLM_VERIFY_TIMEOUT_MS`
- `VITE_PAPER2DRAWIO_MODEL`

6. 只在 `deploy/profiles/muxi.env` 填机器参数。

例如：

- `APP_PYTHON`
- `PAPER2ANY_PYTHON`
- `FRONTEND_PORT`
- `SAM3_GPUS`
- `SAM3_MAX_INSTANCES`

7. 启动。

```bash
bash deploy/start_muxi.sh
```

### NVIDIA 机器

```bash
cp deploy/profiles/nv.env.example deploy/profiles/nv.env
bash deploy/start_nv.sh
```

## 4. 当前项目启动链路

### 一键启动

```bash
bash deploy/start_muxi.sh
```

它会依次执行：

1. `script/prepare_local_models.sh`
2. `script/start_model_servers.sh`
3. `deploy/start.sh`
4. `deploy/start_frontend.sh`

### 停止整套服务

```bash
bash deploy/stop_stack.sh
```

### 仅启动后端

```bash
bash deploy/start.sh
```

### 仅启动前端

```bash
bash deploy/start_frontend.sh
```

## 5. 当前仓库里的实际约定

### 后端业务配置看这里

- `fastapi_app/.env`

### 前端默认展示值看这里

- `frontend-workflow/.env`

### Muxi 机器部署参数看这里

- `deploy/profiles/muxi.env`

## 6. 给开源用户的最简规则

只记一句话：

- 业务 key 写 `fastapi_app/.env`
- 前端默认值写 `frontend-workflow/.env`
- 机器参数写 `deploy/profiles/*.env`

不要把同一套 URL / key 在三处重复写。
