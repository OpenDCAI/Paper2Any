# 🔧 开发者配置指南

## 📖 简介

本文档面向 Paper2Any 项目的开发者，详细讲解如何从零开始配置项目环境、配置模型服务、并成功启动整个系统。

通过本指南，你将学会：
- 如何正确配置前端和后端的环境变量
- 如何理解和使用三层模型配置架构
- 如何配置和启动模型服务器集群
- 如何排查常见的配置问题

## 📋 配置文件概览

Paper2Any 项目包含以下主要配置文件：

| 配置文件 | 路径 | 用途 |
|---------|------|------|
| 前端环境变量 | `frontend-workflow/.env.example` | 配置前端 API 通信、LLM 提供商、Supabase |
| 后端环境变量 | `fastapi_app/.env.example` | 配置后端模型、数据库、API 服务 |
| 模型服务器启动脚本 | `script/start_model_servers.sh` | 配置 MinerU、SAM、OCR 等模型服务 |

### 配置文件之间的关系

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  前端 (frontend-workflow)                                     │
│  配置文件: .env                                               │
│  - VITE_API_KEY: 与后端通信的密钥                             │
│  - VITE_DEFAULT_LLM_API_URL: 默认 LLM API 地址               │
│  - VITE_SUPABASE_URL: 用户认证服务（可选）                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  后端 (fastapi_app)                                          │
│  配置文件: .env                                               │
│  - 三层模型配置架构                                           │
│  - DEFAULT_LLM_API_URL: LLM API 服务地址                     │
│  - SUPABASE_*: 数据库和认证配置（可选）                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  模型服务器集群                                               │
│  启动脚本: script/start_model_servers.sh                     │
│  - MinerU (vLLM): 文档理解模型                               │
│  - SAM: 图像分割模型                                          │
│  - OCR: 光学字符识别服务                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 前端配置详解

### 步骤 1: 创建前端配置文件

```bash
cd frontend-workflow
cp .env.example .env
```

### 步骤 2: 配置内部 API 通信

前端和后端之间通过 API 密钥进行安全通信，**这个密钥必须与后端配置保持一致**。

```bash
# frontend-workflow/.env
VITE_API_KEY=df-internal-2024-workflow-key
```

⚠️ **重要提示**：
- 这个密钥必须与 `fastapi_app/.env` 中的 `API_KEY` 完全一致
- 生产环境中请修改为更安全的密钥
- 不要将包含真实密钥的 `.env` 文件提交到版本控制系统

### 步骤 3: 配置 LLM 提供商

前端需要配置默认的 LLM API 地址，用户可以在界面上选择不同的 API 提供商。

```bash
# frontend-workflow/.env

# 默认 LLM API URL（在 UI 的"API URL"输入框中显示）
VITE_DEFAULT_LLM_API_URL=https://api.apiyi.com/v1

# 可选的 LLM API URL 列表（逗号分隔，用户可在 UI 中选择）
VITE_LLM_API_URLS=https://api.apiyi.com/v1,http://b.apiyi.com:16888/v1,http://123.129.219.111:3000/v1
```

**配置说明**：
- `VITE_DEFAULT_LLM_API_URL`: 前端界面默认显示的 API 地址
- `VITE_LLM_API_URLS`: 用户可以在下拉菜单中选择的 API 地址列表
- 用户可以在生成内容时覆盖这些默认值

**常见 LLM API 提供商**：
- OpenAI 官方: `https://api.openai.com/v1`
- 阿里云百炼: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- DeepSeek: `https://api.deepseek.com/v1`
- 自建代理服务: 根据你的实际部署地址配置

### 步骤 4: 配置 Supabase（可选）

如果你需要用户认证、配额管理和云存储功能，需要配置 Supabase。

```bash
# frontend-workflow/.env

# 取消注释并填写以下配置
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret
```

**获取 Supabase 配置**：
1. 访问 [Supabase Dashboard](https://supabase.com/dashboard)
2. 选择你的项目
3. 进入 Settings → API
4. 复制 Project URL 和 API Keys

**如果不使用 Supabase**：
- 保持这些配置项注释状态
- 项目将使用本地存储和无认证模式运行

### 前端配置完整示例

```bash
# ===========================================
# Internal API Configuration
# ===========================================
VITE_API_KEY=df-internal-2024-workflow-key

# ===========================================
# LLM Provider Configuration
# ===========================================
VITE_DEFAULT_LLM_API_URL=https://api.openai.com/v1
VITE_LLM_API_URLS=https://api.openai.com/v1,https://api.deepseek.com/v1,http://localhost:3000/v1

# ===========================================
# Supabase Configuration (Optional)
# ===========================================
# VITE_SUPABASE_URL=https://your-project.supabase.co
# VITE_SUPABASE_ANON_KEY=your-anon-key
# SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
# SUPABASE_JWT_SECRET=your-jwt-secret
```

## ⚙️ 后端配置详解

### 步骤 1: 创建后端配置文件

```bash
cd fastapi_app
cp .env.example .env
```

### 步骤 2: 配置 Supabase（可选）

如果前端配置了 Supabase，后端也需要相应配置。

```bash
# fastapi_app/.env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 步骤 3: 理解三层模型配置架构 🎯

Paper2Any 采用了灵活的**三层模型配置架构**，让你可以在不同粒度上控制模型选择：

```
Layer 1: 基础模型定义
    ↓
Layer 2: 工作流级别默认模型
    ↓
Layer 3: 角色级别精细控制
```

#### Layer 1: 基础模型定义

定义所有可用的模型名称，这些名称会被后续配置引用。

```bash
# fastapi_app/.env

# ============================================
# Model Configuration - Layer 1: Base Models
# ============================================
MODEL_GPT_4O=gpt-4o
MODEL_GPT_5_1=gpt-5.1
MODEL_CLAUDE_HAIKU=claude-haiku-4-5-20251001
MODEL_GEMINI_PRO_IMAGE=gemini-3-pro-image-preview
MODEL_GEMINI_FLASH_IMAGE=gemini-2.5-flash-image
MODEL_GEMINI_FLASH=gemini-2.5-flash
MODEL_QWEN_VL_OCR=qwen-vl-ocr-2025-11-20

# 默认 LLM API URL（内部服务）
DEFAULT_LLM_API_URL=http://123.129.219.111:3000/v1/
```

**配置说明**：
- 这一层定义了所有可用模型的"别名"
- 你可以根据实际使用的 API 提供商修改模型名称
- `DEFAULT_LLM_API_URL` 是后端调用 LLM 的默认地址

#### Layer 2: 工作流级别默认模型

为每个工作流设置默认模型，快速切换整个工作流的模型。

```bash
# ============================================
# Model Configuration - Layer 2: Workflow-level Defaults
# ============================================

# Paper2PPT 工作流
PAPER2PPT_DEFAULT_MODEL=gpt-5.1
PAPER2PPT_DEFAULT_IMAGE_MODEL=gemini-3-pro-image-preview

# PDF2PPT 工作流
PDF2PPT_DEFAULT_MODEL=gpt-4o
PDF2PPT_DEFAULT_IMAGE_MODEL=gemini-2.5-flash-image

# Paper2Figure 工作流
PAPER2FIGURE_DEFAULT_MODEL=gpt-4o
PAPER2FIGURE_DEFAULT_IMAGE_MODEL=gemini-3-pro-image-preview

# Paper2Video 工作流
PAPER2VIDEO_DEFAULT_MODEL=gpt-4o

# Knowledge Base
KB_EMBEDDING_MODEL=gemini-2.5-flash
KB_CHAT_MODEL=gpt-4o
```

**使用场景**：
- 想要快速切换某个工作流使用的模型
- 例如：将 Paper2PPT 从 GPT-4o 切换到 Claude Haiku
- 只需修改 `PAPER2PPT_DEFAULT_MODEL=claude-haiku-4-5-20251001`

#### Layer 3: 角色级别精细控制

为工作流中的每个具体角色（任务）指定模型，实现最精细的控制。

```bash
# ============================================
# Model Configuration - Layer 3: Role-level (Fine-grained Control)
# ============================================

# Paper2PPT 角色配置
PAPER2PPT_OUTLINE_MODEL=gpt-5.1              # 大纲生成
PAPER2PPT_CONTENT_MODEL=gpt-5.1              # 内容生成
PAPER2PPT_IMAGE_GEN_MODEL=gemini-3-pro-image-preview  # 图像生成
PAPER2PPT_VLM_MODEL=qwen-vl-ocr-2025-11-20   # 视觉语言模型（OCR）
PAPER2PPT_CHART_MODEL=gpt-4o                 # 图表生成
PAPER2PPT_DESC_MODEL=gpt-5.1                 # 图表描述
PAPER2PPT_TECHNICAL_MODEL=claude-haiku-4-5-20251001  # 技术细节

# Paper2Figure 角色配置
PAPER2FIGURE_TEXT_MODEL=gpt-4o
PAPER2FIGURE_IMAGE_MODEL=gemini-3-pro-image-preview
PAPER2FIGURE_VLM_MODEL=qwen-vl-ocr-2025-11-20
PAPER2FIGURE_CHART_MODEL=gpt-4o
PAPER2FIGURE_DESC_MODEL=gpt-5.1
PAPER2FIGURE_TECHNICAL_MODEL=claude-haiku-4-5-20251001
```

**使用场景**：
- 针对特定任务优化模型选择
- 例如：OCR 任务使用专门的视觉模型 `qwen-vl-ocr`
- 技术细节提取使用 Claude Haiku（成本更低）
- 图像生成使用 Gemini Pro（效果更好）

### 步骤 4: 理解配置优先级

三层配置的优先级从高到低：

```
Layer 3 (角色级别) > Layer 2 (工作流级别) > Layer 1 (基础定义)
```

**实际运行逻辑**：
1. 系统首先查找 Layer 3 的角色级别配置
2. 如果未配置，则使用 Layer 2 的工作流级别默认值
3. 如果仍未配置，则使用 Layer 1 定义的基础模型

**实践示例**：

假设你想让 Paper2PPT 的大纲生成使用 Claude Haiku（更便宜），但其他任务仍使用 GPT-5.1：

```bash
# Layer 2: 工作流默认使用 GPT-5.1
PAPER2PPT_DEFAULT_MODEL=gpt-5.1

# Layer 3: 只有大纲生成使用 Claude Haiku
PAPER2PPT_OUTLINE_MODEL=claude-haiku-4-5-20251001
# 其他角色不配置，自动继承 Layer 2 的 gpt-5.1
```

### 步骤 5: 后端配置完整示例

```bash
# ============================================
# Supabase Configuration (Optional)
# ============================================
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key

# ============================================
# Model Configuration - Layer 1: Base Models
# ============================================
MODEL_GPT_4O=gpt-4o
MODEL_GPT_5_1=gpt-5.1
MODEL_CLAUDE_HAIKU=claude-haiku-4-5-20251001
MODEL_GEMINI_PRO_IMAGE=gemini-3-pro-image-preview
MODEL_GEMINI_FLASH_IMAGE=gemini-2.5-flash-image
MODEL_GEMINI_FLASH=gemini-2.5-flash
MODEL_QWEN_VL_OCR=qwen-vl-ocr-2025-11-20

DEFAULT_LLM_API_URL=https://api.openai.com/v1/

# ============================================
# Model Configuration - Layer 2: Workflow-level Defaults
# ============================================
PAPER2PPT_DEFAULT_MODEL=gpt-4o
PAPER2PPT_DEFAULT_IMAGE_MODEL=gemini-3-pro-image-preview

PDF2PPT_DEFAULT_MODEL=gpt-4o
PDF2PPT_DEFAULT_IMAGE_MODEL=gemini-2.5-flash-image

# ============================================
# Model Configuration - Layer 3: Role-level (Fine-grained Control)
# ============================================
# 只配置需要特殊处理的角色，其他角色自动继承 Layer 2
PAPER2PPT_VLM_MODEL=qwen-vl-ocr-2025-11-20
PAPER2PPT_TECHNICAL_MODEL=claude-haiku-4-5-20251001
```

## 🚀 模型服务器配置和启动

### 模型服务器架构

Paper2Any 使用本地模型服务器集群来处理文档解析和图像分割任务：

```
┌─────────────────────────────────────────────────────────────┐
│  MinerU 集群 (vLLM)                                          │
│  - GPU 7, 1, 2, 3                                           │
│  - 端口: 8011, 8012, 8013, 8014                             │
│  - 负载均衡器: 8010                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SAM 集群 (图像分割)                                         │
│  - GPU 4, 5, 6                                              │
│  - 端口: 8021, 8022, 8023                                   │
│  - 负载均衡器: 8020                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  OCR 服务 (CPU)                                              │
│  - 端口: 8003                                                │
│  - Workers: 4                                                │
└─────────────────────────────────────────────────────────────┘
```

### 步骤 1: 配置启动脚本

编辑 `script/start_model_servers.sh` 文件，根据你的硬件配置调整参数。

#### MinerU 配置

```bash
# MinerU Config
MINERU_MODEL="models/MinerU2.5-2509-1.2B"  # 模型路径
MINERU_GPU_UTIL=0.85                        # GPU 显存利用率 (0-1)
MINERU_MAX_SEQS=64                          # 最大并发序列数
MINERU_GPUS=(7 1 2 3)                       # 使用的 GPU ID
MINERU_START_PORT=8011                      # 起始端口号
```

**配置说明**：
- `MINERU_MODEL`: MinerU 模型文件路径，需要提前下载
- `MINERU_GPU_UTIL`: GPU 显存利用率，建议 0.8-0.9
- `MINERU_MAX_SEQS`: 并发处理的序列数，影响吞吐量
- `MINERU_GPUS`: 分配的 GPU 列表，根据你的硬件调整
- `MINERU_START_PORT`: 第一个实例的端口，后续实例递增

#### SAM 配置

```bash
# SAM Config
SAM_GPUS=(4 5 6)                            # 使用的 GPU ID
SAM_START_PORT=8021                         # 起始端口号
```

**配置说明**：
- `SAM_GPUS`: 分配给 SAM 的 GPU 列表
- `SAM_START_PORT`: 第一个 SAM 实例的端口

### 步骤 2: 启动模型服务器

```bash
# 从项目根目录执行
bash script/start_model_servers.sh
```

**启动流程**：
1. 清理旧进程和端口占用
2. 启动 MinerU 集群（每个 GPU 一个实例）
3. 启动 SAM 集群（每个 GPU 一个实例）
4. 启动负载均衡器（MinerU LB: 8010, SAM LB: 8020）
5. 启动 OCR 服务（端口 8003）

**日志文件**：
- MinerU: `logs/mineru_gpu{gpu_id}.log`
- SAM: `logs/sam_{gpu_id}.log`
- 负载均衡器: `logs/mineru_lb.log`, `logs/sam_lb.log`
- OCR: `logs/ocr_server.log`

### 步骤 3: 验证服务运行状态

```bash
# 查看所有日志
tail -f logs/*.log

# 检查 MinerU 负载均衡器
curl http://127.0.0.1:8010/health

# 检查 SAM 负载均衡器
curl http://127.0.0.1:8020/health

# 检查 OCR 服务
curl http://127.0.0.1:8003/health

# 查看端口占用情况
lsof -i:8010,8020,8003
```

## 🎯 完整启动流程

### 启动顺序

按照以下顺序启动整个系统：

```bash
# 1. 启动模型服务器（如果需要本地模型）
bash script/start_model_servers.sh

# 2. 启动后端服务
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动前端服务（新终端）
cd frontend-workflow
npm install  # 首次运行需要安装依赖
npm run dev
```

### 访问应用

- 前端界面: `http://localhost:5173`
- 后端 API 文档: `http://localhost:8000/docs`
- 后端健康检查: `http://localhost:8000/health`

## ❓ 常见配置问题

### 问题 1: API 密钥不匹配

**症状**：前端无法连接后端，返回 401 或 403 错误

**解决方案**：
```bash
# 检查前端配置
cat frontend-workflow/.env | grep VITE_API_KEY

# 检查后端配置（需要在后端代码中查找 API_KEY 配置）
# 确保两者完全一致
```

### 问题 2: 模型配置错误

**症状**：工作流运行时报错 "Model not found" 或 "Invalid model name"

**解决方案**：
1. 检查 Layer 1 是否定义了模型名称
2. 检查 LLM API URL 是否正确
3. 验证 API 提供商是否支持该模型

```bash
# 测试 LLM API 连接
curl -X POST http://your-api-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}'
```

### 问题 3: 端口冲突

**症状**：启动模型服务器时报错 "Address already in use"

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i:8010

# 杀死占用端口的进程
kill -9 <PID>

# 或者重新运行启动脚本（会自动清理）
bash script/start_model_servers.sh
```

### 问题 4: GPU 资源不足

**症状**：MinerU 或 SAM 启动失败，日志显示 "CUDA out of memory"

**解决方案**：
1. 减少 GPU 分配数量
2. 降低 `MINERU_GPU_UTIL` 参数（如 0.7）
3. 减少 `MINERU_MAX_SEQS` 参数（如 32）

```bash
# 编辑启动脚本
vim script/start_model_servers.sh

# 修改配置
MINERU_GPU_UTIL=0.7
MINERU_MAX_SEQS=32
MINERU_GPUS=(7 1)  # 只使用 2 个 GPU
```

### 问题 5: Supabase 连接失败

**症状**：用户认证功能不可用

**解决方案**：
1. 如果不需要用户认证，注释掉所有 Supabase 配置
2. 如果需要，检查 Supabase URL 和 Key 是否正确
3. 验证 Supabase 项目是否正常运行

## 💡 配置最佳实践

### 1. 开发环境 vs 生产环境

**开发环境配置**：
```bash
# 使用本地或测试 API
DEFAULT_LLM_API_URL=http://localhost:3000/v1/

# 使用较小的模型以节省成本
PAPER2PPT_DEFAULT_MODEL=gpt-4o-mini
PAPER2PPT_OUTLINE_MODEL=claude-haiku-4-5-20251001

# 减少 GPU 资源占用
MINERU_GPU_UTIL=0.7
MINERU_GPUS=(0)  # 只使用一个 GPU
```

**生产环境配置**：
```bash
# 使用稳定的 API 服务
DEFAULT_LLM_API_URL=https://api.openai.com/v1/

# 使用高性能模型
PAPER2PPT_DEFAULT_MODEL=gpt-5.1
PAPER2PPT_IMAGE_GEN_MODEL=gemini-3-pro-image-preview

# 充分利用 GPU 资源
MINERU_GPU_UTIL=0.85
MINERU_GPUS=(0 1 2 3)  # 使用多个 GPU
```

### 2. 模型选择建议

**成本优化策略**：
- 大纲生成、技术细节提取：使用 Claude Haiku（成本低）
- 内容生成、图表描述：使用 GPT-4o（平衡性能和成本）
- 图像生成：使用 Gemini Pro（效果好）
- OCR 任务：使用专门的 VLM 模型（qwen-vl-ocr）

**性能优化策略**：
- 关键任务使用最强模型（GPT-5.1, Claude Opus）
- 并行任务使用不同模型避免 API 限流
- 图像任务使用专门的多模态模型

### 3. 安全配置建议

```bash
# ❌ 不要在代码中硬编码密钥
# ❌ 不要将 .env 文件提交到 Git

# ✅ 使用环境变量
export VITE_API_KEY="your-secret-key"

# ✅ 在 .gitignore 中排除配置文件
echo ".env" >> .gitignore
echo "*.env" >> .gitignore

# ✅ 生产环境使用强密钥
VITE_API_KEY=$(openssl rand -hex 32)
```

### 4. 性能优化建议

**GPU 资源优化**：
```bash
# 根据 GPU 显存调整参数
# 24GB GPU: MINERU_GPU_UTIL=0.85, MINERU_MAX_SEQS=64
# 16GB GPU: MINERU_GPU_UTIL=0.75, MINERU_MAX_SEQS=32
# 8GB GPU:  MINERU_GPU_UTIL=0.65, MINERU_MAX_SEQS=16
```

**并发优化**：
```bash
# OCR 服务 workers 数量根据 CPU 核心数调整
# 8 核 CPU: --workers 4
# 16 核 CPU: --workers 8
# 32 核 CPU: --workers 16
```

### 5. 配置文件管理

**推荐的配置文件结构**：
```
project/
├── .env.example          # 配置模板（提交到 Git）
├── .env                  # 本地配置（不提交）
├── .env.development      # 开发环境配置
├── .env.production       # 生产环境配置
└── .env.test             # 测试环境配置
```

**切换环境**：
```bash
# 开发环境
cp .env.development .env

# 生产环境
cp .env.production .env
```

## 📚 相关文档

- [安装指南](../installation.md) - 环境搭建和依赖安装
- [快速开始](../quickstart.md) - 快速体验各项功能
- [CLI 工具](../cli.md) - 命令行工具使用说明

## 🎉 配置完成

恭喜！你已经完成了 Paper2Any 项目的配置。现在可以：

1. 启动模型服务器（如果需要）
2. 启动后端服务
3. 启动前端服务
4. 访问 `http://localhost:5173` 开始使用

如果遇到问题，请参考上面的常见问题部分，或查看项目的 [FAQ 文档](../faq.md)。
