# 安装指南

本指南将帮助您完成 Paper2Any 的安装和环境配置。

## 环境要求

### 系统要求
- **操作系统**: Linux (推荐), Windows 10/11, macOS 10.15+
- **Python**: 3.10 或更高版本
- **内存**: 至少 16GB RAM（推荐 32GB+ 用于大模型推理）
- **存储**: 至少 50GB 可用空间（用于模型缓存和输出文件）
- **GPU**: 可选但推荐（用于加速视觉生成任务）
  - NVIDIA GPU（支持 CUDA 11.8+）
  - 至少 8GB 显存（推荐 16GB+）

### 网络要求
- 稳定的互联网连接（用于下载模型和依赖）
- 能够访问 GitHub, PyPI, HuggingFace

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any
```

### 2. 创建虚拟环境（推荐）

#### 使用 venv (Python 内置)
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 使用 conda
```bash
conda create -n paper2any python=3.10
conda activate paper2any
```

### 3. 安装基础依赖

Paper2Any 提供了多个依赖文件以适应不同场景：

```bash
# 安装核心依赖（必须）
pip install -r requirements-base.txt

# 安装开发依赖（推荐，包含测试和工具）
pip install -r requirements-dev.txt

# 安装 Paper2Any 包（开发模式）
pip install -e .
```

#### Windows 用户注意事项
Windows 用户可以使用 `requirements-win-base.txt` 替代 `requirements-base.txt`：

```bash
pip install -r requirements-win-base.txt
```

### 4. 模型服务配置

Paper2Any 依赖多个外部模型服务来完成各种任务。您需要配置以下服务：

#### 4.1 文本生成模型（必需）
Paper2Any 需要 LLM 服务来处理文本生成任务。您有以下选择：

**选项 A：使用本地部署的模型服务**
```bash
# 示例：使用 Ollama 部署本地模型
ollama pull qwen2.5:7b
ollama serve
```

**选项 B：使用云 API 服务**
- OpenAI GPT 系列（需 API Key）
- 阿里云通义千问（需 API Key）
- DeepSeek（需 API Key）

在 `.env` 文件中配置 API 信息：
```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入您的 API 配置
LLM_API_URL="https://api.openai.com/v1"
LLM_API_KEY="your-api-key"
LLM_MODEL="gpt-4o"
```

#### 4.2 图像生成模型（可选，用于 Paper2Figure/Paper2PPT）
如果使用图像生成功能，需要配置：

- **Stable Diffusion** 或 **DALL-E** API
- 或部署本地 SD WebUI

相关配置可参考 `script/start_model_servers.sh`

#### 4.3 其他模型服务
- **OCR 服务**: 用于提取 PDF 文本（如 PaddleOCR）
- **语音合成**: 用于 Paper2Video（如 Edge-TTS）
- **视频生成**: 用于 Paper2Video（如 Stable Video Diffusion）

### 5. 数据库配置（可选）

Paper2Any 使用 SQLite 作为默认数据库。如需使用其他数据库：

#### SQLite（默认）
无需额外配置，首次运行会自动创建数据库。

#### PostgreSQL
1. 安装 PostgreSQL 和 psycopg2：
   ```bash
   pip install psycopg2-binary
   ```
2. 创建数据库：
   ```sql
   CREATE DATABASE paper2any;
   ```
3. 在 `.env` 中配置连接字符串：
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/paper2any
   ```

### 6. 验证安装

运行以下命令验证安装是否成功：

```bash
# 运行简单测试
python -c "import dataflow_agent; print('DataFlow-Agent installed successfully')"

# 测试 Paper2Any 工作流
python script/run_paper2figure.py --help
```

如果看到帮助信息，说明安装成功。

## Docker 安装（推荐用于生产环境）

### 使用 Docker Compose（最简单）

```bash
# 启动所有服务（包括数据库和模型服务）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 自定义 Docker 构建

```bash
# 构建镜像
docker build -t paper2any:latest .

# 运行容器
docker run -p 7860:7860 -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/models:/app/models \
  paper2any:latest
```

### Docker Compose 配置文件

项目提供的 `docker-compose.yml` 包含三个主要服务：

1. **paper2any-app**: 主应用服务（Gradio + FastAPI）
2. **model-server**: 模型推理服务（可配置）
3. **postgres**: 数据库服务（可选）

## 开发环境配置

### IDE 配置

推荐使用 VS Code 或 PyCharm 作为开发环境：

#### VS Code 配置
1. 安装 Python 扩展
2. 配置工作区设置：
   ```json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true,
     "python.formatting.provider": "black"
   }
   ```

#### PyCharm 配置
1. 设置虚拟环境解释器
2. 启用自动代码格式化
3. 配置运行/调试配置

### 预提交钩子（代码质量）

```bash
# 安装预提交钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

## 故障排除

### 常见问题

#### 1. 依赖安装失败
- **问题**: `pip install` 失败，提示版本冲突
- **解决**: 使用虚拟环境，或尝试：
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install -r requirements-base.txt --no-deps
  ```

#### 2. CUDA 相关错误
- **问题**: 无法导入 torch 或 tensorflow
- **解决**: 确保安装正确版本的 CUDA 工具包：
  ```bash
  # 查看 CUDA 版本
  nvcc --version
  
  # 安装对应版本的 PyTorch
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### 3. 模型下载慢
- **问题**: 下载 HuggingFace 模型速度慢
- **解决**: 使用镜像源：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

#### 4. 内存不足
- **问题**: 运行大模型时内存溢出
- **解决**:
  - 使用 CPU 模式（性能较低）
  - 增加交换空间
  - 使用量化模型

### 获取帮助

如果遇到无法解决的问题：

1. 查看项目的 [GitHub Issues](https://github.com/OpenDCAI/Paper2Any/issues)
2. 搜索类似问题的解决方案
3. 提交新的 Issue（包含详细错误信息）

## 下一步

安装完成后，请继续：

- 📖 [快速开始](quickstart.md) - 学习如何使用 Paper2Any 的基本功能
- 🛠️ [功能指南](guides/) - 深入了解各功能模块
- 🐳 [部署指南](guides/deployment.md) - 学习如何部署到生产环境
