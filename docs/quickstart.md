# 🚀 Paper2Any 快速开始指南

本指南帮助您快速上手 Paper2Any 的核心功能。在完成 [安装指南](./installation.md) 后，您可以通过以下方式快速体验 Paper2Any 的强大能力。

## 📊 快速体验 Paper2Figure：科研绘图

Paper2Figure 支持三种主要绘图模式：模型架构图、技术路线图、实验数据图。

### 方式一：命令行快速生成（推荐）

1. **模型架构图生成**：
   ```bash
   python script/run_paper2figure.py --input "tests/2506.02454v1.pdf" --model architecture
   ```

2. **技术路线图生成**：
   ```bash
   python script/run_paper2technical.py --input "tests/2506.02454v1.pdf"
   ```

3. **实验数据图生成**：
   ```bash
   python script/run_paper2expfigure.py --input "tests/2506.02454v1.pdf"
   ```

### 方式二：Web界面交互体验

1. **启动后端服务**：
   ```bash
   cd fastapi_app
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **启动前端界面**：
   ```bash
   cd frontend-workflow
   npm run dev
   ```

3. **访问界面**：
   打开浏览器访问 `http://localhost:3000`，选择"Paper2Figure"功能模块，上传论文PDF或输入文本即可快速生成。

## 🎬 快速体验 Paper2PPT：论文转演示文稿

### 方式一：命令行生成

```bash
# 从论文PDF生成PPT
python script/run_paper2ppt.py --input "tests/2506.02454v1.pdf"

# 从文本生成PPT
python script/run_paper2ppt.py --text "深度学习在医疗影像分析中的应用"

# 生成长文档PPT（40+页）
python script/run_paper2ppt.py --input "long_paper.pdf" --long_doc
```

### 方式二：Web界面使用

1. 确保后端和前端服务正在运行（同上）
2. 访问 `http://localhost:3000`，选择"Paper2PPT"功能模块
3. 上传论文PDF或输入主题，选择风格模板，点击生成

## 🖼️ 快速体验 PDF2PPT：保持版式的PDF转换

### 方式一：命令行生成

```bash
# 基本转换
python script/run_pdf2ppt_parallel.py --input "tests/test_02.pdf"

# 使用MinerU优化版
python script/run_pdf2ppt_with_paddle_sam_mineru.py --input "tests/test_02.pdf"
```

### 方式二：Web界面使用

1. 访问 `http://localhost:3000`，选择"PDF2PPT"功能模块
2. 上传PDF文件，系统会自动进行智能抠图和版式分析
3. 下载可编辑的PPTX文件

## 🎨 快速体验 Image2PPT：图片转演示文稿

### 方式一：命令行生成

```bash
python script/run_image2ppt.py --image "tests/test_02.png"
```

### 方式二：Web界面使用

1. 访问 `http://localhost:3000`，选择"Image2PPT"功能模块
2. 上传图片文件（支持PNG、JPG、JPEG格式）
3. 系统会自动分析图片内容并生成PPT

## ⚡ 快速脚本说明

Paper2Any 提供了多个快速脚本，位于 `script/` 目录下：

| 脚本文件 | 功能 | 常用参数 |
|---------|------|----------|
| `run_paper2figure.py` | 模型架构图生成 | `--input`, `--model`, `--output_dir` |
| `run_paper2technical.py` | 技术路线图生成 | `--input`, `--style`, `--output_dir` |
| `run_paper2expfigure.py` | 实验数据图生成 | `--input`, `--chart_type`, `--output_dir` |
| `run_paper2ppt.py` | 论文转PPT | `--input`, `--text`, `--long_doc`, `--output` |
| `run_pdf2ppt_parallel.py` | PDF转PPT（并行版） | `--input`, `--output`, `--workers` |
| `run_pdf2ppt_with_paddle_sam_mineru.py` | PDF转PPT（优化版） | `--input`, `--output`, `--gpu_id` |
| `run_image2ppt.py` | 图片转PPT | `--image`, `--output` |

## 🔧 配置说明

### 环境变量配置

在运行前，请确保已配置必要的环境变量：

```bash
# API密钥配置
export DF_API_KEY="your_api_key_here"

# 可选：自定义API端点
export DF_API_URL="http://your-api-gateway/v1/"

# 可选：MinerU GPU资源配置
export MINERU_DEVICES="0,1"  # 使用GPU 0和1
```

### Supabase配置（Web功能必需）

在 `frontend-workflow/.env` 文件中配置：

```bash
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_JWT_SECRET=your_jwt_secret
```

## 📁 输出文件说明

所有生成的输出文件默认保存在以下目录：

- **Paper2Figure输出**：`outputs/paper2fig_ppt/{timestamp}/`
  - `ppt_pages/`：PPT页面图片
  - `clean_backgrounds/`：去背景后的图片
  - `final_output.pptx`：最终PPT文件

- **Paper2PPT输出**：`outputs/paper2ppt/{timestamp}/`
  - `ppt_pages/`：PPT页面
  - `final_output.pptx`：最终PPT文件

- **PDF2PPT输出**：`outputs/pdf2ppt/{timestamp}/`
  - `pages/`：处理后的页面
  - `final_output.pptx`：最终PPT文件

## 🐳 Docker快速体验

如果您不想在本地安装环境，可以使用Docker快速体验：

```bash
# 克隆项目
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 启动所有服务
docker-compose up -d

# 访问Web界面
# 前端：http://localhost:3000
# 后端API：http://localhost:8000
```

## ❓ 常见问题

### Q1: 运行时提示缺少依赖？
A: 请确保已按照 [安装指南](./installation.md) 安装了所有依赖，特别是LaTeX引擎（tectonic）和系统工具（Inkscape、LibreOffice）。

### Q2: 生成速度慢怎么办？
A: 可以尝试以下优化：
1. 使用 `--workers` 参数并行处理（如果脚本支持）
2. 确保已正确配置GPU资源（对于MinerU等需要GPU的组件）
3. 调整模型服务配置，减少等待时间

### Q3: 如何自定义生成风格？
A: 大多数脚本支持 `--style` 或 `--template` 参数，可以指定不同的生成风格。您也可以修改 `dataflow_agent/promptstemplates/` 中的提示词模板来自定义风格。

### Q4: 生成的PPT无法编辑？
A: Paper2Any 生成的PPT是完全可编辑的PPTX格式。如果遇到问题，请确保：
1. 使用最新版本的Microsoft PowerPoint或LibreOffice
2. 检查文件扩展名是否为 `.pptx`
3. 尝试使用脚本的 `--output_format pptx` 参数（如果支持）

## 📚 下一步

- 查看 [详细功能指南](../docs/guides/) 了解每个功能的深度用法
- 了解 [系统架构](../docs/index.md#系统架构) 理解内部工作原理
- 参考 [API文档](../docs/api/) 进行二次开发
- 参与 [贡献指南](../docs/contributing.md) 帮助改进项目

## 🆘 获取帮助

如果在使用过程中遇到问题：
1. 查看 [FAQ](./faq.md) 寻找常见问题解答
2. 提交 [GitHub Issue](https://github.com/OpenDCAI/Paper2Any/issues)
3. 加入 [微信社群](../README.md#wechat-group) 获取实时帮助

---

**开始您的Paper2Any之旅吧！** 🎉
