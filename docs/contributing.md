# 贡献指南

这页只讲当前仓库的真实贡献入口，不再沿用旧的 `DataFlow-Agent` / `gradio_app` 文档结构。

## 1. 基本流程

```bash
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

git checkout -b feature/your-feature
```

建议在提交前至少完成三件事：

1. 本地跑通你改动涉及的链路
2. 补最小必要文档
3. 自查没有把真实密钥或本地 `.env` 提交进仓库

## 2. 本地开发环境

### 后端

```bash
conda create -n paper2any python=3.11 -y
conda activate paper2any

pip install --upgrade pip
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt
```

### 前端

```bash
cd frontend-workflow
npm ci
cd ..
```

### 最小启动方式

```bash
bash deploy/start.sh
bash deploy/start_frontend.sh
```

如果你的改动依赖本地模型服务，再继续使用：

```bash
bash script/prepare_local_models.sh
bash script/start_model_servers.sh
```

## 3. 代码位置建议

不同类型改动通常落在这些目录：

| 目标 | 主要目录 |
| --- | --- |
| 工作流编排 | `dataflow_agent/workflow/` |
| agent / 工具能力 | `dataflow_agent/agentroles/`、`dataflow_agent/toolkits/` |
| 后端 API 与适配层 | `fastapi_app/` |
| 前端页面与交互 | `frontend-workflow/` |
| 部署与启动脚本 | `deploy/`、`script/` |
| 文档 | `docs/`、`mkdocs.yml` |

## 4. 添加新 workflow 的最小思路

通常需要同时处理三层：

1. 在 `dataflow_agent/workflow/` 增加或调整 workflow 注册项
2. 在 `fastapi_app/` 暴露对应 API 或适配层
3. 如果要给用户使用，再补 `frontend-workflow/` 页面入口和 `docs/` 文档

示例模式：

```python
from dataflow_agent.workflow.registry import register


@register("my_workflow")
def create_my_workflow_graph():
    ...
```

实际实现请参考现有 `wf_*.py` 文件，不要只照抄旧文档模板。

## 5. 测试与自查

仓库里既有自动化测试，也有大量依赖外部模型与第三方服务的链路。

建议最少执行：

```bash
pytest
```

如果你的改动影响前后端联调，至少再补一次手工验证：

- 前端页面能正常打开
- 后端 `/health` 正常
- 相关 workflow 不因配置缺失而立即报错

## 6. 文档贡献

如果你改了部署、配置、功能入口或用户可见行为，请同步更新 `docs/`。

本地预览方式：

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```

新增文档时，记得同步修改 `mkdocs.yml` 的 `nav`。
