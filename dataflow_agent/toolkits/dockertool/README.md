# Mini Docker Tool (DataFlow)

该工具用于在最小隔离的 Docker 容器中运行指定的 Python 文件（可选安装 requirements），并在脚本成功完成后保存当前容器环境为镜像 tar，供后续复用。

## 快速开始

- 构建基础镜像（默认 `myorg/dataflow-py:3.11`）：
  - `python -m dataflow.dataflowagent.toolkits.dockertool.build_docker -i myorg/dataflow-py:3.11`

- 运行示例并保存环境：
  - 第三方依赖示例：
    - `python -m dataflow.dataflowagent.toolkits.dockertool.mini_docker --file dataflow/dataflowagent/toolkits/dockertool/examples/hello_thirdparty.py --requirements dataflow/dataflowagent/toolkits/dockertool/examples/requirements_thirdparty.txt --image myorg/dataflow-py:3.11 --save dataflow/dataflowagent/toolkits/dockertool/artifacts/hello_thirdparty.tar --timeout 180`
  - 仅标准库示例：
    - `python -m dataflow.dataflowagent.toolkits.dockertool.mini_docker --file dataflow/dataflowagent/toolkits/dockertool/examples/hello_stdlib.py --image myorg/dataflow-py:3.11 --save dataflow/dataflowagent/toolkits/dockertool/artifacts/hello_stdlib.tar --timeout 120`

运行成功后，会输出 JSON，包含：`success`、`return_code`、`stdout`、`stderr`、`image`、`tag`、`tar_path`、`log_path`。

## 设计要点

- 为避免依赖宿主机挂载的脚本在提交镜像后丢失，运行前会把脚本和 `requirements.txt` 复制到容器内部 `/app_job`。
- 容器以“保活模式”启动，随后通过 `exec_run` 执行 `pip install`（如传入 `requirements`）与 `python -u /app_job/script.py`，以稳定获取退出码和输出。
- 失败时不提交镜像；成功时 `commit` 并保存为 tar 到 `artifacts/`。
- 本地会保存运行日志到 `artifacts/run_logs/`，用于排查问题。

## 目录说明

- `mini_docker.py`：核心工具入口，提供 CLI。
- `build_docker.py`：构建基础镜像脚本（生成 `docker/Dockerfile` 与 `docker/requirements.txt`）。
- `docker/`：镜像构建所需文件（自动生成）。
- `examples/`：示例脚本与示例依赖清单。
- `artifacts/`：输出的镜像 tar 与运行日志（git 忽略 tar 和日志内容）。


