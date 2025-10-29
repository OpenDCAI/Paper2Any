from __future__ import annotations
import asyncio
import argparse
import io
import json
import os
import tarfile
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import docker  # pip install docker
from docker.errors import DockerException, NotFound, APIError, ContainerError

# 使用 build_docker.py 生成的自定义镜像
DOCKER_IMAGE = "myorg/dataflow-py:3.11"
RUN_TIMEOUT = 120  # s


def _run_in_container_sync(
    file_path: Path,
    image: str = DOCKER_IMAGE,
    timeout: int = RUN_TIMEOUT,
) -> Dict[str, Any]:
    """
    旧版：只读挂载、禁网运行（保留兼容）。实际请使用 run_file_in_minidocker。
    """
    client = docker.from_env()

    try:
        client.images.get(image)
    except NotFound:
        client.close()
        return {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": (
                f"image not found: {image}. 请先在宿主机预置镜像（例如 docker load），"
                f"或用 build_docker.py 构建并保存/加载。"
            ),
            "file_path": str(file_path),
        }

    container = None
    try:
        container = client.containers.run(
            image=image,
            command=["python", "/app/script.py"],
            volumes={str(file_path): {"bind": "/app/script.py", "mode": "ro"}},
            detach=True,
            stdout=True,
            stderr=True,
            network_disabled=True,
            read_only=True,
            mem_limit="512m",
            pids_limit=128,
            cpu_quota=100000,
        )

        exit_status = container.wait(timeout=timeout)
        code = exit_status.get("StatusCode", 137)
        result = {
            "success": code == 0,
            "return_code": code,
            "stdout": container.logs(stdout=True, stderr=False).decode(),
            "stderr": container.logs(stdout=False, stderr=True).decode(),
            "file_path": str(file_path),
        }
    except DockerException as e:
        result = {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Docker error: {e}",
            "file_path": str(file_path),
        }
    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass
        client.close()

    return result


def _make_tar_bytes(files: Dict[str, bytes]) -> bytes:
    """
    将内存文件打包为 tar 字节流，用于 put_archive。
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel_path, data in files.items():
            ti = tarfile.TarInfo(name=rel_path)
            ti.size = len(data)
            ti.mtime = int(datetime.now().timestamp())
            tf.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def run_file_in_minidocker(
    file_path: str | Path,
    image: str = DOCKER_IMAGE,
    workdir: Optional[str] = None,
    requirements: Optional[str] = None,
    save_tar_path: Optional[str] = None,
    enable_network: bool = True,
    timeout: int = RUN_TIMEOUT,
) -> Dict[str, Any]:
    """
    将 Python 文件复制进容器，联网安装依赖（可选），运行并在成功后提交镜像并保存为 tar。
    仅保存容器镜像 tar，不额外保存脚本输出文件。
    """
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        return {"success": False, "return_code": -1, "stdout": "", "stderr": f"file not found: {src}"}

    req_path: Optional[Path] = None
    if requirements:
        rp = Path(requirements).expanduser().resolve()
        if not rp.exists():
            return {"success": False, "return_code": -1, "stdout": "", "stderr": f"requirements not found: {rp}"}
        req_path = rp

    client = docker.from_env()

    try:
        client.images.get(image)
    except NotFound:
        client.close()
        return {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": (
                f"image not found: {image}. 请先在宿主机预置镜像（例如 docker load），"
                f"或用 build_docker.py 构建并保存/加载。"
            ),
        }

    # 可选：挂载宿主机工作目录到 /workspace（便于用户脚本访问外部数据）。脚本本身与 requirements 会复制到容器内部。
    volumes = {}
    if workdir:
        volumes[os.path.abspath(workdir)] = {"bind": "/workspace", "mode": "rw"}

    # 通过覆盖 entrypoint 为 sh -lc，一次性执行：pip 安装（可选）+ 运行脚本。
    # 简化为直接依赖容器 stdout/stderr（docker 日志）来获取输出，并在宿主机保存一份副本。
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_filename = f"{Path(file_path).stem}_{run_ts}.log"
    cmd_parts = []
    if req_path:
        cmd_parts.append("python -m pip install --user -r /app/requirements.txt")
    # 直接运行脚本，输出由 docker 日志采集
    cmd_parts.append("python -u /app/script.py")
    cmd_str = " && ".join(cmd_parts)

    # 挂载仅保留用户指定的工作目录，其它文件通过 put_archive 复制进容器，确保 commit 后可用
    mount_volumes = {}
    if volumes:
        mount_volumes.update(volumes)

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    run_logs_dir = artifacts_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    # 改为 run(detach) + exec_run，避免 shell 行为影响退出码和输出
    # 尝试多种保活策略，确保容器处于 running
    container = None
    start_error: Optional[Exception] = None
    strategies = [
        {"entrypoint": ["tail", "-f", "/dev/null"], "command": None},
        {"entrypoint": ["/bin/sh", "-c"], "command": "sleep infinity"},
        {"entrypoint": ["python"], "command": ["-c", "import time; time.sleep(10**9)"]},
    ]
    for strat in strategies:
        try:
            container = client.containers.run(
                image=image,
                entrypoint=strat["entrypoint"],
                command=strat["command"],
                detach=True,
                volumes=mount_volumes,
                network_disabled=not enable_network,
                mem_limit="2g",
                pids_limit=512,
                cpu_quota=200000,
                working_dir="/app",
                environment={"PYTHONUNBUFFERED": "1"},
            )
            start_error = None
            break
        except (APIError, ContainerError, DockerException) as e:
            start_error = e
            continue
    if container is None:
        client.close()
        return {"success": False, "return_code": -1, "stdout": "", "stderr": f"container start error: {start_error}"}

    # 等待容器进入 running 状态，避免后续 exec_run 409 错误
    try:
        waited = 0.0
        max_wait = float(timeout)
        while waited < max_wait:
            container.reload()
            status = getattr(container, "status", None)
            if status == "running":
                break
            if status in {"exited", "dead"}:
                logs = ""
                try:
                    logs = container.logs().decode()
                except Exception:
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    pass
                client.close()
                return {"success": False, "return_code": -1, "stdout": "", "stderr": f"container not running, status={status}. logs: {logs}"}
            await_time = 0.2
            import time as _t
            _t.sleep(await_time)
            waited += await_time
        # 最后再检查一次
        container.reload()
        if getattr(container, "status", None) != "running":
            client.close()
            return {"success": False, "return_code": -1, "stdout": "", "stderr": f"container not running after wait, status={getattr(container, 'status', None)}"}
    except Exception as e:
        logs = ""
        try:
            logs = container.logs().decode()
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": -1, "stdout": "", "stderr": f"container wait error: {e}. logs: {logs}"}

    combined_logs: list[str] = []
    # 创建工作目录并复制脚本/依赖到容器内部，确保 commit 后仍可用
    job_dir = "/app_job"
    try:
        _code, _out = container.exec_run(["/bin/sh", "-c", f"mkdir -p {job_dir}"])
        # 打包文件为 tar 并上传
        files: Dict[str, bytes] = {}
        files["script.py"] = src.read_bytes()
        if req_path:
            files["requirements.txt"] = req_path.read_bytes()
        tar_bytes = _make_tar_bytes(files)
        container.put_archive(job_dir, tar_bytes)
    except Exception as e:
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": -1, "stdout": "", "stderr": f"put files error: {e}"}
    # 先安装依赖（如有）
    if req_path:
        try:
            pip_exit, pip_out = container.exec_run(
                ["python", "-m", "pip", "install", "--user", "-r", f"{job_dir}/requirements.txt"],
                stream=False,
            )
            try:
                combined_logs.append(pip_out.decode())
            except Exception:
                combined_logs.append(pip_out.decode(errors="ignore"))
            if pip_exit not in (0, None):
                # 依赖安装失败
                try:
                    container.remove(force=True)
                except Exception:
                    pass
                client.close()
                return {"success": False, "return_code": pip_exit or -1, "stdout": "".join(combined_logs), "stderr": "pip install failed"}
        except Exception as e:
            try:
                container.remove(force=True)
            except Exception:
                pass
            client.close()
            return {"success": False, "return_code": -1, "stdout": "".join(combined_logs), "stderr": f"pip exec error: {e}"}

    # 运行脚本
    try:
        code, run_out = container.exec_run(["python", "-u", f"{job_dir}/script.py"], stream=False)
        try:
            combined_logs.append(run_out.decode())
        except Exception:
            combined_logs.append(run_out.decode(errors="ignore"))
        if code is None:
            code = 137
    except Exception as e:
        out = "".join(combined_logs)
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": -1, "stdout": out, "stderr": f"run exec error: {e}"}

    stdout_s = "".join(combined_logs)
    stderr_s = ""  # exec_run 已合并输出
    if code != 0 or ("Traceback (most recent call last)" in stdout_s):
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": code, "stdout": stdout_s, "stderr": stderr_s}

    # 成功提交镜像并保存 tar
    repo = f"minidocker/{src.stem}"
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        image_obj = container.commit(repository=repo, tag=tag)
    except DockerException as e:
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": -1, "stdout": stdout_s, "stderr": f"commit error: {e}"}

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    # 将运行日志保存到本地，便于排查问题
    run_logs_dir = artifacts_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_logs_dir / out_filename
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout_s)
            if stderr_s:
                f.write("\n[stderr]\n")
                f.write(stderr_s)
    except Exception:
        # 如果保存日志失败，不影响后续保存 tar
        pass

    tar_path = Path(save_tar_path).expanduser().resolve() if save_tar_path else artifacts_dir / f"{repo.replace('/', '_')}_{tag}.tar"
    try:
        with open(tar_path, "wb") as f:
            for chunk in image_obj.save(named=True):
                f.write(chunk)
    except Exception as e:
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()
        return {"success": False, "return_code": -1, "stdout": stdout_s, "stderr": f"save tar error: {e}"}

    try:
        container.remove(force=True)
    except Exception:
        pass
    client.close()
    return {
        "success": True,
        "return_code": 0,
        "stdout": stdout_s,
        "stderr": stderr_s,
        "image": repo,
        "tag": tag,
        "tar_path": str(tar_path),
        "log_path": str(log_path),
    }


async def _run_py_in_docker(
    file_path: Path,
    image: str = DOCKER_IMAGE,
    timeout: int = RUN_TIMEOUT,
) -> Dict[str, Any]:
    """
    异步封装（旧版），建议换用 run_file_in_minidocker。
    """
    return await asyncio.to_thread(_run_in_container_sync, file_path, image, timeout)


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在 minidocker 中运行 Python 文件并保存镜像 tar")
    parser.add_argument("--file", required=True, help="要运行的 Python 脚本路径")
    parser.add_argument("--image", default=DOCKER_IMAGE, help="基础镜像名:TAG")
    parser.add_argument("--requirements", default=None, help="requirements.txt 路径（可选）")
    parser.add_argument("--save", dest="save_tar", default=None, help="保存 tar 的目标路径（可选）")
    parser.add_argument("--workdir", default=None, help="宿主机工作目录，映射到 /workspace（可选）")
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT, help="运行超时（秒）")
    parser.add_argument("--enable-network", action="store_true", default=True, help="启用容器网络（默认开启）")
    args = parser.parse_args(argv)

    res = run_file_in_minidocker(
        file_path=args.file,
        image=args.image,
        workdir=args.workdir,
        requirements=args.requirements,
        save_tar_path=args.save_tar,
        enable_network=args.enable_network,
        timeout=args.timeout,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("success") else 1


if __name__ == "__main__":
    sys.exit(_cli_main())
