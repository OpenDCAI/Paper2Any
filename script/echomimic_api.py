#!/usr/bin/env python3
"""
EchoMimic 单实例 API：接收 image + audio，调用 EchoMimic 子进程生成数字人视频，返回视频字节。
每个进程同时只处理一个 /infer 请求（占用时返回 503，供 Nginx 或客户端换实例/重试）。
需在启动前设置 CUDA_VISIBLE_DEVICES，由 start_echomimic_apis.sh 按卡启动多实例。

环境变量（可选）：
  ECHOMIMIC_CONFIG      - 配置文件路径
  ECHOMIMIC_SCRIPT      - infer_audio2vid.py 路径
  ECHOMIMIC_CWD         - 子进程 cwd（EchoMimic 项目根）
  ECHOMIMIC_PYTHON      - 子进程 Python 解释器（p2v/echomimic 环境）
  ECHOMIMIC_INFER_TIMEOUT - 单次推理超时秒数，默认 900
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# 项目根，便于导入 dataflow_agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _configure_runtime_tempdir() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runtime_tmp = Path(
        os.getenv("PAPER2ANY_RUNTIME_TMPDIR", str(project_root / "outputs" / "system" / "tmp"))
    ).expanduser().resolve()
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)


_configure_runtime_tempdir()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import uvicorn

# 默认路径；建议在线上通过 ECHOMIMIC_* 环境变量覆盖。
DEFAULT_ECHOMIMIC_CWD = "/data/users/ligang/EchoMimic"
DEFAULT_CONFIG = "/data/users/ligang/EchoMimic/configs/prompts/animation.yaml"
DEFAULT_SCRIPT = "/data/users/ligang/EchoMimic/infer_audio2vid.py"
DEFAULT_PYTHON = os.getenv("ECHOMIMIC_PYTHON", "/root/miniconda3/envs/echomimic/bin/python")
INFER_TIMEOUT = int(os.getenv("ECHOMIMIC_INFER_TIMEOUT", "900"))

app = FastAPI(title="EchoMimic Inference API")

# 单实例并发控制：同时只处理一个 /infer
_infer_lock: None | object = None
_start_time = time.time()
_total_requests = 0
_in_flight = 0


def _get_lock():
    global _infer_lock
    if _infer_lock is None:
        import threading
        _infer_lock = threading.Lock()
    return _infer_lock


def _run_echomimic_subprocess(source_image: str, driving_audio: str, save_video_dir: str) -> Path | None:
    """与 p2v_tool.run_echomimic_inference 一致的子进程调用，返回生成的 mp4 路径。"""
    from ruamel.yaml import YAML
    config_path = os.getenv("ECHOMIMIC_CONFIG", DEFAULT_CONFIG)
    script_path = os.getenv("ECHOMIMIC_SCRIPT", DEFAULT_SCRIPT)
    cwd = os.getenv("ECHOMIMIC_CWD", DEFAULT_ECHOMIMIC_CWD)
    python_bin = os.getenv("ECHOMIMIC_PYTHON", DEFAULT_PYTHON)

    env = os.environ.copy()
    for key in ["PYTHONHASHSEED", "PYTHONPATH"]:
        env.pop(key, None)
    env["PYTHONHASHSEED"] = "random"
    # 确保子进程用到的 Python 与当前一致，避免从别处启动 API 时未传 ECHOMIMIC_PYTHON
    env["ECHOMIMIC_PYTHON"] = python_bin

    audio_basename = os.path.splitext(os.path.basename(driving_audio))[0]
    save_path = os.path.join(save_video_dir, audio_basename)
    config_bak = config_path.replace(".yaml", "_{}.yaml".format(audio_basename))

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml_rt.load(f)
    config_data["test_cases"] = {source_image: [driving_audio]}
    with open(config_bak, "w", encoding="utf-8") as f:
        yaml_rt.dump(config_data, f)

    cmd = [
        python_bin, "-u", script_path,
        "--config", config_bak,
        "--save_path", save_path,
    ]
    import subprocess
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True, timeout=INFER_TIMEOUT)
    finally:
        if os.path.exists(config_bak):
            os.remove(config_bak)

    # 输出一般为 save_path 下 digit_person_withaudio.mp4 或同名目录下
    out_mp4 = Path(save_path) / "digit_person_withaudio.mp4"
    if out_mp4.exists():
        return out_mp4
    # 兼容：可能直接写在 save_video_dir 下
    # for p in Path(save_video_dir).rglob("*.mp4"):
    #     return p
    return None


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """接收 image + audio，返回视频字节（video/mp4）。忙时返回 503。"""
    global _total_requests, _in_flight
    lock = _get_lock()
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Instance busy, try another or retry later")

    _in_flight = 1
    try:
        _total_requests += 1
        with tempfile.TemporaryDirectory(prefix="echomimic_api_") as tmpdir:
            tmp = Path(tmpdir)
            img_path = tmp / (image.filename or "image.png")
            aud_path = tmp / (audio.filename or "audio.wav")
            with open(img_path, "wb") as f:
                f.write(await image.read())
            with open(aud_path, "wb") as f:
                f.write(await audio.read())

            loop = asyncio.get_event_loop()
            out_mp4 = await loop.run_in_executor(
                None,
                lambda: _run_echomimic_subprocess(str(img_path), str(aud_path), str(tmp)),
            )
            if not out_mp4 or not out_mp4.exists():
                raise HTTPException(status_code=500, detail="EchoMimic did not produce output mp4")
            data = out_mp4.read_bytes()
        return Response(content=data, media_type="video/mp4")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"EchoMimic inference failed: {e}\n{tb}")
    finally:
        _in_flight = 0
        lock.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """简单监控：总请求数、当前是否在处理、运行时长。"""
    return {
        "total_requests": _total_requests,
        "current_in_flight": _in_flight,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


def main():
    port = int(os.getenv("PORT", "8040"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
