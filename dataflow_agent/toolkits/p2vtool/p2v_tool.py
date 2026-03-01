from __future__ import annotations

from dataflow_agent.logger import get_logger
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union, Optional
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import json
from PIL import Image, ImageFont, ImageDraw
import shutil
import multiprocessing
import string
import cv2
import numpy as np

log = get_logger(__name__)
import re


def pptx_to_pdf(pptx_path: Union[str, Path], output_dir: Union[str, Path]) -> str:
    """
    使用 LibreOffice 将 PPTX（或 PPT）转为 PDF，供 paper2video 等 workflow 使用。

    Args:
        pptx_path: 输入 PPTX/PPT 文件路径
        output_dir: 输出目录，生成的 PDF 将写入此目录，文件名为 pptx_path 的 stem + .pdf

    Returns:
        生成的 PDF 文件路径（字符串）

    Raises:
        FileNotFoundError: 输入文件不存在
        RuntimeError: LibreOffice 未安装或转换失败
    """
    pptx_path = Path(pptx_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not pptx_path.is_file():
        raise FileNotFoundError(f"PPTX file not found: {pptx_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    soffice_bin = os.environ.get("SOFFICE_BIN") or "libreoffice"
    cmd = [
        soffice_bin,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]
    log.info("[p2v] Converting PPTX to PDF: %s -> %s", pptx_path, output_dir)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"LibreOffice conversion timeout: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "LibreOffice not found. Install it (e.g. apt install libreoffice) to support PPTX conversion."
        ) from e
    pdf_name = pptx_path.with_suffix(".pdf").name
    pdf_path = output_dir / pdf_name
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF conversion failed, expected output: {pdf_path}")
    log.info("[p2v] PDF saved: %s", pdf_path)
    return str(pdf_path)


def get_image_paths(directory_path: str) -> List[str]:
    """
    遍历指定目录及其子目录，查找所有常见的图片文件，并按照日期排序，返回它们的路径字符串列表。
    """
    # 1. 常用图片文件扩展名列表
    image_extensions = [
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.svg', '*.webp'
    ]
    
    base_path = Path(directory_path)
    if not base_path.is_dir():
        print(f"Error: Directory not found at {directory_path}")
        return []

    found_image_paths: List[Path] = []
    
    # 2. 递归遍历目录并收集路径
    for ext in image_extensions:
        found_image_paths.extend(base_path.glob(ext))

    #3. 对找到的图片路径按照文件名日期进行排序，确保顺序
    def natural_sort_key(path: Path):
        numbers = re.findall(r'(\d+)', path.name)
        return tuple(int(n) for n in numbers) if numbers else (float('inf'),)
    
    found_image_paths.sort(key=natural_sort_key)
    return [str(p.resolve()) for p in found_image_paths]

def create_subtitle_image(text, font_size=32, font_path="arial.ttf"):
    # fixme: 硬编码了路径，后续可能需要修改
    if font_path == "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc":
        try:
            font = ImageFont.truetype(font_path, font_size, index=2)
        except Exception as e:
            print(f"[Warning] Failed to load font from '{font_path}': {e}")
            print("Using default font (fixed size, font_size will be ignored!)")
            font = ImageFont.load_default()

    dummy_img = Image.new("RGBA", (70, 70))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    box_w = text_w + 2*padding
    box_h = text_h + 2*padding
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 160))  # semi-transparent black

    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))

    return img

# 根据语音识别结果（带时间戳），生成对应的视频字幕片段
def generate_subtitle_clips(sentence_timesteps_file, video_w, video_h, font_size):
    from moviepy.editor import ImageClip
    clips = []
    with open(sentence_timesteps_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for sentence_timestep in datas:
        # fixme:这里的绝对路径是 支持英文的字体，如果是中文的，需要进行修改
        img = create_subtitle_image(sentence_timestep["text"], font_size=font_size, font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        img_array = np.array(img)
        clip = (ImageClip(img_array, ismask=False)
                .set_duration(sentence_timestep["end"] - sentence_timestep["start"])
                .set_start(sentence_timestep["start"])
                .set_position(("center", video_h - font_size*2)))
        clips.append(clip)
    return clips

# 从sentece_timesteps_path中读取有关的sentence的时间和文本，实际上就是保存的cursor.json文件中的内容
def add_subtitles(video_path, output_path, sentence_timesteps_path, font_size):
    from moviepy.editor import VideoFileClip, CompositeVideoClip
    print("[Step 1] Generating subtitle clips...")
    video = VideoFileClip(video_path)
    subs = generate_subtitle_clips(sentence_timesteps_path, video.w, video.h, font_size)

    print("[Step 2] Rendering final video...")
    final = CompositeVideoClip([video] + subs)
    # 使用cpu
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=12, preset="veryfast")
    # 使用GPU加速
    # final.write_videofile(output_path, codec="h264_nvenc", audio_codec="aac")

def render_cursor_on_video(
    input_video: str,
    output_video: str,
    cursor_points: list,          # list of (time, x, y)
    transition_duration: float = 0.1,
    cursor_size: int = 10,
    cursor_img_path: str = "cursor.png"):

    img = Image.open(cursor_img_path)
    img_resized = img.resize((cursor_size, cursor_size))
    img_resized.save(cursor_img_path)


    def get_video_resolution(path):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]
        return width, height

    w, h = get_video_resolution(input_video)
    print(f"Video resolution: {w}x{h}")

    # 记录了鼠标的移动位置信息，以一个列表的形式
    filters = []

    t_first, _, _ = cursor_points[0]
    # 在视频正式开始时，记录光标轨迹之前，让光标静止悬浮在屏幕的正中央
    if t_first > transition_duration:
        cx = w / 2 - cursor_size / 2
        cy = h / 2 - cursor_size / 2
        global_hold = (
            f"overlay=x={cx}:y={cy-20}:"
            f"enable='between(t,0,{round(t_first - transition_duration, 3)})'"
        )
        filters.append(global_hold)
        
    for i in range(1, len(cursor_points)):
        t0, x0, y0 = cursor_points[i - 1]
        t1, x1, y1 = cursor_points[i]

        hold_start = round(t0, 3)
        hold_end = round(t1 - transition_duration, 3)
        if hold_end > hold_start:
            x_hold = x0 - cursor_size / 2
            y_hold = y0 - cursor_size / 2
            hold_expr = (
                f"overlay=x={x_hold}:y={y_hold}:"
                f"enable='between(t,{hold_start},{hold_end})'"
            )
            filters.append(hold_expr)

        move_start = round(t1 - transition_duration, 3)
        move_end = t1
        dx = x1 - x0
        dy = y1 - y0
        x_expr = f"{x0 - cursor_size/2} + ({dx})*(t-{move_start})/{transition_duration}"
        y_expr = f"{y0 - cursor_size/2} + ({dy})*(t-{move_start})/{transition_duration}"
        move_expr = (
            f"overlay=x={x_expr}:y={y_expr}:"
            f"enable='between(t,{move_start},{move_end})'"
        )
        filters.append(move_expr)

    # 最后一个光标点之后到视频结束：保持光标在最后位置，避免末尾几秒光标消失
    video_duration = get_mp4_duration_ffprobe(input_video)
    t_last, x_last, y_last = cursor_points[-1]
    t_hold_end = round(video_duration, 3)
    t_hold_start = round(t_last, 3)
    if t_hold_end > t_hold_start:
        x_hold = x_last - cursor_size / 2
        y_hold = y_last - cursor_size / 2
        final_hold = (
            f"overlay=x={x_hold}:y={y_hold}:"
            f"enable='between(t,{t_hold_start},{t_hold_end})'"
        )
        filters.append(final_hold)

    filter_lines = []
    stream_in = "[0][1]"
    for i, expr in enumerate(filters):
        stream_out = f"[tmp{i}]" if i < len(filters) - 1 else "[vout]"
        filter_lines.append(f"{stream_in} {expr} {stream_out}")
        stream_in = f"{stream_out}[1]"

    filter_complex = "; ".join(filter_lines)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", cursor_img_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"\n✅ Done! Output saved to: {output_video}")


def render_video_with_cursor_from_json(
    video_path,
    out_video_path,
    json_path,
    cursor_img_path,
    transition_duration=0.1,
    cursor_size=16
):
    with open(json_path, "r") as f:
        data = json.load(f)

    cursor_points = []
    for idx, slide in enumerate(data):
        if idx == 0: start_time = slide["start"]
        else: start_time = slide["start"] + 0.5
        x, y = slide["cursor"]
        cursor_points.append((start_time, x, y))
    
    render_cursor_on_video(
        input_video=video_path,
        output_video=out_video_path,
        cursor_points=cursor_points,
        transition_duration=transition_duration,
        cursor_size=cursor_size,
        cursor_img_path=cursor_img_path
    )
'''========================== 解析生成 数字人 相关的函数  =================================='''
def run_echomimic_inference(args):
    from ruamel.yaml import YAML
    source_image, driving_audio, save_video_dir, config_path, script_path, talking_head_env, gpu_id = args
    
    # 处理可能的 PYTHONSHASHSEED 问题
    env = os.environ.copy()
    keys_to_clear = ["PYTHONHASHSEED", "PYTHONPATH"] 
    for key in keys_to_clear:
        if key in env:
            del env[key]
    
    env["PYTHONHASHSEED"] = "random"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    audio_basename = os.path.splitext(os.path.basename(driving_audio))[0]
    save_path = os.path.join(save_video_dir, f"{audio_basename}")
    config_bak = config_path.replace(".yaml", "_{}.yaml".format(audio_basename))
    
    # 修改原来配置文件中的内容，因为原有文件内容中保存文件的地址不对
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True  # 保留引号
    yaml_rt.indent(mapping=2, sequence=4, offset=2) # 保持缩进风格

    # 1. 读取原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml_rt.load(f)

    # 2. 修改 test_cases
    config_data['test_cases'] = {
        source_image: [driving_audio]
    }
    with open(config_bak, 'w', encoding='utf-8') as f:
        yaml_rt.dump(config_data, f)
    
    cmd = [
        talking_head_env, "-u", script_path,
        "--config", config_bak,
        "--save_path", save_path,
    ]
    log.info(f"Starting Task on GPU {gpu_id}: {audio_basename}")
    # fixme: 硬编码了路径，后续可能需要修改
    result = subprocess.run(cmd, cwd="/data/users/ligang/EchoMimic", env=env)

    if os.path.exists(config_bak):
        os.remove(config_bak)
    return result


# ------------------------- LivePortrait 云数字人 API -------------------------
LIVEPORTRAIT_MODEL = "liveportrait"
LIVEPORTRAIT_DETECT_MODEL = "liveportrait-detect"
LIVEPORTRAIT_UPLOAD_API = "https://dashscope.aliyuncs.com/api/v1/uploads"
LIVEPORTRAIT_VIDEO_SYNTHESIS_API = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
LIVEPORTRAIT_FACE_DETECT_API = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/face-detect"
LIVEPORTRAIT_TASKS_API = "https://dashscope.aliyuncs.com/api/v1/tasks"


def _liveportrait_get_upload_policy(api_key: str, model: str = LIVEPORTRAIT_MODEL) -> dict:
    """获取 DashScope 文件上传凭证，用于后续上传图片/音频并得到 oss:// URL。"""
    import requests
    resp = requests.get(
        LIVEPORTRAIT_UPLOAD_API,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data:
        raise RuntimeError(f"LivePortrait getPolicy response missing data: {data}")
    return data["data"]


def _liveportrait_upload_file(api_key: str, file_path: str, model: str = LIVEPORTRAIT_MODEL) -> str:
    """上传本地文件到 DashScope 临时 OSS，返回 oss://... URL。调用 API 时需加 Header X-DashScope-OssResourceResolve: enable。"""
    import requests
    policy_data = _liveportrait_get_upload_policy(api_key, model)
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"LivePortrait upload: file not found: {file_path}")
    key = f"{policy_data['upload_dir']}/{file_path.name}"
    with open(file_path, "rb") as f:
        files = {
            "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
            "Signature": (None, policy_data["signature"]),
            "policy": (None, policy_data["policy"]),
            "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_path.name, f.read()),
        }
        resp = requests.post(policy_data["upload_host"], files=files, timeout=120)
    resp.raise_for_status()
    return f"oss://{key}"


def liveportrait_face_detect(api_key: str, image_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    使用 LivePortrait-detect 检测人物肖像图是否符合数字人输入规范。
    仅支持 HTTP/OSS 链接，会先将本地文件上传到 DashScope 临时 OSS 再调用 face-detect。
    返回 (passed, message)：通过时 passed=True、message 可能为空；不通过时 passed=False、message 为原因（如 No human face detected）。
    """
    import requests
    image_path = Path(image_path)
    if not image_path.is_file():
        return False, "图像文件不存在"
    image_oss = _liveportrait_upload_file(api_key, str(image_path), model=LIVEPORTRAIT_DETECT_MODEL)
    resp = requests.post(
        LIVEPORTRAIT_FACE_DETECT_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-OssResourceResolve": "enable",
        },
        json={
            "model": LIVEPORTRAIT_DETECT_MODEL,
            "input": {"image_url": image_oss},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "code" in data and data.get("code"):
        return False, data.get("message", data.get("code", "检测接口返回错误"))
    output = data.get("output", data)
    passed = output.get("pass", False)
    message = output.get("message") or ""
    return bool(passed), str(message).strip()


def _liveportrait_submit(api_key: str, image_url: str, audio_url: str, template_id: str = "normal") -> str:
    """提交 LivePortrait 视频合成异步任务，返回 task_id。image_url/audio_url 需为 oss:// 或公网 URL。"""
    import requests
    payload = {
        "model": LIVEPORTRAIT_MODEL,
        "input": {"image_url": image_url, "audio_url": audio_url},
        "parameters": {"template_id": template_id},
    }
    resp = requests.post(
        LIVEPORTRAIT_VIDEO_SYNTHESIS_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("output", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"LivePortrait submit response missing task_id: {data}")
    return task_id


def _liveportrait_poll(api_key: str, task_id: str, poll_interval: float = 5.0, max_wait: float = 600.0) -> Optional[str]:
    """轮询任务状态，成功时返回结果视频 URL，失败返回 None 或抛异常。"""
    import requests
    import time
    url = f"{LIVEPORTRAIT_TASKS_API}/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    start = time.monotonic()
    while (time.monotonic() - start) < max_wait:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        output = data.get("output", data)
        status = output.get("task_status") or output.get("status")
        if status == "SUCCEEDED":
            results = output.get("results") or output.get("result")
            if isinstance(results, dict):
                video_url = results.get("video_url") or results.get("url")
                if video_url:
                    return video_url
            if isinstance(results, list) and results:
                video_url = results[0].get("video_url") or results[0].get("url")
                if video_url:
                    return video_url
            return output.get("video_url") or output.get("video_path")
        if status in ("FAILED", "CANCELED"):
            msg = output.get("message") or output.get("code") or str(data)
            raise RuntimeError(f"LivePortrait task failed: {status} - {msg}")
        time.sleep(poll_interval)
    raise TimeoutError(f"LivePortrait task {task_id} did not finish within {max_wait}s")


def _liveportrait_single(
    api_key: str,
    ref_img_path: str,
    audio_path: str,
    save_path: str | Path,
    template_id: str = "normal",
) -> bool:
    """单段：上传图片+音频，提交任务，轮询并下载视频到 save_path。"""
    import requests
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_oss = _liveportrait_upload_file(api_key, ref_img_path)
    audio_oss = _liveportrait_upload_file(api_key, audio_path)
    task_id = _liveportrait_submit(api_key, image_oss, audio_oss, template_id=template_id)
    video_url = _liveportrait_poll(api_key, task_id)
    if not video_url:
        log.warning("[liveportrait] task %s succeeded but no video_url", task_id)
        return False
    resp = requests.get(video_url, timeout=120)
    resp.raise_for_status()
    save_path.write_bytes(resp.content)
    log.info("[liveportrait] saved %s", save_path)
    return True


def _liveportrait_single_with_retry(
    api_key: str,
    ref_img_path: str,
    audio_path: str,
    save_path: str | Path,
    template_id: str = "normal",
    max_retries: int = 3,
    retry_delay: float = 15.0,
) -> bool:
    """单段 LivePortrait 生成，输出校验失败或异常时重试。"""
    import time
    save_path = Path(save_path)
    last_err = None
    for attempt in range(max_retries):
        try:
            if save_path.is_file():
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            ok = _liveportrait_single(api_key, ref_img_path, audio_path, save_path, template_id=template_id)
            if not ok:
                last_err = RuntimeError("_liveportrait_single returned False")
                if attempt < max_retries - 1:
                    log.warning("[liveportrait] attempt %s/%s failed, retry in %.0fs", attempt + 1, max_retries, retry_delay)
                    time.sleep(retry_delay)
                continue
            if not _validate_talking_video_output(save_path, audio_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass
                last_err = RuntimeError("LivePortrait output validation failed")
                log.warning("[liveportrait] output validation failed (attempt %s/%s), retry in %.0fs", attempt + 1, max_retries, retry_delay)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise last_err
            return True
        except Exception as e:
            last_err = e
            if save_path.is_file():
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            log.warning("[liveportrait] attempt %s/%s failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    if last_err:
        raise last_err
    return False


def _call_echomimic_api_single(
    base_url: str,
    ref_img_path: str,
    audio_path: str,
    save_dir: Path,
    timeout_sec: int = 900,
    max_retries: int = 10,
    retry_delay: float = 30.0,
    task_idx: Optional[int] = None,
) -> bool:
    """
    向 EchoMimic API 发送单次推理请求，将返回的视频字节写入 save_dir/<subdir>/digit_person_withaudio.mp4。
    task_idx 非空时用 save_dir/<task_idx>，保证与 input_list 顺序一一对应，避免多线程下同名音频或乱序导致音画错位。
    若返回 503 则等待后重试（由 Nginx 或服务端换实例）。
    """
    import time
    import httpx

    if task_idx is not None:
        out_dir = save_dir / str(task_idx)
    else:
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = save_dir / audio_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "digit_person_withaudio.mp4"

    url = base_url.rstrip("/") + "/infer"
    last_err = None
    for attempt in range(max_retries):
        try:
            with open(ref_img_path, "rb") as f_img, open(audio_path, "rb") as f_aud:
                files = [
                    ("image", ("image.png", f_img.read(), "image/png")),
                    ("audio", ("audio.wav", f_aud.read(), "audio/wav")),
                ]
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(url, files=files)
            if resp.status_code == 503:
                last_err = httpx.HTTPStatusError("503 Service Unavailable", request=resp.request, response=resp)
                log.warning("[echomimic-api] 503 (attempt %s/%s), retry in %.0fs", attempt + 1, max_retries, retry_delay)
                time.sleep(retry_delay)
                continue
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            log.info("[echomimic-api] wrote %s", out_path)
            return True
        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code == 503 and attempt < max_retries - 1:
                log.warning("[echomimic-api] 503 (attempt %s/%s), retry in %.0fs", attempt + 1, max_retries, retry_delay)
                time.sleep(retry_delay)
                continue
            # 500 等错误时打出服务端返回的 body，便于排查
            try:
                body = e.response.text
                if body:
                    log.error("[echomimic-api] server error %s body: %s", e.response.status_code, body[:2000])
            except Exception:
                pass
            raise
        except Exception as e:
            last_err = e
            log.warning("[echomimic-api] attempt %s/%s failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    if last_err:
        raise last_err
    return True


def talking_gen_per_slide(model_name, input_list, project_root, save_dir, env_path, api_key: Optional[str] = None):
    import multiprocessing as mp
    save_dir = Path(save_dir)

    # 云数字人 LivePortrait：上传图片/音频到 DashScope OSS，提交异步任务，轮询并下载；Key 仅从环境变量 LIVEPORTRAIT_KEY 读取
    if model_name == "liveportrait":
        key = (os.getenv("LIVEPORTRAIT_KEY", "") or "").strip()
        if not key:
            raise ValueError("LivePortrait 需要设置环境变量 LIVEPORTRAIT_KEY")
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(4, len(input_list)) or 1
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (ref_img_path, audio_path) in enumerate(input_list):
                out_dir = save_dir / str(idx)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "digit_person_withaudio.mp4"
                fut = executor.submit(_liveportrait_single_with_retry, key, str(ref_img_path), str(audio_path), out_path)
                futures.append((idx, fut))
            for idx, fut in futures:
                try:
                    fut.result()
                    results.append(None)  # 成功无返回值，与 echomimic 一致
                except Exception as e:
                    log.exception("[liveportrait] task %s failed: %s", idx, e)
                    results.append(subprocess.CompletedProcess(args=[], returncode=1))
        return results

    # 若配置了 EchoMimic API（如 Nginx 入口），则走 HTTP 请求，并行打 8 个实例，503 时重试排队
    
    # fixme: 如果不使用nginx，则这个为空串即可。现在这里硬编码一个本地地址，后续需要修改。
    nginx_echomimic_api_url = "http://localhost:8200"
    if model_name == "echomimic" and nginx_echomimic_api_url:
        timeout_sec = int(os.getenv("ECHOMIMIC_CLIENT_TIMEOUT", "900"))
        max_workers = 8
        from concurrent.futures import ThreadPoolExecutor
        tasks = [
            (str(Path(ref_img_path).resolve()), str(Path(audio_path).resolve()))
            for ref_img_path, audio_path in input_list
        ]
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _call_echomimic_api_single,
                    nginx_echomimic_api_url,
                    ref_img_path,
                    audio_path,
                    save_dir,
                    timeout_sec=timeout_sec,
                    task_idx=idx,
                )
                for idx, (ref_img_path, audio_path) in enumerate(tasks)
            ]
            for idx, fut in enumerate(futures):
                try:
                    fut.result()
                except Exception as e:
                    log.exception("[echomimic-api] task %s failed: %s", idx, e)
                    results[idx] = subprocess.CompletedProcess(args=[], returncode=1)
        return results

    # 原有逻辑：本地多进程子进程调用
    gpu_list = [4, 5, 6, 4, 5, 6, 4, 5, 6]
    num_gpus = len(gpu_list)
    task_list = []
    if model_name == "hallo2":
        # fixme：这个文件路径被硬编码了
        config_path = "/data/users/ligang/models/hallo2/configs/inference/long.yaml"
        script_path = "/data/users/ligang/models/hallo2/scripts/inference_long.py"
    elif model_name == "echomimic":
        config_path = "/data/users/ligang/EchoMimic/configs/prompts/animation.yaml"
        script_path = "/data/users/ligang/EchoMimic/infer_audio2vid.py"
    else:
        config_path = ""
        script_path = ""
    for idx, (ref_img_path, audio_path) in enumerate(input_list):
        ref_img_path = Path(ref_img_path)
        audio_path = Path(audio_path)
        gpu_id = gpu_list[len(task_list) % num_gpus]
        task_list.append([
            str(ref_img_path),
            str(audio_path),
            str(save_dir),
            str(config_path),
            str(script_path),
            env_path,
            gpu_id,
        ])

    results = []
    if num_gpus > 1:
        ctx = mp.get_context("spawn")
        # fixme: 这个错误很致命！！！在这段代码之前，某处执行的代码错误的将“PYTHONHASHSEED”设置为了一个64位整数
        # 而python只支持32位的整数，所以会导致使用ctx.Pool时发生错误，无法初始化一个python解释器
        os.environ["PYTHONHASHSEED"] = "0"
        with ctx.Pool(processes=max(num_gpus, len(task_list))) as pool:
            results = pool.map(run_echomimic_inference, task_list)
    else:
        for task_args in task_list:
            result = run_echomimic_inference(task_args)
            results.append(result)
    return results


def get_audio_paths(slide_audio_dir: Optional[str | Path]):
    '''获取 slide_audio_dir 目录下的所有音频文件路径，并按数字顺序排序返回'''
    if isinstance(slide_audio_dir, str):
        slide_audio_dir = Path(slide_audio_dir)
    slide_audio_paths = [
        p for p in slide_audio_dir.iterdir()
        if p.is_file() and re.search(r'\d+', p.name)
    ]

    def get_sort_key(file_path: Path):
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group()) if match else float('inf')
    
    slide_audio_paths.sort(key=get_sort_key)
    slide_audio_paths = [str(p) for p in slide_audio_paths]
    return slide_audio_paths

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_audio_length(audio_path):
    '''获取音频文件(.wav)的总时长（秒）'''
    import wave
    with wave.open(audio_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate
    
def get_mp4_duration_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


# 数字人输出校验：用于失败时触发重试
MIN_TALKING_VIDEO_SIZE = 1024  # 至少 1KB
DURATION_RATIO_MIN, DURATION_RATIO_MAX = 0.3, 2.5  # 视频时长/音频时长 合理区间


def _validate_talking_video_output(
    video_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    min_size: int = MIN_TALKING_VIDEO_SIZE,
    duration_ratio: Tuple[float, float] = (DURATION_RATIO_MIN, DURATION_RATIO_MAX),
) -> bool:
    """
    校验数字人输出视频是否有效：文件存在、大小足够，可选与音频时长比例合理。
    返回 True 表示通过，False 表示需重试。
    """
    p = Path(video_path)
    if not p.is_file():
        return False
    if p.stat().st_size < min_size:
        log.warning("[talking-video] output too small: %s (%s bytes)", p, p.stat().st_size)
        return False
    if audio_path and Path(audio_path).is_file():
        try:
            video_dur = get_mp4_duration_ffprobe(p)
            audio_dur = get_audio_length(audio_path)
        except Exception as e:
            log.warning("[talking-video] cannot get duration for validation: %s", e)
            return False
        if audio_dur <= 0:
            return True
        ratio = video_dur / audio_dur
        if ratio < duration_ratio[0] or ratio > duration_ratio[1]:
            log.warning("[talking-video] duration ratio out of range: video=%.2fs audio=%.2fs ratio=%.2f", video_dur, audio_dur, ratio)
            return False
    return True


'''========================== 解析生成cursor位置信息相关的函数  =================================='''
_GLOBAL_PIPE_BYTEDANCE_SEED = None
def _infer_cursor(instruction, image_path):
    global _GLOBAL_PIPE_BYTEDANCE_SEED
    from transformers import pipeline
    from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code

    # fixme：修改一下这段代码，最好不要从hf上下载，而是在本地就下载好了，但是这个路径或许需要处理！！！
    if _GLOBAL_PIPE_BYTEDANCE_SEED is None:
        _GLOBAL_PIPE_BYTEDANCE_SEED = pipeline("image-text-to-text", model="/data/users/ligang/models/bytedance-seed")
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You must to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction {}".format(instruction)
    messages = [{"role": "user", "content": [{"type": "image", "url": image_path}, {"type": "text", "text": prompt}]},]
    result = _GLOBAL_PIPE_BYTEDANCE_SEED(text=messages)[0]
    response = result['generated_text'][1]["content"]
    
    ori_image = cv2.imread(image_path)
    #fixme: OpenCV 的 shape 返回的是 (height, width, channels)
    original_image_height, original_image_width = ori_image.shape[:2]
    parsed_dict = parse_action_to_structure_output(
        response,
        factor=1000,
        origin_resized_height=original_image_height,
        origin_resized_width=original_image_width,
        model_type="qwen25vl"
    )

    parsed_pyautogui_code = parsing_response_to_pyautogui_code(
        responses=parsed_dict,
        image_height=original_image_height,
        image_width=original_image_width
    )

    match = re.search(r'pyautogui\.click\(([\d.]+),\s*([\d.]+)', parsed_pyautogui_code)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
    else:
        print(instruction)
    return (x, y)

def cursor_infer(args):
    '''根据说话的内容，得到cursor应该指向的位置'''
    slide_idx, sentence_idx, prompt, cursor_prompt, image_path, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    
    point= _infer_cursor(cursor_prompt, image_path)
    torch.cuda.empty_cache()
    result = {
        'slide': slide_idx, 'sentence': sentence_idx, 'speech_text': prompt, 
        'cursor_prompt': cursor_prompt, 'cursor': point,
    }
    return result

'''========================== 解析生成speech相关的函数  =================================='''
def parse_script_with_cursor(script_text):
    '''
    解析脚本的内容，将其分割成（prompt, cursor_prompt）两部分
    '''
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result

def parse_script(script_text):
    '''
    解析脚本的内容，将多个句子合并成一个句子
    '''
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue   
        lines = page.strip().split("\n")
        result.append(" ".join(lines))
    return result

# fixme: 这里需要判断device，可能需要多加考虑
def _transcribe_with_whisperx_impl(audio_path: str, lang: str = "en") -> str:
    """
    子进程内实际执行 whisperx 转写。假定 CUDA_VISIBLE_DEVICES 已由调用方在子进程 env 中设置。
    """
    import torch
    import whisperx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"transcribe_with_whisperx 使用了 device: {device}")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16" if device == "cuda" else "int8")
    result = model.transcribe(audio_path, language=lang)
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    segments = result_aligned["segments"]
    if lang == "zh":
        text = "".join(seg["text"].strip() for seg in segments)
    else:
        text = " ".join(seg["text"].strip() for seg in segments)
    return text


def transcribe_with_whisperx(audio_path, lang="en", device_id=None):
    '''根据ref_audio生成对应的ref_text，从而在后续使用f5模型时，提供对齐文本，更好的提高最后audio的效果。
    device_id: 指定 GPU 编号时，在子进程中运行 whisperx（CUDA_VISIBLE_DEVICES 在子进程生效）；None 表示当前进程默认 GPU。
    '''
    if device_id is None:
        return _transcribe_with_whisperx_impl(audio_path, lang)

    # 主进程已指定 GPU/CUDA，改 env 无效；在子进程中设置 CUDA_VISIBLE_DEVICES 后执行
    import sys
    import tempfile
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        out_path = f.name
    try:
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from pathlib import Path; "
                "from dataflow_agent.toolkits.p2vtool.p2v_tool import _transcribe_with_whisperx_impl; "
                "text = _transcribe_with_whisperx_impl(sys.argv[1], sys.argv[2]); "
                "Path(sys.argv[3]).write_text(text, encoding='utf-8')"
            ),
            audio_path,
            lang,
            out_path,
        ]
        log.info(f"transcribe_with_whisperx 在子进程运行，device_id={device_id}")
        subprocess.run(cmd, env=env, check=True, timeout=300)
        return Path(out_path).read_text(encoding="utf-8")
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

def _run_f5_in_subprocess(text_prompt: str, save_path: str, ref_audio_path: str, ref_text: str, gpu_id: int) -> None:
    """
    在未初始化 CUDA 的子进程中运行 F5-TTS，以便 CUDA_VISIBLE_DEVICES 生效。
    主进程可能已固定 GPU，直接改 os.environ 无效。
    """
    import sys
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from dataflow_agent.toolkits.p2vtool.p2v_tool import inference_f5; "
            "inference_f5(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])"
        ),
        text_prompt,
        save_path,
        ref_audio_path,
        ref_text,
    ]
    subprocess.run(cmd, env=env, check=True, timeout=300)


def inference_f5(text_prompt, save_path, ref_audio, ref_text):
    '''使用 F5-TTS 模型做语音生成/克隆。通过一段参考音频及其对应的文本，克隆其音色并生成目标文本的语音'''
    from f5_tts.api import F5TTS
    import torch
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        
        # 即使 weights_only=True，这两个类也会被放行
        with torch.serialization.safe_globals([ListConfig, DictConfig]):
            f5tts = F5TTS()
    except ImportError:
        # 如果没有 omegaconf 库，回退到普通实例化
        f5tts = F5TTS()
    f5tts.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text_prompt, file_wave=save_path, seed=None,)

def merge_wav_files(file_list, output_path):
    '''将多个wav文件合并为一个wav文件，实际上是将一张ppt中的多个sentence wav合并为一个wav'''
    from pydub import AudioSegment
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    combined.export(output_path, format="wav")

def speech_task_wrapper_with_f5(task_args):
    """
    单个句子的语音生成任务, 使用 F5-TTS 模型
    """
    (slide_idx, idx, prompt, ref_audio_path, ref_text, speech_result_path, gpu_id) = task_args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 调用 F5-TTS 推理
    inference_f5(prompt, str(speech_result_path), ref_audio_path, ref_text)
    
    # 3. 获取时长
    duration = get_audio_length(str(speech_result_path))
    return slide_idx, idx, duration, str(speech_result_path)

def speech_task_wrapper_with_cloud_tts(task_args):
    """
    单个句子的语音生成任务，优先使用云 TTS（CosyVoice）；若 5 次内均失败则回退到 F5-TTS（需提供 ref_audio_path、ref_text、gpu_list）。
    task_args 可为 8 元组或 10 元组：
    - 8 元组: (slide_idx, idx, prompt, speech_result_path, api_key, tts_model, chat_api_url, tts_voice_name)
    - 10 元组: 上述 8 项 + (gpu_list, speech_language)，用于云 TTS 失败后 F5 回退
    """
    from dataflow_agent.toolkits.multimodaltool.req_tts import (
        generate_speech_and_save_async,
        TTSFallbackToF5Error,
    )
    import asyncio

    if len(task_args) >= 10:
        (slide_idx, idx, prompt, speech_result_path, api_key, tts_model, chat_api_url,
         tts_voice_name, gpu_list, speech_language) = task_args[:10]
        can_fallback_f5 = bool(gpu_list)
    elif len(task_args) >= 8:
        (slide_idx, idx, prompt, speech_result_path, api_key, tts_model, chat_api_url,
         tts_voice_name) = task_args[:8]
        ref_audio_path = ref_text = gpu_list = None
        can_fallback_f5 = False
    else:
        (slide_idx, idx, prompt, speech_result_path, api_key, tts_model, chat_api_url) = task_args[:7]
        tts_voice_name = ""
        ref_audio_path = ref_text = gpu_list = None
        can_fallback_f5 = False

    voice_name = (tts_voice_name or "").strip()

    async def _run_cloud_tts():
        return await generate_speech_and_save_async(
            prompt,
            str(speech_result_path),
            api_url=chat_api_url,
            api_key=api_key,
            model=tts_model,
            voice_name=voice_name,
            max_attempts=5,
        )

    try:
        speech_result_path = asyncio.run(_run_cloud_tts())
    except TTSFallbackToF5Error:
        if not can_fallback_f5:
            raise
        log.warning(f"云 TTS 5 次均失败，回退 F5-TTS: slide_idx={slide_idx}, idx={idx}")
        # 从 speech_result_path 的父目录中任选一个 .wav 作为 ref_audio
        parent_dir = Path(speech_result_path).resolve().parent
        current_name = Path(speech_result_path).name
        wav_files = [p for p in parent_dir.glob("*.wav") if p.name != current_name]
        if not wav_files:
            log.error(f"回退 F5 时父目录下无其他 .wav 可作 ref_audio: {parent_dir}")
            raise
        ref_audio_path = str(wav_files[0])
        ref_text = transcribe_with_whisperx(ref_audio_path, lang=speech_language)
        gpu_id = gpu_list[(slide_idx * 100 + idx) % len(gpu_list)]
        _run_f5_in_subprocess(
            prompt,
            str(speech_result_path),
            ref_audio_path,
            ref_text,
            gpu_id,
        )
        speech_result_path = str(speech_result_path)

    duration = get_audio_length(speech_result_path)
    return slide_idx, idx, duration, speech_result_path


'''========================== 使用beamer生成ppt的函数  =================================='''
def extract_beamer_code(text_str):
    match = re.search(r"(\\documentclass(?:\[[^\]]*\])?\{beamer\}.*?\\end\{document\})", text_str, re.DOTALL)
    return match.group(1) if match else None

def compile_tex(beamer_code_path: str):
    tex_path = Path(beamer_code_path).resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"Tex file {tex_path} does not exist.")
    work_dir = tex_path.parent
    try:
        # 会编译.tex文件，然后创建好一个.pdf文件
        result = subprocess.run(
            ["tectonic", str(tex_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        code_debug_result = "\n".join([result.stdout, result.stderr])
        log.info(f"Beamer 编译成功，输出结果：{code_debug_result}")
        is_beamer_warning = False
        if 'warning' in code_debug_result:
            is_beamer_warning = True
            log.info(f"Beamer 代码存在warning，需要更加完善一下")
        is_beamer_wrong = False
        return is_beamer_wrong, is_beamer_warning, code_debug_result
    except subprocess.CalledProcessError as e:
        log.info(f"Beamer 编译失败: {e.stderr}")
        is_beamer_wrong = True
        is_beamer_warning = True
        code_debug_result = e.stderr
        return is_beamer_wrong, is_beamer_warning, code_debug_result


def is_overfull_warning(code_debug_result: str) -> bool:
    """是否包含 Overfull 类 warning（内容过高/过宽），需要尝试修复。"""
    if not code_debug_result:
        return False
    return "Overfull" in code_debug_result


def is_ignorable_warning_only(code_debug_result: str) -> bool:
    """是否仅包含可忽略的 warning（如访问绝对路径），无需修复。"""
    if not code_debug_result:
        return True
    lower = code_debug_result.lower()
    if "warning" not in lower:
        return True
    # 若只有 absolute path 类提示，视为可忽略
    if "overfull" in lower:
        return False
    if "absolute path" in lower or "accessing absolute path" in lower:
        return True
    return False


def is_table_asset(asset_ref: Any) -> bool:
    """asset_ref 为 Table 时通常为 'Table_1' / 'Table 2' 等形式，无实际文件路径。"""
    if not asset_ref:
        return False
    return str(asset_ref).strip().lower().startswith("table")


def ensure_minueru_output(state: Any) -> None:
    """若 state 无 minueru_output，尝试从 mineru_root 下首个 .md 读取（供 table_extractor 使用）。"""
    if getattr(state, "minueru_output", "") and str(state.minueru_output).strip():
        return
    mineru_root = getattr(state, "mineru_root", "") or ""
    if not mineru_root:
        return
    root = Path(mineru_root).expanduser().resolve()
    if not root.is_dir():
        return
    md_files = list(root.glob("*.md"))
    if not md_files:
        return
    target = md_files[0]
    if len(md_files) > 1:
        for f in md_files:
            if f.stat().st_size > target.stat().st_size:
                target = f
    try:
        state.minueru_output = target.read_text(encoding="utf-8")[:30000]
    except Exception as e:
        log.warning("从 mineru_root 读取 md 失败: %s", e)


def merge_pdfs(pdf_paths: List[str], output_path: Union[str, Path]) -> str:
    """将多份 PDF 按顺序合并为一份。要求 pdf_paths 中路径存在且为 PDF。"""
    if not pdf_paths:
        raise ValueError("merge_pdfs: pdf_paths 不能为空")
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("merge_pdfs 需要 PyMuPDF (pip install pymupdf)")
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = fitz.open()
    for p in pdf_paths:
        path = Path(p).expanduser().resolve()
        if not path.is_file():
            log.warning("merge_pdfs: 跳过不存在的文件 %s", path)
            continue
        with fitz.open(path) as src:
            merged.insert_pdf(src)
    merged.save(out)
    merged.close()
    log.info("merge_pdfs: 已合并 %s 个 PDF -> %s", len(pdf_paths), out)
    return str(out)


def beamer_code_validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """检查tex是否是正确的"""
    from tempfile import TemporaryDirectory

    # 这里的 dir 具体是什么无所谓，因为我latex code中的图像路径是绝对路径
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        # 在临时目录中创建 .tex 文件
        # todo: 这里可能需要修改一下，因为在临时目录下创建文件还是不太行。
        tex_path = temp_dir / "input.tex" 
        
        raw_beamer_code = parsed_result.get("latex_code", "")
        if not raw_beamer_code:
            log.error(f"The content of beamer code is empty!")
            return False, "The content of beamer code is empty!"
        beamer_code = extract_beamer_code(raw_beamer_code)
        try:
            # 1. 写入内容
            tex_path.write_text(beamer_code, encoding='utf-8')

            result = subprocess.run(
                ["tectonic", str(tex_path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            log.info(f"Beamer代码修改完成，没有出现error")
            code_debug_result = "\n".join([result.stdout, result.stderr])
            return True, None
            
        except subprocess.CalledProcessError as e:
            code_debug_result = f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            return False, code_debug_result


def parser_beamer_latex(code: str):
    # 1. 提取 Head: 从 \documentclass 到 \begin{document} 之间的内容
    head_pattern = r'\\documentclass(?:\[[^\]]*\])?\{beamer\}(.*?)\\begin\{document\}'
    head_match = re.search(head_pattern, code, flags=re.DOTALL)
    head_content = head_match.group(1).strip() if head_match else "未找到导言区"

    # 2. 提取所有 Frame (Slides)
    # 逻辑：匹配 \begin{frame} 和 \end{frame} 之间的所有内容
    # 注意：beamer 的 frame 可能带有参数，如 \begin{frame}{标题} 或 \begin{frame}[fragile]
    frame_pattern = r'\\begin\{frame\}.*?(.*?)\\end\{frame\}'
    frames = re.findall(frame_pattern, code, flags=re.DOTALL)
    
    frames_cleaned = [f.strip() for f in frames]

    return head_content, frames_cleaned

def resize_latex_image(code: Union[str, List[str]]):
    # 改进正则：
    # 1. 允许 width= 后面有空格
    # 2. 捕获数值后的单位（如 \textwidth, \linewidth, \columnwidth）
    pattern = r'(\\includegraphics\[[^\]]*width\s*=\s*)([\d.]+)\s*(\\[a-z]+|cm|mm|pt|in)?'
    
    def shrink_width_logic(match):
        prefix = match.group(1)
        current_val = float(match.group(2))
        unit = match.group(3) if match.group(3) else "" # 捕获单位
        
        new_val = max(0.1, current_val - 0.2)
        return f"{prefix}{new_val:.1f}{unit}"
    
    if isinstance(code, str):
        return re.sub(pattern, shrink_width_logic, code)

    if isinstance(code, list):
        new_code = code.copy()
        for i, line in enumerate(new_code):
            if isinstance(line, str) and "includegraphics" in line:
                new_code[i] = re.sub(pattern, shrink_width_logic, line)
        return new_code
    raise TypeError(f"Unsupported code type: {type(code)}")