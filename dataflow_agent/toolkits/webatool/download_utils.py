from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import httpx
from playwright.async_api import Error as PlaywrightError, Page

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


_COMMON_FILE_EXTENSIONS = [
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".bz2",
    ".xz",
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".xml",
    ".tsv",
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".mp4",
    ".avi",
    ".mov",
    ".mp3",
    ".wav",
    ".exe",
    ".msi",
    ".dmg",
    ".deb",
    ".rpm",
    ".parquet",
    ".arrow",
    ".h5",
    ".hdf5",
    ".pkl",
]

_DOWNLOADABLE_TYPES = [
    "application/octet-stream",
    "application/zip",
    "application/x-zip-compressed",
    "application/x-rar-compressed",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "text/csv",
    "application/json",
    "application/xml",
    "image/",
    "video/",
    "audio/",
]


async def check_if_download_link(url: str) -> Dict[str, Any]:
    """
    判断 URL 是否指向可下载文件。
    """
    result: Dict[str, Any] = {
        "is_download": False,
        "reason": "",
        "content_type": "",
        "filename": "",
    }

    url_lower = url.lower()
    for ext in _COMMON_FILE_EXTENSIONS:
        if url_lower.endswith(ext) or f"{ext}?" in url_lower or f"{ext}#" in url_lower:
            result["is_download"] = True
            result["reason"] = f"URL包含文件扩展名: {ext}"
            result["filename"] = url.split("/")[-1].split("?")[0].split("#")[0]
            return result

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.head(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    )
                },
            )
    except Exception as exc:  # pragma: no cover - 网络异常
        result["reason"] = f"HEAD 请求失败: {exc}，无法确定"
        return result

    content_disposition = response.headers.get("Content-Disposition", "")
    if content_disposition and "attachment" in content_disposition.lower():
        result["is_download"] = True
        result["reason"] = "Content-Disposition 包含 attachment"
        if "filename=" in content_disposition:
            import re

            filename_match = re.search(
                r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition
            )
            if filename_match:
                result["filename"] = filename_match.group(1).strip('\'"')
        return result

    content_type = response.headers.get("Content-Type", "").lower()
    result["content_type"] = content_type

    for dtype in _DOWNLOADABLE_TYPES:
        if dtype in content_type:
            result["is_download"] = True
            result["reason"] = f"Content-Type 是可下载类型: {content_type}"
            return result

    if "text/html" in content_type:
        result["is_download"] = False
        result["reason"] = "Content-Type 是 HTML 页面，非文件下载"
        return result

    result["reason"] = "无法确定是否为下载链接"
    return result


async def download_file(page: Page, url: str, save_dir: str) -> Optional[str]:
    """
    使用 Playwright 下载文件，返回保存路径。
    """
    log.info(f"[Playwright] 准备从 {url} 下载文件")
    os.makedirs(save_dir, exist_ok=True)

    download_page = await page.context.new_page()
    try:
        async with download_page.expect_download(timeout=12000) as download_info:
            try:
                await download_page.goto(url, timeout=60000)
            except PlaywrightError as exc:
                if "Download is starting" in str(exc) or "navigation" in str(exc):
                    log.info("下载已通过导航或重定向触发。")
                else:
                    raise exc
        download = await download_info.value
        try:
            await download_page.close()
        except Exception as close_exc:  # pragma: no cover - 尽力关闭
            log.info(f"关闭下载页面时出错（可忽略）: {close_exc}")
        suggested_filename = download.suggested_filename
        save_path = os.path.join(save_dir, suggested_filename)
        log.info(f"文件 '{suggested_filename}' 正在保存中...")
        temp_file_path = await download.path()
        if not temp_file_path:
            log.info("[Playwright] 下载失败，未能获取临时文件路径。")
            await download.delete()
            return None
        shutil.move(temp_file_path, save_path)
        log.info(f"[Playwright] 下载完成: {save_path}")
        return save_path
    except Exception as exc:  # pragma: no cover - 下载异常
        log.info(f"[Playwright] 下载过程中发生意外错误 ({url}): {exc}")
        try:
            await download_page.close()
        except Exception:
            pass
        return None

