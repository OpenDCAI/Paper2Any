# gradio_app/app.py
from __future__ import annotations

import os
import argparse
import socket
import importlib
from pathlib import Path

import gradio as gr

# -----------------------------------------------------------
# 动态加载 pages/ 目录下所有页面
# -----------------------------------------------------------
def load_pages() -> dict[str, gr.Blocks]:
    pages: dict[str, gr.Blocks] = {}
    pages_dir = Path(__file__).parent / "pages"

    for py_file in pages_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"gradio_app.pages.{py_file.stem}"
        module = importlib.import_module(module_name)

        # 约定：每个页面提供 create_<module_name>() 函数
        fn_name = f"create_{py_file.stem}"
        if hasattr(module, fn_name):
            pages[py_file.stem] = getattr(module, fn_name)()

    return pages


# -----------------------------------------------------------
# 构建主 UI
# -----------------------------------------------------------
with gr.Blocks(title="DataFlow Agent Platform") as app:
    gr.Markdown("# 🌊 DataFlow Agent 多功能平台")

    pages = load_pages()

    with gr.Tabs():
        for page_name, page_content in pages.items():
            with gr.Tab(page_name.replace("_", " ").title()):
                page_content.render()


# -----------------------------------------------------------
# 启动入口
# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio App")
    parser.add_argument(
        "--server_port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        help="优先命令行，其次环境变量 GRADIO_SERVER_PORT，默认 7860",
    )
    return parser.parse_args()


def is_port_free(port: int) -> bool:
    """简单探测端口是否空闲"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) != 0


if __name__ == "__main__":
    args = parse_args()
    port = args.server_port
    if not is_port_free(port):
        print(f"⚠️  端口 {port} 已被占用，自动切换到随机空闲端口。"
              " 如需固定端口，请换一个数字或先释放该端口。")
        port = 0  # 让 Gradio 自动选

    app.queue() 
    app.launch(server_name="0.0.0.0", server_port=port)