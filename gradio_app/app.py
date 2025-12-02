# gradio_app/app.py
from __future__ import annotations
import os, argparse, socket, importlib, sys
from pathlib import Path
from typing import Optional, Set

import gradio as gr


# 标签显示名称映射,如果需要自定义页面名称
TAB_NAME_MAP = {
    "operator_write": "Operator Write",
    "PA_frontend": "PromptAgent Frontend",
}

# 页面集合定义：可以按需调整
# key: 启动参数/模式名称
# value: 该模式下要加载的页面文件名集合（不含 .py）
#       使用 None 表示“不过滤，加载全部页面”
PAGE_SETS: dict[str, Optional[Set[str]]] = {
    "all": None,  # 全量加载 pages 目录下的所有页面
    "data": {"operator_write", "PA_frontend","op_assemble_line","pipeline_rec","web_collection"},
    "paper": {"icongen_refine"}
}


def load_pages(allowed: Optional[Set[str]] = None) -> dict[str, gr.Blocks]:
    """自动扫描 pages 目录，根据 allowed 进行白名单过滤后加载页面。"""
    pages: dict[str, gr.Blocks] = {}
    pages_dir = Path(__file__).parent / "pages"

    # 确保项目根目录在 Python 路径中
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    for py_file in pages_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        name = py_file.stem
        # 如果有限定集合，且当前页面不在集合中，则跳过
        if allowed is not None and name not in allowed:
            continue

        try:
            # 尝试两种导入方式
            module = None
            module_name = f"gradio_app.pages.{name}"
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                # 如果从 gradio_app 目录运行，尝试相对导入
                try:
                    module = importlib.import_module(f"pages.{name}")
                except ModuleNotFoundError:
                    raise

            fn_name = f"create_{name}"
            if hasattr(module, fn_name):
                pages[name] = getattr(module, fn_name)()
        except Exception as e:
            print(f"⚠️  跳过页面 {py_file.name}: {e}")
            import traceback
            traceback.print_exc()
    return pages


def create_app(page_names: Optional[Set[str]] = None) -> gr.Blocks:
    """根据给定的 page_names（白名单）创建 Gradio Blocks 应用。"""
    pages = load_pages(page_names)

    with gr.Blocks(title="DataFlow Agent Platform", elem_id="app-root") as app:
        # PromptAgent 前端样式注入，不影响其他页面及整体逻辑
        gr.HTML(
            "<style>"
            ".left-pane-pa{max-height:70vh!important;overflow:auto!important;}"
            ".right-pane-pa{max-height:70vh!important;overflow:auto!important;}"
            ".chat-box-pa{max-height:70vh!important;overflow:auto!important;}"
            "</style>"
        )

        gr.Markdown("# 🌊 DataFlow Agent 多功能平台")
        with gr.Tabs():
            for name, page in pages.items():
                # 优先使用映射表中的名称，否则使用默认转换
                tab_name = TAB_NAME_MAP.get(name, name.replace("_", " ").title())
                with gr.Tab(tab_name):
                    page.render()
    return app


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
    parser.add_argument(
        "--page_set",
        type=str,
        default=os.getenv("DF_PAGE_SET", "all"),
        choices=list(PAGE_SETS.keys()),
        help=(
            "选择要加载的页面集合，例如: "
            + "/".join(PAGE_SETS.keys())
            + "；也可用环境变量 DF_PAGE_SET 覆盖"
        ),
    )
    return parser.parse_args()


def is_port_free(port: int) -> bool:
    """简单探测端口是否空闲"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) != 0


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port_env = os.getenv("GRADIO_SERVER_PORT")

    args = parse_args()

    # 1) 解析端口逻辑（保持你原来的）
    if server_port_env:
        # Docker 环境：使用环境变量指定的端口
        port = int(server_port_env)
        print(f"🐳 Docker 模式：使用环境变量端口 {port}")
    else:
        # 本地开发：使用命令行参数
        port = args.server_port
        if not is_port_free(port):
            print(
                f"⚠️  端口 {port} 已被占用，自动切换到随机空闲端口。"
                " 如需固定端口，请换一个数字或先释放该端口。"
            )
            port = 0  # 让 Gradio 自动选

    # 2) 根据 page_set 决定要加载哪些页面
    page_set_name = args.page_set
    page_names = PAGE_SETS.get(page_set_name)

    if page_names is None:
        print(f"🧩 页面模式: {page_set_name} (加载全部页面)")
    else:
        print(
            f"🧩 页面模式: {page_set_name} "
            f"(仅加载: {', '.join(sorted(page_names))})"
        )

    # 3) 创建 app 并启动应用
    app = create_app(page_names)
    app.queue()
    app.launch(
        server_name=server_name,
        server_port=port,
        share=False,
    )
