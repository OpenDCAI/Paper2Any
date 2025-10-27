import gradio as gr
from pathlib import Path
import importlib

def load_pages():
    """动态加载 pages/ 目录下所有页面"""
    pages = {}
    pages_dir = Path(__file__).parent / "pages"
    
    for py_file in pages_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        
        module_name = f"gradio_app.pages.{py_file.stem}"
        module = importlib.import_module(module_name)
        
        # 约定：每个页面提供 create_xxx_page() 函数
        if hasattr(module, f"create_{py_file.stem}"):
            page_func = getattr(module, f"create_{py_file.stem}")
            pages[py_file.stem] = page_func()
    
    return pages

# 主应用
with gr.Blocks(title="DataFlow Agent Platform") as app:
    gr.Markdown("# 🌊 DataFlow Agent 多功能平台")
    
    pages = load_pages()
    
    with gr.Tabs():
        for page_name, page_content in pages.items():
            with gr.Tab(page_name.replace("_", " ").title()):
                page_content.render()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)