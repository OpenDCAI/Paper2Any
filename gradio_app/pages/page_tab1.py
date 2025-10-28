import gradio as gr

def create_page_tab1() -> gr.Blocks:
    with gr.Blocks() as tab1:
        gr.Markdown("## Tab 1: Mock Content")
        gr.Textbox(label="Input 1", placeholder="Enter something...")
        gr.Button("Submit")
    return tab1