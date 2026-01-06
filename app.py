from __future__ import annotations

import os

from gradio_app.app import PAGE_SETS, create_app


def main() -> None:
    # HF Spaces uses PORT; also keep compatibility with GRADIO_SERVER_PORT.
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    # Ensure Space only loads the intended pages.
    os.environ.setdefault("DF_PAGE_SET", "space")

    page_names = PAGE_SETS["space"]
    app = create_app(page_names)
    app.queue()
    app.launch(server_name=server_name, server_port=port, share=False)


if __name__ == "__main__":
    main()

