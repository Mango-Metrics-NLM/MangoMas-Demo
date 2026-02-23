#!/usr/bin/env python3
"""
MangoMAS Demo — Gradio app entrypoint.

Usage:
    python app.py                          # default port 7860
    GRADIO_PORT=8080 python app.py         # custom port via env
"""

import os

from mangomas_demo.ui.builder import build_app

if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
