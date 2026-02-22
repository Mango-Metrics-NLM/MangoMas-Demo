#!/usr/bin/env python3
"""
MangoMAS Demo — Gradio app entrypoint.

Usage:
    python app.py                          # default port 7860
    python app.py --server-port 8080       # custom port
"""

from mangomas_demo.ui.builder import build_app

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
