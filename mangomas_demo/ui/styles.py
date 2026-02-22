"""
UI Styles — theme CSS and colour tokens for the Gradio app.
"""

THEME_CSS: str = """
.gradio-container {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a0a1a 100%) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.gr-button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    transition: all 0.3s ease !important;
}
.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
.gr-input, .gr-textbox textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}
.gr-panel {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
}
"""
