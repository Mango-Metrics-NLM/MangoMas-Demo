"""Gradio UI — public API re-exports."""

from mangomas_demo.ui.builder import build_app
from mangomas_demo.ui.styles import THEME_CSS

__all__ = ["build_app", "THEME_CSS"]
