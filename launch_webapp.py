#!/usr/bin/env python
"""Launcher for the CSIDE Gradio web app.

Usage:
    uv run python launch_webapp.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `webapp` is importable
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from webapp.app import build_ui  # noqa: E402

if __name__ == "__main__":
    import gradio as gr
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
