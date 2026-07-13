"""Shared theme: color constants (used by both the CSS and the Plotly figures, so
charts never drift out of sync with the page chrome) and the CSS injector."""

from pathlib import Path

import streamlit as st

NAVY = "#0B1220"
PANEL = "#121A2B"
PANEL_BORDER = "#223049"
PANEL_BORDER_BRIGHT = "#2E4066"
ACCENT = "#3B82C4"
ACCENT_2 = "#1FAE85"
WARN = "#D9822B"
DANGER = "#D9534F"
TEXT = "#EAF2FB"
TEXT_DIM = "#8492A6"

PLOTLY_TEMPLATE = "plotly_dark"

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def inject_css():
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def apply_dark_layout(fig, height=380, legend_top=True):
    """Common Plotly layout so every chart in the app looks like one system."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend_top else {},
        font=dict(family="JetBrains Mono, monospace", color=TEXT_DIM, size=12),
    )
    return fig
