"""Panel 2 chart: 2x3 grid of Greeks vs strike or vs spot, B&S overlaid on Heston."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styling import ACCENT, ACCENT_2, TEXT_DIM, apply_dark_layout

GREEK_PLOT_SPECS = [
    ("delta", "Delta (Δ)"), ("gamma", "Gamma (Γ)"), ("vega", "Vega (ν)"),
    ("theta", "Theta (Θ, por día)"), ("rho", "Rho (ρ)"), ("vanna", "Vanna"),
]
_POSITIONS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]


def make_greeks_grid_figure(df_plot, x_col, x_label, x_ref):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[t for _, t in GREEK_PLOT_SPECS])
    for (key, _title), (r_, c_) in zip(GREEK_PLOT_SPECS, _POSITIONS):
        fig.add_trace(
            go.Scatter(x=df_plot[x_col], y=df_plot[f"bs_{key}"], name="B&S",
                       line=dict(color=ACCENT, width=2, dash="dot"),
                       legendgroup="bs", showlegend=(r_ == 1 and c_ == 1)),
            row=r_, col=c_,
        )
        fig.add_trace(
            go.Scatter(x=df_plot[x_col], y=df_plot[f"h_{key}"], name="Heston",
                       line=dict(color=ACCENT_2, width=2.5),
                       legendgroup="heston", showlegend=(r_ == 1 and c_ == 1)),
            row=r_, col=c_,
        )
        fig.add_vline(x=x_ref, line_dash="dash", line_color=TEXT_DIM, row=r_, col=c_)

    fig.update_xaxes(title_text=x_label)
    return apply_dark_layout(fig, height=620, legend_top=False).update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1)
    )
