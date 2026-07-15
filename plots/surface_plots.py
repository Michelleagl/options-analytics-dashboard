"""Bonus Panel 3 chart: 3D implied-volatility surface across strikes and expiries."""

import numpy as np
import plotly.graph_objects as go

from utils.styling import TEXT_DIM


def make_surface_figure(moneyness_grid, days_grid, iv_grid, title="Superficie de volatilidad implícita (mercado)"):
    """moneyness_grid holds raw K/S0 (the interpolation in build_surface_grid happens in
    linear-moneyness space); displayed here in log-moneyness ln(K/S0), the standard
    convention, so the surface reads symmetrically around ATM in either direction."""
    log_moneyness_grid = np.log(moneyness_grid)
    fig = go.Figure(data=[go.Surface(
        x=log_moneyness_grid, y=days_grid, z=iv_grid * 100,
        colorscale="Tealrose", showscale=True,
        colorbar=dict(title="IV (%)", tickfont=dict(color=TEXT_DIM)),
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=14, color=TEXT_DIM)),
        scene=dict(
            xaxis_title="Log-Moneyness ln(K/S₀)",
            yaxis_title="Días al vencimiento",
            zaxis_title="IV (%)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        ),
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
