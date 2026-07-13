"""Panel 1 chart: price vs strike for market / B&S / Heston."""

import plotly.graph_objects as go

from utils.styling import ACCENT, ACCENT_2, TEXT, TEXT_DIM, WARN, apply_dark_layout


def make_price_vs_strike_figure(strikes_sorted, market_curve, bs_curve, heston_curve, S0, strike, option_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes_sorted, y=market_curve, mode="markers", name="Mercado (mid)",
                              marker=dict(color=TEXT, size=7, symbol="circle")))
    fig.add_trace(go.Scatter(x=strikes_sorted, y=bs_curve, mode="lines", name="Black & Scholes",
                              line=dict(color=ACCENT, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=strikes_sorted, y=heston_curve, mode="lines", name="Heston (calibrado)",
                              line=dict(color=ACCENT_2, width=2.5)))
    fig.add_vline(x=S0, line_dash="dash", line_color=TEXT_DIM, annotation_text="Spot", annotation_font_color=TEXT_DIM)
    fig.add_vline(x=strike, line_color=WARN, annotation_text="Seleccionado", annotation_font_color=WARN)
    fig.update_layout(xaxis_title="Strike (K)", yaxis_title=f"Precio del {option_type}")
    return apply_dark_layout(fig, height=380)
