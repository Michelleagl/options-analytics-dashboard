"""Panel 3 centerpiece: market IV smile vs the flat B&S line vs the Heston-implied curve."""

import plotly.graph_objects as go

from utils.styling import ACCENT, ACCENT_2, TEXT, TEXT_DIM, apply_dark_layout


def make_smile_figure(market_df, sigma_atm, heston_df, S0, x_axis="moneyness"):
    """market_df: columns [strike, moneyness, iv_market]. heston_df: columns
    [strike, moneyness, iv_heston]. x_axis: 'moneyness' or 'strike'."""
    x_mkt = market_df[x_axis]
    x_hes = heston_df[x_axis]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_mkt, y=market_df["iv_market"] * 100, mode="markers",
                              name="Mercado (IV implícita)",
                              marker=dict(color=TEXT, size=8, symbol="circle")))
    fig.add_trace(go.Scatter(x=[x_mkt.min(), x_mkt.max()], y=[sigma_atm * 100, sigma_atm * 100],
                              mode="lines", name="B&S (vol plana ATM)",
                              line=dict(color=ACCENT, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x_hes, y=heston_df["iv_heston"] * 100, mode="lines",
                              name="Heston (calibrado)",
                              line=dict(color=ACCENT_2, width=2.5)))
    ref_x = 1.0 if x_axis == "moneyness" else S0
    fig.add_vline(x=ref_x, line_dash="dash", line_color=TEXT_DIM, annotation_text="ATM",
                  annotation_font_color=TEXT_DIM)
    fig.update_layout(
        xaxis_title="Moneyness (K/S₀)" if x_axis == "moneyness" else "Strike (K)",
        yaxis_title="Volatilidad implícita (%)",
    )
    return apply_dark_layout(fig, height=430)
