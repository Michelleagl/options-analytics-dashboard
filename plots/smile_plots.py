"""Panel 3 centerpiece: market IV smile vs the flat B&S line vs the Heston-implied curve."""

import numpy as np
import plotly.graph_objects as go

from utils.styling import ACCENT, ACCENT_2, TEXT, TEXT_DIM, apply_dark_layout


def make_smile_figure(market_df, sigma_atm, heston_df, S0, x_axis="log_moneyness"):
    """market_df: columns [strike, moneyness, iv_market]. heston_df: columns
    [strike, moneyness, iv_heston]. x_axis: 'log_moneyness' or 'strike'.

    Log-moneyness ln(K/S0) rather than raw K/S0 is the standard convention for smile
    plots (brief slide 15: "strike, or log-moneyness") -- equal steps in log-moneyness
    are equal percentage distances from spot in either direction, so the OTM put wing
    and OTM call wing sit symmetrically around ATM instead of being visually compressed
    on one side, the way linear moneyness compresses the put side.
    """
    if x_axis == "log_moneyness":
        x_mkt = np.log(market_df["moneyness"])
        x_hes = np.log(heston_df["moneyness"])
        ref_x = 0.0  # ln(1) = 0
        x_title = "Log-Moneyness ln(K/S₀)"
    else:
        x_mkt = market_df[x_axis]
        x_hes = heston_df[x_axis]
        ref_x = S0
        x_title = "Strike (K)"

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
    fig.add_vline(x=ref_x, line_dash="dash", line_color=TEXT_DIM, annotation_text="ATM",
                  annotation_font_color=TEXT_DIM)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Volatilidad implícita (%)",
    )
    return apply_dark_layout(fig, height=430)
