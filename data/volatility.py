"""Pure calculations for Panel 3 (Volatility Smile / Surface).

Everything here takes already-fetched data as arguments -- no network calls -- so it
stays testable and reusable between the smile page and the surface page.
"""

import numpy as np
import pandas as pd

from models.black_scholes import bs_implied_vol
from models.heston import heston_price


def compute_market_smile(chain_df, S0, r, q, tau, option_type):
    """Market-observed implied vol per strike for one expiry. Falls back to inverting
    the mid price via B&S when yfinance's own impliedVolatility is missing/invalid."""
    sub = chain_df[chain_df["type"] == option_type].sort_values("strike").copy()
    ivs = []
    for _, row in sub.iterrows():
        iv = row["impliedVolatility"]
        if pd.isna(iv) or iv <= 0:
            iv = bs_implied_vol(row["mid"], S0, row["strike"], r, q, tau, option_type)
        ivs.append(iv)
    sub["iv_market"] = ivs
    sub["moneyness"] = sub["strike"] / S0
    return sub[["strike", "moneyness", "iv_market"]].dropna()


def compute_model_smile_heston(strikes, S0, r, q, tau, heston_params, option_type):
    """The 'Heston-implied smile': price each strike with the calibrated Heston model,
    then invert that price back to a B&S implied vol so it plots on the same axis as
    the market smile."""
    v0, theta, kappa, xi, rho = heston_params
    rows = []
    for K in strikes:
        price = heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, option_type)
        iv = bs_implied_vol(price, S0, K, r, q, tau, option_type)
        rows.append({"strike": K, "moneyness": K / S0, "iv_heston": iv})
    return pd.DataFrame(rows)


def build_surface_grid(surface_entries, n_moneyness=25, m_range=(0.8, 1.2)):
    """surface_entries: list of {"days": float, "moneyness": array-like, "iv": array-like}.

    Interpolates each expiry's IV curve onto a common moneyness grid and returns
    (moneyness_grid, days_grid, iv_grid) as 2D arrays ready for a Plotly Surface.
    Returns None if fewer than 2 usable expiries are supplied.
    """
    m_grid = np.linspace(*m_range, n_moneyness)
    days_list, iv_rows = [], []
    for entry in sorted(surface_entries, key=lambda e: e["days"]):
        m = np.asarray(entry["moneyness"], dtype=float)
        iv = np.asarray(entry["iv"], dtype=float)
        order = np.argsort(m)
        m, iv = m[order], iv[order]
        if len(m) < 2:
            continue
        iv_interp = np.interp(m_grid, m, iv, left=np.nan, right=np.nan)
        days_list.append(entry["days"])
        iv_rows.append(iv_interp)
    if len(iv_rows) < 2:
        return None
    iv_grid = np.array(iv_rows)  # shape (n_expiries, n_moneyness)
    days_grid, m_grid_2d = np.meshgrid(days_list, m_grid, indexing="ij")
    return m_grid_2d, days_grid, iv_grid
