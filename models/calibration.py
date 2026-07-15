"""Heston calibration: two-stage global (Latin Hypercube) + local (Levenberg-Marquardt)
optimisation, Feller-condition check, and a local day-over-day snapshot log used for
the calibration-stability check (see sections/calibration.py).

Limitation, stated explicitly: free data sources (yfinance) only give a live snapshot,
not historical option chains. "Re-calibrate on two different days" therefore means the
app builds its own history one run at a time -- each time you use it on a given
ticker/expiry, today's fit is appended to a local log. Comparison is only possible once
the app has been run on 2+ distinct days for that ticker/expiry.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from models.heston import heston_price

HESTON_BOUNDS = [(0.005, 0.20), (0.005, 0.20), (0.10, 6.00), (0.05, 1.50), (-0.95, 0.50)]
HESTON_NAMES = ["v0", "theta", "kappa", "xi", "rho"]

_HISTORY_DIR = Path(__file__).resolve().parent.parent / ".calibration_history"


def heston_loss(params, market_rows, S0, r, q):
    """Sum of spread-weighted squared price residuals -- the calibration objective.
    Exposed standalone (not just as a closure inside calibrate_heston) so it can also
    drive the kappa-xi identifiability valley plot on sections/calibration.py, the same
    way the course notebook reuses its loss() for both calibration and diagnostics."""
    v0, theta, kappa, xi, rho = params
    total = 0.0
    for K, tau, price_mkt, spread, otype in market_rows:
        w = 1.0 / max(spread, 0.01)
        try:
            pm = heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, otype)
            total += (w * (pm - price_mkt)) ** 2
        except Exception:
            total += 1e6
    return total


def build_calibration_rows(chain_df, S0, tau):
    """Select the liquid, near-the-money quotes calibrate_heston should fit for one
    expiry: moneyness 0.80-1.20 with mid > 0.05, relaxed if that leaves too few rows.
    Returns (K, tau, price_mkt, spread, option_type) tuples."""
    moneyness = chain_df["strike"] / S0
    m = chain_df[(moneyness >= 0.80) & (moneyness <= 1.20) & (chain_df["mid"] > 0.05)]
    if len(m) < 5:
        m = chain_df[chain_df["mid"] > 0.02]  # relax if there are too few quotes
    return [
        (row["strike"], tau, row["mid"], max(row["spread"], 0.01), row["type"])
        for _, row in m.iterrows()
    ]


def calibrate_heston(market_rows, S0, r, q, n_candidatos=30, n_refinar=10, seed=1):
    """market_rows: list of tuples (K, tau, price_mkt, spread, option_type).

    Stage 1: Latin Hypercube scan of the bounded parameter space (cheap, global).
    Stage 2: bounded least-squares (Levenberg-Marquardt / TRF) refinement from the
    best few candidates, keeping the lowest-cost result.
    """
    lb = np.array([b[0] for b in HESTON_BOUNDS])
    ub = np.array([b[1] for b in HESTON_BOUNDS])

    def residuals(params):
        v0, theta, kappa, xi, rho = params
        res = []
        for K, tau, price_mkt, spread, otype in market_rows:
            w = 1.0 / max(spread, 0.01)
            try:
                pm = heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, otype)
                res.append(w * (pm - price_mkt))
            except Exception:
                res.append(1e3)
        return res

    muestra = qmc.LatinHypercube(d=len(HESTON_BOUNDS), seed=seed).random(n_candidatos)
    candidatos = lb + muestra * (ub - lb)
    perdidas = [heston_loss(c, market_rows, S0, r, q) for c in candidatos]
    mejores_idx = np.argsort(perdidas)[:n_refinar]

    mejor_fit = None
    for idx in mejores_idx:
        fit = least_squares(residuals, candidatos[idx], bounds=(lb, ub), max_nfev=80)
        if mejor_fit is None or fit.cost < mejor_fit.cost:
            mejor_fit = fit
    return mejor_fit.x, mejor_fit


def feller_condition(params):
    v0, theta, kappa, xi, rho = params
    lhs = 2 * kappa * theta
    rhs = xi**2
    return lhs >= rhs, lhs, rhs


def log_calibration_snapshot(ticker, expiry, S0, params, fit_cost):
    """Append today's calibrated params to a local per-ticker/expiry CSV history."""
    _HISTORY_DIR.mkdir(exist_ok=True)
    path = _HISTORY_DIR / f"{ticker}_{expiry}.csv"
    row = {"date": date.today().isoformat(), "S0": S0, "fit_cost": fit_cost}
    row.update({name: val for name, val in zip(HESTON_NAMES, params)})
    df_new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["date"] != row["date"]]  # keep one snapshot per day
        df_out = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.sort_values("date").to_csv(path, index=False)
    return path


def load_calibration_history(ticker, expiry):
    path = _HISTORY_DIR / f"{ticker}_{expiry}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "S0", "fit_cost"] + HESTON_NAMES)
    return pd.read_csv(path).sort_values("date").reset_index(drop=True)
