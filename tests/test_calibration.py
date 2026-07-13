"""Synthetic-recovery test, mirroring the course's own validation method (notebook
'Heston -- Calibracion de Mercado II', section 1): generate prices from known
parameters, calibrate from a different starting point, and check the fitted surface
reproduces the synthetic market. We deliberately do NOT assert exact recovery of every
parameter -- kappa/theta/xi are known to sit in a flat, poorly-identified valley (see
pages/4_⚙️_Calibration.py) even when the price fit is excellent. v0 and rho are the
two parameters the data actually pins down, so those get a tighter check.
"""

import numpy as np

from models.calibration import calibrate_heston, feller_condition
from models.heston import heston_call

S0, r, q, tau = 100.0, 0.03, 0.0, 0.5
TRUE_PARAMS = dict(v0=0.04, theta=0.05, kappa=1.5, xi=0.4, rho=-0.6)
STRIKES = [85.0, 95.0, 100.0, 105.0, 115.0]


def _synthetic_market_rows():
    rows = []
    for K in STRIKES:
        price = heston_call(S0, K, r, q, tau, **TRUE_PARAMS)
        rows.append((K, tau, price, 0.05, "call"))
    return rows


def test_calibration_recovers_price_surface():
    market_rows = _synthetic_market_rows()
    params, fit_obj = calibrate_heston(market_rows, S0, r, q, n_candidatos=12, n_refinar=2, seed=7)
    v0, theta, kappa, xi, rho = params

    repriced = [heston_call(S0, K, r, q, tau, v0, kappa, theta, xi, rho) for K in STRIKES]
    synthetic = [row[2] for row in market_rows]
    rmse = np.sqrt(np.mean((np.array(repriced) - np.array(synthetic)) ** 2))

    assert rmse < 0.05, f"calibrated surface should reprice the synthetic market closely, got RMSE={rmse:.4f}"


def test_calibration_recovers_v0_and_rho_reasonably():
    """v0 and rho are the well-identified parameters (they set the short-end level and
    the skew, both of which the data constrains tightly); kappa/xi/theta are not
    checked here on purpose."""
    market_rows = _synthetic_market_rows()
    params, fit_obj = calibrate_heston(market_rows, S0, r, q, n_candidatos=12, n_refinar=2, seed=7)
    v0, theta, kappa, xi, rho = params

    assert abs(v0 - TRUE_PARAMS["v0"]) < 0.02
    assert abs(rho - TRUE_PARAMS["rho"]) < 0.3
    assert rho < 0  # sign should be recovered regardless of magnitude precision


def test_feller_condition_flags_correctly():
    ok, lhs, rhs = feller_condition([0.04, 0.05, 1.5, 0.4, -0.6])  # 2*1.5*0.05=0.15 >= 0.4^2=0.16 -> violated
    assert ok is False
    assert lhs == 2 * 1.5 * 0.05
    assert rhs == 0.4**2

    ok2, _, _ = feller_condition([0.04, 0.05, 3.0, 0.2, -0.6])  # 2*3*0.05=0.3 >= 0.2^2=0.04 -> satisfied
    assert ok2 is True
