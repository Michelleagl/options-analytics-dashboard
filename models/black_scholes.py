"""Black-Scholes-Merton closed-form pricing and implied volatility."""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, r, q, tau, sigma, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == "call":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def bs_implied_vol(price, S, K, r, q, tau, option_type="call", tol=1e-7):
    """Bisection search for implied vol. Returns NaN if price is below intrinsic or tau<=0."""
    intrinsic = (
        max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
        if option_type == "call"
        else max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0)
    )
    if price <= intrinsic + 1e-8 or tau <= 0:
        return np.nan
    lo, hi = 1e-6, 5.0
    mid = 0.2
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_price(S, K, r, q, tau, mid, option_type) - price
        if abs(d) < tol:
            return mid
        lo, hi = (mid, hi) if d < 0 else (lo, mid)
    return mid
