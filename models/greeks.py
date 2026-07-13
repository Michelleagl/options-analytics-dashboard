"""Greeks for both pricing engines.

B&S Greeks are closed-form. Heston Greeks are bump-and-reprice (finite differences)
on the same characteristic-function pricer, reparametrized to sigma0 = sqrt(v0) so
vega/vanna/volga come out in the same units as B&S. This asymmetry (closed-form vs
finite-difference) is intentional and documented -- see the Risk Manager's note in
the project brief about not mixing Greek-computation methods silently.

Convention: vega/rho/vanna are per 1 point (0.01) of vol/rate; theta is per calendar
day; volga is per (1 point)^2.
"""

import numpy as np
from scipy.stats import norm

from models.heston import heston_price

GREEK_LABELS = {
    "delta": "Δ Delta",
    "gamma": "Γ Gamma",
    "vega": "ν Vega (por 1pto vol)",
    "theta": "Θ Theta (por día)",
    "rho": "ρ Rho (por 1pto tasa)",
    "vanna": "Vanna (por 1pto vol)",
    "volga": "Volga (por 1pto² vol)",
}


def bs_greeks(S, K, r, q, tau, sigma, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan, vanna=np.nan, volga=np.nan)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    pdf = norm.pdf(d1)
    term1 = (S * sigma * np.exp(-q * tau) * pdf) / (2 * np.sqrt(tau))
    if option_type == "call":
        delta = np.exp(-q * tau) * norm.cdf(d1)
        dPdtau = term1 + r * K * np.exp(-r * tau) * norm.cdf(d2) - q * S * np.exp(-q * tau) * norm.cdf(d1)
        rho_raw = K * tau * np.exp(-r * tau) * norm.cdf(d2)
    else:
        delta = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        dPdtau = term1 - r * K * np.exp(-r * tau) * norm.cdf(-d2) + q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho_raw = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
    gamma = np.exp(-q * tau) * pdf / (S * sigma * np.sqrt(tau))
    vega_raw = S * np.exp(-q * tau) * pdf * np.sqrt(tau)
    vanna_raw = -np.exp(-q * tau) * pdf * d2 / sigma
    volga_raw = vega_raw * d1 * d2 / sigma
    theta_raw = -dPdtau  # dV/dt (calendario) = -dV/dtau
    return dict(
        delta=delta, gamma=gamma,
        vega=vega_raw * 0.01, theta=theta_raw / 365.0, rho=rho_raw * 0.01,
        vanna=vanna_raw * 0.01, volga=volga_raw * 0.0001,
    )


def heston_greeks_fd(S, K, r, q, tau, v0, kappa, theta_lr, xi, rho, option_type="call", full=True):
    sigma0 = np.sqrt(v0)
    dS = 0.005 * S
    dsig = 0.005
    dtau = min(1 / 365.0, max(tau * 0.05, 1e-4))
    dr = 0.0005

    def P(S_, sigma0_, tau_, r_):
        return heston_price(S_, K, r_, q, tau_, sigma0_**2, kappa, theta_lr, xi, rho, option_type)

    p0 = P(S, sigma0, tau, r)
    p_su, p_sd = P(S + dS, sigma0, tau, r), P(S - dS, sigma0, tau, r)
    p_vu, p_vd = P(S, sigma0 + dsig, tau, r), P(S, sigma0 - dsig, tau, r)
    p_tu, p_td = P(S, sigma0, tau + dtau, r), P(S, sigma0, max(tau - dtau, 1e-5), r)

    delta_raw = (p_su - p_sd) / (2 * dS)
    gamma_raw = (p_su - 2 * p0 + p_sd) / dS**2
    vega_raw = (p_vu - p_vd) / (2 * dsig)
    theta_raw = -(p_tu - p_td) / (2 * dtau)

    out = dict(delta=delta_raw, gamma=gamma_raw, vega=vega_raw * 0.01, theta=theta_raw / 365.0)

    if full:
        p_ru, p_rd = P(S, sigma0, tau, r + dr), P(S, sigma0, tau, r - dr)
        rho_raw = (p_ru - p_rd) / (2 * dr)
        p_pp = P(S + dS, sigma0 + dsig, tau, r)
        p_pm = P(S + dS, sigma0 - dsig, tau, r)
        p_mp = P(S - dS, sigma0 + dsig, tau, r)
        p_mm = P(S - dS, sigma0 - dsig, tau, r)
        vanna_raw = (p_pp - p_pm - p_mp + p_mm) / (4 * dS * dsig)
        volga_raw = (p_vu - 2 * p0 + p_vd) / dsig**2
        out["rho"] = rho_raw * 0.01
        out["vanna"] = vanna_raw * 0.01
        out["volga"] = volga_raw * 0.0001
    return out
