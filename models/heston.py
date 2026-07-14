"""Heston (1993) stochastic-volatility pricing via characteristic-function inversion.

Uses the "Little Trap" (g2) formulation of the characteristic function to avoid
the branch-cut discontinuities of the naive complex logarithm (Albrecher et al. 2007).

Priced via scipy's adaptive quadrature rather than a fixed quadrature grid: a fixed
grid (e.g. 128-node Gauss-Legendre over a fixed u_max) was tried and is faster, but
the characteristic function's decay rate in u depends on tau, so a single fixed grid
is inaccurate for short-dated contracts and can even return a negative price for a
deep-OTM short-dated option -- exactly the kind of contract this app is built to
examine (see the Volatility Smile and Live Defense sections). Adaptive quadrature is
slower per call but self-adjusts its refinement to whatever tau/moneyness it's given.
"""

import numpy as np
from scipy.integrate import quad


def heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho):
    i = 1j
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho * xi if j == 1 else kappa
    rd = r - q
    dj = np.sqrt((rho * xi * i * u - bj) ** 2 + xi**2 * (u**2 - 2 * uj * i * u))
    g2j = (bj - rho * xi * i * u + dj) / (bj - rho * xi * i * u - dj)
    Dj = ((bj - rho * xi * i * u + dj) / xi**2) * (
        (1 - np.exp(dj * tau)) / (1 - g2j * np.exp(dj * tau))
    )
    Cj = rd * i * u * tau + (kappa * theta / xi**2) * (
        (bj - rho * xi * i * u + dj) * tau - 2 * np.log((1 - g2j * np.exp(dj * tau)) / (1 - g2j))
    )
    return np.exp(Cj + Dj * v + i * u * x)


def heston_prob_j(j, x, v, tau, K, r, q, kappa, theta, xi, rho):
    lnK = np.log(K)
    ig = lambda u: np.real(
        np.exp(-1j * u * lnK) * heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho) / (1j * u)
    )
    val, _ = quad(ig, 1e-8, 200, limit=200)
    return 0.5 + val / np.pi


def heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    x = np.log(S)
    P1 = heston_prob_j(1, x, v0, tau, K, r, q, kappa, theta, xi, rho)
    P2 = heston_prob_j(2, x, v0, tau, K, r, q, kappa, theta, xi, rho)
    return S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2


def heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    c = heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    return c - S * np.exp(-q * tau) + K * np.exp(-r * tau)  # put-call parity


def heston_price(S, K, r, q, tau, v0, kappa, theta, xi, rho, option_type="call"):
    if option_type == "call":
        return heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    return heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho)
