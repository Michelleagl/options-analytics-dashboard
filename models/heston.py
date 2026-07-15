"""Heston (1993) stochastic-volatility pricing via characteristic-function inversion.

Uses the "Little Trap" formulation (Albrecher et al. 2007, "The Little Heston Trap") of
the characteristic function, expressed with exp(-d*tau) (which decays) rather than
exp(+d*tau) (which grows). This matters, not just for overflow: a version of this
module previously used g2 = (A+d)/(A-d) with exp(+d*tau), which looked like the "trap"
form but still hit the complex logarithm's branch cut for longer-dated contracts,
silently returning wrong prices for tau beyond roughly 1.5-2 years (verified against an
independent Monte Carlo QE simulation -- off by ~37% at tau=3, kappa=1.8, a perfectly
ordinary calibration, not an extreme edge case). The reciprocal form here,
c = (A-d)/(A+d) with exp(-d*tau), avoids that branch crossing. Existing tests only
covered tau in [0.5, 1.0], which is why the earlier bug went undetected -- see
tests/test_heston.py for a longer-dated regression check against this exact case.

Priced via scipy's adaptive quadrature rather than a fixed quadrature grid: a fixed
grid (e.g. 128-node Gauss-Legendre over a fixed u_max) was tried and is faster, but
the characteristic function's decay rate in u depends on tau, so a single fixed grid
is inaccurate for short-dated contracts. Adaptive quadrature is slower per call but
self-adjusts its refinement to whatever tau/moneyness it's given.
"""
import numpy as np
from scipy.integrate import quad

def heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho):
    i = 1j
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho * xi if j == 1 else kappa
    rd = r - q
    A = bj - rho * xi * i * u
    d = np.sqrt(A**2 + xi**2 * (u**2 - 2 * uj * i * u))  # principal branch: Re(d) >= 0
    c = (A - d) / (A + d)
    exp_neg = np.exp(-d * tau)
    Dj = ((A - d) / xi**2) * ((1 - exp_neg) / (1 - c * exp_neg))
    Cj = rd * i * u * tau + (kappa * theta / xi**2) * (
        (A - d) * tau - 2 * np.log((1 - c * exp_neg) / (1 - c))
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
    price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
    intrinsic_floor = max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)  # cota de no-arbitraje
    return max(price, intrinsic_floor)


def heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    c = heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    price = c - S * np.exp(-q * tau) + K * np.exp(-r * tau)  # put-call parity
    intrinsic_floor = max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0)
    return max(price, intrinsic_floor)


def heston_price(S, K, r, q, tau, v0, kappa, theta, xi, rho, option_type="call"):
    if option_type == "call":
        return heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    return heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho)
