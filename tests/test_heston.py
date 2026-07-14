import numpy as np

from models.heston import heston_call, heston_put, heston_price, heston_prob_j
from models.greeks import heston_greeks_fd

S0, K, r, q, tau = 100.0, 100.0, 0.05, 0.0, 1.0
v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.30, -0.7


def test_put_call_parity():
    call = heston_call(S0, K, r, q, tau, v0, kappa, theta, xi, rho)
    put = heston_put(S0, K, r, q, tau, v0, kappa, theta, xi, rho)
    lhs = call - put
    rhs = S0 * np.exp(-q * tau) - K * np.exp(-r * tau)
    assert abs(lhs - rhs) < 1e-6


def test_price_dispatches_by_option_type():
    assert heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, "call") == heston_call(
        S0, K, r, q, tau, v0, kappa, theta, xi, rho
    )
    assert heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, "put") == heston_put(
        S0, K, r, q, tau, v0, kappa, theta, xi, rho
    )


def test_delta_matches_p1_from_characteristic_function():
    """The course's own cross-check (notebook 2): Delta from bump-and-reprice on the
    call price should equal P1 computed directly from the Fourier integral, since
    Delta = P1 falls out of the characteristic function algebra for a call."""
    x = np.log(S0)
    P1 = heston_prob_j(1, x, v0, tau, K, r, q, kappa, theta, xi, rho)
    delta_fd = heston_greeks_fd(S0, K, r, q, tau, v0, kappa, theta, xi, rho, "call", full=False)["delta"]
    assert abs(delta_fd - P1) < 1e-3


def test_call_price_between_intrinsic_and_spot():
    price = heston_call(S0, K, r, q, tau, v0, kappa, theta, xi, rho)
    intrinsic = max(S0 * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
    assert intrinsic <= price <= S0 * np.exp(-q * tau)


def test_long_dated_matches_independent_monte_carlo():
    """Regression test for a real bug: an earlier version of heston_cf_j used the
    "Little Trap" with exp(+d*tau) (g2 = (A+d)/(A-d)), which still crossed the complex
    logarithm's branch cut for longer-dated contracts and silently returned a price
    ~37% too high here (26.27 instead of ~19.14) for a perfectly ordinary calibration
    (kappa=1.8, tau=3) -- not an extreme edge case. Existing tests only covered
    tau in [0.5, 1.0], which is why it went undetected. Ground truth below is from an
    independent Monte Carlo (QE scheme, Andersen 2008) simulation with 300k paths,
    300 steps, seed=42: 19.1391 +/- 0.0898 (95% CI)."""
    S, K, r, q, tau = 100.0, 100.0, 0.04, 0.01, 3.0
    v0_, kappa_, theta_, xi_, rho_ = 0.05, 1.8, 0.06, 0.55, -0.65
    price = heston_call(S, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_)
    assert abs(price - 19.1391) < 0.3, f"expected ~19.14 (Monte Carlo), got {price:.4f}"


def test_put_call_parity_long_dated():
    S, K, r, q, tau = 100.0, 100.0, 0.04, 0.01, 3.0
    v0_, kappa_, theta_, xi_, rho_ = 0.05, 1.8, 0.06, 0.55, -0.65
    call = heston_call(S, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_)
    put = heston_put(S, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_)
    lhs = call - put
    rhs = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    assert abs(lhs - rhs) < 1e-6


def test_intrinsic_floor_prevents_invalid_negative_price():
    """A deep-OTM, very short-dated call is a known hard case for the numerical
    integration (quad can occasionally return a value slightly off from the true
    integral for a fast-decaying, short-tau integrand) -- heston_call/heston_put clamp
    to the no-arbitrage intrinsic-value floor so the app can never show a negative or
    sub-intrinsic price, regardless of integration noise."""
    S, K, r, q, tau = 100.0, 105.0, 0.03, 0.0, 0.02
    v0_, kappa_, theta_, xi_, rho_ = 0.02, 6.0, 0.20, 1.5, -0.84
    price = heston_call(S, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_)
    intrinsic = max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
    assert price >= intrinsic - 1e-9
