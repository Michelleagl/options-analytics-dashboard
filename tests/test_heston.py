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
