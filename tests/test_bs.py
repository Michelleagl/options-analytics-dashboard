import numpy as np

from models.black_scholes import bs_price, bs_implied_vol
from models.greeks import bs_greeks

S0, K, r, q, tau, sigma = 100.0, 105.0, 0.04, 0.01, 0.75, 0.22


def test_put_call_parity():
    call = bs_price(S0, K, r, q, tau, sigma, "call")
    put = bs_price(S0, K, r, q, tau, sigma, "put")
    lhs = call - put
    rhs = S0 * np.exp(-q * tau) - K * np.exp(-r * tau)
    assert abs(lhs - rhs) < 1e-8


def test_implied_vol_round_trip():
    for option_type in ["call", "put"]:
        price = bs_price(S0, K, r, q, tau, sigma, option_type)
        iv = bs_implied_vol(price, S0, K, r, q, tau, option_type)
        assert abs(iv - sigma) < 1e-4


def test_implied_vol_nan_below_intrinsic():
    # a price below intrinsic value is not a valid option price -> NaN
    intrinsic = max(S0 - K * np.exp(-r * tau), 0.0)
    iv = bs_implied_vol(intrinsic - 1.0, S0, K, r, q, tau, "call")
    assert np.isnan(iv)


def test_delta_bounds():
    call_delta = bs_greeks(S0, K, r, q, tau, sigma, "call")["delta"]
    put_delta = bs_greeks(S0, K, r, q, tau, sigma, "put")["delta"]
    assert 0.0 <= call_delta <= 1.0
    assert -1.0 <= put_delta <= 0.0


def test_gamma_positive_and_shared_across_call_put():
    gamma_call = bs_greeks(S0, K, r, q, tau, sigma, "call")["gamma"]
    gamma_put = bs_greeks(S0, K, r, q, tau, sigma, "put")["gamma"]
    assert gamma_call > 0
    assert abs(gamma_call - gamma_put) < 1e-10  # Gamma is identical for call and put
