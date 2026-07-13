"""yfinance-backed market data fetch (spot, dividend yield, expiries, option chains)."""

import os
from pathlib import Path

try:
    # Some managed/AV-scanned Windows machines intercept TLS (e.g. antivirus HTTPS
    # scanning) with a root CA that's trusted by the OS but missing from Python's
    # bundled certifi store, which makes every HTTPS call fail certificate
    # verification. truststore makes the stdlib ssl module use the OS-native trust
    # store instead -- it does NOT disable verification, it verifies against the
    # same roots Windows itself already trusts. This covers requests/urllib3-based
    # calls (e.g. pandas_datareader -> FRED).
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# yfinance's HTTP client (curl_cffi) has its own, separate certificate store that
# truststore can't patch. If a machine-local combined CA bundle has been generated
# (see README: "Local SSL / antivirus interception"), point curl_cffi and the stdlib
# at it too. Purely opt-in: on machines without this file (e.g. Streamlit Cloud),
# nothing changes.
_local_ca_bundle = Path(__file__).resolve().parent.parent / ".certs" / "combined_ca_bundle.pem"
if _local_ca_bundle.exists():
    os.environ.setdefault("CURL_CA_BUNDLE", str(_local_ca_bundle))
    os.environ.setdefault("SSL_CERT_FILE", str(_local_ca_bundle))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_local_ca_bundle))

import numpy as np
import streamlit as st
import yfinance as yf

from data.cleaning import clean_option_rows


@st.cache_data(ttl=1800, show_spinner=False)
def get_underlying_info(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError("Sin datos de precio para este ticker.")
    S0 = float(hist["Close"].iloc[-1])
    try:
        q = float(tk.info.get("dividendYield", 0) or 0)
        if q > 1:  # yfinance sometimes returns a percentage instead of a decimal
            q = q / 100.0
    except Exception:
        q = 0.0
    expiries = list(tk.options)
    return S0, q, expiries


@st.cache_data(ttl=1800, show_spinner=False)
def get_clean_chain(ticker, expiry):
    """Download and clean the option chain (calls and puts) for a given expiry."""
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    return clean_option_rows(chain.calls, chain.puts)


def tau_from_expiry(expiry_str):
    exp = np.datetime64(expiry_str)
    today = np.datetime64("today")
    return max(float((exp - today) / np.timedelta64(365, "D")), 1 / 365)
