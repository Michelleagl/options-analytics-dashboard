"""OCC symbol parsing and session-state bootstrapping."""

import re
from datetime import date

import streamlit as st

# Standard OCC format: ROOT + YYMMDD + C/P + STRIKE*1000 (8 digits)
# Example: SPY260708C00745000 -> SPY, exp 2026-07-08, CALL, strike 745.000
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(symbol):
    symbol = (symbol or "").strip().upper().replace(" ", "")
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root, datestr, cp, strikestr = m.groups()
    yy, mm, dd = int(datestr[0:2]), int(datestr[2:4]), int(datestr[4:6])
    year = 2000 + yy
    try:
        date(year, mm, dd)  # validate the date exists
    except ValueError:
        return None
    return dict(
        ticker=root,
        expiry=f"{year:04d}-{mm:02d}-{dd:02d}",
        option_type="call" if cp == "C" else "put",
        strike=int(strikestr) / 1000.0,
    )


def init_session_state():
    st.session_state.setdefault("ticker_input", "SPY")
    if "occ_pending" not in st.session_state:
        st.session_state["occ_pending"] = None
