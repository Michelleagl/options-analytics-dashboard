"""Risk-free rate proxy."""

import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate():
    """13-week T-Bill (FRED DTB3) as the risk-free rate proxy. Falls back to a fixed
    rate if FRED is unreachable."""
    try:
        import pandas_datareader.data as web
        from datetime import datetime, timedelta

        end = datetime.today()
        start = end - timedelta(days=15)
        df = web.DataReader("DTB3", "fred", start, end)
        return float(df.dropna().iloc[-1, 0]) / 100.0
 
