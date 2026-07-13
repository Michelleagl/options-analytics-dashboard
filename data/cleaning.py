"""Pure option-chain cleaning logic, kept separate from the yfinance fetch so it can
be unit-tested without hitting the network.

Required cleaning steps (brief slide 9): drop zero-bid/zero-ask rows, drop illiquid
rows (no volume AND no open interest), keep bid/ask mid and spread for downstream
weighting.
"""

import pandas as pd


def clean_option_rows(calls_df, puts_df):
    """calls_df/puts_df: raw yfinance option_chain().calls / .puts DataFrames.
    Returns a tidy DataFrame with columns:
    strike, type, bid, ask, mid, spread, volume, openInterest, impliedVolatility.
    """
    rows = []
    for df, otype in [(calls_df, "call"), (puts_df, "put")]:
        for _, row in df.iterrows():
            bid, ask = row.get("bid", 0), row.get("ask", 0)
            vol = row.get("volume", 0) or 0
            if bid is None or ask is None:
                continue
            if bid <= 0 or ask <= 0:
                continue
            if (vol or 0) <= 0 and (row.get("openInterest", 0) or 0) <= 0:
                continue  # discard illiquid: no volume AND no open interest
            mid = (bid + ask) / 2.0
            spread = ask - bid
            rows.append(
                {
                    "strike": float(row["strike"]),
                    "type": otype,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": spread,
                    "volume": vol,
                    "openInterest": row.get("openInterest", 0),
                    "impliedVolatility": row.get("impliedVolatility", float("nan")),
                }
            )
    return pd.DataFrame(rows)
