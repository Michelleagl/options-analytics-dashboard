"""Shared sidebar controls and per-page data/calibration context.

Every page in pages/ calls render_sidebar_controls() first. It owns the ticker/OCC
box/expiry/strike widgets (keyed into st.session_state so selections persist as the
user navigates between pages) and returns an AppContext with everything a page needs:
spot, rate, dividend yield, tau, the cleaned chain, and the sub-chain for the selected
option type. This exists to avoid copy-pasting ~130 lines of sidebar logic into six
separate page files.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from data.market_data import get_underlying_info, get_clean_chain, tau_from_expiry
from data.riskfree import get_risk_free_rate
from models.calibration import calibrate_heston
from utils.helpers import parse_occ_symbol, init_session_state


@dataclass
class AppContext:
    ticker: str
    expiry: str
    option_type: str
    strike: float
    S0: float
    q: float
    r: float
    tau: float
    chain_df: pd.DataFrame
    sub_chain: pd.DataFrame
    strikes_available: list = field(default_factory=list)


def render_sidebar_controls():
    init_session_state()
    st.sidebar.markdown("## ⚙ Configuración del contrato")

    with st.sidebar.expander("📋 Pegar símbolo OCC (opcional)", expanded=False):
        occ_raw = st.text_input("Ej. SPY260708C00745000", key="occ_symbol_box")
        if st.button("Usar este símbolo"):
            parsed = parse_occ_symbol(occ_raw)
            if parsed is None:
                st.error("Formato inválido. Debe ser TICKER + AAMMDD + C/P + 8 dígitos de strike (strike×1000).")
            else:
                st.session_state["ticker_input"] = parsed["ticker"]
                st.session_state["occ_pending"] = parsed
                st.success(
                    f"Detectado: {parsed['ticker']} · {parsed['option_type'].upper()} · "
                    f"K={parsed['strike']:g} · exp {parsed['expiry']}"
                )
                st.rerun()

    ticker = st.sidebar.text_input("Ticker", key="ticker_input").strip().upper()

    data_error = None
    S0, q, expiries = None, 0.0, []
    if ticker:
        try:
            S0, q, expiries = get_underlying_info(ticker)
        except Exception as e:
            data_error = str(e)

    if data_error:
        st.sidebar.error(f"No se pudo descargar '{ticker}': {data_error}")
        st.stop()

    if not expiries:
        st.sidebar.warning("Este ticker no tiene opciones listadas.")
        st.stop()

    # If we just parsed an OCC symbol, pre-select the closest available expiry
    pending = st.session_state.get("occ_pending")
    if pending is not None and pending["ticker"] == ticker:
        if pending["expiry"] in expiries:
            matched_expiry = pending["expiry"]
        else:
            target_dt = np.datetime64(pending["expiry"])
            matched_expiry = min(expiries, key=lambda e: abs(np.datetime64(e) - target_dt))
            st.sidebar.info(
                f"No hay opciones exactas para {pending['expiry']}; usando el vencimiento más "
                f"cercano disponible: {matched_expiry}."
            )
        st.session_state["expiry_select"] = matched_expiry
        st.session_state["type_radio"] = pending["option_type"]

    if "expiry_select" not in st.session_state or st.session_state["expiry_select"] not in expiries:
        st.session_state["expiry_select"] = expiries[min(3, len(expiries) - 1)]

    expiry = st.sidebar.selectbox("Vencimiento (expiry)", expiries, key="expiry_select")
    option_type = st.sidebar.radio("Tipo de contrato", ["call", "put"], horizontal=True, key="type_radio")

    with st.spinner("Descargando y limpiando cadena de opciones..."):
        chain_df = get_clean_chain(ticker, expiry)

    if chain_df.empty:
        st.sidebar.warning("No quedaron quotes líquidas tras la limpieza para este vencimiento.")
        st.stop()

    sub_chain = chain_df[chain_df["type"] == option_type].sort_values("strike")
    strikes_available = sub_chain["strike"].unique().tolist()

    if not strikes_available:
        st.sidebar.warning(f"No hay {option_type}s líquidos para este vencimiento.")
        st.stop()

    # If we just parsed an OCC symbol, pre-select the closest liquid strike
    if pending is not None and pending["ticker"] == ticker:
        nearest_strike = min(strikes_available, key=lambda k: abs(k - pending["strike"]))
        if abs(nearest_strike - pending["strike"]) > 1e-6:
            st.sidebar.info(f"Strike {pending['strike']:g} no está líquido; usando el más cercano: {nearest_strike:g}.")
        st.session_state["strike_select"] = nearest_strike
        st.session_state["occ_pending"] = None  # consumed

    if "strike_select" not in st.session_state or st.session_state["strike_select"] not in strikes_available:
        atm_idx = int(np.argmin(np.abs(np.array(strikes_available) - S0)))
        st.session_state["strike_select"] = strikes_available[atm_idx]

    strike = st.sidebar.selectbox("Strike (K)", strikes_available, key="strike_select")

    r = get_risk_free_rate()
    tau = tau_from_expiry(expiry)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="mono" style="font-size:0.85rem; color:#8492A6; line-height:1.9;">
        S₀ &nbsp;=&nbsp; <span style="color:#EAF2FB;">{S0:.2f}</span><br>
        q &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{q*100:.2f}%</span><br>
        r &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{r*100:.2f}%</span> (13w T-Bill)<br>
        τ &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{tau:.3f}</span> años ({int(tau*365)} días)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Quotes líquidas en este expiry: {len(chain_df)}")

    return AppContext(
        ticker=ticker, expiry=expiry, option_type=option_type, strike=strike,
        S0=S0, q=q, r=r, tau=tau, chain_df=chain_df, sub_chain=sub_chain,
        strikes_available=strikes_available,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def calibrate_heston_for(ticker, expiry, S0, r, q):
    """Cached Heston calibration keyed on plain (ticker, expiry, S0, r, q) -- usable
    outside an AppContext too, e.g. per-leg in the Portfolio page where each leg can
    have its own expiry. Returns (params, fit_obj) or (None, None) if there weren't
    enough liquid quotes to calibrate."""
    df = get_clean_chain(ticker, expiry)
    tau_ = tau_from_expiry(expiry)
    moneyness = df["strike"] / S0
    m = df[(moneyness >= 0.80) & (moneyness <= 1.20) & (df["mid"] > 0.05)]
    if len(m) < 5:
        m = df[df["mid"] > 0.02]  # relax if there are too few quotes
    rows = [
        (row["strike"], tau_, row["mid"], max(row["spread"], 0.01), row["type"])
        for _, row in m.iterrows()
    ]
    if len(rows) < 5:
        return None, None
    params, fit_obj = calibrate_heston(rows, S0, r, q, n_candidatos=20, n_refinar=4)
    return params, fit_obj


def get_heston_calibration(ctx: AppContext):
    """Cached Heston calibration for the current ticker/expiry. Returns (params, fit_obj)
    or (None, None) if there weren't enough liquid quotes to calibrate."""
    return calibrate_heston_for(ctx.ticker, ctx.expiry, ctx.S0, ctx.r, ctx.q)


def render_ticker_strip(ctx: AppContext):
    """Dense monospace status bar shown at the top of every page, terminal-style."""
    days = int(ctx.tau * 365)
    st.markdown(
        f"""
        <div class="ticker-strip">
            <span class="tk-item"><b>{ctx.ticker}</b> {ctx.S0:,.2f}</span>
            <span class="tk-item">{ctx.option_type.upper()} K=<b>{ctx.strike:g}</b></span>
            <span class="tk-item">EXP <b>{ctx.expiry}</b> ({days}d)</span>
            <span class="tk-item">r=<b>{ctx.r*100:.2f}%</b></span>
            <span class="tk-item">q=<b>{ctx.q*100:.2f}%</b></span>
            <span class="tk-item">τ=<b>{ctx.tau:.3f}y</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
