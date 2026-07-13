"""Panel 6 — Live Defense mode.

One clean screen for the moment the professor hands you a ticker and a contract on the
spot (brief slides 18-19): price from both engines vs market, every Greek, the
recommended engine with a one-line justification (Trader feedback, slide 24), Feller
status, and an auto-composed set of talking points so you have something to read from
while you reason out loud.
"""

import numpy as np
import streamlit as st

from models.black_scholes import bs_price, bs_implied_vol
from models.calibration import feller_condition
from models.greeks import bs_greeks, heston_greeks_fd, GREEK_LABELS
from models.heston import heston_price
from models.recommendation import recommend_engine
from utils.styling import inject_css
from utils.context import render_sidebar_controls, render_ticker_strip, get_heston_calibration

st.set_page_config(page_title="Live Defense · Options Analytics Dashboard", page_icon="◆", layout="wide")
inject_css()

ctx = render_sidebar_controls()
render_ticker_strip(ctx)

st.markdown(
    f"""
    <div class="desk-header">
        <h1>Live Defense</h1>
        <p>Pantalla única — precio, Greeks y recomendación para el contrato que te pidan en el momento</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Calibrando Heston a la sonrisa de este vencimiento..."):
    heston_params, fit_obj = get_heston_calibration(ctx)

if heston_params is None:
    st.warning(
        "No hay suficientes quotes líquidas para calibrar Heston en este vencimiento. "
        "Pide al profesor otro ticker/expiry más líquido, o cámbialo tú mismo en el sidebar."
    )
    st.stop()

v0, theta, kappa, xi, rho = heston_params
feller_ok, feller_lhs, feller_rhs = feller_condition(heston_params)

row_sel = ctx.sub_chain[ctx.sub_chain["strike"] == ctx.strike].iloc[0]
market_mid = float(row_sel["mid"])

atm_row = ctx.sub_chain.iloc[(ctx.sub_chain["strike"] - ctx.S0).abs().argsort()[:1]].iloc[0]
sigma_atm = bs_implied_vol(atm_row["mid"], ctx.S0, atm_row["strike"], ctx.r, ctx.q, ctx.tau, ctx.option_type)
if np.isnan(sigma_atm) or sigma_atm <= 0:
    iv = atm_row["impliedVolatility"]
    sigma_atm = float(iv) if not np.isnan(iv) else 0.20

price_bs = bs_price(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, sigma_atm, ctx.option_type)
price_heston = heston_price(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, v0, kappa, theta, xi, rho, ctx.option_type)
err_bs_rel = (price_bs - market_mid) / market_mid * 100 if market_mid else np.nan
err_heston_rel = (price_heston - market_mid) / market_mid * 100 if market_mid else np.nan

moneyness = ctx.strike / ctx.S0
rec = recommend_engine(moneyness, ctx.tau, feller_ok)

g_bs = bs_greeks(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, sigma_atm, ctx.option_type)
g_h = heston_greeks_fd(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, v0, kappa, theta, xi, rho, ctx.option_type, full=True)

# ---------- Big price row ----------
c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="metric-card"><div class="label">Mercado (mid)</div>
<div class="value" style="font-size:2rem;">${market_mid:,.2f}</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card"><div class="label">B&amp;S</div>
<div class="value" style="font-size:2rem;">${price_bs:,.2f}</div>
<div class="sub">error {err_bs_rel:+.1f}%</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card"><div class="label">Heston</div>
<div class="value" style="font-size:2rem; color:#5FDCB4;">${price_heston:,.2f}</div>
<div class="sub">error {err_heston_rel:+.1f}%</div></div>""", unsafe_allow_html=True)

# ---------- Recommendation ----------
rec_color = "#5FDCB4" if rec["engine"] == "Heston" else "#7FB8E8"
st.markdown(
    f"""
    <div class="interp-box" style="border-left-color:{rec_color};">
    <b>Motor recomendado para este contrato: <span style="color:{rec_color};">{rec['engine']}</span></b><br><br>
    {rec['justification']}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Greeks row ----------
st.markdown("##### Greeks")
gcols = st.columns(7)
for col, key in zip(gcols, ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{GREEK_LABELS[key]}</div>
            <div class="sub" style="font-size:0.72rem;">B&amp;S</div>
            <div class="value" style="font-size:1.05rem;">{g_bs[key]:.4f}</div>
            <div class="sub" style="font-size:0.72rem; margin-top:0.3rem;">Heston</div>
            <div class="value" style="font-size:1.05rem; color:#5FDCB4;">{g_h[key]:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Feller ----------
feller_tag = '<span class="tag tag-good">Feller OK</span>' if feller_ok else '<span class="tag tag-danger">Feller violado</span>'
st.markdown(f"{feller_tag} &nbsp; 2κθ={feller_lhs:.4f} vs ξ²={feller_rhs:.4f} &middot; ρ={rho:.2f} · ξ={xi:.3f}", unsafe_allow_html=True)

# ---------- Auto-composed talking points ----------
st.markdown("---")
st.markdown("##### Puntos para defender en voz alta")

closer = "Heston" if abs(err_heston_rel) < abs(err_bs_rel) else "Black & Scholes"
money_word = "cerca del dinero" if abs(moneyness - 1) <= 0.03 else ("fuera del dinero" if (moneyness < 1) == (ctx.option_type == "put") else "dentro del dinero")
theta_word = "pierde" if g_bs["theta"] < 0 else "gana"
skew_word = "negativa (skew típico de equity, protección más cara)" if rho < 0 else "positiva/nula (skew atípico para equity)"

points = [
    f"**Precio**: mercado ${market_mid:,.2f}; B&S ${price_bs:,.2f} ({err_bs_rel:+.1f}%); Heston ${price_heston:,.2f} "
    f"({err_heston_rel:+.1f}%). {closer} está más cerca hoy.",
    f"**Moneyness**: K/S₀={moneyness:.3f}, el contrato está {money_word}, a {ctx.tau*365:.0f} días del vencimiento.",
    f"**Delta**={g_bs['delta']:.3f} (B&S) — sensibilidad direccional inmediata; **Gamma**={g_bs['gamma']:.4f} — "
    f"qué tan rápido cambia esa Delta si el spot se mueve.",
    f"**Theta**={g_bs['theta']:.4f}/día — la posición larga {theta_word} valor cada día que pasa, todo lo demás constante.",
    f"**Vega**: B&S={g_bs['vega']:.4f} vs Heston={g_h['vega']:.4f} — Heston suele salir menor porque un shock a la "
    f"varianza de hoy se diluye con la reversión a la media (κ, θ).",
    f"**Skew**: ρ={rho:.2f} → correlación spot-vol {skew_word}.",
    f"**Motor recomendado**: {rec['engine']} — {rec['justification']}",
]
for p in points:
    st.markdown(f"- {p}")
