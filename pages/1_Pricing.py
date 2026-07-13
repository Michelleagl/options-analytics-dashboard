"""Panel 1 — Pricing & Validation."""

import numpy as np
import pandas as pd
import streamlit as st

from models.black_scholes import bs_price, bs_implied_vol
from models.heston import heston_price
from plots.pricing_plots import make_price_vs_strike_figure
from utils.styling import inject_css
from utils.context import render_sidebar_controls, render_ticker_strip, get_heston_calibration

st.set_page_config(page_title="Pricing · Options Analytics Dashboard", page_icon="◆", layout="wide")
inject_css()

ctx = render_sidebar_controls()
render_ticker_strip(ctx)

st.markdown(
    f"""
    <div class="desk-header">
        <h1>Panel 1 · Pricing &amp; Validación</h1>
        <p>{ctx.ticker} &middot; {ctx.option_type.upper()} K={ctx.strike:g} &middot; exp {ctx.expiry}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Calibrando Heston a la sonrisa de este vencimiento..."):
    heston_params, fit_obj = get_heston_calibration(ctx)

if heston_params is None:
    st.warning(
        "No hay suficientes quotes líquidas para calibrar Heston en este vencimiento. "
        "Prueba otro expiry o ticker más líquido."
    )
    st.stop()

v0, theta, kappa, xi, rho = heston_params
from models.calibration import feller_condition
feller_ok, feller_lhs, feller_rhs = feller_condition(heston_params)

row_sel = ctx.sub_chain[ctx.sub_chain["strike"] == ctx.strike].iloc[0]
market_mid = float(row_sel["mid"])
market_iv = row_sel["impliedVolatility"]

# ATM implied vol (the "flat-vol world" B&S needs)
atm_row = ctx.sub_chain.iloc[(ctx.sub_chain["strike"] - ctx.S0).abs().argsort()[:1]].iloc[0]
sigma_atm = bs_implied_vol(atm_row["mid"], ctx.S0, atm_row["strike"], ctx.r, ctx.q, ctx.tau, ctx.option_type)
if np.isnan(sigma_atm) or sigma_atm <= 0:
    sigma_atm = float(market_iv) if not np.isnan(market_iv) else 0.20

is_atm_reference_strike = abs(ctx.strike - atm_row["strike"]) < 1e-6

price_bs = bs_price(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, sigma_atm, ctx.option_type)
price_heston = heston_price(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, v0, kappa, theta, xi, rho, ctx.option_type)

err_bs_abs = price_bs - market_mid
err_heston_abs = price_heston - market_mid
err_bs_rel = err_bs_abs / market_mid * 100 if market_mid else np.nan
err_heston_rel = err_heston_abs / market_mid * 100 if market_mid else np.nan

if is_atm_reference_strike:
    st.warning(
        f"⚠️ **Estás parado en el strike de referencia ATM (K={ctx.strike:g}).** La σ de B&S se calculó "
        "*a partir de la IV implícita de este mismo contrato*, así que B&S lo reproduce casi exactamente "
        "por construcción (es circular, no es una coincidencia ni un error del pipeline de datos). "
        "Para ver la divergencia real entre B&S y Heston, elige un strike bien OTM o ITM en el sidebar."
    )

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Precio de mercado (mid)</div>
            <div class="value">${market_mid:,.2f}</div>
            <div class="sub" style="color:#8492A6;">bid {row_sel['bid']:.2f} / ask {row_sel['ask']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    color = "#5FDCB4" if abs(err_bs_rel) < abs(err_heston_rel) else "#F0A85C"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Black &amp; Scholes <span class="tag tag-accent">σ_ATM={sigma_atm*100:.1f}%</span></div>
            <div class="value">${price_bs:,.2f}</div>
            <div class="sub" style="color:{color};">error {err_bs_abs:+.2f} ({err_bs_rel:+.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    color = "#5FDCB4" if abs(err_heston_rel) < abs(err_bs_rel) else "#F0A85C"
    feller_tag = '<span class="tag tag-good">Feller OK</span>' if feller_ok else '<span class="tag tag-warn">Feller violado</span>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Heston (calibrado) {feller_tag}</div>
            <div class="value">${price_heston:,.2f}</div>
            <div class="sub" style="color:{color};">error {err_heston_abs:+.2f} ({err_heston_rel:+.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

comp_df = pd.DataFrame(
    {
        "Motor": ["Black & Scholes", "Heston"],
        "Precio": [price_bs, price_heston],
        "Mercado (mid)": [market_mid, market_mid],
        "Error absoluto": [err_bs_abs, err_heston_abs],
        "Error relativo (%)": [err_bs_rel, err_heston_rel],
    }
).set_index("Motor")
st.dataframe(
    comp_df.style.format(
        {"Precio": "${:.2f}", "Mercado (mid)": "${:.2f}", "Error absoluto": "{:+.3f}", "Error relativo (%)": "{:+.2f}%"}
    ),
    use_container_width=True,
)

with st.expander("Parámetros de Heston calibrados en este vencimiento"):
    pcols = st.columns(5)
    labels = ["v₀ (var. inicial)", "θ (var. largo plazo)", "κ (reversión)", "ξ (vol-of-vol)", "ρ (correlación)"]
    for col, lab, val in zip(pcols, labels, heston_params):
        col.metric(lab, f"{val:.4f}")
    st.caption(
        f"Condición de Feller: 2κθ = {feller_lhs:.4f}  vs  ξ² = {feller_rhs:.4f}  →  "
        f"{'se cumple ✅ (varianza no toca cero)' if feller_ok else 'VIOLADA ⚠️ (la varianza puede tocar cero — vigilar estabilidad numérica)'}"
    )

strikes_sorted = np.array(sorted(ctx.strikes_available))
bs_curve = [bs_price(ctx.S0, K, ctx.r, ctx.q, ctx.tau, sigma_atm, ctx.option_type) for K in strikes_sorted]
heston_curve = [heston_price(ctx.S0, K, ctx.r, ctx.q, ctx.tau, v0, kappa, theta, xi, rho, ctx.option_type) for K in strikes_sorted]
market_curve = []
for K in strikes_sorted:
    mrow = ctx.sub_chain[ctx.sub_chain["strike"] == K]
    market_curve.append(float(mrow["mid"].iloc[0]) if not mrow.empty else np.nan)

st.plotly_chart(
    make_price_vs_strike_figure(strikes_sorted, market_curve, bs_curve, heston_curve, ctx.S0, ctx.strike, ctx.option_type),
    use_container_width=True,
)

# ---------- Interpretación automática ----------
moneyness = ctx.strike / ctx.S0
if ctx.option_type == "call":
    money_state = "ATM" if 0.97 <= moneyness <= 1.03 else ("ITM" if moneyness < 0.97 else "OTM")
else:
    money_state = "ATM" if 0.97 <= moneyness <= 1.03 else ("ITM" if moneyness > 1.03 else "OTM")

closer = "Heston" if abs(err_heston_rel) < abs(err_bs_rel) else "Black & Scholes"
gap = abs(abs(err_bs_rel) - abs(err_heston_rel))

if money_state == "ATM":
    circularidad_txt = (
        " De hecho, si tu strike coincide exactamente con el strike usado para calcular esa σ ATM, "
        "el error de B&S es ~0% **por construcción** (es circular: la σ se calculó a partir de este mismo precio), "
        "no porque el modelo esté siendo validado de forma independiente."
        if is_atm_reference_strike else ""
    )
    razon = (
        "En el dinero (ATM), B&S usa una σ derivada de la IV implícita observada justo en este vecindario, "
        "así que es esperable que se acerque bastante al mercado *en este punto específico*. "
        "La sonrisa de volatilidad casi no importa aquí porque las alas del smile apenas afectan al strike ATM."
        f"{circularidad_txt} La prueba real de un modelo de vol plana no es qué tan bien ajusta en el punto "
        "que se usó para calibrarlo, sino qué tan bien ajusta lejos de él (ve a un strike OTM/ITM)."
    )
elif money_state == "OTM":
    razon = (
        f"Este contrato está **fuera del dinero ({moneyness:.2f}× spot)**. B&S sigue usando la volatilidad "
        "plana calibrada en el ATM, así que ignora por completo el *skew* — típicamente subestima (o sobreestima) "
        "el precio en las alas. Heston, al calibrarse contra toda la sonrisa observada, captura ese *skew* "
        "vía ρ (correlación spot-vol) y ξ (vol-of-vol), por eso suele acercarse más al mercado en strikes alejados."
    )
else:
    razon = (
        f"Este contrato está **dentro del dinero ({moneyness:.2f}× spot)**. Aquí el valor intrínseco domina "
        "el precio y ambos motores tienden a converger, aunque Heston puede seguir teniendo una ligera ventaja "
        "si el smile tiene skew pronunciado hacia ese lado."
    )

st.markdown(
    f"""
    <div class="interp-box">
    <b>¿Qué motor está más cerca del mercado?</b><br><br>
    Para este contrato ({ctx.ticker} {ctx.option_type} K={ctx.strike:g}, exp {ctx.expiry}), el motor más cercano
    al mid de mercado es <b>{closer}</b> (diferencia de {gap:.2f} puntos porcentuales de error relativo entre ambos).<br><br>
    {razon}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Nota: σ de B&S es la volatilidad implícita ATM de este mismo vencimiento (mundo de volatilidad plana). "
    "Heston se calibra usando las quotes líquidas (moneyness 0.80–1.20) del vencimiento seleccionado, "
    "vía búsqueda global (Latin Hypercube) + refinamiento local (Levenberg-Marquardt)."
)
