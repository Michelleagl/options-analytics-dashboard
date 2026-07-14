"""
Options Analytics Dashboard
Quantitative Finance — ITESO
Motores: Black-Scholes-Merton y Heston (stochastic vol)

Un solo dashboard continuo (sin tabs): la barra lateral fija el ticker/vencimiento/
strike una vez, y cada fase del análisis (pricing, Greeks, sonrisa, calibración,
portfolio, resumen) es una sección de la misma página, en el mismo orden que
el mandato del proyecto las pide.

Ejecutar con:  streamlit run app.py
"""

import warnings

import streamlit as st

from utils.styling import inject_css
from utils.context import render_sidebar_controls, render_ticker_strip, get_heston_calibration
from sections import pricing, greeks, smile, calibration, portfolio, summary

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Options Analytics Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

ctx = render_sidebar_controls()
render_ticker_strip(ctx)

st.markdown(
    f"""
    <div class="desk-header">
        <h1>◆ Options Analytics Dashboard</h1>
        <p>{ctx.ticker} &middot; {ctx.option_type.upper()} K={ctx.strike:g} &middot; exp {ctx.expiry}
        &mdash; pricing, Greeks, sonrisa, calibración, portfolio y resumen, todo para el mismo contrato</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-nav">
        <a href="#pricing">Pricing</a>
        <a href="#greeks">Greeks</a>
        <a href="#smile">Volatility Smile</a>
        <a href="#calibration">Calibration</a>
        <a href="#portfolio">Portfolio</a>
        <a href="#summary">Summary</a>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Calibrando Heston a la sonrisa de este vencimiento..."):
    heston_params, fit_obj = get_heston_calibration(ctx)

if heston_params is None:
    st.warning(
        "No hay suficientes quotes líquidas para calibrar Heston en este vencimiento. "
        "Prueba otro expiry o ticker más líquido en el sidebar."
    )
    st.stop()

st.markdown('<h2 class="section-title" id="pricing">Pricing &amp; Validación</h2>', unsafe_allow_html=True)
pricing.render(ctx, heston_params, fit_obj)

st.markdown("---")
st.markdown('<h2 class="section-title" id="greeks">Las Greeks</h2>', unsafe_allow_html=True)
greeks.render(ctx, heston_params, fit_obj)

st.markdown("---")
st.markdown('<h2 class="section-title" id="smile">Sonrisa de Volatilidad</h2>', unsafe_allow_html=True)
smile.render(ctx, heston_params, fit_obj)

st.markdown("---")
st.markdown('<h2 class="section-title" id="calibration">Calibración de Heston</h2>', unsafe_allow_html=True)
calibration.render(ctx, heston_params, fit_obj)

st.markdown("---")
st.markdown('<h2 class="section-title" id="portfolio">Portfolio Greeks</h2>', unsafe_allow_html=True)
portfolio.render(ctx)

st.markdown("---")
st.markdown('<h2 class="section-title" id="summary">Summary</h2>', unsafe_allow_html=True)
summary.render(ctx, heston_params, fit_obj)
