"""
Options Analytics Dashboard
Quantitative Finance — ITESO
Motores: Black-Scholes-Merton y Heston (stochastic vol)

Ejecutar con:  streamlit run app.py
"""

import warnings

import streamlit as st

from utils.styling import inject_css
from utils.context import render_sidebar_controls, render_ticker_strip, get_heston_calibration

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
    """
    <div class="desk-header">
        <h1>◆ Options Analytics Dashboard</h1>
        <p>Derivatives desk · pricing, Greeks, smile calibration and portfolio risk for equity options</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="interp-box">
    Selecciona un ticker, vencimiento y strike en la barra lateral — esa selección se mantiene
    al navegar entre páginas. Usa el menú de la izquierda para moverte entre las seis funciones
    del dashboard.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

pages = [
    ("Pricing", "Precio de B&S y Heston lado a lado con el mercado, error absoluto/relativo por motor."),
    ("Greeks", "Δ, Γ, ν, Θ, ρ, Vanna y Volga — de forma cerrada en B&S, bump-and-reprice en Heston."),
    ("Volatility Smile", "La sonrisa de volatilidad: mercado vs B&S (línea plana) vs Heston calibrado, "
                          "más una superficie 3D bonus a través de varios vencimientos."),
    ("Calibration", "Parámetros calibrados de Heston, condición de Feller, y un chequeo de estabilidad "
                     "día-a-día de la calibración."),
    ("Portfolio", "Arma un libro de 2-4 contratos y ve las Greeks netas, el hedge de Delta y el costo "
                   "de mantenerte Gamma-neutral."),
    ("Live Defense", "Pantalla única para la defensa en vivo: precio, Greeks, motor recomendado y "
                       "justificación para cualquier contrato que pida el profesor."),
]

cols = st.columns(3)
for i, (title, desc) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:1rem;">
                <div class="label">{title}</div>
                <div class="sub" style="color:#C9D6E5; font-size:0.85rem; line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.spinner("Calibrando Heston a la sonrisa de este vencimiento..."):
    heston_params, fit_obj = get_heston_calibration(ctx)

if heston_params is None:
    st.warning(
        "No hay suficientes quotes líquidas para calibrar Heston en este vencimiento. "
        "Prueba otro expiry o ticker más líquido antes de entrar a las demás páginas."
    )
else:
    st.caption(
        "Heston ya está calibrado para esta combinación de ticker/vencimiento — las demás páginas "
        "reutilizan este resultado (cacheado) sin recalcular."
    )
