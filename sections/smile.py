"""Section — The Volatility Smile / Surface. The centrepiece of the project: it shows,
visually, why a constant-volatility model is not enough and what stochastic
volatility buys you."""

import numpy as np
import streamlit as st

from data.market_data import get_underlying_info, get_clean_chain, tau_from_expiry
from data.volatility import compute_market_smile, compute_model_smile_heston, build_surface_grid
from models.black_scholes import bs_implied_vol
from models.calibration import feller_condition
from plots.smile_plots import make_smile_figure
from plots.surface_plots import make_surface_figure


def render(ctx, heston_params, fit_obj):
    st.caption(
        "Esta es la prueba visual de por qué un modelo de volatilidad constante no basta: el mercado no precia "
        "todos los strikes con la misma vol implícita, y Heston, calibrado a la sonrisa completa, sí lo reproduce."
    )

    v0, theta, kappa, xi, rho = heston_params
    feller_ok, feller_lhs, feller_rhs = feller_condition(heston_params)

    market_smile_df = compute_market_smile(ctx.chain_df, ctx.S0, ctx.r, ctx.q, ctx.tau, ctx.option_type)

    atm_row = ctx.sub_chain.iloc[(ctx.sub_chain["strike"] - ctx.S0).abs().argsort()[:1]].iloc[0]
    sigma_atm = bs_implied_vol(atm_row["mid"], ctx.S0, atm_row["strike"], ctx.r, ctx.q, ctx.tau, ctx.option_type)
    if np.isnan(sigma_atm) or sigma_atm <= 0:
        market_iv = atm_row["impliedVolatility"]
        sigma_atm = float(market_iv) if not np.isnan(market_iv) else 0.20

    heston_smile_df = compute_model_smile_heston(
        sorted(market_smile_df["strike"].tolist()), ctx.S0, ctx.r, ctx.q, ctx.tau, heston_params, ctx.option_type
    )

    x_axis_choice = st.radio("Eje horizontal:", ["Log-Moneyness (ln(K/S₀))", "Strike (K)"], horizontal=True)
    x_axis = "log_moneyness" if x_axis_choice.startswith("Log-Moneyness") else "strike"

    st.plotly_chart(
        make_smile_figure(market_smile_df, sigma_atm, heston_smile_df, ctx.S0, x_axis=x_axis),
        use_container_width=True,
    )

    # ---------- Interpretación automática ----------
    skew_txt = (
        "negativa (ρ < 0), la firma clásica de equity: cuando el spot cae, la volatilidad sube. Esto infla la IV "
        "de los strikes bajos (puts OTM / protección) respecto a los altos — el mercado cobra más caro el "
        "'seguro contra caídas', exactamente lo que ves en la sonrisa inclinada hacia la izquierda."
        if rho < 0 else
        "positiva o cercana a cero (ρ ≥ 0), algo inusual para equities. La sonrisa calibrada no debería mostrar "
        "el skew típico de 'crash insurance' hacia los strikes bajos."
    )

    st.markdown(
        f"""
        <div class="interp-box">
        <b>Lectura de la sonrisa</b><br><br>
        La línea punteada de B&amp;S es plana por construcción: usa una sola σ (la ATM) para todos los strikes,
        así que ignora el <i>skew</i> que el mercado sí cobra. La curva de Heston se dobla porque ξ (vol-of-vol,
        ξ={xi:.3f} aquí) le da <b>curvatura</b> a las alas y ρ le da <b>inclinación</b> — en esta calibración,
        ρ={rho:.2f}: la correlación spot-vol es {skew_txt}<br><br>
        Si la curva de Heston no se pega bien al mercado en las colas, revisa la condición de Feller
        ({'se cumple' if feller_ok else '⚠️ está violada'} — 2κθ={feller_lhs:.4f} vs ξ²={feller_rhs:.4f}) y el
        rango de moneyness usado para calibrar (0.80–1.20): strikes muy lejanos del dinero, poco líquidos, quedan
        fuera de la calibración y pueden no ajustar tan bien.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### Bonus: superficie de volatilidad 3D")
    st.caption(
        "Extiende la sonrisa a varios vencimientos cercanos. Cada vencimiento requiere descargar y calibrar por "
        "separado, así que esto es opcional y está acotado a unos pocos expiries para no disparar el tiempo de cómputo."
    )

    show_surface = st.checkbox("Construir superficie 3D (mercado) a través de vencimientos cercanos")

    if show_surface:
        try:
            _, _, all_expiries = get_underlying_info(ctx.ticker)
        except Exception as e:
            st.error(f"No se pudo descargar la lista de vencimientos para {ctx.ticker}: {e}")
            return
        if ctx.expiry in all_expiries:
            start_idx = all_expiries.index(ctx.expiry)
        else:
            start_idx = 0
        window = all_expiries[start_idx: start_idx + 6]
        if len(window) < 2:
            window = all_expiries[:6]

        surface_entries = []
        with st.spinner(f"Descargando y calibrando {len(window)} vencimientos para la superficie..."):
            for exp in window:
                try:
                    chain = get_clean_chain(ctx.ticker, exp)
                    tau_e = tau_from_expiry(exp)
                    smile_e = compute_market_smile(chain, ctx.S0, ctx.r, ctx.q, tau_e, ctx.option_type)
                    if len(smile_e) >= 3:
                        surface_entries.append({
                            "days": tau_e * 365,
                            "moneyness": smile_e["moneyness"].tolist(),
                            "iv": smile_e["iv_market"].tolist(),
                        })
                except Exception:
                    continue

        grid = build_surface_grid(surface_entries)
        if grid is None:
            st.warning("No hubo suficientes vencimientos con datos líquidos para construir la superficie.")
        else:
            m_grid, days_grid, iv_grid = grid
            st.plotly_chart(make_surface_figure(m_grid, days_grid, iv_grid), use_container_width=True)
            st.caption(
                f"Superficie construida con {len(surface_entries)} vencimientos ({window[0]} a {window[-1]}), "
                "interpolando la IV de mercado sobre una malla común de moneyness."
            )
