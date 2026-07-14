"""Section — The Greeks."""

import numpy as np
import pandas as pd
import streamlit as st

from models.black_scholes import bs_implied_vol
from models.greeks import bs_greeks, heston_greeks_fd, GREEK_LABELS
from plots.greeks_plots import make_greeks_grid_figure

SWEEP_KEYS = ["delta", "gamma", "vega", "theta", "rho", "vanna"]


@st.cache_data(ttl=1800, show_spinner=False)
def _sweep_greeks_vs_strike(strikes_tuple, S0, r, q, tau, sigma_atm, heston_params_tuple, option_type):
    v0_, theta_, kappa_, xi_, rho_ = heston_params_tuple
    rows = []
    for K in strikes_tuple:
        gb = bs_greeks(S0, K, r, q, tau, sigma_atm, option_type)
        gh = heston_greeks_fd(S0, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_, option_type, full=True)
        rows.append({"K": K, **{f"bs_{k}": gb[k] for k in SWEEP_KEYS}, **{f"h_{k}": gh[k] for k in SWEEP_KEYS}})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def _sweep_greeks_vs_spot(spot_tuple, K, r, q, tau, sigma_atm, heston_params_tuple, option_type):
    v0_, theta_, kappa_, xi_, rho_ = heston_params_tuple
    rows = []
    for S_ in spot_tuple:
        gb = bs_greeks(S_, K, r, q, tau, sigma_atm, option_type)
        gh = heston_greeks_fd(S_, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_, option_type, full=True)
        rows.append({"S": S_, **{f"bs_{k}": gb[k] for k in SWEEP_KEYS}, **{f"h_{k}": gh[k] for k in SWEEP_KEYS}})
    return pd.DataFrame(rows)


def render(ctx, heston_params, fit_obj):
    st.caption(
        "Δ, Γ y Θ de B&S/Heston coinciden casi exactamente cuando el vol-of-vol (ξ) es chico — así se validó "
        "este motor. Vega, Vanna y Volga sí difieren de forma estructural entre modelos: eso es precisamente "
        "lo que la volatilidad estocástica agrega, no un error de cálculo."
    )

    v0, theta, kappa, xi, rho = heston_params

    atm_row = ctx.sub_chain.iloc[(ctx.sub_chain["strike"] - ctx.S0).abs().argsort()[:1]].iloc[0]
    sigma_atm = bs_implied_vol(atm_row["mid"], ctx.S0, atm_row["strike"], ctx.r, ctx.q, ctx.tau, ctx.option_type)
    if np.isnan(sigma_atm) or sigma_atm <= 0:
        market_iv = atm_row["impliedVolatility"]
        sigma_atm = float(market_iv) if not np.isnan(market_iv) else 0.20

    g_bs_sel = bs_greeks(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, sigma_atm, ctx.option_type)
    g_h_sel = heston_greeks_fd(ctx.S0, ctx.strike, ctx.r, ctx.q, ctx.tau, v0, kappa, theta, xi, rho, ctx.option_type, full=True)

    st.markdown("##### Greeks del contrato seleccionado")
    cols = st.columns(6)
    for col, key in zip(cols, ["delta", "gamma", "vega", "theta", "rho", "vanna"]):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{GREEK_LABELS[key]}</div>
                <div class="sub" style="color:#8492A6; font-size:0.75rem;">B&amp;S</div>
                <div class="value" style="font-size:1.15rem;">{g_bs_sel[key]:.4f}</div>
                <div class="sub" style="color:#8492A6; font-size:0.75rem; margin-top:0.4rem;">Heston</div>
                <div class="value" style="font-size:1.15rem; color:#5FDCB4;">{g_h_sel[key]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Vanna (segundo orden) mide cómo cambia Delta ante un shock de volatilidad — o, equivalentemente, "
        "cómo cambia Vega ante un shock de spot. En Heston nace directamente de ρ (correlación spot-vol)."
    )

    with st.expander("Bonus segundo orden: Volga (convexidad de Vega)"):
        st.markdown(
            f"""
            <div class="metric-card" style="max-width:260px;">
                <div class="label">{GREEK_LABELS['volga']}</div>
                <div class="sub" style="color:#8492A6; font-size:0.75rem;">B&amp;S</div>
                <div class="value" style="font-size:1.25rem;">{g_bs_sel['volga']:.6f}</div>
                <div class="sub" style="color:#8492A6; font-size:0.75rem; margin-top:0.4rem;">Heston</div>
                <div class="value" style="font-size:1.25rem; color:#5FDCB4;">{g_h_sel['volga']:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Volga mide la convexidad de Vega (qué tan no-lineal es la exposición a volatilidad). "
            "En B&S existe matemáticamente pero no tiene una fuente estructural — en Heston nace de ξ (vol-of-vol)."
        )

    strikes_sorted = np.array(sorted(ctx.strikes_available))
    strikes_for_sweep = strikes_sorted
    if len(strikes_for_sweep) > 13:
        idx_pick = np.linspace(0, len(strikes_for_sweep) - 1, 13).astype(int)
        strikes_for_sweep = strikes_for_sweep[idx_pick]

    spot_range = np.linspace(0.7 * ctx.S0, 1.3 * ctx.S0, 13)

    with st.spinner("Calculando Greeks a través de strikes y de spot (bump-and-reprice)..."):
        df_strike = _sweep_greeks_vs_strike(tuple(strikes_for_sweep), ctx.S0, ctx.r, ctx.q, ctx.tau, sigma_atm, tuple(heston_params), ctx.option_type)
        df_spot = _sweep_greeks_vs_spot(tuple(spot_range), ctx.strike, ctx.r, ctx.q, ctx.tau, sigma_atm, tuple(heston_params), ctx.option_type)

    axis_choice = st.radio(
        "Ver Greeks a través de:", ["Strike (K)  — S fijo en el spot actual", "Spot (S)  — K fijo en el seleccionado"],
        horizontal=True,
    )
    df_plot = df_strike if axis_choice.startswith("Strike") else df_spot
    x_col = "K" if axis_choice.startswith("Strike") else "S"
    x_label = "Strike (K)" if axis_choice.startswith("Strike") else "Spot (S)"
    x_ref = ctx.strike if axis_choice.startswith("Strike") else ctx.S0

    st.plotly_chart(make_greeks_grid_figure(df_plot, x_col, x_label, x_ref), use_container_width=True)

    # ---------- Interpretación automática ----------
    idx_max_gamma_h = df_strike["h_gamma"].idxmax()
    K_max_gamma_h = df_strike.loc[idx_max_gamma_h, "K"]
    idx_max_gamma_bs = df_strike["bs_gamma"].idxmax()
    K_max_gamma_bs = df_strike.loc[idx_max_gamma_bs, "K"]

    vega_otm_low = df_strike[df_strike["K"] < ctx.S0]["h_vega"].mean()
    vega_otm_high = df_strike[df_strike["K"] > ctx.S0]["h_vega"].mean()
    vega_skew_txt = (
        "más sensible a volatilidad en los strikes bajos (puts OTM)" if vega_otm_low > vega_otm_high
        else "más sensible a volatilidad en los strikes altos (calls OTM)"
    ) if not (np.isnan(vega_otm_low) or np.isnan(vega_otm_high)) else "aproximadamente simétrica en este rango"

    theta_sign_txt = "negativo" if g_bs_sel["theta"] < 0 else "positivo"
    quien_pierde = "el comprador (posición larga)" if g_bs_sel["theta"] < 0 else "el vendedor (posición corta)"

    st.markdown(
        f"""
        <div class="interp-box">
        <b>Lectura de las Greeks para este contrato</b><br><br>
        <b>Gamma:</b> el pico de Γ (Heston) está en K≈{K_max_gamma_h:g} y en B&amp;S en K≈{K_max_gamma_bs:g} —
        ambos cerca del spot (S₀={ctx.S0:.2f}), como se espera: la convexidad del payoff es máxima justo ATM.
        Para quien hace delta-hedging, esto significa que <b>cerca del dinero y cerca del vencimiento hay que
        re-balancear la cobertura con más frecuencia</b> — el Delta cambia rápido ante pequeños movimientos del spot.<br><br>
        <b>Theta:</b> para este {ctx.option_type}, Θ es <b>{theta_sign_txt}</b> ({g_bs_sel['theta']:.4f} por día en B&amp;S).
        Un Θ negativo significa que <b>{quien_pierde}</b> pierde valor cada día que pasa, todo lo demás constante
        — es el "alquiler" que se paga por tener optionalidad.<br><br>
        <b>Vega a través de strikes:</b> en Heston, la sensibilidad a volatilidad resulta {vega_skew_txt}
        (ρ={rho:.2f} en la calibración actual). Esto es exactamente el mecanismo detrás del <i>skew</i> de la
        sonrisa de volatilidad: cuando ρ&lt;0, una caída del spot viene acompañada de un alza en la volatilidad,
        así que los strikes bajos (protección/puts) cargan más sensibilidad — el mercado los precia con IV más alta.<br><br>
        <b>B&amp;S vs Heston:</b> Delta, Gamma y Theta casi no cambian entre motores (ambos miden lo mismo: la forma
        del payoff y el paso del tiempo). Donde sí difieren es en <b>Vega</b> — en Heston es estructuralmente menor
        porque un shock a la varianza de hoy (v₀) se diluye con el tiempo por la reversión a la media (κ, θ) — y en
        <b>Vanna/Volga</b>, que en B&amp;S son casi cero pero en Heston capturan la asimetría y la convexidad de la
        sonrisa. Esa es, literalmente, la estructura adicional que compra la volatilidad estocástica.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Metodología: Greeks de B&S en forma cerrada; Greeks de Heston vía bump-and-reprice (diferencias finitas "
        "centradas) reparametrizando a σ₀=√v₀ para expresar vega/vanna/volga en las mismas unidades que B&S. "
        "Al shockear vega solo se mueve v₀, dejando κ, θ, ξ, ρ (calibrados) fijos — así se aísla el efecto de la "
        "varianza inicial de hoy."
    )
