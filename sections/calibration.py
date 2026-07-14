"""Section — Calibration detail and day-over-day stability check.

Addresses the Quant Developer's feedback (brief slide 26): naive least-squares can find
garbage parameters that still fit today's snapshot but are unstable day to day. This
section documents the two-stage optimiser already in use, flags the Feller condition,
and lets you build up a local history of calibrations to compare over time -- see the
limitation note below about why that history is built locally rather than from paid
historical data.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from models.calibration import (
    feller_condition, log_calibration_snapshot, load_calibration_history, HESTON_NAMES, heston_loss,
)
from utils.styling import ACCENT, ACCENT_2, TEXT_DIM, WARN, DANGER, apply_dark_layout
from utils.context import get_calibration_rows_for


def render(ctx, heston_params, fit_obj):
    v0, theta, kappa, xi, rho = heston_params
    feller_ok, feller_lhs, feller_rhs = feller_condition(heston_params)
    fit_cost = float(fit_obj.cost) if fit_obj is not None else float("nan")

    st.markdown("##### Parámetros calibrados hoy")
    meaning = {
        "v0": "Varianza inicial — nivel de vol de corto plazo, √v₀ ≈ IV cuando τ→0.",
        "theta": "Varianza de largo plazo — nivel al que reversiona la varianza, √θ ≈ IV cuando τ→∞.",
        "kappa": "Velocidad de reversión a la media — qué tan rápido la varianza vuelve a θ.",
        "xi": "Vol-of-vol — controla la curvatura (convexidad) de las alas de la sonrisa.",
        "rho": "Correlación spot-vol — controla la inclinación (skew) de la sonrisa; ρ<0 es la firma típica de equity.",
    }
    pcols = st.columns(5)
    labels = ["v₀", "θ", "κ", "ξ", "ρ"]
    for col, lab, name, val in zip(pcols, labels, HESTON_NAMES, heston_params):
        col.metric(lab, f"{val:.4f}", help=meaning[name])

    feller_tag = '<span class="tag tag-good">Feller OK</span>' if feller_ok else '<span class="tag tag-danger">Feller violado</span>'
    st.markdown(
        f"""
        <div class="interp-box">
        {feller_tag} &nbsp; 2κθ = {feller_lhs:.4f} vs ξ² = {feller_rhs:.4f}.
        {'La varianza nunca toca cero por construcción del proceso — la calibración es numéricamente estable.'
         if feller_ok else
         'La varianza puede tocar cero: vigila la estabilidad del pricer cerca de τ pequeño y v₀ chico, y no te '
         'sorprendas si distintos re-runs de la optimización caen en puntos algo distintos.'}
        Costo final del ajuste (suma de residuos² ponderados): {fit_cost:.4f}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("¿Por qué la calibración a veces 'se mueve' entre corridas? — el problema de identificabilidad κ-ξ"):
        st.markdown(
            """
            La superficie de pérdida de Heston tiene **valles planos**: muchas combinaciones de (κ, ξ) — y en
            menor medida θ — precian la sonrisa casi idéntico, aunque el RMSE en precio sea diminuto. v₀ y ρ
            normalmente sí quedan bien identificados porque controlan el nivel de corto plazo y el skew, que
            los datos fijan con más fuerza.

            **Mitigaciones estándar** (no todas implementadas aquí, pero es lo que haría un desk real):
            - Regularizar hacia los parámetros calibrados ayer (estabilidad temporal).
            - Fijar κ a un valor razonable y calibrar solo los otros cuatro.
            - Usar más de un vencimiento simultáneamente — la estructura temporal ayuda a separar κ de ξ.
            """
        )

        st.caption(
            "El texto de arriba lo describe; el mapa de abajo lo muestra: barre (κ, ξ) en una malla, fija "
            "v₀, θ y ρ en los valores calibrados hoy, y grafica log₁₀(pérdida) en cada punto contra las quotes "
            "reales que se usaron para calibrar -- la misma técnica del notebook de identificabilidad del curso."
        )
        show_valley = st.checkbox("Calcular el valle κ-ξ (una malla ~20×20, puede tardar 10-20s)")
        if show_valley:
            rows_for_valley = get_calibration_rows_for(ctx.ticker, ctx.expiry, ctx.S0)
            with st.spinner("Barriendo (κ, ξ) y repriciando contra el mercado en cada punto..."):
                kappas = np.linspace(0.2, 6.0, 20)
                xis = np.linspace(0.05, 1.5, 20)
                KK, XX = np.meshgrid(kappas, xis)
                L = np.zeros_like(KK)
                for a in range(KK.shape[0]):
                    for b in range(KK.shape[1]):
                        L[a, b] = heston_loss((v0, theta, KK[a, b], XX[a, b], rho), rows_for_valley, ctx.S0, ctx.r, ctx.q)

            fig_valley = go.Figure(data=go.Contour(
                x=kappas, y=xis, z=np.log10(L + 1e-12),
                colorscale="Viridis", contours=dict(showlabels=False),
                colorbar=dict(title="log₁₀(pérdida)"),
            ))
            fig_valley.add_trace(go.Scatter(
                x=[kappa], y=[xi], mode="markers", name="Calibrado hoy",
                marker=dict(color="white", size=12, symbol="star", line=dict(color="black", width=1)),
            ))
            fig_valley.update_layout(xaxis_title="κ (kappa)", yaxis_title="ξ (xi)")
            st.plotly_chart(apply_dark_layout(fig_valley, height=420, legend_top=False), use_container_width=True)
            st.caption(
                "Si la zona oscura (pérdida baja) forma una curva alargada en vez de un punto compacto, eso "
                "*es* el valle: muchos (κ, ξ) distintos explican casi igual de bien las mismas quotes de mercado."
            )

    st.markdown("---")
    st.markdown("##### Chequeo de estabilidad día a día")
    st.caption(
        "Limitación explícita: yfinance solo da un snapshot en vivo, no cadenas de opciones históricas (eso "
        "requiere datos pagados). Por eso este chequeo construye su propio historial: cada vez que corres el "
        "dashboard en este ticker/expiry, puedes guardar la calibración de hoy y compararla contra corridas de "
        "otros días."
    )

    if st.button("💾 Guardar la calibración de hoy"):
        path = log_calibration_snapshot(ctx.ticker, ctx.expiry, ctx.S0, heston_params, fit_cost)
        st.success(f"Snapshot guardado en {path.name}.")

    history = load_calibration_history(ctx.ticker, ctx.expiry)

    if len(history) < 2:
        st.info(
            f"Hay {len(history)} snapshot(s) guardado(s) para {ctx.ticker} {ctx.expiry}. "
            "Guarda al menos dos en días distintos para ver la comparación de estabilidad."
        )
    else:
        st.dataframe(
            history.style.format({"S0": "${:.2f}", "fit_cost": "{:.4f}", **{n: "{:.4f}" for n in HESTON_NAMES}}),
            use_container_width=True,
        )

        fig = go.Figure()
        colors = {"v0": ACCENT, "theta": ACCENT_2, "kappa": WARN, "xi": DANGER, "rho": TEXT_DIM}
        for name in HESTON_NAMES:
            fig.add_trace(go.Scatter(x=history["date"], y=history[name], mode="lines+markers",
                                      name=name, line=dict(color=colors[name], width=2)))
        fig.update_layout(xaxis_title="Fecha de calibración", yaxis_title="Valor del parámetro")
        st.plotly_chart(apply_dark_layout(fig, height=380), use_container_width=True)

        first, last = history.iloc[0], history.iloc[-1]
        drift = {n: last[n] - first[n] for n in HESTON_NAMES}
        biggest = max(drift, key=lambda n: abs(drift[n]))
        st.markdown(
            f"""
            <div class="interp-box">
            Entre {first['date']} y {last['date']}, el parámetro que más se movió fue <b>{biggest}</b>
            (Δ={drift[biggest]:+.4f}). Movimientos grandes en κ o ξ con v₀ y ρ estables son consistentes con
            el problema de identificabilidad descrito arriba — no necesariamente significan que el mercado
            cambió de régimen.
            </div>
            """,
            unsafe_allow_html=True,
        )
