"""Section — Portfolio Greeks.

Addresses the Risk Manager's feedback (brief slide 25): "Greeks without a portfolio are
an academic toy" -- risk lives at the book level. Lets the user build a 2-4 leg book on
the current ticker and see aggregate Delta/Gamma/Vega/Theta/Rho, plus a delta-hedge
estimate. B&S Greeks stay closed-form and Heston Greeks stay bump-and-reprice, same as
every other section -- the two engines' aggregates are reported separately rather than
mixed, per the Risk Manager's explicit "watch out for" flag.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data.market_data import get_underlying_info, get_clean_chain, tau_from_expiry
from models.black_scholes import bs_price, bs_implied_vol
from models.greeks import bs_greeks, heston_greeks_fd
from models.heston import heston_price
from utils.context import calibrate_heston_for

CONTRACT_MULTIPLIER = 100


@st.cache_data(ttl=1800, show_spinner=False)
def _price_and_greeks_for_leg(ticker, expiry, strike, option_type, S0, r, q):
    try:
        chain = get_clean_chain(ticker, expiry)
    except Exception as e:
        return dict(failed=True, error=str(e))
    tau = tau_from_expiry(expiry)
    sub = chain[chain["type"] == option_type]
    atm_row = sub.iloc[(sub["strike"] - S0).abs().argsort()[:1]].iloc[0]
    sigma_atm = bs_implied_vol(atm_row["mid"], S0, atm_row["strike"], r, q, tau, option_type)
    if np.isnan(sigma_atm) or sigma_atm <= 0:
        iv = atm_row["impliedVolatility"]
        sigma_atm = float(iv) if not np.isnan(iv) else 0.20

    price_bs = bs_price(S0, strike, r, q, tau, sigma_atm, option_type)
    g_bs = bs_greeks(S0, strike, r, q, tau, sigma_atm, option_type)

    params, _ = calibrate_heston_for(ticker, expiry, S0, r, q)
    if params is None:
        return dict(price_bs=price_bs, g_bs=g_bs, price_heston=np.nan, g_heston=None, calibrated=False, failed=False)
    v0, theta, kappa, xi, rho = params
    price_heston = heston_price(S0, strike, r, q, tau, v0, kappa, theta, xi, rho, option_type)
    g_heston = heston_greeks_fd(S0, strike, r, q, tau, v0, kappa, theta, xi, rho, option_type, full=True)
    return dict(price_bs=price_bs, g_bs=g_bs, price_heston=price_heston, g_heston=g_heston, calibrated=True, failed=False)


def render(ctx):
    st.caption(f"Libro de 2-4 contratos sobre {ctx.ticker}.")

    if "portfolio_legs" not in st.session_state:
        st.session_state["portfolio_legs"] = []

    try:
        _, _, all_expiries = get_underlying_info(ctx.ticker)
    except Exception as e:
        st.error(f"No se pudo descargar la lista de vencimientos para {ctx.ticker}: {e}")
        return

    st.markdown("##### Agregar contrato")
    with st.form("add_leg_form", clear_on_submit=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns([1.3, 1, 1.3, 1, 1])
        leg_expiry = fc1.selectbox("Vencimiento", all_expiries, index=min(3, len(all_expiries) - 1))
        leg_type = fc2.selectbox("Tipo", ["call", "put"])
        try:
            leg_chain = get_clean_chain(ctx.ticker, leg_expiry)
        except Exception as e:
            leg_chain = pd.DataFrame(columns=["strike", "type"])
            fc3.warning(f"No se pudo descargar este vencimiento: {e}")
        leg_strikes = sorted(leg_chain[leg_chain["type"] == leg_type]["strike"].unique().tolist())
        if leg_strikes:
            atm_idx = int(np.argmin(np.abs(np.array(leg_strikes) - ctx.S0)))
            leg_strike = fc3.selectbox("Strike", leg_strikes, index=atm_idx)
        else:
            leg_strike = None
            fc3.warning("Sin strikes líquidos")
        leg_side = fc4.selectbox("Lado", ["Largo", "Corto"])
        leg_qty = fc5.number_input("Contratos", min_value=1, max_value=50, value=1, step=1)

        submitted = st.form_submit_button("➕ Agregar al libro")
        if submitted and leg_strike is not None:
            if len(st.session_state["portfolio_legs"]) >= 4:
                st.warning("Máximo 4 contratos por libro. Quita uno antes de agregar otro.")
            else:
                st.session_state["portfolio_legs"].append({
                    "expiry": leg_expiry, "type": leg_type, "strike": leg_strike,
                    "side": leg_side, "qty": int(leg_qty),
                })

    legs = st.session_state["portfolio_legs"]

    if not legs:
        st.info("Agrega entre 2 y 4 contratos arriba para ver las Greeks del libro.")
        return

    st.markdown("##### Contratos en el libro")
    legs_display = pd.DataFrame(legs)
    legs_display.index = legs_display.index + 1
    st.dataframe(legs_display, use_container_width=True)

    remove_idx = st.selectbox(
        "Quitar contrato #", options=list(range(1, len(legs) + 1)),
        format_func=lambda i: f"#{i}: {legs[i-1]['side']} {legs[i-1]['qty']}x {legs[i-1]['type'].upper()} "
                               f"K={legs[i-1]['strike']:g} exp {legs[i-1]['expiry']}",
    )
    if st.button("🗑 Quitar"):
        legs.pop(remove_idx - 1)
        st.rerun()

    if len(legs) < 2:
        st.info("Agrega al menos un contrato más (mínimo 2 en el libro) para ver las Greeks agregadas.")
        return

    rows_bs, rows_heston = [], []
    net_bs = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    net_heston = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    any_uncalibrated = False

    with st.spinner("Precificando cada contrato del libro (B&S y Heston)..."):
        for i, leg in enumerate(legs, start=1):
            sign = 1 if leg["side"] == "Largo" else -1
            weight = sign * leg["qty"] * CONTRACT_MULTIPLIER
            res = _price_and_greeks_for_leg(ctx.ticker, leg["expiry"], leg["strike"], leg["type"], ctx.S0, ctx.r, ctx.q)

            if res.get("failed"):
                st.warning(
                    f"Contrato #{i} ({leg['type'].upper()} K={leg['strike']:g} exp {leg['expiry']}) no se pudo "
                    f"descargar: {res['error']}. Se omite del libro."
                )
                continue

            rows_bs.append({"#": i, "Precio": res["price_bs"], **{k: res["g_bs"][k] for k in net_bs}})
            for k in net_bs:
                net_bs[k] += weight * res["g_bs"][k]

            if res["calibrated"]:
                rows_heston.append({"#": i, "Precio": res["price_heston"], **{k: res["g_heston"][k] for k in net_heston}})
                for k in net_heston:
                    net_heston[k] += weight * res["g_heston"][k]
            else:
                any_uncalibrated = True
                rows_heston.append({"#": i, "Precio": np.nan, **{k: np.nan for k in net_heston}})

    if any_uncalibrated:
        st.warning(
            "Al menos un vencimiento del libro no tuvo suficientes quotes líquidas para calibrar Heston; "
            "sus Greeks de Heston se omiten del neto (quedan como NaN en la tabla)."
        )

    if not rows_bs:
        st.warning("No quedaron contratos válidos en el libro tras los errores de descarga.")
        return

    st.markdown("##### Greeks por contrato")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Black & Scholes** (forma cerrada)")
        st.dataframe(pd.DataFrame(rows_bs).set_index("#").style.format("{:.4f}"), use_container_width=True)
    with t2:
        st.markdown("**Heston** (bump-and-reprice)")
        st.dataframe(pd.DataFrame(rows_heston).set_index("#").style.format("{:.4f}"), use_container_width=True)

    st.markdown("##### Greeks netas del libro")
    n1, n2 = st.columns(2)
    GREEK_ROW_LABELS = {"delta": "Δ Delta", "gamma": "Γ Gamma", "vega": "ν Vega", "theta": "Θ Theta/día", "rho": "ρ Rho"}
    for col, net, label in [(n1, net_bs, "B&S"), (n2, net_heston, "Heston")]:
        with col:
            st.markdown(f"**Neto — {label}**")
            cc = st.columns(5)
            for c, k in zip(cc, net_bs):
                c.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="label">{GREEK_ROW_LABELS[k]}</div>
                        <div class="value" style="font-size:1.05rem;">{net[k]:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("##### Delta-hedge y costo de Gamma-neutralidad")

    shares_to_hedge = -net_bs["delta"]
    hedge_cost = shares_to_hedge * ctx.S0
    hedge_side = "comprar" if shares_to_hedge > 0 else "vender"

    atm_row_ctx = ctx.sub_chain.iloc[(ctx.sub_chain["strike"] - ctx.S0).abs().argsort()[:1]].iloc[0]
    sigma_atm_ctx = bs_implied_vol(atm_row_ctx["mid"], ctx.S0, atm_row_ctx["strike"], ctx.r, ctx.q, ctx.tau, ctx.option_type)
    if np.isnan(sigma_atm_ctx) or sigma_atm_ctx <= 0:
        sigma_atm_ctx = 0.20
    gamma_ref = bs_greeks(ctx.S0, ctx.S0, ctx.r, ctx.q, ctx.tau, sigma_atm_ctx, "call")["gamma"]
    price_ref = bs_price(ctx.S0, ctx.S0, ctx.r, ctx.q, ctx.tau, sigma_atm_ctx, "call")

    if gamma_ref and not np.isnan(gamma_ref) and gamma_ref != 0:
        contracts_for_gamma = -net_bs["gamma"] / (gamma_ref * CONTRACT_MULTIPLIER)
        gamma_hedge_cost = abs(contracts_for_gamma) * CONTRACT_MULTIPLIER * price_ref
        gamma_side = "comprar" if contracts_for_gamma > 0 else "vender"
    else:
        contracts_for_gamma, gamma_hedge_cost, gamma_side = np.nan, np.nan, "—"

    st.markdown(
        f"""
        <div class="interp-box">
        <b>Delta-hedge (usando las Greeks de B&amp;S):</b> el libro tiene Delta neta de {net_bs['delta']:,.2f}
        acciones-equivalentes. Para quedar delta-neutral hay que <b>{hedge_side} {abs(shares_to_hedge):,.0f}
        acciones</b> de {ctx.ticker} (~${abs(hedge_cost):,.0f} de notional a spot).<br><br>
        <b>Gamma-neutralidad (aproximada):</b> usando un ATM call de referencia sobre el vencimiento
        seleccionado en el sidebar (Γ={gamma_ref:.5f} por contrato) como instrumento de cobertura, harían falta
        aproximadamente <b>{gamma_side} {abs(contracts_for_gamma):,.1f} contratos</b> de ese ATM
        (~${gamma_hedge_cost:,.0f} de prima) para llevar la Gamma neta a cero. Esto es una aproximación de
        primer orden — cubrir Gamma con un solo instrumento cambia también la Delta del libro, que habría que
        re-cubrir después.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Metodología de Greeks documentada por consistencia: B&S siempre en forma cerrada; Heston siempre "
        "bump-and-reprice sobre el mismo motor de característica de Fourier. Los netos de cada motor se "
        "reportan por separado — nunca se mezclan Greeks de B&S con Greeks de Heston en una misma suma."
    )
