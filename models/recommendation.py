"""Per-contract engine recommendation (Trader desk feedback, brief slide 24):
a trader does not use one model for everything -- short-dated ATM is fine with B&S,
long-dated or off-the-money needs the skew, i.e. Heston.
"""

SHORT_DATED_DAYS = 45
LONG_DATED_DAYS = 90
ATM_BAND = 0.05
FAR_BAND = 0.15


def recommend_engine(moneyness, tau_years, feller_ok):
    """moneyness = strike / spot. Returns {engine, justification}."""
    days = tau_years * 365.0
    near_atm = abs(moneyness - 1.0) <= ATM_BAND
    far_from_atm = abs(moneyness - 1.0) >= FAR_BAND

    if days <= SHORT_DATED_DAYS and near_atm:
        engine = "Black & Scholes"
        justification = (
            f"Corto plazo ({days:.0f}d) y cerca del dinero (K/S={moneyness:.2f}): el skew apenas "
            "mueve el precio en esta esquina de la sonrisa, así que B&S con su vol plana ATM es "
            "suficientemente preciso y mucho más rápido de evaluar."
        )
    elif days >= LONG_DATED_DAYS or far_from_atm:
        engine = "Heston"
        justification = (
            f"{'Vencimiento largo (' + format(days, '.0f') + 'd)' if days >= LONG_DATED_DAYS else 'Lejos del dinero (K/S=' + format(moneyness, '.2f') + ')'}: "
            "aquí el skew de mercado pesa en el precio y B&S, al usar una sola vol plana, lo ignora "
            "por construcción. Heston lo captura vía ρ (skew) y ξ (curvatura de las alas)."
        )
    else:
        engine = "Heston"
        justification = (
            f"Zona intermedia (K/S={moneyness:.2f}, {days:.0f}d): ninguno de los dos motores es "
            "obviamente mejor por default, pero Heston nunca ignora el skew observado, así que es "
            "la opción más conservadora cuando no hay una razón clara para preferir B&S."
        )

    if not feller_ok:
        justification += (
            " Aviso: la condición de Feller está violada en esta calibración (la varianza puede "
            "tocar cero) -- vigila la estabilidad numérica del precio de Heston antes de confiar en él."
        )

    return {"engine": engine, "justification": justification}
