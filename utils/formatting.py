"""Small formatting helpers reused across pages."""


def fmt_money(x, decimals=2):
    return f"${x:,.{decimals}f}"


def fmt_pct(x, decimals=1):
    return f"{x * 100:.{decimals}f}%"


def fmt_signed_pct(x, decimals=1):
    return f"{x:+.{decimals}f}%"
