# Options Analytics Dashboard

Quantitative Finance final project (ITESO) — an options pricing and risk desk in a
browser: Black-Scholes-Merton and Heston (stochastic volatility) side by side,
validated against live market quotes.

## Structure

```
app.py                  Entry point / landing page (sidebar controls live here)
pages/                  One Streamlit page per panel (native multi-page app)
  1_📈_Pricing.py         Panel 1 — price vs market, error by engine
  2_📊_Greeks.py           Panel 2 — Δ Γ ν Θ ρ, Vanna, Volga
  3_🌊_Volatility_Smile.py Panel 3 — the smile, B&S flat line vs Heston, 3D surface bonus
  4_⚙️_Calibration.py      Calibrated Heston params, Feller check, day-over-day stability
  5_💼_Portfolio.py        2-4 leg book, net Greeks, delta-hedge estimate
  6_🎯_Live_Defense.py     One screen for defending a contract on the spot
models/                 Pure pricing math: black_scholes.py, heston.py, greeks.py,
                        calibration.py, recommendation.py — no Streamlit, no I/O
data/                   yfinance/FRED fetch + cleaning + smile/surface calculations
plots/                  Plotly figure builders, one per chart type
utils/                  Theme (styling.py), shared sidebar/state (context.py),
                        OCC parsing (helpers.py), small formatters
tests/                  pytest: put-call parity, IV round-trips, Heston Delta==P1
                        cross-check, synthetic calibration recovery
```

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

Ticker/expiry/strike selection lives in the sidebar and is shared across all six
pages via `st.session_state` (see `utils/context.py`).

## Testing

```
pytest
```

12 tests covering B&S, Heston, and calibration. The calibration test mirrors the
course's own validation method: generate prices from known parameters, calibrate
from a different starting point, and check the fitted surface reproduces the
synthetic market (see `tests/test_calibration.py` for why exact parameter recovery
isn't asserted — kappa/theta/xi sit in a known flat, poorly-identified valley).

## Known limitations (stated up front, not discovered live)

- **Options only quote during market hours.** Outside 9:30–16:00 ET, yfinance
  returns `bid=ask=0` for essentially the whole chain, and the cleaning pipeline
  correctly rejects that as required by the brief ("drop zero-bid rows") — you'll
  see a "no liquid quotes" message rather than a chart. Re-run during market hours,
  or use a cached snapshot.
- **No historical option chains.** The Calibration page's day-over-day stability
  check (`models/calibration.py: log_calibration_snapshot` /
  `load_calibration_history`) builds its own local history one run at a time,
  since free data sources only expose a live snapshot, not a historical surface.
- **European vanilla only**, one market snapshot, no exotics — matches the scope
  the brief's own desk feedback agreed to cut.

## Local SSL / antivirus interception

If `streamlit run app.py` shows "No se pudo descargar" for every ticker even
during market hours, some antivirus products (e.g. Avast's Web/Mail Shield) do
TLS interception with a root certificate that Windows trusts but Python's bundled
`certifi` package doesn't, and `yfinance`'s HTTP client (`curl_cffi`) has its own
cert store that bypasses the OS trust settings truststore normally fixes. Fix:

```powershell
$cert = Get-ChildItem Cert:\CurrentUser\Root | Where-Object Subject -like "*Avast*"
[IO.File]::WriteAllBytes("$env:TEMP\root.cer", $cert.Export('Cert'))
certutil -encode "$env:TEMP\root.cer" "$env:TEMP\root.pem" | Out-Null
python -c "import certifi,shutil; shutil.copy(certifi.where(), '.certs/combined_ca_bundle.pem')"
Get-Content "$env:TEMP\root.pem" >> .certs\combined_ca_bundle.pem
```

`data/market_data.py` picks up `.certs/combined_ca_bundle.pem` automatically if it
exists (it's gitignored — machine-specific, never commit it). On machines without
this quirk, nothing changes.

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repo (`git remote add origin ...`, `git push`).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the
   repo, branch, and `app.py`.
3. `requirements.txt` is already at the repo root, so no extra config is needed.
4. yfinance can be rate-limited harder from shared cloud IPs than from a home
   connection — if the live demo is unreliable, keep a cached chain snapshot as a
   fallback for the in-class defense (the brief itself asks for the dashboard to
   be "offline-capable on a saved snapshot and ready to pull live").
