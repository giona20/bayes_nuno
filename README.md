# NBES — NUNO Bayesian Edge Dashboard

Streamlit dashboard implementing the NBES strategy for Hyperliquid HIP-4
outcome contracts. Wired to three live books (from the Jun 1 / Jun 10 screenshot):

- **BTC > 74032 on Jun 1 8:00 AM** — binary, market ≈ 5% YES
- **BTC > 72551 on Jun 1 8:00 AM** — binary, market ≈ 6% YES
- **May CPI YoY** — 3-way categorical: below 4.3 (45%) / exactly 4.3 (41%) / above 4.3 (12%)

The CPI market is **not** a binary — it has three mutually exclusive buckets, so it
uses a categorical (softmax) prior and an independent edge/decision per bucket,
rather than the YES/NO logit model the BTC binaries use.

**Pipeline:** independent prior → sequential logit updating from weighted
log-likelihood ratios → edge vs. market mid → variance-adjusted quarter-Kelly
sizing, with trade gates (min edge, model SE, time-to-expiry, positive EV).

## Files
- `nbes_engine.py` — pure, testable engine (priors, logit accumulator, edge, Kelly). No I/O.
- `price_feed.py` — live BTC spot (multi-source) + live signal values (funding, OI change, momentum) from Binance public API. No key.
- `hl_outcomes.py` — Hyperliquid HIP-4 client: dynamic market discovery, prices, expiry.
- `calibrate_llr.py` — **calibration**: pulls historical BTC data from Binance, measures real LLRs per signal, writes `calibration.json`.
- `calibration_loader.py` — maps live signal values to the calibrated LLRs.
- `app.py` — Streamlit UI.
- `requirements.txt` — deps (all stdlib for fetching; no new packages).

## Calibrating the signals (turning placeholders into real LLRs)
The LLRs are no longer hand-picked. To generate real ones from history:
```bash
python calibrate_llr.py            # ~720 days of hourly data
python calibrate_llr.py --days 365 # shorter window
```
This writes `calibration.json`. The app loads it and maps the **current** live
signal values (funding rate, OI change, 6h momentum) to the LLR measured for
that value's historical bin. Re-run the script anytime — commit the refreshed
`calibration.json` and the app picks it up.

**Keeping it updated:** schedule `calibrate_llr.py` (e.g. a daily cron or GitHub
Action) so new settled days flow into the LLRs. Thin buckets are auto-shrunk
toward 0 so they can't fabricate a strong signal. If `calibration.json` is
absent, all LLRs default to 0 (prior only) — no fabricated bias.

## Live contract quotes
Each market has a **"Use live Hyperliquid quotes"** toggle. When on, the app
queries Hyperliquid's HIP-4 books and auto-fills the contract prices, showing
which `#N` coin resolved to each outcome so you can verify the match. The
keyword maps that drive matching live in `QUOTE_KEYWORDS` in `app.py` — adjust
them if the live market descriptions differ from the defaults. Recurring markets
rotate their `#N` encodings each period, which is why discovery is dynamic
rather than hardcoded.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy
**Streamlit Community Cloud (free):** push these files to a GitHub repo →
share.streamlit.io → New app → point at `app.py`. Done.

**Docker / your own box:**
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Wiring to live NUNO data (next step)
The signals table currently holds placeholder LLRs. Replace the
`default_signals` block in `app.py` with live pulls:
- **Funding skew** → Loris Tools 34-exchange API
- **OI delta / order-book imbalance** → aggr.trade workspace feed
- **Spot momentum / CVD** → your existing WebSocket order-book worker
- **Book bid/ask** → Hyperliquid HIP-4 outcome market endpoint

The critical missing piece is the **LLR calibration table**: for each signal,
measure the historical hit-rate of `P(signal state | YES outcome)` vs
`P(signal state | NO outcome)` and convert to `ln(ratio)`. Until calibrated,
treat sizing as illustrative only.

## Disclaimer
Educational expected-value modelling tool. Not financial advice. Thin HIP-4
books carry real execution and settlement risk; calibrate on history before
trusting position sizes.
