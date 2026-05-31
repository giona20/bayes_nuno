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
- `app.py` — Streamlit UI.
- `requirements.txt` — deps.

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
