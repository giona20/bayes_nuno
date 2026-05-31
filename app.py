"""
NBES Dashboard — NUNO Bayesian Edge Strategy
Run: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nbes_engine import (
    Signal,
    btc_lognormal_prior,
    categorical_posterior,
    cpi_bucket_prior,
    cpi_normal_prior,
    evaluate,
    inv_logit,
    logit,
    posterior_from_signals,
)

st.set_page_config(page_title="NBES — Bayesian Edge", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stMetric { background:#0e1117; border:1px solid #1f2a37; border-radius:10px;
                  padding:12px 14px; }
      div[data-testid="stMetricValue"] { font-size:1.5rem; }
      .verdict-buy  { color:#1bd97b; font-weight:700; }
      .verdict-no   { color:#f0506e; font-weight:700; }
      .verdict-flat { color:#9aa4b2; font-weight:700; }
      .small { color:#9aa4b2; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NBES — Bayesian Edge Dashboard")
st.caption("Prior → sequential logit updating → edge vs market → variance-adjusted Kelly. "
           "HIP-4 outcome contracts settle 0/1; max loss = stake.")

st.info(
    "**New here? Read this first.** This tool helps you spot when a betting "
    "market's price looks *wrong*. You make your own estimate of how likely "
    "something is (e.g. \"will Bitcoin be above $74,032 tomorrow morning?\"), "
    "the tool compares it to the price the market is charging, and if there's a "
    "big enough gap in your favour it suggests how much to bet. "
    "Open the guide below before touching anything.",
    icon="👋",
)

with st.expander("📖 How to use this — plain-language guide (click to open)"):
    st.markdown(
        """
**What is this market?**
On Hyperliquid you can bet on yes/no questions ("outcome contracts"). A contract
costs between **\\$0 and \\$1**. If you're right it pays **\\$1**; if you're wrong it
pays **\\$0**. A price of **\\$0.60 means the market thinks there's a 60% chance**
it happens. That's the key idea: **price = the market's probability.**

**Where's the money?**
If *you* think the real chance is 75% but the market is only charging 60%
(\\$0.60), you're buying something worth more than you pay. Over many bets like
that, you win. The gap between your estimate and the price is your **"edge."**

**How the tool builds your estimate (3 steps):**
1. **Prior** — a first-guess probability from a simple math model (for Bitcoin,
   based on today's price, the target, time left, and how jumpy the market is).
   *You don't have to calculate this — the tool does.*
2. **Evidence** — you add clues that push the guess up or down (funding rates,
   momentum, etc.). Each clue has a **strength** (how informative) and a
   **direction** (does it make YES more or less likely).
3. **Posterior** — your final, updated estimate after combining the prior with
   all the evidence.

**Then the tool tells you:**
- **Edge** — how far your estimate is from the market price. Bigger = better.
- **Verdict** — *Buy YES*, *Buy NO*, or *No Trade* (if the gap is too small to
  be worth the risk).
- **Stake** — a suggested bet size that grows with your edge and shrinks when
  you're unsure. It's deliberately cautious.

**The dials on the left (you can leave these alone at first):**
- **Bankroll** — total money you're willing to risk overall.
- **Kelly fraction** — how aggressive to bet. **0.25 (quarter) is the safe
  default.** Higher = bigger swings. Don't use 1.0 unless you know why.
- **Min edge to trade** — ignore tiny gaps; only bet when the gap is clearly
  big enough.

**Three honest warnings:**
- The "Evidence" numbers are **placeholders right now.** Until they're calibrated
  on real history, treat the suggested bet sizes as a *demonstration*, not advice.
- These markets are **thin** (little money in them) — your own bet can move the
  price, and you might not be able to sell when you want.
- This is an **educational tool, not financial advice.** You can lose your whole
  stake on any contract.
        """
    )

# ---------------------------------------------------------------------------
# sidebar: bankroll + risk config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Risk config")
    st.caption("Hover the **?** on each item for a plain explanation. "
               "The defaults are sensible — beginners can leave them as-is.")
    bankroll = st.number_input(
        "Bankroll (USDC)", 100.0, 10_000_000.0, 10_000.0, step=500.0,
        help="Total money you're willing to risk across all bets. Bet sizes are "
             "calculated as a fraction of this.")
    kelly_lambda = st.select_slider(
        "Kelly fraction (λ)", options=[0.10, 0.25, 0.50, 1.00], value=0.25,
        help="How aggressive to bet. 0.10 = very cautious, 0.25 = recommended "
             "default, 1.00 = full aggression (risky, big swings — not advised).")
    max_frac = st.slider(
        "Max bankroll per market", 0.01, 0.50, 0.10, 0.01,
        help="Hard cap on any single bet, e.g. 0.10 = never risk more than 10% "
             "of your bankroll on one contract, no matter how good it looks.")
    edge_threshold = st.slider(
        "Min edge to trade (¢)", 0.005, 0.15, 0.03, 0.005,
        help="Ignore small gaps. 0.03 means your estimate must beat the market "
             "price by at least 3 cents before the tool suggests a bet.")
    edge_se = st.slider(
        "Edge std error (model uncertainty)", 0.005, 0.10, 0.02, 0.005,
        help="How unsure you are of your own estimate. Higher = the tool demands "
             "a bigger edge and bets smaller. Keep it honest.")
    fee = st.number_input(
        "Per-side fee", 0.0, 0.01, 0.0005, step=0.0001, format="%.4f",
        help="Trading fee charged each time you buy or sell, as a fraction of "
             "price. Eats into your edge.")
    min_hours = st.number_input(
        "Min hours to expiry", 0.0, 72.0, 0.5, step=0.5,
        help="Don't trade if the contract settles too soon to be worth it.")
    st.markdown("<span class='small'>Thin books — you may be the liquidity. "
                "Keep λ low and threshold above 2× half-spread.</span>",
                unsafe_allow_html=True)

market_type = st.radio(
    "Market (live HIP-4 books, Jun 1 8:00 AM / Jun 10 CPI)",
    ["BTC > 74032 (binary)", "BTC > 72551 (binary)", "May CPI YoY (3-way)"],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# prior block
# ---------------------------------------------------------------------------
is_categorical = market_type == "May CPI YoY (3-way)"

# market presets pulled from the live books in the screenshot
BTC_PRESETS = {
    "BTC > 74032 (binary)": {"strike": 74032.0, "yes_mkt": 0.05, "hours": 11.0},
    "BTC > 72551 (binary)": {"strike": 72551.0, "yes_mkt": 0.06, "hours": 11.0},
}

# ---------------------------------------------------------------------------
# BINARY PATH (BTC markets)
# ---------------------------------------------------------------------------
if not is_categorical:
    preset = BTC_PRESETS[market_type]
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1 · Prior (independent of the book)")
        st.caption("👉 The tool's first-guess probability, before looking at the "
                   "market price. Just enter today's Bitcoin price and the target.")
        c1, c2 = st.columns(2)
        spot = c1.number_input("BTC spot", 1.0, 1_000_000.0, 73_000.0, step=50.0)
        strike = c2.number_input("Strike (>)", 1.0, 1_000_000.0,
                                 preset["strike"], step=50.0)
        c3, c4 = st.columns(2)
        hours = c3.number_input("Hours to expiry (→ Jun 1 8:00 AM)", 0.0, 168.0,
                                preset["hours"], step=0.5)
        vol = c4.slider("Annual vol σ", 0.10, 2.00, 0.55, 0.01)
        prior = btc_lognormal_prior(spot, strike, hours, vol)
        st.caption(f"Lognormal P(BTC > {strike:,.0f} in {hours:.1f}h): **{prior:.3f}**")

    with right:
        st.subheader("2 · Market book")
        st.caption("👉 The current price on Hyperliquid. Remember: a price of "
                   "0.05 means the market thinks there's a 5% chance.")
        st.caption(f"Screenshot shows YES (above) ≈ **{preset['yes_mkt']:.0%}**. "
                   "Enter your observed bid/ask.")
        c1, c2 = st.columns(2)
        yes_bid = c1.number_input("YES bid", 0.0, 1.0,
                                  max(0.0, preset["yes_mkt"] - 0.01), step=0.01)
        yes_ask = c2.number_input("YES ask", 0.0, 1.0,
                                  preset["yes_mkt"] + 0.01, step=0.01)
        mid = 0.5 * (yes_bid + yes_ask)
        spread = yes_ask - yes_bid
        st.caption(f"Mid **{mid:.3f}** · spread **{spread:.3f}** "
                   f"({spread*100:.1f}¢) · implied prob {mid:.1%}")
        if spread > 2 * edge_threshold:
            st.warning("Spread exceeds 2× your edge threshold — likely untradeable.")

# ---------------------------------------------------------------------------
# CATEGORICAL PATH (CPI 3-way)
# ---------------------------------------------------------------------------
else:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("1 · Prior — 3-way bucket model")
        st.caption("👉 This market has THREE possible answers, not yes/no. The "
                   "tool splits your estimate across all three so they add to 100%.")
        st.caption("Market rounds to one decimal around a center (4.3%). "
                   "Buckets: below 4.25 / [4.25,4.35) / ≥4.35.")
        c1, c2 = st.columns(2)
        consensus = c1.number_input("Consensus YoY %", -5.0, 20.0, 4.28, step=0.01)
        dispersion = c2.number_input("Forecast dispersion (std)", 0.01, 2.0,
                                     0.08, step=0.01)
        c3, c4 = st.columns(2)
        center = c3.number_input("Bucket center %", -5.0, 20.0, 4.30, step=0.05)
        half_width = c4.number_input("Bucket half-width", 0.01, 0.50,
                                     0.05, step=0.01)
        hours = st.number_input("Hours to settlement (→ Jun 10 BLS)",
                                0.0, 2000.0, 240.0, step=12.0)
        prior_buckets = cpi_bucket_prior(consensus, dispersion, center, half_width)
        st.caption(" · ".join(f"{k} **{v:.3f}**" for k, v in prior_buckets.items()))

    with right:
        st.subheader("2 · Market book (per bucket)")
        st.caption("👉 The price of each of the three answers. Enter what the "
                   "market is charging for each.")
        st.caption("Prices from screenshot: below 45% · exactly 41% · above 12%.")
        below_mkt = st.number_input("Below 4.3 price", 0.0, 1.0, 0.45, step=0.01)
        exactly_mkt = st.number_input("Exactly 4.3 price", 0.0, 1.0, 0.41, step=0.01)
        above_mkt = st.number_input("Above 4.3 price", 0.0, 1.0, 0.12, step=0.01)
        book_sum = below_mkt + exactly_mkt + above_mkt
        st.caption(f"Book sums to **{book_sum:.2f}** "
                   f"({'overround ' + format((book_sum-1)*100, '.0f') + '¢' if book_sum > 1 else 'underround'}).")
        mkt_buckets = {"below": below_mkt, "exactly": exactly_mkt, "above": above_mkt}

# ---------------------------------------------------------------------------
# evidence / signals (shared)
# ---------------------------------------------------------------------------
st.subheader("3 · Evidence — log-likelihood ratios (calibrate from history)")
st.caption("👉 Clues that adjust your guess. Each row is one clue: a **strength** "
           "(how much it matters) and a **direction** (does it make the answer more "
           "or less likely). Leave the defaults if unsure. ⚠️ These are placeholders "
           "until calibrated on real data — don't trust the bet sizes yet.")
if is_categorical:
    st.caption("For a 3-way market, evidence is log-evidence added to a chosen "
               "bucket (which outcome it favours). Positive = makes that bucket "
               "more likely. Wire to nowcast revisions / DXY / 2Y moves.")
    default_signals = pd.DataFrame([
        {"signal": "Cleveland nowcast revision", "llr": 0.00, "weight": 0.70,
         "bucket": "above", "active": True},
        {"signal": "DXY / 2Y move",              "llr": 0.00, "weight": 0.50,
         "bucket": "above", "active": True},
        {"signal": "Prior-month surprise carry", "llr": 0.00, "weight": 0.40,
         "bucket": "below", "active": False},
    ])
    edited = st.data_editor(
        default_signals, num_rows="dynamic", width='stretch',
        column_config={
            "signal": st.column_config.TextColumn("Signal", width="large"),
            "llr": st.column_config.NumberColumn("Log-ev", step=0.05, format="%.2f"),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0,
                                                    max_value=1.0, step=0.05, format="%.2f"),
            "bucket": st.column_config.SelectboxColumn(
                "Favours", options=["below", "exactly", "above"]),
            "active": st.column_config.CheckboxColumn("On"),
        },
        key="signals_cat",
    )
    deltas = {"below": 0.0, "exactly": 0.0, "above": 0.0}
    for _, row in edited.iterrows():
        if bool(row["active"]):
            deltas[row["bucket"]] += float(row["weight"]) * float(row["llr"])
    post_buckets = categorical_posterior(prior_buckets, deltas)
else:
    st.caption("LLR = ln P(E|YES)/P(E|NO). Positive favours YES (BTC above strike). "
               "Weight ∈ [0,1] decays with age. Wire to Loris funding / aggr.trade.")
    default_signals = pd.DataFrame([
        {"signal": "Spot momentum vs strike", "llr": 0.30, "weight": 0.80, "active": True},
        {"signal": "Funding skew (Loris 34x)", "llr": 0.15, "weight": 0.60, "active": True},
        {"signal": "OI delta direction",       "llr": -0.10, "weight": 0.50, "active": True},
        {"signal": "Order-book imbalance",      "llr": 0.20, "weight": 0.70, "active": True},
        {"signal": "ETF net flow z-score",      "llr": 0.10, "weight": 0.30, "active": False},
    ])
    edited = st.data_editor(
        default_signals, num_rows="dynamic", width='stretch',
        column_config={
            "signal": st.column_config.TextColumn("Signal", width="large"),
            "llr": st.column_config.NumberColumn("LLR", step=0.05, format="%.2f"),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0,
                                                    max_value=1.0, step=0.05, format="%.2f"),
            "active": st.column_config.CheckboxColumn("On"),
        },
        key="signals_bin",
    )
    signals = [
        Signal(name=row["signal"], llr=float(row["llr"]),
               weight=float(row["weight"]), active=bool(row["active"]))
        for _, row in edited.iterrows()
    ]
    posterior = posterior_from_signals(prior, signals)

# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------
st.subheader("4 · Decision")
st.caption("👉 The verdict. **Buy YES** = bet it happens. **Buy NO** = bet it "
           "doesn't. **No Trade** = the gap is too small to be worth it. "
           "**Stake** is the suggested bet size.")

if is_categorical:
    rows = []
    for k in ["below", "exactly", "above"]:
        res_k = evaluate(
            post_buckets[k], mkt_buckets[k] - 0.005, mkt_buckets[k] + 0.005,
            bankroll=bankroll, fee=fee, edge_threshold=edge_threshold,
            edge_se=edge_se, kelly_lambda=kelly_lambda, max_frac=max_frac,
            hours_to_expiry=hours, min_hours=min_hours,
        )
        rows.append({
            "bucket": k, "prior": round(prior_buckets[k], 3),
            "posterior": round(post_buckets[k], 3),
            "market": round(mkt_buckets[k], 3),
            "edge": round(res_k.edge, 3), "side": res_k.side,
            "stake $": round(res_k.stake, 0),
        })
    dec = pd.DataFrame(rows)
    st.dataframe(dec, width='stretch', hide_index=True)
    fires = dec[dec["side"] != "NO_TRADE"]
    if len(fires):
        best = fires.loc[fires["edge"].abs().idxmax()]
        st.markdown(f"### Best edge: <span class='verdict-buy'>{best['side'].replace('_',' ')} "
                    f"'{best['bucket']}'</span> · edge {best['edge']:+.3f} · "
                    f"${best['stake $']:,.0f}", unsafe_allow_html=True)
    else:
        st.markdown("### Verdict: <span class='verdict-flat'>NO TRADE</span> "
                    "— no bucket clears the gate.", unsafe_allow_html=True)
    st.caption("Each bucket evaluated independently against its own price. "
               "Note the book overround — true edges are smaller than raw gaps.")
    # for the shared waterfall section below, expose the 'above' bucket
    prior, posterior, mid = prior_buckets["above"], post_buckets["above"], mkt_buckets["above"]
    signals = []
else:
    res = evaluate(
        posterior, yes_bid, yes_ask,
        bankroll=bankroll, fee=fee,
        edge_threshold=edge_threshold, edge_se=edge_se,
        kelly_lambda=kelly_lambda, max_frac=max_frac,
        hours_to_expiry=hours, min_hours=min_hours,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Prior", f"{prior:.3f}")
    m2.metric("Posterior", f"{posterior:.3f}", f"{posterior - prior:+.3f}")
    m3.metric("Market mid", f"{mid:.3f}")
    m4.metric("Edge", f"{res.edge:+.3f}")
    m5.metric("Stake", f"${res.stake:,.0f}", f"{res.kelly_used:.1%} BR")

    verdict_cls = {"BUY_YES": "verdict-buy", "BUY_NO": "verdict-no",
                   "NO_TRADE": "verdict-flat"}[res.side]
    st.markdown(f"### Verdict: <span class='{verdict_cls}'>{res.side.replace('_',' ')}</span>",
                unsafe_allow_html=True)
    for r in res.reasons:
        st.write(f"• {r}")
    st.caption(f"Full Kelly {res.kelly_full:.1%} → used {res.kelly_used:.1%} "
               f"(λ={kelly_lambda}, variance-shrunk). Exec price {res.exec_price:.3f}.")

# ---------------------------------------------------------------------------
# logit waterfall
# ---------------------------------------------------------------------------
st.subheader("5 · Logit waterfall — how evidence moved the belief")
st.caption("👉 A visual of how each clue nudged your estimate up or down, "
           "starting from the prior and ending at your final probability.")
acc = logit(prior)
rows = [{"step": "Prior", "logit": acc, "prob": prior}]
for s in signals:
    if s.active and s.weight != 0:
        acc += s.contribution
        rows.append({"step": s.name, "logit": acc, "prob": inv_logit(acc),
                     "Δlogit": s.contribution})
wf = pd.DataFrame(rows)
cc1, cc2 = st.columns([2, 1])
cc1.line_chart(wf.set_index("step")["prob"], height=260)
cc2.dataframe(wf.assign(prob=lambda d: d["prob"].round(3),
                        logit=lambda d: d["logit"].round(3)),
              width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# edge sensitivity to posterior (helps see threshold zone)
# ---------------------------------------------------------------------------
st.subheader("6 · Edge sensitivity")
st.caption("👉 Shows how much edge you'd have at different probability estimates. "
           "Where the line rises above the flat threshold line, a bet becomes worth it.")
if is_categorical:
    ask_ref = mkt_buckets["above"] + 0.005
    bid_ref = mkt_buckets["above"] - 0.005
    sens_label = "vs 'above' bucket price"
else:
    ask_ref, bid_ref = yes_ask, yes_bid
    sens_label = "vs YES book"
qs = np.linspace(0.01, 0.99, 99)
edge_yes = qs - (ask_ref + fee)
edge_no = (1 - qs) - ((1 - bid_ref) + fee)
best = np.maximum(edge_yes, edge_no)
sens = pd.DataFrame({"posterior": qs, "best edge": best,
                     "threshold": edge_threshold}).set_index("posterior")
st.line_chart(sens, height=240)
st.caption(f"Best-edge curve {sens_label}; where it clears the threshold a trade fires. "
           f"Current posterior {posterior:.3f}.")

st.divider()
st.caption("NBES v0.1 · Educational tool for modelling expected value. Not financial advice; "
           "calibrate LLRs on real history before trusting sizing. Thin HIP-4 books carry "
           "execution and settlement risk.")
