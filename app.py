"""
NBES Dashboard — NUNO Bayesian Edge Strategy
Run: streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from nbes_engine import (
    Signal,
    btc_lognormal_prior,
    btc_range_prior,
    categorical_posterior,
    cpi_bucket_prior,
    cpi_normal_prior,
    evaluate,
    inv_logit,
    logit,
    posterior_from_signals,
)
from price_feed import fetch_btc_spot
from hl_outcomes import get_outcome_prices

st.set_page_config(page_title="NBES — Bayesian Edge", page_icon="📊", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def get_live_btc(_nonce: int = 0) -> dict:
    """Cached live BTC spot. ttl=30s; _nonce lets a button force a refresh."""
    return fetch_btc_spot()


@st.cache_data(ttl=20, show_spinner="Fetching live quotes from Hyperliquid…")
def get_live_quotes(keyword_items: tuple, _nonce: int = 0) -> dict:
    """Cached live HIP-4 outcome prices. ttl=20s; _nonce forces refresh.
    keyword_items is a hashable tuple of (bucket, (kw, kw, ...))."""
    keyword_map = {b: list(kws) for b, kws in keyword_items}
    return get_outcome_prices(keyword_map)

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
    ["BTC > 74032", "BTC range (3-way)", "May CPI YoY (3-way)"],
    horizontal=True,
    help="'BTC > 74032' is a yes/no bet. 'BTC range' and 'May CPI' are 3-way "
         "markets where the three outcomes add up to 100%.",
)

# ---------------------------------------------------------------------------
# prior block
# ---------------------------------------------------------------------------
cat_kind = None
if market_type == "May CPI YoY (3-way)":
    cat_kind = "cpi"
elif market_type == "BTC range (3-way)":
    cat_kind = "btc_range"
is_categorical = cat_kind is not None

# keyword maps used to match the live Hyperliquid outcome books.
# Matched against outcome name + description + parsed pipe-metadata
# (case-insensitive, ALL keywords must be present). Tuned to the real schema:
#   CPI buckets are named "Below 4.3%" / "Exactly 4.3%" / "Above 4.3%"
#   BTC markets carry pipe-meta: underlying:BTC | targetPrice:NNNNN
QUOTE_KEYWORDS = {
    "btc_range": {
        # The range market appears to be outcomes 133/134/135 (index:0/1/2).
        # Pinned by explicit outcome id since their names are generic
        # ("Recurring Named Outcome"). CONFIRM the index→bucket mapping with
        # diagnose_hl2.py — adjust the ids/order if needed.
        "below":    ["outcome:133"],   # index:0
        "in_range": ["outcome:134"],   # index:1
        "above":    ["outcome:135"],   # index:2
    },
    "cpi": {
        "below":   ["below 4.3"],
        "exactly": ["exactly 4.3"],
        "above":   ["above 4.3"],
    },
}

# market presets pulled from the live books in the screenshots
BTC_PRESETS = {
    "BTC > 74032": {"strike": 74032.0, "yes_mkt": 0.05, "hours": 11.0, "direction": "above"},
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
        direction = preset["direction"]

        # ---- live price ----
        if "px_nonce" not in st.session_state:
            st.session_state.px_nonce = 0
        live = get_live_btc(st.session_state.px_nonce)
        pc1, pc2 = st.columns([3, 1])
        if live["ok"]:
            age = (datetime.now(timezone.utc) - live["ts"]).total_seconds()
            pc1.success(f"🟢 Live BTC ${live['price']:,.0f} "
                        f"({live['source']}, {age:.0f}s ago)")
            default_spot = float(round(live["price"]))
        else:
            pc1.warning("⚠️ Live price unavailable — enter spot manually below.")
            default_spot = 73_000.0
        if pc2.button("↻ Refresh", help="Fetch the latest BTC price now."):
            st.session_state.px_nonce += 1
            st.rerun()

        c1, c2 = st.columns(2)
        spot = c1.number_input(
            "BTC spot", 1.0, 1_000_000.0, default_spot, step=50.0,
            help="Bitcoin's current price. Auto-filled from live data (refreshes "
                 "every 30s); you can override it manually.")
        rel = ">" if direction == "above" else "<"
        strike = c2.number_input(
            f"Target price ({rel})", 1.0, 1_000_000.0, preset["strike"], step=50.0,
            help=f"The price level the bet is about. YES wins if Bitcoin ends "
                 f"{direction} this number.")
        c3, c4 = st.columns(2)
        hours = c3.number_input(
            "Hours to expiry (→ Jun 1 8:00 AM)", 0.0, 168.0, preset["hours"], step=0.5,
            help="How many hours until the bet settles. Less time = less can change.")
        vol = c4.slider(
            "Annual vol σ", 0.10, 2.00, 0.55, 0.01,
            help="How jumpy Bitcoin is. Higher = bigger expected price swings. "
                 "0.55 (55%) is a typical recent value. Leave as-is if unsure.")
        p_above = btc_lognormal_prior(spot, strike, hours, vol)
        prior = p_above if direction == "above" else 1.0 - p_above
        st.caption(f"Lognormal P(BTC {rel} {strike:,.0f} in {hours:.1f}h): **{prior:.3f}**")

    with right:
        st.subheader("2 · Market book")
        st.caption("👉 The current price on Hyperliquid. Remember: a price of "
                   "0.05 means the market thinks there's a 5% chance.")

        if "quote_nonce" not in st.session_state:
            st.session_state.quote_nonce = 0
        bqc1, bqc2 = st.columns([3, 1])
        use_live_bin = bqc1.checkbox(
            "Use live Hyperliquid quote", value=False,
            help="Fetch the current YES price for this contract from Hyperliquid.")
        if bqc2.button("↻ Refresh quote",
                       help="Re-fetch the latest contract price now."):
            st.session_state.quote_nonce += 1
            st.rerun()

        live_yes = None
        if use_live_bin:
            kw_items = (("yes", ("BTC", "74032")),)
            q = get_live_quotes(kw_items, st.session_state.quote_nonce)
            if q["ok"] and q["prices"].get("yes") is not None:
                live_yes = float(q["prices"]["yes"])
                age = (datetime.now(timezone.utc) - q["ts"]).total_seconds()
                st.success(f"🟢 Live YES mid ${live_yes:.3f} "
                           f"({q['resolved'].get('yes') or '?'}, {age:.0f}s ago)")
            else:
                st.warning(f"⚠️ Couldn't match this market live "
                           f"({q.get('error') or 'no price'}). Using manual entry.")

        base = live_yes if live_yes is not None else preset["yes_mkt"]
        if live_yes is None:
            st.caption(f"Screenshot shows YES ({direction}) ≈ **{preset['yes_mkt']:.0%}**. "
                       "Enter your observed bid/ask.")
        c1, c2 = st.columns(2)
        yes_bid = c1.number_input(
            "YES bid", 0.0, 1.0, max(0.0, base - 0.01), step=0.01,
            help="Highest price someone will pay you for a YES contract (where you "
                 "could sell).")
        yes_ask = c2.number_input(
            "YES ask", 0.0, 1.0, min(1.0, base + 0.01), step=0.01,
            help="Lowest price you can buy a YES contract for (where you could buy).")
        mid = 0.5 * (yes_bid + yes_ask)
        spread = yes_ask - yes_bid
        st.caption(f"Mid **{mid:.3f}** · spread **{spread:.3f}** "
                   f"({spread*100:.1f}¢) · implied prob {mid:.1%}")
        if spread > 2 * edge_threshold:
            st.warning("Spread exceeds 2× your edge threshold — likely untradeable.")

# ---------------------------------------------------------------------------
# CATEGORICAL PATH (CPI 3-way  OR  BTC range 3-way)
# ---------------------------------------------------------------------------
else:
    left, right = st.columns([1, 1])

    if cat_kind == "cpi":
        bucket_keys = ["below", "exactly", "above"]
        bucket_labels = {"below": "Below 4.3", "exactly": "Exactly 4.3",
                         "above": "Above 4.3"}
        default_prices = {"below": 0.45, "exactly": 0.41, "above": 0.12}
        with left:
            st.subheader("1 · Prior — 3-way bucket model")
            st.caption("👉 This market has THREE possible answers, not yes/no. The "
                       "tool splits your estimate across all three so they add to 100%.")
            st.caption("Market rounds to one decimal around a center (4.3%). "
                       "Buckets: below 4.25 / [4.25,4.35) / ≥4.35.")
            c1, c2 = st.columns(2)
            consensus = c1.number_input(
                "Consensus YoY %", -5.0, 20.0, 4.28, step=0.01,
                help="The average forecast for the inflation number, from analysts.")
            dispersion = c2.number_input(
                "Forecast dispersion (std)", 0.01, 2.0, 0.08, step=0.01,
                help="How much forecasters disagree. Higher = more uncertainty.")
            c3, c4 = st.columns(2)
            center = c3.number_input(
                "Bucket center %", -5.0, 20.0, 4.30, step=0.05,
                help="The middle value the market rounds to (here 4.3%).")
            half_width = c4.number_input(
                "Bucket half-width", 0.01, 0.50, 0.05, step=0.01,
                help="How wide the middle bucket is. 0.05 means 'exactly 4.3' "
                     "covers 4.25 up to 4.35.")
            hours = st.number_input(
                "Hours to settlement (→ Jun 10 BLS)", 0.0, 2000.0, 240.0, step=12.0,
                help="Hours until the official inflation data is released.")
            prior_buckets = cpi_bucket_prior(consensus, dispersion, center, half_width)
            st.caption(" · ".join(f"{bucket_labels[k]} **{prior_buckets[k]:.3f}**"
                                  for k in bucket_keys))

    else:  # btc_range
        bucket_keys = ["below", "in_range", "above"]
        bucket_labels = {"below": "Below 72551", "in_range": "72551–75512",
                         "above": "Above 75512"}
        default_prices = {"below": 0.08, "in_range": 0.93, "above": 0.01}
        with left:
            st.subheader("1 · Prior — 3-way range model")
            st.caption("👉 This market has THREE outcomes: Bitcoin ends below the "
                       "range, inside it, or above it. They add up to 100%.")
            # live price
            if "px_nonce" not in st.session_state:
                st.session_state.px_nonce = 0
            live = get_live_btc(st.session_state.px_nonce)
            pc1, pc2 = st.columns([3, 1])
            if live["ok"]:
                age = (datetime.now(timezone.utc) - live["ts"]).total_seconds()
                pc1.success(f"🟢 Live BTC ${live['price']:,.0f} "
                            f"({live['source']}, {age:.0f}s ago)")
                default_spot = float(round(live["price"]))
            else:
                pc1.warning("⚠️ Live price unavailable — enter spot manually.")
                default_spot = 73_800.0
            if pc2.button("↻ Refresh", help="Fetch the latest BTC price now."):
                st.session_state.px_nonce += 1
                st.rerun()
            c1, c2 = st.columns(2)
            spot = c1.number_input(
                "BTC spot", 1.0, 1_000_000.0, default_spot, step=50.0,
                help="Bitcoin's current price. Auto-filled from live data.")
            vol = c2.slider(
                "Annual vol σ", 0.10, 2.00, 0.55, 0.01,
                help="How jumpy Bitcoin is. 0.55 (55%) is typical. Leave as-is "
                     "if unsure.")
            c3, c4 = st.columns(2)
            lower = c3.number_input(
                "Range lower", 1.0, 1_000_000.0, 72_551.0, step=50.0,
                help="Bottom of the range. Below this = 'below' outcome.")
            upper = c4.number_input(
                "Range upper", 1.0, 1_000_000.0, 75_512.0, step=50.0,
                help="Top of the range. Above this = 'above' outcome.")
            hours = st.number_input(
                "Hours to expiry (→ Jun 1 8:00 AM)", 0.0, 168.0, 11.0, step=0.5,
                help="Hours until the bet settles.")
            prior_buckets = btc_range_prior(spot, lower, upper, hours, vol)
            st.caption(" · ".join(f"{bucket_labels[k]} **{prior_buckets[k]:.3f}**"
                                  for k in bucket_keys))

    with right:
        st.subheader("2 · Market book (per bucket)")
        st.caption("👉 The price of each of the three outcomes. Pull them live "
                   "from Hyperliquid, or enter manually.")

        if "quote_nonce" not in st.session_state:
            st.session_state.quote_nonce = 0
        kw_items = tuple((b, tuple(kws))
                         for b, kws in QUOTE_KEYWORDS[cat_kind].items())

        qc1, qc2 = st.columns([3, 1])
        use_live = qc1.checkbox(
            "Use live Hyperliquid quotes", value=False,
            help="Fetch current contract prices from Hyperliquid's HIP-4 books. "
                 "If a market can't be matched, that price falls back to manual.")
        if qc2.button("↻ Refresh quotes",
                      help="Re-fetch the latest contract prices now."):
            st.session_state.quote_nonce += 1
            st.rerun()

        live_prices = {}
        if use_live:
            q = get_live_quotes(kw_items, st.session_state.quote_nonce)
            if q["ok"]:
                age = (datetime.now(timezone.utc) - q["ts"]).total_seconds()
                got = {k: v for k, v in q["prices"].items() if v is not None}
                st.success(f"🟢 Live: {len(got)}/{len(bucket_keys)} outcomes "
                           f"matched ({age:.0f}s ago)")
                live_prices = got
                # show what resolved so the user can verify the match
                res_txt = " · ".join(
                    f"{bucket_labels[k]}→{q['resolved'].get(k) or '—'}"
                    for k in bucket_keys)
                st.caption(f"Resolved coins: {res_txt}")
                missing = [bucket_labels[k] for k in bucket_keys
                           if k not in live_prices]
                if missing:
                    st.warning("Couldn't match: " + ", ".join(missing)
                               + " — enter these manually below.")
            else:
                st.warning("⚠️ Live quotes unavailable. Using manual entry.")
                with st.expander("🔍 Why? (diagnostic)"):
                    st.code(q.get("error") or "unknown")
                    if q.get("diag"):
                        for line in q["diag"]:
                            st.text(line)
                    st.caption("If markets matched but prices are blank, the coin "
                               "encoding didn't resolve in allMids. Run "
                               "`diagnose_hl.py` locally and share the output, or "
                               "adjust QUOTE_KEYWORDS in app.py.")

        mkt_buckets = {}
        for k in bucket_keys:
            dflt = float(live_prices.get(k, default_prices[k]))
            mkt_buckets[k] = st.number_input(
                f"{bucket_labels[k]} price", 0.0, 1.0, dflt, step=0.01,
                key=f"price_{cat_kind}_{k}",
                help=f"Market price for the '{bucket_labels[k]}' outcome "
                     f"(0.50 = 50% implied chance). "
                     f"{'Auto-filled from Hyperliquid.' if k in live_prices else ''}")
        book_sum = sum(mkt_buckets.values())
        over = book_sum - 1
        st.caption(f"Book sums to **{book_sum:.2f}** "
                   f"({'overround +' + format(over*100, '.0f') + '¢' if over > 0 else 'underround ' + format(over*100, '.0f') + '¢'}).")

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
               "outcome (which one it favours). Positive = makes that outcome "
               "more likely.")
    if cat_kind == "cpi":
        default_signals = pd.DataFrame([
            {"signal": "Cleveland nowcast revision", "llr": 0.00, "weight": 0.70,
             "bucket": "above", "active": True},
            {"signal": "DXY / 2Y move", "llr": 0.00, "weight": 0.50,
             "bucket": "above", "active": True},
            {"signal": "Prior-month surprise carry", "llr": 0.00, "weight": 0.40,
             "bucket": "below", "active": False},
        ])
    else:  # btc_range
        default_signals = pd.DataFrame([
            {"signal": "Spot drift vs range", "llr": 0.00, "weight": 0.70,
             "bucket": "in_range", "active": True},
            {"signal": "Funding skew (Loris 34x)", "llr": 0.00, "weight": 0.50,
             "bucket": "above", "active": True},
            {"signal": "Realized-vol spike", "llr": 0.00, "weight": 0.40,
             "bucket": "below", "active": False},
        ])
    edited = st.data_editor(
        default_signals, num_rows="dynamic", width='stretch',
        column_config={
            "signal": st.column_config.TextColumn("Signal", width="large"),
            "llr": st.column_config.NumberColumn("Log-ev", step=0.05, format="%.2f"),
            "weight": st.column_config.NumberColumn("Weight", min_value=0.0,
                                                    max_value=1.0, step=0.05, format="%.2f"),
            "bucket": st.column_config.SelectboxColumn("Favours", options=bucket_keys),
            "active": st.column_config.CheckboxColumn("On"),
        },
        key=f"signals_cat_{cat_kind}",
    )
    deltas = {k: 0.0 for k in bucket_keys}
    for _, row in edited.iterrows():
        if bool(row["active"]) and row["bucket"] in deltas:
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
    for k in bucket_keys:
        res_k = evaluate(
            post_buckets[k], mkt_buckets[k] - 0.005, mkt_buckets[k] + 0.005,
            bankroll=bankroll, fee=fee, edge_threshold=edge_threshold,
            edge_se=edge_se, kelly_lambda=kelly_lambda, max_frac=max_frac,
            hours_to_expiry=hours, min_hours=min_hours,
        )
        rows.append({
            "outcome": bucket_labels[k], "prior": round(prior_buckets[k], 3),
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
                    f"'{best['outcome']}'</span> · edge {best['edge']:+.3f} · "
                    f"${best['stake $']:,.0f}", unsafe_allow_html=True)
    else:
        st.markdown("### Verdict: <span class='verdict-flat'>NO TRADE</span> "
                    "— no outcome clears the gate.", unsafe_allow_html=True)
    st.caption("Each outcome evaluated independently against its own price. "
               "Note the book overround — true edges are smaller than raw gaps.")
    # expose the 'above' bucket for the shared waterfall/sensitivity sections
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
