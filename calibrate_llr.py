"""
NBES LLR calibration — turn historical data into real log-likelihood ratios.

WHAT THIS DOES
For each signal (funding skew, OI change, spot momentum), it answers the only
question that makes an LLR real:

    "When this signal was in state X, how often did BTC actually finish ABOVE
     the daily 06:00 UTC reference vs BELOW?"

The log of that ratio IS the LLR. We bin each signal into buckets, count YES vs
NO outcomes per bucket from real history, and write the LLRs to calibration.json
which the app loads. No hand-picked numbers anywhere.

DATA (Binance USDⓈ-M futures, public, no API key)
  - /fapi/v1/klines              hourly BTCUSDT OHLC  -> outcomes + momentum
  - /fapi/v1/fundingRate         historical funding   -> funding signal
  - /futures/data/openInterestHist  historical OI     -> OI-change signal

USAGE (run locally; Streamlit Cloud can also run it on a schedule)
    python calibrate_llr.py                # ~last 720 days, hourly
    python calibrate_llr.py --days 365
It writes calibration.json next to this file. Re-run anytime to incorporate
new data — the app picks up the file on its next load.

NOTE ON HONESTY
LLRs are only as good as the sample. The script records the sample size per
bucket; buckets with too few observations are shrunk toward 0 (no signal)
rather than trusted, so a thin bucket can't fabricate a strong LLR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

FAPI = "https://fapi.binance.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (NBES calibration)"}
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

# settlement reference hour (UTC) for the daily BTC market
SETTLE_HOUR_UTC = 6
# A bin's LLR is only fully trusted once it has FULL_TRUST_OBS observations.
# Below that it is linearly shrunk toward 0 (no signal). Below MIN_TRUST_OBS it
# contributes nothing at all. This is what auto-suppresses a data-starved signal
# like OI (~120 obs/bin from 30d history) until it has accumulated enough
# history to be meaningful, while leaving momentum (3,400+/bin) at full strength.
MIN_TRUST_OBS = 250      # below this, LLR forced to 0 (ignored)
FULL_TRUST_OBS = 600     # at/above this, LLR used at full measured strength
# cap on |LLR| so one noisy bucket can't dominate
LLR_CAP = 1.5


# ---------------------------------------------------------------------------
# HTTP with pagination + retry
# ---------------------------------------------------------------------------

def _get(path: str, params: dict) -> list | dict:
    url = f"{FAPI}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Hourly OHLC, paginated (Binance caps 1500/req)."""
    out, cur = [], start_ms
    while cur < end_ms:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": 1500})
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        nxt = last_open + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1500:
            break
        time.sleep(0.2)
    return out


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list:
    out, cur = [], start_ms
    while cur < end_ms:
        batch = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        out.extend(batch)
        nxt = batch[-1]["fundingTime"] + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return out


def fetch_oi(symbol: str, period: str, start_ms: int, end_ms: int) -> list:
    """openInterestHist only retains ~30 days and rejects older startTimes with
    a 400. Clamp the window to the last 30 days, and never let an OI failure
    abort calibration — funding + momentum still calibrate without it."""
    thirty_days_ms = 30 * 86400_000
    clamped_start = max(start_ms, end_ms - thirty_days_ms + 3600_000)
    out, cur = [], clamped_start
    while cur < end_ms:
        try:
            batch = _get("/futures/data/openInterestHist", {
                "symbol": symbol, "period": period,
                "startTime": cur, "endTime": end_ms, "limit": 500})
        except Exception as e:  # noqa: BLE001
            print(f"  (OI fetch skipped: {type(e).__name__} — "
                  f"funding+momentum will still calibrate)")
            break
        if not batch:
            break
        out.extend(batch)
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 500:
            break
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# build the labelled dataset: for each hour, signal states + the eventual
# outcome (did price at the next 06:00 UTC exceed the price at signal time?)
# ---------------------------------------------------------------------------

def _next_settle_ms(ts_ms: int) -> int:
    from datetime import timedelta
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    settle = dt.replace(hour=SETTLE_HOUR_UTC, minute=0, second=0, microsecond=0)
    if dt.hour >= SETTLE_HOUR_UTC:
        settle = settle + timedelta(days=1)
    return int(settle.timestamp() * 1000)


def build_samples(klines: list, funding: list, oi: list) -> list[dict]:
    """Return per-hour samples: {ts, close, fwd_outcome, funding, oi_chg, mom}."""
    # index closes by hour open-time
    closes = {int(k[0]): float(k[4]) for k in klines}
    times = sorted(closes)
    if not times:
        return []
    # funding: last known rate at/just before each hour
    fund_sorted = sorted(((int(f["fundingTime"]), float(f["fundingRate"]))
                          for f in funding))
    # oi: open interest value over time
    oi_sorted = sorted(((int(o["timestamp"]), float(o["sumOpenInterest"]))
                        for o in oi))

    def _last_at(seq, ts):
        # binary-ish scan; seq small enough
        val = None
        for t, v in seq:
            if t <= ts:
                val = v
            else:
                break
        return val

    samples = []
    for i, ts in enumerate(times):
        close = closes[ts]
        settle_ms = _next_settle_ms(ts)
        # find the close at/after settle
        settle_close = None
        for t in times:
            if t >= settle_ms:
                settle_close = closes[t]
                break
        if settle_close is None:
            continue  # no future settle in data
        outcome = 1 if settle_close > close else 0
        funding_now = _last_at(fund_sorted, ts)
        oi_now = _last_at(oi_sorted, ts)
        oi_prev = _last_at(oi_sorted, ts - 3600_000)
        oi_chg = ((oi_now - oi_prev) / oi_prev) if (oi_now and oi_prev) else None
        # momentum: 6h return up to now
        prev6 = closes.get(ts - 6 * 3600_000)
        mom = ((close - prev6) / prev6) if prev6 else None
        samples.append({"ts": ts, "close": close, "outcome": outcome,
                        "funding": funding_now, "oi_chg": oi_chg, "mom": mom})
    return samples


# ---------------------------------------------------------------------------
# LLR computation per signal
# ---------------------------------------------------------------------------

def _llr_for_bins(samples, key, edges) -> dict:
    """Bin samples by signal value; compute LLR = ln(P(state|YES)/P(state|NO))
    per bin via outcome frequencies. Returns {bin_label: {llr, n, p_yes}}."""
    n_yes = sum(1 for s in samples if s["outcome"] == 1 and s[key] is not None)
    n_no = sum(1 for s in samples if s["outcome"] == 0 and s[key] is not None)
    if n_yes == 0 or n_no == 0:
        return {}
    out = {}
    labels = _bin_labels(edges)
    for lo, hi, lab in labels:
        in_bin = [s for s in samples
                  if s[key] is not None and lo <= s[key] < hi]
        n = len(in_bin)
        yes = sum(1 for s in in_bin if s["outcome"] == 1)
        no = n - yes
        # Laplace smoothing to avoid div-by-zero
        p_state_given_yes = (yes + 1) / (n_yes + len(labels))
        p_state_given_no = (no + 1) / (n_no + len(labels))
        llr = math.log(p_state_given_yes / p_state_given_no)
        # Confidence shrink by sample size:
        #   n < MIN_TRUST_OBS            -> 0 (ignore; too few samples)
        #   MIN_TRUST_OBS..FULL_TRUST    -> linear ramp 0 -> full
        #   n >= FULL_TRUST_OBS          -> full measured strength
        if n < MIN_TRUST_OBS:
            trust = 0.0
        elif n >= FULL_TRUST_OBS:
            trust = 1.0
        else:
            trust = (n - MIN_TRUST_OBS) / (FULL_TRUST_OBS - MIN_TRUST_OBS)
        llr *= trust
        llr = max(-LLR_CAP, min(LLR_CAP, llr))
        out[lab] = {"llr": round(llr, 4), "n": n,
                    "p_yes": round(yes / n, 4) if n else None}
    return out


def _bin_labels(edges):
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append((lo, hi, f"[{lo:g},{hi:g})"))
    return labels


INF = float("inf")

# Number of equal-population quantile bins per signal. Quantile binning is used
# instead of fixed edges because signal distributions are highly skewed (e.g.
# BTC funding sits near a small positive value ~99% of the time) — fixed edges
# dump almost everything into one bin and starve the others. Quantiles guarantee
# each bin has ~equal sample count, so every LLR is backed by real observations.
N_QUANTILE_BINS = {
    "oi_chg": 4,
    "mom": 5,
}


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    """Compute n_bins quantile edges from data, with -inf/inf at the ends.
    De-duplicates collapsed edges (when many values are identical)."""
    vals = sorted(v for v in values if v is not None)
    if len(vals) < n_bins * 2:
        # not enough data for quantiles; fall back to a single passthrough bin
        return [-INF, INF]
    inner = []
    for i in range(1, n_bins):
        q = i / n_bins
        idx = int(q * (len(vals) - 1))
        inner.append(vals[idx])
    # dedupe while preserving order (skewed data can repeat an edge)
    seen, uniq = set(), []
    for e in inner:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return [-INF] + uniq + [INF]


def calibrate(days: int, symbol: str = "BTCUSDT") -> dict:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    print(f"Fetching {days}d of {symbol} … (klines, funding, OI)")
    klines = fetch_klines(symbol, "1h", start_ms, end_ms)
    funding = fetch_funding(symbol, start_ms, end_ms)
    oi = fetch_oi(symbol, "1h", start_ms, end_ms)
    print(f"  klines={len(klines)} funding={len(funding)} oi={len(oi)}")

    samples = build_samples(klines, funding, oi)
    print(f"  labelled samples: {len(samples)}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "days": days, "n_samples": len(samples),
        "settle_hour_utc": SETTLE_HOUR_UTC,
        "signals": {},
    }
    for key, n_bins in N_QUANTILE_BINS.items():
        vals = [s[key] for s in samples if s[key] is not None]
        edges = _quantile_edges(vals, n_bins)
        result["signals"][key] = {
            "bins": _llr_for_bins(samples, key, edges),
            "edges": [e if e not in (INF, -INF) else
                      ("inf" if e == INF else "-inf") for e in edges],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=720)
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()
    res = calibrate(args.days, args.symbol)
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    # human summary
    for sig, data in res["signals"].items():
        print(f"\n{sig}:")
        for lab, b in data["bins"].items():
            print(f"  {lab:>16}  LLR={b['llr']:+.3f}  n={b['n']:>5}  "
                  f"P(yes)={b['p_yes']}")


if __name__ == "__main__":
    main()
