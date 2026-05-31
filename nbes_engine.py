"""
NBES — NUNO Bayesian Edge Strategy: core engine.

Pure, testable functions. No Streamlit, no I/O. The UI imports from here.

Pipeline:
    prior  ->  logit accumulator (sum of weighted LLRs)  ->  posterior
    posterior vs market mid  ->  edge
    edge + edge-uncertainty  ->  variance-adjusted (quarter) Kelly size
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from scipy.stats import norm

# ---------------------------------------------------------------------------
# logit helpers
# ---------------------------------------------------------------------------

_EPS = 1e-9


def clamp_prob(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def logit(p: float) -> float:
    p = clamp_prob(p)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def btc_lognormal_prior(spot: float, strike: float, hours_to_expiry: float,
                        annual_vol: float) -> float:
    """P(BTC > strike at expiry) under a driftless lognormal (risk-neutral-ish).

    annual_vol is a decimal (e.g. 0.55 for 55%). Time scaled from hours.
    """
    if hours_to_expiry <= 0:
        return 1.0 if spot > strike else 0.0
    if annual_vol <= 0 or spot <= 0 or strike <= 0:
        return 0.5
    t = hours_to_expiry / (24.0 * 365.0)
    d = (math.log(spot / strike) - 0.5 * annual_vol ** 2 * t) / (annual_vol * math.sqrt(t))
    return float(norm.cdf(d))


def cpi_normal_prior(consensus: float, dispersion: float, threshold: float,
                     direction: Literal["above", "below"] = "above") -> float:
    """P(CPI YoY print {above|below} threshold) given a normal over forecasts.

    consensus = mean forecast, dispersion = std of the forecast distribution.
    """
    if dispersion <= 0:
        if direction == "above":
            return 1.0 if consensus > threshold else 0.0
        return 1.0 if consensus < threshold else 0.0
    z = (threshold - consensus) / dispersion
    p_above = 1.0 - float(norm.cdf(z))
    return p_above if direction == "above" else 1.0 - p_above


def cpi_bucket_prior(consensus: float, dispersion: float,
                     center: float, half_width: float = 0.05) -> dict[str, float]:
    """Three-outcome categorical prior for a 'rounds-to-X' CPI market.

    Market buckets (e.g. center=4.3, half_width=0.05):
        below   : print <  center - half_width      (< 4.25)
        exactly : center-half_width <= print < center+half_width  ([4.25,4.35))
        above   : print >= center + half_width       (>= 4.35)

    Returns probabilities that sum to 1.0. A normal CDF over the print
    distribution gives the three slices; the middle bucket is the mass the
    binary model could not represent.
    """
    lo = center - half_width
    hi = center + half_width
    if dispersion <= 0:
        if consensus < lo:
            return {"below": 1.0, "exactly": 0.0, "above": 0.0}
        if consensus >= hi:
            return {"below": 0.0, "exactly": 0.0, "above": 1.0}
        return {"below": 0.0, "exactly": 1.0, "above": 0.0}
    c_lo = float(norm.cdf((lo - consensus) / dispersion))
    c_hi = float(norm.cdf((hi - consensus) / dispersion))
    return {
        "below": c_lo,
        "exactly": c_hi - c_lo,
        "above": 1.0 - c_hi,
    }


# ---------------------------------------------------------------------------
# Evidence / signals
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """One piece of evidence.

    llr    : log-likelihood ratio  ln P(E|YES)/P(E|NO).  Sign = which side it
             favours; magnitude = strength.  Calibrate from history, do not
             eyeball.
    weight : calibration / freshness weight in [0,1]. Decay as it ages.
    active : toggle without deleting.
    """
    name: str
    llr: float
    weight: float = 1.0
    active: bool = True
    note: str = ""

    @property
    def contribution(self) -> float:
        return self.weight * self.llr if self.active else 0.0


def posterior_from_signals(prior: float, signals: list[Signal]) -> float:
    """Accumulate weighted LLRs in logit space, return posterior probability."""
    acc = logit(prior)
    for s in signals:
        acc += s.contribution
    return inv_logit(acc)


def categorical_posterior(prior: dict[str, float],
                          deltas: dict[str, float]) -> dict[str, float]:
    """Update a multi-outcome prior with additive log-evidence per bucket.

    prior  : {bucket: prob} summing to 1.
    deltas : {bucket: summed weighted log-evidence} (0 if no signal for it).
    Works in log space then renormalises (softmax), the categorical analogue
    of the binary logit update.
    """
    log_post = {}
    for k, p in prior.items():
        log_post[k] = math.log(clamp_prob(p)) + deltas.get(k, 0.0)
    m = max(log_post.values())
    exps = {k: math.exp(v - m) for k, v in log_post.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


# ---------------------------------------------------------------------------
# Edge + sizing
# ---------------------------------------------------------------------------

@dataclass
class EdgeResult:
    side: Literal["BUY_YES", "BUY_NO", "NO_TRADE"]
    posterior: float
    market_mid: float
    exec_price: float          # ask if buying YES, (1-bid) cost basis if buying NO
    edge: float                # signed: posterior - mkt (in YES terms)
    edge_se: float
    kelly_full: float
    kelly_used: float
    stake: float               # currency
    passes_gate: bool
    reasons: list[str] = field(default_factory=list)


def kelly_binary(q: float, p: float) -> float:
    """Full Kelly fraction for a 0/1 contract bought at price p, true prob q."""
    p = clamp_prob(p)
    return (q - p) / (1.0 - p)


def evaluate(
    posterior: float,
    yes_bid: float,
    yes_ask: float,
    *,
    bankroll: float,
    fee: float = 0.0005,
    edge_threshold: float = 0.03,
    edge_se: float = 0.02,
    kelly_lambda: float = 0.25,
    max_frac: float = 0.10,
    hours_to_expiry: float = 999.0,
    min_hours: float = 0.0,
) -> EdgeResult:
    """Decide side, check gates, size with variance-adjusted quarter Kelly."""
    mid = 0.5 * (yes_bid + yes_ask)
    reasons: list[str] = []

    # Which direction does the posterior point, net of the spread you must cross?
    # BUY YES: pay (ask+fee), worth `posterior`.  Edge = posterior - cost.
    edge_buy_yes = posterior - (yes_ask + fee)
    # BUY NO: a NO contract costs (1 - yes_bid) since you sell YES at the bid.
    # It is worth (1 - posterior).  Edge = (1-posterior) - cost_no.
    cost_no = (1.0 - yes_bid) + fee
    edge_buy_no = (1.0 - posterior) - cost_no

    if edge_buy_yes >= edge_buy_no:
        side = "BUY_YES"
        exec_price = yes_ask + fee
        q = posterior
        signed_edge = edge_buy_yes
    else:
        side = "BUY_NO"
        exec_price = (1.0 - yes_bid) + fee
        q = 1.0 - posterior
        signed_edge = edge_buy_no

    # ---- gates ----
    passes = True
    if abs(signed_edge) < edge_threshold:
        passes = False
        reasons.append(
            f"edge {signed_edge:.3f} < threshold {edge_threshold:.3f}")
    if abs(signed_edge) < edge_se:
        passes = False
        reasons.append(
            f"edge {signed_edge:.3f} within 1 SE ({edge_se:.3f}) — too noisy")
    if hours_to_expiry < min_hours:
        passes = False
        reasons.append(
            f"{hours_to_expiry:.1f}h to expiry < min {min_hours:.1f}h")
    if q <= exec_price:
        passes = False
        reasons.append("no positive expected value at executable price")

    # ---- sizing ----
    f_full = max(0.0, kelly_binary(q, exec_price))
    # variance shrink: small edge relative to its SE -> shrink toward zero
    shrink = 1.0 / (1.0 + (edge_se ** 2) / max(signed_edge ** 2, _EPS))
    f_used = kelly_lambda * f_full * shrink
    f_used = min(f_used, max_frac)
    if not passes:
        f_used = 0.0
        side = "NO_TRADE"

    if passes:
        reasons.append("all gates passed")

    return EdgeResult(
        side=side,
        posterior=posterior,
        market_mid=mid,
        exec_price=exec_price,
        edge=signed_edge,
        edge_se=edge_se,
        kelly_full=f_full,
        kelly_used=f_used,
        stake=f_used * bankroll,
        passes_gate=passes,
        reasons=reasons,
    )
