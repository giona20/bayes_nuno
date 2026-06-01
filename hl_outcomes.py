"""
Hyperliquid HIP-4 outcome-market client.

Pulls live prices for prediction-market (outcome) contracts from the public
Info endpoint. No API key, no auth for reads.

Data model (from Hyperliquid docs):
  - `outcomeMeta` lists questions -> outcomes -> sides. Each side is a token
    addressed as a "#N" coin. Descriptions are pipe-delimited metadata, e.g.
    "class:priceBinary|underlying:BTC|...".
  - `allMids` returns mid price keyed by coin name, including "#N" outcome coins.
  - `l2Book` with {"coin": "#N"} returns the actual bid/ask ladder for one side.
  - For a YES side priced at p, the implied probability of that outcome is ~p
    (settles to 1 if it occurs, 0 otherwise). YES mid + NO mid ~= 1.

Because the exact "#N" encodings and question text are only knowable from a live
`outcomeMeta` call (they rotate each period for recurring markets), this module
DISCOVERS markets at runtime and matches them by keyword, rather than hardcoding
asset IDs that would go stale. Everything degrades gracefully: any failure
returns ok=False so the UI can fall back to manual price entry.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

INFO_URL = "https://api.hyperliquid.xyz/info"
_HEADERS = {"User-Agent": "Mozilla/5.0 (NBES dashboard)",
            "Content-Type": "application/json"}
_TIMEOUT = 6


def _post(payload: dict) -> dict | list:
    req = urllib.request.Request(
        INFO_URL, data=json.dumps(payload).encode(), headers=_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# raw fetchers
# ---------------------------------------------------------------------------

def fetch_outcome_meta() -> dict | list:
    """Raw outcomeMeta: questions, outcomes, side specs, #N encodings."""
    return _post({"type": "outcomeMeta"})


def fetch_all_mids() -> dict:
    """Mid price for every coin, including #N outcome sides."""
    d = _post({"type": "allMids"})
    return d if isinstance(d, dict) else {}


def fetch_l2_book(coin: str) -> dict:
    """L2 book for one coin (e.g. '#20'). Returns {'bid','ask','mid'} or empty."""
    d = _post({"type": "l2Book", "coin": coin})
    # response: {"coin":..., "levels": [ [bids...], [asks...] ]}
    levels = d.get("levels") if isinstance(d, dict) else None
    if not levels or len(levels) < 2:
        return {}
    bids, asks = levels[0], levels[1]
    bid = float(bids[0]["px"]) if bids else None
    ask = float(asks[0]["px"]) if asks else None
    mid = None
    if bid is not None and ask is not None:
        mid = 0.5 * (bid + ask)
    elif bid is not None:
        mid = bid
    elif ask is not None:
        mid = ask
    return {"bid": bid, "ask": ask, "mid": mid}


# ---------------------------------------------------------------------------
# discovery / parsing
# ---------------------------------------------------------------------------

def _iter_outcomes(meta) -> list[dict]:  # retained for back-compat; unused
    outcomes = meta.get("outcomes", []) if isinstance(meta, dict) else []
    return [o for o in outcomes if isinstance(o, dict)]


def discover_markets(meta=None) -> list[dict]:
    """Return simplified market records from the real outcomeMeta schema:
        {outcome_id, name, description, meta (parsed pipe-fields),
         sides: [{label, side_index, coin}]}

    Coin encoding (confirmed from live data): #{outcome_id*10 + side_index},
    where side_index is the position in sideSpecs (0 = first/Yes, 1 = second/No).
    e.g. outcome 131 -> Yes #1310, No #1311.
    """
    if meta is None:
        meta = fetch_outcome_meta()
    outcomes = meta.get("outcomes", []) if isinstance(meta, dict) else []
    records = []
    for o in outcomes:
        if not isinstance(o, dict):
            continue
        oid = o.get("outcome")
        if oid is None:
            continue
        desc = str(o.get("description") or "")
        name = str(o.get("name") or "")
        parsed = _parse_pipe_meta(desc)
        sides = []
        for idx, s in enumerate(o.get("sideSpecs", [])):
            if not isinstance(s, dict):
                continue
            sides.append({
                "label": str(s.get("name") or ""),
                "side_index": idx,
                "coin": f"#{int(oid) * 10 + idx}",
            })
        records.append({
            "outcome_id": int(oid), "name": name, "description": desc,
            "meta": parsed, "sides": sides,
        })
    return records


def _parse_pipe_meta(desc: str) -> dict:
    """Parse pipe-delimited metadata like
    'class:priceBinary|underlying:BTC|targetPrice:74032|period:1d' into a dict.
    Returns {} if the description isn't in that format."""
    if "|" not in desc and ":" not in desc:
        return {}
    out = {}
    for part in desc.split("|"):
        if ":" in part:
            k, _, v = part.partition(":")
            out[k.strip().lower()] = v.strip()
    return out


def match_market(records: list[dict], keywords: list[str]) -> dict | None:
    """Find the first market whose name + description + parsed-meta values
    contain ALL keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for r in records:
        hay = (r["name"] + " " + r["description"] + " "
               + " ".join(str(v) for v in r["meta"].values())).lower()
        if all(k in hay for k in kws):
            return r
    return None


# ---------------------------------------------------------------------------
# dynamic discovery of the CURRENT recurring markets
# ---------------------------------------------------------------------------
# Recurring markets rotate every period (targetPrice + expiry change, and for
# categorical markets the outcome IDs change). Expired markets are removed from
# outcomeMeta, so whatever is present IS current. We therefore select markets by
# their STABLE metadata (class + underlying + period), never by hardcoded id.

def find_btc_binary(records: list[dict]) -> dict | None:
    """The current daily BTC price binary: class=priceBinary, underlying=BTC.
    Returns the record (with live targetPrice + expiry in .meta) or None."""
    best = None
    for r in records:
        m = r["meta"]
        if m.get("class", "").lower() == "pricebinary" \
                and m.get("underlying", "").upper() == "BTC" \
                and "targetprice" in m:
            # prefer the soonest-expiry instance if several are present
            if best is None:
                best = r
            else:
                e1, e2 = parse_expiry(r["meta"].get("expiry")), \
                    parse_expiry(best["meta"].get("expiry"))
                if e1 and (not e2 or e1 < e2):
                    best = r
    return best


def find_btc_range(records: list[dict]) -> list[dict] | None:
    """The current BTC range (multi-outcome) market as an ordered list of
    outcome records [below, in_range, above].

    Strategy: look for priceBucket-class BTC outcomes, OR the 'Recurring Named
    Outcome' triplet tagged index:0/1/2. Returns them ordered by index, or None.
    """
    # 1. explicit priceBucket class with underlying BTC
    buckets = [r for r in records
               if r["meta"].get("class", "").lower() in ("pricebucket", "pricerange")
               and r["meta"].get("underlying", "").upper() == "BTC"]
    if len(buckets) >= 3:
        buckets.sort(key=lambda r: _index_of(r))
        return buckets[:3]

    # 2. fallback: the 'Recurring Named Outcome' index:0/1/2 triplet
    named = [r for r in records
             if "named outcome" in r["name"].lower()
             and "index" in r["meta"]]
    if len(named) >= 3:
        named.sort(key=lambda r: _index_of(r))
        return named[:3]
    return None


def _index_of(rec: dict) -> int:
    """Pull the numeric 'index:N' from meta, else fall back to outcome_id."""
    v = rec["meta"].get("index")
    if v is not None and str(v).lstrip("-").isdigit():
        return int(v)
    return rec["outcome_id"]


def discover_current_markets() -> dict:
    """One-shot discovery of the live recurring markets, fully dynamic.

    Returns {'ok', 'ts', 'error',
             'btc_binary': {targetPrice, expiry, coin_yes, ...} | None,
             'btc_range':  {bounds, outcomes:[{label,coin,index}], expiry} | None}
    """
    res = {"ok": False, "ts": datetime.now(timezone.utc), "error": None,
           "btc_binary": None, "btc_range": None}
    try:
        recs = discover_markets(fetch_outcome_meta())
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        return res

    b = find_btc_binary(recs)
    if b and b["sides"]:
        yes = next((s for s in b["sides"] if s["label"].lower() in ("yes", "y")),
                   b["sides"][0])
        res["btc_binary"] = {
            "outcome_id": b["outcome_id"],
            "target_price": _to_float(b["meta"].get("targetprice")),
            "expiry": parse_expiry(b["meta"].get("expiry")),
            "coin_yes": yes["coin"],
            "period": b["meta"].get("period"),
        }

    rng = find_btc_range(recs)
    if rng:
        outs = []
        for i, r in enumerate(rng):
            yes = next((s for s in r["sides"] if s["label"].lower() in ("yes", "y")),
                       r["sides"][0])
            outs.append({"label": r["name"], "index": _index_of(r),
                         "coin": yes["coin"], "outcome_id": r["outcome_id"]})
        res["btc_range"] = {
            "outcomes": outs,  # ordered below / in_range / above
            "expiry": parse_expiry(rng[0]["meta"].get("expiry")),
        }

    res["ok"] = res["btc_binary"] is not None or res["btc_range"] is not None
    if not res["ok"]:
        res["error"] = "no live BTC recurring markets found in outcomeMeta"
    return res


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# high-level: get live prices for a set of named buckets
# ---------------------------------------------------------------------------

def get_outcome_prices(keyword_map: dict[str, list[str]]) -> dict:
    """Resolve a {bucket_key: [keywords]} map to live mid prices.

    Strategy: one outcomeMeta + one allMids call, then look up each bucket's
    YES-side coin in allMids. Falls back to l2Book per coin if a mid is missing.

    Returns {'ok', 'prices': {bucket: mid|None}, 'source', 'ts', 'error',
             'resolved': {bucket: coin}}.
    """
    result = {"ok": False, "prices": {}, "resolved": {},
              "source": "Hyperliquid", "ts": datetime.now(timezone.utc),
              "error": None}
    try:
        meta = fetch_outcome_meta()
        records = discover_markets(meta)
        mids = fetch_all_mids()
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    if not records:
        result["error"] = "no outcome markets returned"
        return result

    any_hit = False
    diag = []  # per-bucket trace, surfaced on total failure
    by_id = {r["outcome_id"]: r for r in records}
    expiry = None  # first parseable expiry from any matched market
    for bucket, kws in keyword_map.items():
        price = None
        coin = None
        rec = None
        # A bucket can be pinned to an explicit outcome id via "outcome:NNN"
        # (or an int), bypassing fragile text matching. Otherwise match by text.
        pinned = _explicit_outcome_id(kws)
        if pinned is not None:
            rec = by_id.get(pinned)
        else:
            rec = match_market(records, kws)
        if rec and rec["sides"]:
            yes = next((s for s in rec["sides"]
                        if s["label"].lower() in ("yes", "y")), rec["sides"][0])
            coin = yes["coin"]
            if coin is not None:
                price = _lookup_mid(coin, mids)
                if price is None:
                    try:
                        price = fetch_l2_book(coin).get("mid")
                    except Exception:  # noqa: BLE001
                        price = None
            if expiry is None:
                expiry = parse_expiry(rec["meta"].get("expiry"))
            diag.append(f"{bucket}: matched '{rec['name'][:30]}' "
                        f"(outcome {rec['outcome_id']}) coin={coin} price={price}")
        else:
            diag.append(f"{bucket}: no market matched {kws}")
        result["prices"][bucket] = price
        result["resolved"][bucket] = coin
        any_hit = any_hit or (price is not None)

    result["ok"] = any_hit
    result["diag"] = diag
    result["expiry"] = expiry  # tz-aware datetime or None
    if not any_hit:
        result["error"] = "markets found but no live mids resolved | " + " ; ".join(diag)
    return result


def parse_expiry(s) -> "datetime | None":
    """Parse a Hyperliquid expiry tag into a tz-aware UTC datetime.
    Handles 'YYYYMMDD-HHMM' (e.g. '20260601-0600') and a few fallbacks."""
    if not s:
        return None
    s = str(s).strip()
    fmts = ("%Y%m%d-%H%M", "%Y%m%d-%H%M%S", "%Y%m%dT%H%M", "%Y-%m-%dT%H:%M")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # epoch seconds/millis fallback
    if s.isdigit():
        v = int(s)
        if v > 10_000_000_000:  # millis
            v //= 1000
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    return None


def _explicit_outcome_id(kws) -> int | None:
    """If a keyword list pins an outcome explicitly, return its id.
    Accepts ['outcome:133'] or [133] or ['#133']."""
    for k in kws:
        if isinstance(k, int):
            return k
        s = str(k).strip().lower()
        if s.startswith("outcome:"):
            tail = s.split(":", 1)[1].strip().lstrip("#")
            if tail.isdigit():
                return int(tail)
    return None


def _lookup_mid(coin: str, mids: dict) -> float | None:
    """Find a coin's mid in allMids, tolerant of key formatting.
    Tries the coin as-is, without '#', and matching ignoring case."""
    candidates = [coin, coin.lstrip("#"), f"#{coin.lstrip('#')}"]
    for c in candidates:
        if c in mids:
            try:
                return float(mids[c])
            except (TypeError, ValueError):
                return None
    # case-insensitive / stringified scan as a last resort
    target = coin.lstrip("#").lower()
    for k, v in mids.items():
        if str(k).lstrip("#").lower() == target:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_outcome_prices({
        "in_range": ["BTC", "72551", "75512"],
        "below": ["BTC", "below", "72551"],
        "above": ["BTC", "above", "75512"],
    }))
