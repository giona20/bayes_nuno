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

def _iter_outcomes(meta) -> list[dict]:
    """Flatten outcomeMeta into a list of outcome dicts, tolerant of shape.

    We don't know the exact schema version live, so we defensively pull the
    fields we need (name/description/coin encoding) wherever they appear.
    """
    out = []
    container = meta
    if isinstance(meta, dict):
        # common keys seen in docs: "outcomes", "questions", "universe"
        container = meta.get("outcomes") or meta.get("universe") or meta.get("questions") or []
    if not isinstance(container, list):
        return out
    for item in container:
        if not isinstance(item, dict):
            continue
        # a "question" may hold nested outcomes
        nested = item.get("outcomes")
        if isinstance(nested, list):
            for o in nested:
                if isinstance(o, dict):
                    o = {**o, "_question": item.get("description") or item.get("name", "")}
                    out.append(o)
        else:
            out.append(item)
    return out


def discover_markets(meta=None) -> list[dict]:
    """Return simplified market records:
        {name, description, sides: [{label, coin}]}
    Side `coin` is the '#N' string usable with allMids / l2Book.
    """
    if meta is None:
        meta = fetch_outcome_meta()
    records = []
    for o in _iter_outcomes(meta):
        desc = str(o.get("description") or o.get("_question") or o.get("name") or "")
        name = str(o.get("name") or desc[:60])
        sides = []
        side_specs = o.get("sideSpecs") or o.get("sides") or []
        if isinstance(side_specs, list):
            for s in side_specs:
                if not isinstance(s, dict):
                    continue
                label = str(s.get("name") or s.get("side") or "")
                # coin encoding may be under several keys
                coin = s.get("coin") or s.get("assetName") or s.get("encoding")
                if coin is None and "asset" in s:
                    coin = f"#{s['asset']}"
                if coin is not None:
                    sides.append({"label": label, "coin": str(coin)})
        records.append({"name": name, "description": desc, "sides": sides})
    return records


def match_market(records: list[dict], keywords: list[str]) -> dict | None:
    """Find the first market whose name/description contains ALL keywords
    (case-insensitive). Returns the record or None."""
    kws = [k.lower() for k in keywords]
    for r in records:
        hay = (r["name"] + " " + r["description"]).lower()
        if all(k in hay for k in kws):
            return r
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
    for bucket, kws in keyword_map.items():
        rec = match_market(records, kws)
        price = None
        coin = None
        if rec and rec["sides"]:
            # prefer the YES side
            yes = next((s for s in rec["sides"]
                        if s["label"].lower() in ("yes", "y")), rec["sides"][0])
            coin = yes["coin"]
            if coin in mids:
                try:
                    price = float(mids[coin])
                except (TypeError, ValueError):
                    price = None
            if price is None:
                book = fetch_l2_book(coin)
                price = book.get("mid")
        result["prices"][bucket] = price
        result["resolved"][bucket] = coin
        any_hit = any_hit or (price is not None)

    result["ok"] = any_hit
    if not any_hit:
        result["error"] = "markets found but no live mids resolved"
    return result


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_outcome_prices({
        "in_range": ["BTC", "72551", "75512"],
        "below": ["BTC", "below", "72551"],
        "above": ["BTC", "above", "75512"],
    }))
