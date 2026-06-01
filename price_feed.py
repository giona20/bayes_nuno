"""
Live BTC spot price fetcher.

Kept separate from nbes_engine (which is pure / no I/O). Tries several public,
no-auth endpoints in order and returns the first that works, so a single
provider outage or rate-limit doesn't break the app. Hyperliquid is tried
first since that's the venue the markets actually settle against.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

_HEADERS = {"User-Agent": "Mozilla/5.0 (NBES dashboard)"}
_TIMEOUT = 5


def _get_json(url: str, data: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=_HEADERS,
                                 method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def _from_hyperliquid() -> float:
    # Hyperliquid info endpoint: allMids returns mid prices keyed by coin.
    body = json.dumps({"type": "allMids"}).encode()
    d = _get_json("https://api.hyperliquid.xyz/info", data=body)
    return float(d["BTC"])


def _from_coinbase() -> float:
    d = _get_json("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    return float(d["data"]["amount"])


def _from_kraken() -> float:
    d = _get_json("https://api.kraken.com/0/public/Ticker?pair=XBTUSD")
    result = d["result"]
    pair_key = next(iter(result))
    return float(result[pair_key]["c"][0])  # last trade close


def _from_binance() -> float:
    d = _get_json("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
    return float(d["price"])


_SOURCES = [
    ("Hyperliquid", _from_hyperliquid),
    ("Coinbase", _from_coinbase),
    ("Kraken", _from_kraken),
    ("Binance", _from_binance),
]


def fetch_btc_spot() -> dict:
    """Return {'price', 'source', 'ts', 'ok', 'error'}.

    Tries each source in turn; first success wins. Never raises — on total
    failure returns ok=False so the UI can fall back to manual entry.
    """
    errors = []
    for name, fn in _SOURCES:
        try:
            price = fn()
            if price and price > 0:
                return {
                    "price": price,
                    "source": name,
                    "ts": datetime.now(timezone.utc),
                    "ok": True,
                    "error": None,
                }
        except Exception as e:  # noqa: BLE001 - we genuinely want to try the next
            errors.append(f"{name}: {type(e).__name__}")
            continue
    return {
        "price": None,
        "source": None,
        "ts": datetime.now(timezone.utc),
        "ok": False,
        "error": "; ".join(errors) or "all sources failed",
    }


def fetch_live_signals() -> dict:
    """Current values of the calibrated signals, from Binance USDⓈ-M futures
    (public, no key). Returns {funding, oi_chg, mom, ok, error}. Any field that
    fails is None; the caller maps None -> LLR 0 (no bias)."""
    out = {"funding": None, "oi_chg": None, "mom": None,
           "ok": False, "error": None}
    errs = []

    # funding (latest premiumIndex carries lastFundingRate)
    try:
        d = _get_json("https://fapi.binance.com/fapi/v1/premiumIndex"
                      "?symbol=BTCUSDT")
        out["funding"] = float(d["lastFundingRate"])
    except Exception as e:  # noqa: BLE001
        errs.append(f"funding:{type(e).__name__}")

    # OI change: last two hourly OI points
    try:
        d = _get_json("https://fapi.binance.com/futures/data/openInterestHist"
                      "?symbol=BTCUSDT&period=1h&limit=2")
        if isinstance(d, list) and len(d) >= 2:
            prev = float(d[0]["sumOpenInterest"])
            now = float(d[1]["sumOpenInterest"])
            if prev:
                out["oi_chg"] = (now - prev) / prev
    except Exception as e:  # noqa: BLE001
        errs.append(f"oi:{type(e).__name__}")

    # 6h momentum from hourly klines
    try:
        d = _get_json("https://fapi.binance.com/fapi/v1/klines"
                      "?symbol=BTCUSDT&interval=1h&limit=7")
        if isinstance(d, list) and len(d) >= 7:
            c_now = float(d[-1][4])
            c_6h = float(d[0][4])
            if c_6h:
                out["mom"] = (c_now - c_6h) / c_6h
    except Exception as e:  # noqa: BLE001
        errs.append(f"mom:{type(e).__name__}")

    out["ok"] = any(out[k] is not None for k in ("funding", "oi_chg", "mom"))
    out["error"] = "; ".join(errs) or None
    return out


if __name__ == "__main__":
    print(fetch_btc_spot())
    print(fetch_live_signals())
