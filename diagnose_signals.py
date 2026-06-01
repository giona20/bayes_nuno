"""
Diagnostic: why are the live signal values showing as '—' (None)?

Run locally:
    python diagnose_signals.py

Checks two things:
  1. Does the live signal fetch actually return funding / OI / momentum?
  2. Does calibration.json contain real LLRs, and what LLR would the CURRENT
     live values map to?

If (1) returns None for everything, the live endpoints are unreachable from
wherever you run this (e.g. Binance geo-block). If (1) works locally but the
deployed app shows '—', it's the Streamlit Cloud server IP being blocked.
"""

from price_feed import fetch_live_signals
from calibration_loader import load_calibration, llr_for_value, calibration_age_days

print("=" * 60)
print("1. LIVE SIGNAL FETCH")
print("=" * 60)
sig = fetch_live_signals()
for k in ("oi_chg", "mom"):
    print(f"  {k:10} = {sig.get(k)}")
print(f"  ok    = {sig.get('ok')}")
print(f"  error = {sig.get('error')}")

print("\n" + "=" * 60)
print("2. CALIBRATION CONTENT")
print("=" * 60)
calib = load_calibration()
if not calib:
    print("  calibration.json NOT FOUND")
else:
    age = calibration_age_days(calib)
    print(f"  samples={calib.get('n_samples')}  "
          f"age={age:.2f}d" if age is not None else "  (no age)")
    for s, data in calib.get("signals", {}).items():
        bins = data.get("bins", {})
        nonzero = sum(1 for b in bins.values() if b.get("llr"))
        print(f"\n  {s}: {len(bins)} bins, {nonzero} with non-zero LLR")
        for lab, b in bins.items():
            print(f"    {lab:>16}  LLR={b['llr']:+.3f}  n={b['n']}")

    print("\n" + "=" * 60)
    print("3. WHAT THE CURRENT LIVE VALUES MAP TO")
    print("=" * 60)
    for key, sval in (("oi_chg", sig.get("oi_chg")),
                      ("mom", sig.get("mom"))):
        llr, lab = llr_for_value(calib, key, sval)
        print(f"  {key:10} live={sval} -> bin {lab or '(none)'} -> LLR {llr:+.3f}")
