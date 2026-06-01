"""
Apply calibrated LLRs to live signal values.

calibrate_llr.py produces calibration.json (LLR per signal per value-bin from
real history). This module loads it and, given the CURRENT value of a signal,
returns the calibrated LLR for the bin that value falls into. That LLR is real
— measured from outcomes — not a hand-picked guess.

If calibration.json is missing or a signal isn't calibrated, the LLR is 0.0
(no bias), so an uncalibrated app degrades to "prior only" rather than to a
fabricated signal.
"""

from __future__ import annotations

import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def load_calibration(path: str = _PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _edges_to_floats(edges):
    out = []
    for e in edges:
        if e == "inf":
            out.append(float("inf"))
        elif e == "-inf":
            out.append(float("-inf"))
        else:
            out.append(float(e))
    return out


def llr_for_value(calib: dict, signal: str, value) -> tuple[float, str]:
    """Return (llr, bin_label) for a live signal value. (0.0,'') if uncalibrated
    or value is None."""
    if not calib or value is None:
        return 0.0, ""
    sig = calib.get("signals", {}).get(signal)
    if not sig:
        return 0.0, ""
    edges = _edges_to_floats(sig.get("edges", []))
    bins = sig.get("bins", {})
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value < hi:
            lab = f"[{lo:g},{hi:g})"
            b = bins.get(lab)
            return (float(b["llr"]) if b else 0.0), lab
    return 0.0, ""


def calibration_age_days(calib: dict) -> float | None:
    if not calib or "generated_at" not in calib:
        return None
    from datetime import datetime, timezone
    try:
        gen = datetime.fromisoformat(calib["generated_at"])
        return (datetime.now(timezone.utc) - gen).total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return None
