"""
Focused diagnostic #2 — identify the BTC range market.

The first diagnostic showed outcomes 133/134/135 ("Recurring Named Outcome",
index:0/1/2) which is likely the BTC range 3-way. To confirm, we need the
`questions` array (which links a question to its outcome IDs and bounds) and
the live mids for those coins.

Run locally:
    python diagnose_hl2.py

Prints compactly (no truncation):
  1. Every question: id, name, and any outcome links / fields.
  2. Every outcome id with name + description (one line each).
  3. Live mid for every #N coin, grouped by outcome.
"""

import json
import urllib.request

INFO_URL = "https://api.hyperliquid.xyz/info"
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def post(payload):
    req = urllib.request.Request(
        INFO_URL, data=json.dumps(payload).encode(), headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def main():
    meta = post({"type": "outcomeMeta"})
    mids = post({"type": "allMids"})
    mids = mids if isinstance(mids, dict) else {}

    print("=" * 70)
    print("QUESTIONS (full)")
    print("=" * 70)
    for q in meta.get("questions", []):
        # print the entire question dict so we see how it links to outcomes
        print(json.dumps(q, default=str))

    print("\n" + "=" * 70)
    print("OUTCOMES (one line each: id | name | description)")
    print("=" * 70)
    for o in meta.get("outcomes", []):
        oid = o.get("outcome")
        print(f"{oid} | {o.get('name','')} | {o.get('description','')}")

    print("\n" + "=" * 70)
    print("LIVE MIDS for #N coins (coin = outcome*10 + side)")
    print("=" * 70)
    hash_keys = sorted((k for k in mids if str(k).startswith("#")),
                       key=lambda k: int(str(k).lstrip("#")))
    for k in hash_keys:
        oid = int(str(k).lstrip("#")) // 10
        side = int(str(k).lstrip("#")) % 10
        print(f"{k}  (outcome {oid}, side {side})  mid = {mids[k]}")


if __name__ == "__main__":
    main()
