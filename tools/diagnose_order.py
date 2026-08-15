#!/usr/bin/env python3
"""Try to submit ONE small FAK order with explicit options to find out
what's causing `order_version_mismatch` in the live bot.

Tries several configurations on a currently-active BTC market.
Will only succeed if the wallet has USDC and allowance is set.

Run from repo root:
    py -3 tools/diagnose_order.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ["ENABLE_REAL_ORDERS"] = "1"

from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter
# Use v2 throughout — that's what the production order_router uses
from py_clob_client_v2 import (
    MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions, Side,
)


def main() -> int:
    creds = PolymarketCreds.from_keyfile_and_env()
    router = PolymarketOrderRouter(creds=creds, live=True)
    if not router.live or router._client is None:
        print("FAIL: router not live"); return 1
    client = router._client

    # Discover the most recent BTC market from the recorder's REST files
    # (gamma's REST filtering is awkward; the recorder has fresh metadata)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    rest_dir = Path("data") / "raw" / "polymarket" / "btc_updown_5m" / today
    print(f"Looking for recent market metadata in {rest_dir}")
    rest_files = sorted(rest_dir.glob("*.rest.jsonl.zst"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not rest_files:
        print(f"FAIL: no REST files in {rest_dir}"); return 1
    from binance_recorder.compression import iter_zstd_jsonl_with_options
    # Walk recent files newest-first and pick a market whose market_close_s
    # is in the FUTURE (currently-active orderbook).
    import time as _time
    now_s = int(_time.time())
    most_recent_rec = None
    for f in rest_files[:10]:
        for rec in iter_zstd_jsonl_with_options(f, allow_truncated_stream=True,
                                                allow_partial_final_line=True):
            if rec.get("endpoint") != "gamma_discovery": continue
            close_s = int(rec.get("market_close_s") or 0)
            if close_s > now_s + 30:  # at least 30s of life remaining
                most_recent_rec = rec
                # don't break — keep going to find the latest active record
        if most_recent_rec:
            break
    if not most_recent_rec:
        print(f"FAIL: no currently-active market found (wall-clock {now_s})")
        return 1
    print(f"selected active market (closes in {int(most_recent_rec.get('market_close_s') or 0) - now_s}s)")
    sm = (most_recent_rec.get("payload") or {}).get("selected_market") or {}
    slug = most_recent_rec.get("selected_market_slug") or sm.get("slug")
    import json as _json
    token_ids = _json.loads(sm.get("clobTokenIds") or "[]")
    outcomes = _json.loads(sm.get("outcomes") or '["Up","Down"]')
    if len(token_ids) < 2:
        print(f"FAIL: market {slug} has no clob tokens"); return 1
    down_idx = outcomes.index("Down") if "Down" in outcomes else 1
    down_token_id = token_ids[down_idx]
    tick = sm.get("orderPriceMinTickSize", 0.01)
    min_size = sm.get("orderMinSize", 5)
    neg_risk = bool(sm.get("negRisk", False))
    print(f"\nmarket: {slug}")
    print(f"  down_token_id : {down_token_id}")
    print(f"  tick_size     : {tick}")
    print(f"  min_size      : {min_size}")
    print(f"  neg_risk      : {neg_risk}")
    # Also ask the SDK what it thinks
    try:
        sdk_tick = client.get_tick_size(down_token_id)
        sdk_neg  = client.get_neg_risk(down_token_id)
        print(f"  SDK thinks tick={sdk_tick}, neg_risk={sdk_neg}")
    except Exception as e:
        print(f"  SDK metadata query failed: {e}")

    # Use a very low limit price so the order won't match anything (cheap test)
    # but is still valid (above min tick)
    test_price = 0.01
    test_size = max(float(min_size), 5.0)
    print(f"\n=== test order: BUY {test_size} shares DOWN @ ${test_price} (FAK) ===")
    print("(Won't match anything at this price — just testing API acceptance)\n")

    # Use the SAME order construction as production (v2 MarketOrderArgs + FAK).
    # Amount is in USDC notional ($1 minimum).
    notional = round(test_price * test_size, 2)
    print(f"notional: ${notional}")

    configs = [
        ("v2 MarketOrderArgs + tick=0.01 (production code path)",
         PartialCreateOrderOptions(tick_size="0.01")),
        ("v2 MarketOrderArgs + tick=0.01 + neg_risk=False",
         PartialCreateOrderOptions(tick_size="0.01", neg_risk=False)),
        ("v2 MarketOrderArgs + tick=None (auto-detect)",
         None),
    ]
    for label, opts in configs:
        print(f"--- {label} ---")
        try:
            order_args = MarketOrderArgs(
                token_id=down_token_id,
                amount=notional,
                side=Side.BUY,
                price=test_price,
                order_type=OrderType.FAK,
            )
            resp = client.create_and_post_market_order(
                order_args=order_args,
                options=opts,
                order_type=OrderType.FAK,
            )
            print(f"  resp: {resp}")
            success = bool(resp.get("success")) or (resp.get("status") in ("matched","live","delayed"))
            if success:
                print(f"  SUCCESS")
                return 0
        except Exception as e:
            import traceback
            print(f"  {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-12:]:
                print(f"    {line}")
        print()
        time.sleep(1)

    print("All 3 configurations failed. The issue is not signature/neg_risk/tick.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
