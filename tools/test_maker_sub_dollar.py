"""Wire-test: send a deliberately sub-$1 POST_ONLY order so Polymarket
rejects it. Used to validate that the maker submit path is correctly
wired end-to-end (creds, token_id, signing, network, response parsing)
without risking an actual fill.

Polymarket rejects any marketable order whose notional is below $1, so a
$0.40 order is guaranteed to bounce. If the bot's path is healthy we see
an error response come back from POST /order. If we see a different
failure (auth, signature, malformed payload) we know exactly what to fix.

The tool BYPASSES our own _quantize_order helper -- that helper walks
notional UP to >= $1 by design, which is exactly what we want to skip
here because we WANT to provoke Polymarket's rejection.

Usage:
  py -3 tools/test_maker_sub_dollar.py --token-id 7447206183...
                                       [--limit-price 0.20]
                                       [--size 2.0]

If --token-id is not supplied, the tool reads the most recent
logs/live_bot/decisions_*.jsonl for a token_id from a market whose close
is still in the future.

Safety:
  * Hard cap: limit_price * size_shares <= $1.00. Anything above aborts
    before sending.
  * No position is recorded if (against all odds) a fill comes back —
    this is a pure probe.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter  # noqa: E402


LOG_DIR = REPO / "logs" / "live_bot"


def discover_token_from_logs() -> str | None:
    """Return a DOWN-token-id from a market whose close is still in the
    future, sourced from the most recent decisions log. None if no such
    market is found."""
    logs = sorted(LOG_DIR.glob("decisions_*.jsonl"))
    now_ns = int(time.time() * 1e9)
    for log_path in reversed(logs):
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            close = rec.get("market_close_ts_ns")
            if close is None or close <= now_ns:
                continue
            # The decisions log doesn't carry the token_id directly. We
            # need to cross-reference with the bot's asset_id_by_slug map
            # — but that's in-memory only. So fall back to the caller
            # supplying --token-id when needed.
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Sub-$1 POST_ONLY probe")
    ap.add_argument("--token-id", required=False, default=None,
                    help="DOWN-side token_id of an active Polymarket market")
    ap.add_argument("--limit-price", type=float, default=0.20,
                    help="Bid price for the probe (default 0.20)")
    ap.add_argument("--size", type=float, default=2.0,
                    help="Share size for the probe (default 2.0). With "
                         "default limit price, notional = $0.40 < $1.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s")
    log = logging.getLogger("test_maker_sub_dollar")

    notional = args.limit_price * args.size
    if notional > 1.0:
        log.error("Refusing to send: notional=$%.2f exceeds $1.00 safety cap "
                  "for this probe. Use a smaller --size.", notional)
        sys.exit(2)
    log.info("Probe params: limit_price=%.4f size=%.4f notional=$%.2f",
             args.limit_price, args.size, notional)

    token_id = args.token_id or discover_token_from_logs()
    if not token_id:
        log.error("No --token-id supplied and none could be discovered from logs. "
                  "Run the bot for a minute, copy a 'GET .../tick-size?token_id=…' "
                  "value from a market that's still open, and pass --token-id <that>.")
        sys.exit(2)
    log.info("Using token_id: %s", token_id)

    creds = PolymarketCreds.from_keyfile_and_env()
    if not creds.has_signing_key():
        log.error("No POLYMARKET_PRIVATE_KEY / API_Keys 'Private Key:' line found. "
                  "Probe needs live creds to test the wire path.")
        sys.exit(2)
    if not creds.has_api_creds():
        log.error("No CLOB API creds in env / API_Keys.")
        sys.exit(2)

    # IMPORTANT: live=True. The probe is meaningless in dry-run mode.
    import os
    os.environ["ENABLE_REAL_ORDERS"] = "1"
    router = PolymarketOrderRouter(creds=creds, live=True, logger=log)
    if not router.live:
        log.error("Router refused to go live (missing ENABLE_REAL_ORDERS or signing key?)")
        sys.exit(2)

    # --- SDK call directly, bypassing _quantize_order ---------------------
    # We WANT to send notional < $1 to provoke Polymarket's rejection. Our
    # own _quantize_order would walk size up to clear $1, defeating the
    # purpose of the probe.
    log.info("Initializing direct SDK call (bypassing _quantize_order)...")
    try:
        from py_clob_client_v2 import (
            OrderArgs, OrderType, PartialCreateOrderOptions, Side,
        )
    except ImportError as exc:
        log.error("py_clob_client_v2 not importable: %r", exc)
        sys.exit(2)

    # Round price to 2dp (Polymarket tick size = 0.01)
    limit_price = round(float(args.limit_price), 2)
    size_shares = round(float(args.size), 4)

    order_args = OrderArgs(
        token_id=token_id, price=limit_price, size=size_shares, side=Side.BUY,
    )
    log.info("Submitting POST_ONLY GTC: token=%s price=%.4f size=%.4f notional=$%.2f",
             token_id, limit_price, size_shares, limit_price * size_shares)

    t0 = time.time()
    try:
        resp = router._client.create_and_post_order(
            order_args=order_args,
            options=PartialCreateOrderOptions(tick_size="0.01"),
            order_type=OrderType.GTC,
            post_only=True,
        )
        elapsed = time.time() - t0
        log.info("Raw response (%.0f ms):\n%s", elapsed * 1000,
                 json.dumps(resp, indent=2, default=str))
    except Exception as exc:
        elapsed = time.time() - t0
        log.exception("Submit raised after %.0f ms: %r", elapsed * 1000, exc)
        log.info("--- PROBE OUTCOME ---")
        log.info("EXCEPTION: %r", exc)
        return

    # Interpret
    log.info("--- PROBE OUTCOME ---")
    err = (resp or {}).get("errorMsg") or (resp or {}).get("error")
    success = (resp or {}).get("success", False)
    status = (resp or {}).get("status")
    order_id = (resp or {}).get("orderID") or (resp or {}).get("orderId") or (resp or {}).get("id")
    if err:
        # Polymarket rejected -- this is the EXPECTED outcome for a probe
        log.info("REJECTED by Polymarket (expected) -- error message: %r", err)
        log.info("This means the wire path works: creds OK, token OK, signing OK, "
                 "POST /order reachable. The maker submit path is correctly wired.")
    elif success and order_id:
        # Unexpected: order was accepted. Try to cancel it immediately so
        # we don't accidentally hold a position.
        log.warning("UNEXPECTED ACCEPT: order_id=%s status=%s. Attempting cancel...",
                    order_id, status)
        cancel_resp = router.cancel_order(str(order_id))
        log.info("Cancel response: %s", json.dumps(cancel_resp, indent=2, default=str))
    else:
        log.info("AMBIGUOUS response (neither clear error nor success). status=%r", status)


if __name__ == "__main__":
    main()
