#!/usr/bin/env python3
"""One-time Polymarket wallet bootstrap.

Run this script ONCE after you've funded your Polymarket safe wallet
with USDC.e (bridged USDC, not native) on Polygon. It:

  1. Verifies your API + private key are loaded correctly.
  2. Shows your USDC balance.
  3. Shows current USDC allowance to each Polymarket exchange contract.
  4. If any allowance is zero, submits an on-chain `approve()` tx to
     grant unlimited USDC spending authority. Costs ~$0.005 in MATIC
     gas. Persistent forever after.
  5. Verifies the allowance was set successfully.

Run from the repo root:

    py -3 tools/bootstrap_polymarket.py

That's it. You don't need ENABLE_REAL_ORDERS for this — `approve()`
is its own action separate from order placement.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Force live=True for this bootstrap — we explicitly want the
# allowance call to hit Polygon.
os.environ["ENABLE_REAL_ORDERS"] = "1"

from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter   # noqa: E402


def _short(addr: str | None) -> str:
    if not addr: return "<missing>"
    return addr[:8] + "..." + addr[-6:]


def main() -> int:
    print("=" * 70)
    print(" Polymarket wallet bootstrap")
    print("=" * 70)

    creds = PolymarketCreds.from_keyfile_and_env()
    if not creds.has_api_creds():
        print("FAIL: API creds missing (API Key/Secret/Passphrase). Edit API_Keys.")
        return 1
    if not creds.has_signing_key():
        print("FAIL: Private Key missing. Edit API_Keys and add a line like:")
        print("      Private Key: 0x<your 64-char wallet key>")
        return 1
    if not creds.funder:
        print("WARN: Funder Address missing — assuming raw EOA wallet (signature_type=0).")
        print("      If you have a Polymarket safe wallet add it as 'Polymarket Wallet Address: 0x...'")

    print(f"  Signature type    : {creds.signature_type}")
    print(f"  Chain ID          : {creds.chain_id}")
    print(f"  Funder (safe)     : {_short(creds.funder)}")
    print(f"  Private key       : present ({creds.private_key[:8]}...)")

    print("\n[1/3] Initializing CLOB client...")
    router = PolymarketOrderRouter(creds=creds, live=True)
    if not router.live or router._client is None:
        print("FAIL: router didn't initialize live mode")
        return 1
    print(f"      EOA derived from key : {router._client.get_address()}")
    print(f"      Funder (target)      : {creds.funder}")

    print("\n[2/3] Reading current wallet state...")
    pre = router.preflight()
    bal = pre.get("usdc_balance", 0)
    allowances = pre.get("usdc_allowances", {}) or {}
    print(f"      USDC balance         : ${bal:,.2f}")
    print(f"      Allowances:")
    for spender, amt in allowances.items():
        flag = " (NEEDS GRANT)" if amt < 1.0 else " OK"
        print(f"        {_short(spender):<22}  ${amt:,.2f}{flag}")

    if bal < 0.001:
        print()
        print("FAIL: USDC balance is ZERO. Bridge USDC.e to", creds.funder)
        print("      Bot can't place orders without funds.")
        print("      (You can still grant the allowance — it's harmless — but no")
        print("       trade will succeed until USDC arrives.)")

    need_grant = any(amt < 1.0 for amt in allowances.values())
    if not need_grant:
        print("\n[3/3] All allowances already granted — nothing to do. Bot is ready.")
        return 0

    print("\n[3/3] Submitting USDC approve() tx (one-shot, costs ~$0.005 in MATIC)...")
    ok = router.set_max_usdc_allowance()
    if not ok:
        print("FAIL: allowance tx did not succeed. Check MATIC balance on the wallet")
        print("      (need ~0.05 MATIC for gas). Inspect the error above.")
        return 1

    print("\n[verify] Re-reading allowances after grant...")
    post = router.preflight()
    new_allowances = post.get("usdc_allowances", {}) or {}
    for spender, amt in new_allowances.items():
        flag = " GRANTED" if amt >= 1.0 else " STILL ZERO ??"
        print(f"        {_short(spender):<22}  ${amt:,.2f}{flag}")

    if all(amt >= 1.0 for amt in new_allowances.values()):
        print("\nSUCCESS — bot is ready to place real orders.")
        print("Next:   py -3 -m live_bot.main --live")
        return 0
    else:
        print("\nWARNING: allowance still zero after grant — Polygon may be slow.")
        print("Re-run this script in 30s. If it persists, check the tx on PolygonScan.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
