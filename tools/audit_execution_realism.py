"""Depth-aware execution realism audit for shadow ENTER decisions (Phase 2.3).

For every shadow-bot ENTER on days with local raw PM L2 (2026-07-03+), walk the
recorded order book at the delayed fill time and answer:

  - was there enough size at the top ask to fill the full share count?
  - what is the depth-aware effective fill price for N shares (book walk)?
  - how much size sat within the 0.03 slippage cap (capacity per signal)?

The frozen test-split npz stores only top-of-book, and the May-June raw L2
lives on the currently-unplugged D:\\RawDataStorage drive, so this runs on the
shadow OOS days - which is also the sample that matches the current market
regime and the post-fix recorder.

Usage:
    py tools/audit_execution_realism.py                     # all local days
    py tools/audit_execution_realism.py --shares 5.1 --delay-s 2

Output: artifacts/audit_v1/execution_realism_shadow.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd

REPO = Path(__file__).resolve().parents[1]
L2_ROOT = REPO / "data" / "raw" / "polymarket" / "btc_updown_5m"
RES_ROOT = REPO / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"
OUT_DIR = REPO / "artifacts" / "audit_v1"
NS = 1_000_000_000

DECISION_DIRS = sorted(REPO.glob("logs/tcn_shadow_bot_direct_capture*"))


def read_zst_lines(path: Path):
    dctx = zstd.ZstdDecompressor()
    try:
        with open(path, "rb") as f:
            with dctx.stream_reader(f, read_across_frames=True) as r:
                text = io.TextIOWrapper(r, encoding="utf-8", errors="replace")
                while True:
                    try:
                        line = text.readline()
                    except zstd.ZstdError:
                        return
                    if not line:
                        return
                    yield line
    except (OSError, zstd.ZstdError):
        return


def load_token_mapping(date: str) -> dict[str, dict[str, str]]:
    """slug -> {'UP': asset_id, 'DOWN': asset_id}"""
    out: dict[str, dict[str, str]] = {}
    for p in sorted(glob.glob(str(RES_ROOT / date / "*.resolution.jsonl.zst"))):
        for line in read_zst_lines(Path(p)):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = d.get("market_slug")
            tm = d.get("token_mapping")
            if not slug or not tm or slug in out:
                continue
            m = {}
            for row in tm:
                side = str(row.get("normalized_side", "")).upper()
                if side in ("UP", "DOWN"):
                    m[side] = str(row["asset_id"])
            if len(m) == 2:
                out[slug] = m
    return out


def load_enters(date: str) -> list[dict]:
    enters, seen = [], set()
    for ddir in DECISION_DIRS:
        p = ddir / f"tcn_decisions_{date}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("decision") != "ENTER":
                    continue
                key = (d.get("market_slug"), d.get("strategy_id"), d.get("snapshot_ts_ns"))
                if key in seen:
                    continue
                seen.add(key)
                enters.append(d)
    return enters


def book_walk(asks: list[dict], shares: float) -> tuple[float | None, float]:
    """(effective avg price for `shares`, shares available at top level)."""
    remaining = shares
    cost = 0.0
    top_size = float(asks[0]["size"]) if asks else 0.0
    for lvl in asks:
        px, sz = float(lvl["price"]), float(lvl["size"])
        take = min(remaining, sz)
        cost += take * px
        remaining -= take
        if remaining <= 1e-9:
            return cost / shares, top_size
    return None, top_size   # book too thin


def capacity_within(asks: list[dict], cap_price: float) -> float:
    """Total shares available at price <= cap_price."""
    return sum(float(l["size"]) for l in asks if float(l["price"]) <= cap_price + 1e-12)


def audit_day(date: str, shares: float, delay_s: float) -> list[dict]:
    enters = load_enters(date)
    if not enters:
        return []
    mapping = load_token_mapping(date)
    files_by_slug: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(glob.glob(str(L2_ROOT / date / "*.l2.jsonl.zst"))):
        parts = Path(p).name.split("__")
        if len(parts) >= 2:
            files_by_slug[parts[1]].append(Path(p))

    # group enters per (slug, asset) so every market file is read exactly once
    enters_by_slug: dict[str, list[dict]] = defaultdict(list)
    for e in enters:
        enters_by_slug[e.get("market_slug")].append(e)

    rows = []
    for slug, slug_enters in enters_by_slug.items():
        targets = []   # (t_fill_ns, asset, enter, row_slot)
        for e in slug_enters:
            side = str(e.get("side", "")).upper()
            row = {
                "date": date, "market_slug": slug, "strategy_id": e.get("strategy_id"),
                "side": side, "decision_quote": e.get("fill_quote"), "ev": e.get("ev"),
            }
            rows.append(row)
            asset = mapping.get(slug, {}).get(side)
            if asset is None or slug not in files_by_slug:
                row["status"] = "no_mapping_or_l2"
                continue
            targets.append((int(e["snapshot_ts_ns"]) + int(delay_s * NS), asset, row))
        if not targets:
            continue
        # single pass over this market's records; keep the freshest book <= each target
        best: dict[int, tuple[int, list]] = {}
        for p in files_by_slug[slug]:
            for line in read_zst_lines(p):
                if '"l2_book_state"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asset_id = str(d.get("token_asset_id"))
                st = d.get("source_timestamps") or {}
                ts_ms = st.get("message_timestamp_ms") or st.get("timestamp_ms")
                if ts_ms is None:
                    continue
                ts = int(ts_ms) * 1_000_000
                for j, (t_fill, asset, _row) in enumerate(targets):
                    if asset_id == asset and ts <= t_fill:
                        if j not in best or ts > best[j][0]:
                            book = d.get("book") or {}
                            best[j] = (ts, book.get("asks") or [])
        for j, (t_fill, asset, row) in enumerate(targets):
            if j not in best:
                row["status"] = "no_book_before_fill"
                continue
            ts, asks = best[j]
            eff, top_size = book_walk(asks, shares)
            top_ask = float(asks[0]["price"]) if asks else None
            row.update({
                "status": "ok",
                "book_age_at_fill_s": round((t_fill - ts) / NS, 3),
                "top_ask": top_ask,
                "top_ask_size": round(top_size, 2),
                "full_fill_at_top": bool(top_size >= shares),
                "effective_fill_price": round(eff, 5) if eff is not None else None,
                "effective_vs_top_slip": round(eff - top_ask, 5) if (eff is not None and top_ask is not None) else None,
                "capacity_shares_within_slip_cap": round(
                    capacity_within(asks, (top_ask or 0.0) + 0.03), 2) if top_ask is not None else None,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shares", type=float, default=5.1)
    ap.add_argument("--delay-s", type=float, default=2.0)
    ap.add_argument("--dates", default=None, help="comma list; default = all local L2 days")
    args = ap.parse_args()

    if args.dates:
        dates = args.dates.split(",")
    else:
        dates = sorted(p.name for p in L2_ROOT.iterdir() if p.is_dir())
    all_rows: list[dict] = []
    t0 = time.time()
    for d in dates:
        rows = audit_day(d, args.shares, args.delay_s)
        ok = [r for r in rows if r.get("status") == "ok"]
        print(f"{d}: {len(rows)} enters, {len(ok)} with book ({time.time() - t0:.0f}s)", flush=True)
        all_rows.extend(rows)

    ok = [r for r in all_rows if r.get("status") == "ok"]
    def arr(k):
        return np.asarray([r[k] for r in ok if r.get(k) is not None], dtype=np.float64)
    summary = {
        "n_enters": len(all_rows),
        "n_with_book": len(ok),
        "shares": args.shares,
        "delay_s": args.delay_s,
        "full_fill_at_top_rate": round(float(np.mean([r["full_fill_at_top"] for r in ok])), 4) if ok else None,
        "top_ask_size": {
            "p10": round(float(np.percentile(arr("top_ask_size"), 10)), 2),
            "p50": round(float(np.percentile(arr("top_ask_size"), 50)), 2),
            "p90": round(float(np.percentile(arr("top_ask_size"), 90)), 2),
        } if ok else None,
        "effective_vs_top_slip": {
            "mean": round(float(arr("effective_vs_top_slip").mean()), 6),
            "p95": round(float(np.percentile(arr("effective_vs_top_slip"), 95)), 6),
            "max": round(float(arr("effective_vs_top_slip").max()), 6),
        } if ok else None,
        "capacity_shares_within_slip_cap": {
            "p10": round(float(np.percentile(arr("capacity_shares_within_slip_cap"), 10)), 1),
            "p50": round(float(np.percentile(arr("capacity_shares_within_slip_cap"), 50)), 1),
            "p90": round(float(np.percentile(arr("capacity_shares_within_slip_cap"), 90)), 1),
        } if ok else None,
        "note": "May-June test-period L2 lives on D:\\RawDataStorage (unplugged); this covers shadow OOS days only.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "rows": all_rows}
    (OUT_DIR / "execution_realism_shadow.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"-> {OUT_DIR / 'execution_realism_shadow.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
