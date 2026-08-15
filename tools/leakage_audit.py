#!/usr/bin/env python3
"""Forensic leakage audit on a specific suspicious fill.

Checks three things end-to-end against the RAW recorder data:

  CHECK 1 — BTC price freshness:
    For a snapshot at ts T with reported `binance_spot_mid = M`, verify
    M matches the most-recent Binance bookTicker event in raw with
    recv_ts_ns <= T. No future Binance data leaked into the row.

  CHECK 2 — Polymarket L2 freshness:
    For the same snapshot, verify `up_token_best_bid` matches the most
    recent `l2_book_state` event for the UP token with recv_ts_ns <= T.
    No future L2 state leaked.

  CHECK 3 — Resolution provenance:
    Verify the market's `resolved_side` in the parquet was sourced from
    a `market_resolution` raw event with recv_ts_ns >= market_close_ts_ns,
    AND that the resolution event matches the dataset's label.

If all three pass for a sample of high-edge wins, the backtest numbers
are not a leakage artifact.
"""
from __future__ import annotations

import io
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import polars as pl
from market_recorders.unified_reader import UnifiedRawReader
from binance_recorder.compression import iter_zstd_jsonl_with_options


RAW_ROOT = Path("data")


def _ts(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat(timespec="milliseconds")


def check_binance_freshness(snapshot_ts_ns: int, reported_mid: float, day: date) -> dict:
    """Walk raw Binance bookTicker for the relevant hour; find the latest
    bid/ask with recv_ts_ns <= snapshot_ts_ns; reconstruct mid; compare."""
    snap_dt = datetime.fromtimestamp(snapshot_ts_ns / 1e9, tz=UTC)
    hour = snap_dt.hour
    paths = [
        RAW_ROOT / "raw" / "binance" / "spot" / "BTCUSDT" / day.isoformat() / f"{hour:02d}.ws.jsonl.zst",
    ]
    # also include previous hour to handle minute-0 edge
    if hour > 0:
        paths.insert(0, paths[0].with_name(f"{hour-1:02d}.ws.jsonl.zst"))

    last_pre  = None   # (recv_ts_ns, mid)
    first_post = None  # first event with recv_ts_ns > snapshot_ts_ns
    for p in paths:
        if not p.exists(): continue
        for rec in iter_zstd_jsonl_with_options(p, allow_truncated_stream=True, allow_partial_final_line=True):
            if rec.get("event_type") != "bookTicker": continue
            recv = int(rec["recv_ts_ns"])
            pay = rec["payload"]
            b = float(pay["b"]); a = float(pay["a"])
            mid = (b + a) / 2.0
            if recv <= snapshot_ts_ns:
                last_pre = (recv, mid, b, a)
            else:
                first_post = (recv, mid, b, a)
                break
        if first_post: break

    return {
        "reported_mid":         reported_mid,
        "latest_pre_snap":      last_pre,
        "first_post_snap":      first_post,
        "diff_pre_vs_reported": (abs(last_pre[1] - reported_mid) if last_pre else None),
        "verdict": ("PASS" if last_pre and abs(last_pre[1] - reported_mid) < 0.5 else "INVESTIGATE"),
    }


def check_polymarket_l2_freshness(snapshot_ts_ns: int, market_slug: str,
                                   reported_up_bid: float, day: date) -> dict:
    """Find the latest l2_book_state record for the UP token with
    recv_ts_ns <= snapshot_ts_ns; compare its best_bid to the parquet's."""
    folder = RAW_ROOT / "raw" / "polymarket" / "btc_updown_5m" / day.isoformat()
    if not folder.exists():
        return {"verdict": "FAIL", "reason": f"folder missing: {folder}"}
    # Find the L2 file for this slug
    files = list(folder.glob(f"*__{market_slug}__*.l2.jsonl.zst"))
    if not files:
        return {"verdict": "FAIL", "reason": f"no l2 file for {market_slug}"}
    last_pre  = None
    first_post = None
    for p in files:
        for rec in iter_zstd_jsonl_with_options(p, allow_truncated_stream=True, allow_partial_final_line=True):
            if rec.get("record_type") != "l2_book_state":  continue
            if str(rec.get("token_outcome") or "").lower() != "up": continue
            recv = int(rec.get("recv_ts_ns") or 0)
            best_bid = rec.get("best_bid")
            if best_bid is None: continue
            if recv <= snapshot_ts_ns:
                last_pre = (recv, float(best_bid))
            else:
                first_post = (recv, float(best_bid))
                break
        if first_post: break
    return {
        "reported_up_bid":       reported_up_bid,
        "latest_pre_snap":       last_pre,
        "first_post_snap":       first_post,
        "diff_pre_vs_reported":  (abs(last_pre[1] - reported_up_bid) if last_pre else None),
        "verdict": ("PASS" if last_pre and abs(last_pre[1] - reported_up_bid) < 1e-6 else "INVESTIGATE"),
    }


def check_resolution_provenance(market_slug: str, market_close_ts_ns: int,
                                reported_resolved_side: str, day: date) -> dict:
    """Walk raw resolution files for the day (and the next day) and find
    the market_resolution event for this market. Confirm its recv_ts_ns
    is >= market_close_ts_ns and the side matches."""
    from datetime import timedelta
    candidates = []
    for d in (day, day + timedelta(days=1)):
        folder = RAW_ROOT / "raw" / "polymarket" / "resolution" / "btc_updown_5m" / d.isoformat()
        if not folder.exists(): continue
        for p in sorted(folder.glob("*.resolution.jsonl.zst")):
            for rec in iter_zstd_jsonl_with_options(p, allow_truncated_stream=True, allow_partial_final_line=True):
                if rec.get("record_type") != "market_resolution": continue
                if str(rec.get("market_slug") or "") != market_slug: continue
                candidates.append({
                    "recv_ts_ns":       int(rec.get("recv_ts_ns") or 0),
                    "resolved_side":    str(rec.get("resolved_side") or rec.get("winning_outcome") or "").lower(),
                    "file":             str(p),
                })
    if not candidates:
        return {"verdict": "FAIL", "reason": "no resolution event found"}
    earliest = min(candidates, key=lambda r: r["recv_ts_ns"])
    after_close = earliest["recv_ts_ns"] >= market_close_ts_ns
    side_matches = earliest["resolved_side"] == reported_resolved_side
    return {
        "earliest_event":     earliest,
        "after_close":        after_close,
        "delay_after_close_s": (earliest["recv_ts_ns"] - market_close_ts_ns) / 1e9,
        "side_matches":       side_matches,
        "reported_side":      reported_resolved_side,
        "verdict": ("PASS" if after_close and side_matches else "FAIL"),
    }


def main() -> int:
    # Pick 3 suspicious May 22 wins to audit (high edge, big disagreement, DOWN won)
    targets = [
        # (market_slug, snapshot_ts_ns_seed, expected_resolved)
        ("btc-updown-5m-1779410100", None, "down"),
        ("btc-updown-5m-1779413400", None, "down"),
        ("btc-updown-5m-1779417000", None, "down"),
    ]
    day = date(2026, 5, 22)
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{day.isoformat()}.parquet")

    for slug, _, expected_side in targets:
        sub = df.filter(pl.col("market_slug") == slug)
        if len(sub) == 0:
            print(f"\n=== {slug}: NOT IN PARQUET — skip ==="); continue
        # use the snapshot at ttc closest to 60 (where edge_dn is largest in fills)
        sub60 = sub.sort((pl.col("t_to_close_s") - 60).abs()).head(1)
        row = sub60.row(0, named=True)
        snap_ts = int(row["snapshot_ts_ns"])
        market_close = int(row["market_close_ts_ns"])
        binance_mid = float(row["binance_spot_mid"])
        up_bid = float(row["up_token_best_bid"])
        resolved = str(row["resolved_side"]).lower()

        print(f"\n=== {slug}  ttc={row['t_to_close_s']:.1f}s ===")
        print(f"  snapshot  : {_ts(snap_ts)}  (ts_ns={snap_ts})")
        print(f"  close     : {_ts(market_close)}  (ts_ns={market_close})")
        print(f"  parquet says: binance_spot_mid=${binance_mid:.2f}  up_bid={up_bid:.3f}  resolved={resolved}")

        c1 = check_binance_freshness(snap_ts, binance_mid, day)
        print(f"  [1] Binance freshness: {c1['verdict']}")
        if c1.get("latest_pre_snap"):
            lr, lm, lb, la = c1["latest_pre_snap"]
            print(f"      latest pre-snap @ {_ts(lr)}  bid={lb:.2f} ask={la:.2f} mid={lm:.2f}  (Δ={c1['diff_pre_vs_reported']:.3f})")
        if c1.get("first_post_snap"):
            fr, fm, fb, fa = c1["first_post_snap"]
            print(f"      first post-snap@ {_ts(fr)}  bid={fb:.2f} ask={fa:.2f} mid={fm:.2f}")

        c2 = check_polymarket_l2_freshness(snap_ts, slug, up_bid, day)
        print(f"  [2] Polymarket L2 freshness: {c2['verdict']}")
        if c2.get("latest_pre_snap"):
            lr, lb = c2["latest_pre_snap"]
            print(f"      latest pre-snap @ {_ts(lr)}  up_bid={lb:.3f}  (Δ={c2['diff_pre_vs_reported']:.6f})")
        if c2.get("first_post_snap"):
            fr, fb = c2["first_post_snap"]
            print(f"      first post-snap@ {_ts(fr)}  up_bid={fb:.3f}")

        c3 = check_resolution_provenance(slug, market_close, resolved, day)
        print(f"  [3] Resolution provenance: {c3['verdict']}")
        if "earliest_event" in c3:
            e = c3["earliest_event"]
            print(f"      resolution event @ {_ts(e['recv_ts_ns'])}  side={e['resolved_side']}")
            print(f"      delay vs close = {c3['delay_after_close_s']:+.2f}s   "
                  f"side_matches={c3['side_matches']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
