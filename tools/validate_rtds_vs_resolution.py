"""Validate that the PUBLIC Polymarket RTDS Chainlink feed
(wss://ws-live-data.polymarket.com, topic crypto_prices_chainlink) reproduces
the price Polymarket BTC Up/Down 5m markets RESOLVE on.

Tests, over a date range:
  1. RTDS steady-state freshness  (recv_ts - chainlink_ts), excluding the first
     records of each hour file (reconnect/catch-up bursts) -> live viability.
  2. On PAGE-sourced resolutions (which carry ground-truth open/close prices):
     how well does RTDS sampled at open_ts / close_ts match the recorded
     open/close price, under three sampling rules?
  3. On ALL resolutions: does sign(rtds_close - rtds_open) reproduce the
     resolved_side?  (RTDS used both as strike proxy and as close proxy.)
  4. "Lock-in lead": how many seconds before close is the outcome already
     decided by RTDS and stays decided until close?

Usage:
  py -3 tools/validate_rtds_vs_resolution.py --date-from 2026-05-14 --date-to 2026-05-16
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from binance_recorder.compression import iter_zstd_jsonl_with_options as iterz  # noqa: E402

RTDS_DIR = ROOT / "data" / "raw" / "polymarket" / "rtds" / "crypto_prices_chainlink" / "btc_usd"
RES_DIR = ROOT / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"
MS = 1_000_000


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def load_rtds(d0: date, d1: date):
    """Return (ts_ms sorted unique, price[], plus per-record freshness with a
    'first-in-file' flag so we can exclude reconnect bursts)."""
    rows = []  # (chainlink_ts_ms, price, recv_ts_ns, first_in_file)
    for d in daterange(d0, d1):
        ddir = RTDS_DIR / d.isoformat()
        if not ddir.exists():
            continue
        for path in sorted(ddir.glob("*.ws.jsonl.zst")):
            first = True
            for r in iterz(path, allow_truncated_stream=True, allow_partial_final_line=True):
                if r.get("record_type") != "chainlink_live_reference":
                    continue
                try:
                    ts_ms = int(r["chainlink_ts_ms"])
                    px = float(r["btc_usd_price"])
                    recv = int(r["recv_ts_ns"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append((ts_ms, px, recv, first))
                first = False
    rows.sort(key=lambda x: (x[0], x[2]))
    # dedup by chainlink_ts_ms keeping the last (latest recv)
    by_ts = {}
    for ts_ms, px, recv, first in rows:
        by_ts[ts_ms] = (px, recv, first)
    ts_arr = np.array(sorted(by_ts), dtype=np.int64)
    px_arr = np.array([by_ts[t][0] for t in ts_arr], dtype=float)
    fresh_ms = np.array([(by_ts[t][1] - t * MS) / MS for t in ts_arr], dtype=float)
    first_flag = np.array([by_ts[t][2] for t in ts_arr], dtype=bool)
    return ts_arr, px_arr, fresh_ms, first_flag


def load_resolutions(d0: date, d1: date):
    out = []
    for d in daterange(d0, d1):
        ddir = RES_DIR / d.isoformat()
        if not ddir.exists():
            continue
        seen = {}
        for path in sorted(ddir.glob("*.resolution.jsonl.zst")):
            for r in iterz(path, allow_truncated_stream=True, allow_partial_final_line=True):
                if r.get("record_type") != "market_resolution":
                    continue
                slug = r.get("market_slug")
                if not slug:
                    continue
                seen[slug] = r  # last wins
        out.extend(seen.values())
    return out


def sample_at(ts_arr, px_arr, t_ms, rule):
    """Sample RTDS price at chainlink time t_ms (ms)."""
    if rule == "last_le":
        i = np.searchsorted(ts_arr, t_ms, side="right") - 1
    elif rule == "first_ge":
        i = np.searchsorted(ts_arr, t_ms, side="left")
    elif rule == "nearest":
        i = np.searchsorted(ts_arr, t_ms, side="left")
        if i > 0 and (i >= len(ts_arr) or abs(ts_arr[i - 1] - t_ms) <= abs(ts_arr[i] - t_ms)):
            i = i - 1
    else:
        raise ValueError(rule)
    if i < 0 or i >= len(ts_arr):
        return None, None
    return float(px_arr[i]), int(ts_arr[i])


def pct(a, q):
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-from", type=date.fromisoformat, default=date(2026, 5, 14))
    ap.add_argument("--date-to", type=date.fromisoformat, default=date(2026, 5, 16))
    args = ap.parse_args()

    print(f"Loading RTDS {args.date_from}..{args.date_to} ...", flush=True)
    ts_arr, px_arr, fresh_ms, first_flag = load_rtds(args.date_from, args.date_to)
    print(f"  RTDS unique chainlink seconds: {len(ts_arr):,}")
    if len(ts_arr) == 0:
        print("No RTDS data; abort.")
        return
    span_s = (ts_arr[-1] - ts_arr[0]) / 1000.0
    print(f"  span = {span_s/3600:.1f} h, coverage = {len(ts_arr)/max(1,span_s)*100:.1f}% of seconds")

    # --- 1. steady-state freshness (exclude first-in-file reconnect bursts) ---
    steady = fresh_ms[~first_flag]
    print("\n[1] RTDS freshness recv-chainlink (ms), steady-state (excl first-in-file):")
    print(f"    n={steady.size:,}  p50={pct(steady,.5):.0f}  p90={pct(steady,.9):.0f}  "
          f"p99={pct(steady,.99):.0f}  max={np.nanmax(steady):.0f}")
    print(f"    first-in-file records: p50={pct(fresh_ms[first_flag],.5):.0f}ms (the catch-up burst)")

    res = load_resolutions(args.date_from, args.date_to)
    print(f"\nResolutions loaded: {len(res)}")

    # --- 2. page-sourced ground-truth open/close reproduction ---
    print("\n[2] RTDS vs recorded page open/close price (ground truth), by sampling rule:")
    for rule in ("last_le", "first_ge", "nearest"):
        cerr, oerr = [], []
        for r in res:
            op = r.get("page_resolution_open_price")
            cp = r.get("page_resolution_close_price")
            if op is None or cp is None:
                continue
            o_ms = int(r["market_open_ts_ns"]) // MS
            c_ms = int(r["market_close_ts_ns"]) // MS
            ro, _ = sample_at(ts_arr, px_arr, o_ms, rule)
            rc, _ = sample_at(ts_arr, px_arr, c_ms, rule)
            if ro is not None:
                oerr.append(ro - float(op))
            if rc is not None:
                cerr.append(rc - float(cp))
        cerr = np.array(cerr); oerr = np.array(oerr)
        if cerr.size:
            print(f"    rule={rule:8s} n={cerr.size:4d} | close: medAE=${np.median(np.abs(cerr)):.2f} "
                  f"p95=${pct(np.abs(cerr),.95):.2f} exact(<$0.01)={(np.abs(cerr)<0.01).mean()*100:.0f}% "
                  f"| open: medAE=${np.median(np.abs(oerr)):.2f}")

    # --- 3. reproduce resolved_side via sign(rtds_close - rtds_open) ---
    print("\n[3] Reproduce resolved_side from RTDS (strike=rtds_open, close=rtds_close):")
    for rule in ("last_le", "nearest"):
        ok = tot = ties = miss = 0
        for r in res:
            side = r.get("resolved_side")
            if side not in ("up", "down"):
                continue
            o_ms = int(r["market_open_ts_ns"]) // MS
            c_ms = int(r["market_close_ts_ns"]) // MS
            ro, _ = sample_at(ts_arr, px_arr, o_ms, rule)
            rc, _ = sample_at(ts_arr, px_arr, c_ms, rule)
            if ro is None or rc is None:
                miss += 1
                continue
            tot += 1
            pred = "up" if rc >= ro else "down"
            if rc == ro:
                ties += 1
            if pred == side:
                ok += 1
        print(f"    rule={rule:8s} matched={ok}/{tot} = {ok/max(1,tot)*100:.2f}%  "
              f"(ties={ties}, no-rtds-coverage={miss})")

    # --- 4. lock-in lead: seconds before close the outcome is already fixed ---
    print("\n[4] Lock-in lead (how early RTDS already shows the final side and holds it):")
    leads = []
    for r in res:
        side = r.get("resolved_side")
        if side not in ("up", "down"):
            continue
        o_ms = int(r["market_open_ts_ns"]) // MS
        c_ms = int(r["market_close_ts_ns"]) // MS
        strike, _ = sample_at(ts_arr, px_arr, o_ms, "last_le")
        if strike is None:
            continue
        # walk back second-by-second from close; find longest suffix on final side
        lead = 0
        for back in range(0, 61):
            t = c_ms - back * 1000
            p, _ = sample_at(ts_arr, px_arr, t, "last_le")
            if p is None:
                break
            cur = "up" if p >= strike else "down"
            if cur == side:
                lead = back
            else:
                break
        leads.append(lead)
    leads = np.array(leads)
    if leads.size:
        for thr in (1, 2, 3, 5, 10, 20, 30):
            print(f"    outcome already locked >= {thr:2d}s before close: "
                  f"{(leads >= thr).mean()*100:5.1f}% of markets")
        print(f"    median lock-in lead = {np.median(leads):.0f}s  (n={leads.size})")


if __name__ == "__main__":
    main()
