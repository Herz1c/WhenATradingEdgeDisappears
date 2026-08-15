"""(1) RTDS coverage across all available raw days, and
(2) RTDS vs chainlink_public_delayed (the same Chainlink Data Streams BTC/USD
    feed scraped from the public data.chain.link page, i.e. live-chainlink-with-delay).

Confirms RTDS carries the genuine Chainlink price (value agreement at the same
event second) and quantifies HOW MUCH FRESHER RTDS is than the public delayed
feed (delivery-time lead).

Usage:
  py -3 tools/rtds_coverage_and_chainlink_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from binance_recorder.compression import iter_zstd_jsonl_with_options as iterz  # noqa: E402

RTDS_DIR = ROOT / "data" / "raw" / "polymarket" / "rtds" / "crypto_prices_chainlink" / "btc_usd"
PD_DIR = ROOT / "data" / "raw" / "chainlink_public_delayed" / "public_stream_page" / "BTCUSD"
MS = 1_000_000
SEC_NS = 1_000_000_000


def pctl(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


def rtds_day(d: str):
    """Return dict: chainlink_second -> (price, earliest_recv_ns), plus freshness list."""
    ddir = RTDS_DIR / d
    by_sec = {}
    fresh = []
    if not ddir.exists():
        return by_sec, fresh
    for path in sorted(ddir.glob("*.ws.jsonl.zst")):
        first = True
        for r in iterz(path, allow_truncated_stream=True, allow_partial_final_line=True):
            if r.get("record_type") != "chainlink_live_reference":
                first = False
                continue
            try:
                ts_ms = int(r["chainlink_ts_ms"]); px = float(r["btc_usd_price"]); recv = int(r["recv_ts_ns"])
            except (KeyError, TypeError, ValueError):
                first = False
                continue
            sec = ts_ms // 1000
            prev = by_sec.get(sec)
            if prev is None or recv < prev[1]:
                by_sec[sec] = (px, recv)
            if not first:  # exclude reconnect/catch-up burst at file start
                fresh.append((recv - ts_ms * MS) / MS)
            first = False
    return by_sec, fresh


def pd_day(d: str):
    """public_delayed: chainlink_event_second -> (price, earliest_recv_ns)."""
    ddir = PD_DIR / d
    by_sec = {}
    if not ddir.exists():
        return by_sec
    for path in sorted(ddir.glob("*.page.jsonl.zst")):
        for r in iterz(path, allow_truncated_stream=True, allow_partial_final_line=True):
            recv = r.get("recv_ts_ns"); disp = r.get("chainlink_display_ts"); px = r.get("btc_usd_price")
            if recv is None or disp is None or px is None:
                continue
            try:
                recv = int(recv); disp = int(disp); px = float(px)
            except (TypeError, ValueError):
                continue
            ev_ns = disp if abs(disp) >= 10_000_000_000 else disp * SEC_NS
            sec = ev_ns // SEC_NS
            prev = by_sec.get(sec)
            if prev is None or recv < prev[1]:
                by_sec[sec] = (px, recv)
    return by_sec


def main():
    dates = sorted(p.name for p in RTDS_DIR.iterdir() if p.is_dir()) if RTDS_DIR.exists() else []
    print(f"RTDS days available: {len(dates)}  ({dates[0]}..{dates[-1]})\n")

    print("[1] RTDS coverage per day")
    print(f"{'date':12s} {'sec_covered':>11s} {'cov%':>6s} {'gaps>5s':>8s} {'maxgap_s':>9s} {'fresh_p50':>9s} {'fresh_p90':>9s}")
    tot_sec = 0; all_fresh = []
    pd_price_err = []; pd_lead_ms = []; pd_matched = 0; pd_total = 0
    for d in dates:
        rt, fresh = rtds_day(d)
        if not rt:
            continue
        secs = np.array(sorted(rt.keys()), dtype=np.int64)
        diffs = np.diff(secs)
        gaps = int((diffs > 5).sum())
        maxgap = int(diffs.max()) if diffs.size else 0
        tot_sec += len(secs); all_fresh += fresh
        print(f"{d:12s} {len(secs):>11,} {len(secs)/864:>5.1f}% {gaps:>8d} {maxgap:>9d} "
              f"{pctl(fresh,.5):>8.0f}ms {pctl(fresh,.9):>8.0f}ms")

        # compare vs public delayed for this day
        pdd = pd_day(d)
        if pdd:
            common = set(rt.keys()) & set(pdd.keys())
            pd_total += len(pdd)
            pd_matched += len(common)
            for s in common:
                pd_price_err.append(rt[s][0] - pdd[s][0])
                pd_lead_ms.append((pdd[s][1] - rt[s][1]) / MS)  # >0 => RTDS delivered earlier

    print(f"\n  TOTAL sec covered: {tot_sec:,}  | freshness recv-chainlink steady: "
          f"p50={pctl(all_fresh,.5):.0f}ms p90={pctl(all_fresh,.9):.0f}ms p99={pctl(all_fresh,.99):.0f}ms")

    print("\n[2] RTDS vs chainlink_public_delayed (same Chainlink Data Streams btc/usd feed)")
    pe = np.array(pd_price_err); ld = np.array(pd_lead_ms)
    print(f"  matched event-seconds: {pd_matched:,} (of {pd_total:,} public_delayed seconds)")
    if pe.size:
        print(f"  PRICE agreement: medAE=${np.median(np.abs(pe)):.4f}  p95=${pctl(np.abs(pe),.95):.4f}  "
              f"max=${np.max(np.abs(pe)):.4f}  exact(<$0.01)={(np.abs(pe)<0.01).mean()*100:.1f}%")
        print(f"  DELIVERY lead (public_delayed_recv - rtds_recv), seconds:")
        print(f"    median={np.median(ld)/1000:.1f}s  p25={pctl(ld,.25)/1000:.1f}s  p75={pctl(ld,.75)/1000:.1f}s  "
              f"RTDS-earlier={ (ld>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
