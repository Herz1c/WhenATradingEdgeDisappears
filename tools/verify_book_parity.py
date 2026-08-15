"""Verify the LIVE book is actually correct — the REAL test that mid_sum=1.0
could not give. Compares the dense live book-audit log (book_audit_<date>.jsonl,
written by the bot's _book_audit_loop) to the recorder-derived dataset book at
the same instants, and reports the correlation of implied_p_up.

PASS = corr > 0.95 (live book tracks the true book). Anything lower means the
live book is still drifted/desynced and the bot must NOT trade live.

Usage:
  py -3 tools/verify_book_parity.py 2026-06-29
  py -3 tools/verify_book_parity.py 2026-06-29 08:45   # only rows since HH:MM UTC
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else None
    hhmm = sys.argv[2] if len(sys.argv) > 2 else None
    audit_dir = ROOT / "logs" / "fair_value_bot"
    if date is None:
        files = sorted(audit_dir.glob("book_audit_*.jsonl"))
        if not files:
            print("no book_audit_*.jsonl yet — run the bot with FV_BOOK_AUDIT=1 first."); return
        date = files[-1].name[len("book_audit_"):-len(".jsonl")]
    af = audit_dir / f"book_audit_{date}.jsonl"
    dsf = ROOT / "data" / "datasets" / "fair_value_v1" / f"{date}.parquet"
    if not af.exists():
        print(f"missing {af}"); return
    if not dsf.exists():
        print(f"missing dataset {dsf} — build it: py -3 tools/build_fair_value_dataset.py --dates {date} --raw-root data/raw --cex coinbase --workers 8 --rebuild"); return

    live = pd.DataFrame(json.loads(l) for l in af.open(encoding="utf-8"))
    if hhmm:
        h, m = map(int, hhmm.split(":"))
        t0 = int(dt.datetime(*map(int, date.split("-")), h, m, tzinfo=dt.timezone.utc).timestamp() * 1e9)
        live = live[live["ts"] >= t0].copy()
    ds = pd.read_parquet(dsf)
    suffix = f" since {hhmm} UTC" if hhmm else ""
    print(f"live audit rows={len(live)}  dataset rows={len(ds)}  date={date}{suffix}")

    rows = []
    for slug, g in live.groupby("slug"):
        sub = ds[ds["market_slug"] == slug]
        if len(sub) == 0:
            continue
        dn = sub["now_ns"].to_numpy(); dimp = sub["implied_p_up"].to_numpy()
        for r in g.itertuples():
            i = int(np.abs(dn - r.ts).argmin())
            dt_s = abs(dn[i] - r.ts) / 1e9
            if dt_s > 3.0:
                continue  # no close recorder point
            rows.append((r.implied_p_up, float(dimp[i]), r.mid_sum, dt_s, getattr(r, "pm_lag_s", np.nan)))
    if not rows:
        print("no matched points (dataset may not cover the audit window)"); return
    m = pd.DataFrame(rows, columns=["live", "rec", "mid_sum", "dt", "pm_lag_s"])
    corr = np.corrcoef(m["live"], m["rec"])[0, 1]
    mad = (m["live"] - m["rec"]).abs().mean()
    print(f"\nmatched {len(m)} book snapshots (median dt={m['dt'].median()*1000:.0f}ms)")
    print(f"  implied_p_up:  corr = {corr:.3f}   mean|diff| = {mad:.3f}")
    print(f"  live mid_sum==1.0 on {(m['mid_sum']==1.0).mean()*100:.0f}% (note: this was the FALSE signal)")
    if m["pm_lag_s"].notna().any():
        print(f"  pm recorder lag: median={m['pm_lag_s'].median():.2f}s p95={m['pm_lag_s'].quantile(0.95):.2f}s")
    print()
    if corr > 0.95:
        print(f"  PASS [OK]  corr {corr:.3f} > 0.95 — live book tracks the true book. Safe(r) to trade.")
    else:
        print(f"  FAIL [X]  corr {corr:.3f} <= 0.95 — live book STILL diverges. Do NOT trade live.")


if __name__ == "__main__":
    main()
