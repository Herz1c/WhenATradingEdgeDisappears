"""Rank EVERY model feature by how much the LIVE bot's value diverges from the
dataset (backtest) at the same instant. Reads the full-feature book_audit log
(written when FV_BOOK_AUDIT=1 — each row has `f`={all features} and p_model) and
matches each live snapshot to the recorder-built dataset row.

The features at the top of the list are what's making live decisions differ from
the backtest. Run after collecting ~30 min of full-feature audit.

Usage:
  py -3 tools/feature_divergence.py 2026-06-28 18:57   # date + HH:MM UTC start
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else dt.datetime.now(dt.timezone.utc).date().isoformat()
    hhmm = sys.argv[2] if len(sys.argv) > 2 else "00:00"
    h, m = map(int, hhmm.split(":"))
    t0 = int(dt.datetime(*map(int, date.split("-")), h, m, tzinfo=dt.timezone.utc).timestamp() * 1e9)

    af = ROOT / "logs" / "fair_value_bot" / f"book_audit_{date}.jsonl"
    dsf = ROOT / "data" / "datasets" / "fair_value_v1" / f"{date}.parquet"
    rows = [json.loads(l) for l in af.open(encoding="utf-8")]
    rows = [r for r in rows if r["ts"] >= t0 and r.get("f")]
    if not rows:
        print("no full-feature audit rows since start (need FV_BOOK_AUDIT=1 run with the new logging)"); return
    ds = pd.read_parquet(dsf)
    print(f"live audit rows (full-feature): {len(rows)} since {date} {hhmm} UTC | dataset rows {len(ds)}")

    # match each audit row to nearest dataset row of same market (tight 2s)
    feat_names = set()
    pairs = []  # (live_feats, dataset_row, live_pmodel, dataset_pmodel-not-here)
    dsg = {s: g for s, g in ds.groupby("market_slug")}
    for r in rows:
        g = dsg.get(r["slug"])
        if g is None:
            continue
        dn = g["now_ns"].to_numpy()
        i = int(np.abs(dn - r["ts"]).argmin())
        if abs(dn[i] - r["ts"]) / 1e9 > 2.0:
            continue
        drow = g.iloc[i]
        pairs.append((r["f"], drow, r.get("p_model")))
        feat_names |= set(r["f"].keys())
    if not pairs:
        print("no matched points (dataset may lack these markets)"); return
    print(f"matched {len(pairs)} snapshots\n")

    # per-feature divergence
    out = []
    for fn in sorted(feat_names):
        lv, dv = [], []
        for lf, drow, _ in pairs:
            if fn in lf and fn in drow and lf[fn] is not None:
                try:
                    a = float(lf[fn]); b = float(drow[fn])
                    if np.isfinite(a) and np.isfinite(b):
                        lv.append(a); dv.append(b)
                except (TypeError, ValueError):
                    pass
        if len(lv) < 10:
            continue
        lv = np.array(lv); dv = np.array(dv)
        mad = np.abs(lv - dv).mean()
        rng = (dv.max() - dv.min()) or 1.0
        nmad = mad / rng  # normalized by the feature's range (comparable across features)
        corr = np.corrcoef(lv, dv)[0, 1] if lv.std() > 0 and dv.std() > 0 else float("nan")
        out.append((fn, mad, nmad, corr, len(lv)))
    # rank by normalized divergence (worst first)
    out.sort(key=lambda x: -x[2])
    print(f"{'feature':28}{'mean|diff|':>11}{'norm':>7}{'corr':>7}{'n':>6}")
    for fn, mad, nmad, corr, n in out[:25]:
        flag = "  <-- diverges" if (nmad > 0.15 or (corr == corr and corr < 0.8)) else ""
        print(f"{fn:28}{mad:>11.4f}{nmad:>7.2f}{corr:>7.2f}{n:>6}{flag}")

    # p_model divergence
    lpm = [p for (_, _, p) in pairs if p is not None]
    dpm = []
    for lf, drow, p in pairs:
        if p is None or "p" not in drow:
            continue
    print("\n(p_model lives in the audit; dataset p must be computed separately — see verify scripts)")


if __name__ == "__main__":
    main()
