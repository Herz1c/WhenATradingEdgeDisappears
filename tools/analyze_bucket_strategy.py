"""Develop the one-entry-per-TTC-bucket strategy: break down PnL by bucket, and
measure how often a single market fires in multiple buckets at once.

Config: thr=0.15, fill[0.25,0.60], calib 04-22/04-30, 7 OOS days.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa
from search_robust_strategy import base_frame, entries_for, NOTIONAL  # noqa

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]
EVAL = ["2026-05-07", "2026-05-15", "2026-05-21",
        "2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
THR, LO, HI = 0.15, 0.25, 0.60
BUCKET_LABELS = ["(10,20]", "(20,30]", "(30,40]", "(40,50]", "(50,60]"]

model, feats, _ = load_model(ART)
df_cal = load_days(CALIB)
iso = fit_iso(score_model_p(model, feats, df_cal), df_cal["label_up"].to_numpy().astype(np.float64))
base = base_frame(model, feats, EVAL, iso)
e = entries_for(base, THR, LO, HI, 0.0).sort("recv_ts_ns")

bucket = e["bucket"].to_numpy()
pnl = e["pnl"].to_numpy()
won = e["won"].to_numpy()
fill = e["fill"].to_numpy()
slug = e["market_slug"].to_list()
n = len(pnl)
net = float(pnl.sum())

print(f"=== PER-BUCKET PnL (n={n}, net ${net:+.2f}) ===")
print(f"{'bucket':>9} {'n':>4} {'pnl':>9} {'%net':>6} {'win%':>6} {'avg_fill':>8} {'avg_pnl':>8}")
bucket_stats = {}
for b in range(5):
    m = bucket == b
    if not m.any():
        print(f"{BUCKET_LABELS[b]:>9} {0:>4}"); continue
    bp = float(pnl[m].sum())
    bucket_stats[BUCKET_LABELS[b]] = {"n": int(m.sum()), "pnl": bp,
        "pct_net": bp/net if net else None, "win": float(won[m].mean()),
        "avg_fill": float(fill[m].mean()), "avg_pnl": bp/int(m.sum())}
    print(f"{BUCKET_LABELS[b]:>9} {int(m.sum()):>4} {bp:>+9.2f} {100*bp/net if net else 0:>5.1f}% "
          f"{100*won[m].mean():>5.1f}% {fill[m].mean():>8.3f} {bp/int(m.sum()):>+8.3f}")

# --- entries per market (multi-bucket concurrency) ---
cnt = Counter(slug)
per_market_counts = list(cnt.values())
n_markets = len(cnt)
dist = Counter(per_market_counts)   # how many markets had k entries
print(f"\n=== ENTRIES PER MARKET ({n_markets} distinct markets traded, {n} entries) ===")
print(f"{'entries/market':>14} {'#markets':>9} {'%markets':>9} {'%of_entries':>11} {'pnl':>9} {'win%':>6}")
# need pnl/win by market-count group
mk_to_count = cnt
ent_count = np.array([mk_to_count[s] for s in slug])
for k in range(1, 6):
    nm = dist.get(k, 0)
    if nm == 0:
        print(f"{k:>14} {0:>9}"); continue
    mask = ent_count == k
    kp = float(pnl[mask].sum()); kw = float(won[mask].mean())
    print(f"{k:>14} {nm:>9} {100*nm/n_markets:>8.1f}% {100*mask.sum()/n:>10.1f}% "
          f"{kp:>+9.2f} {100*kw:>5.1f}%")

multi = sum(1 for v in per_market_counts if v >= 2)
print(f"\nmarkets with >=2 simultaneous bucket entries: {multi}/{n_markets} "
      f"({100*multi/n_markets:.1f}%)")
print(f"avg entries per traded market: {n/n_markets:.2f}")
ent_in_multi = int((ent_count >= 2).sum())
print(f"entries that belong to a multi-entry market: {ent_in_multi}/{n} "
      f"({100*ent_in_multi/n:.1f}%), their pnl ${float(pnl[ent_count>=2].sum()):+.2f}")
print(f"entries from single-entry markets: {n-ent_in_multi}/{n}, "
      f"pnl ${float(pnl[ent_count==1].sum()):+.2f}")

out = ART / "backtests" / "bucket_strategy_analysis.json"
out.write_text(json.dumps({
    "config": {"thr": THR, "fill": [LO, HI], "eval_days": EVAL},
    "n_entries": n, "net": net, "n_markets": n_markets,
    "per_bucket": bucket_stats,
    "entries_per_market_dist": {str(k): dist.get(k, 0) for k in range(1, 6)},
    "pct_markets_multi": multi/n_markets,
    "avg_entries_per_market": n/n_markets,
}, indent=2, default=float))
print(f"\nsaved -> {out}")
