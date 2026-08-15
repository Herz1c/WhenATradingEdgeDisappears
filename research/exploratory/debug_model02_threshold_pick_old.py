#!/usr/bin/env python3
"""SAME threshold sweep as debug_model02_threshold_pick.py but with the
ARCHIVED Model 02 (artifacts_archive/model_02_v1_2026-05-18_baseline_70pct_winrate)
so we can compare new vs old apples-to-apples on May 15-20.
"""
import io, json, sys
from pathlib import Path
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features
from backtest.fees import FeeCalculator

OOS_DATES = ["2026-05-15", "2026-05-16", "2026-05-17",
             "2026-05-18", "2026-05-19", "2026-05-20"]

ART = Path("artifacts_archive/model_02_v1_2026-05-18_baseline_70pct_winrate")
model = joblib.load(ART / "model.pkl")
feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
print(f"Loaded ARCHIVED Model 02 — {len(feats)} features")

parts = []
for d in OOS_DATES:
    p = Path(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    if not p.exists(): continue
    df = pl.read_parquet(p)
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug", "snapshot_ts_ns"])
df_clean = clean_features(df)
print(f"Loaded {len(df):,} OOS snapshots across {len(OOS_DATES)} days")

X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df_clean.columns:
        s = df_clean.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
raw = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
if cal is not None:
    p_up = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    print("Using calibrated predictions (isotonic) — old model's calibrator")
else:
    p_up = raw
    print("No calibrator on old model")

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
up_ask = df["up_token_best_ask"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid
ttc_band = (ttc >= 10) & (ttc <= 60)

candidate_mask = ttc_band & (edge_dn > 0)
candidate_edges = edge_dn[candidate_mask]
print(f"\nedge_dn distribution on positive-edge candidates (n={len(candidate_edges):,}):")
for p in [50, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]:
    print(f"  p{p:>2}: ${np.percentile(candidate_edges, p):.4f}")


def backtest(thr: float) -> dict:
    fee_calcs = {}
    mask = ttc_band & (edge_dn >= thr)
    idx = np.where(mask)[0]
    seen = set(); fills = []
    for i in idx:
        k = str(market_arr[i])
        if k in seen: continue
        seen.add(k); fills.append(i)
    fills = np.array(fills, dtype=int)
    if len(fills) == 0:
        return {"thr": thr, "fills": 0}
    wins = (resolved[fills] == 0).astype(int)
    entry = down_ask[fills]
    fees = np.empty(len(fills))
    for j, i in enumerate(fills):
        d = str(date_arr[i])
        fc = fee_calcs.setdefault(d, FeeCalculator.for_date(d))
        fees[j] = fc.taker_fee_usd(price=float(entry[j]), size=1.0)
    gross = np.where(wins == 1, 1.0 - entry, -entry)
    net = gross - fees
    daily = {}
    for i, pnl in zip(fills, net):
        d = str(date_arr[i])
        daily[d] = daily.get(d, 0.0) + float(pnl)
    worst_day = min(daily.values()) if daily else 0.0
    n_pos_days = sum(1 for v in daily.values() if v > 0)
    chrono_idx = np.argsort(df["snapshot_ts_ns"].to_numpy()[fills])
    chrono_pnl = net[chrono_idx]
    running = 0.0; peak = 0.0; max_dd = 0.0
    for p in chrono_pnl:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return {
        "thr": thr, "fills": int(len(fills)), "wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "total_pnl_usd": float(net.sum()),
        "per_fill_pnl_usd": float(net.mean()),
        "weekly_pnl_usd": float(net.sum() / 6 * 7),
        "max_drawdown_usd": float(max_dd),
        "worst_day_pnl_usd": float(worst_day),
        "n_pos_days": int(n_pos_days),
        "n_days": len(daily),
        "all_pos_days": bool(n_pos_days == len(daily)),
        "daily": {d: float(v) for d, v in daily.items()},
    }


# Use SAME percentile grid as new model sweep, AND also include $0.10 (the old
# model's historical deployment threshold) for direct comparison
percentiles = [0, 50, 60, 70, 75, 80, 82, 84, 86, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99]
print(f"\n{'pctile':>7} {'$thr':>9} {'fills':>7} {'win%':>6} {'pnl':>9} {'pnl/fill':>10} "
      f"{'wkly':>8} {'maxDD':>8} {'worst_day':>10} {'allPos':>7}")
print("-" * 95)
results = []
for pct in percentiles:
    thr = 0.0 if pct == 0 else float(np.percentile(candidate_edges, pct))
    r = backtest(thr)
    if r["fills"] == 0:
        print(f"  p{pct:>2} {thr:>+8.4f} {0:>7} (no fills)"); continue
    print(f"  p{pct:>2} {thr:>+8.4f} {r['fills']:>7} {r['win_rate']*100:>5.1f}% "
          f"${r['total_pnl_usd']:>+8.2f} ${r['per_fill_pnl_usd']:>+9.4f} "
          f"${r['weekly_pnl_usd']:>+7.2f} ${r['max_drawdown_usd']:>+7.2f} "
          f"${r['worst_day_pnl_usd']:>+9.2f} {('YES' if r['all_pos_days'] else 'NO'):>7}")
    r["percentile"] = pct
    results.append(r)

# Also test the OLD model's original deployment threshold of $0.10
print(f"\n--- Old model's historical \$0.10 threshold ---")
r10 = backtest(0.10)
print(f"  \$0.10: fills={r10['fills']}  win%={r10['win_rate']*100:.1f}  "
      f"total=${r10['total_pnl_usd']:+.2f}  per-fill=${r10['per_fill_pnl_usd']:+.4f}  "
      f"weekly=${r10['weekly_pnl_usd']:+.2f}  maxDD=${r10['max_drawdown_usd']:.2f}")
print(f"  per-day: {r10['daily']}")

# Best PnL + best WR analysis
sorted_pnl = sorted(results, key=lambda r: r["total_pnl_usd"], reverse=True)
sorted_wr = sorted(results, key=lambda r: r["win_rate"], reverse=True)
print(f"\nBest-PnL: p{sorted_pnl[0]['percentile']} thr=${sorted_pnl[0]['thr']:.4f}  "
      f"total=${sorted_pnl[0]['total_pnl_usd']:+.2f}  WR={sorted_pnl[0]['win_rate']*100:.1f}")
print(f"Best-WR : p{sorted_wr[0]['percentile']} thr=${sorted_wr[0]['thr']:.4f}  "
      f"WR={sorted_wr[0]['win_rate']*100:.1f}  total=${sorted_wr[0]['total_pnl_usd']:+.2f}")

Path("docs").mkdir(exist_ok=True)
Path("docs/model02_threshold_pick_OLD_on_15_to_20.json").write_text(json.dumps({
    "oos_dates": OOS_DATES,
    "model_artifact": str(ART),
    "sweep": results,
    "historical_thr_0_10": r10,
}, indent=2, default=str))
print(f"\nSaved to docs/model02_threshold_pick_OLD_on_15_to_20.json")
