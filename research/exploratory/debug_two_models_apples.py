#!/usr/bin/env python3
"""Apples-to-apples head-to-head on the SAME OOS data (May 15-20):
  - log_loss
  - Brier
  - ECE
  - hard accuracy
  - calibration table (predicted vs realized per bin)
  - distribution of p_up

This shows numerically why the new model has 'better' log_loss but
trades WORSE than the old model.
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

OOS = ["2026-05-15","2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20"]

NEW = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
OLD = Path("artifacts_archive/model_02_v1_2026-05-18_baseline_70pct_winrate")

# Load both models
nm = joblib.load(NEW / "model.pkl"); nf = list(json.loads((NEW/"feature_importance.json").read_text()).keys())
om = joblib.load(OLD / "model.pkl"); of = list(json.loads((OLD/"feature_importance.json").read_text()).keys())

parts = []
for d in OOS:
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    parts.append(df)
df = pl.concat(parts, how="diagonal")
df_clean = clean_features(df)

def build_X(feats):
    X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df_clean.columns:
            s = df_clean.get_column(f)
            if s.dtype.is_numeric():
                v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X

def predict(m, feats, cal=True):
    raw = m.predict_proba(build_X(feats))[:, 1]
    if cal:
        c = getattr(m, "_calibrator", None)
        if c is not None: return np.clip(c.predict(raw), 1e-6, 1-1e-6)
    return raw

y = df["resolved_side_label"].to_numpy().astype(int)
p_new = predict(nm, nf, cal=True)
p_old = predict(om, of, cal=True)
p_new_raw = predict(nm, nf, cal=False)
p_old_raw = predict(om, of, cal=False)

eps = 1e-9
def metrics(name, p):
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    br = np.mean((p - y) ** 2)
    hard = ((p >= 0.5).astype(int) == y).mean()
    mae = np.mean(np.abs(p - y))
    # ECE 10 bins
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0: continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"name": name, "log_loss": ll, "brier": br, "ece": ece, "hard_acc": hard, "mae": mae,
            "mean_p_up": p.mean(), "p25_p_up": np.percentile(p, 25), "p75_p_up": np.percentile(p, 75)}

results = []
for name, p in [("NEW calibrated", p_new),
                ("NEW raw",        p_new_raw),
                ("OLD calibrated", p_old),
                ("OLD raw",        p_old_raw)]:
    r = metrics(name, p); results.append(r)

print(f"\nHead-to-head on May 15-20 (n={len(y):,}):")
print(f"{'model':<20} {'log_loss':>10} {'brier':>8} {'ECE':>8} {'hard%':>8} {'MAE':>8} {'mean p_up':>11}")
for r in results:
    print(f"{r['name']:<20} {r['log_loss']:>9.4f} {r['brier']:>7.4f} {r['ece']:>7.4f} "
          f"{r['hard_acc']*100:>7.2f} {r['mae']:>7.4f} {r['mean_p_up']:>10.3f}")

# Calibration table head-to-head
print(f"\n{'='*78}")
print("CALIBRATION TABLE on May 15-20  (per predicted-probability bucket)")
print("='*78")
print(f"{'pred bin':<14} {'NEW pred mean':>15} {'NEW realized':>14} {'NEW err':>10} | "
      f"{'OLD pred mean':>15} {'OLD realized':>14} {'OLD err':>10}")
bins = [(i/10, (i+1)/10) for i in range(10)]
for lo, hi in bins:
    mn = (p_new >= lo) & (p_new < hi)
    mo = (p_old >= lo) & (p_old < hi)
    new_str = f"{'-':>15} {'-':>14} {'-':>10}" if mn.sum() < 50 else \
              f"{p_new[mn].mean():>15.3f} {y[mn].mean():>14.3f} {p_new[mn].mean()-y[mn].mean():>+10.3f}"
    old_str = f"{'-':>15} {'-':>14} {'-':>10}" if mo.sum() < 50 else \
              f"{p_old[mo].mean():>15.3f} {y[mo].mean():>14.3f} {p_old[mo].mean()-y[mo].mean():>+10.3f}"
    n_new = mn.sum() if mn.sum() >= 50 else 0
    n_old = mo.sum() if mo.sum() >= 50 else 0
    print(f"[{lo:.1f}, {hi:.1f}) n=({n_new:>6}/{n_old:>6})  {new_str}  |  {old_str}")

# Bin where strategy actually fires (DOWN signal: low p_up AND market expects higher)
print(f"\n{'='*78}")
print("CALIBRATION BY p_up BUCKET — RESTRICTED TO STRATEGY FILL CONDITIONS")
print("(ttc 10-60s, up_bid > p_up + $0.10)  → these are the cells that drive PnL")
print("'='*78")
up_bid = df["up_token_best_bid"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
ttc_band = (ttc >= 10) & (ttc <= 60)
for name, p in [("NEW calibrated", p_new), ("OLD calibrated", p_old)]:
    edge_dn = up_bid - p
    fire = ttc_band & (edge_dn >= 0.10)
    print(f"\n  {name}: fires on {fire.sum():,} snapshots ({fire.mean()*100:.2f}% of OOS)")
    if fire.sum() == 0: continue
    p_fire = p[fire]; y_fire = y[fire]
    print(f"    mean p_up (in fired)  : {p_fire.mean():.3f}")
    print(f"    realized P(UP) in fired: {y_fire.mean():.3f}")
    print(f"    realized P(DOWN) WR    : {(1 - y_fire.mean())*100:.1f}%")
    print(f"    model error vs realized: {p_fire.mean() - y_fire.mean():+.3f}")
    print(f"    Note: WR for DOWN buyer = 1 - realized P(UP)")
