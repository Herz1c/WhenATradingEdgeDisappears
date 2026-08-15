#!/usr/bin/env python3
"""Model 02 accuracy metrics on OOS resolution_dense_close.

Computes:
 - Calibration error per predicted-probability bin (|pred - actual|)
 - Mean calibration error
 - Mean absolute error (MAE) of predicted P(UP) vs binary outcome
 - Brier score
 - Log loss + baseline ratio
 - Hard-classification accuracy at threshold 0.5
 - Distribution of model predictions
"""
import io, sys, json
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features

feats = list(json.loads(Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/feature_importance.json").read_text()).keys())
model = joblib.load("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl")

OOS = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10","2026-05-11",
       "2026-05-12","2026-05-13","2026-05-14","2026-05-15","2026-05-16"]

parts = []
for d in OOS:
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    parts.append(df)
df = pl.concat(parts, how="diagonal")
df_clean = clean_features(df)

X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df_clean.columns:
        s = df_clean.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
p_up = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
if cal is not None: p_up = np.clip(cal.predict(p_up), 1e-6, 1 - 1e-6)

y = df["resolved_side_label"].to_numpy().astype(int)

print("=" * 78)
print(f"MODEL 02 (dense_close) ACCURACY — OOS May 7-16  (n = {len(y):,})")
print("=" * 78)

# Basic metrics
mae = np.mean(np.abs(p_up - y))
brier = np.mean((p_up - y) ** 2)
eps = 1e-9
log_loss = -np.mean(y * np.log(p_up + eps) + (1 - y) * np.log(1 - p_up + eps))
base_rate = y.mean()
log_loss_baseline = -np.mean(y * np.log(base_rate + eps) + (1 - y) * np.log(1 - base_rate + eps))

print(f"\nPoint-prediction metrics:")
print(f"  MAE    : {mae:.4f}  (mean |predicted P(UP) - realized|)")
print(f"  Brier  : {brier:.4f}  (lower is better; perfect = 0, random = 0.25)")
print(f"  Log loss: {log_loss:.4f}  (baseline constant {base_rate:.3f} = {log_loss_baseline:.4f})")
print(f"  Log-loss skill: {(log_loss_baseline - log_loss) / log_loss_baseline * 100:.2f}% improvement over constant")

# Hard-classification accuracy
pred_class = (p_up >= 0.5).astype(int)
hard_acc = (pred_class == y).mean()
print(f"\nHard-classification accuracy (threshold 0.5):")
print(f"  Accuracy: {hard_acc * 100:.2f}%  (random = 50%)")

# Calibration table
print(f"\n" + "=" * 78)
print("CALIBRATION TABLE — by 10 predicted-probability buckets")
print("=" * 78)
print(f"{'pred bucket':<14} {'n':>8} {'pred mean':>10} {'realized':>10} {'|pred-real|':>13} {'bias':>10}")
bins = [(i/10, (i+1)/10) for i in range(10)]
total_err = 0; total_n = 0
for lo, hi in bins:
    m = (p_up >= lo) & (p_up < hi)
    if m.sum() < 50: continue
    p_mean = p_up[m].mean()
    real = y[m].mean()
    err = abs(p_mean - real)
    bias = p_mean - real
    print(f"[{lo:.1f}, {hi:.1f})    {m.sum():>8,} {p_mean:>9.3f} {real:>9.3f} {err:>12.3f} {bias:>+9.3f}")
    total_err += err * m.sum(); total_n += m.sum()
print(f"\nWeighted mean |pred - realized|: {total_err / total_n:.4f}")
print(f"  (i.e., on average the model's predicted P(UP) is off by this much)")

# Distribution of predictions
print(f"\n" + "=" * 78)
print("PREDICTION DISTRIBUTION")
print("=" * 78)
print(f"  p_up percentiles: p1 {np.percentile(p_up, 1):.3f}  p10 {np.percentile(p_up, 10):.3f}  "
      f"median {np.median(p_up):.3f}  p90 {np.percentile(p_up, 90):.3f}  p99 {np.percentile(p_up, 99):.3f}")
print(f"  fraction predicting > 0.5 (UP-leaning): {(p_up > 0.5).mean() * 100:.1f}%")
print(f"  fraction predicting < 0.5 (DOWN-leaning): {(p_up < 0.5).mean() * 100:.1f}%")
print(f"  fraction near 0.5 ±0.05 (uncertain): {((p_up >= 0.45) & (p_up <= 0.55)).mean() * 100:.1f}%")

# Accuracy by confidence band
print(f"\n" + "=" * 78)
print("ACCURACY BY MODEL CONFIDENCE (distance from 0.5)")
print("=" * 78)
print(f"{'confidence':<18} {'n':>8} {'pred direction WR':>18} {'avg abs error':>15}")
conf = np.abs(p_up - 0.5)
for lo, hi, name in [(0.0, 0.05, "uncertain"), (0.05, 0.15, "slight"), (0.15, 0.30, "moderate"),
                     (0.30, 0.45, "strong"), (0.45, 0.50, "very strong")]:
    m = (conf >= lo) & (conf < hi)
    if m.sum() < 50: continue
    pred_dir = (p_up[m] > 0.5).astype(int)
    wr = (pred_dir == y[m]).mean()
    err = mae if m.sum() == 0 else np.mean(np.abs(p_up[m] - y[m]))
    print(f"{name:<18} {m.sum():>8,} {wr * 100:>16.1f}% {err:>14.4f}")

# OOS-specific accuracy for the actual trade conditions (filter applied)
print(f"\n" + "=" * 78)
print("ACCURACY ON ACTUAL STRATEGY FILLS (DOWN signal, edge ≥ $0.10, ttc 10-60s)")
print("=" * 78)
if "up_token_best_bid" in df.columns:
    df_filtered = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df_filtered = df_filtered.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    # Need to redo predictions on filtered
    df_clean2 = clean_features(df_filtered)
    X2 = np.zeros((len(df_clean2), len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df_clean2.columns:
            s = df_clean2.get_column(f)
            if s.dtype.is_numeric():
                v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                X2[:, i] = np.where(np.isfinite(v), v, 0.0)
    p_up2 = model.predict_proba(X2)[:, 1]
    if cal is not None: p_up2 = np.clip(cal.predict(p_up2), 1e-6, 1 - 1e-6)
    up_bid2 = df_filtered["up_token_best_bid"].to_numpy()
    ttc2 = df_filtered["t_to_close_s"].to_numpy()
    y2 = df_filtered["resolved_side_label"].to_numpy().astype(int)
    edge_dn = up_bid2 - p_up2
    fill_mask = (edge_dn >= 0.10) & (ttc2 >= 10) & (ttc2 <= 60)
    if fill_mask.sum() > 0:
        p_filt = p_up2[fill_mask]
        y_filt = y2[fill_mask]
        mae_f = np.mean(np.abs(p_filt - y_filt))
        print(f"  Fills (pre-dedup): n = {fill_mask.sum():,}")
        print(f"  Mean predicted P(UP): {p_filt.mean():.4f}")
        print(f"  Mean realized P(UP):  {y_filt.mean():.4f}")
        print(f"  MAE on fills:         {mae_f:.4f}")
        print(f"  → model says UP is {p_filt.mean()*100:.1f}% likely on average, actual is {y_filt.mean()*100:.1f}%")
        print(f"     gap of {abs(p_filt.mean() - y_filt.mean())*100:.1f}pp — the strategy exploits this")
