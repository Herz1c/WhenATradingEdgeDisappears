#!/usr/bin/env python3
"""Compare new Model 02 (raw vs calibrated) vs the archived old Model 02
on the same OOS test window (May 15-20).

Tests:
  A) New model RAW predictions (no isotonic) -> strategy backtest
  B) New model CALIBRATED predictions -> strategy backtest
  C) Old model (from archive) -> strategy backtest
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

NEW = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
OLD = Path("artifacts_archive/model_02_v1_2026-05-18_baseline_70pct_winrate")

new_model = joblib.load(NEW / "model.pkl")
new_feats = list(json.loads((NEW / "feature_importance.json").read_text()).keys())
old_model = joblib.load(OLD / "model.pkl")
old_feats = list(json.loads((OLD / "feature_importance.json").read_text()).keys())

print(f"NEW model: features={len(new_feats)}, has _calibrator={hasattr(new_model, '_calibrator')}")
print(f"OLD model: features={len(old_feats)}, has _calibrator={hasattr(old_model, '_calibrator')}")
print(f"Feature lists identical: {new_feats == old_feats}")

# --- Load OOS once ---------------------------------------------------------
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
print(f"\nOOS snapshots: {len(df):,}")

# --- Predict with each variant --------------------------------------------
def predict(model, feats, calibrated=True):
    X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df_clean.columns:
            s = df_clean.get_column(f)
            if s.dtype.is_numeric():
                v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    if calibrated:
        cal = getattr(model, "_calibrator", None)
        if cal is not None:
            return np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw

p_new_raw = predict(new_model, new_feats, calibrated=False)
p_new_cal = predict(new_model, new_feats, calibrated=True)
p_old_cal = predict(old_model, old_feats, calibrated=True)

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
up_ask = df["up_token_best_ask"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
ttc_band = (ttc >= 10) & (ttc <= 60)

# --- Backtest helper ------------------------------------------------------
def backtest(p_up, label, thresholds):
    print(f"\n=== {label} ===")
    print(f"  p_up distribution: mean={p_up.mean():.3f}  med={np.median(p_up):.3f}  "
          f"p10={np.percentile(p_up,10):.3f}  p90={np.percentile(p_up,90):.3f}")
    edge_dn = up_bid - p_up
    down_ask = 1.0 - up_bid
    # Overall direction accuracy (as a sanity check)
    pred_class = (p_up >= 0.5).astype(int)
    hard_acc = (pred_class == resolved).mean()
    print(f"  hard direction accuracy: {hard_acc*100:.2f}%")
    fee_calcs = {}
    for thr in thresholds:
        mask = ttc_band & (edge_dn >= thr)
        idx = np.where(mask)[0]
        seen = set(); fills = []
        for i in idx:
            k = str(market_arr[i])
            if k in seen: continue
            seen.add(k); fills.append(i)
        if not fills:
            print(f"  thr=${thr:.3f}: 0 fills"); continue
        fills = np.array(fills, dtype=int)
        wins = (resolved[fills] == 0).astype(int)
        entry = down_ask[fills]
        fees = np.empty(len(fills))
        for j, i in enumerate(fills):
            fc = fee_calcs.setdefault(str(date_arr[i]), FeeCalculator.for_date(str(date_arr[i])))
            fees[j] = fc.taker_fee_usd(price=float(entry[j]), size=1.0)
        gross = np.where(wins == 1, 1.0 - entry, -entry)
        net = gross - fees
        print(f"  thr=${thr:.3f}  fills={len(fills):>4}  win%={wins.mean()*100:>5.1f}  "
              f"pnl=${net.sum():>+7.2f}  per-fill=${net.mean():>+7.4f}  "
              f"avg_entry=${entry.mean():.3f}")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
backtest(p_new_raw, "NEW MODEL — RAW (no calibration)", thresholds)
backtest(p_new_cal, "NEW MODEL — CALIBRATED (isotonic)", thresholds)
backtest(p_old_cal, "OLD MODEL (archived baseline, w/ its own calibrator)", thresholds)
