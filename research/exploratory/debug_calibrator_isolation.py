#!/usr/bin/env python3
"""Isolate the effect of the calibrator on each model.

Tests the threshold sweep with RAW (uncalibrated) and CALIBRATED predictions
for both OLD (archived) and NEW (just retrained, clean) models on May 15-20.

Goal: figure out whether the old model's edge comes from
  (a) its raw LightGBM (different feature/training-data combination)
  (b) its specific calibrator (fit on different data)
  (c) the interaction of both

Then we know what to preserve when training a NEW model with extended data.
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

OOS = ["2026-05-15","2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20"]
NEW = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
OLD = Path("artifacts_archive/model_02_v1_2026-05-18_baseline_70pct_winrate")

nm = joblib.load(NEW / "model.pkl"); nf = list(json.loads((NEW/"feature_importance.json").read_text()).keys())
om = joblib.load(OLD / "model.pkl"); of = list(json.loads((OLD/"feature_importance.json").read_text()).keys())

# Show calibrator metadata for both
print("NEW model calibration_meta:", getattr(nm, "_calibration_meta", "N/A"))
print("OLD model calibration_meta:", getattr(om, "_calibration_meta", "N/A"))
new_cal = getattr(nm, "_calibrator", None)
old_cal = getattr(om, "_calibrator", None)
if new_cal: print(f"NEW calibrator: {type(new_cal).__name__}, n_thresholds={len(getattr(new_cal,'X_thresholds_',[]))}")
if old_cal: print(f"OLD calibrator: {type(old_cal).__name__}, n_thresholds={len(getattr(old_cal,'X_thresholds_',[]))}")

# --- Load OOS once ---------------------------------------------------------
parts = []
for d in OOS:
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug","snapshot_ts_ns"])
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

p_new_raw = nm.predict_proba(build_X(nf))[:, 1]
p_old_raw = om.predict_proba(build_X(of))[:, 1]
p_new_cal = np.clip(new_cal.predict(p_new_raw), 1e-6, 1-1e-6) if new_cal is not None else p_new_raw
p_old_cal = np.clip(old_cal.predict(p_old_raw), 1e-6, 1-1e-6) if old_cal is not None else p_old_raw

# Also: NEW raw with OLD calibrator applied (mix experiment)
p_new_oldcal = np.clip(old_cal.predict(p_new_raw), 1e-6, 1-1e-6) if old_cal is not None else None
# And: OLD raw with NEW calibrator (other mix)
p_old_newcal = np.clip(new_cal.predict(p_old_raw), 1e-6, 1-1e-6) if new_cal is not None else None

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
ttc_band = (ttc >= 10) & (ttc <= 60)

def backtest(p, label, thresholds=None):
    edge_dn = up_bid - p
    cand = edge_dn[ttc_band & (edge_dn > 0)]
    if len(cand) == 0:
        print(f"\n{label}: no positive-edge candidates"); return
    print(f"\n{label}")
    print(f"  p_up: mean={p.mean():.3f}  p25={np.percentile(p,25):.3f}  p75={np.percentile(p,75):.3f}")
    print(f"  edge_dn pct: p50=${np.percentile(cand,50):.3f}  p70=${np.percentile(cand,70):.3f}  p90=${np.percentile(cand,90):.3f}")
    if thresholds is None:
        thresholds = [0.10, np.percentile(cand,50), np.percentile(cand,70), np.percentile(cand,90)]
        labels = ["$0.10", "p50", "p70", "p90"]
    else:
        labels = [f"${t:.3f}" for t in thresholds]
    fee_calcs = {}
    for thr, lab in zip(thresholds, labels):
        mask = ttc_band & (edge_dn >= thr)
        idx = np.where(mask)[0]
        seen = set(); fills = []
        for i in idx:
            k = str(market_arr[i])
            if k in seen: continue
            seen.add(k); fills.append(i)
        if not fills:
            print(f"    {lab:>10}: 0 fills"); continue
        fills = np.array(fills, dtype=int)
        wins = (resolved[fills] == 0).astype(int)
        entry = 1.0 - up_bid[fills]
        fees = np.array([fee_calcs.setdefault(str(date_arr[i]), FeeCalculator.for_date(str(date_arr[i]))).taker_fee_usd(price=float(entry[j]), size=1.0) for j, i in enumerate(fills)])
        gross = np.where(wins == 1, 1.0 - entry, -entry)
        net = gross - fees
        # per day positive count
        daily_pnl = {}
        for i, n in zip(fills, net):
            d = str(date_arr[i])
            daily_pnl[d] = daily_pnl.get(d, 0.0) + float(n)
        all_pos = all(v > 0 for v in daily_pnl.values())
        print(f"    thr={lab:>9} (=${thr:.4f})  fills={len(fills):>4}  WR={wins.mean()*100:>5.1f}%  "
              f"pnl=${net.sum():>+8.2f}  per-fill=${net.mean():>+7.4f}  allPos={'Y' if all_pos else 'N'}")

backtest(p_old_cal, "1) OLD raw + OLD calibrator   [BASELINE we want]")
backtest(p_old_raw, "2) OLD raw, NO calibrator")
backtest(p_new_cal, "3) NEW raw + NEW calibrator   [current retrain]")
backtest(p_new_raw, "4) NEW raw, NO calibrator")
if p_new_oldcal is not None: backtest(p_new_oldcal, "5) NEW raw + OLD calibrator   [interesting mix]")
if p_old_newcal is not None: backtest(p_old_newcal, "6) OLD raw + NEW calibrator   [interesting mix]")
