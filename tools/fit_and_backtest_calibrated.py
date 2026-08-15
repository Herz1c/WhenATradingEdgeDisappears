"""Fit an isotonic-regression calibrator on the model's test split, then
run the maker backtest on OOS days with the calibrated model. Compare
to the uncalibrated baseline so we can see exactly what calibration does
to PnL.

What this does, end-to-end:
  1) Back up the original model.pkl to model.uncalibrated.pkl
  2) Score the TEST days (2026-05-15..20) -- which were NOT used for
     training the model -- to get (raw_p_up, actual_UP_label) pairs.
  3) Fit isotonic regression: raw_p_up -> calibrated_p_up.
  4) Attach the calibrator to the model as model._calibrator and save
     as model.calibrated.pkl  (we DO NOT overwrite model.pkl by default;
     pass --commit to do that.)
  5) Run the maker backtest on OOS days (2026-05-21, 22, 26, 29) with
     both the uncalibrated and calibrated models. Report side-by-side.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from feature_cleanup import clean_features  # noqa: E402

DC = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
ART_DIR = REPO / "artifacts_cleaned" / "model_02_fair_resolution" / "dense_close" / "lightgbm"
ART = ART_DIR / "model.pkl"
BACKUP = ART_DIR / "model.uncalibrated.pkl"
OUT_CAL = ART_DIR / "model.calibrated.pkl"

TEST_DAYS = ["2026-05-15", "2026-05-16", "2026-05-17",
             "2026-05-18", "2026-05-19", "2026-05-20"]
OOS_DAYS  = ["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-29"]

TTC_MIN_S, TTC_MAX_S = 10.0, 60.0


def predict_raw(model, feats, df):
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return model.predict_proba(X)[:, 1]


def load_days(dates):
    parts = []
    for d in dates:
        p = DC / f"{d}.parquet"
        if not p.exists(): continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null())
        df = df.filter(pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("down_token_best_bid").is_not_null())
        df = df.filter(pl.col("down_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    return pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")


def maker_sim(df, p_up, *, edge_threshold=0.07, min_down_price=0.30,
              max_pos_per_market=2, min_gap_s=10.0, size=1.0):
    """Same maker simulation we've been using. Posts midpoint, fills if
    future down_ask drops to limit before close."""
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()

    edge_dn = up_bid - p_up

    fut_ask = defaultdict(list)
    for i in range(len(df)):
        fut_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt = defaultdict(int)
    last_entry_ns = {}
    gap_ns = int(min_gap_s * 1e9)
    pnl = 0.0; n_sig = 0; n_fill = 0; n_win = 0
    pnl_by_day = defaultdict(float)
    fills_by_day = defaultdict(int)

    for i in range(len(df)):
        if not (TTC_MIN_S <= ttc[i] <= TTC_MAX_S): continue
        if edge_dn[i] < edge_threshold: continue
        if dn_ask[i] < min_down_price: continue
        if dn_ask[i] >= 0.98 or dn_ask[i] <= 0.02: continue
        slug = str(slugs[i])
        if pos_per_mkt[slug] >= max_pos_per_market: continue
        last = last_entry_ns.get(slug)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue
        limit = round((dn_bid[i] + dn_ask[i]) / 2.0, 2)
        if limit < min_down_price: continue
        if not (0.02 <= limit <= 0.98): continue
        n_sig += 1
        pos_per_mkt[slug] += 1
        last_entry_ns[slug] = int(snap_ts[i])
        filled = False
        for t, a in fut_ask[slug]:
            if t <= int(snap_ts[i]): continue
            if t >= int(close_ts[i]): break
            if a <= limit:
                filled = True; break
        if not filled: continue
        n_fill += 1
        won = (resolved[i] == 0)
        pnl_i = (1.0 if won else 0.0) - limit
        pnl += pnl_i * size
        pnl_by_day[str(dates[i])] += pnl_i * size
        fills_by_day[str(dates[i])] += 1
        if won: n_win += 1

    return {
        "n_sig": n_sig, "n_fill": n_fill, "n_win": n_win,
        "fill_rate": n_fill / max(1, n_sig),
        "win_rate": n_win / max(1, n_fill),
        "pnl": pnl,
        "pnl_per_fill": pnl / max(1, n_fill),
        "by_day": {d: {"fills": fills_by_day[d], "pnl": pnl_by_day[d]} for d in fills_by_day},
    }


def main():
    # 1. Back up the original
    if not BACKUP.exists():
        shutil.copy(ART, BACKUP)
        print(f"backed up uncalibrated -> {BACKUP.name}")
    else:
        print(f"backup already exists: {BACKUP.name}")

    model = joblib.load(ART)
    feats = list(json.loads((ART_DIR / "feature_importance.json").read_text()).keys())
    print(f"\nmodel: {type(model).__name__}, {len(feats)} features")

    # 2. Score TEST days for calibration fit
    print(f"\nscoring TEST days for calibration: {TEST_DAYS}")
    test_df = load_days(TEST_DAYS)
    test_df = test_df.filter((pl.col("t_to_close_s") > TTC_MIN_S) &
                              (pl.col("t_to_close_s") < TTC_MAX_S))
    print(f"  {len(test_df)} eligible snapshots")
    test_df_clean = clean_features(test_df)
    raw_p_up_test = predict_raw(model, feats, test_df_clean)
    y_up_test = test_df["resolved_side_label"].to_numpy().astype(int)
    print(f"  raw p_up: mean={raw_p_up_test.mean():.3f}  actual UP rate: {y_up_test.mean():.3f}"
          f"  gap: {(y_up_test.mean() - raw_p_up_test.mean())*100:+.1f}pp")

    # 3. Fit isotonic regression
    print("\nfitting isotonic regression calibrator...")
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    cal.fit(raw_p_up_test, y_up_test)
    # Quick check: post-calibration mean
    calibrated_p_test = cal.predict(raw_p_up_test)
    print(f"  post-calibration mean p_up on test: {calibrated_p_test.mean():.3f}"
          f"  (target = actual UP rate {y_up_test.mean():.3f})")

    # 4. Attach calibrator to model and save
    model._calibrator = cal
    joblib.dump(model, OUT_CAL)
    print(f"  saved calibrated model -> {OUT_CAL.name}")

    # 5. Backtest both on OOS
    print(f"\nloading OOS days: {OOS_DAYS}")
    oos_df = load_days(OOS_DAYS)
    print(f"  {len(oos_df)} eligible snapshots, {oos_df['market_slug'].n_unique()} markets")
    oos_clean = clean_features(oos_df)
    raw_p_up_oos = predict_raw(model, feats, oos_clean)
    cal_p_up_oos = cal.predict(raw_p_up_oos)
    y_oos = oos_df["resolved_side_label"].to_numpy().astype(int)
    print(f"\n  raw       p_up mean = {raw_p_up_oos.mean():.3f}  (actual UP rate {y_oos.mean():.3f}"
          f"  gap {(y_oos.mean()-raw_p_up_oos.mean())*100:+.1f}pp)")
    print(f"  calibrated p_up mean = {cal_p_up_oos.mean():.3f}  "
          f"(gap {(y_oos.mean()-cal_p_up_oos.mean())*100:+.1f}pp)")

    # Sweep thresholds for both models
    print(f"\nMaker backtest on OOS, sweeping thresholds:")
    print(f"{'model':<11} {'thr':>5} {'sig':>5} {'fill':>5} {'fill%':>5} {'win%':>5}"
          f" {'PnL':>8} {'$/fill':>7}")
    print("-" * 70)
    for label, p in [("uncalib", raw_p_up_oos), ("calibrated", cal_p_up_oos)]:
        for thr in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
            r = maker_sim(oos_df, p, edge_threshold=thr)
            print(f"{label:<11} {thr:>5.2f} {r['n_sig']:>5d} {r['n_fill']:>5d}"
                  f" {r['fill_rate']:>5.1%} {r['win_rate']:>5.1%}"
                  f" {r['pnl']:>+8.2f} {r['pnl_per_fill']:>+7.4f}")
        print("-" * 70)

    # 6. Reminder
    print(f"\nORIGINAL model.pkl is UNTOUCHED.")
    print(f"  uncalibrated backup: {BACKUP}")
    print(f"  calibrated version : {OUT_CAL}")
    print(f"\nTo deploy the calibrated model live, also need to remove the")
    print(f"calibrator-stripping guard in src/live_bot/feature_runtime.py")
    print(f"(currently raises if model._calibrator is not None).")


if __name__ == "__main__":
    main()
