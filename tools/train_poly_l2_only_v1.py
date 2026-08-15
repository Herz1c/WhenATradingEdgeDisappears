"""Train poly_l2_only_v1 — a LightGBM binary classifier on the
polymarket_l2_last60s_v1 dataset.

Predicts P(market resolves Up) at every L2 tick in the [-60s, -10s] window
from Polymarket WS features only. Single source of truth: features used
here are the exact 61 columns emitted by src/poly_l2_only/extractor.py, so
inference at train time and live time runs the same code.

Memory plan (16 GB host):
  - 88.4 M rows × 59 features × 4 bytes ≈ 21 GB raw → cannot fit.
  - Subsample to ~1500 rows/market for training (~10 M rows). LightGBM bins
    to 255 buckets so the in-memory Dataset is ~600 MB once constructed.
  - Validation: 15% of TRAIN markets held out by market_slug (GroupKFold-ish
    single split). Used for early stopping.
  - OOS test: untouched, full 19.5 M rows, predicted in 1 M-row chunks.

Calibration: isotonic regression on val-fold predictions vs labels, attached
as `_calibrator` on the saved model (so the live runtime can apply it).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# Make src importable for extractor's FEATURE_COLUMNS.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from poly_l2_only.extractor import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "datasets" / "polymarket_l2_last60s_v1"
ARTIFACT_DIR = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v1"

# Drop features that the sanity report flagged as dead.
DROP_FEATURES = {"neg_risk", "spread_diff"}
ACTIVE_FEATURES: List[str] = [c for c in FEATURE_COLUMNS if c not in DROP_FEATURES]

# Tuning knobs.
ROWS_PER_MARKET_TRAIN = 1500   # cap per-market rows in training sample
ROWS_PER_MARKET_VAL = 600
VAL_MARKET_FRAC = 0.15         # held-out fraction of train markets (early stop)
RANDOM_SEED = 42
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=2000,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    max_bin=255,
    verbosity=-1,
    seed=RANDOM_SEED,
)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

PRED_CHUNK_ROWS = 1_000_000


def ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def load_split(split: str, rows_per_market: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all parquets, filter to `split`, sample `rows_per_market` per market.

    Returns (X float32, y int8, group_ids int32 per row, market_slug_list).
    """
    print(f"[{ts()}] loading {split} split (rows_per_market={rows_per_market}) ...", flush=True)
    lf = pl.scan_parquet(str(DATA_DIR / "*.parquet")).filter(pl.col("split") == split)
    # Per-market head sample. The rows are emitted in time order so head() gives
    # a deterministic, time-contiguous slice. Random sampling is overkill for a
    # baseline and would require holding the full split in memory.
    sampled = (
        lf.with_columns(pl.int_range(pl.len()).over("market_slug").alias("_idx_in_mkt"))
          .filter(pl.col("_idx_in_mkt") < rows_per_market)
          .drop("_idx_in_mkt")
          .collect(engine="streaming")
    )
    n = sampled.height
    print(f"[{ts()}]   collected {n:,} rows from {sampled['market_slug'].n_unique():,} markets", flush=True)

    # Feature matrix (float32) + label (int8) + group ids (int32).
    X = sampled.select(ACTIVE_FEATURES).to_numpy().astype(np.float32, copy=False)
    y = sampled["label_up"].to_numpy().astype(np.int8, copy=False)
    slugs = sampled["market_slug"].to_list()
    # group_id = stable hash of slug → int32
    uniq = {s: i for i, s in enumerate(dict.fromkeys(slugs))}
    g = np.fromiter((uniq[s] for s in slugs), dtype=np.int32, count=len(slugs))
    return X, y, g, sorted(uniq.keys())


def evaluate(y_true: np.ndarray, p_pred: np.ndarray, p_market: np.ndarray, label: str) -> dict:
    """Compute AUC / Brier / LogLoss / Brier-skill vs the market prior."""
    out = {}
    out["n_rows"] = int(len(y_true))
    out["label_up_frac"] = float(y_true.mean())
    out["auc"] = float(roc_auc_score(y_true, p_pred))
    out["pr_auc"] = float(average_precision_score(y_true, p_pred))
    out["brier"] = float(brier_score_loss(y_true, p_pred))
    out["logloss"] = float(log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6)))
    out["market_auc"] = float(roc_auc_score(y_true, p_market))
    out["market_brier"] = float(brier_score_loss(y_true, p_market))
    out["market_logloss"] = float(log_loss(y_true, np.clip(p_market, 1e-6, 1 - 1e-6)))
    # Brier skill score vs market prior (>0 = model beats market).
    out["brier_skill_vs_market"] = float(
        1.0 - out["brier"] / out["market_brier"]) if out["market_brier"] > 0 else 0.0
    out["logloss_delta_vs_market"] = float(out["market_logloss"] - out["logloss"])
    print(f"=== {label} ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.6f}")
        else:
            print(f"  {k:30s} = {v}")
    print()
    return out


def calibration_table(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> List[dict]:
    """Bucketize p_pred into deciles, return per-bin avg_pred vs actual_freq."""
    edges = np.quantile(p_pred, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    rows = []
    for i in range(bins):
        m = (p_pred >= edges[i]) & (p_pred < edges[i + 1])
        if not m.any():
            continue
        rows.append({
            "bin": i,
            "n": int(m.sum()),
            "avg_pred": float(p_pred[m].mean()),
            "actual_freq": float(y_true[m].mean()),
        })
    return rows


def chunked_predict(model: lgb.Booster, parquet_glob: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the OOS test set, predict in chunks. Returns
    (y_true, p_pred_raw, p_market)."""
    # implied_p_up is already an element of ACTIVE_FEATURES; just add label_up.
    lf = pl.scan_parquet(parquet_glob).filter(pl.col("split") == split).select(
        ACTIVE_FEATURES + ["label_up"]
    )
    # Collect via streaming so we don't blow memory; then iterate by slicing.
    df = lf.collect(engine="streaming")
    n = df.height
    print(f"[{ts()}] predicting on {n:,} rows ...", flush=True)
    market_idx = ACTIVE_FEATURES.index("implied_p_up")
    y_chunks, p_chunks, m_chunks = [], [], []
    for start in range(0, n, PRED_CHUNK_ROWS):
        sl = df.slice(start, PRED_CHUNK_ROWS)
        X = sl.select(ACTIVE_FEATURES).to_numpy().astype(np.float32, copy=False)
        y_chunks.append(sl["label_up"].to_numpy().astype(np.int8))
        m_chunks.append(X[:, market_idx].copy())
        p_chunks.append(model.predict(X).astype(np.float32))
    return (np.concatenate(y_chunks), np.concatenate(p_chunks), np.concatenate(m_chunks))


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    # --- 1. Load train split, sample, split off val by market_slug ---
    X_all, y_all, g_all, train_slugs = load_split("train", ROWS_PER_MARKET_TRAIN)
    n_markets = len(train_slugs)
    rng = np.random.default_rng(RANDOM_SEED)
    market_ids_shuf = rng.permutation(n_markets)
    n_val_markets = int(round(n_markets * VAL_MARKET_FRAC))
    val_market_set = set(market_ids_shuf[:n_val_markets].tolist())
    val_mask = np.fromiter((g in val_market_set for g in g_all),
                           dtype=bool, count=len(g_all))
    X_tr, y_tr = X_all[~val_mask], y_all[~val_mask]
    X_va, y_va = X_all[val_mask], y_all[val_mask]
    print(f"[{ts()}] train rows={len(y_tr):,} ({n_markets - n_val_markets:,} markets), "
          f"val rows={len(y_va):,} ({n_val_markets:,} markets)", flush=True)
    print(f"[{ts()}] train label_up frac={y_tr.mean():.4f}, "
          f"val label_up frac={y_va.mean():.4f}", flush=True)
    del X_all, y_all, g_all

    # --- 2. Train LightGBM with early stopping on val ---
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=ACTIVE_FEATURES, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, feature_name=ACTIVE_FEATURES, reference=dtr, free_raw_data=False)
    print(f"[{ts()}] starting LightGBM training ...", flush=True)
    t0 = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(50),
        ],
    )
    train_secs = time.time() - t0
    best_iter = booster.best_iteration
    print(f"[{ts()}] LightGBM done in {train_secs:.1f}s, best_iter={best_iter}", flush=True)

    # --- 3. Calibrate on val (isotonic) ---
    p_va_raw = booster.predict(X_va)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va_raw, y_va)
    p_va_cal = iso.transform(p_va_raw)

    # market prior on val for comparison
    # We didn't keep implied_p_up in X (it IS a feature column already).
    implied_p_up_idx = ACTIVE_FEATURES.index("implied_p_up")
    p_va_market = X_va[:, implied_p_up_idx].astype(np.float32)

    val_metrics_raw = evaluate(y_va, p_va_raw, p_va_market, "VAL (raw model)")
    val_metrics_cal = evaluate(y_va, p_va_cal, p_va_market, "VAL (isotonic-calibrated)")
    val_calib_table = calibration_table(y_va, p_va_cal)

    # Free big arrays before OOS pass.
    del X_tr, y_tr, X_va, y_va, p_va_raw, p_va_cal, p_va_market, dtr, dva

    # --- 4. OOS evaluation on full test set ---
    y_te, p_te_raw, p_te_market = chunked_predict(booster, str(DATA_DIR / "*.parquet"), "test")
    p_te_cal = iso.transform(p_te_raw)
    oos_metrics_raw = evaluate(y_te, p_te_raw, p_te_market, "OOS (raw model)")
    oos_metrics_cal = evaluate(y_te, p_te_cal, p_te_market, "OOS (isotonic-calibrated)")
    oos_calib_table = calibration_table(y_te, p_te_cal)

    # Bonus: edge by TTC bucket — does the model help most in the late tail?
    print("[{}] computing per-ttc-bin metrics ...".format(ts()), flush=True)
    lf_ttc = (pl.scan_parquet(str(DATA_DIR / "*.parquet"))
              .filter(pl.col("split") == "test")
              .select(["ttc_s"])
              .collect(engine="streaming"))
    ttc = lf_ttc["ttc_s"].to_numpy().astype(np.float32)
    bins = [(60, 50), (50, 40), (40, 30), (30, 20), (20, 10)]
    ttc_rows = []
    for lo, hi in bins:
        m = (ttc < lo) & (ttc >= hi)
        if not m.any():
            continue
        ttc_rows.append({
            "ttc_range": f"[{hi},{lo})",
            "n": int(m.sum()),
            "model_brier": float(brier_score_loss(y_te[m], p_te_cal[m])),
            "market_brier": float(brier_score_loss(y_te[m], p_te_market[m])),
            "model_auc": float(roc_auc_score(y_te[m], p_te_cal[m])),
            "market_auc": float(roc_auc_score(y_te[m], p_te_market[m])),
        })
    for r in ttc_rows:
        r["brier_skill_vs_market"] = (
            1.0 - r["model_brier"] / r["market_brier"]) if r["market_brier"] > 0 else 0.0
    print("=== OOS BY TTC BIN ===")
    for r in ttc_rows:
        print(f"  {r['ttc_range']}: n={r['n']:>9,}  "
              f"brier model={r['model_brier']:.5f} mkt={r['market_brier']:.5f}  "
              f"skill={r['brier_skill_vs_market']:+.4f}  "
              f"auc model={r['model_auc']:.4f} mkt={r['market_auc']:.4f}")
    print()

    # --- 5. Save artifacts ---
    booster._calibrator = iso  # type: ignore[attr-defined]
    model_path = ARTIFACT_DIR / "model.pkl"
    joblib.dump(booster, model_path)
    iso_path = ARTIFACT_DIR / "calibrator.pkl"
    joblib.dump(iso, iso_path)
    (ARTIFACT_DIR / "features.json").write_text(json.dumps(ACTIVE_FEATURES, indent=2))

    fi_gain = dict(zip(booster.feature_name(), booster.feature_importance("gain").tolist()))
    fi_split = dict(zip(booster.feature_name(), booster.feature_importance("split").tolist()))
    (ARTIFACT_DIR / "feature_importance.json").write_text(json.dumps(
        {"gain": fi_gain, "split": fi_split}, indent=2))

    manifest = {
        "version": "poly_l2_only_v1",
        "dataset": "polymarket_l2_last60s_v1",
        "feature_count": len(ACTIVE_FEATURES),
        "dropped_features": sorted(DROP_FEATURES),
        "rows_per_market_train": ROWS_PER_MARKET_TRAIN,
        "val_market_frac": VAL_MARKET_FRAC,
        "random_seed": RANDOM_SEED,
        "lgb_params": LGB_PARAMS,
        "num_boost_round_max": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(best_iter),
        "train_secs": round(train_secs, 1),
        "metrics_val_raw": val_metrics_raw,
        "metrics_val_calibrated": val_metrics_cal,
        "metrics_oos_raw": oos_metrics_raw,
        "metrics_oos_calibrated": oos_metrics_cal,
        "calibration_val": val_calib_table,
        "calibration_oos": oos_calib_table,
        "oos_by_ttc_bin": ttc_rows,
        "total_secs": round(time.time() - t_global, 1),
    }
    (ARTIFACT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{ts()}] saved artifacts to {ARTIFACT_DIR}", flush=True)
    print(f"[{ts()}] total wall time: {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
