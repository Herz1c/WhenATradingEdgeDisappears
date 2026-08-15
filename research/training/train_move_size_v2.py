"""Model 8b: Binary big-move classifier with BTC volatility features.

Improvements over Model 8 v1:
  1. TARGET RE-FRAMING: binary "will |mid_move_5s| >= threshold?" instead of regression.
     Directly optimizes the question we care about (volatility regime detection).

  2. BTC VOLATILITY FEATURES (DERIVED per market from live_btc_usd time series):
     - btc_return_3s, btc_return_5s, btc_return_15s   -- BTC momentum lookback
     - btc_abs_return_5s, btc_abs_return_15s           -- BTC magnitude lookback
     - btc_realized_vol_30s                            -- BTC vol (std of 1s returns over 30s)
     Rationale: a token mid jump often originates from a BTC spot move that the
     token's own L2 book cannot foresee.  Token L2 microstructure plus BTC
     volatility together describe both endogenous and exogenous shock channels.

  3. THRESHOLD TUNING: report precision/recall at multiple decision thresholds.

LEAKAGE GUARD (same exclusions as Model 8 v1):
  All `mid_move_*`, `mid_move_sign_*`, `max_favorable_excursion_*`,
  `max_adverse_excursion_*`, `resolved_*`, `flip_*`, `terminal_*` are EXCLUDED.

  The derived BTC features are computed from `live_btc_usd` at the anchor time
  and PAST anchor times -- strictly backward-looking, no leakage.
"""
import sys, json
from pathlib import Path
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss,
)

from model_factory.trainers.logistic_regression_trainer import _to_float_array


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")
TRAIN_DATES = [f"2026-04-{d:02d}" for d in range(19, 31)] + [f"2026-05-{d:02d}" for d in range(1, 5)]
VAL_DATES   = ["2026-05-05"]
TEST_DATES  = ["2026-05-06"]

BIG_MOVE_THRESHOLD = 0.02  # target: |mid_move_5s| >= 0.02

BASE_FEATURES = [
    "anchor_source",
    "event_count_1s", "event_count_3s", "event_count_5s", "event_count_15s",
    "quote_update_count_1s", "quote_update_count_3s", "quote_update_count_5s",
    "trade_count_1s", "trade_count_3s", "trade_count_5s",
    "signed_trade_size_1s", "signed_trade_size_3s", "signed_trade_size_5s",
    "absolute_trade_size_1s", "absolute_trade_size_3s", "absolute_trade_size_5s",
    "spread_mean_5s", "spread_min_5s", "spread_max_5s",
    "imbalance_mean_5s", "imbalance_last", "imbalance_slope_5s",
    "mid_return_1s", "mid_return_3s", "mid_return_5s",
    "quote_churn_rate_5s", "depth_change_rate_5s",
    "one_sided_book", "crossed_book", "stale_book",
    "sequence_completeness_rate",
    "price_to_beat", "live_btc_usd",
    "delta_to_strike", "abs_delta_to_strike", "delta_sign",
    "t_since_open_s", "t_to_close_s", "phase_bucket",
    "live_reference_trusted", "live_bias_mode", "live_applied_bias_age_seconds",
]

# Derived BTC features added in augment_btc_features()
DERIVED_FEATURES = [
    "btc_return_3s", "btc_return_5s", "btc_return_15s",
    "btc_abs_return_5s", "btc_abs_return_15s",
    "btc_realized_vol_30s",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

LOAD_COLS = list(dict.fromkeys(
    BASE_FEATURES + [
        "market_slug", "token_side_normalized", "anchor_ts_ns",
        "mid_move_5s", "sequence_feature_eligible",
    ]
))


# ---------------------------------------------------------------------------
# BTC feature augmentation
# ---------------------------------------------------------------------------

def augment_btc_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add backward-looking BTC volatility/momentum features per market.

    Procedure (no leakage):
      1. Dedup to one row per (market_slug, anchor_ts_ns) since BTC price is
         shared between UP/DOWN tokens at the same instant.
      2. Sort by (market_slug, anchor_ts_ns).
      3. Within each market, compute rolling/lag features on live_btc_usd.
      4. Join back to original 2-row-per-second frame on (market, ts).
    """
    # Dedup BTC time series per market (UP and DOWN share live_btc_usd)
    btc_ts = (
        df.select(["market_slug", "anchor_ts_ns", "live_btc_usd"])
          .unique(subset=["market_slug", "anchor_ts_ns"], keep="first")
          .sort(["market_slug", "anchor_ts_ns"])
    )

    # Build BTC features per market.  Rows are 1-sec granular within each market.
    # Use group_by("market_slug").map_groups with row-wise polars expressions.
    btc_features = btc_ts.with_columns([
        # past returns (% change): live_btc_usd shifted backwards
        ((pl.col("live_btc_usd") / pl.col("live_btc_usd").shift(3).over("market_slug")) - 1.0)
            .alias("btc_return_3s"),
        ((pl.col("live_btc_usd") / pl.col("live_btc_usd").shift(5).over("market_slug")) - 1.0)
            .alias("btc_return_5s"),
        ((pl.col("live_btc_usd") / pl.col("live_btc_usd").shift(15).over("market_slug")) - 1.0)
            .alias("btc_return_15s"),
    ])
    btc_features = btc_features.with_columns([
        pl.col("btc_return_5s").abs().alias("btc_abs_return_5s"),
        pl.col("btc_return_15s").abs().alias("btc_abs_return_15s"),
    ])

    # Rolling std of 1s BTC returns over 30s window -- need 1s return first
    btc_features = btc_features.with_columns(
        ((pl.col("live_btc_usd") / pl.col("live_btc_usd").shift(1).over("market_slug")) - 1.0)
            .alias("_btc_ret_1s")
    ).with_columns(
        pl.col("_btc_ret_1s").rolling_std(window_size=30, min_samples=10)
            .over("market_slug")
            .alias("btc_realized_vol_30s")
    ).drop("_btc_ret_1s")

    # Join back: per (market_slug, anchor_ts_ns) all derived columns
    out = df.join(
        btc_features.select(["market_slug", "anchor_ts_ns"] + DERIVED_FEATURES),
        on=["market_slug", "anchor_ts_ns"],
        how="left",
    )

    # Fill nulls (e.g. first 15 rows per market) with 0 -- equivalent to "no info"
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in DERIVED_FEATURES])
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(dates: list[str], split_name: str) -> pl.DataFrame:
    print(f"  [{split_name}] Loading {len(dates)} day(s)...", flush=True)
    frames = []
    for d in dates:
        f = DATASET_PATH / f"{d}.parquet"
        if not f.exists():
            print(f"    SKIP missing: {d}")
            continue
        schema = pl.scan_parquet(str(f)).collect_schema()
        avail = set(schema.names())
        cols = [c for c in LOAD_COLS if c in avail]
        frames.append(pl.read_parquet(str(f), columns=cols))
    df = pl.concat(frames, how="vertical_relaxed")

    # Eligibility
    if "sequence_feature_eligible" in df.columns:
        df = df.filter(pl.col("sequence_feature_eligible"))

    # Drop NaN target rows (polars NaN != null; need both filters)
    df = df.filter(
        pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan()
    )

    print(f"  [{split_name}] {df.height:,} rows after eligibility+target filter")

    # Add BTC features
    df = augment_btc_features(df)
    print(f"  [{split_name}] BTC features added, total cols={df.width}")

    # Create binary target
    df = df.with_columns(
        (pl.col("mid_move_5s").abs() >= BIG_MOVE_THRESHOLD).cast(pl.Int8).alias("big_move")
    )

    pos_rate = float(df["big_move"].mean())
    print(f"  [{split_name}] big_move (|move|>={BIG_MOVE_THRESHOLD}) positive rate = {pos_rate:.4%}")
    return df


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name, mid_move=None):
    proba = model.predict_proba(X)[:, 1]
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1-1e-6))),
        "brier": float(brier_score_loss(y, proba)),
        "auc": float(roc_auc_score(y, proba)),
        "avg_precision": float(average_precision_score(y, proba)),
    }

    # Threshold sweep
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    out["threshold_table"] = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        n_sel = int(pred.sum())
        if n_sel == 0:
            continue
        tp = int((pred & y).sum())
        prec = tp / n_sel
        rec  = tp / max(int(y.sum()), 1)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out["threshold_table"].append({
            "threshold": t,
            "n_selected": n_sel,
            "pct_selected": float(n_sel / len(y)),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    # Edge analysis at the gate threshold
    if mid_move is not None:
        for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            sel = proba >= t
            if sel.sum() == 0:
                continue
            mm_sel = mid_move[sel]
            mean_abs_move = float(np.nanmean(np.abs(mm_sel)))
            out[f"mean_abs_move_at_t{int(t*100):02d}"] = mean_abs_move

    return out, proba


def print_eval(split_name, m):
    print(f"\n  [{split_name.upper()}]  n={m['n']:,}  pos_rate={m['positive_rate']:.4%}")
    print(f"    log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
          f"auc={m['auc']:.4f}  avg_precision={m['avg_precision']:.4f}")
    print(f"    {'thresh':>7s}  {'n_sel':>10s}  {'pct':>6s}  "
          f"{'precision':>10s}  {'recall':>8s}  {'f1':>6s}  {'mean|move|':>11s}")
    print(f"    {'-'*7}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*11}")
    for row in m["threshold_table"]:
        mam = m.get(f"mean_abs_move_at_t{int(row['threshold']*100):02d}", float("nan"))
        print(f"    {row['threshold']:>7.2f}  {row['n_selected']:>10,}  "
              f"{row['pct_selected']*100:>5.1f}%  {row['precision']:>10.4f}  "
              f"{row['recall']:>8.4f}  {row['f1']:>6.4f}  {mam:>11.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "#"*72)
    print("  MODEL 8b: BINARY BIG-MOVE CLASSIFIER  |  target: |mid_move_5s| >= 0.02")
    print("#"*72)

    print("\n--- Loading data ---")
    train_df = load_split(TRAIN_DATES, "train")
    val_df   = load_split(VAL_DATES,   "val")
    test_df  = load_split(TEST_DATES,  "test")

    print("\n--- Preparing features ---")
    X_train, _ = _to_float_array(train_df.select(ALL_FEATURES))
    y_train = train_df["big_move"].to_numpy().astype(int)

    X_val, _ = _to_float_array(val_df.select(ALL_FEATURES))
    y_val = val_df["big_move"].to_numpy().astype(int)
    mid_val = val_df["mid_move_5s"].to_numpy().astype(float)

    X_test, _ = _to_float_array(test_df.select(ALL_FEATURES))
    y_test = test_df["big_move"].to_numpy().astype(int)
    mid_test = test_df["mid_move_5s"].to_numpy().astype(float)

    print(f"  X_train: {X_train.shape}  pos={y_train.sum():,} ({y_train.mean():.4%})")
    print(f"  X_val:   {X_val.shape}    pos={y_val.sum():,} ({y_val.mean():.4%})")
    print(f"  X_test:  {X_test.shape}   pos={y_test.sum():,} ({y_test.mean():.4%})")

    # Class imbalance: pos rate ~23% (val/test) → ~10% (train if 60% zero-moves).
    # Use scale_pos_weight to upweight positives.
    spw = (1.0 - y_train.mean()) / max(y_train.mean(), 1e-6)
    print(f"  scale_pos_weight = {spw:.3f}")

    print("\n--- Training LightGBM binary classifier ---")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        num_leaves=63,
        min_child_samples=500,
        learning_rate=0.05,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        scale_pos_weight=1.0,  # leave neutral; we will tune threshold instead
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    print(f"  best_iteration: {model.best_iteration_}")

    # Eval
    print("\n--- Evaluation ---")
    val_metrics,  val_proba  = evaluate(model, X_val,  y_val,  "val",  mid_val)
    test_metrics, test_proba = evaluate(model, X_test, y_test, "test", mid_test)

    print_eval("val", val_metrics)
    print_eval("test", test_metrics)

    # Feature importance
    print("\n--- Top 15 features (by gain) ---")
    imp = sorted(
        zip(ALL_FEATURES, model.feature_importances_),
        key=lambda x: -x[1],
    )[:15]
    total = float(model.feature_importances_.sum()) or 1.0
    for name, val in imp:
        print(f"  {val/total:.4f}  {name}")

    # Signed direction analysis: for big-move predictions, check sign accuracy
    print("\n--- Signed direction quality on test ---")
    print("  When the binary model says 'big move likely', is the SIGN of mid_move predictable?")
    print("  (using mid_move_5s itself for test analysis; in production combine with Model 7)")
    print()
    for t in [0.10, 0.20, 0.30, 0.50]:
        sel = test_proba >= t
        if sel.sum() == 0:
            continue
        actual_big = (np.abs(mid_test[sel]) >= BIG_MOVE_THRESHOLD).sum()
        precision = actual_big / sel.sum()
        recall = actual_big / max(int((np.abs(mid_test) >= BIG_MOVE_THRESHOLD).sum()), 1)
        mean_abs = float(np.abs(mid_test[sel]).mean())
        print(f"  prob>={t:.2f}: n_sel={sel.sum():,}  precision={precision:.4f}  "
              f"recall={recall:.4f}  mean|actual_move|={mean_abs:.5f}")

    # Save artifacts
    print("\n--- Saving artifacts ---")
    out_dir = Path("artifacts/model_08b_big_move_classifier/tabular_btc/lightgbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, out_dir / "model.pkl")
    results = {
        "val": val_metrics,
        "test": test_metrics,
        "feature_importance": {name: float(val/total) for name, val in zip(ALL_FEATURES, model.feature_importances_)},
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
        "config": {
            "big_move_threshold": BIG_MOVE_THRESHOLD,
            "train_dates": TRAIN_DATES,
            "val_dates": VAL_DATES,
            "test_dates": TEST_DATES,
            "features": ALL_FEATURES,
        }
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  saved to {out_dir}")

    print("\n=== DONE ===")
