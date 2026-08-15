"""Maker-Defense v2: adds vol-clustering and recent-jump features.

Hypothesis: dramatic moves cluster in time (volatility clustering).  If a
big move just happened, another is more likely.  Add features that capture
this without leakage:

  - past_abs_mid_return_3s / 5s        (already in dataset as mid_return_*; use abs)
  - past_jump_indicator_5s             (1 if any 1s-window in past 5s had |return|>0.02)
  - past_imbalance_volatility_5s       (std of imbalance over past 5s window — proxy)
  - btc_jump_indicator_5s              (1 if BTC moved >0.05% in past 5s)

All derived from BACKWARD-LOOKING data only.
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from model_factory.trainers.logistic_regression_trainer import _to_float_array
from train_move_size_v2 import augment_btc_features, BASE_FEATURES, DERIVED_FEATURES
from train_maker_defense import (
    LOAD_COLS_FULL, TRAIN_DATES, VAL_DATES, TEST_DATES, TARGETS,
    fpr_at_recall, recall_at_fpr,
)


DATASET_PATH = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")

EXTRA_FEATURES = [
    "abs_mid_return_1s",
    "abs_mid_return_3s",
    "abs_mid_return_5s",
    "past_jump_5s",          # 1 if |mid_return_5s| >= 0.02 (past, not future)
    "past_big_jump_5s",      # 1 if |mid_return_5s| >= 0.05
    "btc_jump_5s",           # 1 if |btc_return_5s| >= 0.001 (0.1% in 5s)
    "btc_jump_15s",          # 1 if |btc_return_15s| >= 0.002
    "spread_burst_5s",       # 1 if spread_max_5s >= 0.03 (wide spread = stressed book)
    "depth_collapse_5s",     # 1 if depth_change_rate_5s in top decile (proxy)
]

ALL_FEATURES_V2 = BASE_FEATURES + DERIVED_FEATURES + EXTRA_FEATURES


def augment_v2(df: pl.DataFrame) -> pl.DataFrame:
    df = augment_btc_features(df)
    df = df.with_columns([
        pl.col("mid_return_1s").abs().alias("abs_mid_return_1s"),
        pl.col("mid_return_3s").abs().alias("abs_mid_return_3s"),
        pl.col("mid_return_5s").abs().alias("abs_mid_return_5s"),
        (pl.col("mid_return_5s").abs() >= 0.02).cast(pl.Int8).alias("past_jump_5s"),
        (pl.col("mid_return_5s").abs() >= 0.05).cast(pl.Int8).alias("past_big_jump_5s"),
        (pl.col("btc_return_5s").abs() >= 0.001).cast(pl.Int8).alias("btc_jump_5s"),
        (pl.col("btc_return_15s").abs() >= 0.002).cast(pl.Int8).alias("btc_jump_15s"),
        (pl.col("spread_max_5s") >= 0.03).cast(pl.Int8).alias("spread_burst_5s"),
        (pl.col("depth_change_rate_5s") >= 0.50).cast(pl.Int8).alias("depth_collapse_5s"),
    ])
    return df


def load_split(dates, split_name):
    print(f"  [{split_name}] Loading {len(dates)} day(s)...", flush=True)
    frames = []
    for d in dates:
        f = DATASET_PATH / f"{d}.parquet"
        if not f.exists():
            print(f"    SKIP missing: {d}")
            continue
        schema = pl.scan_parquet(str(f)).collect_schema()
        avail = set(schema.names())
        cols = [c for c in LOAD_COLS_FULL if c in avail]
        frames.append(pl.read_parquet(str(f), columns=cols))
    df = pl.concat(frames, how="vertical_relaxed")

    if "sequence_feature_eligible" in df.columns:
        df = df.filter(pl.col("sequence_feature_eligible"))
    df = df.filter(
        pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan()
    )
    print(f"  [{split_name}] {df.height:,} rows after eligibility filter")
    df = augment_v2(df)
    for t in TARGETS:
        df = df.with_columns(
            (pl.col(t["move_col"]).abs() >= t["threshold"]).cast(pl.Int8).alias(f"target_{t['name']}")
        )
        pos = float(df[f"target_{t['name']}"].mean())
        print(f"  [{split_name}] target {t['name']}: positive_rate={pos:.4%}")
    return df


def train_target(target_cfg, train_df, val_df, test_df):
    name = target_cfg["name"]
    print(f"\n{'='*72}")
    print(f"  TRAINING v2: {name}  |  target = |{target_cfg['move_col']}| >= {target_cfg['threshold']}")
    print(f"{'='*72}")

    X_tr, _ = _to_float_array(train_df.select(ALL_FEATURES_V2))
    y_tr = train_df[f"target_{name}"].to_numpy().astype(int)
    X_va, _ = _to_float_array(val_df.select(ALL_FEATURES_V2))
    y_va = val_df[f"target_{name}"].to_numpy().astype(int)
    X_te, _ = _to_float_array(test_df.select(ALL_FEATURES_V2))
    y_te = test_df[f"target_{name}"].to_numpy().astype(int)

    print(f"  pos rates: train={y_tr.mean():.4%}  val={y_va.mean():.4%}  test={y_te.mean():.4%}")
    print(f"  n_features: {X_tr.shape[1]}")

    model = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        num_leaves=63, min_child_samples=200, learning_rate=0.05,
        n_estimators=1500, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    print(f"  best_iteration: {model.best_iteration_}")

    proba_va = model.predict_proba(X_va)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    auc_v = roc_auc_score(y_va, proba_va)
    auc_t = roc_auc_score(y_te, proba_te)
    ap_v  = average_precision_score(y_va, proba_va)
    ap_t  = average_precision_score(y_te, proba_te)

    print(f"\n  [VAL]  AUC={auc_v:.4f}  AvgPrec={ap_v:.4f}")
    print(f"  [TEST] AUC={auc_t:.4f}  AvgPrec={ap_t:.4f}  lift={ap_t/y_te.mean():.2f}x")

    print("\n  --- TEST: FPR @ target Recall ---")
    print(f"  {'Target Recall':>14s}  {'FPR':>8s}  {'Threshold':>10s}  {'%selected':>10s}")
    fpr_tbl = fpr_at_recall(y_te, proba_te, target_recalls=(0.80, 0.90, 0.95, 0.98, 0.99))
    for row in fpr_tbl:
        sel_pct = float((proba_te >= row["threshold"]).mean())
        print(f"  {row['target_recall']:>14.2%}  {row['fpr']:>8.4f}  "
              f"{row['threshold']:>10.4f}  {sel_pct:>10.2%}")

    print("\n  --- Top 15 features ---")
    imp = sorted(zip(ALL_FEATURES_V2, model.feature_importances_), key=lambda x: -x[1])[:15]
    total = float(model.feature_importances_.sum()) or 1.0
    for n, v in imp:
        marker = " *" if n in EXTRA_FEATURES else ""
        print(f"    {v/total:.4f}  {n}{marker}")

    return {
        "target": name,
        "auc_test": float(auc_t), "ap_test": float(ap_t),
        "pos_rate_test": float(y_te.mean()),
        "lift_over_random": float(ap_t / y_te.mean()),
        "fpr_at_recall": fpr_tbl,
        "best_iter": int(model.best_iteration_ or 0),
        "feature_importance": {n: float(v/total) for n, v in zip(ALL_FEATURES_V2, model.feature_importances_)},
    }, model


if __name__ == "__main__":
    print("\n" + "#"*72)
    print("  MAKER-DEFENSE v2: + vol-clustering & jump features")
    print("#"*72)

    train_df = load_split(TRAIN_DATES, "train")
    val_df   = load_split(VAL_DATES,   "val")
    test_df  = load_split(TEST_DATES,  "test")

    out_root = Path("artifacts/model_08c_maker_defense_v2")
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tgt in TARGETS:
        metrics, model = train_target(tgt, train_df, val_df, test_df)
        all_results[tgt["name"]] = metrics
        save_dir = out_root / tgt["name"]
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / "model.pkl")
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Compare v1 vs v2
    print("\n\n" + "#"*72)
    print("  v1 vs v2 COMPARISON")
    print("#"*72)
    v1_dir = Path("artifacts/model_08c_maker_defense")
    print(f"\n  {'Target':<22s}  {'AUC v1':>8s}  {'AUC v2':>8s}  {'AP v1':>8s}  {'AP v2':>8s}  "
          f"{'FPR@95 v1':>10s}  {'FPR@95 v2':>10s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for name, v2 in all_results.items():
        v1f = v1_dir / name / "metrics.json"
        if v1f.exists():
            with open(v1f) as f: v1 = json.load(f)
            f95_v1 = next((r["fpr"] for r in v1["fpr_at_recall"] if r["target_recall"]==0.95), float("nan"))
            f95_v2 = next((r["fpr"] for r in v2["fpr_at_recall"] if r["target_recall"]==0.95), float("nan"))
            print(f"  {name:<22s}  {v1['auc_test']:>8.4f}  {v2['auc_test']:>8.4f}  "
                  f"{v1['ap_test']:>8.4f}  {v2['ap_test']:>8.4f}  "
                  f"{f95_v1:>10.4f}  {f95_v2:>10.4f}")

    print("\n=== DONE ===")
