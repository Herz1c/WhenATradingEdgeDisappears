"""Phase 6 — Re-rank top-K analysis using Model 4b (markout_to_close)
predictions instead of Model 4 v1 (markout_1s).

This isolates the question: does ranking quotes by predicted held-to-close PnL
(via Model 4b) yield better realized PnL than ranking by 1s markout (v1)?

Outputs side-by-side comparison v1 vs 4b ranking.
"""
import sys, os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import polars as pl
import joblib
import yaml

from model_factory.trainers.logistic_regression_trainer import _to_float_array


TEST_DATE = "2026-05-06"
DATASET_PATH = Path("data/datasets/execution_intent_dataset_v1")
MODEL_03_PATH = Path("artifacts/model_03_fill_probability/lightgbm/model.pkl")
MODEL_04_V1_PATH = Path("artifacts/model_04_adverse_selection/lightgbm/model.pkl")
MODEL_04B_PATH = Path("artifacts/model_04b_markout_to_close/lightgbm/model.pkl")
FS_03 = Path("config/feature_sets/model_03_fill_probability_features_v1.yaml")
FS_04 = Path("config/feature_sets/model_04_adverse_selection_features_v1.yaml")


def load_fs(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))["feature_columns"]


def topk_pnl(predicted_edge, actual_net_pnl_per_attempt, n_total, pcts):
    out = []
    for top_pct in pcts:
        k = int(top_pct * n_total)
        if k < 100:
            continue
        idx = np.argsort(-predicted_edge)[:k]
        net = float(actual_net_pnl_per_attempt[idx].mean())
        out.append((top_pct, k, net))
    return out


def main():
    print("\n" + "#"*72)
    print("  PHASE 6 — RE-RANK: Model 4 v1 vs Model 4b (markout_to_close)")
    print("#"*72)

    fs03 = load_fs(FS_03)
    fs04 = load_fs(FS_04)

    # Load test data
    f = DATASET_PATH / f"{TEST_DATE}.parquet"
    schema = pl.scan_parquet(str(f)).collect_schema()
    avail = set(schema.names())
    pnl_cols = ["filled", "markout_1s", "markout_to_close",
                "net_pnl_to_close", "execution_training_eligible",
                "price_level", "t_to_close_s"]
    load_cols = list(dict.fromkeys(fs03 + fs04 + pnl_cols))
    load_cols = [c for c in load_cols if c in avail]
    df = pl.read_parquet(str(f), columns=load_cols)
    df = df.filter(pl.col("execution_training_eligible"))
    n_total = df.height
    print(f"  rows after eligibility: {n_total:,}")

    # Predict P(fill)
    m03 = joblib.load(MODEL_03_PATH)
    X03 = np.nan_to_num(_to_float_array(df.select(fs03))[0], nan=0.0)
    p_fill_raw = m03.predict_proba(X03)[:, 1]
    cal_03 = getattr(m03, "_calibrator", None)
    if cal_03 is not None:
        p_fill = np.clip(cal_03.predict(p_fill_raw), 1e-6, 1 - 1e-6)
    else:
        p_fill = p_fill_raw

    # Predict markout via both Model 4 v1 and Model 4b
    X04 = np.nan_to_num(_to_float_array(df.select(fs04))[0], nan=0.0)
    m04_v1 = joblib.load(MODEL_04_V1_PATH)
    m04b = joblib.load(MODEL_04B_PATH)
    pred_mko_1s = m04_v1.predict(X04)
    pred_mko_close = m04b.predict(X04)

    edge_v1 = p_fill * pred_mko_1s
    edge_4b = p_fill * pred_mko_close

    # Actual PnL data
    y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
    actual_net_pnl = df["net_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
    actual_net_pnl_per_attempt = np.where(y_filled, actual_net_pnl, 0.0)

    pcts = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

    v1_results = topk_pnl(edge_v1, actual_net_pnl_per_attempt, n_total, pcts)
    v4b_results = topk_pnl(edge_4b, actual_net_pnl_per_attempt, n_total, pcts)

    # Print comparison
    print(f"\n  {'cut':>7s}  {'n':>10s}  {'v1 net_pnl':>13s}  {'4b net_pnl':>13s}  {'lift':>10s}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*13}  {'-'*13}  {'-'*10}")
    for (p1, k1, n1), (p2, k2, n2) in zip(v1_results, v4b_results):
        lift = n2 - n1
        print(f"  top {int(p1*1000)/10:>4.1f}%  {k1:>10,}  {n1:>+13.5f}  {n2:>+13.5f}  {lift:>+10.5f}")

    print("\n  Interpretation:")
    print("  + lift = Model 4b ranks BETTER than v1 (more profit per quote)")
    print("  - lift = v1 ranking was better")

    # Volume-weighted: sum_net_pnl in top-K
    print(f"\n  {'cut':>7s}  {'n':>10s}  {'v1 sum_pnl':>13s}  {'4b sum_pnl':>13s}  {'lift':>10s}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*13}  {'-'*13}  {'-'*10}")
    for top_pct in [0.005, 0.01, 0.05, 0.10, 0.25]:
        k = int(top_pct * n_total)
        idx_v1 = np.argsort(-edge_v1)[:k]
        idx_4b = np.argsort(-edge_4b)[:k]
        sum_v1 = float(actual_net_pnl_per_attempt[idx_v1].sum())
        sum_4b = float(actual_net_pnl_per_attempt[idx_4b].sum())
        lift = sum_4b - sum_v1
        print(f"  top {int(top_pct*1000)/10:>4.1f}%  {k:>10,}  ${sum_v1:>+12,.2f}  ${sum_4b:>+12,.2f}  ${lift:>+9,.2f}")

    # Overlap analysis
    print(f"\n  Top-1% set overlap between v1 and 4b rankings:")
    k1 = int(0.01 * n_total)
    s_v1 = set(np.argsort(-edge_v1)[:k1].tolist())
    s_4b = set(np.argsort(-edge_4b)[:k1].tolist())
    intersect = len(s_v1 & s_4b)
    print(f"    n_top1pct: {k1:,}")
    print(f"    intersection: {intersect:,} ({100*intersect/k1:.1f}%)")
    print(f"    v1-only: {len(s_v1 - s_4b):,}")
    print(f"    4b-only: {len(s_4b - s_v1):,}")

    # Verdict
    print("\n" + "="*72)
    print("  COMPARISON VERDICT")
    print("="*72)
    top10_v1 = next((n for p, _, n in v1_results if abs(p - 0.10) < 1e-6), 0)
    top10_4b = next((n for p, _, n in v4b_results if abs(p - 0.10) < 1e-6), 0)
    top5_v1 = next((n for p, _, n in v1_results if abs(p - 0.05) < 1e-6), 0)
    top5_4b = next((n for p, _, n in v4b_results if abs(p - 0.05) < 1e-6), 0)
    top1_v1 = next((n for p, _, n in v1_results if abs(p - 0.01) < 1e-6), 0)
    top1_4b = next((n for p, _, n in v4b_results if abs(p - 0.01) < 1e-6), 0)
    print()
    print(f"  Per-attempt net_pnl @ top 1%:   v1={top1_v1:+.5f}  4b={top1_4b:+.5f}  lift={top1_4b-top1_v1:+.5f}")
    print(f"  Per-attempt net_pnl @ top 5%:   v1={top5_v1:+.5f}  4b={top5_4b:+.5f}  lift={top5_4b-top5_v1:+.5f}")
    print(f"  Per-attempt net_pnl @ top 10%:  v1={top10_v1:+.5f}  4b={top10_4b:+.5f}  lift={top10_4b-top10_v1:+.5f}")

    if top10_4b > top10_v1 + 0.001:
        print("\n  VERDICT: Model 4b is the better ranker for held-to-close PnL.")
        print("           Replace v1 with 4b in the deployment pipeline.")
    elif top10_v1 > top10_4b + 0.001:
        print("\n  VERDICT: Model 4 v1 (markout_1s) ranks BETTER than 4b for PnL.")
        print("           This is surprising. Toxicity proxy outperforms direct PnL prediction.")
        print("           Likely cause: markout_to_close is too noisy a target.")
    else:
        print("\n  VERDICT: Models 4b and v1 give equivalent ranking quality (within 0.1c).")
        print("           Stick with v1 (cleaner target, better calibration).")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
