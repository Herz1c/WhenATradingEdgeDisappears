"""Model 03 Fill Probability: LR baseline + LightGBM primary.

LEAKAGE AUDIT (full execution_intent leakage_only_columns enforced):
  - All outcome fields (filled, time_to_fill, markout_*, fees, rebate, pnl)
  - All resolution fields (resolved_side, terminal_*, hold_to_close_edge)
  - All replace/cancel lifecycle fields
  - adverse_selection_5s / toxic_fill_5s

  Safe features (decision-time only):
    - intent spec: price_level, limit_price, order_size, post_only, cancel_after_ns
    - book state: state_best_bid/ask, state_mid, state_spread, state_*_depth, state_stale_s
    - market context: t_to_close_s, t_since_open_s, delta_to_strike
    - live BTC: live_btc_usd, live_reference_trusted, live_applied_bias_age_seconds
    - microstructure summary: event_count_*s, imbalance_*, mid_return_*

Strategic role: P(fill | quote spec) is the gate to maker-first viability.
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "src")

from model_factory.trainers.model_specific.model_03_fill_probability_trainer import (
    Model03FillProbabilityTrainer,
)
from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer
from model_factory.dataset_loader import LeakageSafeLoader, DatasetSplit
import numpy as np
import polars as pl


class LRFillProbabilityTrainer(LogisticRegressionTrainer):
    MODEL_ID = "model_03_fill_probability"
    DEFAULT_HYPERPARAMS = {"C": 1.0, "max_iter": 1000}

    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)

    def _compute_baselines(self, split):
        # Reuse Model03 baselines
        return Model03FillProbabilityTrainer._compute_baselines(self, split)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(split: DatasetSplit):
    LEAKAGE_PATTERNS = [
        "markout_", "adverse_selection_", "toxic_fill_", "filled", "full_fill",
        "partial_fill", "fill_uncertain", "fill_ts", "fill_count", "fill_price",
        "fill_size", "fill_notional", "fees_paid", "rebate_earned",
        "gross_pnl_to_close", "net_pnl_to_close", "time_to_fill",
        "resolved_side", "terminal_", "flip_happened", "hold_to_close_edge",
        "cancel_", "replace_", "submitted", "post_only_rejected", "post_only_crossed",
        "maker_fill", "taker_fill", "cancelled", "winning_asset_id", "resolved_ts",
    ]
    features = [c for c in split.train.columns if c != split.target_column]
    violations = []
    for col in features:
        # Allow the safe configured fields: post_only itself (not _rejected) and
        # cancel_after_ns (intent spec, not outcome)
        if col in ("post_only", "cancel_after_ns"):
            continue
        for pat in LEAKAGE_PATTERNS:
            if col.startswith(pat) or col == pat:
                violations.append(col)
                break

    print(f"\n{'='*70}")
    print("  LEAKAGE AUDIT")
    print(f"{'='*70}")
    print(f"  Target column   : {split.target_column}")
    print(f"  Feature columns : {len(features)}")

    y_arr = split.train[split.target_column].cast(pl.Float64).to_numpy()
    valid = ~np.isnan(y_arr)
    y_v = y_arr[valid].astype(int)
    pos = int(y_v.sum())
    print(f"  Train fill rate : {y_v.mean():.4%}  ({pos:,}/{len(y_v):,})")

    if violations:
        print(f"\n  *** LEAKAGE DETECTED *** {violations}")
        raise RuntimeError(f"Leakage columns: {violations}")
    print(f"  STATUS: PASS -- no leakage columns detected")
    for col in ["price_level", "post_only", "state_spread", "t_to_close_s", "delta_to_strike", "live_btc_usd"]:
        print(f"  Safe feature [{col}]: {'PRESENT' if col in features else 'MISSING'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()}")
    print(f"{'='*70}")
    if "_calibration_meta" in metrics:
        print(f"  calibration: {metrics['_calibration_meta']}")
    for split, m in metrics.items():
        if split.startswith("_"):
            continue
        if not isinstance(m, dict) or "log_loss" not in m:
            continue
        ll = m.get("log_loss", float("nan"))
        bs = m.get("brier_score", float("nan"))
        ece = m.get("ece", float("nan"))
        n = m.get("n_rows", "?")
        tmean = m.get("target_mean", float("nan"))
        print(f"  [{split:5s}] ll={ll:.5f}  brier={bs:.5f}  ece={ece:.4f}  "
              f"n={n:,}  fill_rate={tmean:.4%}")

        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_log_loss_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "LOSES"
                print(f"           vs {bl:<25s}: ll={bm['log_loss']:.5f}  "
                      f"dll={delta:+.5f}  [{sign}]")

        if "fill_rate_by_t_to_close" in m:
            print(f"  [fill rate by t_to_close]")
            for bucket, fm in m["fill_rate_by_t_to_close"].items():
                print(f"    t=[{bucket:>7s}s]  n={fm['n']:>8,}  "
                      f"actual={fm['actual_fill_rate']:.4f}  "
                      f"pred={fm['pred_fill_rate']:.4f}  "
                      f"ll={fm['log_loss']:.4f}")

        if "fill_rate_by_price_level" in m:
            print(f"  [fill rate by price_level]")
            for pl_, fm in m["fill_rate_by_price_level"].items():
                print(f"    {pl_:>20s}  n={fm['n']:>8,}  "
                      f"actual={fm['actual_fill_rate']:.4f}  "
                      f"pred={fm['pred_fill_rate']:.4f}")

        if "fill_rate_by_post_only" in m:
            print(f"  [fill rate by post_only]")
            for po, fm in m["fill_rate_by_post_only"].items():
                print(f"    {po:>18s}  n={fm['n']:>8,}  "
                      f"actual={fm['actual_fill_rate']:.4f}  "
                      f"pred={fm['pred_fill_rate']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n\n" + "#"*70)
    print("  MODEL 03 FILL PROBABILITY  |  Phase 6")
    print("#"*70)

    # Leakage audit
    loader = LeakageSafeLoader()
    split_preview = loader.load("model_03_fill_probability", verbose=False)
    leakage_audit(split_preview)
    del split_preview

    # LR baseline
    print("--- Logistic Regression baseline ---")
    lr = LRFillProbabilityTrainer()
    lr_metrics = lr.train()
    print_summary("logistic_regression", lr_metrics)

    # LightGBM primary
    print("\n--- LightGBM ---")
    lgbm = Model03FillProbabilityTrainer()
    lgbm_metrics = lgbm.train()
    print_summary("lightgbm", lgbm_metrics)

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps({
        "logistic_regression": lr_metrics,
        "lightgbm": lgbm_metrics,
    }, indent=2, default=str))
