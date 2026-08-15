"""Model 04 Adverse Selection / Markout: Ridge baseline + LightGBM regression.

LEAKAGE AUDIT (full execution_intent leakage_only_columns enforced):
  - All outcome fields except markout itself (which IS the target)
  - All resolution fields
  - All replace/cancel lifecycle fields

  Dataset filter: filled == True  (only conditional on fills)

Strategic role: E[markout | fill, quote spec].  Combined with Model 03:
    expected_edge_per_quote = P(fill | quote spec) * E[markout | fill] - fees
Answers: "When we DO get filled, do we make money or get adversely selected?"
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "src")

from model_factory.trainers.model_specific.model_04_adverse_selection_trainer import (
    Model04AdverseSelectionTrainer,
)
from model_factory.trainers.linear_regression_trainer import LinearRegressionTrainer
from model_factory.dataset_loader import LeakageSafeLoader, DatasetSplit
import numpy as np
import polars as pl


class RidgeAdverseSelectionTrainer(LinearRegressionTrainer):
    MODEL_ID = "model_04_adverse_selection"
    DEFAULT_HYPERPARAMS = {"alpha": 1.0}

    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)

    def _compute_regression_baselines(self, split):
        return Model04AdverseSelectionTrainer._compute_regression_baselines(self, split)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(split: DatasetSplit):
    LEAKAGE_PATTERNS = [
        "adverse_selection_", "toxic_fill_", "filled", "full_fill",
        "partial_fill", "fill_uncertain", "fill_ts", "fill_count", "fill_price",
        "fill_size", "fill_notional", "fees_paid", "rebate_earned",
        "gross_pnl_to_close", "net_pnl_to_close", "time_to_fill",
        "resolved_side", "terminal_", "flip_happened", "hold_to_close_edge",
        "cancel_request", "cancel_effective", "replace_", "submitted",
        "post_only_rejected", "post_only_crossed",
        "maker_fill", "taker_fill", "cancelled", "winning_asset_id",
        "resolved_ts",
        # Exclude all OTHER markout horizons -- target is the only allowed one
        "markout_",
    ]
    target = split.target_column
    features = [c for c in split.train.columns if c != target]
    violations = []
    for col in features:
        if col in ("post_only", "cancel_after_ns"):
            continue
        for pat in LEAKAGE_PATTERNS:
            if col.startswith(pat) or col == pat:
                # Allow the target itself if it accidentally lands here
                if col == target:
                    break
                violations.append(col)
                break

    print(f"\n{'='*70}")
    print("  LEAKAGE AUDIT (Model 04)")
    print(f"{'='*70}")
    print(f"  Target column   : {target}")
    print(f"  Feature columns : {len(features)}")

    y_arr = split.train[target].cast(pl.Float64).to_numpy()
    valid = ~np.isnan(y_arr)
    y_v = y_arr[valid]
    print(f"  Train markout   : n={len(y_v):,}  "
          f"mean={y_v.mean():.6f}  std={y_v.std():.6f}  "
          f"min={y_v.min():.4f}  max={y_v.max():.4f}")
    adv_rate = float((y_v < -0.005).mean())
    fav_rate = float((y_v > 0.005).mean())
    print(f"  Train adverse(<-0.005) rate: {adv_rate:.4%}")
    print(f"  Train favorable(>+0.005) rate: {fav_rate:.4%}")

    if violations:
        print(f"\n  *** LEAKAGE DETECTED *** {violations}")
        raise RuntimeError(f"Leakage columns: {violations}")
    print(f"  STATUS: PASS -- no leakage columns detected")
    for col in ["price_level", "post_only", "state_spread", "t_to_close_s", "delta_to_strike"]:
        print(f"  Safe feature [{col}]: {'PRESENT' if col in features else 'MISSING'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()}  (Model 04 Markout)")
    print(f"{'='*70}")
    for split, m in metrics.items():
        if split.startswith("_"):
            continue
        if not isinstance(m, dict) or "mae" not in m:
            continue
        mae = m.get("mae", float("nan"))
        rmse = m.get("rmse", float("nan"))
        n = m.get("n_rows", "?")
        tmean = m.get("target_mean", float("nan"))
        pmean = m.get("pred_mean", float("nan"))
        sa = m.get("sign_accuracy", float("nan"))
        print(f"  [{split:5s}] MAE={mae:.5f}  RMSE={rmse:.5f}  "
              f"n={n:,}  actual_mean={tmean:+.5f}  pred_mean={pmean:+.5f}  "
              f"sign_acc={sa:.4f}")

        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_mae_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "LOSES"
                print(f"           vs {bl:<25s}: MAE={bm['mae']:.5f}  "
                      f"dMAE={delta:+.5f}  [{sign}]")

        # Adverse detection
        for thr in [0.001, 0.005, 0.01, 0.02]:
            key = f"adverse_detect_thr_{thr}"
            if key in m:
                ad = m[key]
                print(f"  [adverse<-{thr}] n_actual={ad['n_actual_adverse']:,}  "
                      f"n_pred={ad['n_predicted_adverse']:,}  "
                      f"prec={ad['precision']:.4f}  rec={ad['recall']:.4f}")

        if "markout_by_t_to_close" in m:
            print(f"  [markout by t_to_close]")
            for bucket, mm in m["markout_by_t_to_close"].items():
                print(f"    t=[{bucket:>7s}s]  n={mm['n']:>8,}  "
                      f"actual={mm['actual_mean_markout']:+.5f}  "
                      f"pred={mm['pred_mean_markout']:+.5f}  "
                      f"MAE={mm['mae']:.5f}")

        if "markout_by_order_side" in m:
            print(f"  [markout by order_side]")
            for side, mm in m["markout_by_order_side"].items():
                print(f"    {side:>10s}  n={mm['n']:>8,}  "
                      f"actual={mm['actual_mean_markout']:+.5f}  "
                      f"pred={mm['pred_mean_markout']:+.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n\n" + "#"*70)
    print("  MODEL 04 ADVERSE SELECTION  |  target = markout_1s | filled=True")
    print("#"*70)

    # Leakage audit
    loader = LeakageSafeLoader()
    split_preview = loader.load("model_04_adverse_selection", verbose=False)
    leakage_audit(split_preview)
    del split_preview

    # Ridge baseline
    print("--- Ridge baseline ---")
    ridge = RidgeAdverseSelectionTrainer()
    ridge_metrics = ridge.train()
    print_summary("ridge", ridge_metrics)

    # LightGBM primary
    print("\n--- LightGBM regression ---")
    lgbm = Model04AdverseSelectionTrainer()
    lgbm_metrics = lgbm.train()
    print_summary("lightgbm", lgbm_metrics)

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps({
        "ridge": ridge_metrics,
        "lightgbm": lgbm_metrics,
    }, indent=2, default=str))
