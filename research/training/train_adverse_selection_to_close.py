"""Model 04b: Adverse Selection / Markout-to-CLOSE regression.

Variant of Model 04 trained on `markout_to_close` instead of `markout_1s`.

WHY:
  Model 4 v1 (markout_1s) is a TOXICITY proxy — does the fill go favorable
  in the next 1 second?  That's adverse-selection detection.

  This variant (4b) trains directly on `markout_to_close` — the actual
  held-to-close PnL per share.  That's the held-strategy PnL prediction.

Both are useful:
  - Model 4 v1: when to defensively reject a maker order before it fills
  - Model 4b:   when a fill is expected to be PnL-positive at resolution

Dataset filter: `filled == True` (only conditional on fills) — same as v1.

Artifacts saved to: artifacts/model_04b_markout_to_close/
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "src")

import numpy as np
import polars as pl

from model_factory.trainers.model_specific.model_04_adverse_selection_trainer import (
    Model04AdverseSelectionTrainer,
)
from model_factory.trainers.linear_regression_trainer import LinearRegressionTrainer
from model_factory.dataset_loader import LeakageSafeLoader, DatasetSplit


# Override class to redirect artifact path and target
class Model04bMarkoutToCloseTrainer(Model04AdverseSelectionTrainer):
    MODEL_ID = "model_04_adverse_selection"  # registry entry (filter + dataset)
    # Reuse same hyperparams; just different target

    def _get_artifact_path(self, dataset_variant=None) -> str:
        # AUDIT FIX: redirect artifacts to a separate dir so we don't clobber v1
        from pathlib import Path
        base = Path("artifacts/model_04b_markout_to_close")
        algo = getattr(self, "ALGORITHM_NAME", self.__class__.__name__.lower())
        if dataset_variant:
            base = base / dataset_variant
        base = base / algo
        return str(base)


class RidgeMarkoutToCloseTrainer(LinearRegressionTrainer):
    MODEL_ID = "model_04_adverse_selection"
    DEFAULT_HYPERPARAMS = {"alpha": 1.0}

    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)

    def _compute_regression_baselines(self, split):
        return Model04AdverseSelectionTrainer._compute_regression_baselines(self, split)

    def _get_artifact_path(self, dataset_variant=None) -> str:
        from pathlib import Path
        base = Path("artifacts/model_04b_markout_to_close")
        if dataset_variant:
            base = base / dataset_variant
        base = base / "linear_regression"
        return str(base)


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
        "resolved_ts", "markout_",
    ]
    target = split.target_column
    features = [c for c in split.train.columns if c != target]
    violations = []
    for col in features:
        if col in ("post_only", "cancel_after_ns"):
            continue
        for pat in LEAKAGE_PATTERNS:
            if (col.startswith(pat) or col == pat) and col != target:
                violations.append(col)
                break

    print(f"\n{'='*70}")
    print(f"  LEAKAGE AUDIT (Model 04b)  TARGET = {target}")
    print(f"{'='*70}")
    print(f"  Feature columns: {len(features)}")

    y_arr = split.train[target].cast(pl.Float64).to_numpy()
    valid = ~np.isnan(y_arr)
    y_v = y_arr[valid]
    print(f"  Train target stats: n={len(y_v):,} mean={y_v.mean():+.5f} "
          f"std={y_v.std():.5f} min={y_v.min():.4f} max={y_v.max():.4f}")
    print(f"  P(target > 0):  {float((y_v > 0).mean()):.4%}")
    print(f"  P(target > 0.01): {float((y_v > 0.01).mean()):.4%}")
    print(f"  P(target < -0.01): {float((y_v < -0.01).mean()):.4%}")
    print(f"  P(target < -0.05): {float((y_v < -0.05).mean()):.4%}")

    if violations:
        print(f"\n  *** LEAKAGE DETECTED *** {violations}")
        raise RuntimeError(f"Leakage columns: {violations}")
    print(f"  STATUS: PASS")
    print(f"{'='*70}\n")


def print_summary(algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} (Model 04b: markout_to_close)")
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
        print(f"  [{split:5s}] MAE={mae:.5f} RMSE={rmse:.5f} "
              f"n={n:,} actual={tmean:+.5f} pred={pmean:+.5f} sign_acc={sa:.4f}")
        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_mae_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "LOSES"
                print(f"           vs {bl:<24s}: MAE={bm['mae']:.5f}  dMAE={delta:+.5f}  [{sign}]")
        # Markout by t_to_close
        if "markout_by_t_to_close" in m:
            print(f"  [actual mean by t_to_close]")
            for bucket, mm in m["markout_by_t_to_close"].items():
                print(f"    t=[{bucket:>7s}s]  n={mm['n']:>8,}  "
                      f"actual={mm['actual_mean_markout']:+.5f}  "
                      f"pred={mm['pred_mean_markout']:+.5f}  "
                      f"MAE={mm['mae']:.5f}")


if __name__ == "__main__":
    print("\n\n" + "#"*70)
    print("  MODEL 04b ADVERSE SELECTION (held-to-close PnL)")
    print("  target = markout_to_close  |  filter = filled==True")
    print("#"*70)

    # Leakage audit
    loader = LeakageSafeLoader()
    split_preview = loader.load(
        "model_04_adverse_selection",
        target_override="markout_to_close",
        verbose=False,
    )
    leakage_audit(split_preview)
    del split_preview

    # Ridge baseline
    print("--- Ridge baseline ---")
    ridge = RidgeMarkoutToCloseTrainer()
    ridge_metrics = ridge.train(target_override="markout_to_close")
    print_summary("ridge", ridge_metrics)

    # LightGBM primary
    print("\n--- LightGBM regression ---")
    lgbm = Model04bMarkoutToCloseTrainer()
    lgbm_metrics = lgbm.train(target_override="markout_to_close")
    print_summary("lightgbm", lgbm_metrics)

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps({
        "ridge": ridge_metrics,
        "lightgbm": lgbm_metrics,
    }, indent=2, default=str))
