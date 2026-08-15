"""Model 8 Move Size: Linear Regression baseline + LightGBM on tabular variant.

LEAKAGE AUDIT (same feature set as Model 7, continuous target):
  Excluded (leakage_only=true):
    - mid_move_5s          -- TARGET ITSELF (future 5s price change)
    - mid_move_1s/3s/15s/30s        -- other future horizons
    - mid_move_sign_1s/3s/5s/15s/30s -- future direction labels
    - max_favorable_excursion_5s/15s -- future MFE (peak profit)
    - max_adverse_excursion_5s/15s  -- future MAE (peak loss)
    - resolved_side, flip_happened_after_t, terminal_* -- future resolution

  Safe features (lookback, not lookahead):
    - mid_return_1s/3s/5s  -- PAST returns before anchor
    - imbalance_slope_5s   -- PAST 5s imbalance trend
    - abs_delta_to_strike  -- current BTC distance from strike
"""
import sys, json
sys.path.insert(0, "src")

from model_factory.trainers.model_specific.model_08_move_size_trainer import (
    Model08MoveSizeTrainer,
    Model08LGBMTrainer,
)
from model_factory.dataset_loader import LeakageSafeLoader, DatasetSplit
import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(split: DatasetSplit):
    LEAKAGE_PATTERNS = [
        "mid_move_", "mid_move_sign_", "max_favorable_excursion",
        "max_adverse_excursion", "resolved_side", "flip_happened", "terminal_",
    ]
    features = [c for c in split.train.columns if c != split.target_column]
    violations = []
    for col in features:
        for pat in LEAKAGE_PATTERNS:
            if col.startswith(pat):
                violations.append(col)

    print(f"\n{'='*70}")
    print("  LEAKAGE AUDIT")
    print(f"{'='*70}")
    print(f"  Target column  : {split.target_column}")
    print(f"  Feature columns: {len(features)}")

    y_arr = split.train[split.target_column].to_numpy().astype(float)
    valid = ~np.isnan(y_arr)
    y_v = y_arr[valid]
    n_zero = int((y_v == 0).sum())
    print(f"  Target stats   : mean={np.nanmean(y_v):.6f}  std={np.nanstd(y_v):.6f}  "
          f"min={y_v.min():.4f}  max={y_v.max():.4f}")
    print(f"  Zero moves     : {n_zero:,} / {len(y_v):,} ({100*n_zero/len(y_v):.1f}%)")

    if violations:
        print(f"\n  *** LEAKAGE DETECTED *** {violations}")
        raise RuntimeError(f"Leakage columns: {violations}")
    print(f"  STATUS: PASS -- no leakage columns detected")
    for col in ["mid_return_5s", "imbalance_mean_5s", "abs_delta_to_strike", "t_to_close_s"]:
        print(f"  Safe feature [{col}]: {'PRESENT' if col in features else 'MISSING'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(variant, algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} | variant={variant}")
    print(f"{'='*70}")
    for split, m in metrics.items():
        mae  = m.get("mae", float("nan"))
        rmse = m.get("rmse", float("nan"))
        n    = m.get("n_rows", "?")
        tmean = m.get("target_mean", float("nan"))
        pmean = m.get("pred_mean", float("nan"))
        print(f"  [{split:5s}] MAE={mae:.6f}  RMSE={rmse:.6f}  "
              f"n={n:,}  target_mean={tmean:.6f}  pred_mean={pmean:.6f}")

        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_mae_delta", float("nan"))
                sign  = "BEATS" if delta < 0 else "loses"
                print(f"           vs {bl:<25s}: MAE={bm['mae']:.6f}  "
                      f"d_mae={delta:+.6f}  [{sign}]")

        if "mae_by_pred_magnitude" in m:
            print(f"  [MAE by predicted |move| bucket]")
            for bucket, bm in m["mae_by_pred_magnitude"].items():
                print(f"    pred|move|=[{bucket}]  n={bm['n']:>7,}  "
                      f"MAE={bm['mae']:.6f}  "
                      f"actual_abs={bm['mean_actual_abs']:.6f}  "
                      f"zero_pct={bm['zero_pct']:.2%}")

        if "mispricing_by_time_to_close" in m:
            print(f"  [by t_to_close]")
            for bucket, bm in m["mispricing_by_time_to_close"].items():
                if bm.get("n", 0) > 0:
                    print(f"    t=[{bucket:>7s}s]  n={bm['n']:>7,}  "
                          f"MAE={bm.get('mae', float('nan')):.6f}")

        if "mae_by_session" in m:
            print(f"  [by session]  {m['mae_by_session']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    print("\n\n" + "#"*70)
    print("  MODEL 8 MOVE SIZE | variant=tabular")
    print("#"*70)

    # Leakage audit
    loader = LeakageSafeLoader()
    split_preview = loader.load(
        "model_08_move_size",
        dataset_variant="tabular",
        verbose=False,
    )
    leakage_audit(split_preview)
    del split_preview

    # Linear Regression baseline
    print("--- Linear Regression baseline ---")
    lr = Model08MoveSizeTrainer()
    lr_metrics = lr.train(dataset_variant="tabular")
    print_summary("tabular", "linear_regression", lr_metrics)
    all_results["tabular_lr"] = lr_metrics

    # LightGBM primary
    print("\n--- LightGBM ---")
    lgbm = Model08LGBMTrainer()
    lgbm_metrics = lgbm.train(dataset_variant="tabular")
    print_summary("tabular", "lightgbm", lgbm_metrics)
    all_results["tabular_lgbm"] = lgbm_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2, default=str))
