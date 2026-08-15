"""Model 7 Microstructure Direction: LR baseline + LightGBM on tabular variant.

LEAKAGE AUDIT (verified before training):
  Excluded (leakage_only=true in schema):
    - mid_move_sign_1s/3s/5s/15s/30s  -- future direction labels (TARGET IS ONE OF THESE)
    - mid_move_1s/3s/5s/15s/30s       -- future price changes
    - max_favorable_excursion_5s/15s  -- future price path
    - max_adverse_excursion_5s/15s    -- future price path
    - resolved_side, resolved_side_label -- future resolution
    - flip_happened_after_t            -- future flip
    - terminal_delta_to_strike         -- terminal (future) BTC price vs strike

  Features that look risky but ARE safe:
    - mid_return_1s/3s/5s  -- PAST returns (lookback before anchor), not future
    - imbalance_slope_5s   -- slope of imbalance over PAST 5s
    - delta_to_strike      -- current BTC price vs strike (known at anchor time)
    - phase_bucket         -- derived from t_to_close_s (current time, not future)
"""
import sys, json
sys.path.insert(0, "src")

from model_factory.trainers.model_specific.model_07_microstructure_direction_trainer import (
    Model07MicrostructureDirectionTrainer,
    _map_target,
)
from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer, _to_float_array
from model_factory.dataset_loader import LeakageSafeLoader, DatasetSplit
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# LR baseline — fully self-contained, handles -1/0/+1 target
# ---------------------------------------------------------------------------

class LRMicrostructureDirectionTrainer(LogisticRegressionTrainer):
    MODEL_ID = "model_07_microstructure_direction"
    DEFAULT_HYPERPARAMS = {"C": 0.1, "max_iter": 500}

    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)

    def _fit(self, model, X_train, y_train):
        X, _ = _to_float_array(X_train) if isinstance(X_train, pl.DataFrame) else (np.asarray(X_train, float), None)
        y_raw = (y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)).astype(int)
        y_bin, valid = _map_target(y_raw)
        X_clean = X[valid]
        # NaN imputation
        col_medians = np.where(np.isnan(X_clean).all(axis=0), 0.0, np.nanmedian(X_clean, axis=0))
        nan_mask = np.isnan(X_clean)
        X_clean = X_clean.copy()
        X_clean[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        model.fit(X_clean, y_bin)
        model._nan_fill = col_medians

    def _calibrate(self, model, split: DatasetSplit):
        """Override: calibrate on binary 0/1 labels (excluding zero-move rows)."""
        from sklearn.isotonic import IsotonicRegression
        if split.val.is_empty():
            return model
        y_raw = split.val[split.target_column].to_numpy().astype(int)
        valid = y_raw != 0
        if not valid.any():
            return model
        y_bin = np.where(y_raw[valid] > 0, 1, 0).astype(int)
        X_val_v = split.val.drop(split.target_column).filter(pl.Series(valid))
        raw_probs = self._predict_proba(model, X_val_v)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, y_bin)
        model._calibrator = iso
        return model

    def _evaluate(self, model, split: DatasetSplit):
        # Reuse Model07's evaluation logic (handles -1/0/+1 target properly)
        # but with this model's predict method
        m7 = Model07MicrostructureDirectionTrainer.__new__(Model07MicrostructureDirectionTrainer)
        m7.model_id = self.model_id
        m7.loader = self.loader
        m7.writer = self.writer
        m7._calibrated_predict = lambda mdl, X: self._calibrated_predict(mdl, X)
        return m7._evaluate(model, split)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(split: DatasetSplit):
    LEAKAGE_PATTERNS = [
        "mid_move_sign_", "mid_move_", "max_favorable_excursion",
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
    print(f"  Target column    : {split.target_column}")
    print(f"  Feature columns  : {len(features)}")

    y_arr = split.train[split.target_column].to_numpy().astype(int)
    n_up   = int((y_arr == 1).sum())
    n_down = int((y_arr == -1).sum())
    n_zero = int((y_arr == 0).sum())
    print(f"  Target values    : {sorted(split.train[split.target_column].unique().to_list())}")
    print(f"  Target dist      : up(+1)={n_up:,}  down(-1)={n_down:,}  zero(0)={n_zero:,}  "
          f"zero_pct={100*n_zero/len(y_arr):.1f}%")
    print(f"  UP rate (excl 0) : {n_up/(n_up+n_down)*100:.1f}%")

    if violations:
        print(f"\n  *** LEAKAGE DETECTED *** {violations}")
        raise RuntimeError(f"Leakage columns in feature set: {violations}")
    print(f"  STATUS: PASS — no leakage columns detected")

    for col in ["mid_return_5s", "imbalance_mean_5s", "delta_to_strike", "t_to_close_s"]:
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
        ll  = m.get("log_loss", float("nan"))
        acc = m.get("direction_accuracy", float("nan"))
        n   = m.get("n_rows", "?")
        n_z = m.get("n_zero_moves_excluded", 0)
        up  = m.get("up_rate", float("nan"))
        print(f"  [{split:5s}] log_loss={ll:.5f}  acc={acc:.4f}  "
              f"n={n:,}  zero_excluded={n_z:,}  up_rate={up:.3f}")

        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta_ll  = m.get(f"vs_{bl}_log_loss_delta", float("nan"))
                delta_acc = m.get(f"vs_{bl}_acc_delta", float("nan"))
                sign = "BEATS" if delta_ll < 0 else "loses"
                print(f"           vs {bl:<30s}: ll={bm['log_loss']:.5f}  "
                      f"acc={bm.get('accuracy', float('nan')):.4f}  "
                      f"dll={delta_ll:+.5f}  dacc={delta_acc:+.4f}  [{sign}]")

        if "accuracy_by_delta_bucket" in m:
            print(f"  [accuracy by |delta| bucket]")
            for bucket, bm in m["accuracy_by_delta_bucket"].items():
                print(f"    |delta|={bucket:<12s}  acc={bm['accuracy']:.4f}  "
                      f"up_rate={bm['up_rate']:.3f}  n={bm['n']:,}")

        if "accuracy_by_phase" in m:
            print(f"  [accuracy by phase]  {m['accuracy_by_phase']}")

        if "direction_by_time_to_close" in m:
            print(f"  [by t_to_close]")
            for bucket, bm in m["direction_by_time_to_close"].items():
                if bm.get("rows", 0) > 0:
                    print(f"    t=[{bucket:>7s}s]  rows={bm['rows']:>7,}  "
                          f"ll={bm.get('log_loss', float('nan')):.5f}")

        if "brier_by_session" in m:
            print(f"  [by session]  {m['brier_by_session']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    print("\n\n" + "#"*70)
    print("  MODEL 7 MICROSTRUCTURE DIRECTION | variant=tabular")
    print("#"*70)

    # Leakage audit — load once, check, then discard
    loader = LeakageSafeLoader()
    split_preview = loader.load(
        "model_07_microstructure_direction",
        dataset_variant="tabular",
        verbose=False,
    )
    leakage_audit(split_preview)
    del split_preview

    # LR baseline
    print("--- Logistic Regression baseline ---")
    lr = LRMicrostructureDirectionTrainer()
    lr_metrics = lr.train(dataset_variant="tabular")
    print_summary("tabular", "logistic_regression", lr_metrics)
    all_results["tabular_lr"] = lr_metrics

    # LightGBM primary
    print("\n--- LightGBM ---")
    lgbm = Model07MicrostructureDirectionTrainer()
    lgbm_metrics = lgbm.train(dataset_variant="tabular")
    print_summary("tabular", "lightgbm", lgbm_metrics)
    all_results["tabular_lgbm"] = lgbm_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2, default=str))
