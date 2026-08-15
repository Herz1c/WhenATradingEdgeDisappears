from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_ece,
    compute_edge_by_time_to_close,
    compute_edge_by_session,
    compute_stability_by_phase,
)


def _map_target(y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map int8 sign target (-1/0/+1) to binary (0/1).

    Returns:
        y_binary: 0 = down (-1), 1 = up (+1)
        valid_mask: True for rows where target != 0 (exclude ties)

    Leakage safety: this mapping is purely mechanical, no future
    information is introduced. Zero-move rows are excluded so the
    classifier sees only clear directional labels.
    """
    valid = y_raw != 0
    y_bin = np.where(y_raw[valid] > 0, 1, 0).astype(int)
    return y_bin, valid


class Model07MicrostructureDirectionTrainer(LightGBMTrainer):
    """Trainer for Model 7: token microstructure direction model.

    Target: mid_move_sign_5s in {-1, 0, +1}
      -1 = token mid moved DOWN in next 5 seconds
       0 = no movement (excluded from training and evaluation)
      +1 = token mid moved UP in next 5 seconds

    The trainer maps -1→0, +1→1 and filters out zero-move rows before
    fitting and before every evaluation. This is NOT leakage — the
    exclusion is based on the target value itself, not on any future
    feature.
    """

    MODEL_ID = "model_07_microstructure_direction"
    # Conservative capacity — microstructure noise is high, avoid overfitting
    DEFAULT_HYPERPARAMS = {
        "num_leaves": 31,
        "min_child_samples": 500,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)

    # ------------------------------------------------------------------
    # Calibration override: use binary 0/1 labels (exclude zeros)
    # ------------------------------------------------------------------

    def _calibrate(self, model: Any, split: DatasetSplit) -> Any:
        # AUDIT FIX (2026-05-12): adopt the val 50/50 cal/eval split pattern
        # so the calibrator is not fit and evaluated on the same rows.
        # Same guardrails as BaseTrainer._calibrate.
        from sklearn.isotonic import IsotonicRegression

        if split.val.is_empty():
            model._calibration_meta = {"calibrated": False, "reason": "empty_val"}
            return model

        y_raw = split.val[split.target_column].to_numpy().astype(int)
        nonzero_mask = y_raw != 0
        n_nonzero = int(nonzero_mask.sum())
        y_bin_full = np.where(y_raw > 0, 1, 0).astype(int)
        n_pos_full = int(y_bin_full[nonzero_mask].sum())
        n_neg_full = n_nonzero - n_pos_full

        if n_nonzero < self.MIN_VAL_FOR_CAL or min(n_pos_full, n_neg_full) < self.MIN_POS_FOR_CAL:
            model._calibration_meta = {
                "calibrated": False,
                "reason": f"insufficient non-zero val rows (n={n_nonzero}, "
                          f"n_pos={n_pos_full}, n_neg={n_neg_full})",
            }
            return model

        # 50/50 split of the NON-ZERO rows into cal and eval halves
        nonzero_indices = np.where(nonzero_mask)[0]
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(nonzero_indices))
        cal_local = nonzero_indices[perm[: len(perm) // 2]]
        cal_mask_global = np.zeros(len(y_raw), dtype=bool)
        cal_mask_global[cal_local] = True

        y_bin_cal = y_bin_full[cal_mask_global]
        X_val = split.val.drop(split.target_column)
        # Filter val to non-zero rows for prediction
        full_raw_probs = self._predict_proba(model, X_val)
        raw_probs_cal = full_raw_probs[cal_mask_global]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs_cal, y_bin_cal)
        model._calibrator = iso
        # Eval mask: non-zero rows NOT in cal set (held-out half of non-zeros)
        eval_mask = nonzero_mask & (~cal_mask_global)
        model._val_eval_mask = eval_mask
        model._calibration_meta = {
            "calibrated": True,
            "calibration_set": "val_nonzero_random_half",
            "n_cal": int(cal_mask_global.sum()),
            "n_val_eval": int(eval_mask.sum()),
            "n_pos_cal": int(y_bin_cal.sum()),
        }
        return model

    # ------------------------------------------------------------------
    # Fit override: handle -1/0/+1 target
    # ------------------------------------------------------------------

    def _fit(self, model: Any, X_train: Any, y_train: Any) -> None:
        import lightgbm as lgb
        X = self._to_numpy(X_train)
        y_raw = (y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)).astype(int)
        y_bin, valid = _map_target(y_raw)
        X_clean = X[valid]
        if isinstance(X_train, pl.DataFrame):
            model._feature_names = X_train.columns
        model.fit(X_clean, y_bin)

    # ------------------------------------------------------------------
    # Evaluate override: filter zeros, add microstructure breakdowns
    # ------------------------------------------------------------------

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        baselines = self._compute_baselines(split)

        for name, frame in {"train": split.train, "val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue

            y_raw = frame[split.target_column].to_numpy().astype(int)
            valid = y_raw != 0
            n_total = len(y_raw)
            n_zeros = int((y_raw == 0).sum())

            frame_v = frame.filter(pl.Series(valid))
            y_true = np.where(y_raw[valid] > 0, 1, 0).astype(int)
            X_v = frame_v.drop(split.target_column)
            y_prob = self._calibrated_predict(model, X_v)

            ll = compute_log_loss(y_true, y_prob)
            bs = compute_brier_score(y_true, y_prob)
            ece = compute_ece(y_true, y_prob)
            acc = float(np.mean((y_prob >= 0.5).astype(int) == y_true))
            up_rate = float(y_true.mean())

            entry: dict[str, Any] = {
                "log_loss": ll,
                "brier_score": bs,
                "ece": ece,
                "direction_accuracy": acc,
                "n_rows": int(valid.sum()),
                "n_rows_total": n_total,
                "n_zero_moves_excluded": n_zeros,
                "pct_zero_moves": float(n_zeros / n_total) if n_total > 0 else 0.0,
                "up_rate": up_rate,
            }

            if name in ("val", "test") and name in baselines:
                entry["baselines"] = baselines[name]
                for bl_name, bl_m in baselines[name].items():
                    entry[f"vs_{bl_name}_log_loss_delta"] = ll - bl_m["log_loss"]
                    entry[f"vs_{bl_name}_acc_delta"] = acc - bl_m.get("accuracy", 0.5)

            # Stratified breakdowns
            if "t_to_close_s" in frame_v.columns:
                t = frame_v["t_to_close_s"].to_numpy().astype(float)
                entry["direction_by_time_to_close"] = compute_edge_by_time_to_close(y_true, y_prob, t)

            if "phase_bucket" in frame_v.columns:
                phase = frame_v["phase_bucket"].to_numpy()
                entry["log_loss_by_phase"] = compute_stability_by_phase(
                    y_true, y_prob, phase, compute_log_loss
                )
                entry["accuracy_by_phase"] = {
                    str(ph): float(np.mean((y_prob[phase == ph] >= 0.5).astype(int) == y_true[phase == ph]))
                    for ph in sorted(set(phase)) if (phase == ph).any()
                }

            if "snapshot_hour_utc" in frame_v.columns:
                hour = frame_v["snapshot_hour_utc"].to_numpy()
                entry["brier_by_session"] = compute_edge_by_session(y_true, y_prob, hour)

            # Direction accuracy by delta bucket (key finding: does model add alpha beyond delta?)
            if "abs_delta_to_strike" in frame_v.columns:
                abs_delta = frame_v["abs_delta_to_strike"].fill_null(0.0).to_numpy().astype(float)
                buckets = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, float("inf"))]
                entry["accuracy_by_delta_bucket"] = {}
                for lo, hi in buckets:
                    mask = (abs_delta >= lo) & (abs_delta < hi)
                    if mask.any():
                        acc_b = float(np.mean((y_prob[mask] >= 0.5).astype(int) == y_true[mask]))
                        n_b = int(mask.sum())
                        up_b = float(y_true[mask].mean())
                        entry["accuracy_by_delta_bucket"][f"${lo}_{hi}"] = {
                            "accuracy": acc_b,
                            "up_rate": up_b,
                            "n": n_b,
                        }

            metrics[name] = entry

        return metrics

    # ------------------------------------------------------------------
    # Baselines for gate: must_add_value_above_fair_resolution_layer
    # ------------------------------------------------------------------

    def _compute_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        # Compute train up_rate from non-zero rows
        y_train_raw = split.train[split.target_column].to_numpy().astype(int)
        valid_train = y_train_raw != 0
        y_train_bin = np.where(y_train_raw[valid_train] > 0, 1, 0).astype(int)
        train_up_rate = float(y_train_bin.mean()) if len(y_train_bin) > 0 else 0.5

        result: dict[str, dict[str, Any]] = {}

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue

            y_raw = frame[split.target_column].to_numpy().astype(int)
            valid = y_raw != 0
            y_true = np.where(y_raw[valid] > 0, 1, 0).astype(int)
            frame_v = frame.filter(pl.Series(valid))
            n = len(y_true)
            entry: dict[str, Any] = {}

            # Baseline 1: constant train up rate
            p_const = np.full(n, train_up_rate)
            acc_const = float(np.mean((p_const >= 0.5).astype(int) == y_true))
            entry["constant_up_rate"] = {
                "log_loss": compute_log_loss(y_true, p_const),
                "brier_score": compute_brier_score(y_true, p_const),
                "accuracy": acc_const,
            }

            # Baseline 2: delta_sign — BTC above strike → predict UP token goes up
            # This proxies what the fair resolution model knows (delta_sign feature)
            if "delta_sign" in frame_v.columns:
                delta_sign = frame_v["delta_sign"].fill_null(0).to_numpy().astype(int)
                # delta_sign=+1 means BTC above strike → UP token currently "winning"
                # If UP token is winning, its mid is near 1.0, less room to go up → slight contrarian
                # Use as a neutral 0.5 + small tilt, not a strong signal
                p_delta = np.where(delta_sign > 0, 0.55, 0.45).astype(float)
                p_delta = np.where(delta_sign == 0, 0.5, p_delta)
                acc_delta = float(np.mean((p_delta >= 0.5).astype(int) == y_true))
                entry["delta_sign_heuristic"] = {
                    "log_loss": compute_log_loss(y_true, p_delta),
                    "brier_score": compute_brier_score(y_true, p_delta),
                    "accuracy": acc_delta,
                }

            # Baseline 3: momentum — if past 5s was up (+mid_return_5s > 0), predict up again
            if "mid_return_5s" in frame_v.columns:
                ret5 = frame_v["mid_return_5s"].fill_null(0.0).to_numpy().astype(float)
                p_mom = np.where(ret5 > 0, 0.6, np.where(ret5 < 0, 0.4, 0.5)).astype(float)
                acc_mom = float(np.mean((p_mom >= 0.5).astype(int) == y_true))
                entry["momentum_5s"] = {
                    "log_loss": compute_log_loss(y_true, p_mom),
                    "brier_score": compute_brier_score(y_true, p_mom),
                    "accuracy": acc_mom,
                }

            # Baseline 4: contrarian — if past 5s was up, predict down (mean reversion)
            if "mid_return_5s" in frame_v.columns:
                p_con = np.where(ret5 > 0, 0.4, np.where(ret5 < 0, 0.6, 0.5)).astype(float)
                acc_con = float(np.mean((p_con >= 0.5).astype(int) == y_true))
                entry["contrarian_5s"] = {
                    "log_loss": compute_log_loss(y_true, p_con),
                    "brier_score": compute_brier_score(y_true, p_con),
                    "accuracy": acc_con,
                }

            result[name] = entry

        return result
