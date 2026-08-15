from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl

from model_factory.trainers.base_trainer import BaseTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_edge_sign_accuracy,
    compute_mispricing_by_delta_bucket,
    compute_mispricing_by_time_to_close,
    compute_stability_by_phase,
    compute_edge_by_session,
)


class RegressionBaseTrainer(BaseTrainer):
    """
    Base trainer for continuous regression targets.

    Overrides the classification-specific parts of BaseTrainer:
    - No isotonic calibration
    - Evaluation uses MAE, RMSE, sign accuracy, delta/time stratification
    - Baselines are constant-zero, constant-mean, and delta_to_strike linear
    """

    @abstractmethod
    def _predict(self, model: Any, X: Any) -> np.ndarray:
        """Return continuous scalar predictions, shape (n_samples,)."""

    # Satisfy BaseTrainer's abstract interface — regression routes through _predict
    def _predict_proba(self, model: Any, X: Any) -> np.ndarray:
        return self._predict(model, X)

    def _calibrate(self, model: Any, split: DatasetSplit) -> Any:
        return model

    def _calibrated_predict(self, model: Any, X: Any) -> np.ndarray:
        return self._chunked_predict(model, X)

    def _chunked_predict(self, model: Any, X: Any) -> np.ndarray:
        """AUDIT FIX (2026-05-12): chunked _predict to bound memory on huge eval sets."""
        if isinstance(X, pl.DataFrame) and X.height > self.PREDICT_CHUNK_SIZE:
            chunks = []
            for start in range(0, X.height, self.PREDICT_CHUNK_SIZE):
                end = min(start + self.PREDICT_CHUNK_SIZE, X.height)
                chunks.append(np.asarray(self._predict(model, X[start:end]), dtype=np.float32))
            return np.concatenate(chunks)
        return self._predict(model, X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        baselines = self._compute_regression_baselines(split)

        for name, frame in {"train": split.train, "val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_all = frame[split.target_column].to_numpy().astype(float)
            X_all = frame.drop(split.target_column)
            # AUDIT FIX (2026-05-12): use chunked predict for memory bounds
            y_pred_all = self._chunked_predict(model, X_all)

            # Drop NaN targets for evaluation
            valid = ~np.isnan(y_all)
            y_true = y_all[valid]
            y_pred = y_pred_all[valid]
            frame_v = frame.filter(pl.Series(valid))

            entry: dict[str, Any] = {
                "mae": compute_mae(y_true, y_pred),
                "rmse": compute_rmse(y_true, y_pred),
                "sign_accuracy": compute_edge_sign_accuracy(y_true, y_pred),
                "n_rows": int(valid.sum()),
                "n_rows_total": len(frame),
                "target_mean": float(np.nanmean(y_true)),
                "pred_mean": float(np.nanmean(y_pred)),
                "target_std": float(np.nanstd(y_true)),
            }

            # Baseline comparisons on val/test
            if name in ("val", "test") and name in baselines:
                entry["baselines"] = baselines[name]
                for bl_name, bl_m in baselines[name].items():
                    entry[f"vs_{bl_name}_mae_delta"] = entry["mae"] - bl_m["mae"]

            # Gate metric
            if name in ("val", "test"):
                entry["gate"] = self._compute_gate(y_true, y_pred)

            # Stratified breakdowns (all from frame_v to stay aligned with valid mask)
            if "t_to_close_s" in frame_v.columns:
                t = frame_v["t_to_close_s"].to_numpy().astype(float)
                entry["mispricing_by_time_to_close"] = compute_mispricing_by_time_to_close(y_true, y_pred, t)

            if "delta_to_strike" in frame_v.columns:
                delta = frame_v["delta_to_strike"].to_numpy().astype(float)
                entry["mispricing_by_delta_bucket"] = compute_mispricing_by_delta_bucket(
                    y_true, y_pred, np.abs(delta)
                )

            if "phase_bucket" in frame_v.columns:
                phase = frame_v["phase_bucket"].to_numpy()
                entry["sign_accuracy_by_phase"] = {
                    str(ph): compute_edge_sign_accuracy(y_true[phase == ph], y_pred[phase == ph])
                    for ph in sorted(set(phase))
                }

            if "snapshot_hour_utc" in frame_v.columns:
                hours = np.asarray(frame_v["snapshot_hour_utc"].to_numpy(), dtype=int)
                session_buckets = {
                    "utc_00_08": (hours >= 0) & (hours < 8),
                    "utc_08_16": (hours >= 8) & (hours < 16),
                    "utc_16_24": (hours >= 16) & (hours < 24),
                }
                entry["mae_by_session"] = {
                    k: compute_mae(y_true[m], y_pred[m])
                    for k, m in session_buckets.items() if m.any()
                }

            metrics[name] = entry

        return metrics

    def _compute_gate(self, y_true: np.ndarray, y_pred: np.ndarray, fee_threshold: float = 0.02) -> dict:
        """When model predicts |edge| > fee_threshold, check actual edge direction is correct."""
        confident_long = y_pred > fee_threshold
        confident_short = y_pred < -fee_threshold
        result: dict[str, Any] = {"fee_threshold": fee_threshold}
        if confident_long.any():
            result["mean_actual_edge_when_long"] = float(np.nanmean(y_true[confident_long]))
            result["n_long"] = int(confident_long.sum())
        if confident_short.any():
            result["mean_actual_edge_when_short"] = float(np.nanmean(y_true[confident_short]))
            result["n_short"] = int(confident_short.sum())
        # Pass if model's confident predictions have correct-signed actual edges
        long_ok = result.get("mean_actual_edge_when_long", 0.0) > 0
        short_ok = result.get("mean_actual_edge_when_short", 0.0) < 0
        result["gate_pass"] = long_ok and short_ok
        return result

    def _compute_regression_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        train_mean = float(split.train[split.target_column].cast(pl.Float64).mean() or 0.0)
        result: dict[str, dict[str, Any]] = {}

        # Fit delta_to_strike linear baseline on train once (non-NaN target rows only)
        delta_beta: float = 0.0
        if "delta_to_strike" in split.train.columns:
            train_y_all = split.train[split.target_column].to_numpy().astype(float)
            train_valid = ~np.isnan(train_y_all)
            train_delta = split.train["delta_to_strike"].fill_null(0.0).to_numpy().astype(float)[train_valid]
            train_y = train_y_all[train_valid]
            var_d = np.var(train_delta)
            if var_d > 1e-10:
                delta_beta = float(np.cov(train_delta, train_y)[0, 1] / var_d)

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_all = frame[split.target_column].to_numpy().astype(float)
            valid = ~np.isnan(y_all)
            y_true = y_all[valid]
            frame_v = frame.filter(pl.Series(valid))
            n = len(y_true)
            entry: dict[str, Any] = {}

            # Baseline 1: constant zero (null hypothesis — market is perfectly efficient)
            p_zero = np.zeros(n)
            entry["constant_zero"] = {
                "mae": compute_mae(y_true, p_zero),
                "rmse": compute_rmse(y_true, p_zero),
                "sign_accuracy": float("nan"),
            }

            # Baseline 2: constant train mean
            p_mean = np.full(n, train_mean)
            entry["constant_mean"] = {
                "mae": compute_mae(y_true, p_mean),
                "rmse": compute_rmse(y_true, p_mean),
                "sign_accuracy": compute_edge_sign_accuracy(y_true, p_mean),
            }

            # Baseline 3: delta_to_strike linear (OLS slope fitted on train)
            if "delta_to_strike" in frame_v.columns:
                delta = frame_v["delta_to_strike"].fill_null(0.0).to_numpy().astype(float)
                p_delta = delta * delta_beta
                entry["delta_linear"] = {
                    "mae": compute_mae(y_true, p_delta),
                    "rmse": compute_rmse(y_true, p_delta),
                    "sign_accuracy": compute_edge_sign_accuracy(y_true, p_delta),
                }

            result[name] = entry

        return result

    # Feature importance hook — available to LightGBM subclasses
    def _save_artifacts(self, model: Any, split: DatasetSplit, metrics: dict[str, Any], artifact_path: str) -> None:
        super()._save_artifacts(model, split, metrics, artifact_path)
