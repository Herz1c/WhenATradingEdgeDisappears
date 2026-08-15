from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from model_factory.trainers.linear_regression_trainer import LinearRegressionTrainer
from model_factory.trainers.lightgbm_regression_trainer import LightGBMRegressionTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import (
    compute_mae,
    compute_rmse,
)


def _vol_baselines(split: DatasetSplit) -> dict[str, dict[str, Any]]:
    """Compute regression baselines appropriate for a move-size target.

    Baselines:
      constant_zero  -- predict no move (EMH null: market is efficient)
      constant_mean  -- predict train mean (symmetric dist -> near 0)
      realized_vol   -- predict |mid_return_5s| as proxy for next-5s magnitude
                        (if past 5s was volatile, expect similar volatility)
    """
    train_mean = float(split.train[split.target_column].cast(pl.Float64).mean() or 0.0)
    result: dict[str, dict[str, Any]] = {}

    for name, frame in {"val": split.val, "test": split.test}.items():
        if frame.is_empty():
            continue
        y_all = frame[split.target_column].to_numpy().astype(float)
        valid = ~np.isnan(y_all)
        y_true = y_all[valid]
        frame_v = frame.filter(pl.Series(valid))
        n = len(y_true)
        entry: dict[str, Any] = {}

        # Baseline 1: constant zero (efficient market)
        entry["constant_zero"] = {
            "mae": compute_mae(y_true, np.zeros(n)),
            "rmse": compute_rmse(y_true, np.zeros(n)),
        }

        # Baseline 2: constant train mean (near 0 for symmetric distribution)
        entry["constant_mean"] = {
            "mae": compute_mae(y_true, np.full(n, train_mean)),
            "rmse": compute_rmse(y_true, np.full(n, train_mean)),
        }

        # Baseline 3: realized vol proxy -- |past 5s return| predicts |next 5s move|
        # This is the key baseline: does knowing recent volatility help?
        if "mid_return_5s" in frame_v.columns:
            abs_ret = np.abs(frame_v["mid_return_5s"].fill_null(0.0).to_numpy().astype(float))
            # Predict signed move = 0 (no direction), magnitude = abs_ret
            # Since mid_move_5s is signed, realized vol baseline predicts 0
            # But for MAE comparison we use abs_ret as the prediction of |move|
            # Compute MAE of |y_true| vs abs_ret for a fair apples-to-apples comparison
            entry["realized_vol_5s"] = {
                "mae": compute_mae(y_true, np.zeros(n)),   # still predicts 0 direction
                "mae_abs": compute_mae(np.abs(y_true), abs_ret),  # magnitude comparison
                "rmse": compute_rmse(y_true, np.zeros(n)),
            }

        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# Linear Regression baseline (stub was already LinearRegressionTrainer)
# ---------------------------------------------------------------------------

class Model08MoveSizeTrainer(LinearRegressionTrainer):
    """Linear regression baseline for Model 8: move-size prediction."""

    MODEL_ID = "model_08_move_size"
    DEFAULT_HYPERPARAMS = {"alpha": 1.0}

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)

    def _compute_regression_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        return _vol_baselines(split)


# ---------------------------------------------------------------------------
# LightGBM primary
# ---------------------------------------------------------------------------

class Model08LGBMTrainer(LightGBMRegressionTrainer):
    """LightGBM regressor for Model 8: move-size prediction."""

    MODEL_ID = "model_08_move_size"
    # Same conservative capacity as Model 7 -- high zero-move rate means
    # the model must generalise across a spike at 0 and tails
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

    def _compute_regression_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        return _vol_baselines(split)

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics = super()._evaluate(model, split)

        # Add |move| breakdown by confidence bucket (predicted |move| vs actual |move|)
        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty() or name not in metrics:
                continue
            y_all = frame[split.target_column].to_numpy().astype(float)
            valid = ~np.isnan(y_all)
            y_true = y_all[valid]
            frame_v = frame.filter(pl.Series(valid))
            y_pred = self._predict(model, frame_v.drop(split.target_column))

            abs_pred = np.abs(y_pred)
            abs_true = np.abs(y_true)

            # MAE on |move| by predicted-magnitude bucket
            buckets = [(0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, float("inf"))]
            mae_by_pred_magnitude: dict[str, Any] = {}
            for lo, hi in buckets:
                mask = (abs_pred >= lo) & (abs_pred < hi)
                if mask.any():
                    mae_by_pred_magnitude[f"{lo:.3f}_{hi:.3f}"] = {
                        "n": int(mask.sum()),
                        "mae": compute_mae(y_true[mask], y_pred[mask]),
                        "mean_actual_abs": float(np.mean(abs_true[mask])),
                        "mean_pred_abs": float(np.mean(abs_pred[mask])),
                        "zero_pct": float(np.mean(y_true[mask] == 0)),
                    }
            metrics[name]["mae_by_pred_magnitude"] = mae_by_pred_magnitude

        return metrics
