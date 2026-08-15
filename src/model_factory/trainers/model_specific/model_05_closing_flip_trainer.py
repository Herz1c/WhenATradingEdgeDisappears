from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_flip_precision_recall,
    compute_flip_rate_by_delta_bucket,
)


class Model05ClosingFlipTrainer(LightGBMTrainer):
    """Trainer for Model 5: closing-flip model."""

    MODEL_ID = "model_05_closing_flip"
    # Conservative vs Model 2 — flip events are rare, need large leaves to avoid overfitting rare class
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

    def _compute_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        train_flip_rate = float(split.train[split.target_column].cast(pl.Float64).mean())
        result: dict[str, dict[str, Any]] = {}

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            n = len(y_true)
            entry: dict[str, Any] = {}

            # Baseline 1: constant train flip rate
            p_const = np.full(n, train_flip_rate)
            entry["constant_flip_rate"] = {
                "log_loss": compute_log_loss(y_true, p_const),
                "brier_score": compute_brier_score(y_true, p_const),
            }

            # Baseline 2: abs_delta_to_strike heuristic — closer to strike = more likely to flip
            # p_flip = sigmoid(-|delta| * 0.15): at |delta|=0 -> 0.5, at |delta|=10 -> 0.22, at |delta|=50 -> 0.0007
            if "abs_delta_to_strike" in frame.columns:
                abs_delta = frame["abs_delta_to_strike"].fill_null(0.0).to_numpy().astype(float)
                p_abs = 1.0 / (1.0 + np.exp(abs_delta * 0.15))
                p_abs = np.clip(p_abs, 0.01, 0.99)
                entry["abs_delta_heuristic"] = {
                    "log_loss": compute_log_loss(y_true, p_abs),
                    "brier_score": compute_brier_score(y_true, p_abs),
                }

            result[name] = entry

        return result

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics = super()._evaluate(model, split)

        # Add flip-specific stratification to val/test
        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty() or name not in metrics:
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            y_prob = self._calibrated_predict(model, frame.drop(split.target_column))

            metrics[name]["flip_precision_recall_t50"] = compute_flip_precision_recall(y_prob, y_true, threshold=0.50)
            metrics[name]["flip_precision_recall_t15"] = compute_flip_precision_recall(y_prob, y_true, threshold=0.15)

            if "abs_delta_to_strike" in frame.columns:
                abs_delta = frame["abs_delta_to_strike"].fill_null(0.0).to_numpy().astype(float)
                metrics[name]["flip_rate_by_delta_bucket"] = compute_flip_rate_by_delta_bucket(y_true, abs_delta)

        return metrics
