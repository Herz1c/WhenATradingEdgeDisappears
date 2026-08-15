"""Model 4: Adverse Selection / Markout regression.

Target: `markout_1s` (continuous, signed) — what's the expected mark-to-market
move 1s after a fill?  Positive = favorable (we filled at a good price);
negative = adverse (we filled into a stale book just before the market moved
against us).

Strategic role:
  Combined with Model 3 fill probability, gives:
      expected_edge_per_quote = P(fill) * E[markout | fill] - fees
  Answers: "When we DO get filled, do we make money or get adversely selected?"

Dataset filter: `filled == True` (only conditional on fills) — applied by
LeakageSafeLoader via the registry's `dataset.filter` field.

Leakage guard: all outcome fields are in leakage_only_columns.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl

from model_factory.trainers.lightgbm_regression_trainer import LightGBMRegressionTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import compute_mae, compute_rmse


class Model04AdverseSelectionTrainer(LightGBMRegressionTrainer):
    """LightGBM regressor for markout_1s on filled orders."""

    MODEL_ID = "model_04_adverse_selection"

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
        """Markout baselines:
          1. constant_zero (no markout assumption)
          2. constant_mean (train mean markout)
          3. order_side_conditioned (mean markout per buy/sell)
        """
        train_y = split.train[split.target_column].cast(pl.Float64).to_numpy()
        valid_tr = ~np.isnan(train_y)
        train_mean = float(train_y[valid_tr].mean()) if valid_tr.any() else 0.0
        result: dict[str, dict[str, Any]] = {}

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_all = frame[split.target_column].cast(pl.Float64).to_numpy()
            valid = ~np.isnan(y_all)
            y_true = y_all[valid]
            n = len(y_true)
            if n == 0:
                continue
            entry: dict[str, Any] = {}

            entry["constant_zero"] = {
                "mae": compute_mae(y_true, np.zeros(n)),
                "rmse": compute_rmse(y_true, np.zeros(n)),
            }
            entry["constant_mean"] = {
                "mae": compute_mae(y_true, np.full(n, train_mean)),
                "rmse": compute_rmse(y_true, np.full(n, train_mean)),
                "train_mean": train_mean,
            }

            # order_side conditioned
            if "order_side" in frame.columns and "order_side" in split.train.columns:
                tr_side = split.train["order_side"].fill_null("unknown").to_numpy()
                tr_y_arr = split.train[split.target_column].cast(pl.Float64).to_numpy()
                tr_valid = ~np.isnan(tr_y_arr)
                side_means: dict[str, float] = {}
                for sd in np.unique(tr_side):
                    m = (tr_side == sd) & tr_valid
                    if m.sum() >= 100:
                        side_means[str(sd)] = float(tr_y_arr[m].mean())
                eval_side = frame["order_side"].fill_null("unknown").to_numpy()[valid]
                p_side = np.array([side_means.get(str(v), train_mean) for v in eval_side])
                entry["order_side_conditioned"] = {
                    "mae": compute_mae(y_true, p_side),
                    "rmse": compute_rmse(y_true, p_side),
                }

            result[name] = entry
        return result

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics = super()._evaluate(model, split)

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty() or name not in metrics:
                continue
            y_all = frame[split.target_column].cast(pl.Float64).to_numpy()
            valid = ~np.isnan(y_all)
            y_true = y_all[valid]
            if len(y_true) == 0:
                continue
            frame_v = frame.filter(pl.Series(valid))
            X = frame_v.drop(split.target_column)
            y_pred = self._predict(model, X)

            # Signed-direction accuracy: do we predict the SIGN of markout correctly?
            # Only count rows where actual markout is non-zero
            nonzero = y_true != 0
            if nonzero.any():
                sign_acc = float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])))
                metrics[name]["sign_accuracy"] = sign_acc

            # Adverse fill detection: precision/recall of "markout < -threshold"
            for adv_thresh in [0.001, 0.005, 0.01, 0.02]:
                actual_adverse = y_true < -adv_thresh
                pred_adverse = y_pred < -adv_thresh
                if actual_adverse.sum() > 0 and pred_adverse.sum() > 0:
                    tp = int((pred_adverse & actual_adverse).sum())
                    prec = tp / max(pred_adverse.sum(), 1)
                    rec = tp / max(actual_adverse.sum(), 1)
                    metrics[name][f"adverse_detect_thr_{adv_thresh}"] = {
                        "n_actual_adverse": int(actual_adverse.sum()),
                        "n_predicted_adverse": int(pred_adverse.sum()),
                        "precision": prec,
                        "recall": rec,
                    }

            # Markout by t_to_close bucket
            if "t_to_close_s" in frame_v.columns:
                t = frame_v["t_to_close_s"].fill_null(-1.0).to_numpy().astype(float)
                buckets = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 9e9)]
                m_by_ttc: dict[str, Any] = {}
                for lo, hi in buckets:
                    mask = (t >= lo) & (t < hi)
                    if mask.sum() < 50:
                        continue
                    m_by_ttc[f"{lo}_{hi if hi < 9e8 else 'inf'}"] = {
                        "n": int(mask.sum()),
                        "actual_mean_markout": float(y_true[mask].mean()),
                        "pred_mean_markout": float(y_pred[mask].mean()),
                        "mae": compute_mae(y_true[mask], y_pred[mask]),
                    }
                metrics[name]["markout_by_t_to_close"] = m_by_ttc

            # Markout by order_side
            if "order_side" in frame_v.columns:
                side = frame_v["order_side"].fill_null("unknown").to_numpy()
                m_by_side: dict[str, Any] = {}
                for sd in np.unique(side):
                    mask = side == sd
                    if mask.sum() < 50:
                        continue
                    m_by_side[str(sd)] = {
                        "n": int(mask.sum()),
                        "actual_mean_markout": float(y_true[mask].mean()),
                        "pred_mean_markout": float(y_pred[mask].mean()),
                        "mae": compute_mae(y_true[mask], y_pred[mask]),
                    }
                metrics[name]["markout_by_order_side"] = m_by_side

        return metrics
