"""Model 3: Fill Probability classifier.

Target: `filled` (boolean) — given a hypothetical maker quote spec at a
decision timestamp, will this order get filled before cancel/expiry?

Strategic role:
  P(fill | quote spec) is the prerequisite for any maker-first edge analysis.
  Combined with Model 4 markout, gives:
      expected_edge_per_quote = P(fill) * E[markout | fill] - fees
  Without this model, we cannot answer "is maker-first viable?".

Leakage guard (enforced by LeakageSafeLoader from leakage_only_columns):
  - All outcome fields (filled, full_fill, time_to_fill, markout_*, etc.)
  - All resolution fields (resolved_side, terminal_*, hold_to_close_edge)
  - All replace/cancel lifecycle fields
"""
from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer
from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.metrics import (
    compute_log_loss,
    compute_brier_score,
)


class Model03FillProbabilityTrainer(LightGBMTrainer):
    """LightGBM classifier for fill probability."""

    MODEL_ID = "model_03_fill_probability"

    # Conservative capacity given ~2M+ training rows and 48 features
    DEFAULT_HYPERPARAMS = {
        "num_leaves": 63,
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
        """Fill-rate baselines:
          1. constant_fill_rate (train mean fill rate)
          2. constant_zero (predict never-filled)
          3. post_only_conditioned (different fill rate for post_only vs not)
          4. price_level_conditioned (different fill rate per price_level)
        """
        train_rate = float(split.train[split.target_column].cast(pl.Float64).mean())
        result: dict[str, dict[str, Any]] = {}

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            n = len(y_true)
            entry: dict[str, Any] = {}

            # Baseline 1: constant fill rate
            p_const = np.full(n, train_rate)
            entry["constant_fill_rate"] = {
                "log_loss": compute_log_loss(y_true, p_const),
                "brier_score": compute_brier_score(y_true, p_const),
            }

            # Baseline 2: post_only conditioned (categorical)
            if "post_only" in frame.columns and "post_only" in split.train.columns:
                tr_po = split.train["post_only"].fill_null(False).to_numpy().astype(bool)
                tr_y = split.train[split.target_column].to_numpy().astype(float)
                rate_po_true = float(tr_y[tr_po].mean()) if tr_po.any() else train_rate
                rate_po_false = float(tr_y[~tr_po].mean()) if (~tr_po).any() else train_rate
                eval_po = frame["post_only"].fill_null(False).to_numpy().astype(bool)
                p_cond = np.where(eval_po, rate_po_true, rate_po_false)
                p_cond = np.clip(p_cond, 0.01, 0.99)
                entry["post_only_conditioned"] = {
                    "log_loss": compute_log_loss(y_true, p_cond),
                    "brier_score": compute_brier_score(y_true, p_cond),
                }

            # Baseline 3: price_level conditioned (bid_at_best, bid_inside, etc.)
            if "price_level" in frame.columns and "price_level" in split.train.columns:
                tr_pl = split.train["price_level"].fill_null("unknown").to_numpy()
                tr_y = split.train[split.target_column].to_numpy().astype(float)
                level_rates: dict[str, float] = {}
                for lvl in np.unique(tr_pl):
                    mask = tr_pl == lvl
                    if mask.sum() >= 100:
                        level_rates[str(lvl)] = float(tr_y[mask].mean())
                eval_pl = frame["price_level"].fill_null("unknown").to_numpy()
                p_lvl = np.array([level_rates.get(str(v), train_rate) for v in eval_pl])
                p_lvl = np.clip(p_lvl, 0.01, 0.99)
                entry["price_level_conditioned"] = {
                    "log_loss": compute_log_loss(y_true, p_lvl),
                    "brier_score": compute_brier_score(y_true, p_lvl),
                }

            result[name] = entry
        return result

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        """Standard eval + stratified fill-rate breakdown by t_to_close, price_level."""
        metrics = super()._evaluate(model, split)

        # Stratified eval for val/test
        val_eval_mask = getattr(model, "_val_eval_mask", None)
        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty() or name not in metrics:
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            X = frame.drop(split.target_column)
            y_prob = self._calibrated_predict(model, X)

            # AUDIT FIX: apply val_eval_mask consistently so y_true / y_prob /
            # stratification arrays stay aligned (calibrator was fit on the
            # other half of val).
            row_mask = None
            if name == "val" and val_eval_mask is not None and len(val_eval_mask) == len(y_true):
                row_mask = val_eval_mask
                y_true = y_true[row_mask]
                y_prob = y_prob[row_mask]

            def _aligned(arr):
                return arr[row_mask] if row_mask is not None else arr

            # Fill rate by t_to_close
            if "t_to_close_s" in frame.columns:
                t_to_close = _aligned(frame["t_to_close_s"].fill_null(-1.0).to_numpy().astype(float))
                buckets = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 9e9)]
                fill_by_ttc: dict[str, dict[str, Any]] = {}
                for lo, hi in buckets:
                    mask = (t_to_close >= lo) & (t_to_close < hi)
                    if mask.sum() < 50:
                        continue
                    fill_by_ttc[f"{lo}_{hi if hi < 9e8 else 'inf'}"] = {
                        "n": int(mask.sum()),
                        "actual_fill_rate": float(y_true[mask].mean()),
                        "pred_fill_rate": float(y_prob[mask].mean()),
                        "log_loss": compute_log_loss(y_true[mask], y_prob[mask]),
                        "brier_score": compute_brier_score(y_true[mask], y_prob[mask]),
                    }
                metrics[name]["fill_rate_by_t_to_close"] = fill_by_ttc

            # Fill rate by price_level
            if "price_level" in frame.columns:
                price_lvl = _aligned(frame["price_level"].fill_null("unknown").to_numpy())
                fill_by_pl: dict[str, dict[str, Any]] = {}
                for lvl in np.unique(price_lvl):
                    mask = price_lvl == lvl
                    if mask.sum() < 50:
                        continue
                    fill_by_pl[str(lvl)] = {
                        "n": int(mask.sum()),
                        "actual_fill_rate": float(y_true[mask].mean()),
                        "pred_fill_rate": float(y_prob[mask].mean()),
                        "log_loss": compute_log_loss(y_true[mask], y_prob[mask]),
                    }
                metrics[name]["fill_rate_by_price_level"] = fill_by_pl

            # Fill rate by post_only
            if "post_only" in frame.columns:
                po = _aligned(frame["post_only"].fill_null(False).to_numpy().astype(bool))
                fill_by_po: dict[str, dict[str, Any]] = {}
                for po_val, key in [(True, "post_only"), (False, "non_post_only")]:
                    mask = po == po_val
                    if mask.sum() < 50:
                        continue
                    fill_by_po[key] = {
                        "n": int(mask.sum()),
                        "actual_fill_rate": float(y_true[mask].mean()),
                        "pred_fill_rate": float(y_prob[mask].mean()),
                        "log_loss": compute_log_loss(y_true[mask], y_prob[mask]),
                    }
                metrics[name]["fill_rate_by_post_only"] = fill_by_po

        return metrics
