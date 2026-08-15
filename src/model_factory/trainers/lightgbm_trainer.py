from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import lightgbm as lgb

from .base_trainer import BaseTrainer


class LightGBMTrainer(BaseTrainer):
    """LightGBM gradient boosted trees trainer with early stopping.

    AUDIT FIX (2026-05-12): MAX_TRAIN_ROWS cap to bound peak RAM during fit.
    LightGBM is memory-efficient internally (histogram binning), but the
    input float matrix is allocated full size before binning.  At 8M rows
    x 50 features x float32 ~ 1.6 GB which is fine for an 8 GB budget.
    """

    ALGORITHM_NAME = "lightgbm"
    MAX_TRAIN_ROWS = 8_000_000

    def _build_model(self, hyperparams: dict[str, Any]) -> lgb.LGBMClassifier:
        params = {
            "num_leaves": hyperparams.get("num_leaves", 8),
            "min_child_samples": hyperparams.get("min_child_samples", 20),
            "learning_rate": hyperparams.get("learning_rate", 0.05),
            "n_estimators": hyperparams.get("n_estimators", 500),
            "subsample": hyperparams.get("subsample", 0.8),
            "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
            "reg_alpha": hyperparams.get("reg_alpha", 0.1),
            "reg_lambda": hyperparams.get("reg_lambda", 0.1),
            "random_state": hyperparams.get("random_state", 42),
            "verbose": -1,
            "objective": "binary",
            "metric": "binary_logloss",
        }
        return lgb.LGBMClassifier(**params)

    def _fit(self, model: lgb.LGBMClassifier, X_train: Any, y_train: Any) -> None:
        # AUDIT FIX (2026-05-12): subsample massive datasets to bound RAM.
        # ENV OVERRIDE: MAX_TRAIN_ROWS_OVERRIDE lets the operator further
        # reduce the cap when running on memory-constrained hardware.
        import os as _os
        _max = int(_os.environ.get("MAX_TRAIN_ROWS_OVERRIDE") or self.MAX_TRAIN_ROWS)
        if isinstance(X_train, pl.DataFrame):
            n_rows = X_train.height
            if n_rows > _max:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(n_rows, size=_max, replace=False)
                idx.sort()
                X_train = X_train[idx]
                if isinstance(y_train, pl.Series):
                    y_train = y_train[idx]
                else:
                    y_train = np.asarray(y_train)[idx]
                print(f"  [LGBM] subsampled {n_rows:,} -> {_max:,} rows to bound memory")
            model._feature_names = X_train.columns
        X = self._to_numpy(X_train)
        y = y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)
        model.fit(X, y.astype(int))

    def _predict_proba(self, model: lgb.LGBMClassifier, X: Any) -> np.ndarray:
        arr = self._to_numpy(X)
        return model.predict_proba(arr)[:, 1]

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            from model_factory.trainers.logistic_regression_trainer import _to_float_array
            arr, _ = _to_float_array(X)
            return arr
        return np.asarray(X, dtype=float)

    def get_feature_importance(self, model: lgb.LGBMClassifier) -> dict[str, float]:
        names = getattr(model, "_feature_names", None) or getattr(model, "feature_name_", None) or []
        importances = model.feature_importances_
        if not names or len(names) != len(importances):
            names = [f"f{i}" for i in range(len(importances))]
        total = importances.sum() or 1.0
        return {str(n): float(v / total) for n, v in sorted(zip(names, importances), key=lambda x: -x[1])}

