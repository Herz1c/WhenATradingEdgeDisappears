from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import lightgbm as lgb

from .regression_base_trainer import RegressionBaseTrainer
from .logistic_regression_trainer import _to_float_array


class LightGBMRegressionTrainer(RegressionBaseTrainer):
    """LightGBM regressor for continuous targets.

    AUDIT FIX (2026-05-12): MAX_TRAIN_ROWS cap to bound peak RAM.
    """

    ALGORITHM_NAME = "lightgbm"
    MAX_TRAIN_ROWS = 8_000_000

    def _build_model(self, hyperparams: dict[str, Any]) -> lgb.LGBMRegressor:
        params = {
            "num_leaves": hyperparams.get("num_leaves", 31),
            "min_child_samples": hyperparams.get("min_child_samples", 50),
            "learning_rate": hyperparams.get("learning_rate", 0.05),
            "n_estimators": hyperparams.get("n_estimators", 500),
            "subsample": hyperparams.get("subsample", 0.8),
            "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
            "reg_alpha": hyperparams.get("reg_alpha", 0.1),
            "reg_lambda": hyperparams.get("reg_lambda", 1.0),
            "random_state": hyperparams.get("random_state", 42),
            "verbose": -1,
            "objective": "regression",
            "metric": "mae",
        }
        return lgb.LGBMRegressor(**params)

    def _fit(self, model: lgb.LGBMRegressor, X_train: Any, y_train: Any) -> None:
        # AUDIT FIX (2026-05-12): subsample massive datasets to bound RAM.
        # NaN target rows are dropped FIRST, then subsample from the survivors.
        # ENV OVERRIDE: MAX_TRAIN_ROWS_OVERRIDE further reduces the cap on
        # memory-constrained machines.
        import os as _os
        _max = int(_os.environ.get("MAX_TRAIN_ROWS_OVERRIDE") or self.MAX_TRAIN_ROWS)
        y = (y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)).astype(float)
        valid = ~np.isnan(y)
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.filter(pl.Series(valid))
            y = y[valid]
            if X_train.height > _max:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(X_train.height, size=_max, replace=False)
                idx.sort()
                X_train = X_train[idx]
                y = y[idx]
                print(f"  [LGBM-Reg] subsampled to {_max:,} rows to bound memory")
            model._feature_names = X_train.columns
            X = self._to_numpy(X_train)
        else:
            X = np.asarray(X_train)[valid]
            y = y[valid]
        model.fit(X, y)

    def _predict(self, model: lgb.LGBMRegressor, X: Any) -> np.ndarray:
        return model.predict(self._to_numpy(X)).astype(float)

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            arr, _ = _to_float_array(X)
            return arr
        return np.asarray(X, dtype=float)

    def get_feature_importance(self, model: lgb.LGBMRegressor) -> dict[str, float]:
        names = getattr(model, "_feature_names", None) or []
        importances = model.feature_importances_
        if not names or len(names) != len(importances):
            names = [f"f{i}" for i in range(len(importances))]
        total = importances.sum() or 1.0
        return {str(n): float(v / total) for n, v in sorted(zip(names, importances), key=lambda x: -x[1])}
