from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .regression_base_trainer import RegressionBaseTrainer
from .logistic_regression_trainer import _to_float_array


class LinearRegressionTrainer(RegressionBaseTrainer):
    """Ridge regression with standard scaling + median NaN imputation.

    DEPLOYMENT NOTE: the saved Pipeline includes SimpleImputer so
    `joblib.load(model.pkl).predict(X)` works directly on raw NaN-bearing
    arrays.

    AUDIT FIX (2026-05-12): MAX_TRAIN_ROWS cap (3M) for memory.
    """

    ALGORITHM_NAME = "linear_regression"
    MAX_TRAIN_ROWS_RIDGE = 3_000_000

    def _build_model(self, hyperparams: dict[str, Any]) -> Pipeline:
        # AUDIT FIX (2026-05-12): SimpleImputer inside Pipeline so saved pkl
        # is self-contained and survives load+predict without external wrapper.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=hyperparams.get("alpha", 1.0))),
        ])

    def _fit(self, model: Pipeline, X_train: Any, y_train: Any) -> None:
        # AUDIT FIX (2026-05-12): NaN target filter + subsample for memory.
        y = (y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)).astype(float)
        valid = ~np.isnan(y)
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.filter(pl.Series(valid))
            y = y[valid]
            if X_train.height > self.MAX_TRAIN_ROWS_RIDGE:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(X_train.height, size=self.MAX_TRAIN_ROWS_RIDGE, replace=False)
                idx.sort()
                X_train = X_train[idx]
                y = y[idx]
                print(f"  [Ridge] subsampled to {self.MAX_TRAIN_ROWS_RIDGE:,} rows to bound memory")
            X, _ = _to_float_array(X_train)
        else:
            X = np.asarray(X_train, float)[valid]
            y = y[valid]
        model.fit(X, y)
        try:
            imputer = model.named_steps.get("imputer")
            if imputer is not None and hasattr(imputer, "statistics_"):
                model._nan_fill = imputer.statistics_
        except Exception:
            pass

    def _predict(self, model: Pipeline, X: Any) -> np.ndarray:
        arr, _ = _to_float_array(X) if isinstance(X, pl.DataFrame) else (np.asarray(X, float), None)
        return model.predict(arr).astype(float)
