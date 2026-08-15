from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base_trainer import BaseTrainer


def _to_float_array(df: pl.DataFrame, dtype: type = np.float32) -> tuple[np.ndarray, list[str]]:
    """Convert a polars DataFrame to a float numpy array.

    - Boolean columns cast to float.
    - String/categorical columns encoded as integer codes.
    - All remaining numeric columns cast to float.
    Returns (array, kept_column_names).

    AUDIT FIX (2026-05-12): Pre-allocates a single output buffer in float32
    by default (half the memory of float64) and writes columns in-place.
    Previous version allocated 3 copies of the matrix (~3x peak memory).
    """
    # First pass: determine which columns are convertible and their order
    convertible_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == pl.Boolean:
            convertible_cols.append(col)
        elif s.dtype in (pl.Utf8, pl.Categorical, pl.String):
            convertible_cols.append(col)
        else:
            try:
                _ = s.cast(pl.Float64, strict=False)
                convertible_cols.append(col)
            except Exception:
                pass  # skip unconvertible columns

    if not convertible_cols:
        raise ValueError("No convertible columns found in DataFrame.")

    n_rows = df.height
    n_cols = len(convertible_cols)
    # Pre-allocate one output buffer; write columns in-place to avoid 3x peak memory
    arr = np.empty((n_rows, n_cols), dtype=dtype)
    for j, col in enumerate(convertible_cols):
        s = df[col]
        if s.dtype == pl.Boolean:
            arr[:, j] = s.cast(pl.Float64).to_numpy().astype(dtype, copy=False)
        elif s.dtype in (pl.Utf8, pl.Categorical, pl.String):
            arr[:, j] = s.cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy().astype(dtype, copy=False)
        else:
            arr[:, j] = s.cast(pl.Float64).to_numpy().astype(dtype, copy=False)
    return arr, convertible_cols


class LogisticRegressionTrainer(BaseTrainer):
    """Sklearn logistic regression with standard scaling + median NaN imputation.

    DEPLOYMENT NOTE: the saved Pipeline includes SimpleImputer so
    `joblib.load(model.pkl).predict_proba(X)` works directly on raw NaN-bearing
    arrays.  No external wrapper needed.

    MEMORY NOTE: sklearn LR's fit copies the float matrix internally (~2x
    peak memory).  For datasets > MAX_TRAIN_ROWS, we subsample at fit time
    to keep memory bounded.  LightGBM doesn't have this constraint.
    """

    ALGORITHM_NAME = "logistic_regression"
    # AUDIT FIX (2026-05-12): cap LR training to 5M rows.  With 50 features
    # at float32 that's ~1 GB matrix; sklearn LR doubles it internally to ~2 GB.
    # Well within budget for 16 GB systems with Python at idle.
    MAX_TRAIN_ROWS_LR = 3_000_000

    def _build_model(self, hyperparams: dict[str, Any]) -> Pipeline:
        params = {
            "C": hyperparams.get("C", 1.0),
            "max_iter": hyperparams.get("max_iter", 1000),
            "solver": hyperparams.get("solver", "lbfgs"),
            "random_state": hyperparams.get("random_state", 42),
        }
        # AUDIT FIX (2026-05-12):
        # Previously NaN fill was applied outside the Pipeline and attached as
        # an attribute (_nan_fill).  This left the bare LogisticRegression in
        # model.pkl crashing on NaN at production load+predict.  Embed
        # SimpleImputer inside the Pipeline so the saved artifact is self-
        # contained and Just Works.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])

    def _fit(self, model: Pipeline, X_train: Any, y_train: Any) -> None:
        # AUDIT FIX (2026-05-12): subsample huge datasets to bound LR memory.
        # sklearn LR copies the float matrix internally; with 22M+ rows the
        # peak hits 10+ GB.  Cap at MAX_TRAIN_ROWS_LR (default 3M) which is
        # plenty for converged LR coefficients on this feature scale.
        if isinstance(X_train, pl.DataFrame):
            n_rows = X_train.height
            if n_rows > self.MAX_TRAIN_ROWS_LR:
                # Stratified-ish subsample: random sample preserves base rate
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(n_rows, size=self.MAX_TRAIN_ROWS_LR, replace=False)
                idx.sort()
                X_train = X_train[idx]
                if isinstance(y_train, pl.Series):
                    y_train = y_train[idx]
                else:
                    y_train = np.asarray(y_train)[idx]
                print(f"  [LR] subsampled {n_rows:,} -> {self.MAX_TRAIN_ROWS_LR:,} rows to bound memory")
            X, _ = _to_float_array(X_train)
        else:
            X = np.asarray(X_train, float)
        y = y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)
        model.fit(X, y.astype(int))
        # Backward compat: legacy callers may still expect _nan_fill
        try:
            imputer = model.named_steps.get("imputer")
            if imputer is not None and hasattr(imputer, "statistics_"):
                model._nan_fill = imputer.statistics_
        except Exception:
            pass

    def _predict_proba(self, model: Pipeline, X: Any) -> np.ndarray:
        arr, _ = _to_float_array(X) if isinstance(X, pl.DataFrame) else (np.asarray(X, float), None)
        # Pipeline now handles NaN imputation internally
        return model.predict_proba(arr)[:, 1]
