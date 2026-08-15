from __future__ import annotations

from .logistic_regression_trainer import LogisticRegressionTrainer


class CatBoostTrainer(LogisticRegressionTrainer):
    """CatBoost training harness shell; concrete training is intentionally deferred."""

