from __future__ import annotations

from model_factory.trainers.lightgbm_regression_trainer import LightGBMRegressionTrainer


class Model06MispricingTrainer(LightGBMRegressionTrainer):
    """Trainer for Model 6: mispricing — predicts hold_to_close_edge_vs_mid."""

    MODEL_ID = "model_06_mispricing"
    DEFAULT_HYPERPARAMS = {
        "num_leaves": 63,
        "min_child_samples": 200,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)
