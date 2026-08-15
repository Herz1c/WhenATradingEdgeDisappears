from __future__ import annotations

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer


class Model01BiasTrainer(LightGBMTrainer):
    """Trainer for Model 1: 5-minute interval bias model."""

    MODEL_ID = "model_01_bias"
    DEFAULT_HYPERPARAMS = {"num_leaves": 8, "min_child_samples": 50, "learning_rate": 0.05, "n_estimators": 200}

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)

