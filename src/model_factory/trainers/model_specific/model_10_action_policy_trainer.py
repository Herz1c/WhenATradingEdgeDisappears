from __future__ import annotations

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer


class Model10ActionPolicyTrainer(LightGBMTrainer):
    """Trainer shell for Model 10: action / execution policy."""

    MODEL_ID = "model_10_action_policy"

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)

