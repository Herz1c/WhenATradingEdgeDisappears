from __future__ import annotations

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer


class Model09RegimeEncoderTrainer(LightGBMTrainer):
    """Trainer shell for Model 9: BTC regime encoder."""

    MODEL_ID = "model_09_regime_encoder"

    def __init__(self) -> None:
        super().__init__(model_id=self.MODEL_ID)

