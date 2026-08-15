from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python3 src/model_factory/trainers/model_specific/model_02_fair_resolution_trainer.py`
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from model_factory.trainers.lightgbm_trainer import LightGBMTrainer


class Model02FairResolutionTrainer(LightGBMTrainer):
    """Trainer for Model 2: fair-resolution model."""

    MODEL_ID = "model_02_fair_resolution"
    # Larger capacity than Model 1 — 683K train rows warrant more leaves and trees.
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


def _print_summary(variant: str, metrics: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  Model 02 dense_close (LightGBM)  variant={variant}")
    print(f"{'='*70}")
    for split_name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        ll = m.get("log_loss", float("nan"))
        bs = m.get("brier_score", float("nan"))
        ece = m.get("ece", 0)
        n = m.get("n_rows", 0) or 0
        try:
            n_str = f"{int(n):,}"
        except (TypeError, ValueError):
            n_str = str(n)
        print(f"  [{split_name:5s}] log_loss={ll:.5f}  brier={bs:.5f}  ece={ece:.4f}  n={n_str}")
    cal_meta = metrics.get("_calibration_meta")
    if cal_meta:
        print(f"  calibration: {cal_meta}")


if __name__ == "__main__":
    print("Training Model 02 fair_resolution (dense_close LightGBM).")
    print("Split: train Apr 19 - May 9 | val May 10 - May 14 | test May 15 - May 20")
    trainer = Model02FairResolutionTrainer()
    metrics = trainer.train(dataset_variant="dense_close")
    _print_summary("dense_close", metrics)
    print("\n=== FULL METRICS JSON ===")
    print(json.dumps(metrics, indent=2, default=str))

