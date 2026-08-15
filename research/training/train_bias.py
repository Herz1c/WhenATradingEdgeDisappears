"""Run Model 1 Bias training: LR baseline + LightGBM on all three dataset variants."""
import sys
import json

sys.path.insert(0, "src")

from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer
from model_factory.trainers.model_specific.model_01_bias_trainer import Model01BiasTrainer


class LRBiasTrainer(LogisticRegressionTrainer):
    MODEL_ID = "model_01_bias"
    DEFAULT_HYPERPARAMS = {"C": 0.1, "max_iter": 500}
    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)


VARIANTS = ["preopen", "first15s", "first30s"]


def summarise(variant, algo, metrics):
    print(f"\n{'='*60}")
    print(f"  {algo.upper()} | variant={variant}")
    print(f"{'='*60}")
    if "_calibration_meta" in metrics:
        cm = metrics["_calibration_meta"]
        print(f"  calibration: {cm}")
    for split, m in metrics.items():
        if split.startswith("_"):
            continue
        if not isinstance(m, dict) or "log_loss" not in m:
            continue
        print(f"  [{split}]  log_loss={m['log_loss']:.5f}  brier={m['brier_score']:.5f}  ece={m.get('ece',0):.4f}  n={m.get('n_rows','?')}")
        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_log_loss_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "LOSES"
                print(f"         vs {bl}: baseline_ll={bm['log_loss']:.5f}  delta={delta:+.5f}  [{sign}]")


if __name__ == "__main__":
    all_results = {}

    for variant in VARIANTS:
        print(f"\n\n{'#'*60}")
        print(f"  VARIANT: {variant}")
        print(f"{'#'*60}")

        # LR baseline
        lr = LRBiasTrainer()
        lr_metrics = lr.train(dataset_variant=variant)
        summarise(variant, "logistic_regression", lr_metrics)
        all_results[f"{variant}_lr"] = lr_metrics

        # LightGBM
        lgbm = Model01BiasTrainer()
        lgbm_metrics = lgbm.train(dataset_variant=variant)
        summarise(variant, "lightgbm", lgbm_metrics)
        all_results[f"{variant}_lgbm"] = lgbm_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2))
