"""Model 5 Closing Flip: LR baseline + LightGBM on dense_close."""
import sys, json
sys.path.insert(0, "src")

from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer
from model_factory.trainers.model_specific.model_05_closing_flip_trainer import Model05ClosingFlipTrainer


class LRClosingFlipTrainer(LogisticRegressionTrainer):
    MODEL_ID = "model_05_closing_flip"
    DEFAULT_HYPERPARAMS = {"C": 0.1, "max_iter": 500}
    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)


def print_summary(variant, algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} | variant={variant}")
    print(f"{'='*70}")
    for split, m in metrics.items():
        if not isinstance(m, dict):
            continue
        ll = m.get("log_loss", float("nan"))
        bs = m.get("brier_score", float("nan"))
        ece = m.get("ece", float("nan"))
        n = m.get("n_rows", 0) or 0
        try:
            n_str = f"{int(n):,}"
        except (TypeError, ValueError):
            n_str = str(n)
        flip_rate = m.get("target_mean", float("nan"))
        print(f"  [{split:5s}] log_loss={ll:.5f}  brier={bs:.5f}  ece={ece:.4f}  n={n_str}  flip_rate={flip_rate:.4f}")
        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_log_loss_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "loses"
                print(f"           vs {bl:<30s}: ll={bm['log_loss']:.5f}  delta={delta:+.5f}  [{sign}]")
        if "flip_precision_recall" in m:
            pr = m["flip_precision_recall"]
            prec = pr.get("precision", float("nan"))
            rec = pr.get("recall", float("nan"))
            print(f"  [flip p/r at 0.5]  precision={prec:.4f}  recall={rec:.4f}")
        if "flip_rate_by_delta_bucket" in m:
            print(f"  [flip rate by |delta| bucket]")
            for bucket, rate in m["flip_rate_by_delta_bucket"].items():
                if rate is not None:
                    print(f"    |delta|=[{bucket:>8s}$]  flip_rate={rate:.4f}")
        if "edge_by_time_to_close" in m:
            print(f"  [by t_to_close]")
            for bucket, bm in m["edge_by_time_to_close"].items():
                if bm.get("rows", 0) > 0:
                    print(f"    t=[{bucket:>7s}s]  rows={bm['rows']:>7,}  ll={bm.get('log_loss', float('nan')):.5f}")
        if "brier_by_session" in m:
            print(f"  [by session]  {m['brier_by_session']}")


if __name__ == "__main__":
    all_results = {}

    print("\n\n" + "#"*70)
    print("  MODEL 5 CLOSING FLIP | variant=dense_close")
    print("#"*70)

    lr = LRClosingFlipTrainer()
    lr_metrics = lr.train(dataset_variant="dense_close")
    print_summary("dense_close", "logistic_regression", lr_metrics)
    all_results["dense_close_lr"] = lr_metrics

    lgbm = Model05ClosingFlipTrainer()
    lgbm_metrics = lgbm.train(dataset_variant="dense_close")
    print_summary("dense_close", "lightgbm", lgbm_metrics)
    all_results["dense_close_lgbm"] = lgbm_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2))
