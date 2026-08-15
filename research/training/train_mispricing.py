"""Model 6 Mispricing: Ridge regression baseline + LightGBM on dense_close."""
import sys, json
sys.path.insert(0, "src")

from model_factory.trainers.linear_regression_trainer import LinearRegressionTrainer
from model_factory.trainers.model_specific.model_06_mispricing_trainer import Model06MispricingTrainer


class LRMispricingTrainer(LinearRegressionTrainer):
    MODEL_ID = "model_06_mispricing"
    DEFAULT_HYPERPARAMS = {"alpha": 1.0}
    def __init__(self):
        super().__init__(model_id=self.MODEL_ID)


def print_summary(algo, metrics):
    print(f"\n{'='*70}")
    print(f"  {algo.upper()} | variant=dense_close")
    print(f"{'='*70}")
    for split, m in metrics.items():
        mae = m["mae"]
        rmse = m["rmse"]
        sa = m.get("sign_accuracy", float("nan"))
        n = m.get("n_rows", "?")
        tmean = m.get("target_mean", float("nan"))
        print(f"  [{split:5s}] mae={mae:.5f}  rmse={rmse:.5f}  sign_acc={sa:.3f}  n={n:,}  target_mean={tmean:.4f}")
        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_mae_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "loses"
                print(f"           vs {bl:<25s}: mae={bm['mae']:.5f}  delta={delta:+.5f}  [{sign}]")
        if "gate" in m:
            g = m["gate"]
            passed = g.get("gate_pass", False)
            status = "PASS" if passed else "FAIL"
            print(f"           gate [{status}]  long_edge={g.get('mean_actual_edge_when_long', float('nan')):.4f}  "
                  f"short_edge={g.get('mean_actual_edge_when_short', float('nan')):.4f}")
        if "mispricing_by_time_to_close" in m:
            print(f"  [t_to_close breakdown]")
            for bucket, bm in m["mispricing_by_time_to_close"].items():
                if bm.get("rows", 0) > 0:
                    print(f"    t=[{bucket:>7s}s]  rows={bm['rows']:>7,}  mae={bm.get('mae', float('nan')):.5f}  "
                          f"sign_acc={bm.get('sign_accuracy', float('nan')):.3f}  "
                          f"mean_edge={bm.get('mean_actual_edge', float('nan')):+.4f}")
        if "sign_accuracy_by_phase" in m:
            print(f"  [sign acc by phase]  {m['sign_accuracy_by_phase']}")
        if "mae_by_session" in m:
            print(f"  [mae by session]  {m['mae_by_session']}")


if __name__ == "__main__":
    all_results = {}

    print("\n\n" + "#"*70)
    print("  MODEL 6 MISPRICING | variant=dense_close")
    print("#"*70)

    print("\n-- Ridge Regression baseline --")
    lr = LRMispricingTrainer()
    lr_metrics = lr.train(dataset_variant="dense_close")
    print_summary("ridge_regression", lr_metrics)
    all_results["dense_close_lr"] = lr_metrics

    print("\n-- LightGBM --")
    lgbm = Model06MispricingTrainer()
    lgbm_metrics = lgbm.train(dataset_variant="dense_close")
    print_summary("lightgbm", lgbm_metrics)
    all_results["dense_close_lgbm"] = lgbm_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2))
