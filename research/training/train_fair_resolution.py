"""Model 2 Fair-Resolution: LR baseline + LightGBM on coarse and dense_close."""
import sys, json
sys.path.insert(0, "src")

from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer
from model_factory.trainers.model_specific.model_02_fair_resolution_trainer import Model02FairResolutionTrainer


class LRFairResolutionTrainer(LogisticRegressionTrainer):
    MODEL_ID = "model_02_fair_resolution"
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
        ece = m.get("ece", 0)
        n = m.get("n_rows", 0) or 0
        try:
            n_str = f"{int(n):,}"
        except (TypeError, ValueError):
            n_str = str(n)
        print(f"  [{split:5s}] log_loss={ll:.5f}  brier={bs:.5f}  ece={ece:.4f}  n={n_str}")
        if "baselines" in m:
            for bl, bm in m["baselines"].items():
                delta = m.get(f"vs_{bl}_log_loss_delta", float("nan"))
                sign = "BEATS" if delta < 0 else "loses"
                print(f"           vs {bl:<30s}: ll={bm['log_loss']:.5f}  delta={delta:+.5f}  [{sign}]")
        if "edge_by_time_to_close" in m:
            print(f"  [t_to_close breakdown]")
            for bucket, bm in m["edge_by_time_to_close"].items():
                if bm.get("rows", 0) > 0:
                    print(f"    t=[{bucket:>7s}s]  rows={bm['rows']:>7,}  ll={bm.get('log_loss', float('nan')):.5f}")
        if "log_loss_by_phase" in m:
            print(f"  [by phase]  {m['log_loss_by_phase']}")
        if "brier_by_session" in m:
            print(f"  [by session]  {m['brier_by_session']}")


if __name__ == "__main__":
    all_results = {}

    # -- COARSE ----------------------------------------------------------
    print("\n\n" + "#"*70)
    print("  VARIANT: coarse  (1s snapshots, open -> T-60s)")
    print("#"*70)

    lr = LRFairResolutionTrainer()
    lr_metrics = lr.train(dataset_variant="coarse")
    print_summary("coarse", "logistic_regression", lr_metrics)
    all_results["coarse_lr"] = lr_metrics

    lgbm = Model02FairResolutionTrainer()
    lgbm_metrics = lgbm.train(dataset_variant="coarse")
    print_summary("coarse", "lightgbm", lgbm_metrics)
    all_results["coarse_lgbm"] = lgbm_metrics

    # -- DENSE CLOSE -----------------------------------------------------
    print("\n\n" + "#"*70)
    print("  VARIANT: dense_close  (250ms/100ms, T-60s -> close)")
    print("#"*70)

    lgbm_dc = Model02FairResolutionTrainer()
    lgbm_dc_metrics = lgbm_dc.train(dataset_variant="dense_close")
    print_summary("dense_close", "lightgbm", lgbm_dc_metrics)
    all_results["dense_close_lgbm"] = lgbm_dc_metrics

    print("\n\n=== FULL RESULTS JSON ===")
    print(json.dumps(all_results, indent=2))
