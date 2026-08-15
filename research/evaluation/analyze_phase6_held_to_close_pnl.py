"""Phase 6 DEFINITIVE viability test — actual held-to-close PnL.

Where the previous `analyze_phase6_maker_first_viability.py` used
`markout_1s` (a 1-second toxicity proxy), THIS script uses the actual
`net_pnl_to_close` and `gross_pnl_to_close` columns directly from the
dataset.  No modeling on the PnL side, just data.

The ranking is still done via Model 03 (P(fill)) and Model 04
(predicted markout_1s) — but the bucketed outcomes are real held-to-close
PnL, after the dataset's holding policy.

Output answers definitively: "if we deploy top-K%-ranked maker quotes,
what's the actual PnL we'd have earned?"
"""
import sys, os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import polars as pl
import joblib
import yaml

from model_factory.trainers.logistic_regression_trainer import _to_float_array


TEST_DATE = "2026-05-06"
DATASET_PATH = Path("data/datasets/execution_intent_dataset_v1")
MODEL_03_PATH = Path("artifacts/model_03_fill_probability/lightgbm/model.pkl")
MODEL_04_PATH = Path("artifacts/model_04_adverse_selection/lightgbm/model.pkl")
FS_03 = Path("config/feature_sets/model_03_fill_probability_features_v1.yaml")
FS_04 = Path("config/feature_sets/model_04_adverse_selection_features_v1.yaml")


def load_fs(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))["feature_columns"]


def main():
    print("\n" + "#"*72)
    print("  PHASE 6 DEFINITIVE: HELD-TO-CLOSE PnL ANALYSIS")
    print("#"*72)

    fs03 = load_fs(FS_03)
    fs04 = load_fs(FS_04)
    print(f"  Model 03 features: {len(fs03)}, Model 04 features: {len(fs04)}")

    f = DATASET_PATH / f"{TEST_DATE}.parquet"
    schema = pl.scan_parquet(str(f)).collect_schema()
    avail = set(schema.names())
    pnl_cols = [
        "filled", "fill_price", "fill_size",
        "markout_1s", "markout_to_close",
        "gross_pnl_to_close", "net_pnl_to_close",
        "fees_paid", "rebate_earned",
        "execution_training_eligible", "markout_to_close_available",
        "order_side", "price_level", "post_only", "t_to_close_s",
        "phase_bucket", "holding_policy",
    ]
    load_cols = list(dict.fromkeys(fs03 + fs04 + pnl_cols))
    load_cols = [c for c in load_cols if c in avail]
    df = pl.read_parquet(str(f), columns=load_cols)
    df = df.filter(pl.col("execution_training_eligible"))
    n_total = df.height
    print(f"  rows after eligibility: {n_total:,}")

    # Predict P(fill) via Model 03
    print("\n--- Predictions ---")
    m03 = joblib.load(MODEL_03_PATH)
    X03 = np.nan_to_num(_to_float_array(df.select(fs03))[0], nan=0.0)
    p_fill_raw = m03.predict_proba(X03)[:, 1]
    cal_03 = getattr(m03, "_calibrator", None)
    if cal_03 is not None:
        p_fill = np.clip(cal_03.predict(p_fill_raw), 1e-6, 1 - 1e-6)
    else:
        p_fill = p_fill_raw
    print(f"  P(fill): mean={p_fill.mean():.4f}, range=[{p_fill.min():.4f}, {p_fill.max():.4f}]")

    # Predict E[markout_1s] via Model 04
    m04 = joblib.load(MODEL_04_PATH)
    X04 = np.nan_to_num(_to_float_array(df.select(fs04))[0], nan=0.0)
    expected_mko_1s = m04.predict(X04)
    print(f"  E[markout_1s]: mean={expected_mko_1s.mean():.5f}, "
          f"range=[{expected_mko_1s.min():.4f}, {expected_mko_1s.max():.4f}]")

    # Ranking by combined predicted edge
    predicted_edge = p_fill * expected_mko_1s
    print(f"  predicted edge per quote: mean={predicted_edge.mean():.6f} "
          f"std={predicted_edge.std():.6f}")

    # ACTUAL outcome arrays from data
    y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
    actual_mko_1s = df["markout_1s"].cast(pl.Float64).fill_null(0).to_numpy()
    actual_mko_close = df["markout_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
    actual_gross_pnl = df["gross_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
    actual_net_pnl = df["net_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
    fees = df["fees_paid"].cast(pl.Float64).fill_null(0).to_numpy() if "fees_paid" in df.columns else np.zeros(n_total)
    rebates = df["rebate_earned"].cast(pl.Float64).fill_null(0).to_numpy() if "rebate_earned" in df.columns else np.zeros(n_total)

    # For unfilled rows, PnL must be 0 (we never took the position)
    actual_net_pnl_per_attempt = np.where(y_filled, actual_net_pnl, 0.0)
    actual_gross_pnl_per_attempt = np.where(y_filled, actual_gross_pnl, 0.0)
    actual_mko_close_per_attempt = np.where(y_filled, actual_mko_close, 0.0)
    actual_mko_1s_per_attempt = np.where(y_filled, actual_mko_1s, 0.0)

    print(f"\n  Filled rate: {y_filled.mean():.4%}")
    print(f"  Actual realized per-quote-attempt averages:")
    print(f"    markout_1s:        {actual_mko_1s_per_attempt.mean():+.6f}")
    print(f"    markout_to_close:  {actual_mko_close_per_attempt.mean():+.6f}")
    print(f"    gross_pnl_to_close:{actual_gross_pnl_per_attempt.mean():+.6f}")
    print(f"    net_pnl_to_close:  {actual_net_pnl_per_attempt.mean():+.6f}")

    print(f"\n  Among FILLED orders only:")
    if y_filled.any():
        print(f"    n_filled: {y_filled.sum():,}")
        print(f"    mean markout_1s:        {actual_mko_1s[y_filled].mean():+.5f}")
        print(f"    mean markout_to_close:  {actual_mko_close[y_filled].mean():+.5f}")
        print(f"    mean gross_pnl_to_close:{actual_gross_pnl[y_filled].mean():+.5f}")
        print(f"    mean net_pnl_to_close:  {actual_net_pnl[y_filled].mean():+.5f}")
        print(f"    mean fees_paid: {fees[y_filled].mean():+.6f}")
        print(f"    mean rebate_earned: {rebates[y_filled].mean():+.6f}")

    # =====================================================================
    # TOP-K SELECTIVE ANALYSIS with ACTUAL PnL
    # =====================================================================
    print("\n" + "="*72)
    print("  TOP-K SELECTIVE QUOTING — ACTUAL HELD-TO-CLOSE PnL")
    print("="*72)
    print(f"  {'cut':>7s} {'n_quotes':>10s} {'P(fill)':>8s} {'actual_FR':>10s}"
          f" {'mko_1s':>10s} {'mko_close':>11s} {'gross_pnl':>11s} {'net_pnl':>11s}")
    print(f"  {'-'*7} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*11}")
    results_topk = []
    for top_pct in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00]:
        k = int(top_pct * n_total)
        if k < 100:
            continue
        idx = np.argsort(-predicted_edge)[:k]
        mko1s = float(actual_mko_1s_per_attempt[idx].mean())
        mkoclose = float(actual_mko_close_per_attempt[idx].mean())
        gross = float(actual_gross_pnl_per_attempt[idx].mean())
        net = float(actual_net_pnl_per_attempt[idx].mean())
        p_f = float(p_fill[idx].mean())
        afr = float(y_filled[idx].mean())
        print(f"  top {int(top_pct*1000)/10:>4.1f}% {k:>10,} {p_f:>8.4f} {afr:>10.4f}"
              f" {mko1s:>+10.5f} {mkoclose:>+11.5f} {gross:>+11.5f} {net:>+11.5f}")
        results_topk.append({
            "top_pct": top_pct, "n": k, "p_fill_avg": p_f, "actual_fr": afr,
            "actual_mko_1s_per_attempt": mko1s,
            "actual_mko_close_per_attempt": mkoclose,
            "actual_gross_pnl_per_attempt": gross,
            "actual_net_pnl_per_attempt": net,
        })

    # =====================================================================
    # Per-price-level held-to-close PnL
    # =====================================================================
    print("\n" + "="*72)
    print("  BREAKDOWN BY price_level (ACTUAL hold-to-close net_pnl)")
    print("="*72)
    if "price_level" in df.columns:
        plevels = df["price_level"].fill_null("unknown").to_numpy()
        print(f"  {'level':<26s} {'n':>9s} {'FR':>7s} {'mko_1s':>10s} {'mko_close':>11s} {'net_pnl':>11s}")
        for lvl in sorted(set(plevels.tolist())):
            mask = plevels == lvl
            if mask.sum() < 100:
                continue
            afr = float(y_filled[mask].mean())
            mko1s = float(actual_mko_1s_per_attempt[mask].mean())
            mkoclose = float(actual_mko_close_per_attempt[mask].mean())
            net = float(actual_net_pnl_per_attempt[mask].mean())
            print(f"  {str(lvl):<26s} {int(mask.sum()):>9,} {afr:>7.4f} "
                  f"{mko1s:>+10.5f} {mkoclose:>+11.5f} {net:>+11.5f}")

    # =====================================================================
    # Per-t_to_close held-to-close PnL
    # =====================================================================
    print("\n" + "="*72)
    print("  BREAKDOWN BY t_to_close (ACTUAL hold-to-close net_pnl)")
    print("="*72)
    if "t_to_close_s" in df.columns:
        ttc = df["t_to_close_s"].fill_null(-1).cast(pl.Float64).to_numpy()
        print(f"  {'bucket':<10s} {'n':>9s} {'FR':>7s} {'mko_1s':>10s} {'mko_close':>11s} {'net_pnl':>11s}")
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 9e9)]:
            mask = (ttc >= lo) & (ttc < hi)
            if mask.sum() < 100:
                continue
            afr = float(y_filled[mask].mean())
            mko1s = float(actual_mko_1s_per_attempt[mask].mean())
            mkoclose = float(actual_mko_close_per_attempt[mask].mean())
            net = float(actual_net_pnl_per_attempt[mask].mean())
            label = f"{lo}_{int(hi) if hi < 9e8 else 'inf'}s"
            print(f"  {label:<10s} {int(mask.sum()):>9,} {afr:>7.4f} "
                  f"{mko1s:>+10.5f} {mkoclose:>+11.5f} {net:>+11.5f}")

    # =====================================================================
    # Total PnL volume in top-K (multiplied by avg order size)
    # =====================================================================
    print("\n" + "="*72)
    print("  TOTAL VOLUME-WEIGHTED PnL IN TOP-K (sum of net_pnl, not avg)")
    print("="*72)
    print(f"  {'cut':>7s} {'n_quotes':>10s} {'sum_net_pnl':>15s} "
          f"{'sum_fees':>10s} {'sum_rebates':>12s} {'fills':>8s}")
    for top_pct in [0.01, 0.05, 0.10, 0.25, 1.00]:
        k = int(top_pct * n_total)
        idx = np.argsort(-predicted_edge)[:k]
        sum_pnl = float(actual_net_pnl_per_attempt[idx].sum())
        sum_fees = float(fees[idx][y_filled[idx]].sum())
        sum_rebates = float(rebates[idx][y_filled[idx]].sum())
        n_fills = int(y_filled[idx].sum())
        print(f"  top {int(top_pct*100):>3d}% {k:>10,} {sum_pnl:>+15.4f} "
              f"{sum_fees:>10.2f} {sum_rebates:>+12.4f} {n_fills:>8,}")

    # =====================================================================
    # Final verdict
    # =====================================================================
    print("\n" + "="*72)
    print("  DEFINITIVE VERDICT (held-to-close PnL)")
    print("="*72)
    top1_idx = np.argsort(-predicted_edge)[:int(0.01 * n_total)]
    top5_idx = np.argsort(-predicted_edge)[:int(0.05 * n_total)]
    top10_idx = np.argsort(-predicted_edge)[:int(0.10 * n_total)]
    top25_idx = np.argsort(-predicted_edge)[:int(0.25 * n_total)]

    top1_net = float(actual_net_pnl_per_attempt[top1_idx].mean())
    top5_net = float(actual_net_pnl_per_attempt[top5_idx].mean())
    top10_net = float(actual_net_pnl_per_attempt[top10_idx].mean())
    top25_net = float(actual_net_pnl_per_attempt[top25_idx].mean())
    all_net = float(actual_net_pnl_per_attempt.mean())

    print(f"\n  ACTUAL net_pnl_to_close PER QUOTE ATTEMPT:")
    print(f"    top  1%:  {top1_net:+.6f}")
    print(f"    top  5%:  {top5_net:+.6f}")
    print(f"    top 10%:  {top10_net:+.6f}")
    print(f"    top 25%:  {top25_net:+.6f}")
    print(f"    all 100%: {all_net:+.6f}")

    print()
    if top10_net > 0.001:
        print("  VERDICT: MAKER-FIRST IS PROFITABLE (held-to-close PnL)")
        print("           Top decile has positive realized net PnL.")
    elif top5_net > 0.001:
        print("  VERDICT: MAKER-FIRST IS PROFITABLE BUT TIGHT")
        print("           Only top 5% has positive realized PnL.")
    elif top1_net > 0.001:
        print("  VERDICT: MAKER-FIRST IS MARGINALLY VIABLE")
        print("           Only the top 1% has positive PnL.  Tight strategy.")
    elif all_net > -0.0001:
        print("  VERDICT: BREAKEVEN — no clear PnL edge on held-to-close.")
        print("           1s markout edge does NOT translate to held PnL.")
    else:
        print("  VERDICT: MAKER-FIRST NOT PROFITABLE on held-to-close")
        print("           1s markout was misleading — adverse selection")
        print("           dominates when held to resolution.")

    # Save full results
    out = {
        "test_date": TEST_DATE,
        "n_rows": int(n_total),
        "n_filled": int(y_filled.sum()),
        "actual_fill_rate": float(y_filled.mean()),
        "overall_actual_net_pnl_per_attempt": all_net,
        "topk_results": results_topk,
        "verdict_top1_net": top1_net,
        "verdict_top5_net": top5_net,
        "verdict_top10_net": top10_net,
        "verdict_top25_net": top25_net,
    }
    out_path = Path("artifacts/phase6_held_to_close_pnl_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
