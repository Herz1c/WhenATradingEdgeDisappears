"""Phase 6 Strategic Analysis: Is Maker-First Viable?

Combines Model 03 (fill probability) and Model 04 (markout) to compute:

    expected_edge_per_quote = P(fill | quote) * E[markout | fill, quote] - fees

For every row in the test split:
  1. Get P(fill) from Model 03
  2. Get E[markout_1s | fill] from Model 04
  3. Compute net expected edge after fees/rebates
  4. Aggregate by strategic regimes (post_only, price_level, t_to_close)

Strategic verdict:
  - If average expected_edge across all maker scenarios > 0 after fees: VIABLE
  - If selective scenarios > 0: CONDITIONAL (need to identify which subset)
  - If all scenarios <= 0: NOT VIABLE (pivot strategy required)
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


# Fee schedule assumptions (Polymarket maker rebate / taker fee schedule)
# Conservative defaults — adjust based on actual schedule
MAKER_REBATE_BPS = 0.0    # current Polymarket: no maker rebate
MAKER_FEE_BPS    = 0.0    # current Polymarket: no maker fee on UMA-style
TAKER_FEE_BPS    = 2.0    # 2 bps approx for protocol fee


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

MODEL_03_PATH = Path("artifacts/model_03_fill_probability/lightgbm/model.pkl")
MODEL_04_PATH = Path("artifacts/model_04_adverse_selection/lightgbm/model.pkl")

# Test parquet for May 6
TEST_DATE = "2026-05-06"
DATASET_PATH = Path("data/datasets/execution_intent_dataset_v1")

# Feature sets
FS_03_PATH = Path("config/feature_sets/model_03_fill_probability_features_v1.yaml")
FS_04_PATH = Path("config/feature_sets/model_04_adverse_selection_features_v1.yaml")


def load_model(p):
    if not p.exists():
        # Try alternate path with no `default/` segment
        alt = p.parent.parent / p.name
        if alt.exists():
            return joblib.load(alt)
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)


def load_feature_set(p):
    fs = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
    return fs["feature_columns"]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#"*72)
    print("  PHASE 6 STRATEGIC ANALYSIS: MAKER-FIRST VIABILITY")
    print("#"*72)

    # Load models
    print("\n--- Loading models ---")
    m03 = load_model(MODEL_03_PATH)
    m04 = load_model(MODEL_04_PATH)
    print(f"  Model 03 type: {type(m03).__name__}")
    print(f"  Model 04 type: {type(m04).__name__}")

    fs03 = load_feature_set(FS_03_PATH)
    fs04 = load_feature_set(FS_04_PATH)

    # Load test data with full schema (we need the joint dataset)
    print(f"\n--- Loading test data ({TEST_DATE}) ---")
    f = DATASET_PATH / f"{TEST_DATE}.parquet"
    schema = pl.scan_parquet(str(f)).collect_schema()
    avail = set(schema.names())
    load_cols = list(dict.fromkeys(
        fs03 + fs04 + [
            "filled", "markout_1s", "markout_5s",
            "execution_training_eligible", "order_side", "price_level",
            "post_only", "t_to_close_s", "phase_bucket",
            "fees_paid", "rebate_earned",
        ]
    ))
    load_cols = [c for c in load_cols if c in avail]
    df = pl.read_parquet(str(f), columns=load_cols)
    df = df.filter(pl.col("execution_training_eligible"))
    print(f"  rows after eligibility: {df.height:,}")

    # Predict
    print("\n--- Generating predictions ---")
    X03_df = df.select(fs03)
    X03, _ = _to_float_array(X03_df)
    p_fill = m03.predict_proba(np.nan_to_num(X03, nan=0.0))[:, 1]
    # Apply Model 03 calibrator if present
    cal_03 = getattr(m03, "_calibrator", None)
    if cal_03 is not None:
        p_fill = np.clip(cal_03.predict(p_fill), 1e-6, 1 - 1e-6)
    print(f"  P(fill) range: [{p_fill.min():.4f}, {p_fill.max():.4f}]  mean={p_fill.mean():.4f}")

    X04_df = df.select(fs04)
    X04, _ = _to_float_array(X04_df)
    expected_markout = m04.predict(np.nan_to_num(X04, nan=0.0))
    print(f"  E[markout] range: [{expected_markout.min():.5f}, {expected_markout.max():.5f}]  "
          f"mean={expected_markout.mean():.5f}")

    # Expected edge per quote = P(fill) * E[markout | fill] - taker_fee_if_filled
    # For maker quotes, fees are typically 0 at Polymarket; rebate is also 0
    # The edge is purely from markout
    expected_edge = p_fill * expected_markout
    print(f"  Expected edge per quote: mean={expected_edge.mean():.6f}  "
          f"std={expected_edge.std():.6f}")

    # Actual outcomes for verification
    y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
    y_markout = df["markout_1s"].cast(pl.Float64).to_numpy()
    fees = df["fees_paid"].cast(pl.Float64).fill_null(0).to_numpy() if "fees_paid" in df.columns else np.zeros(df.height)
    rebates = df["rebate_earned"].cast(pl.Float64).fill_null(0).to_numpy() if "rebate_earned" in df.columns else np.zeros(df.height)

    actual_realized_markout = np.where(y_filled, np.nan_to_num(y_markout, nan=0.0), 0.0)
    actual_realized_net = actual_realized_markout - fees + rebates

    print(f"\n  Actual fill rate: {y_filled.mean():.4%}")
    print(f"  Actual realized markout (per-quote-attempt, 0 if no fill): mean={actual_realized_markout.mean():+.6f}")
    print(f"  Actual realized net (after fees/rebate): mean={actual_realized_net.mean():+.6f}")

    # --------------------------------------------------------------
    # Stratified analysis
    # --------------------------------------------------------------

    def report_group(name, mask, label):
        if mask.sum() < 100:
            return
        n = int(mask.sum())
        p_f = float(p_fill[mask].mean())
        e_m = float(expected_markout[mask].mean())
        e_edge = float(expected_edge[mask].mean())
        actual_fr = float(y_filled[mask].mean())
        # Actual mean markout over filled rows only
        sub_filled = mask & y_filled
        if sub_filled.sum() > 0:
            a_mko = float(np.nan_to_num(y_markout[sub_filled], nan=0.0).mean())
        else:
            a_mko = float("nan")
        # Actual realized edge per quote attempt
        a_realized = float(actual_realized_net[mask].mean())
        print(f"    {label:30s}  n={n:>7,}  "
              f"P(fill)={p_f:.3f}  E[mko]={e_m:+.5f}  "
              f"E[edge]={e_edge:+.6f}  ||  "
              f"actual_FR={actual_fr:.3f}  "
              f"actual_mko_when_filled={a_mko:+.5f}  "
              f"actual_net_per_attempt={a_realized:+.6f}")

    print("\n" + "="*72)
    print("  BREAKDOWN BY post_only")
    print("="*72)
    for po_val, lbl in [(True, "post_only=TRUE"), (False, "post_only=FALSE")]:
        if "post_only" in df.columns:
            mask = df["post_only"].fill_null(False).to_numpy().astype(bool) == po_val
            report_group("post_only", mask, lbl)

    print("\n" + "="*72)
    print("  BREAKDOWN BY price_level")
    print("="*72)
    if "price_level" in df.columns:
        levels = df["price_level"].fill_null("unknown").to_numpy()
        for lvl in np.unique(levels):
            mask = levels == lvl
            report_group("price_level", mask, f"price_level={lvl}")

    print("\n" + "="*72)
    print("  BREAKDOWN BY t_to_close BUCKET")
    print("="*72)
    if "t_to_close_s" in df.columns:
        ttc = df["t_to_close_s"].fill_null(-1).cast(pl.Float64).to_numpy()
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 9e9)]:
            mask = (ttc >= lo) & (ttc < hi)
            report_group("t_to_close", mask, f"{lo}_{hi if hi < 9e8 else 'inf'}s")

    # --------------------------------------------------------------
    # Top-K analysis: if we only acted on the highest-expected-edge quotes,
    # what would the average realized edge be?
    # --------------------------------------------------------------
    print("\n" + "="*72)
    print("  TOP-K SELECTIVE QUOTING: only act on top-K% predicted edge")
    print("="*72)
    print(f"  {'cut':>6s}  {'n_quotes':>9s}  {'P(fill)_avg':>11s}  "
          f"{'E[mko]':>10s}  {'E[edge]':>10s}  "
          f"{'actual_FR':>10s}  {'actual_net/attempt':>20s}")
    for top_pct in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        k = int(top_pct * len(expected_edge))
        if k < 100:
            continue
        idx = np.argsort(-expected_edge)[:k]
        sel_mask = np.zeros(len(expected_edge), dtype=bool)
        sel_mask[idx] = True
        n = int(sel_mask.sum())
        p_f = float(p_fill[sel_mask].mean())
        e_m = float(expected_markout[sel_mask].mean())
        e_edge = float(expected_edge[sel_mask].mean())
        a_fr = float(y_filled[sel_mask].mean())
        a_realized = float(actual_realized_net[sel_mask].mean())
        print(f"  top {int(top_pct*100):>3d}%  "
              f"{n:>9,}  {p_f:>11.4f}  {e_m:>+10.5f}  "
              f"{e_edge:>+10.6f}  {a_fr:>10.4f}  {a_realized:>+20.6f}")

    # --------------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------------
    print("\n" + "="*72)
    print("  STRATEGIC VERDICT")
    print("="*72)
    overall_actual_realized = float(actual_realized_net.mean())
    overall_expected = float(expected_edge.mean())
    # Top decile
    top10 = int(0.10 * len(expected_edge))
    top10_idx = np.argsort(-expected_edge)[:top10]
    top10_realized = float(actual_realized_net[top10_idx].mean())

    print(f"\n  Overall E[edge per quote attempt]:   {overall_expected:+.6f}")
    print(f"  Overall ACTUAL realized per attempt:  {overall_actual_realized:+.6f}")
    print(f"\n  Top 10% selective realized:           {top10_realized:+.6f}")
    print(f"  Top 5% selective realized:            {float(actual_realized_net[np.argsort(-expected_edge)[:int(0.05 * len(expected_edge))]].mean()):+.6f}")
    print(f"  Top 1% selective realized:            {float(actual_realized_net[np.argsort(-expected_edge)[:int(0.01 * len(expected_edge))]].mean()):+.6f}")

    print()
    if top10_realized > 0.001:
        print("  VERDICT: MAKER-FIRST IS CONDITIONALLY VIABLE")
        print("           Top decile of model-ranked quotes shows positive net edge.")
        print("           Strategy: only quote when expected edge is in top 10-25%.")
    elif overall_actual_realized > 0.0001:
        print("  VERDICT: MAKER-FIRST MARGINALLY POSITIVE")
        print("           Bulk quoting shows tiny positive edge; size carefully.")
    elif overall_actual_realized > -0.0001:
        print("  VERDICT: MAKER-FIRST NEUTRAL (essentially zero edge)")
        print("           Strategy pivot needed: hybrid maker/taker or pure taker.")
    else:
        print("  VERDICT: MAKER-FIRST IS NOT VIABLE")
        print("           Net edge is negative even on best-ranked subset.")
        print("           Strategy pivot required.")

    # Save full results
    out_path = Path("artifacts/phase6_maker_first_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "n_test_rows": int(df.height),
        "overall_p_fill_mean": float(p_fill.mean()),
        "overall_expected_markout_mean": float(expected_markout.mean()),
        "overall_expected_edge_mean": overall_expected,
        "overall_actual_realized_net_mean": overall_actual_realized,
        "top10pct_realized": top10_realized,
        "actual_fill_rate": float(y_filled.mean()),
    }
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
