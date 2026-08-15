"""Phase 6 — Small-money sizing analysis.

Given target $45/day ($1K/month), how many top-edge quotes per day do we need
to hit it on out-of-sample days?  Uses existing models (Model 03 + 4b).
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


DATASET_PATH = Path("data/datasets/execution_intent_dataset_v1")
MODEL_03_PATH = Path("artifacts/model_03_fill_probability/lightgbm/model.pkl")
MODEL_04B_PATH = Path("artifacts/model_04b_markout_to_close/lightgbm/model.pkl")
FS_03 = "config/feature_sets/model_03_fill_probability_features_v1.yaml"
FS_04 = "config/feature_sets/model_04_adverse_selection_features_v1.yaml"

OOS_DAYS = ["2026-05-05", "2026-05-06"]
DAILY_TARGET_USD = 45  # $1K/month


def load_fs(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))["feature_columns"]


def main():
    print("\n" + "#"*72)
    print("  PHASE 6 SIZING ANALYSIS  |  Target: $45/day ($1K/month)")
    print("#"*72)

    fs03 = load_fs(FS_03)
    fs04 = load_fs(FS_04)
    m03 = joblib.load(MODEL_03_PATH)
    m04b = joblib.load(MODEL_04B_PATH)

    summary = []

    for date_str in OOS_DAYS:
        f = DATASET_PATH / f"{date_str}.parquet"
        if not f.exists():
            print(f"\n  {date_str}: SKIP (no file)")
            continue

        schema = pl.scan_parquet(str(f)).collect_schema()
        avail = set(schema.names())
        # Dedupe
        cols = list(dict.fromkeys(
            fs03 + fs04 + [
                "filled", "fill_size", "fill_price", "fill_notional",
                "net_pnl_to_close", "execution_training_eligible",
                "price_level", "order_side", "t_to_close_s",
            ]
        ))
        cols = [c for c in cols if c in avail]
        df = pl.read_parquet(str(f), columns=cols)
        df = df.filter(pl.col("execution_training_eligible"))
        n_total = df.height

        # Predict
        X03 = np.nan_to_num(_to_float_array(df.select(fs03))[0], nan=0.0)
        p_fill_raw = m03.predict_proba(X03)[:, 1]
        cal_03 = getattr(m03, "_calibrator", None)
        p_fill = np.clip(cal_03.predict(p_fill_raw), 1e-6, 1 - 1e-6) if cal_03 else p_fill_raw
        X04 = np.nan_to_num(_to_float_array(df.select(fs04))[0], nan=0.0)
        pred_mko = m04b.predict(X04)
        predicted_edge = p_fill * pred_mko

        y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
        actual_net = df["net_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
        actual_notional = df["fill_notional"].cast(pl.Float64).fill_null(0).to_numpy() if "fill_notional" in df.columns else np.zeros(n_total)
        per_att = np.where(y_filled, actual_net, 0.0)

        # Rank descending
        sorted_idx = np.argsort(-predicted_edge)
        sorted_pnl = per_att[sorted_idx]
        sorted_filled = y_filled[sorted_idx]
        sorted_notional = actual_notional[sorted_idx]
        cum_pnl = np.cumsum(sorted_pnl)
        cum_fills = np.cumsum(sorted_filled.astype(int))
        cum_notional = np.cumsum(sorted_notional)

        crossed = np.where(cum_pnl >= DAILY_TARGET_USD)[0]
        if len(crossed) == 0:
            print(f"\n  {date_str}:  ERROR — cannot reach ${DAILY_TARGET_USD} target on this day")
            print(f"    max cum PnL = ${cum_pnl[-1]:+,.2f}")
            continue

        n_quotes = int(crossed[0] + 1)
        n_fills = int(cum_fills[crossed[0]])
        traded_notional = float(cum_notional[crossed[0]])

        # Stats of these top-N quotes
        top_n_idx = sorted_idx[:n_quotes]
        top_n_p_fill = float(p_fill[top_n_idx].mean())
        top_n_pred_mko = float(pred_mko[top_n_idx].mean())
        top_n_actual_fill_rate = float(y_filled[top_n_idx].mean())
        avg_pnl_per_attempt = DAILY_TARGET_USD / n_quotes
        avg_pnl_per_fill = DAILY_TARGET_USD / max(n_fills, 1)

        print(f"\n  === {date_str} ===")
        print(f"  Total opportunities on day:     {n_total:,}")
        print(f"  Quotes needed for ${DAILY_TARGET_USD}/day: {n_quotes:,}  ({100*n_quotes/n_total:.4f}% of day's opportunities)")
        print(f"  Actually filled at this depth:  {n_fills:,}  ({100*top_n_actual_fill_rate:.1f}% fill rate on selected)")
        print(f"  Avg P(fill) of selected:        {top_n_p_fill:.3f}")
        print(f"  Avg pred markout of selected:   ${top_n_pred_mko:.4f}")
        print(f"  Notional traded:                ${traded_notional:,.2f}")
        print(f"  Avg PnL per quote attempt:      ${avg_pnl_per_attempt:.5f}")
        print(f"  Avg PnL per fill:               ${avg_pnl_per_fill:.5f}")
        print(f"  ROI on notional traded:         {100*DAILY_TARGET_USD/max(traded_notional,1):.3f}%")

        summary.append({
            "date": date_str,
            "n_total_opportunities": n_total,
            "n_quotes_for_target": n_quotes,
            "n_fills_for_target": n_fills,
            "notional_traded": traded_notional,
            "p_fill_avg_selected": top_n_p_fill,
            "actual_fill_rate_selected": top_n_actual_fill_rate,
            "avg_pnl_per_attempt": avg_pnl_per_attempt,
            "avg_pnl_per_fill": avg_pnl_per_fill,
            "roi_on_notional": DAILY_TARGET_USD / max(traded_notional, 1),
        })

    # =====================================================================
    # Capital required analysis
    # =====================================================================
    if summary:
        print("\n" + "="*72)
        print("  CAPITAL REQUIREMENT ANALYSIS")
        print("="*72)
        print(f"\n  Polymarket BTC 5m markets typically resolve in <=300 seconds.")
        print(f"  We assume positions held until market close (worst-case capital lock).")
        print()
        avg_n_quotes = np.mean([s["n_quotes_for_target"] for s in summary])
        avg_n_fills = np.mean([s["n_fills_for_target"] for s in summary])
        avg_notional = np.mean([s["notional_traded"] for s in summary])

        # Estimate max simultaneous open positions (worst case)
        # If a typical market is 5 minutes and we trade evenly throughout the day,
        # peak concurrent = (n_fills / 24 hours) × 5 minutes
        peak_concurrent_estimate = avg_n_fills / (24 * 60 / 5)  # fills per 5-min slot
        capital_5min_concurrent = peak_concurrent_estimate * 5.10  # avg $5.10 notional
        # If we cap concurrent at 20 (sensible inventory limit)
        capital_capped_20 = 20 * 5.10

        print(f"  Avg quotes/day needed:         {avg_n_quotes:>10,.0f}")
        print(f"  Avg fills/day:                 {avg_n_fills:>10,.0f}")
        print(f"  Avg notional traded/day:       ${avg_notional:>10,.2f}")
        print()
        print(f"  Capital estimate scenarios:")
        print(f"    Naive worst-case (all open):    ${avg_n_quotes * 5.10:>10,.0f}")
        print(f"    Avg concurrent (~5min hold):    ${capital_5min_concurrent:>10,.0f}")
        print(f"    Capped at 20 concurrent quotes: ${capital_capped_20:>10,.0f}")
        print()
        print(f"  Monthly return on capital at $45/day × 22 days = $1,000:")
        print(f"    Naive:     {100 * 1000 / (avg_n_quotes * 5.10):.2f}%/month")
        print(f"    Realistic: {100 * 1000 / max(capital_5min_concurrent, 1):.2f}%/month")
        print(f"    Capped:    {100 * 1000 / capital_capped_20:.2f}%/month")

        # SAFETY MARGIN: what if we want $30/day (66% of target)?
        print()
        print("="*72)
        print("  ROBUSTNESS: what if some days achieve only 50% of target?")
        print("="*72)
        print(f"\n  At $45/day target × 22 days = $990/month")
        print(f"  If average day = 50% of target (= $22.50/day) -> $495/month ~= HALF the goal")
        print(f"  Recommendation: trade for $90/day target (2× target safety buffer)")
        print(f"  to robustly hit $1K/month even on weaker regime days.")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
