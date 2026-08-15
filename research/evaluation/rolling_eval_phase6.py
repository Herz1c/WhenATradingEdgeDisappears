"""Phase 6 — Multi-day rolling evaluation of maker-first strategy.

For each day in the available window, applies existing Model 03 (fill prob)
+ Model 04b (markout_to_close) to compute the realized top-K net_pnl_to_close.

Flags each day as in_train / val / test based on the original training split:
  Train:  Apr 19 - May 4
  Val:    May 5
  Test:   May 6

The in-train days reveal "how well the model fits its training distribution"
(overfit upper bound). The val+test days are the real out-of-sample signal.

Also computes a SMALL-MONEY sizing analysis: given a $1K/month target
($45/day), what selectivity is required at order_size=10?
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

# Training split (from registry)
TRAIN_DAYS = set(f"2026-04-{d:02d}" for d in range(19, 31))
TRAIN_DAYS |= set(f"2026-05-{d:02d}" for d in range(1, 5))
VAL_DAY = "2026-05-05"
TEST_DAY = "2026-05-06"

DAILY_TARGET_USD = 45  # $1K/month / ~22 trading days


def load_fs(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))["feature_columns"]


def evaluate_day(date_str: str, m03, m04b, fs03, fs04):
    """Evaluate ranked selectivity on a single day's parquet."""
    f = DATASET_PATH / f"{date_str}.parquet"
    if not f.exists():
        return None

    schema = pl.scan_parquet(str(f)).collect_schema()
    avail = set(schema.names())
    cols = list(dict.fromkeys(
        fs03 + fs04 + [
            "filled", "fill_size", "fill_price", "fill_notional",
            "net_pnl_to_close", "markout_to_close",
            "execution_training_eligible",
            "price_level", "order_side", "t_to_close_s",
        ]
    ))
    cols = [c for c in cols if c in avail]
    df = pl.read_parquet(str(f), columns=cols)
    df = df.filter(pl.col("execution_training_eligible"))
    n_total = df.height
    if n_total == 0:
        return None

    # Predict
    X03 = np.nan_to_num(_to_float_array(df.select(fs03))[0], nan=0.0)
    p_fill_raw = m03.predict_proba(X03)[:, 1]
    cal_03 = getattr(m03, "_calibrator", None)
    p_fill = np.clip(cal_03.predict(p_fill_raw), 1e-6, 1 - 1e-6) if cal_03 else p_fill_raw

    X04 = np.nan_to_num(_to_float_array(df.select(fs04))[0], nan=0.0)
    pred_mko_close = m04b.predict(X04)

    predicted_edge = p_fill * pred_mko_close

    y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
    actual_net = df["net_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
    actual_net_per_attempt = np.where(y_filled, actual_net, 0.0)
    actual_notional = df["fill_notional"].cast(pl.Float64).fill_null(0).to_numpy() if "fill_notional" in df.columns else np.zeros(n_total)

    # Top-K stats
    pcts = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
    topk = []
    for pct in pcts:
        k = int(pct * n_total)
        if k < 50:
            continue
        idx = np.argsort(-predicted_edge)[:k]
        n_sel = int(len(idx))
        n_fills = int(y_filled[idx].sum())
        per_att = float(actual_net_per_attempt[idx].mean())
        sum_pnl = float(actual_net_per_attempt[idx].sum())
        sum_notional = float(actual_notional[idx][y_filled[idx]].sum())
        topk.append({
            "top_pct": pct, "n_selected": n_sel, "n_fills": n_fills,
            "per_attempt_pnl": per_att,
            "sum_pnl": sum_pnl,
            "sum_notional_traded": sum_notional,
        })

    return {
        "date": date_str,
        "n_total": n_total,
        "fill_rate_total": float(y_filled.mean()),
        "mean_net_pnl_all_attempts": float(actual_net_per_attempt.mean()),
        "topk": topk,
    }


def main():
    print("\n" + "#"*72)
    print("  PHASE 6 — MULTI-DAY ROLLING EVAL")
    print("#"*72)

    fs03 = load_fs(FS_03)
    fs04 = load_fs(FS_04)
    m03 = joblib.load(MODEL_03_PATH)
    m04b = joblib.load(MODEL_04B_PATH)
    print(f"  Model 03 loaded: {type(m03).__name__}")
    print(f"  Model 04b loaded: {type(m04b).__name__}")

    # All available days
    all_files = sorted(DATASET_PATH.glob("*.parquet"))
    dates = [f.stem for f in all_files]
    print(f"  Days available: {len(dates)} ({dates[0]} -> {dates[-1]})")

    results = []
    for date_str in dates:
        if date_str in TRAIN_DAYS:
            tag = "TRAIN (in-sample)"
        elif date_str == VAL_DAY:
            tag = "VAL (out-of-sample)"
        elif date_str == TEST_DAY:
            tag = "TEST (out-of-sample)"
        else:
            tag = "?"
        print(f"\n  --- {date_str}  [{tag}] ---", flush=True)
        r = evaluate_day(date_str, m03, m04b, fs03, fs04)
        if r is None:
            print(f"    SKIP (no data)")
            continue
        r["tag"] = tag
        results.append(r)
        # Quick per-day summary
        top25 = next((t for t in r["topk"] if abs(t["top_pct"] - 0.25) < 1e-6), None)
        top5 = next((t for t in r["topk"] if abs(t["top_pct"] - 0.05) < 1e-6), None)
        top1 = next((t for t in r["topk"] if abs(t["top_pct"] - 0.01) < 1e-6), None)
        bulk = next((t for t in r["topk"] if abs(t["top_pct"] - 1.0) < 1e-6), None)
        print(f"    n={r['n_total']:>10,}  FR={r['fill_rate_total']:.4f}")
        if top1:  print(f"    top  1%: per_att=${top1['per_attempt_pnl']:+.4f}  sum=${top1['sum_pnl']:+,.2f}")
        if top5:  print(f"    top  5%: per_att=${top5['per_attempt_pnl']:+.4f}  sum=${top5['sum_pnl']:+,.2f}")
        if top25: print(f"    top 25%: per_att=${top25['per_attempt_pnl']:+.4f}  sum=${top25['sum_pnl']:+,.2f}")
        if bulk:  print(f"    bulk:    per_att=${bulk['per_attempt_pnl']:+.4f}  sum=${bulk['sum_pnl']:+,.2f}")

    # =====================================================================
    # Aggregate stats per top-K bucket, split by in-sample vs out-of-sample
    # =====================================================================
    print("\n\n" + "="*72)
    print("  AGGREGATE: per-attempt PnL distribution by top-K bucket")
    print("="*72)
    pcts = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
    groups = {
        "IN-SAMPLE (train days)": [r for r in results if "TRAIN" in r["tag"]],
        "OUT-OF-SAMPLE (val+test)": [r for r in results if "TRAIN" not in r["tag"]],
    }
    print(f"\n  {'top%':>6s}  {'group':<28s}  {'n_days':>7s}  {'mean':>9s}  {'std':>9s}  {'min':>9s}  {'max':>9s}")
    print(f"  {'-'*6}  {'-'*28}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for pct in pcts:
        for g_name, g_results in groups.items():
            values = []
            for r in g_results:
                t = next((x for x in r["topk"] if abs(x["top_pct"] - pct) < 1e-6), None)
                if t:
                    values.append(t["per_attempt_pnl"])
            if not values:
                continue
            arr = np.array(values)
            print(f"  top {int(pct*1000)/10:>3.1f}%  {g_name:<28s}  {len(arr):>7d}  "
                  f"{arr.mean():>+9.5f}  {arr.std():>9.5f}  "
                  f"{arr.min():>+9.5f}  {arr.max():>+9.5f}")

    # =====================================================================
    # Daily total PnL summary — what would maker-first generate per day?
    # =====================================================================
    print("\n\n" + "="*72)
    print("  DAILY TOTAL PnL: maker-first at order_size=10 (current dataset)")
    print("="*72)
    print(f"\n  Goal: $1K/month = ~${DAILY_TARGET_USD}/day")
    print()
    print(f"  {'top%':>6s}  {'group':<28s}  {'mean $/day':>13s}  {'median $/day':>13s}  {'min $/day':>13s}  {'max $/day':>13s}")
    for pct in [0.005, 0.01, 0.05, 0.10, 0.25]:
        for g_name, g_results in groups.items():
            values = []
            for r in g_results:
                t = next((x for x in r["topk"] if abs(x["top_pct"] - pct) < 1e-6), None)
                if t:
                    values.append(t["sum_pnl"])
            if not values:
                continue
            arr = np.array(values)
            print(f"  top {int(pct*1000)/10:>3.1f}%  {g_name:<28s}  ${arr.mean():>+12,.2f}  ${np.median(arr):>+12,.2f}  ${arr.min():>+12,.2f}  ${arr.max():>+12,.2f}")

    # =====================================================================
    # Small-money sizing analysis
    # =====================================================================
    print("\n\n" + "="*72)
    print(f"  SMALL-MONEY SIZING: hit ${DAILY_TARGET_USD}/day target conservatively")
    print("="*72)
    print(f"\n  Strategy: filter to top-K%, then take only the N highest-edge quotes per day.")
    print(f"  At order_size=10 shares × ~$0.51 fill price = ~$5.10 notional per quote.")
    print()

    # For each out-of-sample day, find min n_quotes required to hit $45/day
    out_of_sample = groups["OUT-OF-SAMPLE (val+test)"]
    for r in out_of_sample:
        date_str = r["date"]
        f = DATASET_PATH / f"{date_str}.parquet"
        if not f.exists():
            continue
        schema = pl.scan_parquet(str(f)).collect_schema()
        avail = set(schema.names())
        # AUDIT FIX: dedupe column list (fs03 ∪ fs04 has overlap)
        cols = list(dict.fromkeys(fs03 + fs04 + ["filled", "net_pnl_to_close", "fill_notional", "execution_training_eligible"]))
        cols = [c for c in cols if c in avail]
        df = pl.read_parquet(str(f), columns=cols)
        df = df.filter(pl.col("execution_training_eligible"))
        X03 = np.nan_to_num(_to_float_array(df.select(fs03))[0], nan=0.0)
        p_fill_raw = m03.predict_proba(X03)[:, 1]
        cal_03 = getattr(m03, "_calibrator", None)
        p_fill = np.clip(cal_03.predict(p_fill_raw), 1e-6, 1 - 1e-6) if cal_03 else p_fill_raw
        X04 = np.nan_to_num(_to_float_array(df.select(fs04))[0], nan=0.0)
        pred_mko = m04b.predict(X04)
        predicted_edge = p_fill * pred_mko
        y_filled = df["filled"].fill_null(False).to_numpy().astype(bool)
        actual_net = df["net_pnl_to_close"].cast(pl.Float64).fill_null(0).to_numpy()
        per_att = np.where(y_filled, actual_net, 0.0)

        sorted_idx = np.argsort(-predicted_edge)
        sorted_pnl = per_att[sorted_idx]
        cum_pnl = np.cumsum(sorted_pnl)
        # Find smallest N s.t. cum_pnl[N] >= DAILY_TARGET
        crossed = np.where(cum_pnl >= DAILY_TARGET_USD)[0]
        if len(crossed) > 0:
            n_quotes_to_target = int(crossed[0] + 1)
            n_fills_at_target = int(y_filled[sorted_idx[:n_quotes_to_target]].sum())
            avg_pnl_per_attempt = DAILY_TARGET_USD / n_quotes_to_target
            avg_pnl_per_fill = DAILY_TARGET_USD / max(n_fills_at_target, 1)
            print(f"  {date_str} [{r['tag']:<20s}]:")
            print(f"    n_quotes_needed = {n_quotes_to_target:,}  "
                  f"(={100*n_quotes_to_target/r['n_total']:.4f}% of day's opportunities)")
            print(f"    n_fills_actual  = {n_fills_at_target:,}")
            print(f"    avg PnL per quote attempt = ${avg_pnl_per_attempt:.5f}")
            print(f"    avg PnL per fill          = ${avg_pnl_per_fill:.5f}")
            # Estimate capital required = max simultaneous open position
            # Assume worst case: all quotes open at $5.10 notional simultaneously
            cap_est = n_quotes_to_target * 5.10
            print(f"    rough capital cap (all simultaneous): ${cap_est:,.0f}")
            cap_concurrent_50 = min(50, n_quotes_to_target) * 5.10  # cap at 50 concurrent
            print(f"    realistic capital (max 50 concurrent quotes): ~${cap_concurrent_50:.0f}")
            print()
        else:
            print(f"  {date_str} [{r['tag']:<20s}]:  CANNOT HIT TARGET (max cum_pnl=${cum_pnl[-1]:+,.2f})\n")

    # =====================================================================
    # Final consistency verdict
    # =====================================================================
    print("\n" + "="*72)
    print("  CONSISTENCY VERDICT")
    print("="*72)
    out_top25 = []
    for r in out_of_sample:
        t = next((x for x in r["topk"] if abs(x["top_pct"] - 0.25) < 1e-6), None)
        if t:
            out_top25.append(t["per_attempt_pnl"])
    in_top25 = []
    for r in groups["IN-SAMPLE (train days)"]:
        t = next((x for x in r["topk"] if abs(x["top_pct"] - 0.25) < 1e-6), None)
        if t:
            in_top25.append(t["per_attempt_pnl"])
    print(f"\n  Top-25% per-attempt PnL across {len(in_top25)} in-sample + {len(out_top25)} OOS days:")
    print(f"    IN-SAMPLE:  mean ${np.mean(in_top25):+.5f}  std {np.std(in_top25):.5f}  "
          f"range [${min(in_top25):+.5f}, ${max(in_top25):+.5f}]")
    print(f"    OUT-OF-SAMPLE: mean ${np.mean(out_top25):+.5f}  std {np.std(out_top25):.5f}  "
          f"range [${min(out_top25):+.5f}, ${max(out_top25):+.5f}]")
    print()
    drop = np.mean(out_top25) - np.mean(in_top25)
    if abs(drop) < 0.05:
        print(f"  Out-of-sample drop from in-sample: ${drop:+.5f}  -> CONSISTENT (no major drift)")
    elif drop < -0.05:
        print(f"  Out-of-sample drop: ${drop:+.5f}  -> MILD DRIFT (overfitting in-sample)")
    else:
        print(f"  Out-of-sample LIFT: ${drop:+.5f}  -> unusual (OOS better than train??)")

    # Save full JSON
    out_path = Path("artifacts/phase6_rolling_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump({
            "daily_target_usd": DAILY_TARGET_USD,
            "results_per_day": results,
        }, fout, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
