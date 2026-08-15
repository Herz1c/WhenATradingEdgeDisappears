#!/usr/bin/env python3
"""Pick the deployment threshold for the new Model 02 DOWN-side taker.

Method:
  - Sweep edge_threshold by percentile of the edge_dn (= up_bid - p_up)
    distribution, not by fixed dollar values. This adapts to the new
    calibrated probability distribution automatically.
  - Evaluate each percentile threshold on the 6 OOS days May 15-20 (May 21
    not built — partial day data).
  - Report per-threshold: n fills, win rate, per-fill PnL, total weekly PnL,
    max-drawdown, worst-day PnL, all-positive-days flag.
  - Recommend a conservative threshold: one step ABOVE the percentile
    with maximum total PnL (more selective = fewer fills but safer).

The threshold reported is dollar-denominated so it plugs directly into the
live taker engine's `edge_threshold_usd` field.

Strategy config (held fixed):
  - DOWN side only
  - t_to_close in [10s, 60s]
  - 1 position per market
  - 1 contract per fill
  - Margin cap: $100 (won't bind at this volume)
"""
import io, json, sys, math
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features
from backtest.fees import FeeCalculator

OOS_DATES = ["2026-05-15", "2026-05-16", "2026-05-17",
             "2026-05-18", "2026-05-19", "2026-05-20"]

ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
model = joblib.load(ART / "model.pkl")
feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
print(f"Loaded retrained Model 02 (dense_close) — {len(feats)} features")

# --- Load OOS data ----------------------------------------------------------

parts = []
for d in OOS_DATES:
    p = Path(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    if not p.exists():
        print(f"  WARN missing {d}"); continue
    df = pl.read_parquet(p)
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug", "snapshot_ts_ns"])
df_clean = clean_features(df)
print(f"Loaded {len(df):,} OOS snapshots across {len(OOS_DATES)} days")

# --- Predict P(UP) ----------------------------------------------------------

X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df_clean.columns:
        s = df_clean.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
raw = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
if cal is not None:
    p_up = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    print(f"Using calibrated predictions (isotonic)")
else:
    p_up = raw
    print(f"WARN no calibrator on model")

# --- Pre-compute signals ----------------------------------------------------

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
up_ask = df["up_token_best_ask"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
edge_dn = up_bid - p_up      # we buy DOWN at down_ask = 1 - up_bid
down_ask = 1.0 - up_bid

# Filter to ttc band
ttc_band = (ttc >= 10) & (ttc <= 60)

# Only positive-edge candidates form the threshold distribution
candidate_mask = ttc_band & (edge_dn > 0)
candidate_edges = edge_dn[candidate_mask]
print(f"\nedge_dn distribution on positive-edge candidates (n={len(candidate_edges):,}):")
for p in [50, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]:
    print(f"  p{p:>2}: ${np.percentile(candidate_edges, p):.4f}")


# --- Backtest one threshold -------------------------------------------------

def backtest(thr: float) -> dict:
    """Run the DOWN-only taker with this $ threshold, dedup per market."""
    fee_calcs: dict[str, FeeCalculator] = {}
    mask = ttc_band & (edge_dn >= thr)
    idx = np.where(mask)[0]
    seen = set()
    fills = []
    for i in idx:
        k = str(market_arr[i])
        if k in seen: continue
        seen.add(k); fills.append(i)
    fills = np.array(fills, dtype=int)
    if len(fills) == 0:
        return {"thr": thr, "fills": 0}

    wins = (resolved[fills] == 0).astype(int)
    entry = down_ask[fills]
    # Fees
    fees = np.empty(len(fills))
    for j, i in enumerate(fills):
        d = str(date_arr[i])
        fc = fee_calcs.setdefault(d, FeeCalculator.for_date(d))
        fees[j] = fc.taker_fee_usd(price=float(entry[j]), size=1.0)
    gross = np.where(wins == 1, 1.0 - entry, -entry)
    net = gross - fees

    # Per-day
    daily = {}
    for i, pnl in zip(fills, net):
        d = str(date_arr[i])
        daily[d] = daily.get(d, 0.0) + float(pnl)
    worst_day = min(daily.values()) if daily else 0.0
    best_day = max(daily.values()) if daily else 0.0
    n_pos_days = sum(1 for v in daily.values() if v > 0)
    # Running drawdown (chronological)
    chrono = fills[np.argsort(df["snapshot_ts_ns"].to_numpy()[fills])]
    chrono_pnl = net[np.argsort(df["snapshot_ts_ns"].to_numpy()[fills])]
    running = 0.0; peak = 0.0; max_dd = 0.0
    for p in chrono_pnl:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        "thr": thr,
        "fills": int(len(fills)),
        "wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "total_pnl_usd": float(net.sum()),
        "per_fill_pnl_usd": float(net.mean()),
        "weekly_pnl_usd": float(net.sum() / 6 * 7),   # 6 OOS days -> normalize to 7
        "max_drawdown_usd": float(max_dd),
        "worst_day_pnl_usd": float(worst_day),
        "best_day_pnl_usd": float(best_day),
        "n_pos_days": int(n_pos_days),
        "n_days": len(daily),
        "all_pos_days": bool(n_pos_days == len(daily)),
        "daily": {d: float(v) for d, v in daily.items()},
    }


# --- Sweep ------------------------------------------------------------------

# Percentile grid of candidate edges; convert each to a dollar threshold
percentiles = [0, 50, 60, 70, 75, 80, 82, 84, 86, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99]
print(f"\n{'pctile':>7} {'$thr':>9} {'fills':>7} {'win%':>6} {'pnl':>9} {'pnl/fill':>10} "
      f"{'wkly':>8} {'maxDD':>8} {'worst_day':>10} {'allPos':>7}")
print("-" * 95)
results = []
for pct in percentiles:
    if pct == 0:
        thr = 0.0
    else:
        thr = float(np.percentile(candidate_edges, pct))
    r = backtest(thr)
    if r["fills"] == 0:
        print(f"  p{pct:>2} {thr:>+8.4f} {0:>7} (no fills)")
        continue
    print(f"  p{pct:>2} {thr:>+8.4f} {r['fills']:>7} {r['win_rate']*100:>5.1f}% "
          f"${r['total_pnl_usd']:>+8.2f} ${r['per_fill_pnl_usd']:>+9.4f} "
          f"${r['weekly_pnl_usd']:>+7.2f} ${r['max_drawdown_usd']:>+7.2f} "
          f"${r['worst_day_pnl_usd']:>+9.2f} {('YES' if r['all_pos_days'] else 'NO'):>7}")
    r["percentile"] = pct
    results.append(r)


# --- Pick optimum + conservative recommendation -----------------------------

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Total PnL champion (unconditional)
results_sorted_pnl = sorted(results, key=lambda r: r["total_pnl_usd"], reverse=True)
optimum = results_sorted_pnl[0]

print(f"\nBest-PnL threshold (unconditional, do NOT use directly):")
print(f"  p{optimum['percentile']:>2}  thr=${optimum['thr']:.4f}  "
      f"fills={optimum['fills']}  win%={optimum['win_rate']*100:.1f}  "
      f"total=${optimum['total_pnl_usd']:+.2f}  weekly~${optimum['weekly_pnl_usd']:+.2f}  "
      f"maxDD=${optimum['max_drawdown_usd']:.2f}")

# Apply gates from user spec: WR >= 65%, all positive days, PnL positive
GATE_WR = 0.65
passing = [r for r in results
           if r["win_rate"] >= GATE_WR
           and r["all_pos_days"]
           and r["total_pnl_usd"] > 0]
if not passing:
    print(f"\n NO threshold passes all gates (WR>=65%, all days positive, PnL>0).")
    print(f"   Recommend NOT deploying. Best available below the WR gate:")
    for r in results_sorted_pnl[:3]:
        print(f"     p{r['percentile']:>2} thr=${r['thr']:.4f}  WR={r['win_rate']*100:.1f}  "
              f"PnL=${r['total_pnl_usd']:+.2f}")
    conservative = None
else:
    # Optimum WITHIN gate-passing set (max PnL)
    passing_sorted = sorted(passing, key=lambda r: r["total_pnl_usd"], reverse=True)
    gate_optimum = passing_sorted[0]
    # One step above (higher percentile = more selective)
    opt_idx = next(i for i, r in enumerate(results) if r["percentile"] == gate_optimum["percentile"])
    # Find the next-higher percentile that ALSO passes the gate
    conservative = None
    for r in results[opt_idx + 1:]:
        if r["win_rate"] >= GATE_WR and r["all_pos_days"] and r["total_pnl_usd"] > 0:
            conservative = r; break
    if conservative is None:
        conservative = gate_optimum
        print(f"\nNote: no percentile above gate_optimum (p{gate_optimum['percentile']}) "
              f"passes the gate; using gate_optimum itself as conservative pick.")

    print(f"\nBest-PnL threshold within gate (WR>=65%, all days positive):")
    print(f"  p{gate_optimum['percentile']:>2}  thr=${gate_optimum['thr']:.4f}  "
          f"fills={gate_optimum['fills']}  win%={gate_optimum['win_rate']*100:.1f}  "
          f"total=${gate_optimum['total_pnl_usd']:+.2f}  weekly~${gate_optimum['weekly_pnl_usd']:+.2f}")

    print(f"\n** RECOMMENDED (conservative — one step above gate optimum): **")
    print(f"  Percentile: p{conservative['percentile']}")
    print(f"  edge_threshold_usd = {conservative['thr']:.4f}")
    print(f"  Expected fills/week: ~{conservative['fills']/6*7:.0f}")
    print(f"  Win rate: {conservative['win_rate']*100:.1f}%")
    print(f"  Per-fill PnL: ${conservative['per_fill_pnl_usd']:+.4f}")
    print(f"  Weekly PnL (normalized to 7 days): ${conservative['weekly_pnl_usd']:+.2f}")
    print(f"  Max drawdown observed: ${conservative['max_drawdown_usd']:.2f}")
    print(f"  Worst day PnL: ${conservative['worst_day_pnl_usd']:+.2f}")
    print(f"  All positive days: {'YES' if conservative['all_pos_days'] else 'NO'} "
          f"({conservative['n_pos_days']}/{conservative['n_days']})")
    print(f"  ✓ Win rate ≥ 65% — calibration intact")
    print(f"  ✓ All {conservative['n_days']} OOS days positive")

# Save full result
Path("docs").mkdir(exist_ok=True)
Path("docs/model02_threshold_pick_result.json").write_text(json.dumps({
    "oos_dates": OOS_DATES,
    "model_artifact": str(ART),
    "candidate_edge_distribution": {
        f"p{p}": float(np.percentile(candidate_edges, p)) for p in [50, 75, 80, 85, 90, 95, 99]
    },
    "sweep": results,
    "optimum_unconditional": optimum,
    "recommended_conservative": conservative,
    "gate_wr_min": 0.65,
}, indent=2, default=str))
print(f"\nSaved full sweep to docs/model02_threshold_pick_result.json")
