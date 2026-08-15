#!/usr/bin/env python3
"""Same threshold sweep as debug_model02_threshold_pick.py BUT with
Polymarket's real $1 minimum-order constraint.

At entry price p, $1 notional buys (1/p) shares. PnL on a win is
((1/p) * $1) - $1 = (1-p)/p (gross of fees). PnL on a loss is -$1.
Fees scale with share count: fee = shares * 0.072 * p*(1-p) per Polymarket schedule.

This is the correct sizing for live deployment.
"""
import io, json, sys
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

OOS_DATES = ["2026-05-15","2026-05-16","2026-05-17",
             "2026-05-18","2026-05-19","2026-05-20"]
ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
NOTIONAL_PER_FILL_USD = 1.00     # Polymarket's minimum order size

model = joblib.load(ART / "model.pkl")
feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
print(f"Loaded model — features={len(feats)}  has_calibrator={hasattr(model, '_calibrator')}")

parts = []
for d in OOS_DATES:
    p = Path(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    if not p.exists(): continue
    df = pl.read_parquet(p)
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug","snapshot_ts_ns"])
df_clean = clean_features(df)
print(f"Loaded {len(df):,} OOS snapshots across {len(OOS_DATES)} days")

X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df_clean.columns:
        s = df_clean.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
raw = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
p_up = np.clip(cal.predict(raw), 1e-6, 1-1e-6) if cal is not None else raw
print(f"Using {'CALIBRATED' if cal is not None else 'RAW'} predictions")

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
up_ask = df["up_token_best_ask"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid
ttc_band = (ttc >= 10) & (ttc <= 60)

candidate_mask = ttc_band & (edge_dn > 0)
candidate_edges = edge_dn[candidate_mask]
print(f"\nedge_dn percentiles (positive-edge candidates n={len(candidate_edges):,}):")
for p in [50, 60, 70, 75, 80, 85, 90, 95, 99]:
    print(f"  p{p:>2}: ${np.percentile(candidate_edges, p):.4f}")


def backtest(thr: float) -> dict:
    fee_calcs = {}
    mask = ttc_band & (edge_dn >= thr)
    idx = np.where(mask)[0]
    seen = set(); fills = []
    for i in idx:
        k = str(market_arr[i])
        if k in seen: continue
        seen.add(k); fills.append(i)
    fills = np.array(fills, dtype=int)
    if len(fills) == 0:
        return {"thr": thr, "fills": 0}

    entry = down_ask[fills]
    # Polymarket min-order sizing: $1 notional → shares = 1/entry
    shares = NOTIONAL_PER_FILL_USD / entry
    wins = (resolved[fills] == 0).astype(int)
    # Fees: scale with share count
    fees = np.empty(len(fills))
    for j, i in enumerate(fills):
        d = str(date_arr[i])
        fc = fee_calcs.setdefault(d, FeeCalculator.for_date(d))
        fees[j] = fc.taker_fee_usd(price=float(entry[j]), size=float(shares[j]))
    # PnL on a win: payoff = shares * $1 (each share pays $1 if DOWN wins).
    # On a loss: payoff = 0. Cost paid: shares * entry = $1 (the notional).
    payoff = np.where(wins == 1, shares * 1.0, 0.0)
    cost = NOTIONAL_PER_FILL_USD   # always $1
    gross = payoff - cost
    net = gross - fees

    # Aggregate per-day
    daily = {}
    for i, pnl in zip(fills, net):
        d = str(date_arr[i])
        daily[d] = daily.get(d, 0.0) + float(pnl)
    worst_day = min(daily.values()) if daily else 0.0
    n_pos_days = sum(1 for v in daily.values() if v > 0)
    chrono_pnl = net[np.argsort(df["snapshot_ts_ns"].to_numpy()[fills])]
    running = 0.0; peak = 0.0; max_dd = 0.0
    for x in chrono_pnl:
        running += x
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        "thr": thr, "fills": int(len(fills)),
        "wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "total_pnl_usd": float(net.sum()),
        "per_fill_pnl_usd": float(net.mean()),
        "weekly_pnl_usd": float(net.sum() / 6 * 7),
        "max_drawdown_usd": float(max_dd),
        "worst_day_pnl_usd": float(worst_day),
        "n_pos_days": int(n_pos_days),
        "n_days": len(daily),
        "all_pos_days": bool(n_pos_days == len(daily)),
        "avg_entry": float(entry.mean()),
        "avg_shares_per_fill": float(shares.mean()),
        "total_capital_deployed": float(NOTIONAL_PER_FILL_USD * len(fills)),
        "daily": {d: float(v) for d, v in daily.items()},
    }


percentiles = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 93, 94, 95, 97, 99]
print(f"\n{'pctile':>7} {'$thr':>9} {'fills':>6} {'avg_entry':>10} {'sh/fill':>9} "
      f"{'win%':>6} {'pnl':>9} {'pnl/fill':>10} {'weekly':>9} {'maxDD':>8} {'worst_day':>10} {'allPos':>7}")
print("-" * 130)
results = []
for pct in percentiles:
    thr = 0.0 if pct == 0 else float(np.percentile(candidate_edges, pct))
    r = backtest(thr)
    if r["fills"] == 0:
        print(f"  p{pct:>2} {thr:>+8.4f} {0:>6} (no fills)"); continue
    print(f"  p{pct:>2} {thr:>+8.4f} {r['fills']:>6} {r['avg_entry']:>9.3f} {r['avg_shares_per_fill']:>8.2f} "
          f"{r['win_rate']*100:>5.1f}% ${r['total_pnl_usd']:>+8.2f} ${r['per_fill_pnl_usd']:>+9.4f} "
          f"${r['weekly_pnl_usd']:>+8.2f} ${r['max_drawdown_usd']:>+7.2f} "
          f"${r['worst_day_pnl_usd']:>+9.2f} {('YES' if r['all_pos_days'] else 'NO'):>7}")
    r["percentile"] = pct
    results.append(r)

print("\n" + "=" * 80)
print("RECOMMENDATION (Polymarket \$1-min-order sizing)")
print("=" * 80)

# Best PnL unconditional
opt = max(results, key=lambda r: r["total_pnl_usd"])
# Best gate-passing (WR>=65%, all positive days)
passing = [r for r in results if r["win_rate"] >= 0.65 and r["all_pos_days"] and r["total_pnl_usd"] > 0]
if not passing:
    print("no threshold passes all gates"); conservative = None
else:
    gate_opt = max(passing, key=lambda r: r["total_pnl_usd"])
    # one step above
    opt_idx = next(i for i, r in enumerate(results) if r["percentile"] == gate_opt["percentile"])
    conservative = None
    for r in results[opt_idx + 1:]:
        if r["win_rate"] >= 0.65 and r["all_pos_days"] and r["total_pnl_usd"] > 0:
            conservative = r; break
    if conservative is None: conservative = gate_opt

    print(f"\nBest gate-passing (within WR>=65% & all positive days):")
    print(f"  p{gate_opt['percentile']:>2}  thr=${gate_opt['thr']:.4f}  fills={gate_opt['fills']}  "
          f"WR={gate_opt['win_rate']*100:.1f}%  weekly_PnL=${gate_opt['weekly_pnl_usd']:+.2f}")

    print(f"\n** RECOMMENDED (conservative — one step above gate optimum): **")
    print(f"  edge_threshold_usd = {conservative['thr']:.4f}  (p{conservative['percentile']})")
    print(f"  Expected fills/week: ~{conservative['fills']/6*7:.0f}")
    print(f"  Capital deployed per fill: ${NOTIONAL_PER_FILL_USD:.2f}  (Polymarket minimum)")
    print(f"  Avg shares per fill: {conservative['avg_shares_per_fill']:.2f}")
    print(f"  Win rate: {conservative['win_rate']*100:.1f}%")
    print(f"  Per-fill PnL: ${conservative['per_fill_pnl_usd']:+.4f}")
    print(f"  Weekly PnL: ${conservative['weekly_pnl_usd']:+.2f} on \$100 bankroll")
    print(f"  Max drawdown observed: ${conservative['max_drawdown_usd']:.2f}")
    print(f"  Worst day PnL: ${conservative['worst_day_pnl_usd']:+.2f}")
    print(f"  All positive days: {'YES' if conservative['all_pos_days'] else 'NO'}")

print(f"\n=== UNCONDITIONAL OPTIMUM (best total PnL, may not pass gate) ===")
print(f"  p{opt['percentile']:>2}  thr=${opt['thr']:.4f}  WR={opt['win_rate']*100:.1f}%  "
      f"weekly=${opt['weekly_pnl_usd']:+.2f}  maxDD=${opt['max_drawdown_usd']:.2f}")

# Concurrent margin sanity
print(f"\n=== CAPACITY CHECK ===")
fills_per_day = (conservative['fills'] / 6) if conservative else 0
avg_concurrent = fills_per_day / 86400 * 300   # 5 min hold
print(f"  Expected fills/day: ~{fills_per_day:.0f}")
print(f"  Avg concurrent positions (5-min holds): ~{avg_concurrent:.2f}")
print(f"  Avg notional locked: ~${avg_concurrent * NOTIONAL_PER_FILL_USD:.2f}  of \$100 bankroll")

Path("docs").mkdir(exist_ok=True)
Path("docs/model02_threshold_pick_1usd_min_order.json").write_text(json.dumps({
    "oos_dates": OOS_DATES,
    "model_artifact": str(ART),
    "notional_per_fill_usd": NOTIONAL_PER_FILL_USD,
    "sweep": results,
    "recommended_conservative": conservative,
    "unconditional_optimum": opt,
}, indent=2, default=str))
print(f"\nSaved to docs/model02_threshold_pick_1usd_min_order.json")
