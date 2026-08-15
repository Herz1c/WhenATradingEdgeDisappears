#!/usr/bin/env python3
"""Compare position-sizing schemes on the recommended v2 model + p65 threshold.

Same fills, same WR, same fee model (Polymarket post-2026-03-30:
fee = shares × 0.072 × price × (1-price)). Only varies the size of each fill.

Schemes tested:
  A. Fixed $1 (current baseline / Polymarket minimum)
  B. Linear in down_ask: size = clip($1 + 4*(down_ask-0.30), $1, $5)
     "buy more when market thinks DOWN is more likely"
  C. Linear in edge_dn: size = clip($1 + k*(edge-thr), $1, $5)
     "buy more when our signal is stronger relative to market"
  D. Tiered by down_ask: $1 / $2 / $3 / $4 in [0,0.3]/[0.3,0.5]/[0.5,0.7]/[0.7,1.0]
     (the scheme I proposed)
  E. Quarter-Kelly: size = max($1, min($5, 0.25 * bankroll * kelly_fraction))
     where kelly_fraction = (assumed_WR × (1/entry) - 1) / ((1/entry) - 1)
     using observed WR of 75.6% as the assumed win rate
  F. Capped Kelly with calibrated WR per entry-band

Constraints applied to ALL schemes:
  - Per-fill cap: $5 max (defensive)
  - Concurrent margin cap: $30 (30% of $100 bankroll) — chronological lockup
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

OOS_DATES = ["2026-05-15","2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20"]
ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
THRESHOLD = 0.3067    # p65 conservative
MIN_NOTIONAL = 1.00   # Polymarket minimum
MAX_NOTIONAL = 5.00   # defensive per-fill cap
MAX_MARGIN_USD = 30.0 # concurrent margin cap on $100 bankroll
ASSUMED_WR = 0.756    # observed WR at threshold p65

model = joblib.load(ART / "model.pkl")
feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
print(f"Model loaded; calibrator present: {hasattr(model, '_calibrator')}")

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

up_bid = df["up_token_best_bid"].to_numpy().astype(float)
ttc = df["t_to_close_s"].to_numpy().astype(float)
resolved = df["resolved_side_label"].to_numpy().astype(int)
date_arr = df["date"].to_numpy()
market_arr = df["market_slug"].to_numpy()
snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64) if "market_close_ts_ns" in df.columns else snap_ts + 300_000_000_000
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid

# Determine fills (same for all sizing schemes — threshold-gated, dedup per market)
ttc_band = (ttc >= 10) & (ttc <= 60)
mask = ttc_band & (edge_dn >= THRESHOLD)
idx = np.where(mask)[0]
seen = set(); fills = []
for i in idx:
    k = str(market_arr[i])
    if k in seen: continue
    seen.add(k); fills.append(i)
fills = np.array(fills, dtype=int)
print(f"Fills at threshold ${THRESHOLD:.4f}: {len(fills)}")
fill_entry = down_ask[fills]
fill_edge = edge_dn[fills]
fill_wins = (resolved[fills] == 0).astype(int)
fill_dates = date_arr[fills]
fill_snap_ts = snap_ts[fills]
fill_close_ts = close_ts[fills]

# Sizing schemes
def scheme_A(entry, edge):
    return MIN_NOTIONAL  # fixed $1

def scheme_B(entry, edge):
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, 1.0 + 4.0 * (entry - 0.30)))

def scheme_C(entry, edge):
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, 1.0 + 8.0 * (edge - THRESHOLD)))

def scheme_D(entry, edge):
    # Tiered by entry (= market's implied P(DOWN))
    if entry < 0.30:   return 1.0
    elif entry < 0.50: return 2.0
    elif entry < 0.70: return 3.0
    else:              return 4.0

def scheme_E(entry, edge):
    # Quarter-Kelly using assumed WR=75.6%
    payout = 1.0 / entry
    wr = ASSUMED_WR
    kelly = (wr * payout - 1.0) / (payout - 1.0)
    if kelly <= 0: return MIN_NOTIONAL
    bet = 0.25 * 100.0 * kelly  # quarter Kelly on $100 bankroll
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, bet))

def scheme_F(entry, edge):
    # Calibrated-WR Kelly: assume WR varies linearly with edge
    # At threshold (edge=0.3067), WR=0.756; at edge=0.55, observed WR=0.67
    # Linear interpolation: WR = 0.756 - 0.36 * (edge - 0.3067)
    wr_est = max(0.55, min(0.85, 0.756 - 0.36 * (edge - THRESHOLD)))
    payout = 1.0 / entry
    kelly = (wr_est * payout - 1.0) / (payout - 1.0)
    if kelly <= 0: return MIN_NOTIONAL
    bet = 0.25 * 100.0 * kelly
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, bet))

schemes = [
    ("A: Fixed $1 (baseline)", scheme_A),
    ("B: Linear in entry price", scheme_B),
    ("C: Linear in edge_dn", scheme_C),
    ("D: Tiered by entry [users idea]", scheme_D),
    ("E: ¼-Kelly (fixed WR=0.756)", scheme_E),
    ("F: ¼-Kelly (edge-conditional WR)", scheme_F),
]

# Simulate each scheme with margin-cap enforcement (chronological)
def run(scheme_fn):
    fee_calcs = {}
    chrono_order = np.argsort(fill_snap_ts)
    notional_history = []
    pnl_per_fill = []
    skipped_margin = 0
    open_positions = []  # heap-like list of (close_ts, notional)
    import heapq
    margin_in_use = 0.0
    for idx in chrono_order:
        i = fills[idx]
        e = fill_entry[idx]
        ed = fill_edge[idx]
        w = fill_wins[idx]
        # Release positions that have closed
        ts = fill_snap_ts[idx]
        while open_positions and open_positions[0][0] <= ts:
            ct, nt = heapq.heappop(open_positions)
            margin_in_use -= nt

        desired = scheme_fn(float(e), float(ed))
        # Enforce concurrent-margin cap
        if margin_in_use + desired > MAX_MARGIN_USD:
            available = MAX_MARGIN_USD - margin_in_use
            if available < MIN_NOTIONAL:
                skipped_margin += 1
                notional_history.append(0.0)
                pnl_per_fill.append(0.0)
                continue
            desired = available  # downsize but still fill
        notional = desired
        shares = notional / e
        # Fee
        d = str(fill_dates[idx])
        fc = fee_calcs.setdefault(d, FeeCalculator.for_date(d))
        fee = fc.taker_fee_usd(price=float(e), size=float(shares))
        # PnL
        payoff = shares * 1.0 if w == 1 else 0.0
        net = payoff - notional - fee
        pnl_per_fill.append(net)
        notional_history.append(notional)
        # Lock margin until close
        heapq.heappush(open_positions, (int(fill_close_ts[idx]), notional))
        margin_in_use += notional

    # Compute statistics
    pnl_arr = np.array(pnl_per_fill)
    notional_arr = np.array(notional_history)
    active = notional_arr > 0
    n_fills = int(active.sum())
    daily = {}
    for j, idx in enumerate(chrono_order):
        if notional_history[j] == 0: continue
        d = str(fill_dates[idx])
        daily[d] = daily.get(d, 0.0) + pnl_per_fill[j]
    worst_day = min(daily.values()) if daily else 0.0
    n_pos_days = sum(1 for v in daily.values() if v > 0)
    cumul = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cumul)
    dd = peak - cumul
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    return {
        "n_fills": n_fills,
        "skipped_margin": skipped_margin,
        "total_pnl_usd": float(pnl_arr.sum()),
        "per_fill_pnl_usd": float(pnl_arr[active].mean()) if active.sum() > 0 else 0.0,
        "weekly_pnl_usd": float(pnl_arr.sum() / 6 * 7),
        "win_rate": float((pnl_arr[active] > 0).mean()) if active.sum() > 0 else 0.0,
        "max_drawdown_usd": max_dd,
        "worst_day": worst_day,
        "best_day": max(daily.values()) if daily else 0.0,
        "n_pos_days": n_pos_days, "n_days": len(daily),
        "all_pos_days": bool(n_pos_days == len(daily)),
        "mean_notional": float(notional_arr[active].mean()) if active.sum() > 0 else 0.0,
        "max_notional": float(notional_arr.max()) if len(notional_arr) > 0 else 0.0,
        "total_deployed": float(notional_arr.sum()),
    }


print(f"\n{'scheme':<36} {'fills':>5} {'skip':>5} {'mean_$':>8} {'max_$':>7} {'PnL':>9} "
      f"{'pnl/fill':>10} {'weekly':>9} {'WR':>6} {'maxDD':>8} {'worst':>9} {'allPos':>7}")
print("-" * 130)
results = {}
for label, fn in schemes:
    r = run(fn)
    results[label] = r
    print(f"{label:<36} {r['n_fills']:>5} {r['skipped_margin']:>5} ${r['mean_notional']:>6.2f} "
          f"${r['max_notional']:>5.2f} ${r['total_pnl_usd']:>+7.2f} ${r['per_fill_pnl_usd']:>+9.4f} "
          f"${r['weekly_pnl_usd']:>+7.2f} {r['win_rate']*100:>5.1f}% ${r['max_drawdown_usd']:>+7.2f} "
          f"${r['worst_day']:>+8.2f} {('YES' if r['all_pos_days'] else 'NO'):>7}")

# Save
Path("docs").mkdir(exist_ok=True)
Path("docs/sizing_schemes_comparison.json").write_text(json.dumps({
    "model": str(ART),
    "threshold": THRESHOLD,
    "constraints": {"min_notional": MIN_NOTIONAL, "max_notional": MAX_NOTIONAL,
                    "max_margin_usd": MAX_MARGIN_USD},
    "schemes": results,
}, indent=2, default=str))

# Pick winner
best = max(results.items(), key=lambda kv: kv[1]["total_pnl_usd"])
print(f"\nBEST total PnL: {best[0]}")
print(f"  PnL=${best[1]['total_pnl_usd']:+.2f}  WR={best[1]['win_rate']*100:.1f}%  "
      f"maxDD=${best[1]['max_drawdown_usd']:.2f}  weekly=${best[1]['weekly_pnl_usd']:+.2f}")

# Best risk-adjusted: pnl / maxDD
best_calmar = max(results.items(), key=lambda kv: kv[1]["total_pnl_usd"] / max(0.01, kv[1]["max_drawdown_usd"]))
calmar_val = best_calmar[1]["total_pnl_usd"] / max(0.01, best_calmar[1]["max_drawdown_usd"])
print(f"\nBEST risk-adjusted (PnL/maxDD): {best_calmar[0]}")
print(f"  ratio={calmar_val:.1f}  PnL=${best_calmar[1]['total_pnl_usd']:+.2f}  maxDD=${best_calmar[1]['max_drawdown_usd']:.2f}")
