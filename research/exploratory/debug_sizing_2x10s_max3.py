#!/usr/bin/env python3
"""Scheme C (linear-in-edge sizing) BUT with:
  - MAX bet = $3 (instead of $5)
  - 2 entries per market (instead of 1)
  - Minimum 10s gap between entries on same market
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

OOS_DATES = ["2026-05-15","2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20"]
ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
THRESHOLD = 0.3067
MIN_NOTIONAL = 1.00
MAX_NOTIONAL = 3.00                       # was 5.00 in prior scheme C
MAX_POSITIONS_PER_MARKET = 2              # was 1 in prior scheme C
MIN_SECONDS_BETWEEN_ENTRIES = 10.0
MAX_MARGIN_USD = 30.0                     # concurrent margin cap
EDGE_K = 8.0                              # scheme C slope

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
close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64) if "market_close_ts_ns" in df.columns \
           else snap_ts + 300_000_000_000
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid
ttc_band = (ttc >= 10) & (ttc <= 60)

# Apply 2-entries-per-market with 10s gap rule
candidate_mask = ttc_band & (edge_dn >= THRESHOLD)
candidate_idx = np.where(candidate_mask)[0]

# Sort by time (we already sorted by market then snap_ts_ns earlier, but for chronological
# replay we need time-order across markets)
chrono_order = np.argsort(snap_ts[candidate_idx])
candidate_idx_chrono = candidate_idx[chrono_order]

position_count_per_market: dict[str, int] = {}
last_entry_ts_per_market: dict[str, int] = {}
gap_ns = int(MIN_SECONDS_BETWEEN_ENTRIES * 1e9)

fills = []
n_skipped_per_market_cap = 0
n_skipped_gap = 0
for i in candidate_idx_chrono:
    mk = str(market_arr[i])
    cur = position_count_per_market.get(mk, 0)
    if cur >= MAX_POSITIONS_PER_MARKET:
        n_skipped_per_market_cap += 1
        continue
    last_ts = last_entry_ts_per_market.get(mk)
    if last_ts is not None and (snap_ts[i] - last_ts) < gap_ns:
        n_skipped_gap += 1
        continue
    fills.append(i)
    position_count_per_market[mk] = cur + 1
    last_entry_ts_per_market[mk] = int(snap_ts[i])

fills = np.array(fills, dtype=int)
print(f"\nFills: {len(fills)} (after 2-entries-per-market + 10s gap rule)")
print(f"  skipped (market cap full): {n_skipped_per_market_cap}")
print(f"  skipped (within 10s of prior entry in same market): {n_skipped_gap}")
print(f"  unique markets traded: {len(position_count_per_market)}")
print(f"  markets with 2 entries: {sum(1 for v in position_count_per_market.values() if v == 2)}")
print(f"  markets with 1 entry:  {sum(1 for v in position_count_per_market.values() if v == 1)}")

# Sizing — Scheme C with MAX=$3
fill_entry = down_ask[fills]
fill_edge = edge_dn[fills]
fill_wins = (resolved[fills] == 0).astype(int)
fill_date = date_arr[fills]
fill_snap = snap_ts[fills]
fill_close = close_ts[fills]

def size_C(edge):
    return float(np.clip(1.0 + EDGE_K * max(0.0, edge - THRESHOLD), MIN_NOTIONAL, MAX_NOTIONAL))

# Simulate with margin cap (chronological)
import heapq
fee_calcs = {}
chrono_iter = np.argsort(fill_snap)
open_heap: list[tuple[int, float]] = []
margin_in_use = 0.0
pnl_per_fill = []
notional_per_fill = []
n_skipped_margin = 0

for idx in chrono_iter:
    i = fills[idx]
    e = float(fill_entry[idx])
    ed = float(fill_edge[idx])
    w = int(fill_wins[idx])
    ts = int(fill_snap[idx])
    # Free positions that have closed
    while open_heap and open_heap[0][0] <= ts:
        _, freed = heapq.heappop(open_heap)
        margin_in_use -= freed
    notional = size_C(ed)
    if margin_in_use + notional > MAX_MARGIN_USD:
        avail = MAX_MARGIN_USD - margin_in_use
        if avail < MIN_NOTIONAL:
            n_skipped_margin += 1
            notional_per_fill.append(0.0); pnl_per_fill.append(0.0); continue
        notional = avail
    shares = notional / e
    d = str(fill_date[idx])
    fc = fee_calcs.setdefault(d, FeeCalculator.for_date(d))
    fee = fc.taker_fee_usd(price=e, size=shares)
    payoff = shares * 1.0 if w == 1 else 0.0
    net = payoff - notional - fee
    notional_per_fill.append(notional)
    pnl_per_fill.append(net)
    heapq.heappush(open_heap, (int(fill_close[idx]), notional))
    margin_in_use += notional

pnl = np.array(pnl_per_fill)
nots = np.array(notional_per_fill)
active = nots > 0

# Per-day
daily = {}
for j, idx in enumerate(chrono_iter):
    if nots[j] == 0: continue
    d = str(fill_date[idx])
    daily[d] = daily.get(d, 0.0) + pnl_per_fill[j]
# Drawdown
cum = np.cumsum(pnl)
peak = np.maximum.accumulate(cum) if len(cum) > 0 else np.array([0])
dd = peak - cum
max_dd = float(dd.max()) if len(dd) > 0 else 0.0

n_act = int(active.sum())
wr = float((pnl[active] > 0).mean()) if n_act else 0.0
print(f"\nResults:")
print(f"  active fills:        {n_act}")
print(f"  skipped (margin):    {n_skipped_margin}")
print(f"  win rate:            {wr*100:.1f}%")
print(f"  mean notional:       ${nots[active].mean():.2f}")
print(f"  max notional:        ${nots.max():.2f}")
print(f"  total notional:      ${nots.sum():.2f}")
print(f"  total PnL:           ${pnl.sum():+.2f}")
print(f"  per-fill PnL:        ${pnl[active].mean():+.4f}")
print(f"  weekly PnL (x7/6):   ${pnl.sum() / 6 * 7:+.2f}")
print(f"  max drawdown:        ${max_dd:.2f}")
print(f"  Calmar:              {pnl.sum() / max(0.01, max_dd):.1f}")
print(f"  worst day:           ${min(daily.values()):+.2f}")
print(f"  best day:            ${max(daily.values()):+.2f}")
print(f"  all days positive:   {all(v > 0 for v in daily.values())}  ({sum(1 for v in daily.values() if v > 0)}/{len(daily)})")
print(f"  per-day:")
for d, v in sorted(daily.items()):
    print(f"    {d}: ${v:+.2f}")

# Side-by-side with prior scheme C variants
print(f"\n=== SIDE-BY-SIDE on the same model + threshold ===")
print(f"{'config':<45} {'fills':>5} {'WR':>6} {'mean$':>7} {'pnl':>9} {'weekly':>9} {'maxDD':>8} {'calmar':>7}")
prior_results = [
    ("A. fixed $1, 1 entry/market",         352,  75.6, 1.00,  292.05, 340.72,  4.84, 60.4),
    ("C. linear-in-edge max $5, 1/market",  352,  75.6, 1.93,  727.07, 848.25, 10.66, 68.2),
    ("Current: max $3, 2/market w/ 10s gap", n_act, wr*100, float(nots[active].mean()),
                                              pnl.sum(), pnl.sum()/6*7, max_dd, pnl.sum()/max(0.01,max_dd)),
]
for label, fills_n, wr_pct, mean_d, p, wk, md, cal_v in prior_results:
    print(f"{label:<45} {fills_n:>5} {wr_pct:>5.1f}% ${mean_d:>5.2f} ${p:>+7.2f} ${wk:>+7.2f} ${md:>+6.2f} {cal_v:>6.1f}")

Path("docs").mkdir(exist_ok=True)
Path("docs/sizing_2entries_max3.json").write_text(json.dumps({
    "model": str(ART),
    "threshold": THRESHOLD,
    "scheme": "C (linear in edge_dn)",
    "max_notional": MAX_NOTIONAL,
    "max_positions_per_market": MAX_POSITIONS_PER_MARKET,
    "min_seconds_between_entries": MIN_SECONDS_BETWEEN_ENTRIES,
    "max_margin_usd": MAX_MARGIN_USD,
    "n_fills_active": n_act,
    "n_skipped_per_market_cap": n_skipped_per_market_cap,
    "n_skipped_gap": n_skipped_gap,
    "n_skipped_margin": n_skipped_margin,
    "total_pnl_usd": float(pnl.sum()),
    "per_fill_pnl_usd": float(pnl[active].mean()) if n_act else 0.0,
    "weekly_pnl_usd": float(pnl.sum() / 6 * 7),
    "win_rate": float(wr),
    "max_drawdown_usd": max_dd,
    "calmar": float(pnl.sum() / max(0.01, max_dd)),
    "mean_notional": float(nots[active].mean()) if n_act else 0.0,
    "max_notional_observed": float(nots.max()) if len(nots) else 0.0,
    "daily_pnl": daily,
}, indent=2))
print(f"\nSaved to docs/sizing_2entries_max3.json")
