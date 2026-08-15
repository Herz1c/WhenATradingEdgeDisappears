#!/usr/bin/env python3
"""Two questions to test the 'retail-inefficiency hypothesis':

A) Does the edge DECAY across the OOS window?
   If win rate trends down day-by-day, that's a competitor learning.
   If it's flat, that's a structural inefficiency.

B) Does the edge concentrate in retail-heavy hours?
   Polymarket retail is most active US evening (00:00-06:00 UTC) and
   EU evening (18:00-22:00 UTC). If win rate is higher in those hours,
   that's evidence we're picking off retail flow.
"""
import io, sys, json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features

feats = list(json.loads(Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/feature_importance.json").read_text()).keys())
model = joblib.load("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl")

OOS = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10","2026-05-11",
       "2026-05-12","2026-05-13","2026-05-14","2026-05-15","2026-05-16"]

parts = []
for d in OOS:
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug", "snapshot_ts_ns"])
df_clean = clean_features(df)
X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df_clean.columns:
        s = df_clean.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
p_up = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
if cal is not None: p_up = np.clip(cal.predict(p_up), 1e-6, 1-1e-6)

up_bid = df["up_token_best_bid"].to_numpy()
up_ask = df["up_token_best_ask"].to_numpy()
ttc = df["t_to_close_s"].to_numpy()
market = df["market_slug"].to_numpy()
snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
resolved = df["resolved_side_label"].to_numpy()
edge_dn = up_bid - p_up
mask = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
idx = np.where(mask)[0]
seen = set(); fills = []
for i in idx:
    k = str(market[i])
    if k in seen: continue
    seen.add(k); fills.append(i)
fills = np.array(fills)
wins = (resolved[fills] == 0).astype(int)
dates_arr = df["date"].to_numpy()
ts_fills = snap_ts[fills]
entry = (1.0 - up_bid[fills])

# ── A) Edge decay across days ────────────────────────────────────────────────
print("=" * 78)
print("A) EDGE DECAY ACROSS OOS WINDOW (day-by-day)")
print("=" * 78)
print(f"{'date':<12} {'n':>5} {'wins':>5} {'win%':>7} {'pnl_per_trade':>15}")
prev_wr = None
for d in OOS:
    m = dates_arr[fills] == d
    if m.sum() == 0: continue
    nn = m.sum(); kk = wins[m].sum()
    wr = kk/nn
    avg_entry = entry[m].mean()
    fee = 0.072 * avg_entry * (1 - avg_entry)
    ept = wr * (1 - avg_entry) - (1 - wr) * avg_entry - fee
    trend = ""
    if prev_wr is not None:
        diff = wr - prev_wr
        if abs(diff) > 0.03:
            trend = " ⬆" if diff > 0 else " ⬇"
    print(f"{d:<12} {nn:>5} {kk:>5} {wr*100:>6.1f}%{trend}  ${ept:>+12.4f}")
    prev_wr = wr

# Linear trend in win rate vs day index
day_idx = np.array([OOS.index(d) for d in dates_arr[fills]])
slope, intercept = np.polyfit(day_idx, wins, 1)
print(f"\nLinear trend in win rate: slope = {slope*100:+.3f}pp/day  intercept = {intercept*100:.1f}%")
print(f"  Over 10 days, win rate would drift {slope*10*100:+.2f}pp")
# Statistical significance via correlation
from scipy.stats import pearsonr
r, p = pearsonr(day_idx, wins)
print(f"  Pearson r = {r:+.4f}  p = {p:.4f}  (p<0.05 means real trend)")

# ── B) Edge by hour of day ───────────────────────────────────────────────────
print("\n" + "=" * 78)
print("B) WIN RATE BY HOUR OF DAY (UTC) — retail concentration check")
print("=" * 78)
hours = np.array([datetime.fromtimestamp(t/1e9, tz=timezone.utc).hour for t in ts_fills])
print(f"{'hour UTC':<10} {'n':>5} {'wins':>5} {'win%':>7} {'note':<20}")
hour_notes = {
    0: "US evening peak", 1: "US evening peak", 2: "US evening peak",
    3: "US late night", 4: "US late night", 5: "AS morning",
    13: "EU afternoon", 14: "EU/US-east overlap", 15: "US morning",
    18: "EU evening", 19: "EU evening peak", 20: "EU evening peak", 21: "EU late",
    22: "US evening", 23: "US evening peak",
}
for h in range(24):
    m = hours == h
    if m.sum() < 5: continue
    nn = m.sum(); kk = wins[m].sum()
    note = hour_notes.get(h, "")
    print(f"  {h:>2}:00    {nn:>5} {kk:>5} {kk/nn*100:>6.1f}%  {note}")

# Retail vs off-retail hours
retail_hours = set([0,1,2,3,4,18,19,20,21,22,23])
off_retail = set(range(24)) - retail_hours
ret_m = np.isin(hours, list(retail_hours))
off_m = np.isin(hours, list(off_retail))
print(f"\nRetail-peak hours ({sorted(retail_hours)}):    n={ret_m.sum()}  win%={wins[ret_m].mean()*100:.1f}")
print(f"Off-retail hours ({sorted(off_retail)}):       n={off_m.sum()}  win%={wins[off_m].mean()*100:.1f}")

# ── C) Edge by market price level (where does retail mispricing concentrate?) ─
print("\n" + "=" * 78)
print("C) WIN RATE BY MARKET MID (where retail mispricing concentrates)")
print("=" * 78)
# entry = down_ask = 1 - up_bid. Up_mid roughly = (up_bid + up_ask)/2.
up_mid_fills = (up_bid[fills] + up_ask[fills]) / 2
print(f"{'up_mid band':<14} {'n':>5} {'wins':>5} {'win%':>7}")
for lo, hi in [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.0)]:
    m = (up_mid_fills >= lo) & (up_mid_fills < hi)
    if m.sum() < 10: continue
    print(f"[{lo:.2f}, {hi:.2f})   {m.sum():>5} {wins[m].sum():>5} {wins[m].mean()*100:>6.1f}%")

print("\nInterpretation:")
print("  - If win rate is HIGH where up_mid is in 'gambler-overpriced' bands (0.50-0.80),")
print("    that supports the retail-mispricing hypothesis.")
print("  - If win rate is flat across all bands, edge is not specifically retail-driven.")
