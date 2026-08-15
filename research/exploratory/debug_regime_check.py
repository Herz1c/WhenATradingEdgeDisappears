#!/usr/bin/env python3
"""Regime check — was the OOS window a single BTC regime?

Tests:
1. BTC price trajectory over OOS (May 7-16)
2. Overall resolution distribution (UP vs DOWN across ALL markets)
3. Strategy fill rate by BTC daily direction (up days vs down days)
4. Win rate by BTC daily direction
5. Win rate when market is "trending up" vs "trending down" intraday
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

# ── 1. BTC price trajectory ──────────────────────────────────────────────────
print("=" * 78)
print("1. BTC PRICE TRAJECTORY OVER OOS WINDOW (May 7-16)")
print("=" * 78)
if "binance_spot_mid" in df.columns:
    # Daily BTC summary
    print(f"{'date':<12} {'open':>10} {'mean':>10} {'close':>10} {'low':>10} {'high':>10} {'%chg':>7}")
    daily_open_close = {}
    for d in OOS:
        sub = df.filter(pl.col("date") == d).filter(pl.col("binance_spot_mid").is_not_null())
        if len(sub) == 0: continue
        sub = sub.sort("snapshot_ts_ns")
        prices = sub["binance_spot_mid"].to_numpy()
        open_p = prices[0]; close_p = prices[-1]
        chg = (close_p / open_p - 1) * 100
        daily_open_close[d] = (open_p, close_p, chg)
        print(f"{d:<12} ${open_p:>9,.0f} ${prices.mean():>9,.0f} ${close_p:>9,.0f} "
              f"${prices.min():>9,.0f} ${prices.max():>9,.0f} {chg:>+6.2f}%")
    first_day = list(daily_open_close.keys())[0]
    last_day = list(daily_open_close.keys())[-1]
    first_open = daily_open_close[first_day][0]
    last_close = daily_open_close[last_day][1]
    overall_chg = (last_close / first_open - 1) * 100
    print(f"\nOOS window: ${first_open:,.0f} -> ${last_close:,.0f}  ({overall_chg:+.2f}%)")

# ── 2. Overall UP vs DOWN resolution rate ────────────────────────────────────
print("\n" + "=" * 78)
print("2. OVERALL UP vs DOWN RESOLUTION RATE (all markets in OOS)")
print("=" * 78)
# Each market resolves once; we use unique markets
unique_markets = df.unique(subset=["market_slug"], keep="first")
all_resolutions = unique_markets["resolved_side_label"].to_numpy()
print(f"  Total unique markets in OOS: {len(all_resolutions)}")
print(f"  UP wins:   {(all_resolutions == 1).sum()} ({(all_resolutions == 1).mean()*100:.1f}%)")
print(f"  DOWN wins: {(all_resolutions == 0).sum()} ({(all_resolutions == 0).mean()*100:.1f}%)")

# Per-day
print(f"\n  Per-day UP win rate:")
for d in OOS:
    sub = unique_markets.filter(pl.col("date") == d) if "date" in unique_markets.columns else None
    if sub is None or len(sub) == 0:
        # date isn't in unique_markets; group manually
        sub = df.filter(pl.col("date") == d).unique(subset=["market_slug"], keep="first")
    r = sub["resolved_side_label"].to_numpy()
    if len(r) == 0: continue
    print(f"    {d}: UP wins {(r==1).sum()}/{len(r)} = {(r==1).mean()*100:.1f}%")

# ── 3. Our strategy fills ────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("3. STRATEGY FILL RATE BY BTC DAILY DIRECTION")
print("=" * 78)
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
date_fills = df["date"].to_numpy()[fills]

print(f"  {'date':<12} {'btc_chg':>8} {'fills':>6} {'wins':>5} {'WR':>7} {'all_up_rate':>13}")
for d in OOS:
    chg = daily_open_close.get(d, (None,None,None))[2]
    m = date_fills == d
    sub_all = df.filter(pl.col("date") == d).unique(subset=["market_slug"], keep="first")
    all_up = (sub_all["resolved_side_label"].to_numpy() == 1).mean() if len(sub_all) else 0
    n = m.sum(); k = wins[m].sum() if n else 0
    chg_s = f"{chg:+.2f}%" if chg is not None else "n/a"
    wr_s = f"{k/n*100:.1f}%" if n else "n/a"
    print(f"  {d:<12} {chg_s:>8} {n:>6} {k:>5} {wr_s:>7} {all_up*100:>11.1f}%")

# ── 4. Win rate by BTC daily direction ───────────────────────────────────────
print("\n" + "=" * 78)
print("4. STRATEGY WIN RATE: UP DAYS vs DOWN DAYS")
print("=" * 78)
up_days = [d for d in OOS if daily_open_close.get(d, (None,None,0))[2] > 0]
down_days = [d for d in OOS if daily_open_close.get(d, (None,None,0))[2] <= 0]
print(f"  BTC UP days:   {up_days}")
print(f"  BTC DOWN days: {down_days}")

up_m = np.isin(date_fills, up_days)
dn_m = np.isin(date_fills, down_days)
if up_m.sum() > 0:
    print(f"\n  On BTC UP days:   {up_m.sum()} fills, wins={wins[up_m].sum()} ({wins[up_m].mean()*100:.1f}%)")
if dn_m.sum() > 0:
    print(f"  On BTC DOWN days: {dn_m.sum()} fills, wins={wins[dn_m].sum()} ({wins[dn_m].mean()*100:.1f}%)")

# ── 5. Win rate by intraday BTC direction ────────────────────────────────────
print("\n" + "=" * 78)
print("5. WIN RATE BY INTRADAY BTC SHORT-TERM DIRECTION")
print("=" * 78)
if "btc_return_30s" in df.columns:
    ret30 = df["btc_return_30s"].to_numpy()[fills]
elif "btc_return_5s" in df.columns:
    ret30 = df["btc_return_5s"].to_numpy()[fills]
else:
    ret30 = None
if ret30 is not None:
    finite = np.isfinite(ret30)
    if finite.sum() > 100:
        vals = ret30[finite]
        qs = np.quantile(vals, [0.25, 0.5, 0.75])
        for lo, hi, name in [(-1e9, qs[0], "BTC dropping hard"), (qs[0], qs[1], "BTC mildly down"),
                              (qs[1], qs[2], "BTC mildly up"), (qs[2], 1e9, "BTC ripping up")]:
            m = (ret30 >= lo) & (ret30 < hi) & finite
            if m.sum() < 20: continue
            print(f"  {name} (return [{lo:.5f},{hi:.5f})): n={m.sum()}, win%={wins[m].mean()*100:.1f}")
