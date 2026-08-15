#!/usr/bin/env python3
"""Deeper audit: stale-quote arbitrage check + latency stress + market efficiency check.

The big question after audit part 1:
- entry [0.50, 0.70): n=363, win=79.3%, realized_DOWN_avg=0.21
- Market is pricing P(DOWN) = 0.60, actual is 0.79. That's 19% market mispricing.
- Why so big? Either market is irrational, or our backtest is using stale quotes.

Tests here:
1. Check stale-book flags & churn rate on filled trades
2. Compare quote staleness when signal fires vs random snapshots
3. LATENCY STRESS: re-price entry using up_ask from N seconds later
   (simulates the gap between snapshot and order arrival)
4. Look at price velocity at fill — are we picking up momentum-stale quotes?
"""
import io, sys, json, math
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features

feat_path = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/feature_importance.json")
feats = list(json.loads(feat_path.read_text()).keys())
model = joblib.load("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl")

OOS = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10","2026-05-11",
       "2026-05-12","2026-05-13","2026-05-14","2026-05-15","2026-05-16"]

# Load with NO subsampling so we have all snapshots per market for latency testing
parts = []
for d in OOS:
    df = pl.read_parquet(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort(["market_slug", "snapshot_ts_ns"])
print(f"Loaded {len(df)} full-resolution snapshots across {len(OOS)} OOS days")
print(f"Unique markets: {df['market_slug'].n_unique()}")

# Look for staleness columns
stale_cols = [c for c in df.columns if "stale" in c.lower() or "churn" in c.lower() or "update" in c.lower()]
print(f"\nStaleness/churn columns: {stale_cols}")

df_clean = clean_features(df)

# Predict
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
down_ask = 1.0 - up_bid

# Apply our filter, dedupe per market (one entry per market)
mask = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
idx = np.where(mask)[0]
# Need to dedupe by market_slug — keep first
seen = set(); fills = []
for i in idx:
    k = str(market[i])
    if k in seen: continue
    seen.add(k); fills.append(i)
print(f"\nFilled trades (after dedup): {len(fills)}")

# AUDIT A: churn / staleness on fills
print("\n" + "=" * 78)
print("AUDIT A — Quote churn / staleness on fills vs population")
print("=" * 78)
for col in ["up_token_price_change_updates_5s", "up_token_best_bid_ask_updates_5s",
            "up_token_best_bid_ask_updates_1s", "stale_book"]:
    if col not in df.columns: continue
    arr = df[col].to_numpy()
    if df.schema[col] == pl.Boolean:
        arr = arr.astype(int)
    arr_fills = arr[fills]
    print(f"\n{col}:")
    print(f"  All snapshots   : median={np.nanmedian(arr):.2f}  mean={np.nanmean(arr):.3f}")
    print(f"  Filled snapshots: median={np.nanmedian(arr_fills):.2f}  mean={np.nanmean(arr_fills):.3f}")

# AUDIT B: market efficiency check — is up_bid moving with BTC price?
print("\n" + "=" * 78)
print("AUDIT B — Is up_bid moving at all in the seconds before fill?")
print("=" * 78)
# For each fill, look at the previous 5 snapshots of the SAME market and see if up_bid changed
# This requires grouping by market and looking back in time
fill_market_set = {market[i] for i in fills}
# Build per-market sorted snapshot indices
per_market: dict[str, list[int]] = {}
for i, mk in enumerate(market):
    per_market.setdefault(str(mk), []).append(i)
# Already sorted because df sorted by (market_slug, snapshot_ts_ns)

stale_count_5s = 0
stale_count_10s = 0
bid_change_5s_list = []
for i in fills:
    mk = str(market[i])
    idxs = per_market[mk]
    pos = idxs.index(i)
    ts_now = snap_ts[i]
    bid_now = up_bid[i]
    # Walk back to find snapshot ~5s before
    snap_5s_ago_bid = bid_now
    for j in range(pos - 1, max(-1, pos - 30), -1):
        ts_back = snap_ts[idxs[j]]
        if (ts_now - ts_back) >= 5e9:
            snap_5s_ago_bid = up_bid[idxs[j]]
            break
    if snap_5s_ago_bid == bid_now: stale_count_5s += 1
    bid_change_5s_list.append(abs(bid_now - snap_5s_ago_bid))
print(f"  Filled snapshots where up_bid was IDENTICAL 5s before: {stale_count_5s}/{len(fills)} "
      f"({stale_count_5s/len(fills)*100:.1f}%)")
arr = np.array(bid_change_5s_list)
print(f"  abs(up_bid_now - up_bid_5s_ago): median={np.median(arr):.4f}  p90={np.quantile(arr, 0.9):.4f}")

# AUDIT C: LATENCY STRESS — what if we have to use up_bid from 1s/3s/5s later?
print("\n" + "=" * 78)
print("AUDIT C — LATENCY STRESS")
print("=" * 78)
print("Simulating: order placed at snapshot_ts but executes against book at snapshot_ts + L seconds.")
print("If strategy collapses with even 1-2s latency, the 'edge' is stale-quote arbitrage.")

from backtest.fees import FeeCalculator
date_arr = df["date"].to_numpy()

for latency_s in [0, 1, 3, 5, 10]:
    pnl_sum = 0.0
    wins = 0; losses = 0; skipped = 0; n = 0
    signal_persisted = 0
    for i in fills:
        mk = str(market[i])
        idxs = per_market[mk]
        pos = idxs.index(i)
        ts_now = snap_ts[i]
        # Find snapshot >= ts_now + latency
        future_pos = None
        for j in range(pos + 1, min(pos + 200, len(idxs))):
            if (snap_ts[idxs[j]] - ts_now) >= latency_s * 1e9:
                future_pos = j; break
        if latency_s > 0 and future_pos is None:
            skipped += 1; continue
        if latency_s == 0:
            future_pos = pos
        future_i = idxs[future_pos]
        new_up_bid = up_bid[future_i]
        new_up_ask = up_ask[future_i]
        new_down_ask = 1.0 - new_up_bid
        # Would we still place the order at this future time? (signal must persist)
        if (new_up_bid - p_up[future_i]) < 0.10:
            # Signal evaporated — assume we don't fill
            continue
        signal_persisted += 1
        won = (resolved[i] == 0)
        fc = FeeCalculator.for_date(str(date_arr[i]))
        fee = fc.taker_fee_usd(price=float(new_down_ask), size=1.0)
        pnl = (1.0 - new_down_ask) if won else (-new_down_ask)
        pnl -= fee
        pnl_sum += pnl
        n += 1
        if pnl > 0: wins += 1
        elif pnl < 0: losses += 1
    persist_rate = signal_persisted / max(1, len(fills) - skipped) * 100
    print(f"  latency = {latency_s:>2}s: filled={n}  skipped(no_future_data)={skipped}  "
          f"signal_still_valid={persist_rate:>5.1f}%  win%={wins/max(1,n)*100:>5.1f}  "
          f"total_pnl=${pnl_sum:+.2f}  per-fill=${pnl_sum/max(1,n):+.4f}")

# AUDIT D: BTC price velocity at fill
print("\n" + "=" * 78)
print("AUDIT D — Is the strategy reading BTC momentum the market hasn't priced in?")
print("=" * 78)
for c in ["btc_return_1s", "btc_return_3s", "btc_return_5s", "btc_return_10s"]:
    if c not in df.columns: continue
    arr = df[c].to_numpy()
    arr_fills = arr[fills]
    valid = np.isfinite(arr_fills)
    if valid.sum() == 0: continue
    print(f"  {c}: median(all)={np.nanmedian(arr):+.5f}  median(fills)={np.nanmedian(arr_fills[valid]):+.5f}  "
          f"mean(fills)={np.nanmean(arr_fills[valid]):+.5f}")
print("(If fill-conditioned mean is non-zero, the strategy fires preferentially on BTC trending one way.)")

print("\n--- DONE ---")
