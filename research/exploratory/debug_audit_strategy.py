#!/usr/bin/env python3
"""Honest audit of the Model 02 DOWN taker strategy.

Looking for:
1. Data leakage in Model 02 features (any field that uses post-snapshot info)
2. Stale/glitched quotes (bid > ask, crossed books, ridiculous spreads)
3. Already-resolved markets (snapshots where outcome is essentially determined)
4. Liquidity check (is the displayed down_ask backed by real depth?)
5. Spread distribution on filled rows (are we trading wide-spread anomalies?)
6. Entry price vs t_to_close (do entries get cheaper as resolution gets closer?)
7. Sanity: snapshot_ts_ns vs market_close_ts_ns gap matches t_to_close_s
"""
import io, sys, json, math
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features

OOS = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10","2026-05-11",
       "2026-05-12","2026-05-13","2026-05-14","2026-05-15","2026-05-16"]

# Load Model 02 features
feat_path = Path('artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/feature_importance.json')
feats = list(json.loads(feat_path.read_text()).keys())
model = joblib.load('artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl')

print("=" * 78)
print("AUDIT 1 — Model 02's 72 features (looking for future-info leakage)")
print("=" * 78)

# Classify each feature
suspicious_keywords = ["close", "resolv", "outcome", "final", "future", "after", "hold_to",
                       "flip_happened", "terminal", "next_", "_at_t", "settled", "_t_plus"]
neutral_keywords = ["snapshot_", "minute_utc", "second_of_day", "hour_utc", "day_of_week",
                    "live_", "spot_", "perp_", "binance_", "hyperliquid_",
                    "delta_to_strike", "abs_delta_to_strike", "price_to_beat",
                    "best_bid", "best_ask", "mid", "spread", "depth", "imbalance",
                    "_updates_", "_change_updates_", "time_spent_",
                    "up_token_", "down_token_", "complete", "without_token"]

suspicious = []
neutral = []
unknown = []
for f in feats:
    flow = f.lower()
    if any(k in flow for k in suspicious_keywords):
        suspicious.append(f)
    elif any(k in flow for k in neutral_keywords):
        neutral.append(f)
    else:
        unknown.append(f)

print(f"\nClassified {len(feats)} features:")
print(f"  Suspicious (future-info-like): {len(suspicious)}")
for f in suspicious: print(f"    - {f}")
print(f"\n  Likely safe (snapshot-time):   {len(neutral)}")
print(f"  Needs manual review:           {len(unknown)}")
for f in unknown[:25]: print(f"    - {f}")

# Now check t_to_close_s feature — is it in features? Yes that's a snapshot-time field,
# t_to_close shrinks as time progresses, model uses it. Not leakage; it's known at snap.
# But `time_spent_above_strike_recent_s` — what's "recent"? If recent extends post-snap, that's leakage.

# Load a sample to verify
print("\n" + "=" * 78)
print("AUDIT 2 — Spread / book sanity on filled trades")
print("=" * 78)

# Re-create the filter and look at filled trades
parts = []
for d in OOS:
    df = pl.read_parquet(f'data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet')
    df = df.filter(pl.col("resolved_side_label").is_not_null())
    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.filter(pl.col("up_token_best_ask") > 0.01).filter(pl.col("up_token_best_bid") < 0.99)
    if len(df) > 30000: df = df.sample(n=30000, seed=42)
    df = df.with_columns(pl.lit(d).alias("date"))
    parts.append(df)
df = pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")
df = clean_features(df)

# Predict P(UP)
X = np.zeros((len(df), len(feats)), dtype=np.float32)
for i, f in enumerate(feats):
    if f in df.columns:
        s = df.get_column(f)
        if s.dtype.is_numeric():
            v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
            X[:, i] = np.where(np.isfinite(v), v, 0.0)
p_up = model.predict_proba(X)[:, 1]
cal = getattr(model, "_calibrator", None)
if cal is not None: p_up = np.clip(cal.predict(p_up), 1e-6, 1-1e-6)

up_bid = df["up_token_best_bid"].to_numpy()
up_ask = df["up_token_best_ask"].to_numpy()
up_spread = up_ask - up_bid
ttc = df["t_to_close_s"].to_numpy()
market = df["market_slug"].to_numpy()
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid
resolved = df["resolved_side_label"].to_numpy()

# Filter: DOWN-side fills, t_to_close 10-60s, edge >= 0.10, dedupe per market
mask = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
idx = np.where(mask)[0]
seen = set(); fills = []
for i in idx:
    k = (str(market[i]), "DOWN")
    if k in seen: continue
    seen.add(k); fills.append(i)

print(f"\nTotal filled trades: {len(fills)}")

# Spread analysis
up_sp_fills = up_spread[fills]
print(f"\nup_token_spread on filled trades:")
print(f"  min/p10/p25/median/p75/p90/max: ${up_sp_fills.min():.4f} / ${np.quantile(up_sp_fills, 0.1):.4f} / "
      f"${np.quantile(up_sp_fills, 0.25):.4f} / ${np.median(up_sp_fills):.4f} / "
      f"${np.quantile(up_sp_fills, 0.75):.4f} / ${np.quantile(up_sp_fills, 0.9):.4f} / ${up_sp_fills.max():.4f}")
print(f"  mean: ${up_sp_fills.mean():.4f}")
print(f"  >$0.10 spread (wide): {(up_sp_fills > 0.10).sum()} / {len(up_sp_fills)} ({(up_sp_fills > 0.10).mean()*100:.1f}%)")
print(f"  >$0.20 spread (very wide): {(up_sp_fills > 0.20).sum()} / {len(up_sp_fills)} ({(up_sp_fills > 0.20).mean()*100:.1f}%)")

# Crossed / zero spread
crossed = (up_bid >= up_ask)[fills].sum()
print(f"  Crossed book (bid >= ask): {crossed}")

# Audit 3: already-resolved markets
print("\n" + "=" * 78)
print("AUDIT 3 — Already-resolved markets (cheap-entry, near-certain outcome)")
print("=" * 78)
down_ask_fills = down_ask[fills]
print(f"down_ask on filled trades (avg entry price):")
print(f"  <$0.05 (deep OTM, near-certain DOWN): {(down_ask_fills < 0.05).sum()} / {len(down_ask_fills)} "
      f"({(down_ask_fills < 0.05).mean()*100:.1f}%)")
print(f"  <$0.10: {(down_ask_fills < 0.10).sum()} ({(down_ask_fills < 0.10).mean()*100:.1f}%)")
print(f"  <$0.20: {(down_ask_fills < 0.20).sum()} ({(down_ask_fills < 0.20).mean()*100:.1f}%)")
print(f"  >$0.50: {(down_ask_fills > 0.50).sum()} ({(down_ask_fills > 0.50).mean()*100:.1f}%)")

# Win rate by entry-price band
print(f"\nWin rate by entry-price band:")
for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.0)]:
    m = (down_ask_fills >= lo) & (down_ask_fills < hi)
    if m.sum() == 0: continue
    wins = ((resolved[fills][m] == 0)).mean()
    print(f"  entry [{lo:.2f}, {hi:.2f}): n={m.sum():>4}  win%={wins*100:.1f}  realized_DOWN_avg={resolved[fills][m].mean():.2f}")

# Audit 4: liquidity check — depth at down_ask
print("\n" + "=" * 78)
print("AUDIT 4 — Liquidity at down_ask (depth on the side we'd lift)")
print("=" * 78)
# down_ask depth = up_bid depth (the bid we'd cross to buy DOWN ≈ sell UP into bid)
# Wait — to buy DOWN we lift DOWN's ask. The depth IS on down_token_ask side.
if "down_token_ask_depth_total" in df.columns:
    dn_depth = df["down_token_ask_depth_total"].to_numpy()
    dn_depth_fills = dn_depth[fills]
    print(f"down_token_ask_depth_total at fill snapshots:")
    print(f"  min/p10/p50/p90/max: {dn_depth_fills.min():.0f} / {np.quantile(dn_depth_fills, 0.1):.0f} / "
          f"{np.median(dn_depth_fills):.0f} / {np.quantile(dn_depth_fills, 0.9):.0f} / {dn_depth_fills.max():.0f}")
    print(f"  <1 contract depth: {(dn_depth_fills < 1).sum()} ({(dn_depth_fills < 1).mean()*100:.1f}%)")
    print(f"  <10 contracts: {(dn_depth_fills < 10).sum()} ({(dn_depth_fills < 10).mean()*100:.1f}%)")
    print(f"  >=100 contracts: {(dn_depth_fills >= 100).sum()} ({(dn_depth_fills >= 100).mean()*100:.1f}%)")
else:
    print("  WARN no down_token_ask_depth_total column")

# Audit 5: timing sanity
print("\n" + "=" * 78)
print("AUDIT 5 — Timestamp sanity (snapshot_ts + t_to_close_s == market_close_ts?)")
print("=" * 78)
if "snapshot_ts_ns" in df.columns and "market_close_ts_ns" in df.columns:
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)[fills]
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)[fills]
    ttc_fills = ttc[fills]
    derived_ttc = (close_ts - snap_ts) / 1e9
    err = derived_ttc - ttc_fills
    print(f"abs error between (close_ts - snap_ts)/1e9 and t_to_close_s: ")
    print(f"  median {np.median(np.abs(err)):.3f}s, p90 {np.quantile(np.abs(err), 0.9):.3f}s, max {np.max(np.abs(err)):.3f}s")
    # If max error is small (< 1s), timestamps are consistent

# Audit 6: snapshot_ts vs current time at decision — no future info
print("\n" + "=" * 78)
print("AUDIT 6 — Does Model 02's 'time_spent_above_strike_recent_s' look future-leaky?")
print("=" * 78)
if "time_spent_above_strike_recent_s" in df.columns:
    tsab = df["time_spent_above_strike_recent_s"].to_numpy()[fills]
    print(f"time_spent_above_strike_recent_s on fills:")
    print(f"  min/median/max: {tsab.min():.1f} / {np.median(tsab):.1f} / {tsab.max():.1f} seconds")
    # Compare to t_to_close: if "recent" looks back N seconds, it can't be > (snapshot_age) seconds
    # Check if any tsab > t_since_open
    if "t_since_open_s" in df.columns:
        ts_open = df["t_since_open_s"].to_numpy()[fills]
        violations = (tsab > ts_open + 1).sum()   # +1s tolerance
        print(f"  cases where time_spent > t_since_open (would be impossible without future info): {violations}/{len(tsab)}")

    # Also check: if "recent" includes future, it'd correlate with realized outcome strongly
    resolved_arr = resolved[fills]
    corr_with_resolved = np.corrcoef(tsab, resolved_arr)[0,1]
    print(f"  corr(time_spent_above_strike_recent_s, resolved_side_label): {corr_with_resolved:+.3f}")
    print(f"  → If this is very high (>0.7), feature reveals the outcome.")

# Audit 7: is `price_to_beat` future-info?
print("\n" + "=" * 78)
print("AUDIT 7 — Does price_to_beat encode future BTC price?")
print("=" * 78)
if "price_to_beat" in df.columns:
    ptb = df["price_to_beat"].to_numpy()[fills]
    print(f"price_to_beat on fills: min/median/max: {ptb.min():.2f} / {np.median(ptb):.2f} / {ptb.max():.2f}")
    # price_to_beat looks like a market strike. Should not be future-conditional.
    resolved_arr = resolved[fills]
    if "binance_spot_mid" in df.columns:
        spot = df["binance_spot_mid"].to_numpy()[fills]
        # On a UP market, resolves=1 iff spot at close > strike. snapshot spot would only correlate
        # if it's near the close (high t_to_close stationarity).
        side_above_strike = (spot > ptb).astype(int)
        agree = (side_above_strike == resolved_arr).mean()
        print(f"  snapshot binance_spot_mid > price_to_beat → resolved_side_label=1?  agree: {agree*100:.1f}%")
        print(f"  (If ~99%+, model is just reading the answer off current spot vs strike, not predicting.)")
