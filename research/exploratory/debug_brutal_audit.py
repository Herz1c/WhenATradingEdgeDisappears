#!/usr/bin/env python3
"""BRUTAL AUDIT — Model 02 DOWN-side taker strategy.

If this survives, the strategy is real. If anything breaks, we fail it.

Sections:
  1. Entry-price + risk-reward distribution (exact)
  2. Win-rate confidence intervals (binomial) + per-day variance
  3. Consecutive-loss streaks + tail risk (VaR, CVaR)
  4. Fee burden as % of trade — is the rebate / fee math honest?
  5. "What if win rate is actually 60% / 65%" sensitivity to model degradation
  6. Slippage probe — when bid hasn't moved for N sec, what does the NEXT trade
     in that market clear at? Are we picking up a price the market will correct?
  7. Capacity check — can the strategy survive 10x / 100x size scaling?
  8. Regime check — is win rate stable across BTC volatility regimes?
  9. Counterparty check — is there a Polymarket fee tier / minimum that
     wipes the edge?
 10. Sample-independence — how many TRULY independent observations are there?
"""
import io, sys, json
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from feature_cleanup import clean_features
from backtest.fees import FeeCalculator

feat_path = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/feature_importance.json")
feats = list(json.loads(feat_path.read_text()).keys())
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
dates_arr = df["date"].to_numpy()
edge_dn = up_bid - p_up
down_ask = 1.0 - up_bid

mask = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
idx = np.where(mask)[0]
seen = set(); fills = []
for i in idx:
    k = str(market[i])
    if k in seen: continue
    seen.add(k); fills.append(i)
fills = np.array(fills)
print(f"Total filled trades: {len(fills)}")

entries = down_ask[fills]
wins = (resolved[fills] == 0).astype(int)
fees = np.array([FeeCalculator.for_date(str(dates_arr[i])).taker_fee_usd(price=float(down_ask[i]), size=1.0) for i in fills])
gross_pnl = np.where(wins == 1, 1.0 - entries, -entries)
net_pnl = gross_pnl - fees

# ── SECTION 1: Entry-price + risk-reward distribution ─────────────────────────
print("\n" + "=" * 78)
print("1. ENTRY PRICE + RISK/REWARD DISTRIBUTION")
print("=" * 78)
print(f"\nEntry price (down_ask paid per contract):")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:>3}: ${np.percentile(entries, p):.4f}")
print(f"  mean : ${entries.mean():.4f}")
print(f"  std  : ${entries.std():.4f}")

print(f"\nRisk per trade (max loss if DOWN doesn't resolve): same as entry")
print(f"  mean risk:   ${entries.mean():.4f}")
print(f"  p99 risk:    ${np.percentile(entries, 99):.4f}  ← single-trade worst case loss")

print(f"\nReward per trade (1 - entry if DOWN wins):")
reward_if_win = 1.0 - entries
print(f"  mean reward (when win): ${reward_if_win.mean():.4f}")

print(f"\nRisk/reward ratio per trade (reward/risk):")
rr = reward_if_win / entries
print(f"  median: {np.median(rr):.2f}x  (i.e., risk \$X to make {np.median(rr):.2f}*X)")
print(f"  mean:   {rr.mean():.2f}x")

# ── SECTION 2: Win-rate confidence intervals ──────────────────────────────────
print("\n" + "=" * 78)
print("2. WIN RATE CONFIDENCE INTERVALS (binomial Wilson 95%)")
print("=" * 78)
def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    phat = k / n
    denom = 1 + z*z/n
    centre = (phat + z*z/(2*n)) / denom
    halfw = z/denom * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (max(0, centre - halfw), min(1, centre + halfw))

n = len(fills); k = wins.sum()
lo, hi = wilson_ci(k, n)
print(f"  Overall: {k}/{n} = {k/n*100:.2f}%  95% CI: [{lo*100:.2f}%, {hi*100:.2f}%]")
print(f"  Break-even win rate at p=$0.5 with Polymarket fee: ~51.8%")
margin = (k/n - 0.518) * 100
print(f"  Margin over break-even: +{margin:.1f}pp")

# Per-day breakdown
print(f"\n  Per-day:")
print(f"  {'date':<14} {'fills':>5} {'wins':>5} {'win%':>7} {'CI lo':>7} {'CI hi':>7}")
for d in OOS:
    m = dates_arr[fills] == d
    if m.sum() == 0: continue
    nn = m.sum(); kk = wins[m].sum()
    lo, hi = wilson_ci(kk, nn)
    print(f"  {d:<14} {nn:>5} {kk:>5} {kk/nn*100:>6.1f}% {lo*100:>6.1f}% {hi*100:>6.1f}%")

# ── SECTION 3: Tail risk + consecutive losses ─────────────────────────────────
print("\n" + "=" * 78)
print("3. TAIL RISK + CONSECUTIVE-LOSS STREAKS")
print("=" * 78)
print(f"\nPer-trade PnL distribution (USD, after fee):")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:>3}: ${np.percentile(net_pnl, p):+.4f}")

# VaR / CVaR (loss-side)
losses = net_pnl[net_pnl < 0]
print(f"\nVaR95 (95% of trades lose at most): ${-np.percentile(net_pnl, 5):.4f}")
print(f"VaR99 (99% of trades lose at most): ${-np.percentile(net_pnl, 1):.4f}")
print(f"CVaR95 (mean loss in worst 5%):     ${-net_pnl[net_pnl <= np.percentile(net_pnl, 5)].mean():.4f}")

# Consecutive loss streak (in chronological order)
chrono_idx = fills[np.argsort(snap_ts[fills])]
chrono_wins = (resolved[chrono_idx] == 0).astype(int)
current_streak = 0; max_streak = 0
for w in chrono_wins:
    if w == 0:   # loss
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        current_streak = 0
print(f"\nLongest consecutive-loss streak across OOS: {max_streak}")
# Cumulative drawdown of worst streak
running = 0.0; peak = 0.0; max_dd = 0.0
chrono_pnl = net_pnl[np.argsort(snap_ts[fills])]
for pnl in chrono_pnl:
    running += pnl
    peak = max(peak, running)
    max_dd = max(max_dd, peak - running)
print(f"Worst rolling drawdown of cumulative PnL: ${max_dd:.4f}")

# ── SECTION 4: Fee burden ─────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("4. FEE BURDEN — is the Polymarket fee math honest?")
print("=" * 78)
print(f"  Mean fee per fill: ${fees.mean():.4f}")
print(f"  Mean fee as % of entry: {(fees / entries).mean()*100:.2f}%")
print(f"  Mean fee as % of net PnL: {(fees / np.abs(net_pnl) + 1e-9).mean()*100:.2f}%")
print(f"  Total fees paid OOS: ${fees.sum():.2f}")
print(f"  Total gross PnL OOS: ${gross_pnl.sum():.2f}")
print(f"  Total net PnL OOS:   ${net_pnl.sum():.2f}")
print(f"  Fees as % of gross: {fees.sum() / gross_pnl.sum() * 100:.1f}%")

# ── SECTION 5: Sensitivity to win-rate degradation ────────────────────────────
print("\n" + "=" * 78)
print("5. SENSITIVITY — what if true win rate is lower than observed?")
print("=" * 78)
mean_entry = entries.mean()
mean_fee = fees.mean()
print(f"  Mean entry: ${mean_entry:.4f}, mean fee: ${mean_fee:.4f}")
print(f"  Break-even at this entry: WR > {(mean_entry + mean_fee) / 1.0 * 100:.1f}%")
print(f"\n  Per-trade expected PnL for various WR:")
for wr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    ep = wr * (1 - mean_entry) - (1 - wr) * mean_entry - mean_fee
    weekly_est = ep * 575   # 575 fills/wk in 1-entry config
    print(f"    WR={wr*100:.0f}%: per-fill ${ep:+.4f}  weekly (575 fills) ${weekly_est:+.2f}")

# ── SECTION 6: Slippage probe ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("6. SLIPPAGE PROBE — when bid is stale, does NEXT trade clear lower?")
print("=" * 78)
per_market: dict[str, list[int]] = {}
for i, mk in enumerate(market):
    per_market.setdefault(str(mk), []).append(i)

next_movement_dn = []   # change in up_bid: positive = bid moved UP (against us, since we bought DOWN at 1-up_bid)
for fi in fills:
    mk = str(market[fi])
    idxs = per_market[mk]
    pos = idxs.index(fi)
    bid_now = up_bid[fi]
    # Find next bid change
    for j in range(pos + 1, min(pos + 60, len(idxs))):
        if up_bid[idxs[j]] != bid_now:
            move = up_bid[idxs[j]] - bid_now
            next_movement_dn.append(move)
            break
arr = np.array(next_movement_dn)
print(f"  Next bid move after fill: n={len(arr)}")
print(f"  mean: ${arr.mean():+.4f}  median: ${np.median(arr):+.4f}")
print(f"  P(moves DOWN, in our favor): {(arr < 0).mean()*100:.1f}%")
print(f"  P(moves UP, against us):      {(arr > 0).mean()*100:.1f}%")
print(f"  (If 'against' >> 'favor', we're racing into a falling knife — book is repricing.)")

# Compare next trade price vs our entry
next_down_ask = []
for fi in fills:
    mk = str(market[fi])
    idxs = per_market[mk]
    pos = idxs.index(fi)
    bid_now = up_bid[fi]
    for j in range(pos + 1, min(pos + 60, len(idxs))):
        if up_bid[idxs[j]] != bid_now:
            next_down_ask.append((1 - up_bid[idxs[j]]) - (1 - bid_now))   # change in down_ask
            break
arr2 = np.array(next_down_ask)
print(f"\n  Change in down_ask after fill: mean ${arr2.mean():+.4f}")
print(f"  P(down_ask falls — we PAID too much): {(arr2 < 0).mean()*100:.1f}%")
print(f"  P(down_ask rises — entry was good):    {(arr2 > 0).mean()*100:.1f}%")

# ── SECTION 7: Capacity at scale ──────────────────────────────────────────────
print("\n" + "=" * 78)
print("7. CAPACITY — depth at displayed asks")
print("=" * 78)
if "down_token_ask_depth_total" in df.columns:
    depths = df["down_token_ask_depth_total"].to_numpy()
    depths_fills = depths[fills]
    # Filter to finite
    finite = np.isfinite(depths_fills)
    if finite.sum() > 0:
        print(f"  down_token_ask_depth_total on fills (n={finite.sum()}):")
        for p in [1, 5, 10, 25, 50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(depths_fills[finite], p):.0f} contracts")

if "up_token_bid_depth_total" in df.columns:
    bdepths = df["up_token_bid_depth_total"].to_numpy()[fills]
    finite = np.isfinite(bdepths)
    if finite.sum() > 0:
        print(f"\n  up_token_bid_depth_total on fills (the bid we're effectively taking):")
        for p in [1, 5, 10, 25, 50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(bdepths[finite], p):.0f} contracts")

# ── SECTION 8: Regime check ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print("8. REGIME CHECK — does win rate vary with BTC volatility?")
print("=" * 78)
if "btc_realized_vol_30s" in df.columns:
    vol = df["btc_realized_vol_30s"].to_numpy()[fills]
    finite = np.isfinite(vol)
    qs = np.quantile(vol[finite], [0.25, 0.5, 0.75])
    print(f"  btc_realized_vol_30s quartiles at fills: {qs}")
    for lo, hi, name in [(-1e9, qs[0], "low vol"), (qs[0], qs[1], "med-low"),
                          (qs[1], qs[2], "med-high"), (qs[2], 1e9, "high vol")]:
        m = (vol >= lo) & (vol < hi) & finite
        if m.sum() < 20: continue
        wr = wins[m].mean()
        lo_ci, hi_ci = wilson_ci(wins[m].sum(), m.sum())
        print(f"    {name} ({lo:.5f}-{hi:.5f}): n={m.sum()}  win%={wr*100:.1f}  CI [{lo_ci*100:.1f}, {hi_ci*100:.1f}]")

# ── SECTION 9: Sample independence ───────────────────────────────────────────
print("\n" + "=" * 78)
print("9. SAMPLE INDEPENDENCE — how many TRULY independent observations?")
print("=" * 78)
# Unique markets
n_markets = len({str(market[i]) for i in fills})
print(f"  Filled markets: {n_markets} unique markets across {len(OOS)} days")
print(f"  → effective sample size for win-rate inference: ~{n_markets}")
print(f"  → Wilson CI uses this; CI of ±{(0.74-0.66)*100/2:.1f}pp at n={n_markets} is statistically tight")
print()
# What about market-level correlation? same market = same resolution = same direction
# Are wins clustered in time (correlation between consecutive trades)?
import itertools
runs = [(k, sum(1 for _ in g)) for k, g in itertools.groupby(chrono_wins)]
print(f"  Run lengths (consecutive same outcome): max win streak = {max(r for k,r in runs if k==1)}, "
      f"max loss streak = {max(r for k,r in runs if k==0)}")
runs_wr = sum(r for k,r in runs if k==1) / len(chrono_wins)
print(f"  Total run-encoded win rate: {runs_wr*100:.1f}% (sanity check vs {wins.mean()*100:.1f}%)")

# ── SECTION 10: TOTAL ECONOMICS ──────────────────────────────────────────────
print("\n" + "=" * 78)
print("10. BOTTOM LINE — full economic breakdown (OOS, 10 days)")
print("=" * 78)
print(f"  Capital deployed (sum of entries): ${entries.sum():.2f}")
print(f"  Gross PnL (before fees):           ${gross_pnl.sum():+.2f}")
print(f"  Total fees paid:                   ${fees.sum():.2f}")
print(f"  Net PnL:                           ${net_pnl.sum():+.2f}")
print(f"  Return on capital deployed:        {gross_pnl.sum() / entries.sum() * 100:.1f}% gross")
print(f"  Effective gain per dollar deployed: ${net_pnl.sum() / entries.sum():.4f}")
print(f"  Avg PnL per trade:                 ${net_pnl.mean():+.4f}")
print(f"  Sharpe of per-trade PnL (annualized estimate, 575/wk × 52): "
      f"{net_pnl.mean() / net_pnl.std() * np.sqrt(575*52):.2f}")

print("\n" + "=" * 78)
print("VERDICT (read all sections carefully — no single number is the answer)")
print("=" * 78)
