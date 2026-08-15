#!/usr/bin/env python3
"""Sub-second latency analysis using bid LIFETIME distribution.

The dataset has 1-second-per-market resolution, so we can't directly probe
at 100-250ms. Instead we measure how long each up_bid stays at its current
value when our signal fires — if the median bid lifetime is multi-second,
then a 250ms latency captures it cleanly.

Method: for each filled snapshot, walk forward in same-market snapshots
and count how many seconds until the up_bid actually moves. That's the
"survival time" of the quote we'd be racing against.
"""
import io, sys, json
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
ttc = df["t_to_close_s"].to_numpy()
market = df["market_slug"].to_numpy()
snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
edge_dn = up_bid - p_up

# Apply our filter, dedupe per market
mask = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
idx = np.where(mask)[0]
seen = set(); fills = []
for i in idx:
    k = str(market[i])
    if k in seen: continue
    seen.add(k); fills.append(i)

# Build per-market sorted index list
per_market: dict[str, list[int]] = {}
for i, mk in enumerate(market):
    per_market.setdefault(str(mk), []).append(i)

# For each fill, find the time-until up_bid first CHANGES from current value
# (i.e., bid survival time after our signal)
survival_times_s = []
for i in fills:
    mk = str(market[i])
    idxs = per_market[mk]
    pos = idxs.index(i)
    ts_now = snap_ts[i]
    bid_now = up_bid[i]
    survival = None
    for j in range(pos + 1, min(pos + 300, len(idxs))):
        ix = idxs[j]
        if up_bid[ix] != bid_now:
            survival = (snap_ts[ix] - ts_now) / 1e9
            break
    # If never changed, mark as max observed
    if survival is None:
        if pos + 300 >= len(idxs):
            # Just cap at remaining horizon
            survival = (snap_ts[idxs[-1]] - ts_now) / 1e9 if idxs[-1] > i else 60.0
        else:
            survival = 60.0   # observed at least 5 minutes
    survival_times_s.append(survival)

surv = np.array(survival_times_s)
print(f"Bid SURVIVAL TIME (time until up_bid changes) on {len(fills)} filled snapshots:")
print(f"  min/p10/p25/median/p75/p90/p95/p99/max: ")
print(f"  {surv.min():.2f} / {np.quantile(surv,0.1):.2f} / {np.quantile(surv,0.25):.2f} / "
      f"{np.median(surv):.2f} / {np.quantile(surv,0.75):.2f} / {np.quantile(surv,0.9):.2f} / "
      f"{np.quantile(surv,0.95):.2f} / {np.quantile(surv,0.99):.2f} / {surv.max():.2f}  seconds")
print(f"  mean: {surv.mean():.2f}s")

print(f"\nSurvival at various latencies:")
for lat in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    rate = (surv >= lat).mean()
    print(f"  bid still at original price after {lat:>4.2f}s: {rate*100:>5.1f}% of fills")

print(f"\nIMPLICATION for your VPS (100-250ms tick-to-trade):")
p100 = (surv >= 0.1).mean()
p250 = (surv >= 0.25).mean()
print(f"  At 100ms latency: signal survival = {p100*100:.1f}%")
print(f"  At 250ms latency: signal survival = {p250*100:.1f}%")
print(f"  Paper-perfect PnL ~= $131/wk (1 entry/market) or $246/wk (2 entries, 10s gap)")
print(f"  Expected live PnL at 100-250ms: ${131*p250:.0f}-${131*p100:.0f}/wk (1 entry) "
      f"or ${246*p250:.0f}-${246*p100:.0f}/wk (2 entries)")
