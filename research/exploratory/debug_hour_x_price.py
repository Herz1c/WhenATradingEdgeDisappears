#!/usr/bin/env python3
"""Joint distribution: hour-of-day × entry-price-band → win rate.

The 90% WR at US-evening peak hours doesn't mean 90% on $0.50 entries.
It probably means the strategy fires on CHEAPER entries during those hours
(where retail is pricing DOWN at $0.20-0.30 because they're chasing UP).
"""
import io, sys, json
from datetime import datetime, timezone
from pathlib import Path

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
entry = (1.0 - up_bid[fills])  # down_ask paid
ts_fills = snap_ts[fills]
hours = np.array([datetime.fromtimestamp(t/1e9, tz=timezone.utc).hour for t in ts_fills])

# US-peak hours
us_peak = (hours >= 22) | (hours <= 4)
asia_morn = (hours == 5)

# ── 1. Entry price distribution by hour bucket ───────────────────────────────
print("=" * 78)
print("1. ENTRY PRICE DISTRIBUTION BY HOUR BUCKET")
print("=" * 78)
for label, m in [("All OOS",        np.ones(len(fills), dtype=bool)),
                 ("US-peak (22-4 UTC)", us_peak),
                 ("Asia morn (5 UTC)", asia_morn),
                 ("Other hours",    ~us_peak & ~asia_morn)]:
    if m.sum() == 0: continue
    e = entry[m]
    print(f"\n  {label}  (n={m.sum()}):")
    print(f"    p10/p25/median/p75/p90: ${np.percentile(e,10):.3f} / ${np.percentile(e,25):.3f} / "
          f"${np.median(e):.3f} / ${np.percentile(e,75):.3f} / ${np.percentile(e,90):.3f}")
    print(f"    mean: ${e.mean():.3f}  std: ${e.std():.3f}")

# ── 2. Joint table: hour × entry-price-band → win rate ───────────────────────
print("\n" + "=" * 78)
print("2. JOINT WIN-RATE TABLE — hour × entry-price band")
print("=" * 78)
bands = [(0.00, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 1.00)]
hour_buckets = [
    ("US-peak (22-04)", us_peak),
    ("EU-peak (18-21)", (hours >= 18) & (hours <= 21)),
    ("EU/US-day (10-17)", (hours >= 10) & (hours <= 17)),
    ("Asia (05-09)",     (hours >= 5) & (hours <= 9)),
]
print(f"{'band':<14} {'US-peak':>14} {'EU-peak':>14} {'EU/US-day':>14} {'Asia':>14}")
for lo, hi in bands:
    band_m = (entry >= lo) & (entry < hi)
    parts_row = [f"[{lo:.2f}, {hi:.2f})"]
    for label, hm in hour_buckets:
        m = band_m & hm
        if m.sum() < 5:
            parts_row.append(f"{'n=' + str(m.sum()):>14}")
        else:
            wr = wins[m].mean() * 100
            parts_row.append(f"{wr:>5.1f}% (n={m.sum():>3})")
    print(f"  {parts_row[0]:<12} {parts_row[1]:>14} {parts_row[2]:>14} {parts_row[3]:>14} {parts_row[4]:>14}")

# ── 3. The 90% WR fillings — what entries drove them? ────────────────────────
print("\n" + "=" * 78)
print("3. WHERE IS THE 88-92% WIN RATE COMING FROM?")
print("=" * 78)
hours_top = [0, 23]   # US evening peak hours
for h in hours_top:
    m = hours == h
    if m.sum() == 0: continue
    e = entry[m]
    w = wins[m]
    print(f"\n  Hour {h:02d}:00 UTC (n={m.sum()}, overall WR={w.mean()*100:.1f}%):")
    print(f"    entry distribution: p25=${np.percentile(e,25):.3f}  med=${np.median(e):.3f}  p75=${np.percentile(e,75):.3f}")
    # Sub-buckets within this hour
    for lo, hi in bands:
        bm = (e >= lo) & (e < hi)
        if bm.sum() < 2: continue
        print(f"      entry [{lo:.2f}, {hi:.2f}): n={bm.sum()}  wins={w[bm].sum()}  WR={w[bm].mean()*100:.1f}%")

# ── 4. What does up_bid look like in the high-WR cheap-entry cells? ──────────
print("\n" + "=" * 78)
print("4. MICRO-EXAMPLES — low entry-price cells (where 88%+ WR concentrates)")
print("=" * 78)
m = (entry < 0.30)
print(f"  Filled trades with entry < $0.30: n={m.sum()}, WR={wins[m].mean()*100:.1f}%")
if m.sum() > 0:
    print(f"  up_bid range here: {up_bid[fills][m].min():.3f} - {up_bid[fills][m].max():.3f}")
    print(f"  model_p_up range:  {p_up[fills][m].min():.3f} - {p_up[fills][m].max():.3f}")
    print(f"  edge_dn (up_bid - p_up) range: {(up_bid[fills][m] - p_up[fills][m]).min():.3f} - "
          f"{(up_bid[fills][m] - p_up[fills][m]).max():.3f}")
    print(f"  → Retail pricing UP at {up_bid[fills][m].mean():.3f}, model says P(UP)={p_up[fills][m].mean():.3f}")
    print(f"    Realized P(UP) for these: {(resolved[fills][m]==1).mean():.3f}")
