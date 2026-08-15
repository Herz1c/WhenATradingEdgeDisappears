#!/usr/bin/env python3
"""Exhaustive exploration of Model 02 as a pure taker signal.

Step 1: OOS calibration audit — is Model 02 well-calibrated against
        resolved_side_label? AUC 0.881 means good discrimination but
        calibration could still be off.

Step 2: Conditional accuracy — under what conditions does the model
        beat the market? Slice by:
        - t_to_close (more deterministic close to resolution)
        - market mid (efficiency varies with price level)
        - edge size (model's confidence over market)

Step 3: Strategy sweep — for each (threshold, t_to_close range, price band),
        compute win rate, per-trade PnL, after-fee PnL, total weekly PnL.
        Find any condition with >51.8% win rate (taker break-even at p=0.5).

Step 4: Best config — grade on 20 MC weeks from pure OOS pool.
"""
import io, json, sys, math, random, statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib

from backtest.fees import FeeCalculator
from backtest.grading import grade_strategy
from feature_cleanup import clean_features

# ── Load Model 02 ────────────────────────────────────────────────────────────

ART = "artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl"
FI = Path(ART).parent / "feature_importance.json"
model = joblib.load(ART)
feats = list(json.loads(FI.read_text()).keys())
print(f"Loaded Model 02 dense_close: {len(feats)} features")


def predict_p_up(df: pl.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def load_oos(dates, max_per_day=15_000, seed=42):
    parts = []
    for d in dates:
        p = Path(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
        if not p.exists(): continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        if max_per_day and len(df) > max_per_day:
            df = df.sample(n=max_per_day, seed=seed, with_replacement=False)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    return pl.concat(parts, how="diagonal") if parts else pl.DataFrame()


OOS_DATES = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
             "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
             "2026-05-15","2026-05-16"]


# ── Step 1: Calibration audit ────────────────────────────────────────────────

print("\n" + "=" * 78)
print("STEP 1: OOS CALIBRATION AUDIT")
print("=" * 78)

df = load_oos(OOS_DATES, max_per_day=15_000)
print(f"Loaded {len(df)} OOS snapshots")
df = clean_features(df)
p_up = predict_p_up(df)
y = df["resolved_side_label"].to_numpy().astype(int)
up_bid = df["up_token_best_bid"].to_numpy().astype(float)
up_ask = df["up_token_best_ask"].to_numpy().astype(float)
up_mid = (up_bid + up_ask) / 2
ttc = df["t_to_close_s"].to_numpy().astype(float)

# Bin predictions and check realized win rate
print(f"\nModel 02 calibration:")
print(f"{'pred bin':<14} {'n':>7} {'mean_pred':>11} {'realized P(UP)':>15} {'calib err':>12}")
print("-" * 65)
bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in bins:
    m = (p_up >= lo) & (p_up < hi)
    if m.sum() < 100: continue
    mean_p = p_up[m].mean()
    real_p = y[m].mean()
    err = mean_p - real_p
    print(f"[{lo:.1f}, {hi:.1f})    {m.sum():>7} {mean_p:>11.3f} {real_p:>15.3f} {err:>+12.3f}")

print(f"\nMarket calibration (compare up_token_mid to realized):")
print(f"{'mid bin':<14} {'n':>7} {'mean_mid':>11} {'realized P(UP)':>15} {'err':>10}")
print("-" * 60)
for lo, hi in bins:
    m = (up_mid >= lo) & (up_mid < hi)
    if m.sum() < 100: continue
    mean_p = up_mid[m].mean()
    real_p = y[m].mean()
    err = mean_p - real_p
    print(f"[{lo:.1f}, {hi:.1f})    {m.sum():>7} {mean_p:>11.3f} {real_p:>15.3f} {err:>+10.3f}")


# ── Step 2: Edge vs realized win rate ────────────────────────────────────────

print("\n" + "=" * 78)
print("STEP 2: WIN RATE BY (EDGE, T_TO_CLOSE)")
print("=" * 78)
print("For each (model_P - market_ask) edge band, compute realized win rate")
print("when we WOULD have bought UP token. Break-even at p=$0.5 needs >51.8% win rate.")

edge_up = p_up - up_ask     # model - what we'd pay to buy UP
edge_dn = up_bid - p_up     # what we'd receive selling UP - model_P_UP (i.e., buying DOWN)

print(f"\n{'edge band':<14} {'ttc band':<14} {'n_buy_UP':>10} {'win%':>8} {'n_buy_DN':>10} {'win%':>8}")
print("-" * 75)
for edge_lo in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
    for ttc_lo, ttc_hi in [(0, 30), (30, 60), (60, 120), (120, 300)]:
        mttc = (ttc >= ttc_lo) & (ttc < ttc_hi)
        # Buy UP
        m_up = (edge_up >= edge_lo) & mttc
        if m_up.sum() > 50:
            win_up = (y[m_up] == 1).mean()
        else:
            win_up = float('nan')
        # Buy DOWN
        m_dn = (edge_dn >= edge_lo) & mttc
        if m_dn.sum() > 50:
            win_dn = (y[m_dn] == 0).mean()
        else:
            win_dn = float('nan')
        n_up_s = m_up.sum() if m_up.sum() > 0 else 0
        n_dn_s = m_dn.sum() if m_dn.sum() > 0 else 0
        print(f"≥${edge_lo:.2f}      ttc {ttc_lo:>3}-{ttc_hi:<3}s   {n_up_s:>10} {win_up*100 if not math.isnan(win_up) else 0:>7.1f}%  {n_dn_s:>10} {win_dn*100 if not math.isnan(win_dn) else 0:>7.1f}%")
    print()


# ── Step 3: PnL/trade by condition (after taker fee) ─────────────────────────

print("=" * 78)
print("STEP 3: AFTER-FEE PnL PER TRADE (taker fee = Polymarket schedule)")
print("=" * 78)
# Precompute fees for each row
fee_cal = FeeCalculator.for_date("2026-05-09")   # all OOS days same schedule

print(f"{'edge band':<10} {'ttc band':<14} {'n':>6} {'win%':>7} {'avg_entry':>10} {'mean PnL':>12} {'net/trade':>11}")
print("-" * 80)
for edge_lo in [0.05, 0.08, 0.10, 0.15, 0.20]:
    for ttc_lo, ttc_hi in [(10, 60), (30, 90), (60, 180)]:
        mttc = (ttc >= ttc_lo) & (ttc < ttc_hi)
        m_up = (edge_up >= edge_lo) & mttc
        m_dn = (edge_dn >= edge_lo) & mttc
        m = m_up | m_dn
        if m.sum() < 100: continue
        entries = np.where(m_up, up_ask, up_bid)  # buy UP at ask, buy DOWN at (1-up_bid) ~ down_ask
        # Approximate: buying DOWN at down_ask, which is approximately (1 - up_bid)
        # For simplicity treat entry as up_ask for UP, (1 - up_bid) for DOWN
        down_ask = 1.0 - up_bid
        entries[m_dn] = down_ask[m_dn]
        # Realized payoff
        won = np.where(m_up, y == 1, y == 0)
        payoffs = np.where(won, 1.0, 0.0)
        # Fees (Polymarket: peak at p=0.5)
        fees = np.array([fee_cal.taker_fee_usd(price=e, size=1.0) for e in entries[m]])
        gross = payoffs[m] - entries[m]
        net = gross - fees
        win_rate = won[m].mean()
        print(f"≥${edge_lo:.2f}    ttc {ttc_lo:>3}-{ttc_hi:<3}s   {m.sum():>6} {win_rate*100:>6.1f}% "
              f"${entries[m].mean():>+9.4f} ${gross.mean():>+10.4f} ${net.mean():>+10.4f}")
    print()


# ── Step 4: Save best results for next step ──────────────────────────────────

# Find the best per-trade-PnL condition
print("\n" + "=" * 78)
print("STEP 4: SAVING DIAGNOSTICS")
print("=" * 78)

results = {
    "n_oos_snapshots": int(len(df)),
    "calibration_bins": [],
}
for lo, hi in bins:
    m = (p_up >= lo) & (p_up < hi)
    if m.sum() > 100:
        results["calibration_bins"].append({
            "bin": [lo, hi], "n": int(m.sum()),
            "mean_pred": float(p_up[m].mean()),
            "realized_p_up": float(y[m].mean()),
            "calibration_err": float(p_up[m].mean() - y[m].mean()),
        })
Path("docs").mkdir(exist_ok=True)
Path("docs/model02_taker_audit.json").write_text(json.dumps(results, indent=2))
print(f"Saved to docs/model02_taker_audit.json")
