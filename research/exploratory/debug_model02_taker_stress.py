#!/usr/bin/env python3
"""Stress tests for Model 02 DOWN-side taker strategy.

1) Per-day breakdown across May 7-16 (no MC sampling overlap).
2) Edge threshold sweep (higher thresholds should still be profitable).
3) Fee stress test (1.5× and 2× fees).
4) Compare in-sample (May 4-6, training overlap) vs OOS (May 7-16).
5) Random label test — shuffle resolved_side_label to confirm it goes to 0.
"""
import io, sys, json, statistics
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib

from backtest.model02_taker_engine import run_taker_backtest, _load_model, _predict_p_up
from backtest.fees import FeeCalculator
from feature_cleanup import clean_features

OOS_DATES = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
             "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
             "2026-05-15","2026-05-16"]
IS_DATES = ["2026-04-22","2026-04-26","2026-04-30","2026-05-04"]   # spread across training window

base_cfg = dict(
    edge_threshold_usd=0.10,
    allow_up_side=False, allow_dn_side=True,
    min_t_to_close_s=10, max_t_to_close_s=60,
    max_margin_usd=100.0, max_per_day=15_000,
)

# ── Test 1: per-day breakdown ────────────────────────────────────────────────
print("=" * 78)
print("TEST 1: PER-DAY BREAKDOWN (each day independently)")
print("=" * 78)
print(f"{'date':<14} {'fills':>7} {'win%':>7} {'pnl':>10} {'dd':>8}")
for d in OOS_DATES:
    s = run_taker_backtest([d], **base_cfg)
    print(f"{d:<14} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+6.2f}")

# ── Test 2: edge threshold sweep ─────────────────────────────────────────────
print("\n" + "=" * 78)
print("TEST 2: EDGE THRESHOLD SWEEP (on full OOS window, DOWN only)")
print("=" * 78)
print(f"{'edge':>6} {'fills':>7} {'win%':>7} {'pnl':>10} {'dd':>8} {'per-fill':>11}")
for thr in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    cfg = dict(base_cfg); cfg["edge_threshold_usd"] = thr
    s = run_taker_backtest(OOS_DATES, **cfg)
    pf = s['net_pnl_usd'] / s['n_fills'] if s['n_fills'] else 0
    print(f"≥${thr:.2f} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+6.2f} ${pf:>+10.4f}")

# ── Test 3: fee stress ───────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("TEST 3: FEE STRESS (full OOS window, base config)")
print("=" * 78)
print(f"{'fee_mult':>10} {'fills':>7} {'win%':>7} {'pnl':>10} {'per-fill':>11}")
for fm in [1.0, 1.2, 1.5, 2.0, 3.0]:
    cfg = dict(base_cfg); cfg["fee_scale"] = fm
    s = run_taker_backtest(OOS_DATES, **cfg)
    pf = s['net_pnl_usd'] / s['n_fills'] if s['n_fills'] else 0
    print(f"{fm:>9.1f}x {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${pf:>+10.4f}")

# ── Test 4: IS vs OOS ────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("TEST 4: IN-SAMPLE vs OOS (training overlap vs post-training)")
print("=" * 78)
for label, dates in [("IS (Apr 22, 26, 30, May 4)", IS_DATES),
                     ("OOS (May 7-16)", OOS_DATES)]:
    s = run_taker_backtest(dates, **base_cfg)
    pf = s['net_pnl_usd'] / s['n_fills'] if s['n_fills'] else 0
    print(f"{label:<35} fills={s['n_fills']:>6} win%={s['win_rate']*100:>5.1f} "
          f"pnl=${s['net_pnl_usd']:+8.2f} per-fill=${pf:+8.4f}")

# ── Test 5: Random-label sanity (shuffle resolved_side, should produce ~0) ───
print("\n" + "=" * 78)
print("TEST 5: RANDOM-LABEL SANITY (shuffle resolved_side_label → PnL should ~= 0)")
print("=" * 78)

import random
def shuffled_backtest(dates):
    """Run the taker logic but with resolved_side_label randomly shuffled."""
    from backtest.model02_taker_engine import load_dense_week
    df = load_dense_week(dates, max_per_day=15_000)
    df = clean_features(df)
    model, feats = _load_model()
    p_up = _predict_p_up(model, feats, df)
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int).copy()
    np.random.default_rng(seed=7).shuffle(resolved)   # break the prediction-outcome link
    edge_dn = up_bid - p_up
    down_ask = 1.0 - up_bid
    m = (edge_dn >= 0.10) & (ttc >= 10) & (ttc <= 60)
    if m.sum() == 0: return {"n": 0, "win_rate": 0, "mean_pnl": 0}
    won = (resolved[m] == 0)
    fc = FeeCalculator.for_date("2026-05-09")
    fees = np.array([fc.taker_fee_usd(price=p, size=1.0) for p in down_ask[m]])
    pnl = np.where(won, 1.0, 0.0) - down_ask[m] - fees
    return {"n": int(m.sum()), "win_rate": float(won.mean()), "mean_pnl": float(pnl.mean()),
            "total_pnl": float(pnl.sum())}

r = shuffled_backtest(OOS_DATES)
print(f"  Shuffled labels: n={r['n']}  win%={r['win_rate']*100:.1f}  "
      f"total_pnl=${r['total_pnl']:+.2f}  per-fill=${r['mean_pnl']:+.4f}")
print(f"  → If close to $0, the signal is genuine (not data leak).")

print("\n--- DONE ---")
