#!/usr/bin/env python3
"""Direct test: how profitable is buying when model 02 fair_value > mid + threshold?

This bypasses the HybridStrategy and tests model 02 as a pure taker signal.
"""
import io, sys, time, math
from collections import defaultdict
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import polars as pl
import numpy as np

from trading.signals import SignalEngine
from backtest.fees import FeeCalculator
from backtest.precomputed_signals import batch_precompute
from backtest.multi_market_engine import load_week
from feature_cleanup import clean_features


def test_taker(week_dates, edge_threshold_usd: float, signals: SignalEngine,
               target_field: str = "fair_value", horizon: str = "close"):
    """Buy when model_pred > mid + threshold; sell when model_pred < mid - threshold.

    horizon: 'close' uses hold_to_close_edge_vs_mid (true realized PnL at resolution),
             'mid_5s' uses mid_move_5s,
             'mid_30s' uses mid_move_30s.
    """
    df = load_week(week_dates, max_anchors_per_day=2000, seed=42)
    df = clean_features(df)
    df = batch_precompute(df, signals)
    fee_calcs = {}

    total_pnl = 0.0
    n_trades = 0
    n_wins = 0
    n_losses = 0
    daily_pnl = defaultdict(float)

    pred_col = f"_pred_{target_field}"
    for row in df.iter_rows(named=True):
        pred = row.get(pred_col)
        if pred is None or not math.isfinite(pred):
            continue
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        bid = max(0.001, mid - spread / 2)
        ask = min(0.999, mid + spread / 2)
        date = str(row.get("date") or "?")

        if horizon == "close":
            hold_edge = row.get("hold_to_close_edge_vs_mid")
            if hold_edge is None or not math.isfinite(hold_edge):
                continue
            future = mid + float(hold_edge)
        elif horizon == "mid_30s":
            mm = row.get("mid_move_30s")
            if mm is None or not math.isfinite(mm):
                continue
            future = mid + float(mm)
        else:
            mm = row.get("mid_move_5s")
            if mm is None or not math.isfinite(mm):
                continue
            future = mid + float(mm)

        edge_up = pred - ask
        edge_dn = bid - pred

        if edge_up >= edge_threshold_usd:
            fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date))
            fee = fc.taker_fee_usd(price=ask, size=1.0)
            pnl = (future - ask) * 1.0 - fee
        elif edge_dn >= edge_threshold_usd:
            fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date))
            fee = fc.taker_fee_usd(price=bid, size=1.0)
            pnl = (bid - future) * 1.0 - fee
        else:
            continue

        total_pnl += pnl
        daily_pnl[date] += pnl
        n_trades += 1
        if pnl > 0: n_wins += 1
        elif pnl < 0: n_losses += 1

    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": (n_wins / n_trades) if n_trades else 0.0,
        "total_pnl_usd": total_pnl,
        "daily_pnl": dict(daily_pnl),
        "edge_threshold_usd": edge_threshold_usd,
    }


def main():
    signals = SignalEngine(artifact_root="artifacts_cleaned")
    week = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
            "2026-05-08", "2026-05-09", "2026-05-10"]
    print()
    print(f"{'threshold':>10} {'horizon':>10} {'n_trades':>10} {'win_rate':>10} {'pnl_usd':>12} {'pnl/trade':>11}")
    for horizon in ["mid_5s", "mid_30s", "close"]:
        for thr in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
            r = test_taker(week, thr, signals, "fair_value", horizon=horizon)
            pt = r['total_pnl_usd'] / max(1, r['n_trades'])
            print(f"  ${thr:.3f}   {horizon:>10} {r['n_trades']:>10} "
                  f"{r['win_rate']:>10.1%} ${r['total_pnl_usd']:>+11.2f} ${pt:>+10.4f}")


if __name__ == "__main__":
    main()
