#!/usr/bin/env python3
"""Test model 04b (markout-to-close) as a direct taker signal.

For each anchor, model_04b.predict() outputs E[markout_at_close].
Strategy: BUY YES if prediction > +threshold, SELL YES if prediction < -threshold.
Realized PnL = hold_to_close_edge_vs_mid * direction - fee.
"""
import io, sys, math
from collections import defaultdict
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.fees import FeeCalculator
from backtest.precomputed_signals import batch_precompute
from backtest.multi_market_engine import load_week
from feature_cleanup import clean_features


def run(week_dates, signals, pred_col, threshold, horizon_col="hold_to_close_edge_vs_mid",
        fee_kind="taker"):
    df = load_week(week_dates, max_anchors_per_day=2000, seed=42)
    df = clean_features(df)
    df = batch_precompute(df, signals)
    fee_calcs = {}

    total = 0.0; n = 0; w = 0; l = 0
    long_n = 0; short_n = 0
    daily = defaultdict(float)
    for row in df.iter_rows(named=True):
        pred = row.get(pred_col)
        if pred is None or not math.isfinite(pred): continue
        hold = row.get(horizon_col)
        if hold is None or not math.isfinite(hold): continue
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        bid = max(0.001, mid - spread/2); ask = min(0.999, mid + spread/2)
        date = str(row.get("date") or "?")

        # pred is expected markout vs mid (USD per contract held to close)
        if pred >= threshold:
            # LONG
            fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date))
            entry = ask if fee_kind == "taker" else mid
            fee = fc.taker_fee_usd(price=entry, size=1.0) if fee_kind == "taker" else -fc.maker_rebate_usd(price=entry, size=1.0)
            realized = (mid + float(hold)) - entry - fee
            long_n += 1
        elif pred <= -threshold:
            # SHORT
            fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date))
            entry = bid if fee_kind == "taker" else mid
            fee = fc.taker_fee_usd(price=entry, size=1.0) if fee_kind == "taker" else -fc.maker_rebate_usd(price=entry, size=1.0)
            realized = entry - (mid + float(hold)) - fee
            short_n += 1
        else:
            continue
        total += realized; daily[date] += realized; n += 1
        if realized > 0: w += 1
        elif realized < 0: l += 1
    return dict(n=n, long_n=long_n, short_n=short_n, w=w, l=l, total=total, daily=daily)


def main():
    signals = SignalEngine(artifact_root="artifacts_cleaned")
    week = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
            "2026-05-08", "2026-05-09", "2026-05-10"]

    print(f"\n{'model':<35} {'thr':>7} {'fee':>6} {'n':>7} {'long/short':>12} "
          f"{'win%':>7} {'pnl':>10} {'pnl/tr':>9}")
    print("-" * 100)
    for pred_col, label in [
        ("_pred_expected_markout_to_close", "04b markout_to_close"),
        ("_pred_mispricing_predicted", "06 mispricing"),
        ("_pred_expected_markout_1s", "04 adverse_sel_1s"),
    ]:
        for thr in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
            for fee_kind in ["taker", "maker"]:
                r = run(week, signals, pred_col, thr, fee_kind=fee_kind)
                if r["n"] == 0: continue
                wr = r["w"] / r["n"]
                pt = r["total"] / r["n"]
                print(f"{label:<35} {thr:>6.3f}  {fee_kind:>5}  {r['n']:>7} "
                      f"{r['long_n']:>5}/{r['short_n']:<6} {wr:>6.1%} "
                      f"${r['total']:>+8.2f} ${pt:>+7.4f}")


if __name__ == "__main__":
    main()
