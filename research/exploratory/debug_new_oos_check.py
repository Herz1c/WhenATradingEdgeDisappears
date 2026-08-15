#!/usr/bin/env python3
"""Quick Model 02 DOWN strategy sanity check on new May 17-20 OOS data."""
import io, sys, json
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from backtest.model02_taker_engine import run_taker_backtest

NEW_OOS = ["2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20"]

print("=" * 78)
print("MODEL 02 DOWN STRATEGY — NEW OOS DAYS (May 17-20, just built)")
print("=" * 78)

print(f"\n--- Per-day (edge>=0.10, ttc 10-60, dedup per market) ---")
print(f"{'date':<14} {'fills':>7} {'win%':>7} {'pnl':>10} {'dd':>8}")
totals = {"fills": 0, "wins": 0, "pnl": 0.0}
for d in NEW_OOS:
    s = run_taker_backtest(
        [d], edge_threshold_usd=0.10,
        allow_up_side=False, allow_dn_side=True,
        min_t_to_close_s=10, max_t_to_close_s=60,
        max_margin_usd=100.0, max_per_day=30_000,
        max_positions_per_market=1,
    )
    totals["fills"] += s["n_fills"]
    totals["wins"] += s["n_wins"]
    totals["pnl"] += s["net_pnl_usd"]
    print(f"{d:<14} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+6.2f}")
total_wr = totals["wins"] / max(1, totals["fills"]) * 100
print(f"\n{'TOTAL':<14} {totals['fills']:>7} {total_wr:>6.1f}% ${totals['pnl']:>+8.2f}")
print(f"\nReference (May 7-16 OOS): WR 70%, $4218 net over 10 days, ~$0.31/fill")
print(f"NEW (May 17-20):           WR {total_wr:.1f}%, ${totals['pnl']:.2f} net over 4 days, "
      f"~${totals['pnl']/max(1,totals['fills']):.4f}/fill")
