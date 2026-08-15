#!/usr/bin/env python3
"""Validate the ExecutionIntentEngine with hand-tuned filter configs."""
import io, sys, time
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks

signals = SignalEngine(artifact_root="artifacts_cleaned")
single_week = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
               "2026-05-08", "2026-05-09", "2026-05-10"]

CAP = 100000  # effectively uncapped — just see raw signal economics
configs = [
    ("accept-all", IntentFilter(max_concurrent_positions=CAP)),
    ("fp>=0.75", IntentFilter(min_fill_prob=0.75, max_concurrent_positions=CAP)),
    ("fp>=0.75 + ttc<=30", IntentFilter(min_fill_prob=0.75, max_t_to_close_s=30, max_concurrent_positions=CAP)),
    ("fp>=0.85 + ttc<=30", IntentFilter(min_fill_prob=0.85, max_t_to_close_s=30, max_concurrent_positions=CAP)),
    ("fp>=0.75 + ttc<=60", IntentFilter(min_fill_prob=0.75, max_t_to_close_s=60, max_concurrent_positions=CAP)),
    ("fp>=0.80 + ttc 10-30 + sev<0.10", IntentFilter(min_fill_prob=0.80, min_t_to_close_s=10, max_t_to_close_s=30, max_p_severe=0.10, max_concurrent_positions=CAP)),
    ("fp>=0.75 + ttc<=45 + mod<0.20", IntentFilter(min_fill_prob=0.75, max_t_to_close_s=45, max_p_moderate=0.20, max_concurrent_positions=CAP)),
]

print(f"{'config':<48} {'fills':>7} {'win%':>7} {'pnl':>10} {'max_dd':>9} {'fees':>8} {'rebates':>9}")
print("-" * 105)
for name, filt in configs:
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(single_week, max_intents_per_day=30_000)
    print(f"{name:<48} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+7.2f} "
          f"${s['gross_fees_usd']:>+6.2f} ${s['gross_rebates_usd']:>+7.2f}")

# Test on 5 MC weeks
print("\n=== BEST FILTER ACROSS 5 RANDOM MC WEEKS ===")
best = IntentFilter(min_fill_prob=0.75, max_t_to_close_s=30,
                    max_p_severe=0.15, quote_size=1, max_concurrent_positions=CAP)
weeks = generate_mc_weeks(n_weeks=5, seed=0)
pnls = []
for i, w in enumerate(weeks):
    eng = ExecutionIntentEngine(signals, best, starting_balance_usd=100.0)
    s = eng.run(w, max_intents_per_day=30_000)
    pnls.append(s['net_pnl_usd'])
    print(f"  week {i+1}: fills={s['n_fills']:>5}  "
          f"net=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
          f"worst_day=${s['worst_day_pnl_usd']:+6.2f}")
print(f"  Mean: ${sum(pnls)/len(pnls):+.2f}  Min: ${min(pnls):+.2f}  Max: ${max(pnls):+.2f}")
