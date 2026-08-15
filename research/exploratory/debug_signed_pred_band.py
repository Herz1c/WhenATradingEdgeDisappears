#!/usr/bin/env python3
"""Test the signed_pred band (0.05, 0.10] strategy across multiple weeks."""
import io, sys
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks

signals = SignalEngine(artifact_root="artifacts_cleaned")

configs = [
    ("baseline (filled-only, no filter)", IntentFilter(max_concurrent_positions=10**9)),
    ("signed_pred (0.05, 0.10]", IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, max_concurrent_positions=10**9)),
    ("signed_pred (0.03, 0.12]", IntentFilter(signed_pred_min=0.03, signed_pred_max=0.12, max_concurrent_positions=10**9)),
    ("signed_pred (0.05, 0.10] + ttc<=60", IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, max_t_to_close_s=60, max_concurrent_positions=10**9)),
    ("signed_pred (0.05, 0.10] + ttc<=30", IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, max_t_to_close_s=30, max_concurrent_positions=10**9)),
    ("signed_pred (0.05, 0.10] + sev<0.10", IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, max_p_severe=0.10, max_concurrent_positions=10**9)),
    ("signed_pred (0.05, 0.10] + fp>=0.75", IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, min_fill_prob=0.75, max_concurrent_positions=10**9)),
]

# In-sample week
single_week = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
               "2026-05-08", "2026-05-09", "2026-05-10"]

print(f"{'config':<55} {'fills':>7} {'win%':>7} {'pnl':>9} {'dd':>9}")
print("-" * 95)
for name, filt in configs:
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(single_week, max_intents_per_day=50_000, filled_only=True)
    print(f"{name:<55} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+7.2f} ${s['max_drawdown_usd']:>+7.2f}")

# MC across 5 weeks for the top picks
print("\n=== TOP CANDIDATE: signed_pred (0.05, 0.10] + max_p_severe<0.10 ACROSS 10 MC WEEKS ===")
best = IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10, max_p_severe=0.10,
                    max_concurrent_positions=10**9)
weeks = generate_mc_weeks(n_weeks=10, seed=0)
pnls = []; dds = []; worst = []
for i, w in enumerate(weeks):
    eng = ExecutionIntentEngine(signals, best, starting_balance_usd=100.0)
    s = eng.run(w, max_intents_per_day=50_000, filled_only=True)
    pnls.append(s['net_pnl_usd']); dds.append(s['max_drawdown_usd']); worst.append(s['worst_day_pnl_usd'])
    print(f"  week {i+1}: fills={s['n_fills']:>5}  "
          f"net=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
          f"worst_day=${s['worst_day_pnl_usd']:+6.2f}")
print(f"  Mean PnL: ${sum(pnls)/len(pnls):+.2f}  Min: ${min(pnls):+.2f}  Max: ${max(pnls):+.2f}")
print(f"  Mean MaxDD: ${sum(dds)/len(dds):+.2f}  Mean WorstDay: ${sum(worst)/len(worst):+.2f}")
print(f"  Disqual (worst<-10): {sum(1 for x in worst if x < -10)}/{len(worst)}")
