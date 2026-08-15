#!/usr/bin/env python3
"""Official Calmar-style grade of the hand-tuned winning config across 20 MC weeks."""
import io, sys, json, statistics
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks
from backtest.grading import grade_strategy

signals = SignalEngine(artifact_root="artifacts_cleaned")

# The winning config from the manual exploration
filt = IntentFilter(
    signed_pred_min=0.05, signed_pred_max=0.10,
    max_p_severe=0.10,
    max_concurrent_positions=10**6,   # margin cap is the real constraint
)

weeks = generate_mc_weeks(n_weeks=20, seed=42)
print(f"Grading 'signed_pred (0.05, 0.10] + sev<0.10' across 20 MC weeks (margin cap $100)...")
print()

per_week = []
for i, w in enumerate(weeks):
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(w, max_intents_per_day=15_000, filled_only=True, max_margin_usd=100.0)
    per_week.append(s)
    print(f"  w{i+1:>2}: fills={s['n_fills']:>5}  win%={s['win_rate']*100:>5.1f}  "
          f"pnl=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
          f"worst=${s['worst_day_pnl_usd']:+6.2f}  best=${s['best_day_pnl_usd']:+6.2f}")

g = grade_strategy(per_week)
print()
print(f"=== OFFICIAL GRADE ===")
for k, v in g.items():
    print(f"  {k} = {v}")

pnls = [w['net_pnl_usd'] for w in per_week]
print()
print(f"Distribution:")
print(f"  Mean PnL: ${statistics.mean(pnls):+.2f} / week")
print(f"  Median:   ${statistics.median(pnls):+.2f}")
print(f"  Min/Max:  ${min(pnls):+.2f} / ${max(pnls):+.2f}")
print(f"  Std:      ${statistics.stdev(pnls):.2f}")
print(f"  Positive weeks: {sum(1 for p in pnls if p > 0)}/{len(pnls)}")

# Save
from pathlib import Path
Path("docs").mkdir(exist_ok=True)
Path("docs/winning_config_grade.json").write_text(json.dumps({
    "filter": {"signed_pred_min": 0.05, "signed_pred_max": 0.10, "max_p_severe": 0.10,
               "max_concurrent_positions": 10**6, "max_margin_usd": 100.0},
    "grade": g,
    "per_week": per_week,
    "weeks": weeks,
}, indent=2, default=str))
print("\nSaved to docs/winning_config_grade.json")
