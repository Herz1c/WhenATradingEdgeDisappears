#!/usr/bin/env python3
"""Pure OOS validation: grade winner ONLY on days outside training window.

Training: 2026-04-19 .. 2026-05-06.
Pure OOS:  2026-05-07 .. 2026-05-16  (10 days).
With 1-week embargo: 2026-05-13 .. 2026-05-16 (4 days).

We build "weeks" from the OOS pool only (7 unique days). For the 1-week-embargo
case, only 4 days exist, so we test that as a single 4-day "mini-week".
"""
import io, sys, json, statistics, random
from pathlib import Path
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.grading import grade_strategy

signals = SignalEngine(artifact_root="artifacts_cleaned")

filt = IntentFilter(
    signed_pred_min=0.05, signed_pred_max=0.10, max_p_severe=0.10,
    max_concurrent_positions=10**6,
)

# Pure OOS pool: May 7 onwards (post training)
oos_pool = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
            "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
            "2026-05-15","2026-05-16"]

# 20 MC weeks sampled from OOS pool only
weeks = []
for i in range(20):
    rng = random.Random(1000 + i)
    week = sorted(rng.sample(oos_pool, 7))
    weeks.append(week)

print("=" * 70)
print("PURE OOS GRADE — 20 MC weeks built from May 7-16 ONLY (post training)")
print("=" * 70)

per_week = []
for i, w in enumerate(weeks):
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(w, max_intents_per_day=15_000, filled_only=True, max_margin_usd=100.0)
    per_week.append(s)
    print(f"  w{i+1:>2}: fills={s['n_fills']:>5}  win%={s['win_rate']*100:>5.1f}  "
          f"pnl=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
          f"worst=${s['worst_day_pnl_usd']:+6.2f}")

g = grade_strategy(per_week)
print(f"\n=== PURE-OOS GRADE ===")
print(f"  score={g['score']:+.3f}  rejected={g['rejected']}")
print(f"  median_pnl=${g['median_net_pnl_usd']:+.2f}  mean=${g['mean_net_pnl_usd']:+.2f}")
print(f"  p10=${g['p10_net_pnl_usd']:+.2f}  positive_weeks={g['n_weeks_with_positive_pnl']}/{g['n_weeks']}")
print(f"  mean_calmar={g['mean_calmar_clean']:.2f}  disq_rate={g['disqualification_rate']:.2f}")

pnls = [w['net_pnl_usd'] for w in per_week]
print(f"  min/max=${min(pnls):+.2f} / ${max(pnls):+.2f}  std=${statistics.stdev(pnls):.2f}")

# Also test the 4-day pure post-embargo window directly
print("\n" + "=" * 70)
print("DEEP-OOS (post 1-week embargo): single 4-day run on May 13-16")
print("=" * 70)
deep_oos = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]
eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
s = eng.run(deep_oos, max_intents_per_day=30_000, filled_only=True, max_margin_usd=100.0)
print(f"  fills={s['n_fills']}  win%={s['win_rate']*100:.1f}  "
      f"pnl=${s['net_pnl_usd']:+.2f}  dd=${s['max_drawdown_usd']:.2f}")
print(f"  daily: {dict(sorted(s['daily_pnl'].items()))}")

# Save
Path("docs").mkdir(exist_ok=True)
Path("docs/pure_oos_grade.json").write_text(json.dumps({
    "filter": filt.__dict__,
    "oos_pool": oos_pool,
    "weeks": weeks,
    "grade": g,
    "per_week": per_week,
    "deep_oos_4day": s,
}, indent=2, default=str))
print("\nSaved to docs/pure_oos_grade.json")
