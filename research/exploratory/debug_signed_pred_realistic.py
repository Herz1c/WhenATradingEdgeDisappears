#!/usr/bin/env python3
"""Realistic test of signed_pred band strategy with proper margin model."""
import io, sys
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks

signals = SignalEngine(artifact_root="artifacts_cleaned")

# Test on in-sample (May 4-10, partial training overlap)
in_sample = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
             "2026-05-08", "2026-05-09", "2026-05-10"]

# True OOS (May 10-16, all post-training)
true_oos = ["2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13",
            "2026-05-14", "2026-05-15", "2026-05-16"]

filt = IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10,
                    max_p_severe=0.10, max_concurrent_positions=200)

print("Realistic config: signed_pred (0.05, 0.10] + sev<0.10")
print(f"  margin cap: $200 (~200 contracts × $1)")
print()

for label, week in [("IN-SAMPLE (May 4-10)", in_sample),
                    ("TRUE OOS (May 10-16)", true_oos)]:
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(week, max_intents_per_day=100_000, filled_only=True, max_margin_usd=200.0)
    print(f"=== {label} ===")
    print(f"  fills={s['n_fills']}  skipped_margin={s['n_skipped_for_margin']}  "
          f"win%={s['win_rate']*100:.1f}")
    print(f"  net=${s['net_pnl_usd']:+.2f}  max_dd=${s['max_drawdown_usd']:.2f}  "
          f"worst_day=${s['worst_day_pnl_usd']:+.2f}")
    print(f"  daily: " + ", ".join(f"{d}=${p:+.1f}" for d, p in sorted(s['daily_pnl'].items())))
    print()

# Test margin caps to find what's realistic for $100
print("=== MARGIN-CAP SWEEP ON IN-SAMPLE ===")
print(f"{'cap':>6} {'fills':>7} {'skipped':>8} {'pnl':>9} {'max_dd':>9} {'worst':>8}")
for cap in [50, 100, 200, 500, 1000, 5000]:
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(in_sample, max_intents_per_day=100_000, filled_only=True, max_margin_usd=cap)
    print(f"  ${cap:>4} {s['n_fills']:>7} {s['n_skipped_for_margin']:>8} "
          f"${s['net_pnl_usd']:>+7.2f} ${s['max_drawdown_usd']:>+7.2f} "
          f"${s['worst_day_pnl_usd']:>+6.2f}")

# Sanity: 20 MC weeks at $100 margin cap (realistic)
print("\n=== 20 MC WEEKS, $100 margin cap ===")
weeks = generate_mc_weeks(n_weeks=20, seed=0)
import statistics
pnls = []; worsts = []; dds = []; nfills = []
for i, w in enumerate(weeks):
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(w, max_intents_per_day=100_000, filled_only=True, max_margin_usd=100.0)
    pnls.append(s['net_pnl_usd']); worsts.append(s['worst_day_pnl_usd']);
    dds.append(s['max_drawdown_usd']); nfills.append(s['n_fills'])
    print(f"  w{i+1:>2}: fills={s['n_fills']:>5}  net=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  worst=${s['worst_day_pnl_usd']:+6.2f}")
print(f"\n  PnL    : mean=${statistics.mean(pnls):+.2f}  med=${statistics.median(pnls):+.2f}  min=${min(pnls):+.2f}  max=${max(pnls):+.2f}")
print(f"  Max DD : mean=${statistics.mean(dds):.2f}  med=${statistics.median(dds):.2f}  max=${max(dds):.2f}")
print(f"  Worst  : mean=${statistics.mean(worsts):+.2f}  med=${statistics.median(worsts):+.2f}  min=${min(worsts):+.2f}")
print(f"  Disqual (worst_day < -$10): {sum(1 for x in worsts if x < -10)}/{len(worsts)}")
print(f"  Positive weeks: {sum(1 for x in pnls if x > 0)}/{len(pnls)}")
