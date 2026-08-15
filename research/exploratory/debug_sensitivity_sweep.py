#!/usr/bin/env python3
"""Sensitivity sweep ±20% on each parameter of the winning config.

For each parameter, perturb +20% and -20% (with sensible discrete steps for
the categorical/integer ones) and re-grade across 10 MC weeks. A robust
strategy survives small perturbations; a brittle one collapses.
"""
import io, sys, json, statistics
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks
from backtest.grading import grade_strategy

signals = SignalEngine(artifact_root="artifacts_cleaned")
weeks = generate_mc_weeks(n_weeks=6, seed=42)   # 6 weeks for tractable runtime

base_filt_dict = dict(
    signed_pred_min=0.05,
    signed_pred_max=0.10,
    max_p_severe=0.10,
    max_concurrent_positions=10**6,
)


def evaluate(d):
    filt = IntentFilter(**d)
    per_week = []
    for w in weeks:
        eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
        s = eng.run(w, max_intents_per_day=8_000, filled_only=True, max_margin_usd=100.0)
        per_week.append(s)
    return grade_strategy(per_week), per_week


print("=" * 80)
print(f"SENSITIVITY SWEEP — {len(weeks)} MC weeks, ±20% on each param")
print("=" * 80)
print(f"{'param':<30} {'value':>10} {'mean_pnl':>12} {'med_pnl':>11} {'mean_calmar':>12} {'pos/n':>9} {'disq%':>7}")
print("-" * 95)

# Baseline
g0, _ = evaluate(base_filt_dict)
print(f"{'(baseline)':<30} {'':>10} ${g0['mean_net_pnl_usd']:>+10.2f} ${g0['median_net_pnl_usd']:>+9.2f} "
      f"{g0['mean_calmar_clean']:>11.2f} {g0['n_weeks_with_positive_pnl']}/{g0['n_weeks']:<5} "
      f"{g0['disqualification_rate']*100:>6.0f}%")

# Perturbations
perturbations = [
    ("signed_pred_min", [0.04, 0.06]),   # ±0.01 (±20% of 0.05)
    ("signed_pred_max", [0.08, 0.12]),   # ±0.02 (±20% of 0.10)
    ("max_p_severe",    [0.08, 0.12]),   # ±20% of 0.10
]

results = {"baseline": g0}
for param, vals in perturbations:
    for v in vals:
        d = deepcopy(base_filt_dict)
        d[param] = v
        g, _ = evaluate(d)
        results[f"{param}={v}"] = g
        print(f"{param + '=' + str(v):<30} {'':>10} "
              f"${g['mean_net_pnl_usd']:>+10.2f} ${g['median_net_pnl_usd']:>+9.2f} "
              f"{g['mean_calmar_clean']:>11.2f} "
              f"{g['n_weeks_with_positive_pnl']}/{g['n_weeks']:<5} "
              f"{g['disqualification_rate']*100:>6.0f}%")

# Margin cap is engine-level not filter-level
print()
print("MARGIN CAP SWEEP:")
print(f"{'margin_cap':>10} {'mean_pnl':>12} {'med_pnl':>11} {'mean_calmar':>12} {'pos/n':>9}")
for cap in [50, 80, 100, 120, 150, 200]:
    filt = IntentFilter(**base_filt_dict)
    per_week = []
    for w in weeks:
        eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
        s = eng.run(w, max_intents_per_day=8_000, filled_only=True, max_margin_usd=cap)
        per_week.append(s)
    g = grade_strategy(per_week)
    print(f"${cap:>4} {'':<5} ${g['mean_net_pnl_usd']:>+10.2f} ${g['median_net_pnl_usd']:>+9.2f} "
          f"{g['mean_calmar_clean']:>11.2f} {g['n_weeks_with_positive_pnl']}/{g['n_weeks']:<5}")
    results[f"margin_cap=${cap}"] = g

Path("docs").mkdir(exist_ok=True)
Path("docs/sensitivity_sweep.json").write_text(json.dumps(results, indent=2, default=str))
print("\nSaved to docs/sensitivity_sweep.json")

# Verdict
print("\n=== VERDICT ===")
base_pnl = g0['mean_net_pnl_usd']
robust = True
for name, g in results.items():
    if name == "baseline": continue
    delta_pct = (g['mean_net_pnl_usd'] - base_pnl) / abs(base_pnl) * 100
    if g['rejected'] or g['mean_net_pnl_usd'] < 0:
        robust = False
        print(f"  {name}: pnl=${g['mean_net_pnl_usd']:+.2f}  ({delta_pct:+.0f}% vs baseline) — REJECTED")
    elif abs(delta_pct) > 50:
        print(f"   {name}: pnl=${g['mean_net_pnl_usd']:+.2f}  ({delta_pct:+.0f}% vs baseline) — large drift")
    else:
        print(f"  ✓ {name}: pnl=${g['mean_net_pnl_usd']:+.2f}  ({delta_pct:+.0f}% vs baseline) — robust")
print("STRATEGY IS ROBUST UNDER PERTURBATION." if robust else "STRATEGY IS BRITTLE — some perturbations break it.")
