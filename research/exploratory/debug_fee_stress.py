#!/usr/bin/env python3
"""Fee stress test: re-grade winner under 1.0x, 1.2x, 1.5x, and 2.0x fee scaling.

`fee_scale` multiplies BOTH taker fees (we don't pay any as a pure maker) AND
the maker rebate base. So 1.5x fee_scale means rebate also scales up — that's
not actually adversarial for us. To properly stress, we want fees UP and
rebates DOWN.

We model this by scaling the rebate manually inside the run.
"""
import io, sys, json, statistics
from pathlib import Path
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.mc_weeks import generate_mc_weeks
from backtest.grading import grade_strategy
from backtest.fees import FeeCalculator

signals = SignalEngine(artifact_root="artifacts_cleaned")
weeks = generate_mc_weeks(n_weeks=6, seed=42)   # 6 weeks for tractable runtime

filt = IntentFilter(
    signed_pred_min=0.05, signed_pred_max=0.10, max_p_severe=0.10,
    max_concurrent_positions=10**6,
)


def grade_with_rebate_multiplier(rebate_mult: float):
    """Run all weeks with rebate scaled by `rebate_mult`. Monkey-patches the
    FeeCalculator.maker_rebate_usd method temporarily."""
    orig = FeeCalculator.maker_rebate_usd

    def patched(self, price, size):
        return orig(self, price, size) * rebate_mult

    FeeCalculator.maker_rebate_usd = patched
    try:
        per_week = []
        for w in weeks:
            eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
            s = eng.run(w, max_intents_per_day=8_000, filled_only=True, max_margin_usd=100.0)
            per_week.append(s)
        return grade_strategy(per_week), per_week
    finally:
        FeeCalculator.maker_rebate_usd = orig


results = {}
print(f"{'rebate_mult':>12} {'mean_pnl':>12} {'med_pnl':>11} {'calmar':>9} {'pos/n':>9} {'disq%':>7}")
print("-" * 70)
for rm in [1.0, 0.8, 0.5, 0.0, -1.0]:
    g, pw = grade_with_rebate_multiplier(rm)
    label = f"{rm:.1f}x"
    if rm == 0:
        label = "0 (no rebate)"
    elif rm < 0:
        label = f"{rm}x (PENALTY)"
    results[label] = g
    print(f"{label:>12} ${g['mean_net_pnl_usd']:>+10.2f} ${g['median_net_pnl_usd']:>+9.2f} "
          f"{g['mean_calmar_clean']:>8.2f} "
          f"{g['n_weeks_with_positive_pnl']}/{g['n_weeks']:<5} "
          f"{g['disqualification_rate']*100:>6.0f}%")

Path("docs").mkdir(exist_ok=True)
Path("docs/fee_stress.json").write_text(json.dumps(results, indent=2, default=str))
print("\nSaved to docs/fee_stress.json")

print("\n=== VERDICT ===")
for label, g in results.items():
    if g['rejected']:
        print(f"  {label}: REJECTED (mean=${g['mean_net_pnl_usd']:+.2f})")
    elif g['mean_net_pnl_usd'] < 0:
        print(f"  {label}: negative mean PnL (${g['mean_net_pnl_usd']:+.2f})")
    else:
        print(f"  ✓ {label}: passes (mean=${g['mean_net_pnl_usd']:+.2f})")
