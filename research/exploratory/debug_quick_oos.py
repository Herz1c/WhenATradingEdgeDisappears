#!/usr/bin/env python3
"""Fast in-sample vs OOS comparison."""
import io, sys
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter

signals = SignalEngine(artifact_root="artifacts_cleaned")

in_sample = ["2026-05-04", "2026-05-05", "2026-05-06"]
true_oos_1 = ["2026-05-10", "2026-05-11", "2026-05-12"]
true_oos_2 = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]

filt = IntentFilter(signed_pred_min=0.05, signed_pred_max=0.10,
                    max_p_severe=0.10, max_concurrent_positions=200)

print("Filter: signed_pred (0.05, 0.10] + sev<0.10, margin_cap=$100")
print()
for label, dates in [("IN-SAMPLE May 4-6", in_sample),
                     ("OOS May 10-12", true_oos_1),
                     ("OOS May 13-16", true_oos_2)]:
    eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
    s = eng.run(dates, max_intents_per_day=30_000, filled_only=True, max_margin_usd=100.0)
    print(f"=== {label} ===")
    print(f"  fills={s['n_fills']}  skipped_for_margin={s['n_skipped_for_margin']}  "
          f"win%={s['win_rate']*100:.1f}  elapsed={s['elapsed_sec']:.1f}s")
    print(f"  pnl=${s['net_pnl_usd']:+.2f}  max_dd=${s['max_drawdown_usd']:.2f}  "
          f"worst=${s['worst_day_pnl_usd']:+.2f}  best=${s['best_day_pnl_usd']:+.2f}")
    print(f"  daily: {dict(sorted(s['daily_pnl'].items()))}")
    print()
