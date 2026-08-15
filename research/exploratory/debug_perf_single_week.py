#!/usr/bin/env python3
"""Time a single week of backtest after precompute optimization."""
import io, sys, time
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.multi_market_engine import MultiMarketEngine
from backtest.search_space import fixed_baseline_config, fixed_defensive_config

signals = SignalEngine(artifact_root="artifacts_cleaned")
week_dates = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10"]

for name, cfg in [("baseline", fixed_baseline_config()), ("defensive", fixed_defensive_config())]:
    t0 = time.time()
    eng = MultiMarketEngine(cfg, signals, starting_balance_usd=100.0, seed=42)
    s = eng.run(week_dates, max_anchors_per_day=2000)
    elapsed = time.time() - t0
    print(f"\n=== {name.upper()} ({elapsed:.1f}s) ===")
    print(f"  anchors: {s['n_anchors_processed']}  fills: {s['n_fills']}  win_rate: {s['win_rate']:.1%}")
    print(f"  net pnl: ${s['net_pnl_usd']:+.2f}  max_dd: ${s['max_drawdown_usd']:.2f}")
    print(f"  fees ${s['gross_fees_usd']:.2f}  rebates ${s['gross_rebates_usd']:.2f}")
    print(f"  daily: " + ", ".join(f"{d}=${p:+.2f}" for d, p in sorted(s['daily_pnl'].items())))
