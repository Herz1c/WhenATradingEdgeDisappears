#!/usr/bin/env python3
"""Validate the directional-maker config: maker quotes gated on Model 04b sign."""
import io, sys, time
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from backtest.multi_market_engine import MultiMarketEngine
from backtest.search_space import (
    fixed_baseline_config, fixed_directional_maker_config
)
from backtest.mc_weeks import generate_mc_weeks

signals = SignalEngine(artifact_root="artifacts_cleaned")

# Test the same week we tested before for direct comparison
single_week = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
               "2026-05-08", "2026-05-09", "2026-05-10"]

# And a small MC sample
weeks = generate_mc_weeks(n_weeks=5, seed=0)

def make_filtered_cfg(gate, severe_thr, extreme_thr, quote_edge=0.5):
    cfg = fixed_directional_maker_config(gate)
    cfg.signals.severe_defense_threshold_frac = severe_thr
    cfg.signals.extreme_freeze_threshold_frac = extreme_thr
    cfg.signals.moderate_lean_threshold_frac = min(severe_thr * 0.7, 0.15)
    cfg.signals.quote_edge_ticks = quote_edge
    return cfg

configs = [
    ("baseline (all 4 sides)", fixed_baseline_config()),
    ("dir-gate $0.05, no defense", fixed_directional_maker_config(0.05)),
    ("dir-gate $0.05 + 08c-defense 0.20/0.15", make_filtered_cfg(0.05, 0.20, 0.15)),
    ("dir-gate $0.05 + 08c-defense 0.10/0.08", make_filtered_cfg(0.05, 0.10, 0.08)),
    ("dir-gate $0.05 + 08c-defense 0.05/0.05", make_filtered_cfg(0.05, 0.05, 0.05)),
    ("dir-gate $0.10 + 08c-defense 0.10/0.08", make_filtered_cfg(0.10, 0.10, 0.08)),
    ("dir-gate $0.20 + 08c-defense 0.10/0.08", make_filtered_cfg(0.20, 0.10, 0.08)),
    ("dir-gate $0.10 + 08c 0.10/0.08, wide quotes", make_filtered_cfg(0.10, 0.10, 0.08, quote_edge=3.0)),
]

for name, cfg in configs:
    t0 = time.time()
    eng = MultiMarketEngine(cfg, signals, starting_balance_usd=100.0, seed=42)
    s = eng.run(single_week, max_anchors_per_day=2000)
    el = time.time() - t0
    print(f"\n=== {name} ({el:.1f}s, single week 2026-05-04..10) ===")
    print(f"  fills={s['n_fills']}  win_rate={s['win_rate']:.1%}  "
          f"net=${s['net_pnl_usd']:+.2f}  max_dd=${s['max_drawdown_usd']:.2f}")
    print(f"  worst_day=${s['worst_day_pnl_usd']:+.2f}  best_day=${s['best_day_pnl_usd']:+.2f}")

print("\n=== TOP CONFIG ACROSS 5 RANDOM MC WEEKS ===")
best_cfg = make_filtered_cfg(0.10, 0.10, 0.08)
mc_pnls = []
for i, w in enumerate(weeks):
    eng = MultiMarketEngine(best_cfg, signals, starting_balance_usd=100.0, seed=42 + i)
    s = eng.run(w, max_anchors_per_day=2000)
    mc_pnls.append(s['net_pnl_usd'])
    print(f"  week {i+1}: fills={s['n_fills']:>5}  net=${s['net_pnl_usd']:+7.2f}  "
          f"dd=${s['max_drawdown_usd']:5.2f}  worst_day=${s['worst_day_pnl_usd']:+7.2f}")
print(f"  Mean: ${sum(mc_pnls)/len(mc_pnls):+.2f}  "
      f"Min: ${min(mc_pnls):+.2f}  Max: ${max(mc_pnls):+.2f}")
