#!/usr/bin/env python3
"""Test extreme configs to find any profitable parameterization."""
import io, sys, time
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from trading.signals import SignalEngine
from trading.config import StrategyConfig
from backtest.multi_market_engine import MultiMarketEngine
from backtest.search_space import fixed_baseline_config

signals = SignalEngine(artifact_root="artifacts_cleaned")
week_dates = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10"]


def cfg_with(**overrides):
    cfg = fixed_baseline_config()
    for k, v in overrides.items():
        sect, attr = k.split(".", 1)
        setattr(getattr(cfg, sect), attr, v)
    return cfg


configs = {
    "ultra_defensive": cfg_with(
        **{"risk.enforce_edge_check_in_uniform_mode": True,
           "risk.min_edge_required_ticks": 1.0,
           "signals.severe_defense_threshold_frac": 0.10,
           "signals.extreme_freeze_threshold_frac": 0.08,
           "signals.moderate_lean_threshold_frac": 0.15,
           "signals.quote_edge_ticks": 5.0,
           "selection.max_concurrent_markets": 3,
           "selection.require_min_book_depth_contracts": 50.0}),
    "wide_quotes_only": cfg_with(
        **{"risk.enforce_edge_check_in_uniform_mode": True,
           "risk.min_edge_required_ticks": 0.5,
           "signals.quote_edge_ticks": 8.0,
           "signals.severe_defense_threshold_frac": 0.65,  # disabled
           "signals.extreme_freeze_threshold_frac": 0.55,  # disabled
           "selection.max_concurrent_markets": 20}),
    "tight_with_strict_edge": cfg_with(
        **{"risk.enforce_edge_check_in_uniform_mode": True,
           "risk.min_edge_required_ticks": 2.0,  # strong edge required
           "signals.quote_edge_ticks": 1.0,
           "signals.severe_defense_threshold_frac": 0.65,
           "selection.max_concurrent_markets": 10}),
    "taker_aggressive": cfg_with(
        **{"risk.enforce_edge_check_in_uniform_mode": True,
           "risk.min_edge_required_ticks": 5.0,  # don't make
           "risk.max_taker_trades_per_hour": 30,
           "signals.taker_min_edge_ticks": 2.5,
           "signals.taker_min_fill_prob": 0.60,
           "signals.quote_edge_ticks": 10.0,  # essentially refuse to maker
           "selection.max_concurrent_markets": 10}),
}

for name, cfg in configs.items():
    t0 = time.time()
    eng = MultiMarketEngine(cfg, signals, starting_balance_usd=100.0, seed=42)
    s = eng.run(week_dates, max_anchors_per_day=2000)
    elapsed = time.time() - t0
    print(f"\n=== {name.upper()} ({elapsed:.1f}s) ===")
    print(f"  fills={s['n_fills']}  win_rate={s['win_rate']:.1%}  "
          f"net=${s['net_pnl_usd']:+.2f}  max_dd=${s['max_drawdown_usd']:.2f}")
    print(f"  fees=${s['gross_fees_usd']:.2f}  rebates=${s['gross_rebates_usd']:.2f}")
    print(f"  worst_day=${s['worst_day_pnl_usd']:+.2f}  best_day=${s['best_day_pnl_usd']:+.2f}")
