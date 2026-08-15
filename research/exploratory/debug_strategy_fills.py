#!/usr/bin/env python3
"""Debug: load 200 anchors from a single day and trace why no fills happen.

Reports counts at each gate:
  - anchors loaded
  - entry filter (passes_entry_filters): pass/fail by reason
  - intents emitted by type
  - fills by intent type
"""
from __future__ import annotations
import io, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import polars as pl
import random

from trading.config import StrategyConfig
from trading.signals import SignalEngine
from trading.risk import RiskManager
from trading.strategy import HybridStrategy
from trading.entities import PlaceQuote, CancelQuote, TakerOrder, FlattenInventory
from backtest.fees import FeeCalculator
from backtest.fill_model import FillSimulator
from backtest.multi_market_engine import row_to_event_and_features
from feature_cleanup import clean_features


def main():
    date = "2026-05-09"
    p = Path(f"data/datasets/microstructure_sequence_dataset_v1_tabular/{date}.parquet")
    df = pl.read_parquet(p)
    if "sequence_feature_eligible" in df.columns:
        df = df.filter(pl.col("sequence_feature_eligible"))
    df = df.filter(pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan())
    df = clean_features(df)
    df = df.sample(n=200, seed=42, with_replacement=False).sort("anchor_ts_ns")

    # Inspect data distribution
    spread = df.get_column("spread_mean_5s").to_numpy()
    depth_bid = df.get_column("state_bid_depth_total").to_numpy() if "state_bid_depth_total" in df.columns else None
    print(f"--- DATA SAMPLE (n=200, {date}) ---")
    print(f"spread_mean_5s: min={spread.min():.4f} med={float(pl.Series(spread).median()):.4f} max={spread.max():.4f}")
    if depth_bid is not None:
        import numpy as np
        print(f"state_bid_depth_total: min={np.nanmin(depth_bid):.1f} med={float(pl.Series(depth_bid).median()):.1f} max={np.nanmax(depth_bid):.1f}")
    print(f"unique markets: {df['market_slug'].n_unique()}")

    cfg = StrategyConfig.market_maker()
    cfg.risk.max_daily_loss_usd = 20.0
    cfg.risk.max_drawdown_from_session_peak_usd = 10.0
    cfg.log_every_decision = False
    sigs_engine = SignalEngine(artifact_root="artifacts_cleaned")
    risk = RiskManager(cfg.risk)
    strat = HybridStrategy(cfg, sigs_engine, risk)
    fee_calc = FeeCalculator.for_date(date)
    sim = FillSimulator(cfg.execution, fee_calc, random.Random(42))

    n_intents = Counter()
    n_fills = Counter()
    n_anchors = 0
    n_no_intents = 0
    n_emit_rejected = 0
    entry_fails = Counter()

    # Patch _passes_entry_filters to track failures
    orig = strat._passes_entry_filters
    def traced(state, sigs):
        cfg2 = strat.cfg; tick = cfg2.execution.tick_size
        if state.t_to_close_s <= cfg2.risk.no_quote_in_last_n_seconds:
            entry_fails["t_to_close"] += 1; return False
        if state.book.spread <= 0:
            entry_fails["spread<=0"] += 1; return False
        if state.book.spread > cfg2.selection.max_book_spread_to_quote_ticks * tick:
            entry_fails["spread_too_wide"] += 1; return False
        if state.book.bid_depth < cfg2.selection.require_min_book_depth_contracts:
            entry_fails["bid_depth"] += 1; return False
        if state.book.ask_depth < cfg2.selection.require_min_book_depth_contracts:
            entry_fails["ask_depth"] += 1; return False
        if sigs.p_severe_7c_3s is not None and sigs.p_severe_7c_3s >= cfg2.signals.severe_defense_threshold_frac:
            entry_fails[f"severe_defense({sigs.p_severe_7c_3s:.2f})"] += 1; return False
        return True
    strat._passes_entry_filters = traced

    for row in df.iter_rows(named=True):
        n_anchors += 1
        try:
            event, feat = row_to_event_and_features(row)
        except Exception as e:
            entry_fails[f"event_build_err:{type(e).__name__}"] += 1
            continue
        sigs = sigs_engine.predict(strat.market_states.get(event.market_id) or strat._update_market_state(event), feat)
        intents = strat.on_market_event(event, feat)
        if not intents:
            n_no_intents += 1
            continue
        for it in intents:
            n_intents[type(it).__name__] += 1
            if isinstance(it, FlattenInventory):
                pos = risk.positions.get(it.market_id, 0.0)
                fill = sim._simulate_flatten(it, row, pos)
            else:
                fill = sim.simulate(it, row, sigs=sigs)
            if fill:
                n_fills[type(it).__name__] += 1

    print()
    print(f"--- RESULTS ---")
    print(f"anchors processed:   {n_anchors}")
    print(f"anchors with 0 intents emitted: {n_no_intents}")
    print(f"entry filter fails:  {dict(entry_fails)}")
    print(f"intents emitted:     {dict(n_intents)}")
    print(f"fills:               {dict(n_fills)}")
    print(f"sample signals on first market: p_severe={sigs.p_severe_7c_3s}, p_extreme={sigs.p_extreme_10c_5s}, fair_value={sigs.fair_value}")


if __name__ == "__main__":
    main()
