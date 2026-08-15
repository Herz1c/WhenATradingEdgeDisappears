"""Tiny offline simulator for sanity-testing HybridStrategy.

This is NOT a full backtester (that's a separate, larger build). It exists
to confirm the strategy module wires up correctly: signals load, mode
transitions fire, risk limits engage, and intents are emitted in a sane
order.  Use it as the canary before building the real backtester.

Run:
    py -3 -m trading.simulator
or
    py -3 src/trading/simulator.py
"""
from __future__ import annotations
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from trading.config import StrategyConfig
from trading.signals import SignalEngine
from trading.risk import RiskManager
from trading.strategy import HybridStrategy
from trading.entities import (
    MarketEvent, OrderBookSnapshot, Fill, PlaceQuote, CancelQuote,
    TakerOrder, FlattenInventory,
)


def synth_event(market_id: str, ts_ns: int, mid: float, spread: float = 0.02,
                depth: float = 100.0) -> MarketEvent:
    bid = mid - spread / 2
    ask = mid + spread / 2
    book = OrderBookSnapshot(
        market_id=market_id, token_side="YES",
        best_bid=bid, best_ask=ask,
        bid_depth=depth, ask_depth=depth,
        mid=mid, microprice=mid, spread=spread, ts_ns=ts_ns,
    )
    return MarketEvent(
        ts_ns=ts_ns, market_id=market_id, token_side="YES",
        event_type="book_update", book=book,
    )


def synth_feature_dict(book_mid: float = 0.5, t_since_open_s: float = 60.0) -> dict:
    """Return a minimal feature dict with sensible defaults.

    Models that need features not provided will be zero-filled — for sanity
    testing this produces deterministic but uninformative predictions.
    """
    return {
        "anchor_source": 1,
        "event_count_1s": 5, "event_count_3s": 12, "event_count_5s": 25, "event_count_15s": 80,
        "quote_update_count_1s": 5, "quote_update_count_3s": 12, "quote_update_count_5s": 25,
        "trade_count_1s": 0, "trade_count_3s": 1, "trade_count_5s": 2,
        "signed_trade_size_1s": 0, "signed_trade_size_3s": 5, "signed_trade_size_5s": 10,
        "absolute_trade_size_1s": 0, "absolute_trade_size_3s": 5, "absolute_trade_size_5s": 10,
        "spread_mean_5s": 0.02, "spread_min_5s": 0.01, "spread_max_5s": 0.03,
        "imbalance_mean_5s": 0.0, "imbalance_last": 0.0, "imbalance_slope_5s": 0.0,
        "mid_return_1s": 0.0, "mid_return_3s": 0.0, "mid_return_5s": 0.0,
        "quote_churn_rate_5s": 2.0, "depth_change_rate_5s": 0.0,
        "one_sided_book": 0, "crossed_book": 0, "stale_book": 0,
        "sequence_completeness_rate": 1.0,
        "delta_to_strike": 1000.0, "abs_delta_to_strike": 1000.0, "delta_sign": 1,
        "t_since_open_s": t_since_open_s, "t_to_close_s": 3600.0, "phase_bucket": 2,
        "live_reference_trusted": 1, "live_bias_mode": 0,
        # Cleaned features (model expects these post-cleanup)
        "live_bias_age_log_clipped": 5.0, "live_bias_age_bucket": 1,
        "sequence_length_actual": 50,
        # State and price columns the framework models reference
        "price_level": book_mid, "state_stale_s": 0.1,
        "state_bid_depth_total": 200.0, "state_ask_depth_total": 200.0,
    }


def run_synthetic_session(n_events: int = 100) -> None:
    print("=" * 80)
    print("HybridStrategy synthetic-event sanity simulator")
    print("=" * 80)

    cfg = StrategyConfig.conservative()
    print(f"Config: conservative (kelly={cfg.risk.kelly_fraction}, "
          f"max_daily_loss=${cfg.risk.max_daily_loss_usd}, "
          f"max_concurrent_markets={cfg.selection.max_concurrent_markets})")

    sigs = SignalEngine(artifact_root="artifacts_cleaned")
    risk = RiskManager(cfg.risk)
    strat = HybridStrategy(cfg, sigs, risk)

    market_id = "synth-market-1"
    ts_ns = 1_700_000_000 * 10**9
    mid = 0.50
    rng = np.random.default_rng(42)

    counts = {"PlaceQuote": 0, "CancelQuote": 0, "TakerOrder": 0, "FlattenInventory": 0}
    total_intents = 0

    for i in range(n_events):
        # Random walk on mid
        mid += float(rng.normal(0, 0.001))
        mid = max(0.01, min(0.99, mid))
        ts_ns += 100_000_000  # 100 ms apart

        ev = synth_event(market_id, ts_ns, mid)
        feat = synth_feature_dict(book_mid=mid, t_since_open_s=60 + i * 0.1)
        intents = strat.on_market_event(ev, feat)
        total_intents += len(intents)

        for it in intents:
            counts[type(it).__name__] += 1

        # Simulate a maker fill once in a while
        if isinstance(intents[0] if intents else None, PlaceQuote) and rng.random() < 0.05:
            q = intents[0]
            fill = Fill(
                market_id=market_id, token_side="YES",
                order_side=q.order_side, price=q.price, size=q.size,
                fees_paid=0.0, rebate_earned=0.0001 * q.size,
                client_order_id=q.client_order_id, ts_ns=ts_ns,
            )
            state = strat.market_states[market_id]
            fill_intents = strat.on_fill(fill, state, feat)
            total_intents += len(fill_intents)
            for it in fill_intents:
                counts[type(it).__name__] += 1

    print(f"\nProcessed {n_events} events, emitted {total_intents} intents:")
    for k, v in counts.items():
        print(f"  {k:<18} {v}")

    print(f"\nSession summary:")
    for k, v in risk.session_summary().items():
        print(f"  {k:<28} {v}")

    print(f"\nFinal market state:")
    st = strat.market_states.get(market_id)
    if st:
        print(f"  market_id={st.market_id}")
        print(f"  current mode={strat.modes[st.market_id].value}")
        print(f"  position={risk.position_for(market_id):.0f}")

    print(f"\nDecision log: {len(strat.decision_log)} entries")
    # Print first 5 and last 5 decisions
    for entry in strat.decision_log[:5]:
        print(f"  {entry}")
    if len(strat.decision_log) > 10:
        print(f"  ... ({len(strat.decision_log)-10} more) ...")
    for entry in strat.decision_log[-5:]:
        print(f"  {entry}")

    print("\nSimulator complete.  If counts look sane, the strategy is wired correctly.")


if __name__ == "__main__":
    run_synthetic_session(n_events=100)
