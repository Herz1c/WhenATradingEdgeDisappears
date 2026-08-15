"""Hybrid maker/taker trading strategy for Polymarket.

Design doc: docs/trading_strategy_design.md

Quick start:
    from trading.config import StrategyConfig
    from trading.signals import SignalEngine
    from trading.risk import RiskManager
    from trading.strategy import HybridStrategy

    cfg = StrategyConfig()  # all conservative defaults
    sigs = SignalEngine(artifact_root="artifacts_cleaned")
    risk = RiskManager(cfg.risk)
    strat = HybridStrategy(cfg, sigs, risk)

    # In your event loop:
    for event in market_event_stream:
        intents = strat.on_market_event(event)
        for intent in intents:
            execute(intent)
"""
from .config import StrategyConfig, RiskConfig, ExecutionConfig
from .entities import (
    MarketEvent, MarketState, OrderIntent, PlaceQuote, CancelQuote,
    TakerOrder, FlattenInventory, Position, Fill, SignalSnapshot, Mode,
)
from .signals import SignalEngine
from .risk import RiskManager
from .strategy import HybridStrategy

__all__ = [
    "StrategyConfig", "RiskConfig", "ExecutionConfig",
    "MarketEvent", "MarketState", "OrderIntent", "PlaceQuote", "CancelQuote",
    "TakerOrder", "FlattenInventory", "Position", "Fill", "SignalSnapshot", "Mode",
    "SignalEngine", "RiskManager", "HybridStrategy",
]
