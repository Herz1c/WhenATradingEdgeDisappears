"""Data classes shared across the trading module.

All types are immutable (frozen dataclasses) where possible to make the
strategy easier to reason about and to enable safe backtesting/replay.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal

# ── Identifiers and primitives ────────────────────────────────────────────────

MarketId = str       # e.g. "btc-above-80k-may-2026"
TokenSide = Literal["YES", "NO"]
OrderSide = Literal["BID", "ASK"]   # BID = we're buying, ASK = we're selling


# ── Market and book state ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderBookSnapshot:
    """Top-of-book state for one (market, token side) pair."""
    market_id: MarketId
    token_side: TokenSide
    best_bid: float
    best_ask: float
    bid_depth: float
    ask_depth: float
    mid: float                  # (bid + ask) / 2
    microprice: float           # depth-weighted mid
    spread: float               # ask - bid
    ts_ns: int


@dataclass
class MarketState:
    """Mutable per-market state the strategy maintains between events.

    Updated on every event, read every time we make a decision.
    """
    market_id: MarketId
    token_side: TokenSide
    book: OrderBookSnapshot
    t_since_open_s: float
    t_to_close_s: float
    phase_bucket: int                  # 0=preopen, 1=open, 2=mid, 3=late, 4=close

    # Our state in this market
    position: float = 0.0              # signed: positive = long YES (or NO)
    avg_entry_price: float = 0.0
    open_quote_ids: dict = field(default_factory=dict)
    last_quote_ts_ns: int = 0
    blackout_until_ns: int = 0
    consecutive_losses: int = 0
    realized_pnl: float = 0.0
    n_fills_today: int = 0


@dataclass(frozen=True)
class MarketEvent:
    """A single event from the market data stream."""
    ts_ns: int
    market_id: MarketId
    token_side: TokenSide
    event_type: Literal["book_update", "trade", "top_of_book"]
    book: OrderBookSnapshot
    # For trades:
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[Literal["BUY", "SELL"]] = None


# ── Signals snapshot ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalSnapshot:
    """All model outputs for one market at one instant.

    Computed by SignalEngine just before each strategy decision.  Any field
    can be None if the model can't score at this moment (e.g. insufficient
    history).  Strategy code MUST handle None gracefully.
    """
    market_id: MarketId
    ts_ns: int

    # Pricing
    fair_value: Optional[float] = None              # from model 02
    mispricing_predicted: Optional[float] = None    # from model 06 (signed)

    # Execution
    fill_prob_bid: Optional[float] = None           # from model 03 at our bid
    fill_prob_ask: Optional[float] = None           # from model 03 at our ask

    # Risk (defense)
    p_severe_7c_3s: Optional[float] = None          # from 08c v3 agg+event
    p_moderate_5c_5s: Optional[float] = None
    p_extreme_10c_5s: Optional[float] = None

    # Direction
    p_up_5s: Optional[float] = None                 # from model 07

    # Late-day
    p_closing_flip: Optional[float] = None          # from model 05

    # Post-fill
    expected_markout_1s: Optional[float] = None     # from model 04
    expected_markout_to_close: Optional[float] = None # from model 04b


# ── Strategy modes ────────────────────────────────────────────────────────────

class Mode(Enum):
    QUIET     = "QUIET"
    QUOTING   = "QUOTING"
    POST_FILL = "POST_FILL"
    TAKER     = "TAKER"
    BLACKOUT  = "BLACKOUT"


# ── Action intents (what the strategy emits) ──────────────────────────────────

@dataclass(frozen=True)
class PlaceQuote:
    market_id: MarketId
    token_side: TokenSide
    order_side: OrderSide
    price: float
    size: float
    time_in_force_ms: int = 5_000
    post_only: bool = True
    client_order_id: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class CancelQuote:
    market_id: MarketId
    client_order_id: str
    rationale: str = ""


@dataclass(frozen=True)
class TakerOrder:
    market_id: MarketId
    token_side: TokenSide
    order_side: OrderSide
    price: float                  # limit price, willing to cross to this
    size: float
    client_order_id: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class FlattenInventory:
    """Immediately cross the spread to flatten a position."""
    market_id: MarketId
    token_side: TokenSide
    target_position: float = 0.0
    rationale: str = ""


# Union type for the strategy's output
OrderIntent = PlaceQuote | CancelQuote | TakerOrder | FlattenInventory


# ── Fill / execution feedback ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Fill:
    market_id: MarketId
    token_side: TokenSide
    order_side: OrderSide
    price: float
    size: float
    fees_paid: float
    rebate_earned: float
    client_order_id: str
    ts_ns: int


@dataclass(frozen=True)
class Position:
    market_id: MarketId
    token_side: TokenSide
    size: float
    avg_entry_price: float
    realized_pnl: float
    unrealized_pnl: float
