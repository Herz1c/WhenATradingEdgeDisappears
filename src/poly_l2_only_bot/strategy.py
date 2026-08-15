"""Strategy parameters + pure decision function for the VALIDATED final
strategy (poly_l2_only_v2), frozen 2026-06-11.

This is the simple, general, non-overfit strategy validated out-of-sample:
  - ONE entry per market (no re-entries -- rank-2+ entries lose).
  - TTC window (10, 50] s  -- the four buckets near close; the (50,60] bucket
    was dropped (it makes ~0 PnL and has the worst win-rate).
  - Both sides: pick the higher-EV side; require EV/share >= MIN_EDGE.
  - Fill band [0.25, 0.60]  -- buy cheap, where payoff asymmetry is positive.
  - $1 notional for now.
  - GTC taker order (crosses the book, buys what it can, rests the remainder).
    NEVER FAK/FOK.

Out-of-sample over 7 days: +$274, max DD $28 (DD/PnL 0.10), PF 1.72, 7/7 days +.
Survives a 2s execution delay (stays profitable, PF 1.32).

Decision flow per WS frame (called from the bot hot path):
  1. Per-market gate: done flag / max entries reached.
  2. Compute ev_up / ev_down from calibrated p_up.
  3. Pick the higher-EV qualifying side (EV/share >= MIN_EDGE).
  4. Apply the fill-band filter [FILL_MIN, FILL_MAX].
  5. Return Decision or None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Strategy version + tag ──────────────────────────────────────────────────
STRATEGY_VERSION = "poly_l2_only_v2_final_1usd"
STRATEGY_FROZEN_DATE = "2026-06-11"

# ── TTC scanning window: (10, 50] s — drop the (50,60] bucket ────────────────
# Caller skips ticks with ttc_s < TTC_MIN_S or ttc_s >= TTC_MAX_S, which keeps
# exactly the four buckets [10,20),[20,30),[30,40),[40,50) used in the backtest.
TTC_MIN_S = 10.0
TTC_MAX_S = 50.0

# ── Per-market entry rules ──────────────────────────────────────────────────
MAX_ENTRIES_PER_MARKET = 1          # one entry per market — re-entries lose
COOLDOWN_S = 0.0                    # moot with a single entry
COOLDOWN_NS = int(COOLDOWN_S * 1_000_000_000)

# ── Edge + fill band ────────────────────────────────────────────────────────
MIN_EDGE = 0.15                     # EV per share threshold (USD)
FILL_MIN = 0.25
FILL_MAX = 0.60

# ── Position sizing ─────────────────────────────────────────────────────────
NOTIONAL_USD = 1.5                  # $1.50 entries to start

# ── Execution mode ──────────────────────────────────────────────────────────
# GTC taker limit: crosses the book to buy what it can, rests the remainder.
# Explicitly NOT FAK/FOK (those give no fills).
ORDER_MODE_GTC = "gtc"

# Optional execution-time crash-guard: if the price has left the band by the
# time we submit, the bot may skip the order. Kept here as a single knob.
EXEC_CRASH_GUARD = True


@dataclass(slots=True)
class Decision:
    """What a winning tick decides — the bot turns this into an order call."""
    side: str             # "UP" or "DOWN"
    fill_price: float     # best_ask we expect to pay
    ev_per_share: float   # USD EV per share at decision time
    p_model: float        # final calibrated P(Up)
    order_mode: str       # ORDER_MODE_GTC
    zone: str             # ttc-window label, kept for log compatibility
    ttc_s: float
    ts_ns: int
    notional_usd: float = NOTIONAL_USD


@dataclass(slots=True)
class MarketStrategyState:
    """Per-market trading state. Lives next to MarketState in the bot."""
    n_entries: int = 0
    last_entry_ts_ns: int = 0
    done: bool = False        # set True once max entries hit; skips future ticks


def decide(
    *,
    p_up: float,
    best_ask_up: float,
    best_ask_down: float,
    ttc_s: float,
    ts_ns: int,
    strat_state: MarketStrategyState,
) -> Optional[Decision]:
    """The trading decision. Returns Decision to enter, or None to skip.

    Caller has already updated MarketState and confirmed ttc is in
    [TTC_MIN_S, TTC_MAX_S]. Caller marks strat_state on a successful return.
    """
    # Per-market gate.
    if strat_state.done or strat_state.n_entries >= MAX_ENTRIES_PER_MARKET:
        strat_state.done = True
        return None

    # Per-side EV. We buy the higher-EV side if it clears the edge + fill band.
    ev_up = (p_up - best_ask_up) if (0.0 < best_ask_up < 1.0) else -1.0
    ev_dn = ((1.0 - p_up) - best_ask_down) if (0.0 < best_ask_down < 1.0) else -1.0

    if ev_up >= ev_dn:
        side, fill, ev = "UP", best_ask_up, ev_up
    else:
        side, fill, ev = "DOWN", best_ask_down, ev_dn

    if ev < MIN_EDGE:
        return None
    if not (FILL_MIN <= fill <= FILL_MAX):
        return None

    return Decision(
        side=side,
        fill_price=fill,
        ev_per_share=ev,
        p_model=p_up,
        order_mode=ORDER_MODE_GTC,
        zone="ttc_10_50",
        ttc_s=ttc_s,
        ts_ns=ts_ns,
    )
