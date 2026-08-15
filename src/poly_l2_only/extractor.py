"""Pure state + feature extraction for Polymarket BTC up/down 5-min markets.

Frames arrive as dicts (one per WS message). Four event types we care about:
    book              : full book snapshot for one token
    price_change      : incremental update (book also reconstructed in payload)
    last_trade_price  : trade execution
    best_bid_ask      : top-of-book update

Each frame carries `token_outcome` in {"Up", "Down"} — the side this event
belongs to. State maintains a per-token view; features cross-reference both.

No external data. No future leakage. State updates only from frames already
seen at the emit time.

Design notes (perf-driven):
- We do NOT keep the full L2 ladder in state. The recorder already provides
  best_bid/ask, spread, depth_totals, level_counts as scalars, plus the top
  level inside book.bids[0]/asks[0]. That's all we need for the feature set.
- Rolling trade/book-event windows are maintained incrementally (running
  sums + per-window deques) so state_to_features is O(1).
- Floats are parsed at update time and stored as numerics in state so the
  feature emit path does no string conversion.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import math


# ----- constants -------------------------------------------------------------

NS_PER_S = 1_000_000_000
SIDE_UP = "Up"
SIDE_DOWN = "Down"

WIN_1S_NS = 1 * NS_PER_S
WIN_5S_NS = 5 * NS_PER_S
WIN_10S_NS = 10 * NS_PER_S

EMIT_EVENT_TYPES = ("book", "price_change", "last_trade_price", "best_bid_ask")


# ----- safe float parsing (used only in update_state, not in features) ------

def _f(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else default
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ----- per-token state -------------------------------------------------------

@dataclass
class TokenState:
    """Rolling view of one token (Up or Down) within a single market."""
    # Top of book (most-recent values from any reporting event).
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    top_bid_size: float = 0.0
    top_ask_size: float = 0.0

    # Pre-aggregated by the recorder.
    bid_size_total: float = 0.0
    ask_size_total: float = 0.0
    level_count_bid: float = 0.0
    level_count_ask: float = 0.0

    # Last trade summary.
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    last_trade_side: float = 0.0   # +1 BUY, -1 SELL, 0 unknown
    last_trade_ts_ns: int = 0

    # Per-window rolling deques + running sums (size/buy/sell vol, count).
    trades_1s: Deque[Tuple[int, float, float]] = field(default_factory=deque)
    trades_5s: Deque[Tuple[int, float, float]] = field(default_factory=deque)
    trades_10s: Deque[Tuple[int, float, float]] = field(default_factory=deque)
    cnt_1s: int = 0
    cnt_5s: int = 0
    cnt_10s: int = 0
    buy_vol_5s: float = 0.0
    sell_vol_5s: float = 0.0

    # Book event timestamps (1s and 5s rolling counts).
    book_ts_1s: Deque[int] = field(default_factory=deque)
    book_ts_5s: Deque[int] = field(default_factory=deque)
    book_evts_1s: int = 0
    book_evts_5s: int = 0

    last_ts_ns: int = 0
    last_book_state_seq: int = 0


# ----- per-market state ------------------------------------------------------

@dataclass
class MarketState:
    event_id: str = ""
    event_slug: str = ""
    market_id: str = ""
    market_open_s: int = 0
    market_close_s: int = 0
    tick_size: float = 0.01
    neg_risk: float = 0.0

    asset_to_side: Dict[str, str] = field(default_factory=dict)
    up: TokenState = field(default_factory=TokenState)
    down: TokenState = field(default_factory=TokenState)

    # When True, top-of-book (best_bid/ask) is taken ONLY from authoritative
    # `book` snapshots + `best_bid_ask` events, NOT from `price_change`-
    # reconstructed values. The LIVE bot sets this so a dropped price_change
    # delta can't permanently drift its top-of-book (the next best_bid_ask
    # self-heals it). The dataset builder leaves it False, so training data is
    # byte-identical; verified live==dataset top-of-book to mean diff 0.0009.
    bba_authoritative: bool = False


# ----- rolling-window helpers -----------------------------------------------

def _push_trade(t: TokenState, ts: int, size: float, side: float) -> None:
    """Append a trade and update all three rolling windows."""
    # Window 1s
    cutoff = ts - WIN_1S_NS
    dq = t.trades_1s
    while dq and dq[0][0] < cutoff:
        dq.popleft()
        t.cnt_1s -= 1
    dq.append((ts, size, side))
    t.cnt_1s += 1

    # Window 5s (also tracks buy/sell volume)
    cutoff = ts - WIN_5S_NS
    dq = t.trades_5s
    while dq and dq[0][0] < cutoff:
        ots, osize, oside = dq.popleft()
        t.cnt_5s -= 1
        if oside > 0:
            t.buy_vol_5s -= osize
        elif oside < 0:
            t.sell_vol_5s -= osize
    dq.append((ts, size, side))
    t.cnt_5s += 1
    if side > 0:
        t.buy_vol_5s += size
    elif side < 0:
        t.sell_vol_5s += size

    # Window 10s
    cutoff = ts - WIN_10S_NS
    dq = t.trades_10s
    while dq and dq[0][0] < cutoff:
        dq.popleft()
        t.cnt_10s -= 1
    dq.append((ts, size, side))
    t.cnt_10s += 1


def _push_book_event(t: TokenState, ts: int) -> None:
    """Append a book event ts and update 1s & 5s rolling counts."""
    cutoff_1 = ts - WIN_1S_NS
    dq = t.book_ts_1s
    while dq and dq[0] < cutoff_1:
        dq.popleft()
        t.book_evts_1s -= 1
    dq.append(ts)
    t.book_evts_1s += 1

    cutoff_5 = ts - WIN_5S_NS
    dq = t.book_ts_5s
    while dq and dq[0] < cutoff_5:
        dq.popleft()
        t.book_evts_5s -= 1
    dq.append(ts)
    t.book_evts_5s += 1


def _decay_to_now(t: TokenState, now_ns: int) -> None:
    """Expire deque entries whose timestamps are older than `now_ns - window`.
    Used right before reading rolling stats so that a long gap with no trades
    doesn't leave stale counts pinned."""
    # 1s trades
    cutoff = now_ns - WIN_1S_NS
    dq = t.trades_1s
    while dq and dq[0][0] < cutoff:
        dq.popleft()
        t.cnt_1s -= 1
    # 5s trades (+ volumes)
    cutoff = now_ns - WIN_5S_NS
    dq = t.trades_5s
    while dq and dq[0][0] < cutoff:
        ots, osize, oside = dq.popleft()
        t.cnt_5s -= 1
        if oside > 0:
            t.buy_vol_5s -= osize
        elif oside < 0:
            t.sell_vol_5s -= osize
    # 10s trades
    cutoff = now_ns - WIN_10S_NS
    dq = t.trades_10s
    while dq and dq[0][0] < cutoff:
        dq.popleft()
        t.cnt_10s -= 1
    # 1s book events
    cutoff = now_ns - WIN_1S_NS
    dq = t.book_ts_1s
    while dq and dq[0] < cutoff:
        dq.popleft()
        t.book_evts_1s -= 1
    # 5s book events
    cutoff = now_ns - WIN_5S_NS
    dq = t.book_ts_5s
    while dq and dq[0] < cutoff:
        dq.popleft()
        t.book_evts_5s -= 1


# ----- state update ----------------------------------------------------------

def _ensure_market_meta(state: MarketState, frame: Dict[str, Any]) -> None:
    if state.event_id:
        return
    state.event_id = str(frame.get("selected_event_id") or state.event_id or "")
    state.event_slug = str(frame.get("selected_event_slug") or state.event_slug or "")
    state.market_id = str(frame.get("selected_market_id") or frame.get("market") or state.market_id or "")
    state.market_open_s = int(frame.get("market_open_s") or state.market_open_s or 0)
    state.market_close_s = int(frame.get("market_close_s") or state.market_close_s or 0)
    state.tick_size = _f(frame.get("tick_size"), state.tick_size or 0.01)
    nr = frame.get("neg_risk")
    if nr is not None:
        state.neg_risk = 1.0 if nr is True else 0.0
    for aid, out in zip(frame.get("selected_asset_ids") or [],
                        frame.get("selected_outcomes") or []):
        state.asset_to_side[str(aid)] = str(out)


def _token_for_frame(state: MarketState, frame: Dict[str, Any]) -> Optional[TokenState]:
    side = frame.get("token_outcome")
    if side == SIDE_UP:
        return state.up
    if side == SIDE_DOWN:
        return state.down
    aid = str(frame.get("token_asset_id") or "")
    side = state.asset_to_side.get(aid)
    if side == SIDE_UP:
        return state.up
    if side == SIDE_DOWN:
        return state.down
    return None


def update_state(state: MarketState, frame: Dict[str, Any]) -> bool:
    """Mutate `state` from one raw L2 frame. Returns True if consumed.
    Single source of truth — trainer and live bot both call this."""
    et = frame.get("event_type")
    if et not in EMIT_EVENT_TYPES:
        return False
    _ensure_market_meta(state, frame)
    token = _token_for_frame(state, frame)
    if token is None:
        return False

    ts_ns = int(frame.get("recv_ts_ns") or 0)
    token.last_ts_ns = ts_ns

    if et == "book" or et == "price_change":
        # Top-of-book: `book` (snapshot) always authoritative; `price_change`
        # (reconstructed from deltas) is SKIPPED when bba_authoritative so a
        # dropped delta can't drift the bot's top-of-book. Depth/level/book-evt
        # fields below still update from price_change either way.
        if not (state.bba_authoritative and et == "price_change"):
            token.best_bid = _f(frame.get("best_bid"))
            token.best_ask = _f(frame.get("best_ask"))
            token.spread = _f(frame.get("spread"))
        lc = frame.get("level_counts") or {}
        token.level_count_bid = float(lc.get("bid") or 0)
        token.level_count_ask = float(lc.get("ask") or 0)
        dt = frame.get("depth_totals") or {}
        token.bid_size_total = _f(dt.get("bid_size_total"))
        token.ask_size_total = _f(dt.get("ask_size_total"))
        # Top level sizes — read first entry if present.
        book = frame.get("book") or {}
        bids = book.get("bids") or ()
        asks = book.get("asks") or ()
        if bids:
            b0 = bids[0]
            token.top_bid_size = _f(b0.get("size"))
        else:
            token.top_bid_size = 0.0
        if asks:
            a0 = asks[0]
            token.top_ask_size = _f(a0.get("size"))
        else:
            token.top_ask_size = 0.0
        token.last_book_state_seq = int(frame.get("book_state_seq") or 0)
        _push_book_event(token, ts_ns)

    elif et == "best_bid_ask":
        token.best_bid = _f(frame.get("best_bid"))
        token.best_ask = _f(frame.get("best_ask"))
        token.spread = _f(frame.get("spread"))
        bss = frame.get("book_state_seq")
        if bss is not None:
            token.last_book_state_seq = int(bss)
        _push_book_event(token, ts_ns)

    elif et == "last_trade_price":
        price = _f(frame.get("price"))
        size = _f(frame.get("size"))
        side_raw = frame.get("side")
        if side_raw == "BUY":
            side = 1.0
        elif side_raw == "SELL":
            side = -1.0
        else:
            side = 0.0
        token.last_trade_price = price
        token.last_trade_size = size
        token.last_trade_side = side
        token.last_trade_ts_ns = ts_ns
        _push_trade(token, ts_ns, size, side)

    return True


# ----- feature extraction (O(1) per emit) -----------------------------------

# Column list defined first so we can build dicts by direct assignment.
FEATURE_COLUMNS: Tuple[str, ...] = (
    # Up token
    "up_best_bid", "up_best_ask", "up_mid", "up_spread", "up_microprice",
    "up_top_bid_size", "up_top_ask_size",
    "up_depth_total_bid", "up_depth_total_ask", "up_depth_imbalance",
    "up_level_count_bid", "up_level_count_ask",
    "up_last_trade_price", "up_last_trade_size", "up_last_trade_side",
    "up_last_trade_age_s",
    "up_trade_count_1s", "up_trade_count_5s", "up_trade_count_10s",
    "up_trade_buy_vol_5s", "up_trade_sell_vol_5s",
    "up_book_evts_1s", "up_book_evts_5s",
    # Down token
    "down_best_bid", "down_best_ask", "down_mid", "down_spread", "down_microprice",
    "down_top_bid_size", "down_top_ask_size",
    "down_depth_total_bid", "down_depth_total_ask", "down_depth_imbalance",
    "down_level_count_bid", "down_level_count_ask",
    "down_last_trade_price", "down_last_trade_size", "down_last_trade_side",
    "down_last_trade_age_s",
    "down_trade_count_1s", "down_trade_count_5s", "down_trade_count_10s",
    "down_trade_buy_vol_5s", "down_trade_sell_vol_5s",
    "down_book_evts_1s", "down_book_evts_5s",
    # Market / cross-token
    "ttc_s", "ttc_log", "tick_size", "neg_risk",
    "mid_sum", "mid_skew", "mid_up_implied",
    "bb_sum", "ba_sum", "spread_sum", "spread_diff",
    "implied_p_up", "last_trade_arb_gap",
    "trade_count_5s_total", "book_evts_5s_total",
)


def state_to_features(state: MarketState, now_ns: int) -> Dict[str, float]:
    """Build the full feature vector at time `now_ns`. O(1) — pure reads off
    the running counters plus a final decay pass to expire stale window entries
    when this emit is far past the last event of a token."""
    up = state.up
    dn = state.down
    _decay_to_now(up, now_ns)
    _decay_to_now(dn, now_ns)

    # ---- per-token features ----
    up_bb, up_ba = up.best_bid, up.best_ask
    up_mid = (up_bb + up_ba) * 0.5 if (up_bb > 0 and up_ba > 0) else 0.0
    up_spread = up.spread if up.spread > 0 else (up_ba - up_bb if up_ba > up_bb > 0 else 0.0)
    up_top_b, up_top_a = up.top_bid_size, up.top_ask_size
    up_micro = (up_bb * up_top_a + up_ba * up_top_b) / (up_top_b + up_top_a) \
        if (up_top_b + up_top_a) > 0 and up_bb > 0 and up_ba > 0 else 0.0
    up_dt = up.bid_size_total + up.ask_size_total
    up_dimb = ((up.bid_size_total - up.ask_size_total) / up_dt) if up_dt > 0 else 0.0
    up_trade_age = ((now_ns - up.last_trade_ts_ns) / NS_PER_S) if up.last_trade_ts_ns else 999.0
    if up_trade_age > 999.0:
        up_trade_age = 999.0

    dn_bb, dn_ba = dn.best_bid, dn.best_ask
    dn_mid = (dn_bb + dn_ba) * 0.5 if (dn_bb > 0 and dn_ba > 0) else 0.0
    dn_spread = dn.spread if dn.spread > 0 else (dn_ba - dn_bb if dn_ba > dn_bb > 0 else 0.0)
    dn_top_b, dn_top_a = dn.top_bid_size, dn.top_ask_size
    dn_micro = (dn_bb * dn_top_a + dn_ba * dn_top_b) / (dn_top_b + dn_top_a) \
        if (dn_top_b + dn_top_a) > 0 and dn_bb > 0 and dn_ba > 0 else 0.0
    dn_dt = dn.bid_size_total + dn.ask_size_total
    dn_dimb = ((dn.bid_size_total - dn.ask_size_total) / dn_dt) if dn_dt > 0 else 0.0
    dn_trade_age = ((now_ns - dn.last_trade_ts_ns) / NS_PER_S) if dn.last_trade_ts_ns else 999.0
    if dn_trade_age > 999.0:
        dn_trade_age = 999.0

    # ---- meta ----
    close_ns = state.market_close_s * NS_PER_S
    ttc_s = (close_ns - now_ns) / NS_PER_S
    if ttc_s < 0:
        ttc_s = 0.0
    ttc_log = math.log1p(ttc_s)

    # ---- cross-token ----
    mid_sum = up_mid + dn_mid
    mid_skew = up_mid - dn_mid
    mid_up_implied = up_mid - (1.0 - dn_mid)
    bb_sum = up_bb + dn_bb
    ba_sum = up_ba + dn_ba
    spread_sum = up_spread + dn_spread
    spread_diff = up_spread - dn_spread
    if up_mid > 0 and dn_mid > 0:
        implied_p_up = (up_mid + (1.0 - dn_mid)) * 0.5
    elif up_mid > 0:
        implied_p_up = up_mid
    elif dn_mid > 0:
        implied_p_up = 1.0 - dn_mid
    else:
        implied_p_up = 0.5
    if up.last_trade_price > 0 and dn.last_trade_price > 0:
        last_trade_arb_gap = up.last_trade_price - (1.0 - dn.last_trade_price)
    else:
        last_trade_arb_gap = 0.0

    feats = {
        "up_best_bid": up_bb,
        "up_best_ask": up_ba,
        "up_mid": up_mid,
        "up_spread": up_spread,
        "up_microprice": up_micro,
        "up_top_bid_size": up_top_b,
        "up_top_ask_size": up_top_a,
        "up_depth_total_bid": up.bid_size_total,
        "up_depth_total_ask": up.ask_size_total,
        "up_depth_imbalance": up_dimb,
        "up_level_count_bid": up.level_count_bid,
        "up_level_count_ask": up.level_count_ask,
        "up_last_trade_price": up.last_trade_price,
        "up_last_trade_size": up.last_trade_size,
        "up_last_trade_side": up.last_trade_side,
        "up_last_trade_age_s": up_trade_age,
        "up_trade_count_1s": float(up.cnt_1s),
        "up_trade_count_5s": float(up.cnt_5s),
        "up_trade_count_10s": float(up.cnt_10s),
        "up_trade_buy_vol_5s": up.buy_vol_5s,
        "up_trade_sell_vol_5s": up.sell_vol_5s,
        "up_book_evts_1s": float(up.book_evts_1s),
        "up_book_evts_5s": float(up.book_evts_5s),
        "down_best_bid": dn_bb,
        "down_best_ask": dn_ba,
        "down_mid": dn_mid,
        "down_spread": dn_spread,
        "down_microprice": dn_micro,
        "down_top_bid_size": dn_top_b,
        "down_top_ask_size": dn_top_a,
        "down_depth_total_bid": dn.bid_size_total,
        "down_depth_total_ask": dn.ask_size_total,
        "down_depth_imbalance": dn_dimb,
        "down_level_count_bid": dn.level_count_bid,
        "down_level_count_ask": dn.level_count_ask,
        "down_last_trade_price": dn.last_trade_price,
        "down_last_trade_size": dn.last_trade_size,
        "down_last_trade_side": dn.last_trade_side,
        "down_last_trade_age_s": dn_trade_age,
        "down_trade_count_1s": float(dn.cnt_1s),
        "down_trade_count_5s": float(dn.cnt_5s),
        "down_trade_count_10s": float(dn.cnt_10s),
        "down_trade_buy_vol_5s": dn.buy_vol_5s,
        "down_trade_sell_vol_5s": dn.sell_vol_5s,
        "down_book_evts_1s": float(dn.book_evts_1s),
        "down_book_evts_5s": float(dn.book_evts_5s),
        "ttc_s": ttc_s,
        "ttc_log": ttc_log,
        "tick_size": state.tick_size,
        "neg_risk": state.neg_risk,
        "mid_sum": mid_sum,
        "mid_skew": mid_skew,
        "mid_up_implied": mid_up_implied,
        "bb_sum": bb_sum,
        "ba_sum": ba_sum,
        "spread_sum": spread_sum,
        "spread_diff": spread_diff,
        "implied_p_up": implied_p_up,
        "last_trade_arb_gap": last_trade_arb_gap,
        "trade_count_5s_total": float(up.cnt_5s + dn.cnt_5s),
        "book_evts_5s_total": float(up.book_evts_5s + dn.book_evts_5s),
    }
    return feats
