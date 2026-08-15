"""HybridStrategy — the policy that turns signals into order intents.

Design priorities (in order):
  1. Don't lose money beyond risk limits     (defensive bias)
  2. Capture maker rebates                   (steady income)
  3. Pull quotes before adverse moves        (08c v3 is the killer signal)
  4. Take only when clearly +EV after fees   (rare, opportunistic)
  5. Flatten on signal of toxic fill         (04 adverse selection)

Strategy flow per market event:
  1. Update market state with new event
  2. Compute SignalSnapshot via SignalEngine
  3. Check entry filters (market eligibility)
  4. Compute desired action based on current mode + signals
  5. Pass each candidate intent through RiskManager.check()
  6. Emit only passing intents
  7. Log everything (decision + rationale)
"""
from __future__ import annotations
import math
import time
from collections import defaultdict
from typing import Optional

from .entities import (
    MarketEvent, MarketState, SignalSnapshot, OrderIntent,
    PlaceQuote, CancelQuote, TakerOrder, FlattenInventory, Fill, Mode,
    MarketId,
)
from .config import StrategyConfig
from .signals import SignalEngine
from .risk import RiskManager


class HybridStrategy:
    """Maker-first hybrid strategy with defensive cancels and rare taker takes."""

    def __init__(self, cfg: StrategyConfig, signals: SignalEngine, risk: RiskManager):
        self.cfg = cfg
        self.signals = signals
        self.risk = risk
        # Per-market mode and state
        self.modes: dict[MarketId, Mode] = defaultdict(lambda: Mode.QUIET)
        self.market_states: dict[MarketId, MarketState] = {}
        self.cooldown_until_ns: dict[MarketId, int] = defaultdict(int)
        self.last_quote_intent_ns: dict[MarketId, int] = defaultdict(int)
        self.decision_log: list[dict] = []

    # ── Public entry points ──────────────────────────────────────────────────

    def on_market_event(self, event: MarketEvent,
                        feature_dict: dict[str, float],
                        event_features: Optional[dict[str, float]] = None,
                        ) -> list[OrderIntent]:
        """Process a market event. Return all order intents to be sent."""
        # 1. Update internal market state
        state = self._update_market_state(event)

        # 2. Get signals
        sigs = self.signals.predict(state, feature_dict, event_features)

        # 3. Check defense FIRST — any severe / extreme signal triggers cancel
        intents: list[OrderIntent] = []
        defense_intents = self._check_defense(state, sigs)
        if defense_intents:
            intents.extend(defense_intents)
            # If hard defense fired, freeze quoting for cooldown_seconds
            if any(isinstance(i, CancelQuote) and "extreme" in i.rationale.lower()
                   for i in defense_intents):
                self.cooldown_until_ns[state.market_id] = (
                    event.ts_ns + self.cfg.signals.defense_cooldown_seconds * 1_000_000_000
                )
                self.modes[state.market_id] = Mode.BLACKOUT

        # 4. If in cooldown / blackout, no further action
        if event.ts_ns < self.cooldown_until_ns[state.market_id]:
            return self._filter(intents)
        if self.modes[state.market_id] == Mode.BLACKOUT:
            # Cooldown elapsed — return to QUIET
            self.modes[state.market_id] = Mode.QUIET

        # 5. Mode-specific logic
        mode = self.modes[state.market_id]
        if mode == Mode.QUIET:
            intents.extend(self._consider_entering_quoting(state, sigs))
        elif mode == Mode.QUOTING:
            intents.extend(self._maintain_quotes(state, sigs))
        elif mode == Mode.POST_FILL:
            intents.extend(self._manage_post_fill(state, sigs))

        # 6. Always check taker opportunity (rate-limited inside check)
        taker_intent = self._consider_taker(state, sigs)
        if taker_intent:
            intents.append(taker_intent)

        return self._filter(intents)

    def on_fill(self, fill: Fill, state: MarketState,
                feature_dict: dict[str, float]) -> list[OrderIntent]:
        """Process a fill. May trigger post-fill management."""
        self.risk.on_fill(fill)
        # If position size > 0, we are now in POST_FILL mode
        if abs(self.risk.position_for(fill.market_id)) > 1e-9:
            self.modes[fill.market_id] = Mode.POST_FILL
            sigs = self.signals.predict(state, feature_dict)
            return self._filter(self._manage_post_fill(state, sigs))
        else:
            # Position flat (this was a closing fill)
            self.modes[fill.market_id] = Mode.QUIET
            return []

    def on_clock_tick(self, ts_ns: int) -> list[OrderIntent]:
        """Periodic check.  For now: nothing to do beyond event-driven logic.

        Future: could detect stale quotes (TIF expired but no cancel ack yet).
        """
        return []

    # ── Internal: defense ────────────────────────────────────────────────────

    def _check_defense(self, state: MarketState, sigs: SignalSnapshot) -> list[OrderIntent]:
        """Trigger cancels based on 08c models and late-day flip signal."""
        intents: list[OrderIntent] = []
        cfg = self.cfg.signals

        # Extreme (highest priority)
        if (sigs.p_extreme_10c_5s is not None
            and sigs.p_extreme_10c_5s >= cfg.extreme_freeze_threshold_frac):
            intents.extend(self._cancel_all_quotes(state,
                rationale=f"08c extreme={sigs.p_extreme_10c_5s:.3f}>={cfg.extreme_freeze_threshold_frac}"))
            return intents

        # Severe
        if (sigs.p_severe_7c_3s is not None
            and sigs.p_severe_7c_3s >= cfg.severe_defense_threshold_frac):
            intents.extend(self._cancel_all_quotes(state,
                rationale=f"08c severe={sigs.p_severe_7c_3s:.3f}>={cfg.severe_defense_threshold_frac}"))
            return intents

        # Closing flip (only in last N seconds)
        if (state.t_to_close_s <= cfg.closing_flip_window_seconds
            and sigs.p_closing_flip is not None
            and sigs.p_closing_flip >= cfg.closing_flip_defense_threshold_frac):
            intents.extend(self._cancel_all_quotes(state,
                rationale=f"closing flip={sigs.p_closing_flip:.3f} in last {state.t_to_close_s:.0f}s"))
            return intents

        return intents

    def _cancel_all_quotes(self, state: MarketState, rationale: str) -> list[CancelQuote]:
        if not state.open_quote_ids:
            return []
        return [CancelQuote(market_id=state.market_id, client_order_id=oid, rationale=rationale)
                for oid in state.open_quote_ids]

    # ── Internal: quoting ────────────────────────────────────────────────────

    def _consider_entering_quoting(self, state: MarketState,
                                   sigs: SignalSnapshot) -> list[OrderIntent]:
        """Decide whether to start quoting this market."""
        if not self._passes_entry_filters(state, sigs):
            return []
        self.modes[state.market_id] = Mode.QUOTING
        return self._maintain_quotes(state, sigs)

    def _maintain_quotes(self, state: MarketState,
                         sigs: SignalSnapshot) -> list[OrderIntent]:
        """Place / refresh quotes centered around the best available anchor.

        Use model 02 fair value if available AND within max drift; otherwise
        fall back to the book mid.  A real MM should always be quoting *something*
        — defending against the worst case of bad fair-value estimates by
        falling back to the mid is far better than refusing to quote.
        """
        # Rate-limit re-quoting
        if (state.book.ts_ns - self.last_quote_intent_ns[state.market_id]
            < self.cfg.execution.requote_min_interval_ms * 1_000_000):
            return []

        cfg = self.cfg
        tick = cfg.execution.tick_size
        mid = state.book.mid

        # Use fair value if reliable, else fall back to mid
        if (sigs.fair_value is not None
            and abs(sigs.fair_value - mid) <= cfg.signals.max_fair_value_drift_ticks * tick):
            F = sigs.fair_value
        else:
            F = mid  # FALLBACK — quote around the mid

        # Compute base quote prices: F ± quote_edge_ticks
        bid_price = round((F - cfg.signals.quote_edge_ticks * tick) / tick) * tick
        ask_price = round((F + cfg.signals.quote_edge_ticks * tick) / tick) * tick

        # Apply direction skew (model 07)
        if (sigs.p_up_5s is not None
            and abs(sigs.p_up_5s - 0.5) > cfg.signals.direction_skew_threshold_frac - 0.5):
            skew = cfg.signals.direction_skew_ticks * tick * (1 if sigs.p_up_5s > 0.5 else -1)
            bid_price += skew
            ask_price += skew

        # Soft defense (moderate signal): widen on the weak side
        if (sigs.p_moderate_5c_5s is not None
            and sigs.p_moderate_5c_5s >= cfg.signals.moderate_lean_threshold_frac):
            # Direction not specified by 08c moderate — widen both sides slightly
            bid_price -= 0.5 * tick
            ask_price += 0.5 * tick

        # Don't quote inside the spread (no crossing accidentally)
        bid_price = min(bid_price, state.book.best_bid)
        ask_price = max(ask_price, state.book.best_ask)

        # Compute expected edge per side
        edge_bid = self._compute_edge(state, sigs, "BID", bid_price)
        edge_ask = self._compute_edge(state, sigs, "ASK", ask_price)

        # Directional gate using model 04b (markout-to-close).
        # When enabled, we only quote on the side that model 04b says is
        # profitable to hold (BID if E[markout] > +gate; ASK if < -gate).
        gate = cfg.signals.markout_to_close_directional_gate_usd
        allow_bid = True
        allow_ask = True
        if gate > 0.0:
            em_close = sigs.expected_markout_to_close
            if em_close is None:
                allow_bid = allow_ask = False
            else:
                allow_bid = (em_close >= +gate)
                allow_ask = (em_close <= -gate)

        intents: list[OrderIntent] = []
        # In uniform-size (market-maker) mode, by default skip the edge-required
        # check — we want to be on both sides whenever the book exists.  If
        # `enforce_edge_check_in_uniform_mode` is True, gate on edge anyway
        # (useful when adverse selection makes always-quoting unprofitable).
        skip_edge_check = (cfg.risk.uniform_quote_size is not None
                           and cfg.risk.uniform_quote_size > 0
                           and not cfg.risk.enforce_edge_check_in_uniform_mode)
        if allow_bid and (skip_edge_check or edge_bid >= cfg.risk.min_edge_required_ticks * tick):
            size_bid = self._compute_size(state, sigs, "BID", edge_bid)
            if size_bid > 0:
                intents.append(PlaceQuote(
                    market_id=state.market_id,
                    token_side=state.token_side,
                    order_side="BID",
                    price=bid_price,
                    size=size_bid,
                    time_in_force_ms=cfg.execution.default_quote_time_in_force_ms,
                    post_only=True,
                    client_order_id=self._gen_oid(state, "BID"),
                    rationale=f"F={F:.4f} edge={edge_bid:.4f}",
                ))
        if allow_ask and (skip_edge_check or edge_ask >= cfg.risk.min_edge_required_ticks * tick):
            size_ask = self._compute_size(state, sigs, "ASK", edge_ask)
            if size_ask > 0:
                intents.append(PlaceQuote(
                    market_id=state.market_id,
                    token_side=state.token_side,
                    order_side="ASK",
                    price=ask_price,
                    size=size_ask,
                    time_in_force_ms=cfg.execution.default_quote_time_in_force_ms,
                    post_only=True,
                    client_order_id=self._gen_oid(state, "ASK"),
                    rationale=f"F={F:.4f} edge={edge_ask:.4f}",
                ))

        if intents:
            self.last_quote_intent_ns[state.market_id] = state.book.ts_ns
        return intents

    # ── Internal: post-fill management ───────────────────────────────────────

    def _manage_post_fill(self, state: MarketState, sigs: SignalSnapshot) -> list[OrderIntent]:
        """After a fill, decide hold vs flatten based on 04 adverse selection."""
        cfg = self.cfg.signals
        pos = self.risk.position_for(state.market_id)
        if abs(pos) < 1e-9:
            self.modes[state.market_id] = Mode.QUIET
            return []

        # If adverse selection predicts a bad markout, flatten now
        if (sigs.expected_markout_1s is not None
            and sigs.expected_markout_1s <= cfg.flatten_if_predicted_markout_below_ticks
                                            * self.cfg.execution.tick_size):
            return [FlattenInventory(
                market_id=state.market_id,
                token_side=state.token_side,
                target_position=0.0,
                rationale=f"04 markout1s={sigs.expected_markout_1s:.4f} below threshold",
            )]
        # Otherwise hold — return to QUOTING mode but with no opposite quote until flat
        return []

    # ── Internal: taker opportunity ──────────────────────────────────────────

    def _consider_taker(self, state: MarketState,
                        sigs: SignalSnapshot) -> Optional[TakerOrder]:
        """Cross the spread only if model 06 mispricing exceeds fees + safety."""
        if sigs.mispricing_predicted is None:
            return None
        cfg = self.cfg
        tick = cfg.execution.tick_size

        # Required edge: must overcome (taker_fee + half_spread + safety_margin)
        half_spread = state.book.spread / 2
        required_edge = (cfg.execution.taker_fee_per_contract_ticks * tick
                         + half_spread
                         + cfg.signals.taker_min_edge_ticks * tick)

        delta = sigs.mispricing_predicted
        if abs(delta) < required_edge:
            return None

        # Only take if fill prob is high
        fp = sigs.fill_prob_bid if delta > 0 else sigs.fill_prob_ask
        if fp is None or fp < cfg.signals.taker_min_fill_prob:
            return None

        side = "BID" if delta > 0 else "ASK"
        price = state.book.best_ask if side == "BID" else state.book.best_bid
        size = min(cfg.risk.max_size_per_quote_contracts,
                   self.risk.headroom_per_market(state.market_id))
        if size <= 0:
            return None

        return TakerOrder(
            market_id=state.market_id,
            token_side=state.token_side,
            order_side=side,
            price=price,
            size=size,
            client_order_id=self._gen_oid(state, f"taker_{side}"),
            rationale=f"06 mispricing={delta:+.4f} > required={required_edge:.4f}",
        )

    # ── Internal: edge & sizing ──────────────────────────────────────────────

    def _compute_edge(self, state: MarketState, sigs: SignalSnapshot,
                      side: str, price: float) -> float:
        """Expected edge per contract in price units.

        BID (we buy): we profit when price rises after fill, lose when it
            falls — adverse_cost = max(0, -E[markout_1s]).
        ASK (we sell): we profit when price falls, lose when it rises —
            adverse_cost = max(0, +E[markout_1s]).
        Edge = half_spread - adverse_cost + rebate.
        """
        tick = self.cfg.execution.tick_size
        half_spread = state.book.spread / 2
        em = sigs.expected_markout_1s if sigs.expected_markout_1s is not None else 0.0
        if side == "BID":
            adverse_cost = max(0.0, -em)
        else:  # ASK
            adverse_cost = max(0.0, +em)
        rebate = self.cfg.execution.maker_rebate_per_contract_ticks * tick
        return half_spread - adverse_cost + rebate

    def _compute_size(self, state: MarketState, sigs: SignalSnapshot,
                      side: str, edge: float) -> float:
        """Position sizing.

        Mode A (uniform_quote_size set): use fixed size, with vol scaler + headroom.
        Mode B (Kelly): fractional Kelly × edge/variance.
        """
        cfg = self.cfg

        # Mode A — uniform size (market-maker style)
        uniform_mode = (cfg.risk.uniform_quote_size is not None
                        and cfg.risk.uniform_quote_size > 0)
        if uniform_mode:
            base = float(cfg.risk.uniform_quote_size)
        else:
            # Mode B — Kelly
            if edge <= 0:
                return 0.0
            variance = max(1e-6, state.book.spread ** 2)
            kelly_size = cfg.risk.kelly_fraction * edge / variance
            base = min(cfg.risk.max_size_per_quote_contracts, kelly_size)

        # Volatility scaler — reduce size when 08c moderate is elevated.
        # Skip in uniform mode: any scale < 1.0 + floor() would zero a 1-contract
        # quote, which is not the intent ("widen quotes" is handled separately
        # via moderate_lean_threshold_frac in _maintain_quotes).
        if (cfg.risk.use_volatility_scaler
                and sigs.p_moderate_5c_5s is not None
                and not uniform_mode):
            vol_scale = 1.0 / (1.0 + cfg.risk.vol_scaler_strength * sigs.p_moderate_5c_5s)
            base *= vol_scale

        # Honor headroom
        base = min(base, cfg.risk.max_size_per_quote_contracts)
        base = min(base, self.risk.headroom_per_market(state.market_id))
        base = min(base, self.risk.global_headroom())

        # Uniform mode rounds (so vol-adjacent fractional sizes stay sane);
        # Kelly mode floors (truncation is correct for "don't bother if tiny").
        if uniform_mode:
            return max(0.0, float(round(base)))
        return max(0.0, math.floor(base))

    # ── Internal: filters ────────────────────────────────────────────────────

    def _passes_entry_filters(self, state: MarketState, sigs: SignalSnapshot) -> bool:
        """Minimal sanity checks before quoting.

        For a market-maker strategy: the DEFAULT is to quote.  We only refuse
        in clearly broken conditions (no book, late-day, defense fired).  All
        the "is this a great spot?" logic happens in sizing and pricing, not
        in the entry gate.
        """
        cfg = self.cfg
        tick = cfg.execution.tick_size

        # Late-day cutoff
        if state.t_to_close_s <= cfg.risk.no_quote_in_last_n_seconds:
            return False

        # Book must exist (positive spread, positive depth)
        if state.book.spread <= 0 or state.book.spread > cfg.selection.max_book_spread_to_quote_ticks * tick:
            return False
        if state.book.bid_depth < cfg.selection.require_min_book_depth_contracts:
            return False
        if state.book.ask_depth < cfg.selection.require_min_book_depth_contracts:
            return False

        # Active defense — don't enter a market while 08c severe is hot
        if (sigs.p_severe_7c_3s is not None
            and sigs.p_severe_7c_3s >= cfg.signals.severe_defense_threshold_frac):
            return False
        return True

    # ── Internal: bookkeeping ────────────────────────────────────────────────

    def _update_market_state(self, event: MarketEvent) -> MarketState:
        """Maintain a MarketState per market_id."""
        st = self.market_states.get(event.market_id)
        if st is None:
            st = MarketState(
                market_id=event.market_id,
                token_side=event.token_side,
                book=event.book,
                t_since_open_s=0.0,
                t_to_close_s=86400.0,
                phase_bucket=1,
            )
            self.market_states[event.market_id] = st
        else:
            st.book = event.book
        return st

    def _gen_oid(self, state: MarketState, suffix: str) -> str:
        return f"{state.market_id}_{state.book.ts_ns}_{suffix}"

    def _filter(self, intents: list[OrderIntent]) -> list[OrderIntent]:
        """Apply risk-manager check to each intent; drop rejected; log all."""
        out: list[OrderIntent] = []
        for i in intents:
            res = self.risk.check(i)
            if res.allowed:
                if isinstance(i, PlaceQuote):
                    self.risk.on_quote_placed()
                elif isinstance(i, CancelQuote):
                    self.risk.on_quote_cancelled()
                elif isinstance(i, TakerOrder):
                    self.risk.on_taker()
                out.append(i)
                if self.cfg.log_every_decision:
                    self._log("emit", None, intent=type(i).__name__, rationale=i.rationale if hasattr(i, "rationale") else "")
            else:
                if self.cfg.log_skipped_quotes:
                    self._log("reject", None, intent=type(i).__name__, reason=res.reason)
        return out

    def _log(self, kind: str, state: Optional[MarketState], **kw) -> None:
        entry = {"kind": kind, "ts": time.time()}
        if state is not None:
            entry["market_id"] = state.market_id
        entry.update(kw)
        self.decision_log.append(entry)
