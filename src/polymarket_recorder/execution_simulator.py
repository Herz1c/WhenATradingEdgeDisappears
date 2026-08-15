from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .execution_fee_schedule import FeeSchedule
from .execution_markout import compute_markout_value
from .execution_replay import BookState, ExecutionReplayIndex, SECOND_NS, TokenReplay


def _nan() -> float:
    return float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def _round_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return round(float(price), 6)
    return round(round(float(price) / tick_size) * tick_size, 6)


def generate_order_intent_id(
    *,
    market_slug: str,
    token_asset_id: str,
    source_sequence_id: str = "",
    order_ts_ns: int,
    order_side: str,
    limit_price: float,
    order_size: float,
    cancel_after_ns: int | None,
    price_level: str,
) -> str:
    key = "|".join(
        [
            str(market_slug),
            str(token_asset_id),
            str(source_sequence_id or ""),
            str(int(order_ts_ns)),
            str(order_side).lower(),
            f"{float(limit_price):.6f}",
            f"{float(order_size):.6f}",
            "" if cancel_after_ns is None else str(int(cancel_after_ns)),
            str(price_level),
        ]
    )
    return hashlib.blake2b(key.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionSimulatorConfig:
    fee_bps: float = 0.0
    rebate_bps: float = 0.0
    fee_schedule: FeeSchedule | None = None
    cancel_latency_ns: int = 0
    replace_latency_ns: int = 0
    stale_book_threshold_ns: int = 30 * SECOND_NS
    default_tick_size: float = 0.01
    queue_ahead_fraction: float = 0.0
    allow_unknown_trade_side_fill: bool = True
    markout_horizons_s: tuple[int, ...] = (1, 3, 5, 15)
    post_only_reject_crossing: bool = True
    require_live_reference: bool = False

    def resolved_fee_schedule(self) -> FeeSchedule:
        if self.fee_schedule is not None:
            return self.fee_schedule
        return FeeSchedule()


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    market_slug: str
    token_asset_id: str
    token_side_normalized: str
    order_ts_ns: int
    market_close_ts_ns: int
    market_deploy_ts_ns: int
    order_side: str
    limit_price: float
    order_size: float
    price_level: str
    post_only: bool = True
    cancel_after_ns: int | None = None
    replace_after_ns: int | None = None
    replace_limit_price: float | None = None
    replace_order_size: float | None = None
    replace_price_level: str = ""
    replace_post_only: bool = True
    holding_policy: str = "cancel_or_close"
    source_sequence_id: str = ""
    resolved_side: str = ""
    scenario_name: str = "grid_v1"

    @classmethod
    def create(
        cls,
        *,
        market_slug: str,
        token_asset_id: str,
        token_side_normalized: str,
        order_ts_ns: int,
        market_close_ts_ns: int,
        order_side: str,
        limit_price: float,
        order_size: float,
        price_level: str,
        market_deploy_ts_ns: int | None = None,
        post_only: bool = True,
        cancel_after_ns: int | None = None,
        replace_after_ns: int | None = None,
        replace_limit_price: float | None = None,
        replace_order_size: float | None = None,
        replace_price_level: str = "",
        replace_post_only: bool = True,
        holding_policy: str = "cancel_or_close",
        source_sequence_id: str = "",
        resolved_side: str = "",
        scenario_name: str = "grid_v1",
    ) -> "OrderIntent":
        intent_id = generate_order_intent_id(
            market_slug=market_slug,
            token_asset_id=token_asset_id,
            source_sequence_id=source_sequence_id,
            order_ts_ns=order_ts_ns,
            order_side=order_side,
            limit_price=limit_price,
            order_size=order_size,
            cancel_after_ns=cancel_after_ns,
            price_level=price_level,
        )
        if replace_after_ns is not None or replace_limit_price is not None or replace_order_size is not None:
            intent_id = hashlib.blake2b(
                "|".join(
                    [
                        intent_id,
                        "" if replace_after_ns is None else str(int(replace_after_ns)),
                        "" if replace_limit_price is None else f"{float(replace_limit_price):.6f}",
                        "" if replace_order_size is None else f"{float(replace_order_size):.6f}",
                        str(replace_price_level or ""),
                    ]
                ).encode("utf-8"),
                digest_size=12,
            ).hexdigest()
        return cls(
            intent_id=intent_id,
            market_slug=str(market_slug),
            token_asset_id=str(token_asset_id),
            token_side_normalized=str(token_side_normalized).upper(),
            order_ts_ns=int(order_ts_ns),
            market_close_ts_ns=int(market_close_ts_ns),
            market_deploy_ts_ns=int(market_deploy_ts_ns if market_deploy_ts_ns is not None else order_ts_ns),
            order_side=str(order_side).lower(),
            limit_price=float(limit_price),
            order_size=float(order_size),
            price_level=str(price_level),
            post_only=bool(post_only),
            cancel_after_ns=None if cancel_after_ns is None else int(cancel_after_ns),
            replace_after_ns=None if replace_after_ns is None else int(replace_after_ns),
            replace_limit_price=None if replace_limit_price is None else float(replace_limit_price),
            replace_order_size=None if replace_order_size is None else float(replace_order_size),
            replace_price_level=str(replace_price_level or ""),
            replace_post_only=bool(replace_post_only),
            holding_policy=str(holding_policy),
            source_sequence_id=str(source_sequence_id or ""),
            resolved_side=str(resolved_side or "").upper(),
            scenario_name=str(scenario_name),
        )


RESULT_COLUMNS = [
    "simulator_id",
    "intent_id",
    "source_sequence_id",
    "scenario_name",
    "market_slug",
    "token_asset_id",
    "token_side_normalized",
    "resolved_side",
    "order_ts_ns",
    "market_close_ts_ns",
    "order_side",
    "price_level",
    "limit_price",
    "order_size",
    "post_only",
    "holding_policy",
    "cancel_after_ns",
    "replace_after_ns",
    "replace_limit_price",
    "replace_order_size",
    "replace_price_level",
    "replace_post_only",
    "cancel_request_ts_ns",
    "cancel_effective_ts_ns",
    "replace_requested",
    "replace_request_ts_ns",
    "replace_effective_ts_ns",
    "replaced",
    "replace_post_only_rejected",
    "state_recv_ts_ns",
    "state_stale_s",
    "state_best_bid",
    "state_best_ask",
    "state_mid",
    "state_spread",
    "state_bid_depth_total",
    "state_ask_depth_total",
    "state_tick_size",
    "live_reference_recv_ts_ns",
    "live_btc_usd",
    "live_price_valid",
    "live_reference_trusted",
    "live_price_degraded",
    "live_source_count",
    "live_bias_mode",
    "live_bias_active",
    "live_bias_carried_forward",
    "live_applied_bias_age_seconds",
    "one_sided_book",
    "crossed_book",
    "stale_book",
    "submitted",
    "post_only_rejected",
    "post_only_crossed_book",
    "filled",
    "full_fill",
    "partial_fill",
    "fill_uncertain",
    "fill_ts_ns",
    "last_fill_ts_ns",
    "time_to_fill_s",
    "time_to_last_fill_s",
    "fill_count",
    "fill_price",
    "fill_size",
    "fill_notional",
    "maker_fill",
    "taker_fill",
    "cancelled",
    "cancel_effective",
    "fee_schedule_id",
    "maker_fee_bps",
    "taker_fee_bps",
    "maker_rebate_bps",
    "taker_rebate_bps",
    "fees_paid",
    "rebate_earned",
    "gross_pnl_to_close",
    "net_pnl_to_close",
    "markout_1s",
    "markout_3s",
    "markout_5s",
    "markout_15s",
    "markout_to_close",
    "markout_1s_available",
    "markout_3s_available",
    "markout_5s_available",
    "markout_15s_available",
    "markout_to_close_available",
    "adverse_selection_5s",
    "toxic_fill_5s",
    "execution_exclusion_reason",
]


class ExecutionSimulator:
    def __init__(
        self,
        replay_index: ExecutionReplayIndex,
        config: ExecutionSimulatorConfig | None = None,
        *,
        live_reference_reader: Any | None = None,
    ) -> None:
        self.replay_index = replay_index
        self.config = config or ExecutionSimulatorConfig()
        self.live_reference_reader = live_reference_reader

    def simulate_intent(self, intent: OrderIntent) -> dict[str, Any]:
        replay = self.replay_index.get(intent.market_slug, intent.token_asset_id)
        if replay is None:
            return self._base_result(intent, None, "replay_missing")

        state_cache = getattr(self, "_state_cache", None)
        state_key = (intent.market_slug, intent.token_asset_id, int(intent.order_ts_ns))
        if state_cache is not None and state_key in state_cache:
            state = state_cache[state_key]
        else:
            state = replay.state_before(intent.order_ts_ns)
            if state_cache is not None:
                state_cache[state_key] = state
        result = self._base_result(intent, state, "")
        live_reference_reason = self._attach_live_reference(result, intent)
        if live_reference_reason:
            result["execution_exclusion_reason"] = live_reference_reason
            return result
        state_block_reason = self._state_block_reason(state)
        if state_block_reason:
            result["execution_exclusion_reason"] = state_block_reason
            return result

        if intent.order_side not in {"buy", "sell"}:
            result["execution_exclusion_reason"] = "invalid_order_side"
            return result
        if intent.token_side_normalized not in {"UP", "DOWN"}:
            result["execution_exclusion_reason"] = "invalid_token_side"
            return result
        if intent.order_size <= 0 or intent.limit_price <= 0:
            result["execution_exclusion_reason"] = "invalid_order_terms"
            return result

        result["submitted"] = True
        crossed = self._post_only_crosses(intent, state)
        result["post_only_crossed_book"] = crossed
        if intent.post_only and crossed and self.config.post_only_reject_crossing:
            result["post_only_rejected"] = True
            result["execution_exclusion_reason"] = "post_only_crossed_book"
            return result

        cancel_effective_ts_ns = self._cancel_effective_ts_ns(intent)
        result["cancel_request_ts_ns"] = self._cancel_request_ts_ns(intent)
        result["cancel_effective_ts_ns"] = cancel_effective_ts_ns
        replace_effective_ts_ns = self._replace_effective_ts_ns(intent)
        result["replace_effective_ts_ns"] = replace_effective_ts_ns

        active_end_ts_ns = min(cancel_effective_ts_ns, intent.market_close_ts_ns)
        initial_end_ts_ns = (
            min(active_end_ts_ns, replace_effective_ts_ns)
            if intent.replace_after_ns is not None
            else active_end_ts_ns
        )
        fills = self._find_fills(
            replay=replay,
            intent=intent,
            state=state,
            start_ts_ns=intent.order_ts_ns,
            end_exclusive_ts_ns=initial_end_ts_ns,
            limit_price=float(intent.limit_price),
            remaining_size=float(intent.order_size),
        )
        total_filled = sum(fill["size"] for fill in fills)
        remaining_size = max(0.0, float(intent.order_size) - total_filled)

        if intent.replace_after_ns is not None and replace_effective_ts_ns < active_end_ts_ns and remaining_size > 0:
            replacement_state = replay.state_before(replace_effective_ts_ns)
            replacement_price = intent.replace_limit_price if intent.replace_limit_price is not None else intent.limit_price
            replacement_size = min(
                remaining_size,
                intent.replace_order_size if intent.replace_order_size is not None else remaining_size,
            )
            result["replaced"] = True
            if not replacement_state.valid:
                result["execution_exclusion_reason"] = "replace_book_state_unavailable"
            elif replacement_size <= 0 or replacement_price <= 0:
                result["execution_exclusion_reason"] = "invalid_replace_terms"
            else:
                replace_crossed = self._post_only_crosses_at_price(intent, replacement_state, float(replacement_price))
                if intent.replace_post_only and replace_crossed and self.config.post_only_reject_crossing:
                    result["replace_post_only_rejected"] = True
                else:
                    fills.extend(
                        self._find_fills(
                            replay=replay,
                            intent=intent,
                            state=replacement_state,
                            start_ts_ns=replace_effective_ts_ns,
                            end_exclusive_ts_ns=active_end_ts_ns,
                            limit_price=float(replacement_price),
                            remaining_size=float(replacement_size),
                        )
                    )

        self._apply_fill_summary(result, fills, intent)
        if result["filled"]:
            self._apply_markouts(result, replay, intent)
            self._apply_economics(result, intent)

        cancel_happens_before_close = cancel_effective_ts_ns <= intent.market_close_ts_ns
        result["cancelled"] = (not result["full_fill"]) and cancel_happens_before_close
        result["cancel_effective"] = cancel_happens_before_close
        return result

    def simulate_many(self, intents: list[OrderIntent]) -> pd.DataFrame:
        previous_state_cache = getattr(self, "_state_cache", None)
        previous_live_cache = getattr(self, "_live_reference_cache", None)
        self._state_cache = {}
        self._live_reference_cache = {}
        try:
            rows = [self.simulate_intent(intent) for intent in intents]
            return pd.DataFrame(rows, columns=RESULT_COLUMNS)
        finally:
            if previous_state_cache is None:
                try:
                    delattr(self, "_state_cache")
                except AttributeError:
                    pass
            else:
                self._state_cache = previous_state_cache
            if previous_live_cache is None:
                try:
                    delattr(self, "_live_reference_cache")
                except AttributeError:
                    pass
            else:
                self._live_reference_cache = previous_live_cache

    def _base_result(self, intent: OrderIntent, state: BookState | None, reason: str) -> dict[str, Any]:
        simulator_id = hashlib.blake2b(
            f"{intent.intent_id}|execution_simulator_v1".encode("utf-8"), digest_size=12
        ).hexdigest()
        return {
            "simulator_id": simulator_id,
            "intent_id": intent.intent_id,
            "source_sequence_id": intent.source_sequence_id,
            "scenario_name": intent.scenario_name,
            "market_slug": intent.market_slug,
            "token_asset_id": intent.token_asset_id,
            "token_side_normalized": intent.token_side_normalized,
            "resolved_side": intent.resolved_side,
            "order_ts_ns": int(intent.order_ts_ns),
            "market_close_ts_ns": int(intent.market_close_ts_ns),
            "order_side": intent.order_side,
            "price_level": intent.price_level,
            "limit_price": float(intent.limit_price),
            "order_size": float(intent.order_size),
            "post_only": bool(intent.post_only),
            "holding_policy": intent.holding_policy,
            "cancel_after_ns": intent.cancel_after_ns,
            "replace_after_ns": intent.replace_after_ns,
            "replace_limit_price": intent.replace_limit_price if intent.replace_limit_price is not None else _nan(),
            "replace_order_size": intent.replace_order_size if intent.replace_order_size is not None else _nan(),
            "replace_price_level": intent.replace_price_level,
            "replace_post_only": bool(intent.replace_post_only),
            "cancel_request_ts_ns": self._cancel_request_ts_ns(intent),
            "cancel_effective_ts_ns": self._cancel_effective_ts_ns(intent),
            "replace_requested": intent.replace_after_ns is not None,
            "replace_request_ts_ns": self._replace_request_ts_ns(intent),
            "replace_effective_ts_ns": self._replace_effective_ts_ns(intent),
            "replaced": False,
            "replace_post_only_rejected": False,
            "state_recv_ts_ns": state.recv_ts_ns if state is not None else 0,
            "state_stale_s": state.stale_s if state is not None else math.inf,
            "state_best_bid": state.best_bid if state is not None else _nan(),
            "state_best_ask": state.best_ask if state is not None else _nan(),
            "state_mid": state.mid if state is not None else _nan(),
            "state_spread": state.spread if state is not None else _nan(),
            "state_bid_depth_total": state.bid_depth_total if state is not None else _nan(),
            "state_ask_depth_total": state.ask_depth_total if state is not None else _nan(),
            "state_tick_size": state.tick_size if state is not None else self.config.default_tick_size,
            "live_reference_recv_ts_ns": 0,
            "live_btc_usd": _nan(),
            "live_price_valid": False,
            "live_reference_trusted": False,
            "live_price_degraded": False,
            "live_source_count": 0,
            "live_bias_mode": "unavailable",
            "live_bias_active": False,
            "live_bias_carried_forward": False,
            "live_applied_bias_age_seconds": _nan(),
            "one_sided_book": state.one_sided_book if state is not None else True,
            "crossed_book": state.crossed_book if state is not None else False,
            "stale_book": (state.stale_s * SECOND_NS > self.config.stale_book_threshold_ns) if state is not None else True,
            "submitted": False,
            "post_only_rejected": False,
            "post_only_crossed_book": False,
            "filled": False,
            "full_fill": False,
            "partial_fill": False,
            "fill_uncertain": False,
            "fill_ts_ns": 0,
            "last_fill_ts_ns": 0,
            "time_to_fill_s": _nan(),
            "time_to_last_fill_s": _nan(),
            "fill_count": 0,
            "fill_price": _nan(),
            "fill_size": 0.0,
            "fill_notional": 0.0,
            "maker_fill": False,
            "taker_fill": False,
            "cancelled": False,
            "cancel_effective": False,
            "fee_schedule_id": self.config.resolved_fee_schedule().schedule_id,
            "maker_fee_bps": self.config.resolved_fee_schedule().maker_fee_bps,
            "taker_fee_bps": self.config.resolved_fee_schedule().taker_fee_bps,
            "maker_rebate_bps": self.config.resolved_fee_schedule().maker_rebate_bps,
            "taker_rebate_bps": self.config.resolved_fee_schedule().taker_rebate_bps,
            "fees_paid": 0.0,
            "rebate_earned": 0.0,
            "gross_pnl_to_close": _nan(),
            "net_pnl_to_close": _nan(),
            "markout_1s": _nan(),
            "markout_3s": _nan(),
            "markout_5s": _nan(),
            "markout_15s": _nan(),
            "markout_to_close": _nan(),
            "markout_1s_available": False,
            "markout_3s_available": False,
            "markout_5s_available": False,
            "markout_15s_available": False,
            "markout_to_close_available": False,
            "adverse_selection_5s": False,
            "toxic_fill_5s": False,
            "execution_exclusion_reason": reason,
        }

    def _state_block_reason(self, state: BookState) -> str:
        if not state.valid:
            return "book_state_unavailable"
        if state.one_sided_book:
            return "one_sided_book"
        if state.crossed_book:
            return "crossed_book"
        if state.stale_s * SECOND_NS > self.config.stale_book_threshold_ns:
            return "stale_book"
        return ""

    def _post_only_crosses(self, intent: OrderIntent, state: BookState) -> bool:
        return self._post_only_crosses_at_price(intent, state, float(intent.limit_price))

    def _post_only_crosses_at_price(self, intent: OrderIntent, state: BookState, limit_price: float) -> bool:
        if intent.order_side == "buy":
            return state.best_ask is not None and float(limit_price) >= state.best_ask
        if intent.order_side == "sell":
            return state.best_bid is not None and float(limit_price) <= state.best_bid
        return False

    def _cancel_request_ts_ns(self, intent: OrderIntent) -> int:
        if intent.cancel_after_ns is None:
            return int(intent.market_close_ts_ns)
        return min(int(intent.order_ts_ns + intent.cancel_after_ns), int(intent.market_close_ts_ns))

    def _cancel_effective_ts_ns(self, intent: OrderIntent) -> int:
        return min(self._cancel_request_ts_ns(intent) + int(self.config.cancel_latency_ns), int(intent.market_close_ts_ns))

    def _replace_request_ts_ns(self, intent: OrderIntent) -> int:
        if intent.replace_after_ns is None:
            return 0
        return min(int(intent.order_ts_ns + intent.replace_after_ns), int(intent.market_close_ts_ns))

    def _replace_effective_ts_ns(self, intent: OrderIntent) -> int:
        request_ts_ns = self._replace_request_ts_ns(intent)
        if request_ts_ns <= 0:
            return 0
        latency = int(self.config.cancel_latency_ns) + int(self.config.replace_latency_ns)
        return min(request_ts_ns + latency, int(intent.market_close_ts_ns))

    def _attach_live_reference(self, result: dict[str, Any], intent: OrderIntent) -> str:
        if self.live_reference_reader is None:
            return "live_reference_reader_missing" if self.config.require_live_reference else ""
        live_cache = getattr(self, "_live_reference_cache", None)
        live_key = int(intent.order_ts_ns)
        if live_cache is not None and live_key in live_cache:
            live_reference = live_cache[live_key]
        else:
            live_reference = self.live_reference_reader.latest_asof(asof_ts_ns=live_key)
            if live_cache is not None:
                live_cache[live_key] = live_reference
        if live_reference is None:
            return "live_reference_unavailable" if self.config.require_live_reference else ""
        recv_ts_ns = int(live_reference.get("recv_ts_ns") or 0)
        if recv_ts_ns > int(intent.order_ts_ns):
            return "live_reference_future_row"
        price = live_reference.get("btc_usd_price")
        price_valid = bool(live_reference.get("price_valid")) and price is not None and not _is_nan(price)
        bias_active = bool(live_reference.get("bias_active"))
        bias_carried = bool(live_reference.get("bias_carried_forward"))
        trusted = bool(price_valid and (bias_active or bias_carried))
        result["live_reference_recv_ts_ns"] = recv_ts_ns
        result["live_btc_usd"] = float(price) if price_valid else _nan()
        result["live_price_valid"] = bool(price_valid)
        result["live_reference_trusted"] = trusted
        result["live_price_degraded"] = bool(live_reference.get("degraded"))
        result["live_source_count"] = int(live_reference.get("source_count") or 0)
        result["live_bias_mode"] = str(live_reference.get("bias_mode") or "unavailable")
        result["live_bias_active"] = bias_active
        result["live_bias_carried_forward"] = bias_carried
        applied_age = live_reference.get("applied_bias_age_seconds")
        result["live_applied_bias_age_seconds"] = _nan() if applied_age is None else float(applied_age)
        if self.config.require_live_reference and not trusted:
            return "live_reference_untrusted"
        return ""

    def _queue_ahead_size(self, intent: OrderIntent, state: BookState) -> float:
        if self.config.queue_ahead_fraction <= 0:
            return 0.0
        depth = state.bid_depth_total if intent.order_side == "buy" else state.ask_depth_total
        return max(0.0, float(depth or 0.0) * float(self.config.queue_ahead_fraction))

    def _find_fills(
        self,
        *,
        replay: TokenReplay,
        intent: OrderIntent,
        state: BookState,
        start_ts_ns: int,
        end_exclusive_ts_ns: int,
        limit_price: float,
        remaining_size: float,
    ) -> list[dict[str, Any]]:
        if int(end_exclusive_ts_ns) <= int(start_ts_ns) or remaining_size <= 0:
            return []
        if len(replay._trade_ts) == 0:
            return []
        left = replay._trade_ts.searchsorted(int(start_ts_ns), side="right")
        right = replay._trade_ts.searchsorted(int(end_exclusive_ts_ns), side="left")
        if right <= left:
            return []

        queue_ahead = self._queue_ahead_size(intent, state)
        queue_remaining = queue_ahead
        remaining = float(remaining_size)
        saw_uncertain = False
        fills: list[dict[str, Any]] = []
        for trade_index in range(int(left), int(right)):
            trade_price_f = float(replay._trade_prices[trade_index])
            if _is_nan(trade_price_f):
                continue
            side_code = int(replay._trade_side_codes[trade_index])
            trade_size = max(0.0, float(replay._trade_sizes[trade_index]))
            if trade_size <= 0:
                continue
            if intent.order_side == "buy":
                side_ok = side_code == -1 or (self.config.allow_unknown_trade_side_fill and side_code == 0)
                price_ok = trade_price_f <= float(limit_price)
            elif intent.order_side == "sell":
                side_ok = side_code == 1 or (self.config.allow_unknown_trade_side_fill and side_code == 0)
                price_ok = trade_price_f >= float(limit_price)
            else:
                side_ok = False
                price_ok = False
            if not (side_ok and price_ok):
                continue
            saw_uncertain = saw_uncertain or side_code == 0
            executable_size = trade_size
            if queue_remaining > 0:
                consumed_by_queue = min(queue_remaining, executable_size)
                queue_remaining -= consumed_by_queue
                executable_size -= consumed_by_queue
            if executable_size <= 0:
                continue
            fill_size = min(remaining, executable_size)
            if fill_size <= 0:
                continue
            passive_price = _round_price(float(limit_price), state.tick_size)
            fills.append(
                {
                    "ts_ns": int(replay._trade_ts[trade_index]),
                    "size": float(fill_size),
                    "price": passive_price,
                    "uncertain": bool(saw_uncertain),
                }
            )
            remaining -= fill_size
            if remaining <= 1e-12:
                break
        return fills

    def _trade_can_fill_intent(self, intent: OrderIntent, trade_price: float, trade_side: str, *, limit_price: float) -> bool:
        if intent.order_side == "buy":
            side_ok = trade_side == "sell" or (self.config.allow_unknown_trade_side_fill and trade_side == "unknown")
            return side_ok and trade_price <= float(limit_price)
        if intent.order_side == "sell":
            side_ok = trade_side == "buy" or (self.config.allow_unknown_trade_side_fill and trade_side == "unknown")
            return side_ok and trade_price >= float(limit_price)
        return False

    def _apply_fill_summary(self, result: dict[str, Any], fills: list[dict[str, Any]], intent: OrderIntent) -> None:
        if not fills:
            return
        fill_size = sum(float(fill["size"]) for fill in fills)
        fill_notional = sum(float(fill["size"]) * float(fill["price"]) for fill in fills)
        first_ts_ns = int(fills[0]["ts_ns"])
        last_ts_ns = int(fills[-1]["ts_ns"])
        result["filled"] = fill_size > 0
        result["full_fill"] = fill_size >= float(intent.order_size) - 1e-12
        result["partial_fill"] = 0 < fill_size < float(intent.order_size) - 1e-12
        result["fill_uncertain"] = any(bool(fill.get("uncertain")) for fill in fills)
        result["fill_ts_ns"] = first_ts_ns
        result["last_fill_ts_ns"] = last_ts_ns
        result["time_to_fill_s"] = (first_ts_ns - intent.order_ts_ns) / SECOND_NS
        result["time_to_last_fill_s"] = (last_ts_ns - intent.order_ts_ns) / SECOND_NS
        result["fill_count"] = len(fills)
        result["fill_size"] = float(fill_size)
        result["fill_notional"] = float(fill_notional)
        result["fill_price"] = fill_notional / fill_size if fill_size > 0 else _nan()
        result["maker_fill"] = bool(intent.post_only and not result["post_only_rejected"])
        result["taker_fill"] = bool(not result["maker_fill"])

    def _signed_markout(self, intent: OrderIntent, fill_price: float, fill_size: float, future_mid: float) -> float:
        return compute_markout_value(
            fill_price=fill_price,
            fill_size=fill_size,
            fill_side=intent.order_side,
            future_mid=future_mid,
        )

    def _apply_markouts(self, result: dict[str, Any], replay: TokenReplay, intent: OrderIntent) -> None:
        fill_ts_ns = int(result["fill_ts_ns"])
        fill_price = float(result["fill_price"])
        fill_size = float(result["fill_size"])
        for horizon_s in self.config.markout_horizons_s:
            value_col = f"markout_{horizon_s}s"
            available_col = f"markout_{horizon_s}s_available"
            target_ts_ns = fill_ts_ns + horizon_s * SECOND_NS
            if target_ts_ns > intent.market_close_ts_ns:
                result[value_col] = _nan()
                result[available_col] = False
                continue
            future_mid = replay.mid_at_or_before(target_ts_ns)
            if future_mid is None:
                result[value_col] = _nan()
                result[available_col] = False
                continue
            result[value_col] = self._signed_markout(intent, fill_price, fill_size, float(future_mid))
            result[available_col] = True

        terminal_mid = replay.terminal_mid_before_close(intent.market_close_ts_ns)
        if terminal_mid is not None:
            result["markout_to_close"] = self._signed_markout(intent, fill_price, fill_size, float(terminal_mid))
            result["markout_to_close_available"] = True

        markout_5s = result.get("markout_5s")
        result["adverse_selection_5s"] = bool(not _is_nan(markout_5s) and float(markout_5s) < 0)
        result["toxic_fill_5s"] = bool(not _is_nan(markout_5s) and float(markout_5s) < -0.02)

    def _apply_economics(self, result: dict[str, Any], intent: OrderIntent) -> None:
        if not result["filled"]:
            return
        fill_size = float(result["fill_size"])
        fill_price = float(result["fill_price"])
        schedule = self.config.resolved_fee_schedule()
        taker_fee_equivalent = schedule.taker_fee(fill_price, fill_size, int(intent.market_deploy_ts_ns))
        if result["maker_fill"]:
            fees_paid = schedule.maker_fee(fill_price, fill_size)
            rebate_earned = schedule.maker_rebate(taker_fee_equivalent, int(intent.market_deploy_ts_ns))
        else:
            fees_paid = taker_fee_equivalent
            rebate_earned = 0.0
        result["fees_paid"] = fees_paid
        result["rebate_earned"] = rebate_earned

        gross_pnl = _nan()
        resolved_side = intent.resolved_side.upper()
        if resolved_side in {"UP", "DOWN"}:
            payout = 1.0 if resolved_side == intent.token_side_normalized else 0.0
            gross_per_share = payout - fill_price if intent.order_side == "buy" else fill_price - payout
            gross_pnl = gross_per_share * fill_size
        elif not _is_nan(result["markout_to_close"]):
            gross_pnl = float(result["markout_to_close"])

        result["gross_pnl_to_close"] = gross_pnl
        if not _is_nan(result["gross_pnl_to_close"]):
            result["net_pnl_to_close"] = float(result["gross_pnl_to_close"]) - result["fees_paid"] + result["rebate_earned"]
