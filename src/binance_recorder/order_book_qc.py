from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_levels(levels: Any) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(levels, list):
        return normalized
    for level in levels:
        if not isinstance(level, list | tuple) or len(level) < 2:
            continue
        price = str(level[0])
        quantity = str(level[1])
        normalized[price] = quantity
    return normalized


def _quantity_is_zero(quantity: str) -> bool:
    try:
        return Decimal(quantity) == 0
    except Exception:
        return quantity.strip() == "0"


def _best_level(levels: dict[str, str], *, reverse: bool) -> tuple[str, str] | None:
    if not levels:
        return None
    ordered = sorted((Decimal(price), price, qty) for price, qty in levels.items())
    _, price, qty = ordered[-1] if reverse else ordered[0]
    return price, qty


def _book_crossed(bids: dict[str, str], asks: dict[str, str]) -> bool:
    best_bid = _best_level(bids, reverse=True)
    best_ask = _best_level(asks, reverse=False)
    if best_bid is None or best_ask is None:
        return False
    return Decimal(best_bid[0]) >= Decimal(best_ask[0])


@dataclass(slots=True)
class BinanceOrderBookObservation:
    state: str
    local_update_id: int | None
    requires_resnapshot: bool
    quality_class: str | None
    quality_flags: tuple[str, ...]
    lifecycle_events: list[dict[str, Any]]


@dataclass(slots=True)
class BinanceOrderBookIntegrityMonitor:
    symbol: str
    market: str
    parity_check_update_lag_tolerance: int = 20
    local_update_id: int | None = None
    initialized: bool = False
    resync_required: bool = False
    awaiting_bridge_event: bool = False
    _buffered_events: list[dict[str, Any]] = field(default_factory=list)
    _bids: dict[str, str] = field(default_factory=dict)
    _asks: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self.local_update_id = None
        self.initialized = False
        self.resync_required = False
        self.awaiting_bridge_event = False
        self._buffered_events.clear()
        self._bids.clear()
        self._asks.clear()

    def _enter_resync(
        self,
        *,
        state: str = "resync_required",
        local_update_id: int | None,
        lifecycle_events: list[dict[str, Any]],
        quality_flags: tuple[str, ...],
        buffered_events: list[dict[str, Any]] | None = None,
    ) -> BinanceOrderBookObservation:
        self.initialized = False
        self.resync_required = True
        self.awaiting_bridge_event = False
        if buffered_events is not None:
            self._buffered_events = list(buffered_events)
        return BinanceOrderBookObservation(
            state=state,
            local_update_id=local_update_id,
            requires_resnapshot=True,
            quality_class="continuity_compromised",
            quality_flags=quality_flags,
            lifecycle_events=lifecycle_events,
        )

    def observe_depth_event(self, payload: dict[str, Any]) -> BinanceOrderBookObservation:
        first_update_id = _to_int(payload.get("U"))
        final_update_id = _to_int(payload.get("u"))
        previous_final_update_id = _to_int(payload.get("pu"))
        if first_update_id is None or final_update_id is None:
            return BinanceOrderBookObservation(
                state="continuity_uncertain",
                local_update_id=self.local_update_id,
                requires_resnapshot=False,
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[],
            )
        if not self.initialized:
            self._buffered_events.append(payload)
            return BinanceOrderBookObservation(
                state="buffering_snapshot",
                local_update_id=self.local_update_id,
                requires_resnapshot=False,
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[],
            )
        if final_update_id <= (self.local_update_id or 0):
            return BinanceOrderBookObservation(
                state="stale_event_ignored",
                local_update_id=self.local_update_id,
                requires_resnapshot=False,
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[],
            )
        bridge_aligned = first_update_id <= (self.local_update_id or 0) + 1 <= final_update_id
        previous_id_mismatch = previous_final_update_id is not None and previous_final_update_id != self.local_update_id
        if self.awaiting_bridge_event:
            if not bridge_aligned:
                return self._enter_resync(
                    local_update_id=self.local_update_id,
                    quality_flags=("snapshot_alignment_failed",),
                    buffered_events=[payload],
                    lifecycle_events=[
                        {
                            "event": "snapshot_alignment_failed",
                            "symbol": self.symbol,
                            "market": self.market,
                            "local_update_id": self.local_update_id,
                            "event_first_update_id": first_update_id,
                            "event_final_update_id": final_update_id,
                            "event_previous_final_update_id": previous_final_update_id,
                            "reason": "first_live_event_not_aligned_to_snapshot",
                        },
                        {
                            "event": "snapshot_reset_required",
                            "reason": "snapshot_alignment_failed",
                        },
                    ],
                )
        else:
            gap_detected = previous_final_update_id is None and first_update_id > (self.local_update_id or 0) + 1
            continuity_broken = previous_id_mismatch or gap_detected
            if continuity_broken:
                return self._enter_resync(
                    local_update_id=self.local_update_id,
                    quality_flags=("order_book_continuity_broken",),
                    buffered_events=[payload],
                    lifecycle_events=[
                        {
                            "event": "order_book_continuity_broken",
                            "symbol": self.symbol,
                            "market": self.market,
                            "local_update_id": self.local_update_id,
                            "event_first_update_id": first_update_id,
                            "event_final_update_id": final_update_id,
                            "event_previous_final_update_id": previous_final_update_id,
                            "reason": "previous_id_mismatch" if previous_id_mismatch else "update_gap",
                        },
                        {
                            "event": "snapshot_reset_required",
                            "reason": "order_book_continuity_broken",
                        },
                    ],
                )
        next_bids = dict(self._bids)
        next_asks = dict(self._asks)
        self._apply_levels(payload, bids=next_bids, asks=next_asks)
        if _book_crossed(next_bids, next_asks):
            return self._enter_resync(
                local_update_id=self.local_update_id,
                quality_flags=("order_book_invalid_local_state",),
                buffered_events=[payload],
                lifecycle_events=[
                    {
                        "event": "order_book_invalid_local_state",
                        "symbol": self.symbol,
                        "market": self.market,
                        "reason": "crossed_local_book",
                        "event_first_update_id": first_update_id,
                        "event_final_update_id": final_update_id,
                        "local_best_bid": _best_level(next_bids, reverse=True),
                        "local_best_ask": _best_level(next_asks, reverse=False),
                    },
                    {
                        "event": "snapshot_reset_required",
                        "reason": "crossed_local_book",
                    },
                ],
            )
        self._bids = next_bids
        self._asks = next_asks
        self.local_update_id = final_update_id
        self.awaiting_bridge_event = False
        return BinanceOrderBookObservation(
            state="healthy",
            local_update_id=self.local_update_id,
            requires_resnapshot=False,
            quality_class=None,
            quality_flags=(),
            lifecycle_events=[],
        )

    def load_snapshot(self, payload: dict[str, Any], *, reason: str) -> BinanceOrderBookObservation:
        snapshot_update_id = _to_int(payload.get("lastUpdateId"))
        if snapshot_update_id is None:
            return BinanceOrderBookObservation(
                state="continuity_uncertain",
                local_update_id=self.local_update_id,
                requires_resnapshot=True,
                quality_class="continuity_compromised",
                quality_flags=("snapshot_invalid",),
                lifecycle_events=[{"event": "snapshot_reset_required", "reason": "invalid_snapshot"}],
            )
        buffered = list(self._buffered_events)
        self._bids = _normalize_levels(payload.get("bids"))
        self._asks = _normalize_levels(payload.get("asks"))
        self.local_update_id = snapshot_update_id
        self.initialized = True
        self.resync_required = False
        self.awaiting_bridge_event = True
        self._buffered_events = [event for event in buffered if (_to_int(event.get("u")) or 0) > snapshot_update_id]
        if self._buffered_events:
            first = self._buffered_events[0]
            first_lower = _to_int(first.get("U")) or (snapshot_update_id + 1)
            first_upper = _to_int(first.get("u")) or snapshot_update_id
            if not (first_lower <= snapshot_update_id + 1 <= first_upper):
                return self._enter_resync(
                    local_update_id=self.local_update_id,
                    quality_flags=("snapshot_alignment_failed",),
                    buffered_events=self._buffered_events,
                    lifecycle_events=[
                        {
                            "event": "snapshot_alignment_failed",
                            "snapshot_last_update_id": snapshot_update_id,
                            "first_buffered_first_update_id": first_lower,
                            "first_buffered_final_update_id": first_upper,
                            "reason": "first_buffered_event_not_aligned",
                        }
                    ],
                )
        while self._buffered_events:
            event = self._buffered_events.pop(0)
            follow_up = self.observe_depth_event(event)
            if follow_up.requires_resnapshot:
                return follow_up
        return BinanceOrderBookObservation(
            state="healthy",
            local_update_id=self.local_update_id,
            requires_resnapshot=False,
            quality_class=None,
            quality_flags=(),
            lifecycle_events=[
                {
                    "event": "order_book_resync_started" if reason != "bootstrap" else "order_book_snapshot_loaded",
                    "snapshot_last_update_id": snapshot_update_id,
                    "reason": reason,
                },
                {
                    "event": "order_book_resynced",
                    "snapshot_last_update_id": snapshot_update_id,
                    "reason": reason,
                },
            ],
        )

    def observe_periodic_snapshot(self, payload: dict[str, Any]) -> BinanceOrderBookObservation:
        snapshot_update_id = _to_int(payload.get("lastUpdateId"))
        if not self.initialized or snapshot_update_id is None or self.local_update_id is None:
            return BinanceOrderBookObservation(
                state="continuity_uncertain",
                local_update_id=self.local_update_id,
                requires_resnapshot=False,
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[],
            )
        update_gap = abs(snapshot_update_id - self.local_update_id)
        if update_gap > self.parity_check_update_lag_tolerance:
            return BinanceOrderBookObservation(
                state="parity_inconclusive",
                local_update_id=self.local_update_id,
                requires_resnapshot=False,
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[
                    {
                        "event": "order_book_parity_inconclusive",
                        "snapshot_last_update_id": snapshot_update_id,
                        "local_update_id": self.local_update_id,
                        "update_gap": update_gap,
                    }
                ],
            )
        snapshot_bids = _normalize_levels(payload.get("bids"))
        snapshot_asks = _normalize_levels(payload.get("asks"))
        local_best_bid = _best_level(self._bids, reverse=True)
        local_best_ask = _best_level(self._asks, reverse=False)
        snapshot_best_bid = _best_level(snapshot_bids, reverse=True)
        snapshot_best_ask = _best_level(snapshot_asks, reverse=False)
        if local_best_bid != snapshot_best_bid or local_best_ask != snapshot_best_ask:
            return self._enter_resync(
                state="parity_mismatch",
                local_update_id=self.local_update_id,
                quality_flags=("order_book_parity_mismatch",),
                lifecycle_events=[
                    {
                        "event": "order_book_parity_mismatch",
                        "snapshot_last_update_id": snapshot_update_id,
                        "local_update_id": self.local_update_id,
                        "local_best_bid": local_best_bid,
                        "local_best_ask": local_best_ask,
                        "snapshot_best_bid": snapshot_best_bid,
                        "snapshot_best_ask": snapshot_best_ask,
                    },
                    {
                        "event": "snapshot_reset_required",
                        "reason": "order_book_parity_mismatch",
                    }
                ],
            )
        return BinanceOrderBookObservation(
            state="parity_verified",
            local_update_id=self.local_update_id,
            requires_resnapshot=False,
            quality_class=None,
            quality_flags=(),
            lifecycle_events=[
                {
                    "event": "order_book_parity_verified",
                    "snapshot_last_update_id": snapshot_update_id,
                    "local_update_id": self.local_update_id,
                }
            ],
        )

    def _apply_levels(self, payload: dict[str, Any], *, bids: dict[str, str], asks: dict[str, str]) -> None:
        for price, quantity in _normalize_levels(payload.get("b")).items():
            if _quantity_is_zero(quantity):
                bids.pop(price, None)
            else:
                bids[price] = quantity
        for price, quantity in _normalize_levels(payload.get("a")).items():
            if _quantity_is_zero(quantity):
                asks.pop(price, None)
            else:
                asks[price] = quantity
