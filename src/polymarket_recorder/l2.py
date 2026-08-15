from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from .utils import build_polymarket_source_timestamps


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, InvalidOperation, ValueError):
        return None


def _spread(best_bid: str | None, best_ask: str | None) -> str | None:
    bid_dec = _decimal(best_bid)
    ask_dec = _decimal(best_ask)
    if bid_dec is None or ask_dec is None:
        return None
    return format(ask_dec - bid_dec, "f")


@dataclass(slots=True)
class _AssetBookState:
    bids: dict[str, str] = field(default_factory=dict)
    asks: dict[str, str] = field(default_factory=dict)
    state_seq: int = 0
    last_hash: str | None = None
    tick_size: str | None = None
    min_order_size: str | None = None
    neg_risk: bool | None = None
    last_trade_price: str | None = None
    market: str | None = None


class PolymarketL2StateTracker:
    def __init__(self) -> None:
        self._states: dict[str, _AssetBookState] = {}

    def process_event(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # `source_timestamps` lets high-rate consumers (recorder/live bot) pass
        # a precomputed value instead of paying the recursive payload walk once
        # per emitted record. Omitted -> computed here, exactly as before.
        if not isinstance(payload, dict):
            return []
        event_type = str(payload.get("event_type") or "")
        if event_type == "book":
            record = self._apply_book(payload, source_timestamps)
            return [record] if record is not None else []
        if event_type == "price_change":
            return self._apply_price_changes(payload, source_timestamps)
        if event_type == "best_bid_ask":
            record = self._top_of_book_observation(payload, source_timestamps)
            return [record] if record is not None else []
        if event_type == "last_trade_price":
            record = self._trade_execution(payload, source_timestamps)
            return [record] if record is not None else []
        if event_type == "tick_size_change":
            record = self._tick_size_update(payload, source_timestamps)
            return [record] if record is not None else []
        return []

    def _state(self, asset_id: str) -> _AssetBookState:
        if asset_id not in self._states:
            self._states[asset_id] = _AssetBookState()
        return self._states[asset_id]

    def drop_asset(self, asset_id: str) -> None:
        """Forget a closed asset's book state so `_states` doesn't grow
        unbounded over a long-lived run (memory leak -> event-loop slowdown ->
        more dropped deltas -> worse book drift). Safe no-op if unknown."""
        self._states.pop(str(asset_id), None)

    def _snapshot_record(
        self,
        *,
        asset_id: str,
        event_type: str,
        state: _AssetBookState,
        source_payload: dict[str, Any],
        changed_levels: list[dict[str, Any]] | None = None,
        snapshot_origin: str,
        extra: dict[str, Any] | None = None,
        source_timestamps: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bids = self._sorted_levels(state.bids, reverse=True)
        asks = self._sorted_levels(state.asks, reverse=False)
        best_bid = bids[0]["price"] if bids else None
        best_ask = asks[0]["price"] if asks else None
        payload: dict[str, Any] = {
            "record_type": "l2_book_state",
            "event_type": event_type,
            "token_asset_id": asset_id,
            "market": state.market or source_payload.get("market"),
            "book_state_seq": state.state_seq,
            "snapshot_origin": snapshot_origin,
            "book_hash": state.last_hash,
            "tick_size": state.tick_size,
            "min_order_size": state.min_order_size,
            "neg_risk": state.neg_risk,
            "last_trade_price": state.last_trade_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": _spread(best_bid, best_ask),
            "level_counts": {"bid": len(bids), "ask": len(asks)},
            "depth_totals": {
                "bid_size_total": self._sum_sizes(bids),
                "ask_size_total": self._sum_sizes(asks),
            },
            "book": {"bids": bids, "asks": asks},
            "changed_levels": changed_levels or [],
            "timestamp_semantics": "source_metadata",
            "source_timestamps": source_timestamps
            if source_timestamps is not None
            else build_polymarket_source_timestamps(source_payload),
            "event_payload": source_payload,
        }
        if extra:
            payload.update(extra)
        return payload

    def _apply_book(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        asset_id = payload.get("asset_id")
        if asset_id is None:
            return None
        asset_key = str(asset_id)
        state = self._state(asset_key)
        state.bids = self._levels_to_map(payload.get("bids", []))
        state.asks = self._levels_to_map(payload.get("asks", []))
        state.market = str(payload.get("market") or state.market or "")
        state.last_hash = str(payload.get("hash")) if payload.get("hash") is not None else state.last_hash
        state.tick_size = payload.get("tick_size") or state.tick_size
        state.min_order_size = payload.get("min_order_size") or state.min_order_size
        state.neg_risk = payload.get("neg_risk") if payload.get("neg_risk") is not None else state.neg_risk
        state.last_trade_price = payload.get("last_trade_price") or state.last_trade_price
        state.state_seq += 1
        return self._snapshot_record(
            asset_id=asset_key,
            event_type="book",
            state=state,
            source_payload=payload,
            snapshot_origin="ws_book_snapshot",
            source_timestamps=source_timestamps,
        )

    def _apply_price_changes(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        grouped_changes: dict[str, list[dict[str, Any]]] = {}
        for change in payload.get("price_changes", []):
            if not isinstance(change, dict) or change.get("asset_id") is None:
                continue
            grouped_changes.setdefault(str(change["asset_id"]), []).append(change)
        records: list[dict[str, Any]] = []
        for asset_id, changes in grouped_changes.items():
            state = self._state(asset_id)
            for change in changes:
                side = str(change.get("side") or "").upper()
                levels = state.bids if side == "BUY" else state.asks
                price = str(change.get("price"))
                size = str(change.get("size"))
                if size == "0":
                    levels.pop(price, None)
                else:
                    levels[price] = size
                state.market = str(payload.get("market") or state.market or "")
                state.last_hash = str(change.get("hash")) if change.get("hash") is not None else state.last_hash
            state.state_seq += 1
            records.append(
                self._snapshot_record(
                    asset_id=asset_id,
                    event_type="price_change",
                    state=state,
                    source_payload={**payload, "price_changes": changes},
                    changed_levels=changes,
                    snapshot_origin="ws_price_change_reconstructed",
                    source_timestamps=source_timestamps,
                )
            )
        return records

    def _top_of_book_observation(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        asset_id = payload.get("asset_id")
        if asset_id is None:
            return None
        asset_key = str(asset_id)
        state = self._state(asset_key)
        return {
            "record_type": "l2_top_of_book",
            "event_type": "best_bid_ask",
            "token_asset_id": asset_key,
            "market": payload.get("market") or state.market,
            "book_state_seq": state.state_seq,
            "tick_size": state.tick_size,
            "best_bid": payload.get("best_bid"),
            "best_ask": payload.get("best_ask"),
            "spread": payload.get("spread") or _spread(payload.get("best_bid"), payload.get("best_ask")),
            "timestamp_semantics": "source_metadata",
            "source_timestamps": source_timestamps
            if source_timestamps is not None
            else build_polymarket_source_timestamps(payload),
            "event_payload": payload,
        }

    def _trade_execution(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        asset_id = payload.get("asset_id")
        if asset_id is None:
            return None
        asset_key = str(asset_id)
        state = self._state(asset_key)
        if payload.get("price") is not None:
            state.last_trade_price = str(payload["price"])
        return {
            "record_type": "trade_execution",
            "event_type": "last_trade_price",
            "token_asset_id": asset_key,
            "market": payload.get("market") or state.market,
            "price": payload.get("price"),
            "size": payload.get("size"),
            "side": payload.get("side"),
            "fee_rate_bps": payload.get("fee_rate_bps"),
            "tick_size": state.tick_size,
            "timestamp_semantics": "source_metadata",
            "source_timestamps": source_timestamps
            if source_timestamps is not None
            else build_polymarket_source_timestamps(payload),
            "event_payload": payload,
        }

    def _tick_size_update(
        self,
        payload: dict[str, Any],
        source_timestamps: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        asset_id = payload.get("asset_id")
        if asset_id is None:
            return None
        asset_key = str(asset_id)
        state = self._state(asset_key)
        if payload.get("new_tick_size") is not None:
            state.tick_size = str(payload["new_tick_size"])
        return {
            "record_type": "tick_size_update",
            "event_type": "tick_size_change",
            "token_asset_id": asset_key,
            "market": payload.get("market") or state.market,
            "old_tick_size": payload.get("old_tick_size"),
            "new_tick_size": payload.get("new_tick_size"),
            "book_state_seq": state.state_seq,
            "timestamp_semantics": "source_metadata",
            "source_timestamps": source_timestamps
            if source_timestamps is not None
            else build_polymarket_source_timestamps(payload),
            "event_payload": payload,
        }

    @staticmethod
    def _levels_to_map(levels: Any) -> dict[str, str]:
        mapped: dict[str, str] = {}
        if not isinstance(levels, list):
            return mapped
        for level in levels:
            if not isinstance(level, dict):
                continue
            price = level.get("price")
            size = level.get("size")
            if price is None or size is None:
                continue
            mapped[str(price)] = str(size)
        return mapped

    @staticmethod
    def _sorted_levels(levels: dict[str, str], *, reverse: bool) -> list[dict[str, str]]:
        # float keys give the same ordering as Decimal for these tick-grid price
        # strings and cut the per-record sort cost roughly in half; this method
        # runs twice per emitted record on a 300-600 msg/s stream.
        def sort_key(item: tuple[str, str]) -> float:
            try:
                return float(item[0])
            except (TypeError, ValueError):
                return 0.0

        return [
            {"price": price, "size": size}
            for price, size in sorted(levels.items(), key=sort_key, reverse=reverse)
        ]

    @staticmethod
    def _sum_sizes(levels: list[dict[str, str]]) -> str:
        total = Decimal("0")
        for level in levels:
            size = _decimal(level.get("size"))
            if size is not None:
                total += size
        return format(total, "f")
