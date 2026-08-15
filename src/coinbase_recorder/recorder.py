from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import orjson
from websockets import connect

from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.file_quality import HourlyQualityTracker
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_timestamp_fields, recv_timestamp_source, utc_now_ns

from .continuity import CoinbaseSequenceTracker
from .config import CoinbaseAdvancedRecorderConfig


class CoinbaseSequenceResetRequired(RuntimeError):
    """Raised when Coinbase sequence continuity requires a controlled reconnect."""


def _lifecycle(event: str, product_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "coinbase",
        "venue": "advanced",
        "product_id": product_id,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


def _source_timestamps(payload: dict[str, Any]) -> dict[str, Any]:
    timestamps: dict[str, Any] = {}
    if "timestamp" in payload:
        timestamps.update(build_timestamp_fields("message_timestamp", payload["timestamp"]))
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        for group_name, key in (("trades", "time"), ("tickers", "time"), ("updates", "event_time")):
            for item in event.get(group_name, []):
                if isinstance(item, dict) and key in item:
                    timestamps.update(build_timestamp_fields(f"{group_name}_{key}", item[key]))
                    return timestamps
        if "current_time" in event:
            timestamps.update(build_timestamp_fields("heartbeat_current_time", event["current_time"]))
    return timestamps


def _first_event_timestamp_ns(payload: dict[str, Any]) -> int | None:
    timestamps = _source_timestamps(payload)
    for key in ("updates_event_time_ns", "trades_time_ns", "heartbeat_current_time_ns", "message_timestamp_ns"):
        value = timestamps.get(key)
        if isinstance(value, int):
            return value
    return None


def _normalize_l2_updates(updates: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(updates, list):
        return normalized
    for item in updates:
        if not isinstance(item, dict):
            continue
        quantity = item.get("new_quantity")
        if quantity is None:
            quantity = item.get("quantity")
        normalized_item = {
            "side": item.get("side"),
            "price_level": item.get("price_level"),
            "quantity": quantity,
            "event_time": item.get("event_time"),
        }
        normalized.append({key: value for key, value in normalized_item.items() if value is not None})
    return normalized


def _normalized(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        channel = payload.get("channel")
        events = payload.get("events", [])
        if not isinstance(events, list) or not events:
            return {"event_name": "heartbeat"} if channel == "heartbeats" else None
        event = events[0]
        if not isinstance(event, dict):
            return None
        if channel == "ticker":
            ticker = event.get("tickers", [None])[0]
            if not isinstance(ticker, dict):
                return None
            return {
                "event_name": "ticker",
                "product_id": ticker.get("product_id") or payload.get("product_id"),
                "price": ticker.get("price"),
                "best_bid": ticker.get("best_bid"),
                "best_bid_quantity": ticker.get("best_bid_quantity"),
                "best_ask": ticker.get("best_ask"),
                "best_ask_quantity": ticker.get("best_ask_quantity"),
            }
        if channel == "market_trades":
            trades = event.get("trades", [])
            if not isinstance(trades, list) or not trades:
                return None
            first = trades[0] if isinstance(trades[0], dict) else {}
            last = trades[-1] if isinstance(trades[-1], dict) else {}
            return {
                "event_name": "market_trades",
                "product_id": first.get("product_id") or payload.get("product_id"),
                "trade_count": len(trades),
                "first_price": first.get("price"),
                "last_price": last.get("price"),
                "first_side": first.get("side"),
                "last_side": last.get("side"),
            }
        if channel in {"level2", "l2_data"}:
            updates = event.get("updates", [])
            event_ts_ns = _first_event_timestamp_ns(payload)
            normalized_updates = _normalize_l2_updates(updates)
            return {
                "event_name": "level2",
                "channel": channel,
                "product_id": event.get("product_id") or payload.get("product_id"),
                "type": event.get("type"),
                "event_ts_ns": event_ts_ns,
                "event_ts_iso": ns_to_iso(event_ts_ns) if event_ts_ns is not None else None,
                "update_count": len(normalized_updates),
                "updates": normalized_updates,
                "first_side": normalized_updates[0].get("side") if normalized_updates else None,
                "first_price_level": normalized_updates[0].get("price_level") if normalized_updates else None,
                "first_quantity": normalized_updates[0].get("quantity") if normalized_updates else None,
            }
        if channel == "heartbeats":
            return {"event_name": "heartbeat"}
    except Exception:
        return None
    return None


class CoinbaseAdvancedRecorderService:
    def __init__(self, config: CoinbaseAdvancedRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="coinbase_advanced", config=config)
        static_fields = self._runtime.manifest_fields(
            {"source": "coinbase", "venue": "advanced", "product_id": config.product_id}
        )
        self._ws_quality = HourlyQualityTracker()
        self._continuity = CoinbaseSequenceTracker(heartbeat_stale_seconds=config.heartbeat_stale_seconds)
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_extra_fields_provider=self._ws_quality.manifest_extra_fields,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
        )

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._ws_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        self._lifecycle_writer.write(_lifecycle("service_start", self.config.product_id, {"mode": "public_advanced_ws"}))
        try:
            await self._websocket_loop()
        finally:
            self._lifecycle_writer.write(_lifecycle("service_stop", self.config.product_id, {"mode": "public_advanced_ws"}))
            self._ws_writer.close()
            self._lifecycle_writer.close()

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            self._advance_writers()
            connection_id = generate_connection_id("coinbase")
            try:
                await self._consume_websocket(connection_id)
                delay = self.config.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._lifecycle_writer.write(
                    _lifecycle("websocket_error", self.config.product_id, {"connection_id": connection_id, "error": repr(exc)})
                )
                await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                delay = min(self.config.reconnect_max_seconds, max(self.config.reconnect_min_seconds, delay * 2))

    async def _consume_websocket(self, connection_id: str) -> None:
        async with connect(
            self.config.websocket_url,
            ping_interval=self.config.ping_interval_seconds,
            ping_timeout=self.config.ping_timeout_seconds,
            max_size=self.config.max_message_size_bytes,
        ) as ws:
            self._continuity.reset()
            for channel in self.config.channels:
                await ws.send(
                    orjson.dumps(
                        {"type": "subscribe", "channel": channel, "product_ids": [self.config.product_id]}
                    ).decode("utf-8")
                )
            self._lifecycle_writer.write(
                _lifecycle(
                    "websocket_connected",
                    self.config.product_id,
                    {"connection_id": connection_id, "channels": list(self.config.channels)},
                )
            )
            async for message in ws:
                recv_ts_ns = utc_now_ns()
                self._advance_writers(recv_ts_ns)
                payload = orjson.loads(message)
                observation = self._continuity.observe(
                    payload if isinstance(payload, dict) else {},
                    recv_ts_ns=recv_ts_ns,
                )
                record = {
                    "record_type": "ws_event",
                    "source": "coinbase",
                    "venue": "advanced",
                    "product_id": self.config.product_id,
                    "connection_id": connection_id,
                    "channel": payload.get("channel"),
                    "sequence_num": payload.get("sequence_num"),
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": ns_to_iso(recv_ts_ns),
                    "recv_ts_source": recv_timestamp_source(),
                    "payload": payload,
                    "source_timestamps": _source_timestamps(payload if isinstance(payload, dict) else {}),
                    "normalized": _normalized(payload if isinstance(payload, dict) else {}),
                    "continuity_state": observation.state,
                }
                self._ws_writer.write(record)
                for event in observation.lifecycle_events:
                    details = {
                        "connection_id": connection_id,
                        "channel": payload.get("channel") if isinstance(payload, dict) else None,
                    }
                    details.update({key: value for key, value in event.items() if key != "event"})
                    self._lifecycle_writer.write(
                        _lifecycle(
                            str(event["event"]),
                            self.config.product_id,
                            details,
                        )
                    )
                if observation.quality_class is not None:
                    self._ws_quality.mark(
                        ts_ns=recv_ts_ns,
                        quality_class=observation.quality_class,
                        flags=observation.quality_flags,
                    )
                if observation.requires_reset:
                    raise CoinbaseSequenceResetRequired(observation.state)
