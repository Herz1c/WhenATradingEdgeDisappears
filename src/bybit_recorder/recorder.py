from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx
import orjson
from websockets import connect

from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.asyncio_utils import cancel_and_await
from market_recorders.file_quality import HourlyQualityTracker
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import (
    build_capture_timing,
    build_clock_probe,
    build_timestamp_fields,
    recv_timestamp_source,
    utc_now_ns,
)

from .config import BybitRecorderConfig
from .integrity import BybitContinuityMonitor


def _lifecycle(event: str, symbol: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "bybit",
        "symbol": symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


def _source_timestamps(payload: dict[str, Any]) -> dict[str, Any]:
    timestamps: dict[str, Any] = {}
    for field in ("ts", "time", "T", "updatedTime", "nextFundingTime", "cts"):
        if field in payload:
            timestamps.update(build_timestamp_fields(field.lower(), payload[field]))
    if isinstance(payload.get("data"), list) and payload["data"]:
        first = payload["data"][0]
        if isinstance(first, dict):
            for field in ("T", "time"):
                if field in first:
                    timestamps.update(build_timestamp_fields(f"item_{field.lower()}", first[field]))
    elif isinstance(payload.get("data"), dict):
        data = payload["data"]
        for field in ("u", "seq", "cts"):
            if field in data and field == "cts":
                timestamps.update(build_timestamp_fields(field.lower(), data[field]))
    return timestamps


class BybitRecorderService:
    def __init__(self, config: BybitRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._ws_quality = HourlyQualityTracker()
        self._continuity = BybitContinuityMonitor()
        self._runtime = build_runtime_metadata(recorder_service="bybit", config=config)
        self._last_lifecycle_heartbeat_ns = 0
        static_fields = self._runtime.manifest_fields({"source": "bybit", "venue": "linear", "symbol": config.symbol})
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_extra_fields_provider=self._ws_quality.manifest_extra_fields,
        )
        self._rest_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="rest",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
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
        self._rest_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)
        self._maybe_write_lifecycle_heartbeat(ts_ns)

    def _maybe_write_lifecycle_heartbeat(self, now_ns: int) -> None:
        interval_ns = int(self.config.lifecycle_heartbeat_interval_seconds * 1_000_000_000)
        if interval_ns <= 0:
            return
        if self._last_lifecycle_heartbeat_ns and now_ns - self._last_lifecycle_heartbeat_ns < interval_ns:
            return
        self._last_lifecycle_heartbeat_ns = now_ns
        self._lifecycle_writer.write(
            _lifecycle(
                "service_heartbeat",
                self.config.symbol,
                self._runtime.lifecycle_fields(
                    extra_fields={
                        "mode": "public_linear_ws_rest",
                        "heartbeat_interval_seconds": self.config.lifecycle_heartbeat_interval_seconds,
                    }
                ),
            )
        )

    async def run(self) -> None:
        timeout = httpx.Timeout(self.config.http_timeout_seconds)
        async with httpx.AsyncClient(base_url=self.config.rest_base_url, timeout=timeout) as client:
            rest_task = asyncio.create_task(self._rest_loop(client), name="bybit:rest")
            self._lifecycle_writer.write(
                _lifecycle(
                    "service_start",
                    self.config.symbol,
                    self._runtime.lifecycle_fields(
                        include_config_snapshot=True,
                        extra_fields={"mode": "public_linear_ws_rest"},
                    ),
                )
            )
            try:
                await self._websocket_loop()
            finally:
                cancelled_during_cleanup = await cancel_and_await(rest_task)
                self._lifecycle_writer.write(
                    _lifecycle(
                        "service_stop",
                        self.config.symbol,
                        self._runtime.lifecycle_fields(extra_fields={"mode": "public_linear_ws_rest"}),
                    )
                )
                self._ws_writer.close()
                self._rest_writer.close()
                self._lifecycle_writer.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _rest_loop(self, client: httpx.AsyncClient) -> None:
        last_metrics_ns = 0
        while True:
            self._advance_writers()
            await self._fetch_rest(client, endpoint="server_time", path="/v5/market/time", params={})
            await self._fetch_rest(
                client,
                endpoint="orderbook_snapshot",
                path="/v5/market/orderbook",
                params={"category": "linear", "symbol": self.config.symbol, "limit": 50},
            )
            await self._fetch_rest(
                client,
                endpoint="tickers",
                path="/v5/market/tickers",
                params={"category": "linear", "symbol": self.config.symbol},
            )
            await self._fetch_rest(
                client,
                endpoint="recent_trade",
                path="/v5/market/recent-trade",
                params={"category": "linear", "symbol": self.config.symbol, "limit": 200},
            )
            now_ns = utc_now_ns()
            if now_ns - last_metrics_ns >= int(self.config.metrics_rest_poll_interval_seconds * 1_000_000_000):
                await self._fetch_rest(
                    client,
                    endpoint="open_interest",
                    path="/v5/market/open-interest",
                    params={
                        "category": "linear",
                        "symbol": self.config.symbol,
                        "intervalTime": self.config.open_interest_interval,
                    },
                )
                await self._fetch_rest(
                    client,
                    endpoint="kline",
                    path="/v5/market/kline",
                    params={"category": "linear", "symbol": self.config.symbol, "interval": "5", "limit": 30},
                )
                last_metrics_ns = now_ns
            await asyncio.sleep(self.config.current_rest_poll_interval_seconds)

    async def _fetch_rest(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str,
        path: str,
        params: dict[str, Any],
    ) -> None:
        request_start_ns = utc_now_ns()
        response = await client.get(path, params=params)
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        recv_ts_ns = utc_now_ns()
        record = {
            "record_type": "rest_observation",
            "source": "bybit",
            "venue": "linear",
            "symbol": self.config.symbol,
            "endpoint": endpoint,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "payload": payload,
            "source_timestamps": _source_timestamps(payload if isinstance(payload, dict) else {}),
        }
        if endpoint == "server_time" and isinstance(payload, dict):
            result = payload.get("result", {})
            record["clock_probe"] = build_clock_probe(
                "server_time",
                result.get("timeNano") or result.get("timeSecond"),
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            )
        self._rest_writer.write(record)

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            connection_id = generate_connection_id("bybit")
            try:
                await self._consume_websocket(connection_id)
                delay = self.config.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._lifecycle_writer.write(
                    _lifecycle("websocket_error", self.config.symbol, {"connection_id": connection_id, "error": repr(exc)})
                )
                await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                delay = min(self.config.reconnect_max_seconds, max(self.config.reconnect_min_seconds, delay * 2))

    async def _consume_websocket(self, connection_id: str) -> None:
        async with connect(
            self.config.websocket_url,
            max_queue=self.config.websocket_max_queue,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            await ws.send(orjson.dumps({"op": "subscribe", "args": list(self.config.topics)}).decode("utf-8"))
            self._lifecycle_writer.write(
                _lifecycle("websocket_connected", self.config.symbol, {"connection_id": connection_id, "topics": self.config.topics})
            )
            async for message in ws:
                recv_ts_ns = utc_now_ns()
                self._advance_writers(recv_ts_ns)
                payload = orjson.loads(message)
                observation = self._continuity.observe(payload if isinstance(payload, dict) else {})
                record = {
                    "record_type": "ws_event",
                    "source": "bybit",
                    "venue": "linear",
                    "symbol": self.config.symbol,
                    "connection_id": connection_id,
                    "topic": payload.get("topic"),
                    "type": payload.get("type"),
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": ns_to_iso(recv_ts_ns),
                    "recv_ts_source": recv_timestamp_source(),
                    "payload": payload,
                    "source_timestamps": _source_timestamps(payload if isinstance(payload, dict) else {}),
                    "continuity_state": observation.continuity_state,
                }
                self._ws_writer.write(record)
                if observation.quality_class is not None:
                    self._ws_quality.mark(
                        ts_ns=recv_ts_ns,
                        quality_class=observation.quality_class,
                        flags=observation.quality_flags,
                    )
                for event in observation.lifecycle_events:
                    details = {"connection_id": connection_id}
                    details.update({key: value for key, value in event.items() if key != "event"})
                    self._lifecycle_writer.write(_lifecycle(str(event["event"]), self.config.symbol, details))
