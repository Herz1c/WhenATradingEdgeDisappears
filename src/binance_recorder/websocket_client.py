from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any

import httpx
import orjson
from websockets import ConnectionClosed, connect

from binance_recorder.config import RecorderConfig
from binance_recorder.lifecycle import lifecycle_event
from binance_recorder.order_book_qc import BinanceOrderBookIntegrityMonitor
from binance_recorder.source_metadata import extract_binance_source_timestamps, normalize_binance_payload
from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.asyncio_utils import cancel_and_await
from market_recorders.file_quality import HourlyQualityTracker
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import (
    build_capture_timing,
    build_clock_probe,
    recv_timestamp_source,
    utc_now_ns,
)


class BinanceRecorderService:
    def __init__(self, config: RecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._ws_quality = HourlyQualityTracker()
        self._order_book_monitor: BinanceOrderBookIntegrityMonitor | None = None
        self._runtime = build_runtime_metadata(recorder_service="binance_spot", config=config)
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=self._runtime.manifest_fields({
                "source": "binance",
                "venue": "spot",
                "symbol": config.symbol,
            }),
            manifest_extra_fields_provider=self._ws_quality.manifest_extra_fields,
            logger=self.logger,
        )
        self._snapshot_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="snapshot",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=self._runtime.manifest_fields({
                "source": "binance",
                "venue": "spot",
                "symbol": config.symbol,
            }),
            logger=self.logger,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=self._runtime.manifest_fields({
                "source": "binance",
                "venue": "spot",
                "symbol": config.symbol,
            }),
            logger=self.logger,
        )

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._ws_writer.advance_time(now_ns=ts_ns)
        self._snapshot_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        timeout = httpx.Timeout(self.config.http_timeout_seconds)
        async with httpx.AsyncClient(base_url=self.config.rest_base_url, timeout=timeout) as client:
            self._write_lifecycle("service_start", {"mode": "spot_combined_streams"})
            snapshot_task = asyncio.create_task(self._snapshot_and_probe_loop(client), name="binance_spot:snapshot")
            try:
                await self._websocket_loop(client)
            finally:
                cancelled_during_cleanup = await cancel_and_await(snapshot_task)
                self._write_lifecycle("service_stop", {"mode": "spot_combined_streams"})
                self._ws_writer.close()
                self._snapshot_writer.close()
                self._lifecycle_writer.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _snapshot_and_probe_loop(self, client: httpx.AsyncClient) -> None:
        last_snapshot_ns = 0
        last_server_time_ns = 0
        while True:
            self._advance_writers()
            now_ns = utc_now_ns()
            if now_ns - last_server_time_ns >= int(self.config.server_time_poll_interval_seconds * 1_000_000_000):
                await self._fetch_server_time(client)
                last_server_time_ns = utc_now_ns()
            if now_ns - last_snapshot_ns >= int(self.config.periodic_snapshot_interval_seconds * 1_000_000_000):
                await self._fetch_snapshot(client)
                last_snapshot_ns = utc_now_ns()
            await asyncio.sleep(1.0)

    async def _fetch_server_time(self, client: httpx.AsyncClient) -> None:
        request_start_ns = utc_now_ns()
        response = await client.get("/api/v3/time")
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        event = lifecycle_event(
            event="server_time_probe",
            symbol=self.config.symbol,
            market="spot",
            details={
                "endpoint": "server_time",
                "capture_timing": build_capture_timing(
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                ),
                "clock_probe": build_clock_probe(
                    "server_time",
                    payload.get("serverTime"),
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                    unit_hint="millisecond",
                ),
                "raw": {"public_json": payload},
            },
        )
        self._lifecycle_writer.write(event)

    async def _fetch_snapshot(self, client: httpx.AsyncClient, *, capture_reason: str = "periodic") -> dict[str, Any]:
        request_start_ns = utc_now_ns()
        response = await client.get(
            "/api/v3/depth",
            params={"symbol": self.config.symbol, "limit": self.config.snapshot_limit},
        )
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        recv_ts_ns = utc_now_ns()
        record = {
            "record_type": "snapshot",
            "source": "binance",
            "venue": "spot",
            "symbol": self.config.symbol,
            "endpoint": "depth_snapshot",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "raw": {"public_json": payload},
            "normalized": {
                "last_update_id": payload.get("lastUpdateId"),
                "bid_level_count": len(payload.get("bids", [])),
                "ask_level_count": len(payload.get("asks", [])),
                "snapshot_limit": self.config.snapshot_limit,
                "capture_reason": capture_reason,
            },
        }
        self._snapshot_writer.write(record)
        if capture_reason == "periodic" and self._order_book_monitor is not None:
            observation = self._order_book_monitor.observe_periodic_snapshot(payload)
            self._handle_order_book_observation(
                observation,
                connection_id=None,
                recv_ts_ns=recv_ts_ns,
                context="periodic_snapshot",
            )
        return payload

    async def _websocket_loop(self, client: httpx.AsyncClient) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            connection_id = generate_connection_id("binance-spot")
            try:
                await self._consume_websocket(connection_id, client)
                delay = self.config.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._write_lifecycle(
                    "websocket_error",
                    {"connection_id": connection_id, "error": repr(exc)},
                )
                await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                delay = min(self.config.reconnect_max_seconds, max(self.config.reconnect_min_seconds, delay * 2))

    async def _consume_websocket(self, connection_id: str, client: httpx.AsyncClient) -> None:
        stream_path = "/".join(self.config.streams)
        url = (
            f"{self.config.websocket_base_url}/stream?streams={stream_path}"
            f"&timeUnit={self.config.websocket_time_unit}"
        )
        self._write_lifecycle("websocket_connecting", {"connection_id": connection_id, "url": url})
        self._order_book_monitor = BinanceOrderBookIntegrityMonitor(
            symbol=self.config.symbol,
            market="spot",
            parity_check_update_lag_tolerance=self.config.order_book_parity_check_update_lag_tolerance,
        )
        async with connect(url, max_queue=self.config.websocket_max_queue, ping_interval=20, ping_timeout=20) as ws:
            self._write_lifecycle("websocket_connected", {"connection_id": connection_id, "url": url})
            pending_snapshot_reason = "bootstrap"
            pending_snapshot_task: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(
                self._fetch_snapshot(client, capture_reason=pending_snapshot_reason),
                name=f"binance_spot:snapshot:{connection_id}",
            )
            try:
                async for message in ws:
                    recv_ts_ns = utc_now_ns()
                    self._advance_writers(recv_ts_ns)
                    if pending_snapshot_task is not None and pending_snapshot_task.done():
                        snapshot_payload = await pending_snapshot_task
                        observation = self._order_book_monitor.load_snapshot(
                            snapshot_payload,
                            reason=pending_snapshot_reason,
                        )
                        self._handle_order_book_observation(
                            observation,
                            connection_id=connection_id,
                            recv_ts_ns=recv_ts_ns,
                            context=pending_snapshot_reason,
                        )
                        pending_snapshot_task = None
                        if observation.requires_resnapshot:
                            pending_snapshot_reason = "resync"
                            pending_snapshot_task = asyncio.create_task(
                                self._fetch_snapshot(client, capture_reason=pending_snapshot_reason),
                                name=f"binance_spot:snapshot:{connection_id}:resync",
                            )
                    envelope = orjson.loads(message)
                    payload = envelope.get("data", {})
                    stream = envelope.get("stream")
                    normalized = normalize_binance_payload(payload, stream=stream)
                    order_book_observation = None
                    if isinstance(payload, dict) and payload.get("e") == "depthUpdate" and self._order_book_monitor is not None:
                        order_book_observation = self._order_book_monitor.observe_depth_event(payload)
                        self._handle_order_book_observation(
                            order_book_observation,
                            connection_id=connection_id,
                            recv_ts_ns=recv_ts_ns,
                            context="depth_update",
                        )
                        if order_book_observation.requires_resnapshot and pending_snapshot_task is None:
                            pending_snapshot_reason = "resync"
                            pending_snapshot_task = asyncio.create_task(
                                self._fetch_snapshot(client, capture_reason=pending_snapshot_reason),
                                name=f"binance_spot:snapshot:{connection_id}:resync",
                            )
                    record = {
                        "record_type": "ws_event",
                        "source": "binance",
                        "venue": "spot",
                        "symbol": self.config.symbol,
                        "stream": stream,
                        "connection_id": connection_id,
                        "event_type": payload.get("e") or normalized.get("normalized_event_type"),
                        "recv_ts_ns": recv_ts_ns,
                        "recv_ts_iso": ns_to_iso(recv_ts_ns),
                        "recv_ts_source": recv_timestamp_source(),
                        "payload": payload,
                        "source_timestamps": extract_binance_source_timestamps(
                            payload,
                            unit_hint="microsecond" if self.config.websocket_time_unit == "MICROSECOND" else None,
                            stream=stream,
                        ),
                        "normalized": normalized,
                    }
                    if order_book_observation is not None:
                        record["order_book_status"] = order_book_observation.state
                        record["order_book_local_update_id"] = order_book_observation.local_update_id
                    self._ws_writer.write(record)
            finally:
                if pending_snapshot_task is not None:
                    cancelled_during_cleanup = await cancel_and_await(pending_snapshot_task)
                    if cancelled_during_cleanup:
                        raise asyncio.CancelledError
        self._write_lifecycle("websocket_closed", {"connection_id": connection_id, "url": url})
        self._order_book_monitor = None

    def _write_lifecycle(self, event: str, details: dict[str, Any] | None = None) -> None:
        self._lifecycle_writer.write(
            lifecycle_event(event=event, symbol=self.config.symbol, market="spot", details=details)
        )

    def _handle_order_book_observation(
        self,
        observation: Any,
        *,
        connection_id: str | None,
        recv_ts_ns: int,
        context: str,
    ) -> None:
        if observation is None:
            return
        if getattr(observation, "quality_class", None) is not None:
            self._ws_quality.mark(
                ts_ns=recv_ts_ns,
                quality_class=observation.quality_class,
                flags=getattr(observation, "quality_flags", ()),
            )
        for event in getattr(observation, "lifecycle_events", []):
            details = {
                "connection_id": connection_id,
                "context": context,
            }
            details.update({key: value for key, value in event.items() if key != "event"})
            self._write_lifecycle(str(event["event"]), details)
