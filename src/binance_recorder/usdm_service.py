from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx
import orjson
from websockets import connect

from binance_recorder.lifecycle import lifecycle_event
from binance_recorder.order_book_qc import BinanceOrderBookIntegrityMonitor
from binance_recorder.source_metadata import extract_binance_source_timestamps, normalize_binance_payload
from binance_recorder.usdm_config import BinanceUsdmRecorderConfig
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


class BinanceUsdmRecorderService:
    def __init__(self, config: BinanceUsdmRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._ws_quality = HourlyQualityTracker()
        self._order_book_monitor: BinanceOrderBookIntegrityMonitor | None = None
        self._runtime = build_runtime_metadata(recorder_service="binance_usdm", config=config)
        static_fields = self._runtime.manifest_fields({"source": "binance", "venue": "usdm", "symbol": config.symbol})
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_extra_fields_provider=self._ws_quality.manifest_extra_fields,
            logger=self.logger,
        )
        self._rest_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="rest",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            logger=self.logger,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            logger=self.logger,
        )

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._ws_writer.advance_time(now_ns=ts_ns)
        self._rest_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        timeout = httpx.Timeout(self.config.http_timeout_seconds)
        async with httpx.AsyncClient(base_url=self.config.rest_base_url, timeout=timeout) as client:
            self._write_lifecycle("service_start", {"mode": "futures_public_ws_rest"})
            rest_task = asyncio.create_task(self._rest_loop(client), name="binance_usdm:rest")
            try:
                await self._websocket_loop(client)
            finally:
                cancelled_during_cleanup = await cancel_and_await(rest_task)
                self._write_lifecycle("service_stop", {"mode": "futures_public_ws_rest"})
                self._ws_writer.close()
                self._rest_writer.close()
                self._lifecycle_writer.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _rest_loop(self, client: httpx.AsyncClient) -> None:
        last_metrics_ns = 0
        self._write_lifecycle(
            "rest_loop_started",
            {
                "current_poll_interval_seconds": self.config.current_rest_poll_interval_seconds,
                "metrics_poll_interval_seconds": self.config.metrics_rest_poll_interval_seconds,
            },
        )
        while True:
            self._advance_writers()
            try:
                last_metrics_ns = await self._rest_cycle(client, last_metrics_ns=last_metrics_ns)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._write_lifecycle("rest_error", {"error": repr(exc)})
                await asyncio.sleep(
                    self.config.reconnect_min_seconds
                    + random.random() * self.config.reconnect_jitter_seconds
                )
                continue
            await asyncio.sleep(self.config.current_rest_poll_interval_seconds)

    async def _rest_cycle(self, client: httpx.AsyncClient, *, last_metrics_ns: int) -> int:
        await self._fetch_server_time(client)
        await self._fetch_depth_snapshot(client)
        await self._fetch_premium_index(client)
        await self._fetch_open_interest(client)
        now_ns = utc_now_ns()
        if now_ns - last_metrics_ns >= int(self.config.metrics_rest_poll_interval_seconds * 1_000_000_000):
            await self._fetch_klines(client)
            last_metrics_ns = now_ns
        return last_metrics_ns

    async def _fetch_server_time(self, client: httpx.AsyncClient) -> None:
        request_start_ns = utc_now_ns()
        response = await client.get("/fapi/v1/time")
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        self._lifecycle_writer.write(
            lifecycle_event(
                event="server_time_probe",
                symbol=self.config.symbol,
                market="usdm",
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
        )

    async def _fetch_depth_snapshot(
        self,
        client: httpx.AsyncClient,
        *,
        capture_reason: str = "periodic",
    ) -> dict[str, Any]:
        request_start_ns = utc_now_ns()
        response = await client.get(
            "/fapi/v1/depth",
            params={"symbol": self.config.symbol, "limit": self.config.depth_snapshot_limit},
        )
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        recv_ts_ns = utc_now_ns()
        record = {
            "record_type": "rest_observation",
            "source": "binance",
            "venue": "usdm",
            "symbol": self.config.symbol,
            "endpoint": "depth_snapshot",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "raw": {
                "public_json": payload,
                "request": {
                    "path": "/fapi/v1/depth",
                    "params": {"symbol": self.config.symbol, "limit": self.config.depth_snapshot_limit},
                },
            },
            "normalized": {
                "last_update_id": payload.get("lastUpdateId"),
                "bid_level_count": len(payload.get("bids", [])),
                "ask_level_count": len(payload.get("asks", [])),
                "snapshot_limit": self.config.depth_snapshot_limit,
                "capture_reason": capture_reason,
            },
        }
        self._rest_writer.write(record)
        if capture_reason == "periodic" and self._order_book_monitor is not None:
            observation = self._order_book_monitor.observe_periodic_snapshot(payload)
            self._handle_order_book_observation(
                observation,
                connection_id=None,
                recv_ts_ns=recv_ts_ns,
                context="periodic_snapshot",
            )
        return payload

    async def _fetch_premium_index(self, client: httpx.AsyncClient) -> None:
        await self._fetch_rest(
            client,
            endpoint="premium_index",
            path="/fapi/v1/premiumIndex",
            params={"symbol": self.config.symbol},
        )

    async def _fetch_open_interest(self, client: httpx.AsyncClient) -> None:
        await self._fetch_rest(
            client,
            endpoint="open_interest",
            path="/fapi/v1/openInterest",
            params={"symbol": self.config.symbol},
        )

    async def _fetch_klines(self, client: httpx.AsyncClient) -> None:
        await self._fetch_rest(
            client,
            endpoint="klines",
            path="/fapi/v1/klines",
            params={
                "symbol": self.config.symbol,
                "interval": self.config.history_period,
                "limit": self.config.history_limit,
            },
        )

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
            "source": "binance",
            "venue": "usdm",
            "symbol": self.config.symbol,
            "endpoint": endpoint,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "raw": {"public_json": payload, "request": {"path": path, "params": params}},
        }
        if isinstance(payload, dict):
            normalized_payload = {"e": "markPriceUpdate", **payload} if endpoint == "premium_index" else payload
            record["source_timestamps"] = extract_binance_source_timestamps(
                normalized_payload,
                unit_hint="millisecond",
            )
            record["normalized"] = normalize_binance_payload(
                normalized_payload,
            )
        self._rest_writer.write(record)

    async def _websocket_loop(self, client: httpx.AsyncClient) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            connection_id = generate_connection_id("binance-usdm")
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
        url = f"{self.config.websocket_base_url}/stream?streams={'/'.join(self.config.streams)}"
        self._write_lifecycle("websocket_connecting", {"connection_id": connection_id, "url": url})
        self._order_book_monitor = BinanceOrderBookIntegrityMonitor(
            symbol=self.config.symbol,
            market="usdm",
            parity_check_update_lag_tolerance=self.config.order_book_parity_check_update_lag_tolerance,
        )
        async with connect(url, max_queue=self.config.websocket_max_queue, ping_interval=20, ping_timeout=20) as ws:
            self._write_lifecycle("websocket_connected", {"connection_id": connection_id, "url": url})
            pending_snapshot_reason = "bootstrap"
            pending_snapshot_task: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(
                self._fetch_depth_snapshot(client, capture_reason=pending_snapshot_reason),
                name=f"binance_usdm:snapshot:{connection_id}",
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
                                self._fetch_depth_snapshot(client, capture_reason=pending_snapshot_reason),
                                name=f"binance_usdm:snapshot:{connection_id}:resync",
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
                                self._fetch_depth_snapshot(client, capture_reason=pending_snapshot_reason),
                                name=f"binance_usdm:snapshot:{connection_id}:resync",
                            )
                    record = {
                        "record_type": "ws_event",
                        "source": "binance",
                        "venue": "usdm",
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
                            unit_hint="millisecond",
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
            lifecycle_event(event=event, symbol=self.config.symbol, market="usdm", details=details)
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
            details = {"connection_id": connection_id, "context": context}
            details.update({key: value for key, value in event.items() if key != "event"})
            self._write_lifecycle(str(event["event"]), details)
