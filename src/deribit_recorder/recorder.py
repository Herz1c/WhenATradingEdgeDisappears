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
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import (
    build_capture_timing,
    build_recursive_epoch_timestamp_fields,
    recv_timestamp_source,
    utc_now_ns,
)

from .config import DeribitRecorderConfig

_TS_FIELDS = (
    "timestamp",
    "creation_timestamp",
    "time",
    "usIn",
    "usOut",
    "usDiff",
)


def _lifecycle(event: str, currency: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "deribit",
        "market_type": "options",
        "symbol": currency,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


def _source_timestamps(payload: Any) -> dict[str, Any]:
    return build_recursive_epoch_timestamp_fields(payload, candidate_field_names=_TS_FIELDS)


def _timestamp_semantics(endpoint: str | None = None) -> str:
    del endpoint
    return "metadata_only_sparse"


class DeribitRecorderService:
    def __init__(self, config: DeribitRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="deribit", config=config)
        self._last_lifecycle_heartbeat_ns = 0
        static_fields = self._runtime.manifest_fields(
            {"source": "deribit", "market_type": "options", "symbol": config.currency}
        )
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
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
                self.config.currency,
                self._runtime.lifecycle_fields(
                    extra_fields={
                        "mode": "public_options_ws_rest",
                        "heartbeat_interval_seconds": self.config.lifecycle_heartbeat_interval_seconds,
                    }
                ),
            )
        )

    async def run(self) -> None:
        timeout = httpx.Timeout(self.config.http_timeout_seconds)
        async with httpx.AsyncClient(base_url=self.config.rest_base_url, timeout=timeout) as client:
            rest_task = asyncio.create_task(self._rest_loop(client), name="deribit:rest")
            self._lifecycle_writer.write(
                _lifecycle(
                    "service_start",
                    self.config.currency,
                    self._runtime.lifecycle_fields(
                        include_config_snapshot=True,
                        extra_fields={"mode": "public_options_ws_rest"},
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
                        self.config.currency,
                        self._runtime.lifecycle_fields(extra_fields={"mode": "public_options_ws_rest"}),
                    )
                )
                self._ws_writer.close()
                self._rest_writer.close()
                self._lifecycle_writer.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _rest_loop(self, client: httpx.AsyncClient) -> None:
        last_iv_ns = 0
        last_summary_ns = 0
        last_instruments_ns = 0
        while True:
            self._advance_writers()
            now_ns = utc_now_ns()
            now_ms = now_ns // 1_000_000
            if now_ns - last_iv_ns >= int(self.config.iv_poll_interval_seconds * 1_000_000_000):
                await self._get(
                    client,
                    endpoint="volatility_index",
                    path="/api/v2/public/get_volatility_index_data",
                    params={
                        "currency": self.config.currency,
                        "resolution": 60,
                        "start_timestamp": now_ms - 3_600_000,
                        "end_timestamp": now_ms,
                    },
                )
                last_iv_ns = now_ns
            if now_ns - last_summary_ns >= int(self.config.summary_poll_interval_seconds * 1_000_000_000):
                await self._get(
                    client,
                    endpoint="book_summary",
                    path="/api/v2/public/get_book_summary_by_currency",
                    params={"currency": self.config.currency, "kind": "option"},
                )
                await self._get(
                    client,
                    endpoint="index_price",
                    path="/api/v2/public/get_index_price",
                    params={"index_name": "btc_usd"},
                )
                last_summary_ns = now_ns
            if now_ns - last_instruments_ns >= int(self.config.instruments_poll_interval_seconds * 1_000_000_000):
                await self._get(
                    client,
                    endpoint="instruments",
                    path="/api/v2/public/get_instruments",
                    params={"currency": self.config.currency, "kind": "option", "expired": "false"},
                )
                last_instruments_ns = now_ns
            await asyncio.sleep(1.0)

    async def _get(
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
            "source": "deribit",
            "market_type": "options",
            "symbol": self.config.currency,
            "endpoint": endpoint,
            "timestamp_semantics": _timestamp_semantics(endpoint),
            "url": str(response.request.url),
            "http_status": response.status_code,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "payload": payload,
            "source_timestamps": _source_timestamps(payload),
        }
        self._rest_writer.write(record)

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            connection_id = generate_connection_id("deribit")
            try:
                await self._consume_websocket(connection_id)
                delay = self.config.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._lifecycle_writer.write(
                    _lifecycle("websocket_error", self.config.currency, {"connection_id": connection_id, "error": repr(exc)})
                )
                await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                delay = min(self.config.reconnect_max_seconds, max(self.config.reconnect_min_seconds, delay * 2))

    async def _consume_websocket(self, connection_id: str) -> None:
        async with connect(
            self.config.websocket_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=self.config.websocket_max_size_bytes,
        ) as ws:
            await ws.send(
                orjson.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "public/subscribe",
                        "params": {"channels": list(self.config.ws_channels)},
                    }
                ).decode("utf-8")
            )
            self._lifecycle_writer.write(
                _lifecycle(
                    "websocket_connected",
                    self.config.currency,
                    {"connection_id": connection_id, "ws_url": self.config.websocket_url},
                )
            )
            local_msg_seq = 0
            async for message in ws:
                recv_ts_ns = utc_now_ns()
                self._advance_writers(recv_ts_ns)
                payload = orjson.loads(message)
                local_msg_seq += 1
                params = payload.get("params") if isinstance(payload, dict) else None
                topic = params.get("channel") if isinstance(params, dict) else "price_index_ws"
                record = {
                    "record_type": "ws_event",
                    "source": "deribit",
                    "market_type": "options",
                    "symbol": self.config.currency,
                    "connection_id": connection_id,
                    "local_msg_seq": local_msg_seq,
                    "ws_url": self.config.websocket_url,
                    "topic": topic,
                    "timestamp_semantics": _timestamp_semantics(),
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": ns_to_iso(recv_ts_ns),
                    "recv_ts_source": recv_timestamp_source(),
                    "payload": payload,
                    "source_timestamps": _source_timestamps(payload),
                }
                self._ws_writer.write(record)
