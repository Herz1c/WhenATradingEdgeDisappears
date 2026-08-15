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
    build_recursive_epoch_timestamp_fields,
    recv_timestamp_source,
    utc_now_ns,
)

from .config import HyperliquidRecorderConfig
from .health import HyperliquidRestCircuitBreaker

_TS_FIELDS = (
    "time",
    "timestamp",
    "ts",
    "created_at",
    "createdAt",
    "updated_at",
    "updatedAt",
    "fundingTime",
)


def _lifecycle(event: str, symbol: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "hyperliquid",
        "market_type": "linear",
        "symbol": symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


def _source_timestamps(payload: Any) -> dict[str, Any]:
    return build_recursive_epoch_timestamp_fields(payload, candidate_field_names=_TS_FIELDS)


def _topic(payload: dict[str, Any]) -> str | None:
    if payload.get("channel") is not None:
        return str(payload["channel"])
    data = payload.get("data")
    if isinstance(data, dict) and data.get("type") is not None:
        return str(data["type"])
    if payload.get("subscription") is not None:
        subscription = payload.get("subscription")
        if isinstance(subscription, dict) and subscription.get("type") is not None:
            return str(subscription["type"])
    return str(payload.get("type")) if payload.get("type") is not None else None


def _extract_meta_and_asset_ctx(payload: Any, symbol: str) -> dict[str, Any]:
    if not (isinstance(payload, list) and len(payload) >= 2):
        return {}
    meta = payload[0]
    contexts = payload[1]
    if not (isinstance(meta, dict) and isinstance(contexts, list)):
        return {}
    universe = meta.get("universe", [])
    if not isinstance(universe, list):
        return {}
    for asset_meta, asset_ctx in zip(universe, contexts, strict=False):
        if not (isinstance(asset_meta, dict) and isinstance(asset_ctx, dict)):
            continue
        if str(asset_meta.get("name") or "").upper() != symbol.upper():
            continue
        extracted = {
            "coin": symbol,
            "fundingRate": asset_ctx.get("fundingRate", asset_ctx.get("funding")),
            "openInterest": asset_ctx.get("openInterest"),
            "markPx": asset_ctx.get("markPx"),
            "midPx": asset_ctx.get("midPx"),
        }
        return {key: value for key, value in extracted.items() if value is not None}
    return {}


def _rest_timestamp_semantics(endpoint: str) -> str:
    if endpoint == "fundingHistory":
        return "historical_series"
    return "source_event"


def _rest_endpoint_role(endpoint: str) -> str:
    if endpoint == "fundingHistory":
        return "historical_context"
    return "realtime_context"


class HyperliquidRecorderService:
    def __init__(self, config: HyperliquidRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._rest_quality = HourlyQualityTracker()
        self._runtime = build_runtime_metadata(recorder_service="hyperliquid", config=config)
        self._last_lifecycle_heartbeat_ns = 0
        self._rest_health = HyperliquidRestCircuitBreaker(
            degraded_failure_threshold=config.rest_degraded_failure_threshold,
            suspended_failure_threshold=config.rest_suspended_failure_threshold,
            degraded_backoff_seconds=config.rest_degraded_backoff_seconds,
            suspended_backoff_seconds=config.rest_suspended_backoff_seconds,
        )
        static_fields = self._runtime.manifest_fields(
            {"source": "hyperliquid", "market_type": "linear", "symbol": config.symbol}
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
            manifest_extra_fields_provider=self._rest_quality.manifest_extra_fields,
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
            rest_task = asyncio.create_task(self._rest_loop(client), name="hyperliquid:rest")
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
        self._lifecycle_writer.write(
            _lifecycle(
                "rest_loop_started",
                self.config.symbol,
                {
                    "current_poll_interval_seconds": self.config.current_rest_poll_interval_seconds,
                    "metrics_poll_interval_seconds": self.config.metrics_rest_poll_interval_seconds,
                },
            )
        )
        while True:
            self._advance_writers()
            gate = self._rest_health.before_cycle(now_ns=utc_now_ns())
            for event in gate.lifecycle_events:
                self._lifecycle_writer.write(_lifecycle(str(event["event"]), self.config.symbol, dict(event)))
            if gate.skip_cycle:
                self._rest_quality.mark(
                    ts_ns=utc_now_ns(),
                    quality_class="stale_context",
                    flags=("rest_suspended",),
                    details={"rest_health_state": gate.current_state},
                )
                await asyncio.sleep(gate.sleep_seconds or self.config.current_rest_poll_interval_seconds)
                continue
            try:
                last_metrics_ns = await self._rest_cycle(client, last_metrics_ns=last_metrics_ns)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                health = self._rest_health.observe_failure(
                    now_ns=utc_now_ns(),
                    status_code=status_code,
                    reason=exc.__class__.__name__,
                )
                details = {"error": repr(exc)}
                if isinstance(exc, httpx.HTTPStatusError):
                    details["status_code"] = exc.response.status_code
                    details["endpoint"] = exc.request.url.path
                details["rest_health_state"] = health.current_state
                self._lifecycle_writer.write(
                    _lifecycle("rest_error", self.config.symbol, details)
                )
                for event in health.lifecycle_events:
                    self._lifecycle_writer.write(_lifecycle(str(event["event"]), self.config.symbol, dict(event)))
                self._rest_quality.mark(
                    ts_ns=utc_now_ns(),
                    quality_class="stale_context" if health.current_state in {"degraded", "suspended"} else None,
                    flags=(f"rest_{health.current_state}",) if health.current_state != "healthy" else (),
                    details={"rest_health_state": health.current_state},
                )
                await asyncio.sleep(
                    health.sleep_seconds
                    + random.random() * self.config.reconnect_jitter_seconds
                )
                continue
            health = self._rest_health.observe_success()
            for event in health.lifecycle_events:
                self._lifecycle_writer.write(_lifecycle(str(event["event"]), self.config.symbol, dict(event)))
            await asyncio.sleep(self.config.current_rest_poll_interval_seconds)

    async def _rest_cycle(self, client: httpx.AsyncClient, *, last_metrics_ns: int) -> int:
        await self._post_info(
            client,
            endpoint="metaAndAssetCtxs",
            body={"type": "metaAndAssetCtxs"},
            extracted=_extract_meta_and_asset_ctx,
        )
        await self._post_info(
            client,
            endpoint="l2Book",
            body={"type": "l2Book", "coin": self.config.symbol},
        )
        now_ns = utc_now_ns()
        if now_ns - last_metrics_ns >= int(self.config.metrics_rest_poll_interval_seconds * 1_000_000_000):
            now_ms = now_ns // 1_000_000
            await self._post_info(
                client,
                endpoint="fundingHistory",
                body={
                    "type": "fundingHistory",
                    "coin": self.config.symbol,
                    "startTime": now_ms - 3_600_000,
                },
            )
            last_metrics_ns = now_ns
        return last_metrics_ns

    async def _post_info(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str,
        body: dict[str, Any],
        extracted: Any | None = None,
    ) -> None:
        request_start_ns = utc_now_ns()
        response = await client.post("/info", json=body)
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        payload = response.json()
        recv_ts_ns = utc_now_ns()
        record = {
            "record_type": "rest_observation",
            "source": "hyperliquid",
            "market_type": "linear",
            "symbol": self.config.symbol,
            "endpoint": endpoint,
            "endpoint_role": _rest_endpoint_role(endpoint),
            "timestamp_semantics": _rest_timestamp_semantics(endpoint),
            "rest_health_state": self._rest_health.current_state,
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
        if callable(extracted):
            record["extracted"] = extracted(payload, self.config.symbol)
        self._rest_writer.write(record)

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            connection_id = generate_connection_id("hyperliquid")
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
            ping_interval=20,
            ping_timeout=20,
            max_size=self.config.websocket_max_size_bytes,
        ) as ws:
            for payload in self.config.subscriptions:
                await ws.send(orjson.dumps(payload).decode("utf-8"))
            self._lifecycle_writer.write(
                _lifecycle(
                    "websocket_connected",
                    self.config.symbol,
                    {"connection_id": connection_id, "ws_url": self.config.websocket_url},
                )
            )
            local_msg_seq = 0
            async for message in ws:
                recv_ts_ns = utc_now_ns()
                self._advance_writers(recv_ts_ns)
                payload = orjson.loads(message)
                local_msg_seq += 1
                record = {
                    "record_type": "ws_event",
                    "source": "hyperliquid",
                    "market_type": "linear",
                    "symbol": self.config.symbol,
                    "connection_id": connection_id,
                    "local_msg_seq": local_msg_seq,
                    "ws_url": self.config.websocket_url,
                    "topic": _topic(payload if isinstance(payload, dict) else {}),
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": ns_to_iso(recv_ts_ns),
                    "recv_ts_source": recv_timestamp_source(),
                    "payload": payload,
                    "source_timestamps": _source_timestamps(payload),
                }
                self._ws_writer.write(record)
