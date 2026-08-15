from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import orjson
from websockets import connect

from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.asyncio_utils import cancel_and_await
from market_recorders.file_quality import HourlyQualityTracker
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_epoch_timestamp_fields, recv_timestamp_source, utc_now_ns

from .rtds_config import PolymarketRtdsConfig
from .rtds_qc import RtdsBacklogAnalyzer, RtdsQcThresholds


def build_subscription_payload(config: PolymarketRtdsConfig) -> dict[str, Any]:
    return {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": config.topic,
                "type": config.subscription_type,
                "filters": orjson.dumps({"symbol": config.symbol}).decode("utf-8"),
            }
        ],
    }


def build_source_timestamps(message: dict[str, Any]) -> dict[str, Any]:
    source_timestamps: dict[str, Any] = {}
    source_timestamps.update(
        build_epoch_timestamp_fields("message_timestamp", message.get("timestamp"), unit_hint="millisecond")
    )
    payload = message.get("payload")
    if isinstance(payload, dict):
        source_timestamps.update(
            build_epoch_timestamp_fields("price_timestamp", payload.get("timestamp"), unit_hint="millisecond")
        )
    return source_timestamps


def _build_freshness(*, recv_ts_ns: int, source_timestamps: dict[str, Any]) -> dict[str, Any]:
    reference_ts_ns = None
    if source_timestamps.get("price_timestamp_ms") is not None:
        reference_ts_ns = int(source_timestamps["price_timestamp_ms"]) * 1_000_000
    elif source_timestamps.get("message_timestamp_ms") is not None:
        reference_ts_ns = int(source_timestamps["message_timestamp_ms"]) * 1_000_000
    if reference_ts_ns is None:
        return {}
    return {
        "reference_ts_ns": reference_ts_ns,
        "reference_ts_iso": ns_to_iso(reference_ts_ns),
        "source_age_ms": round((recv_ts_ns - reference_ts_ns) / 1_000_000, 3),
    }


def _lifecycle(
    event: str,
    config: PolymarketRtdsConfig,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    record = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "polymarket",
        "component": "rtds_chainlink",
        "backend": "rtds_chainlink",
        "topic": config.topic,
        "symbol": config.symbol,
        "subscribed_symbol": config.symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        record.update(details)
    return record


class PolymarketRtdsRecorderService:
    def __init__(self, config: PolymarketRtdsConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._ws_quality = HourlyQualityTracker()
        self._runtime = build_runtime_metadata(recorder_service="polymarket_rtds", config=config)
        static_fields = self._runtime.manifest_fields({
            "source": "polymarket_rtds_chainlink",
            "topic": config.topic,
            "symbol": config.symbol,
        })
        counter_fields = ("source",)
        self._ws_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="ws",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_counter_fields=counter_fields,
            manifest_enabled=config.manifest_enabled,
            manifest_extra_fields_provider=self._ws_quality.manifest_extra_fields,
            logger=self.logger,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_counter_fields=("record_type", "event"),
            manifest_enabled=config.manifest_enabled,
            logger=self.logger,
        )
        self._last_message_ns = utc_now_ns()

    def _build_qc_analyzer(self) -> RtdsBacklogAnalyzer:
        return RtdsBacklogAnalyzer(
            RtdsQcThresholds(
                freshness_spike_ms=self.config.freshness_spike_threshold_ms,
                silent_gap_ms=self.config.silent_gap_threshold_ms,
                burst_followup_gap_ms=self.config.burst_followup_gap_threshold_ms,
                burst_freshness_improvement_ms=self.config.burst_freshness_improvement_threshold_ms,
                burst_confirmation_ticks=self.config.burst_confirmation_ticks,
            )
        )

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._ws_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    def build_record(
        self,
        message: dict[str, Any],
        *,
        connection_id: str,
        local_msg_seq: int,
        recv_ts_ns: int | None = None,
    ) -> dict[str, Any] | None:
        recv_ts_ns = recv_ts_ns or utc_now_ns()
        payload = message.get("payload")
        live_price = None
        chainlink_ts_ms = None
        if isinstance(payload, dict):
            if payload.get("value") is not None:
                live_price = str(payload.get("value"))
            if payload.get("full_accuracy_value") is not None:
                if live_price is None:
                    try:
                        live_price = format(int(str(payload["full_accuracy_value"])) / 10**18, "f")
                    except Exception:
                        live_price = None
            chainlink_ts_ms = payload.get("timestamp")
        if chainlink_ts_ms is None:
            chainlink_ts_ms = message.get("timestamp")
        try:
            chainlink_ts_ms = int(chainlink_ts_ms) if chainlink_ts_ms is not None else None
        except Exception:
            chainlink_ts_ms = None
        if live_price is None or chainlink_ts_ms is None:
            return None
        return {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "chainlink_ts_ms": chainlink_ts_ms,
            "chainlink_ts_iso": ns_to_iso(chainlink_ts_ms * 1_000_000),
            "btc_usd_price": live_price,
            "source": "polymarket_rtds_chainlink",
            "recv_minus_chainlink_ms": round((recv_ts_ns - (chainlink_ts_ms * 1_000_000)) / 1_000_000, 3),
        }

    async def run(self) -> None:
        self._lifecycle_writer.write(_lifecycle("recorder_start", self.config))
        delay = self.config.reconnect_min_seconds
        try:
            while True:
                self._advance_writers()
                connection_id = generate_connection_id("poly-rtds")
                self._lifecycle_writer.write(
                    _lifecycle(
                        "websocket_connect_start",
                        self.config,
                        details={"connection_id": connection_id, "ws_url": self.config.websocket_url},
                    )
                )
                try:
                    await self._consume(connection_id)
                    delay = self.config.reconnect_min_seconds
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._lifecycle_writer.write(
                        _lifecycle(
                            "exception",
                            self.config,
                            details={"connection_id": connection_id, "error": repr(exc)},
                        )
                    )
                    self._lifecycle_writer.write(
                        _lifecycle(
                            "reconnect",
                            self.config,
                            details={
                                "connection_id": connection_id,
                                "delay_seconds": round(delay, 3),
                                "error": repr(exc),
                            },
                        )
                    )
                    await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                    delay = min(
                        self.config.reconnect_max_seconds,
                        max(self.config.reconnect_min_seconds, delay * 2),
                    )
        finally:
            self._lifecycle_writer.write(_lifecycle("shutdown", self.config))
            self._ws_writer.close()
            self._lifecycle_writer.close()

    async def _consume(self, connection_id: str) -> None:
        async with connect(
            self.config.websocket_url,
            open_timeout=self.config.connect_open_timeout_seconds,
            close_timeout=self.config.connect_close_timeout_seconds,
            max_queue=self.config.websocket_max_queue,
            ping_interval=None,
        ) as ws:
            self._last_message_ns = utc_now_ns()
            self._lifecycle_writer.write(
                _lifecycle(
                    "websocket_connect",
                    self.config,
                    details={"connection_id": connection_id, "ws_url": self.config.websocket_url},
                )
            )
            subscribe_payload = build_subscription_payload(self.config)
            await ws.send(orjson.dumps(subscribe_payload).decode("utf-8"))
            self._lifecycle_writer.write(
                _lifecycle(
                    "subscribe_sent",
                    self.config,
                    details={
                        "connection_id": connection_id,
                        "subscription": subscribe_payload,
                    },
                )
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws), name="polymarket:rtds:heartbeat")
            local_msg_seq = 0
            qc_analyzer = self._build_qc_analyzer()
            consecutive_critical_stale_records = 0
            try:
                while True:
                    message = await asyncio.wait_for(
                        ws.recv(),
                        timeout=self.config.health_timeout_seconds,
                    )
                    self._last_message_ns = utc_now_ns()
                    local_msg_seq += 1
                    recv_ts_ns = utc_now_ns()
                    self._advance_writers(recv_ts_ns)
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", errors="replace")
                    if isinstance(message, str):
                        try:
                            payload = orjson.loads(message)
                        except orjson.JSONDecodeError:
                            continue
                        if isinstance(payload, dict):
                            record = self.build_record(
                                payload,
                                connection_id=connection_id,
                                local_msg_seq=local_msg_seq,
                                recv_ts_ns=recv_ts_ns,
                            )
                            if record is not None:
                                self._ws_writer.write(record)
                                _, qc_events = qc_analyzer.observe(record)
                                source_age_ms = float(record.get("recv_minus_chainlink_ms") or 0.0)
                                if source_age_ms >= self.config.stale_reconnect_source_age_ms:
                                    consecutive_critical_stale_records += 1
                                else:
                                    consecutive_critical_stale_records = 0
                                reconnect_reason: str | None = None
                                reconnect_details: dict[str, Any] = {
                                    "connection_id": connection_id,
                                    "local_msg_seq": local_msg_seq,
                                    "recv_minus_chainlink_ms": source_age_ms,
                                    "consecutive_critical_stale_records": consecutive_critical_stale_records,
                                }
                                for qc_event in qc_events:
                                    self._ws_quality.mark(
                                        ts_ns=recv_ts_ns,
                                        quality_class="stale_context",
                                        flags=(str(qc_event["event"]),),
                                    )
                                    self._lifecycle_writer.write(
                                        _lifecycle(
                                            qc_event["event"],
                                            self.config,
                                            details={
                                                "connection_id": connection_id,
                                                "local_msg_seq": local_msg_seq,
                                                **qc_event,
                                            },
                                        )
                                    )
                                    if (
                                        str(qc_event["event"]) == "silent_gap_detected"
                                        and float(qc_event.get("recv_gap_ms") or 0.0)
                                        >= self.config.stale_reconnect_silent_gap_ms
                                    ):
                                        reconnect_reason = "silent_gap_detected"
                                        reconnect_details.update(qc_event)
                                    elif str(qc_event["event"]) == "backlog_burst_detected":
                                        reconnect_reason = "backlog_burst_detected"
                                        reconnect_details.update(qc_event)
                                if (
                                    reconnect_reason is None
                                    and consecutive_critical_stale_records
                                    >= self.config.stale_reconnect_consecutive_records
                                ):
                                    reconnect_reason = "critical_source_age"
                                if reconnect_reason is not None:
                                    self._lifecycle_writer.write(
                                        _lifecycle(
                                            "stale_stream_reconnect_requested",
                                            self.config,
                                            details={
                                                **reconnect_details,
                                                "reason": reconnect_reason,
                                                "stale_reconnect_source_age_ms": self.config.stale_reconnect_source_age_ms,
                                                "stale_reconnect_consecutive_records": self.config.stale_reconnect_consecutive_records,
                                            },
                                        )
                                    )
                                    await ws.close()
                                    return
                            continue
            finally:
                cancelled_during_cleanup = await cancel_and_await(heartbeat_task)
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError
                self._lifecycle_writer.write(
                    _lifecycle(
                        "websocket_disconnect",
                        self.config,
                        details={
                            "connection_id": connection_id,
                            "ws_url": self.config.websocket_url,
                            "close_code": ws.close_code,
                            "close_reason": ws.close_reason,
                        },
                    )
                )

    async def _heartbeat_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.config.ping_interval_seconds)
            self._advance_writers()
            await ws.send("PING")
