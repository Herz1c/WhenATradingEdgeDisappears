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
from market_recorders.time_utils import build_capture_timing, recv_timestamp_source, utc_now_ns

from .config import ChainlinkRecorderConfig
from .datastreams_auth import build_auth_headers, datastreams_websocket_url
from .live_reference import build_direct_live_record


class DatastreamsWsClient:
    def __init__(
        self,
        config: ChainlinkRecorderConfig,
        *,
        data_writer: HourlyRotatingJsonlWriter,
        lifecycle_writer: HourlyRotatingJsonlWriter,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.data_writer = data_writer
        self.lifecycle_writer = lifecycle_writer
        self.logger = logger or logging.getLogger(__name__)

    def _feed_ids(self) -> list[str]:
        feed_id = self.config.datastreams_feed_id_btcusd
        return [str(feed_id)] if feed_id else []

    def _ws_endpoint(self) -> tuple[str, str]:
        return datastreams_websocket_url(
            self.config.datastreams_ws_url,
            feed_ids=self._feed_ids(),
        )

    def _connect_url_and_headers(self) -> tuple[str, str, dict[str, str]]:
        ws_url, full_path = self._ws_endpoint()
        headers = build_auth_headers(
            api_key=self.config.datastreams_api_key,
            api_secret=self.config.datastreams_api_secret,
            method="GET",
            full_path=full_path,
        )
        if not headers:
            raise RuntimeError(
                self.config.availability_message()
                or "Direct Chainlink Data Streams WebSocket auth headers could not be built"
            )
        return ws_url, full_path, headers

    @staticmethod
    def _status_code_from_exception(exc: BaseException) -> int | None:
        for candidate in (
            getattr(exc, "status_code", None),
            getattr(getattr(exc, "response", None), "status_code", None),
            getattr(getattr(exc, "response", None), "status", None),
        ):
            if isinstance(candidate, int):
                return candidate
        return None

    @classmethod
    def classify_exception_event(cls, exc: BaseException) -> str:
        status_code = cls._status_code_from_exception(exc)
        if status_code in {401, 403}:
            return "auth_error"
        return "websocket_error"

    def _lifecycle(
        self,
        event: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recv_ts_ns = utc_now_ns()
        record = {
            "record_type": "lifecycle_event",
            "event": event,
            "source": "chainlink",
            "component": "live_reference",
            "service": "datastreams_ws",
            "backend": "datastreams_ws",
            "symbol": self.config.symbol,
            "feed_id": self.config.datastreams_feed_id_btcusd,
            "live_source_type": "chainlink_datastreams_ws",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
        }
        if details:
            record.update(details)
        return {key: value for key, value in record.items() if value is not None}

    def _write_lifecycle(self, event: str, *, details: dict[str, Any] | None = None) -> None:
        self.lifecycle_writer.write(self._lifecycle(event, details=details))

    def build_record(
        self,
        payload: dict[str, Any],
        *,
        connection_id: str = "chainlink-ds-test",
        local_msg_seq: int = 0,
        recv_ts_ns: int | None = None,
    ) -> dict[str, Any] | None:
        recv_ts_ns = recv_ts_ns or utc_now_ns()
        return build_direct_live_record(
            payload,
            backend="datastreams_ws",
            symbol=self.config.symbol,
            recv_ts_ns=recv_ts_ns,
            recv_ts_iso=ns_to_iso(recv_ts_ns),
            recv_ts_source=recv_timestamp_source(),
            endpoint_url=self._ws_endpoint()[0],
            capture_timing=build_capture_timing(
                request_start_ns=recv_ts_ns,
                response_end_ns=recv_ts_ns,
            ),
            connection_id=connection_id,
            local_msg_seq=local_msg_seq,
        )

    async def run(self) -> None:
        delay = self.config.reconnect_min_seconds
        self._write_lifecycle(
            "service_start",
            details={
                "mode": "signed_ws_stream",
                "endpoint_url": self.config.datastreams_ws_url,
                "feed_ids": self._feed_ids(),
            },
        )
        try:
            while True:
                connection_id = generate_connection_id("chainlink-ds")
                endpoint_url, full_path = self._ws_endpoint()
                self._write_lifecycle(
                    "websocket_connecting",
                    details={
                        "connection_id": connection_id,
                        "endpoint_url": endpoint_url,
                        "ws_path": full_path,
                    },
                )
                try:
                    await self._consume(connection_id)
                    delay = self.config.reconnect_min_seconds
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    status_code = self._status_code_from_exception(exc)
                    event = self.classify_exception_event(exc)
                    self._write_lifecycle(
                        event,
                        details={
                            "connection_id": connection_id,
                            "endpoint_url": endpoint_url,
                            "ws_path": full_path,
                            "status_code": status_code,
                            "error": repr(exc),
                        },
                    )
                    self._write_lifecycle(
                        "reconnect",
                        details={
                            "connection_id": connection_id,
                            "endpoint_url": endpoint_url,
                            "ws_path": full_path,
                            "status_code": status_code,
                            "delay_seconds": round(delay, 3),
                            "error": repr(exc),
                        },
                    )
                    await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                    delay = min(
                        self.config.reconnect_max_seconds,
                        max(self.config.reconnect_min_seconds, delay * 2),
                    )
        finally:
            self._write_lifecycle(
                "shutdown",
                details={"mode": "signed_ws_stream", "endpoint_url": self.config.datastreams_ws_url},
            )

    async def _consume(self, connection_id: str) -> None:
        ws_url, full_path, headers = self._connect_url_and_headers()
        async with connect(
            ws_url,
            extra_headers=headers,
            ping_interval=None,
            open_timeout=self.config.http_timeout_seconds,
            close_timeout=self.config.http_timeout_seconds,
        ) as ws:
            self._write_lifecycle(
                "websocket_connected",
                details={
                    "connection_id": connection_id,
                    "endpoint_url": ws_url,
                    "ws_path": full_path,
                },
            )
            self._write_lifecycle(
                "start_stream",
                details={
                    "connection_id": connection_id,
                    "endpoint_url": ws_url,
                    "ws_path": full_path,
                    "feed_ids": self._feed_ids(),
                },
            )
            heartbeat = asyncio.create_task(self._ping_loop(ws), name="chainlink:datastreams:ping")
            local_msg_seq = 0
            try:
                async for message in ws:
                    recv_ts_ns = utc_now_ns()
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", errors="replace")
                    try:
                        payload = orjson.loads(message)
                    except orjson.JSONDecodeError as exc:
                        self._write_lifecycle(
                            "decode_error",
                            details={
                                "connection_id": connection_id,
                                "raw_msg_seq": local_msg_seq + 1,
                                "error": repr(exc),
                                "raw_message_preview": message[:500],
                            },
                        )
                        continue
                    if not isinstance(payload, dict):
                        self._write_lifecycle(
                            "decode_error",
                            details={
                                "connection_id": connection_id,
                                "raw_msg_seq": local_msg_seq + 1,
                                "error": "non_dict_payload",
                                "payload_type": type(payload).__name__,
                            },
                        )
                        continue
                    local_msg_seq += 1
                    try:
                        record = self.build_record(
                            payload,
                            connection_id=connection_id,
                            local_msg_seq=local_msg_seq,
                            recv_ts_ns=recv_ts_ns,
                        )
                    except Exception as exc:
                        self._write_lifecycle(
                            "decode_error",
                            details={
                                "connection_id": connection_id,
                                "raw_msg_seq": local_msg_seq,
                                "error": repr(exc),
                            },
                        )
                        continue
                    if record is None:
                        continue
                    self.data_writer.write(record)
            finally:
                cancelled_during_cleanup = await cancel_and_await(heartbeat)
                self._write_lifecycle(
                    "websocket_disconnect",
                    details={
                        "connection_id": connection_id,
                        "endpoint_url": ws_url,
                        "ws_path": full_path,
                        "close_code": getattr(ws, "close_code", None),
                        "close_reason": getattr(ws, "close_reason", None),
                    },
                )
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _ping_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(20.0)
            pong = await ws.ping()
            await pong
