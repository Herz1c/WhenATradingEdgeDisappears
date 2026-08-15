from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from binance_recorder.utils import ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.http_utils import build_async_http_client, request_with_retries
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_epoch_timestamp_fields, recv_timestamp_source, utc_now_ns


@dataclass(slots=True)
class FearAndGreedRecorderConfig:
    out: Path
    rest_url: str = "https://api.alternative.me/fng/?limit=1&format=json"
    poll_interval_seconds: float = 300.0
    flush_interval_seconds: float = 60.0
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 2
    http_retry_backoff_seconds: float = 0.5
    http_max_connections: int = 10
    http_max_keepalive_connections: int = 2
    http_keepalive_expiry_seconds: float = 5.0
    http_trust_env: bool = False
    change_only: bool = True

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("fng",)

    @classmethod
    def from_env(cls, *, out: Path) -> "FearAndGreedRecorderConfig":
        return cls(
            out=out,
            rest_url=os.getenv("FNG_REST_URL", "https://api.alternative.me/fng/?limit=1&format=json"),
            poll_interval_seconds=float(os.getenv("FNG_POLL_INTERVAL_SECONDS", "300.0")),
            flush_interval_seconds=float(os.getenv("FNG_FLUSH_INTERVAL_SECONDS", "60.0")),
            http_timeout_seconds=float(os.getenv("FNG_HTTP_TIMEOUT_SECONDS", "15.0")),
            http_max_retries=int(os.getenv("FNG_HTTP_MAX_RETRIES", "2")),
            http_retry_backoff_seconds=float(os.getenv("FNG_HTTP_RETRY_BACKOFF_SECONDS", "0.5")),
            http_max_connections=int(os.getenv("FNG_HTTP_MAX_CONNECTIONS", "10")),
            http_max_keepalive_connections=int(os.getenv("FNG_HTTP_MAX_KEEPALIVE_CONNECTIONS", "2")),
            http_keepalive_expiry_seconds=float(os.getenv("FNG_HTTP_KEEPALIVE_EXPIRY_SECONDS", "5.0")),
            http_trust_env=os.getenv("FNG_HTTP_TRUST_ENV", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            change_only=os.getenv("FNG_CHANGE_ONLY", "true").strip().lower() not in {"0", "false", "no", "off"},
        )


def build_fng_record(payload: dict[str, Any], *, recv_ts_ns: int | None = None) -> dict[str, Any]:
    recv_ts_ns = recv_ts_ns or utc_now_ns()
    first = payload.get("data", [None])[0] if isinstance(payload.get("data"), list) else None
    timestamp_value = None
    if isinstance(first, dict):
        timestamp_value = first.get("timestamp")
    source_timestamps = build_epoch_timestamp_fields("timestamp", timestamp_value, unit_hint="second")
    extracted = {}
    if isinstance(first, dict):
        try:
            extracted["value"] = int(first["value"])
        except Exception:
            pass
        if first.get("value_classification") is not None:
            extracted["value_classification"] = str(first["value_classification"])
    return {
        "record_type": "rest_observation",
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
        "source": "alternative_me",
        "endpoint": "fear_and_greed",
        "url": "https://api.alternative.me/fng/?limit=1&format=json",
        "http_status": 200,
        "payload": payload,
        "source_timestamps": source_timestamps,
        "extracted": extracted,
    }


def fng_change_signature(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    source_timestamps = record.get("source_timestamps", {}) if isinstance(record.get("source_timestamps"), dict) else {}
    return (
        extracted.get("value"),
        extracted.get("value_classification"),
        source_timestamps.get("timestamp_s"),
    )


class FearAndGreedRecorderService:
    def __init__(self, config: FearAndGreedRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="fng", config=config)
        self._writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="fng",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=self._runtime.manifest_fields(
                {"source": "alternative_me", "endpoint": "fear_and_greed"}
            ),
        )
        self._last_signature: tuple[Any, Any, Any] | None = None

    async def run(self) -> None:
        async with build_async_http_client(
            timeout_seconds=self.config.http_timeout_seconds,
            max_connections=self.config.http_max_connections,
            max_keepalive_connections=self.config.http_max_keepalive_connections,
            keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
            trust_env=self.config.http_trust_env,
        ) as client:
            try:
                while True:
                    try:
                        await self._poll_once(client)
                    except asyncio.CancelledError:
                        raise
                    except (httpx.HTTPError, ValueError) as exc:
                        self.logger.warning(
                            "Fear and Greed poll failed",
                            extra={
                                "service_name": "fng",
                                "source": "alternative_me",
                                "mode": "public_rest_polling",
                                "rest_url": self.config.rest_url,
                                "error": repr(exc),
                            },
                        )
                    await asyncio.sleep(self.config.poll_interval_seconds)
            finally:
                self._writer.close()

    async def _poll_once(self, client: httpx.AsyncClient) -> None:
        recv_ts_ns = utc_now_ns()
        self._writer.advance_time(now_ns=recv_ts_ns)
        outcome = await request_with_retries(
            client,
            "GET",
            self.config.rest_url,
            max_retries=self.config.http_max_retries,
            retry_backoff_seconds=self.config.http_retry_backoff_seconds,
        )
        response = outcome.response
        payload = response.json()
        record = build_fng_record(payload if isinstance(payload, dict) else {"data": payload}, recv_ts_ns=recv_ts_ns)
        record["url"] = str(response.request.url)
        record["http_status"] = response.status_code
        record["request_attempt_count"] = outcome.attempt_count
        signature = fng_change_signature(record)
        if self.config.change_only and self._last_signature == signature:
            return
        self._writer.write(record)
        self._last_signature = signature
