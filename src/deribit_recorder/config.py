from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DeribitRecorderConfig:
    out: Path
    currency: str = "BTC"
    websocket_url: str = "wss://www.deribit.com/ws/api/v2"
    rest_base_url: str = "https://www.deribit.com"
    iv_poll_interval_seconds: float = 60.0
    summary_poll_interval_seconds: float = 60.0
    instruments_poll_interval_seconds: float = 300.0
    flush_interval_seconds: float = 2.0
    http_timeout_seconds: float = 15.0
    reconnect_backoff_min_seconds: float = 1.0
    reconnect_backoff_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    websocket_max_size_bytes: int = 8 * 1024 * 1024
    lifecycle_heartbeat_interval_seconds: float = 60.0

    @property
    def reconnect_min_seconds(self) -> float:
        return self.reconnect_backoff_min_seconds

    @property
    def reconnect_max_seconds(self) -> float:
        return self.reconnect_backoff_max_seconds

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("deribit", "options", self.currency)

    @property
    def ws_channels(self) -> tuple[str, ...]:
        return ("deribit_price_index.btc_usd",)

    @classmethod
    def from_env(cls, *, out: Path) -> "DeribitRecorderConfig":
        return cls(
            out=out,
            currency=os.getenv("DERIBIT_CURRENCY", "BTC").upper(),
            websocket_url=os.getenv("DERIBIT_WS_URL", "wss://www.deribit.com/ws/api/v2"),
            rest_base_url=os.getenv("DERIBIT_REST_BASE_URL", "https://www.deribit.com"),
            iv_poll_interval_seconds=float(os.getenv("DERIBIT_IV_POLL_INTERVAL_SECONDS", "60.0")),
            summary_poll_interval_seconds=float(
                os.getenv("DERIBIT_SUMMARY_POLL_INTERVAL_SECONDS", "60.0")
            ),
            instruments_poll_interval_seconds=float(
                os.getenv("DERIBIT_INSTRUMENTS_POLL_INTERVAL_SECONDS", "300.0")
            ),
            flush_interval_seconds=float(os.getenv("DERIBIT_FLUSH_INTERVAL_SECONDS", "2.0")),
            http_timeout_seconds=float(os.getenv("DERIBIT_HTTP_TIMEOUT_SECONDS", "15.0")),
            reconnect_backoff_min_seconds=float(
                os.getenv("DERIBIT_RECONNECT_BACKOFF_MIN_SECONDS", "1.0")
            ),
            reconnect_backoff_max_seconds=float(
                os.getenv("DERIBIT_RECONNECT_BACKOFF_MAX_SECONDS", "30.0")
            ),
            reconnect_jitter_seconds=float(os.getenv("DERIBIT_RECONNECT_JITTER_SECONDS", "0.5")),
            websocket_max_size_bytes=int(os.getenv("DERIBIT_WS_MAX_SIZE_BYTES", str(8 * 1024 * 1024))),
            lifecycle_heartbeat_interval_seconds=float(
                os.getenv("DERIBIT_LIFECYCLE_HEARTBEAT_INTERVAL_SECONDS", "60.0")
            ),
        )
