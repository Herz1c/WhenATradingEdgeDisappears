from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HyperliquidRecorderConfig:
    out: Path
    symbol: str = "BTC"
    websocket_url: str = "wss://api.hyperliquid.xyz/ws"
    rest_base_url: str = "https://api.hyperliquid.xyz"
    current_rest_poll_interval_seconds: float = 15.0
    metrics_rest_poll_interval_seconds: float = 60.0
    rest_degraded_failure_threshold: int = 2
    rest_suspended_failure_threshold: int = 4
    rest_degraded_backoff_seconds: float = 15.0
    rest_suspended_backoff_seconds: float = 60.0
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
        return ("hyperliquid", "linear", self.symbol)

    @property
    def subscriptions(self) -> tuple[dict[str, Any], ...]:
        return (
            {"method": "subscribe", "subscription": {"type": "trades", "coin": self.symbol}},
            {"method": "subscribe", "subscription": {"type": "l2Book", "coin": self.symbol}},
            {"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": self.symbol}},
        )

    @classmethod
    def from_env(cls, *, out: Path) -> "HyperliquidRecorderConfig":
        return cls(
            out=out,
            symbol=os.getenv("HYPERLIQUID_SYMBOL", "BTC").upper(),
            websocket_url=os.getenv("HYPERLIQUID_WS_URL", "wss://api.hyperliquid.xyz/ws"),
            rest_base_url=os.getenv("HYPERLIQUID_REST_BASE_URL", "https://api.hyperliquid.xyz"),
            current_rest_poll_interval_seconds=float(
                os.getenv("HYPERLIQUID_CURRENT_REST_POLL_INTERVAL_SECONDS", "15.0")
            ),
            metrics_rest_poll_interval_seconds=float(
                os.getenv("HYPERLIQUID_METRICS_REST_POLL_INTERVAL_SECONDS", "60.0")
            ),
            rest_degraded_failure_threshold=int(
                os.getenv("HYPERLIQUID_REST_DEGRADED_FAILURE_THRESHOLD", "2")
            ),
            rest_suspended_failure_threshold=int(
                os.getenv("HYPERLIQUID_REST_SUSPENDED_FAILURE_THRESHOLD", "4")
            ),
            rest_degraded_backoff_seconds=float(
                os.getenv("HYPERLIQUID_REST_DEGRADED_BACKOFF_SECONDS", "15.0")
            ),
            rest_suspended_backoff_seconds=float(
                os.getenv("HYPERLIQUID_REST_SUSPENDED_BACKOFF_SECONDS", "60.0")
            ),
            flush_interval_seconds=float(os.getenv("HYPERLIQUID_FLUSH_INTERVAL_SECONDS", "2.0")),
            http_timeout_seconds=float(os.getenv("HYPERLIQUID_HTTP_TIMEOUT_SECONDS", "15.0")),
            reconnect_backoff_min_seconds=float(
                os.getenv("HYPERLIQUID_RECONNECT_BACKOFF_MIN_SECONDS", "1.0")
            ),
            reconnect_backoff_max_seconds=float(
                os.getenv("HYPERLIQUID_RECONNECT_BACKOFF_MAX_SECONDS", "30.0")
            ),
            reconnect_jitter_seconds=float(os.getenv("HYPERLIQUID_RECONNECT_JITTER_SECONDS", "0.5")),
            websocket_max_size_bytes=int(
                os.getenv("HYPERLIQUID_WS_MAX_SIZE_BYTES", str(8 * 1024 * 1024))
            ),
            lifecycle_heartbeat_interval_seconds=float(
                os.getenv("HYPERLIQUID_LIFECYCLE_HEARTBEAT_INTERVAL_SECONDS", "60.0")
            ),
        )
