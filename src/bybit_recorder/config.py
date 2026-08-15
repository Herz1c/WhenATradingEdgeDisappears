from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BybitRecorderConfig:
    out: Path
    symbol: str = "BTCUSDT"
    websocket_url: str = "wss://stream.bybit.com/v5/public/linear"
    rest_base_url: str = "https://api.bybit.com"
    flush_interval_seconds: float = 2.0
    current_rest_poll_interval_seconds: float = 15.0
    metrics_rest_poll_interval_seconds: float = 60.0
    open_interest_interval: str = "5min"
    http_timeout_seconds: float = 15.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    websocket_max_queue: int = 10_000
    manifest_enabled: bool = True
    lifecycle_heartbeat_interval_seconds: float = 60.0

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("bybit", "linear", self.symbol)

    @property
    def topics(self) -> tuple[str, ...]:
        return (
            f"orderbook.50.{self.symbol}",
            f"publicTrade.{self.symbol}",
            f"tickers.{self.symbol}",
            f"allLiquidation.{self.symbol}",
        )

    @classmethod
    def from_env(cls, *, out: Path) -> "BybitRecorderConfig":
        return cls(
            out=out,
            symbol=os.getenv("BYBIT_SYMBOL", "BTCUSDT").upper(),
            websocket_url=os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear"),
            rest_base_url=os.getenv("BYBIT_REST_BASE_URL", "https://api.bybit.com"),
            flush_interval_seconds=float(os.getenv("BYBIT_FLUSH_INTERVAL_SECONDS", "2.0")),
            current_rest_poll_interval_seconds=float(
                os.getenv("BYBIT_CURRENT_REST_POLL_INTERVAL_SECONDS", "15.0")
            ),
            metrics_rest_poll_interval_seconds=float(
                os.getenv("BYBIT_METRICS_REST_POLL_INTERVAL_SECONDS", "60.0")
            ),
            open_interest_interval=os.getenv("BYBIT_OPEN_INTEREST_INTERVAL", "5min"),
            http_timeout_seconds=float(os.getenv("BYBIT_HTTP_TIMEOUT_SECONDS", "15.0")),
            reconnect_min_seconds=float(os.getenv("BYBIT_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("BYBIT_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("BYBIT_RECONNECT_JITTER_SECONDS", "0.5")),
            websocket_max_queue=int(os.getenv("BYBIT_WS_MAX_QUEUE", "10000")),
            lifecycle_heartbeat_interval_seconds=float(
                os.getenv("BYBIT_LIFECYCLE_HEARTBEAT_INTERVAL_SECONDS", "60.0")
            ),
        )
