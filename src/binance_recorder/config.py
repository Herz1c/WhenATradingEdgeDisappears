from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RecorderConfig:
    out: Path
    symbol: str = "BTCUSDT"
    websocket_base_url: str = "wss://stream.binance.com:9443"
    rest_base_url: str = "https://api.binance.com"
    snapshot_limit: int = 5000
    websocket_time_unit: str = "MICROSECOND"
    flush_interval_seconds: float = 2.0
    server_time_poll_interval_seconds: float = 30.0
    periodic_snapshot_interval_seconds: float = 300.0
    order_book_parity_check_update_lag_tolerance: int = 20
    http_timeout_seconds: float = 15.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    websocket_max_queue: int = 10_000
    manifest_enabled: bool = True

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("binance", "spot", self.symbol)

    @property
    def streams(self) -> tuple[str, ...]:
        symbol = self.symbol.lower()
        return (
            f"{symbol}@trade",
            f"{symbol}@aggTrade",
            f"{symbol}@depth@100ms",
            f"{symbol}@bookTicker",
        )

    @classmethod
    def from_env(cls, *, out: Path) -> "RecorderConfig":
        return cls(
            out=out,
            symbol=os.getenv("BINANCE_SYMBOL", "BTCUSDT").upper(),
            websocket_base_url=os.getenv("BINANCE_WS_BASE_URL", "wss://stream.binance.com:9443"),
            rest_base_url=os.getenv("BINANCE_REST_BASE_URL", "https://api.binance.com"),
            snapshot_limit=int(os.getenv("BINANCE_SNAPSHOT_LIMIT", "5000")),
            websocket_time_unit=os.getenv("BINANCE_WS_TIME_UNIT", "MICROSECOND").upper(),
            flush_interval_seconds=float(os.getenv("BINANCE_FLUSH_INTERVAL_SECONDS", "2.0")),
            server_time_poll_interval_seconds=float(
                os.getenv("BINANCE_SERVER_TIME_POLL_INTERVAL_SECONDS", "30.0")
            ),
            periodic_snapshot_interval_seconds=float(
                os.getenv("BINANCE_PERIODIC_SNAPSHOT_INTERVAL_SECONDS", "300.0")
            ),
            order_book_parity_check_update_lag_tolerance=int(
                os.getenv("BINANCE_ORDER_BOOK_PARITY_CHECK_UPDATE_LAG_TOLERANCE", "20")
            ),
            http_timeout_seconds=float(os.getenv("BINANCE_HTTP_TIMEOUT_SECONDS", "15.0")),
            reconnect_min_seconds=float(os.getenv("BINANCE_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("BINANCE_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("BINANCE_RECONNECT_JITTER_SECONDS", "0.5")),
            websocket_max_queue=int(os.getenv("BINANCE_WS_MAX_QUEUE", "10000")),
        )
