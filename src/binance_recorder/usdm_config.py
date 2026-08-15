from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BinanceUsdmRecorderConfig:
    out: Path
    symbol: str = "BTCUSDT"
    websocket_base_url: str = "wss://fstream.binance.com"
    rest_base_url: str = "https://fapi.binance.com"
    flush_interval_seconds: float = 2.0
    current_rest_poll_interval_seconds: float = 15.0
    metrics_rest_poll_interval_seconds: float = 60.0
    depth_snapshot_limit: int = 1000
    order_book_parity_check_update_lag_tolerance: int = 20
    history_period: str = "5m"
    history_limit: int = 30
    http_timeout_seconds: float = 15.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    websocket_max_queue: int = 10_000
    manifest_enabled: bool = True

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("binance", "usdm", self.symbol)

    @property
    def streams(self) -> tuple[str, ...]:
        symbol = self.symbol.lower()
        return (
            f"{symbol}@trade",
            f"{symbol}@aggTrade",
            f"{symbol}@depth@100ms",
            f"{symbol}@bookTicker",
            f"{symbol}@markPrice@1s",
            "!forceOrder@arr",
        )

    @classmethod
    def from_env(cls, *, out: Path) -> "BinanceUsdmRecorderConfig":
        return cls(
            out=out,
            symbol=os.getenv("BINANCE_USDM_SYMBOL", "BTCUSDT").upper(),
            websocket_base_url=os.getenv("BINANCE_USDM_WS_BASE_URL", "wss://fstream.binance.com"),
            rest_base_url=os.getenv("BINANCE_USDM_REST_BASE_URL", "https://fapi.binance.com"),
            flush_interval_seconds=float(os.getenv("BINANCE_USDM_FLUSH_INTERVAL_SECONDS", "2.0")),
            current_rest_poll_interval_seconds=float(
                os.getenv("BINANCE_USDM_CURRENT_REST_POLL_INTERVAL_SECONDS", "15.0")
            ),
            metrics_rest_poll_interval_seconds=float(
                os.getenv("BINANCE_USDM_METRICS_REST_POLL_INTERVAL_SECONDS", "60.0")
            ),
            depth_snapshot_limit=int(os.getenv("BINANCE_USDM_DEPTH_SNAPSHOT_LIMIT", "1000")),
            order_book_parity_check_update_lag_tolerance=int(
                os.getenv("BINANCE_USDM_ORDER_BOOK_PARITY_CHECK_UPDATE_LAG_TOLERANCE", "20")
            ),
            history_period=os.getenv("BINANCE_USDM_HISTORY_PERIOD", "5m"),
            history_limit=int(os.getenv("BINANCE_USDM_HISTORY_LIMIT", "30")),
            http_timeout_seconds=float(os.getenv("BINANCE_USDM_HTTP_TIMEOUT_SECONDS", "15.0")),
            reconnect_min_seconds=float(os.getenv("BINANCE_USDM_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("BINANCE_USDM_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("BINANCE_USDM_RECONNECT_JITTER_SECONDS", "0.5")),
            websocket_max_queue=int(os.getenv("BINANCE_USDM_WS_MAX_QUEUE", "10000")),
        )
