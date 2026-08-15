from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CoinbaseAdvancedRecorderConfig:
    out: Path
    product_id: str = "BTC-USD"
    websocket_url: str = "wss://advanced-trade-ws.coinbase.com"
    channels: tuple[str, ...] = ("heartbeats", "market_trades", "ticker", "level2")
    flush_interval_seconds: float = 2.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    ping_interval_seconds: float = 20.0
    ping_timeout_seconds: float = 20.0
    heartbeat_stale_seconds: float = 5.0
    max_message_size_bytes: int = 8 * 1024 * 1024
    manifest_enabled: bool = True

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("coinbase", "advanced", self.product_id)

    @classmethod
    def from_env(cls, *, out: Path) -> "CoinbaseAdvancedRecorderConfig":
        return cls(
            out=out,
            product_id=os.getenv("COINBASE_PRODUCT_ID", "BTC-USD").upper(),
            websocket_url=os.getenv("COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com"),
            flush_interval_seconds=float(os.getenv("COINBASE_FLUSH_INTERVAL_SECONDS", "2.0")),
            reconnect_min_seconds=float(os.getenv("COINBASE_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("COINBASE_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("COINBASE_RECONNECT_JITTER_SECONDS", "0.5")),
            ping_interval_seconds=float(os.getenv("COINBASE_PING_INTERVAL_SECONDS", "20.0")),
            ping_timeout_seconds=float(os.getenv("COINBASE_PING_TIMEOUT_SECONDS", "20.0")),
            heartbeat_stale_seconds=float(os.getenv("COINBASE_HEARTBEAT_STALE_SECONDS", "5.0")),
            max_message_size_bytes=int(os.getenv("COINBASE_WS_MAX_SIZE_BYTES", str(8 * 1024 * 1024))),
        )
