from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _symbol_path_component(symbol: str) -> str:
    return symbol.strip().lower().replace("/", "_")


@dataclass(slots=True)
class PolymarketRtdsConfig:
    out: Path
    websocket_url: str = "wss://ws-live-data.polymarket.com"
    topic: str = "crypto_prices_chainlink"
    subscription_type: str = "*"
    symbol: str = "btc/usd"
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    flush_interval_seconds: float = 2.0
    manifest_enabled: bool = True
    log_level: str = "INFO"
    ping_interval_seconds: float = 5.0
    health_timeout_seconds: float = 20.0
    connect_open_timeout_seconds: float = 20.0
    connect_close_timeout_seconds: float = 10.0
    websocket_max_queue: int = 50_000
    freshness_spike_threshold_ms: float = 10_000.0
    silent_gap_threshold_ms: float = 10_000.0
    burst_followup_gap_threshold_ms: float = 1_500.0
    burst_freshness_improvement_threshold_ms: float = 5_000.0
    burst_confirmation_ticks: int = 2
    stale_reconnect_source_age_ms: float = 30_000.0
    stale_reconnect_consecutive_records: int = 3
    stale_reconnect_silent_gap_ms: float = 15_000.0

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("polymarket", "rtds", self.topic, _symbol_path_component(self.symbol))

    @classmethod
    def from_env(cls, *, out: Path) -> "PolymarketRtdsConfig":
        return cls(
            out=out,
            websocket_url=os.getenv("POLYMARKET_RTDS_WS_URL", "wss://ws-live-data.polymarket.com"),
            topic=os.getenv("POLYMARKET_RTDS_TOPIC", "crypto_prices_chainlink"),
            subscription_type=os.getenv("POLYMARKET_RTDS_SUBSCRIPTION_TYPE", "*"),
            symbol=os.getenv("POLYMARKET_RTDS_SYMBOL", "btc/usd"),
            reconnect_min_seconds=float(os.getenv("POLYMARKET_RTDS_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("POLYMARKET_RTDS_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("POLYMARKET_RTDS_RECONNECT_JITTER_SECONDS", "0.5")),
            flush_interval_seconds=float(os.getenv("POLYMARKET_RTDS_FLUSH_INTERVAL_SECONDS", "2.0")),
            manifest_enabled=os.getenv("POLYMARKET_RTDS_MANIFEST_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            log_level=os.getenv("POLYMARKET_RTDS_LOG_LEVEL", "INFO"),
            ping_interval_seconds=float(os.getenv("POLYMARKET_RTDS_PING_INTERVAL_SECONDS", "5.0")),
            health_timeout_seconds=float(os.getenv("POLYMARKET_RTDS_HEALTH_TIMEOUT_SECONDS", "20.0")),
            connect_open_timeout_seconds=float(
                os.getenv("POLYMARKET_RTDS_CONNECT_OPEN_TIMEOUT_SECONDS", "20.0")
            ),
            connect_close_timeout_seconds=float(
                os.getenv("POLYMARKET_RTDS_CONNECT_CLOSE_TIMEOUT_SECONDS", "10.0")
            ),
            websocket_max_queue=int(os.getenv("POLYMARKET_RTDS_WS_MAX_QUEUE", "50000")),
            freshness_spike_threshold_ms=float(
                os.getenv("POLYMARKET_RTDS_FRESHNESS_SPIKE_THRESHOLD_MS", "10000.0")
            ),
            silent_gap_threshold_ms=float(
                os.getenv("POLYMARKET_RTDS_SILENT_GAP_THRESHOLD_MS", "10000.0")
            ),
            burst_followup_gap_threshold_ms=float(
                os.getenv("POLYMARKET_RTDS_BURST_FOLLOWUP_GAP_THRESHOLD_MS", "1500.0")
            ),
            burst_freshness_improvement_threshold_ms=float(
                os.getenv("POLYMARKET_RTDS_BURST_FRESHNESS_IMPROVEMENT_THRESHOLD_MS", "5000.0")
            ),
            burst_confirmation_ticks=int(
                os.getenv("POLYMARKET_RTDS_BURST_CONFIRMATION_TICKS", "2")
            ),
            stale_reconnect_source_age_ms=float(
                os.getenv("POLYMARKET_RTDS_STALE_RECONNECT_SOURCE_AGE_MS", "30000.0")
            ),
            stale_reconnect_consecutive_records=int(
                os.getenv("POLYMARKET_RTDS_STALE_RECONNECT_CONSECUTIVE_RECORDS", "3")
            ),
            stale_reconnect_silent_gap_ms=float(
                os.getenv("POLYMARKET_RTDS_STALE_RECONNECT_SILENT_GAP_MS", "15000.0")
            ),
        )
