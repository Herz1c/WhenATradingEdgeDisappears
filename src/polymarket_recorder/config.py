from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PolymarketRecorderConfig:
    out: Path
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    data_api_base_url: str = "https://data-api.polymarket.com"
    market_page_base_url: str = "https://polymarket.com/event"
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    gamma_page_limit: int = 200
    gamma_max_pages: int = 15
    gamma_poll_interval_seconds: float = 2.0
    clob_quote_poll_interval_seconds: float = 1.0
    clob_batch_quote_poll_interval_seconds: float = 1.0
    history_poll_interval_seconds: float = 5.0
    trade_history_poll_interval_seconds: float = 1.0
    metadata_poll_interval_seconds: float = 10.0
    metrics_poll_interval_seconds: float = 15.0
    rewards_poll_interval_seconds: float = 60.0
    event_detail_poll_interval_seconds: float = 30.0
    strike_poll_interval_seconds: float = 5.0
    strike_mode: str = "first_success_or_on_change"
    strike_debug_include_html: bool = False
    market_slug_prefix: str = "btc-updown-5m"
    market_interval_seconds: int = 300
    market_active_preopen_seconds: int = 150
    market_discovery_lookback_intervals: int = 2
    market_discovery_lookahead_intervals: int = 24
    market_stale_grace_seconds: float = 15.0
    market_boundary_check_interval_seconds: float = 0.25
    market_prewindow_resync_ttc_seconds: float = 90.0
    market_prewindow_resync_width_seconds: float = 20.0
    # Keep a closed market's writers open this long so book events still in
    # flight (or briefly queued in-process) at close time land in its files
    # instead of being dropped -- the decision-window tail must be replayable.
    market_close_linger_seconds: float = 30.0
    # In-process ingest queue between the ws reader task and the record writer.
    # If depth ever exceeds the hard limit the writer cannot keep up at all;
    # we drop the backlog and hard-reconnect for a fresh book (fresh > complete).
    ingest_queue_hard_limit: int = 200_000
    ingest_stats_interval_seconds: float = 10.0
    resolution_closed_lookback_intervals: int = 288
    resolution_watch_preclose_seconds: int = 150
    resolution_rest_poll_interval_seconds: float = 30.0
    resolution_rest_resolved_grace_seconds: float = 120.0
    heartbeat_interval_seconds: float = 5.0
    heartbeat_health_timeout_seconds: float = 20.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    switch_grace_seconds: float = 1.0
    connect_open_timeout_seconds: float = 20.0
    connect_close_timeout_seconds: float = 10.0
    websocket_max_queue: int = 50_000
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 2
    http_retry_backoff_seconds: float = 0.5
    http_max_connections: int = 20
    http_max_keepalive_connections: int = 5
    http_keepalive_expiry_seconds: float = 5.0
    http_trust_env: bool = False
    http_degraded_failure_threshold: int = 2
    http_suspended_failure_threshold: int = 4
    http_degraded_backoff_seconds: float = 5.0
    http_suspended_backoff_seconds: float = 30.0
    flush_interval_seconds: float = 2.0
    rest_loop_sleep_seconds: float = 0.25
    manifest_enabled: bool = True
    enable_token_ws_split: bool = False
    enable_token_rest_split: bool = False
    api_keys_path: Path = Path("API_Keys")

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("polymarket", "btc_updown_5m")

    @classmethod
    def from_env(cls, *, out: Path) -> "PolymarketRecorderConfig":
        return cls(
            out=out,
            gamma_base_url=os.getenv("POLYMARKET_GAMMA_BASE_URL", "https://gamma-api.polymarket.com"),
            clob_base_url=os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"),
            data_api_base_url=os.getenv("POLYMARKET_DATA_API_BASE_URL", "https://data-api.polymarket.com"),
            market_page_base_url=os.getenv("POLYMARKET_MARKET_PAGE_BASE_URL", "https://polymarket.com/event"),
            websocket_url=os.getenv(
                "POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
            ),
            gamma_page_limit=int(os.getenv("POLYMARKET_GAMMA_PAGE_LIMIT", "200")),
            gamma_max_pages=int(os.getenv("POLYMARKET_GAMMA_MAX_PAGES", "15")),
            gamma_poll_interval_seconds=float(os.getenv("POLYMARKET_GAMMA_POLL_INTERVAL_SECONDS", "2.0")),
            clob_quote_poll_interval_seconds=float(
                os.getenv("POLYMARKET_CLOB_QUOTE_POLL_INTERVAL_SECONDS", "1.0")
            ),
            clob_batch_quote_poll_interval_seconds=float(
                os.getenv("POLYMARKET_CLOB_BATCH_QUOTE_POLL_INTERVAL_SECONDS", "1.0")
            ),
            history_poll_interval_seconds=float(
                os.getenv("POLYMARKET_HISTORY_POLL_INTERVAL_SECONDS", "5.0")
            ),
            trade_history_poll_interval_seconds=float(
                os.getenv("POLYMARKET_TRADE_HISTORY_POLL_INTERVAL_SECONDS", "1.0")
            ),
            metadata_poll_interval_seconds=float(
                os.getenv("POLYMARKET_METADATA_POLL_INTERVAL_SECONDS", "10.0")
            ),
            metrics_poll_interval_seconds=float(
                os.getenv("POLYMARKET_METRICS_POLL_INTERVAL_SECONDS", "15.0")
            ),
            rewards_poll_interval_seconds=float(
                os.getenv("POLYMARKET_REWARDS_POLL_INTERVAL_SECONDS", "60.0")
            ),
            event_detail_poll_interval_seconds=float(
                os.getenv("POLYMARKET_EVENT_DETAIL_POLL_INTERVAL_SECONDS", "30.0")
            ),
            strike_poll_interval_seconds=float(
                os.getenv("POLYMARKET_STRIKE_POLL_INTERVAL_SECONDS", "5.0")
            ),
            strike_mode=os.getenv("POLYMARKET_STRIKE_MODE", "first_success_or_on_change").strip().lower(),
            strike_debug_include_html=os.getenv("POLYMARKET_STRIKE_DEBUG_INCLUDE_HTML", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            market_slug_prefix=os.getenv("POLYMARKET_MARKET_SLUG_PREFIX", "btc-updown-5m"),
            market_interval_seconds=int(os.getenv("POLYMARKET_MARKET_INTERVAL_SECONDS", "300")),
            market_active_preopen_seconds=int(os.getenv("POLYMARKET_MARKET_ACTIVE_PREOPEN_SECONDS", "150")),
            market_discovery_lookback_intervals=int(
                os.getenv("POLYMARKET_DISCOVERY_LOOKBACK_INTERVALS", "2")
            ),
            market_discovery_lookahead_intervals=int(
                os.getenv("POLYMARKET_DISCOVERY_LOOKAHEAD_INTERVALS", "24")
            ),
            market_stale_grace_seconds=float(os.getenv("POLYMARKET_MARKET_STALE_GRACE_SECONDS", "15.0")),
            market_boundary_check_interval_seconds=float(
                os.getenv("POLYMARKET_MARKET_BOUNDARY_CHECK_INTERVAL_SECONDS", "0.25")
            ),
            market_prewindow_resync_ttc_seconds=float(
                os.getenv("POLYMARKET_PREWINDOW_RESYNC_TTC_SECONDS", "90.0")
            ),
            market_prewindow_resync_width_seconds=float(
                os.getenv("POLYMARKET_PREWINDOW_RESYNC_WIDTH_SECONDS", "20.0")
            ),
            market_close_linger_seconds=float(
                os.getenv("POLYMARKET_MARKET_CLOSE_LINGER_SECONDS", "30.0")
            ),
            ingest_queue_hard_limit=int(
                os.getenv("POLYMARKET_INGEST_QUEUE_HARD_LIMIT", "200000")
            ),
            ingest_stats_interval_seconds=float(
                os.getenv("POLYMARKET_INGEST_STATS_INTERVAL_SECONDS", "10.0")
            ),
            resolution_closed_lookback_intervals=int(
                os.getenv("POLYMARKET_RESOLUTION_CLOSED_LOOKBACK_INTERVALS", "288")
            ),
            resolution_watch_preclose_seconds=int(
                os.getenv("POLYMARKET_RESOLUTION_WATCH_PRECLOSE_SECONDS", "150")
            ),
            resolution_rest_poll_interval_seconds=float(
                os.getenv("POLYMARKET_RESOLUTION_REST_POLL_INTERVAL_SECONDS", "30.0")
            ),
            resolution_rest_resolved_grace_seconds=float(
                os.getenv("POLYMARKET_RESOLUTION_REST_RESOLVED_GRACE_SECONDS", "120.0")
            ),
            heartbeat_interval_seconds=float(os.getenv("POLYMARKET_HEARTBEAT_INTERVAL_SECONDS", "5.0")),
            heartbeat_health_timeout_seconds=float(
                os.getenv("POLYMARKET_HEARTBEAT_HEALTH_TIMEOUT_SECONDS", "20.0")
            ),
            reconnect_min_seconds=float(os.getenv("POLYMARKET_RECONNECT_MIN_SECONDS", "1.0")),
            reconnect_max_seconds=float(os.getenv("POLYMARKET_RECONNECT_MAX_SECONDS", "30.0")),
            reconnect_jitter_seconds=float(os.getenv("POLYMARKET_RECONNECT_JITTER_SECONDS", "0.5")),
            switch_grace_seconds=float(os.getenv("POLYMARKET_SWITCH_GRACE_SECONDS", "1.0")),
            connect_open_timeout_seconds=float(
                os.getenv("POLYMARKET_CONNECT_OPEN_TIMEOUT_SECONDS", "20.0")
            ),
            connect_close_timeout_seconds=float(
                os.getenv("POLYMARKET_CONNECT_CLOSE_TIMEOUT_SECONDS", "10.0")
            ),
            websocket_max_queue=int(os.getenv("POLYMARKET_WS_MAX_QUEUE", "50000")),
            http_timeout_seconds=float(os.getenv("POLYMARKET_HTTP_TIMEOUT_SECONDS", "15.0")),
            http_max_retries=int(os.getenv("POLYMARKET_HTTP_MAX_RETRIES", "2")),
            http_retry_backoff_seconds=float(
                os.getenv("POLYMARKET_HTTP_RETRY_BACKOFF_SECONDS", "0.5")
            ),
            http_max_connections=int(os.getenv("POLYMARKET_HTTP_MAX_CONNECTIONS", "20")),
            http_max_keepalive_connections=int(
                os.getenv("POLYMARKET_HTTP_MAX_KEEPALIVE_CONNECTIONS", "5")
            ),
            http_keepalive_expiry_seconds=float(
                os.getenv("POLYMARKET_HTTP_KEEPALIVE_EXPIRY_SECONDS", "5.0")
            ),
            http_trust_env=os.getenv("POLYMARKET_HTTP_TRUST_ENV", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            http_degraded_failure_threshold=int(
                os.getenv("POLYMARKET_HTTP_DEGRADED_FAILURE_THRESHOLD", "2")
            ),
            http_suspended_failure_threshold=int(
                os.getenv("POLYMARKET_HTTP_SUSPENDED_FAILURE_THRESHOLD", "4")
            ),
            http_degraded_backoff_seconds=float(
                os.getenv("POLYMARKET_HTTP_DEGRADED_BACKOFF_SECONDS", "5.0")
            ),
            http_suspended_backoff_seconds=float(
                os.getenv("POLYMARKET_HTTP_SUSPENDED_BACKOFF_SECONDS", "30.0")
            ),
            flush_interval_seconds=float(os.getenv("POLYMARKET_FLUSH_INTERVAL_SECONDS", "2.0")),
            rest_loop_sleep_seconds=float(os.getenv("POLYMARKET_REST_LOOP_SLEEP_SECONDS", "0.25")),
            manifest_enabled=os.getenv("POLYMARKET_MANIFEST_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            enable_token_ws_split=os.getenv("POLYMARKET_ENABLE_TOKEN_WS_SPLIT", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            enable_token_rest_split=os.getenv("POLYMARKET_ENABLE_TOKEN_REST_SPLIT", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            api_keys_path=Path(os.getenv("POLYMARKET_API_KEYS_PATH", "API_Keys")),
        )
