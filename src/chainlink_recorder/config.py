from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from market_recorders.keyfile import load_key_value_file


@dataclass(slots=True)
class ChainlinkRecorderConfig:
    out: Path
    symbol: str = "BTCUSD"
    backend: str = "auto"
    page_url: str = "https://data.chain.link/streams/btc-usd-cexprice-streams"
    poll_interval_seconds: float = 2.0
    http_timeout_seconds: float = 15.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    flush_interval_seconds: float = 2.0
    user_agent: str = "Mozilla/5.0 (compatible; market-recorders/0.1)"
    store_raw_html: bool = False
    datastreams_rest_url: str = "https://api.dataengine.chain.link"
    datastreams_ws_url: str = "wss://ws.dataengine.chain.link"
    datastreams_api_key: str | None = None
    datastreams_api_secret: str | None = None
    datastreams_feed_id_btcusd: str | None = None
    live_reports_query: str = "LIVE_STREAM_REPORTS_QUERY"

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("chainlink", self.selected_backend or "unconfigured", self.symbol)

    def missing_datastreams_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.datastreams_api_key:
            missing.append("datastreams_api_key")
        if not self.datastreams_api_secret:
            missing.append("datastreams_api_secret")
        if not self.datastreams_feed_id_btcusd:
            missing.append("datastreams_feed_id_btcusd")
        return tuple(missing)

    def direct_datastreams_requested(self) -> bool:
        return any(
            (
                self.datastreams_api_key,
                self.datastreams_api_secret,
                self.datastreams_feed_id_btcusd,
            )
        )

    def direct_datastreams_ready(self) -> bool:
        return not self.missing_datastreams_fields()

    @property
    def selected_backend(self) -> str | None:
        if self.backend in {"public_page", "public_stream_page"}:
            return None
        if self.backend != "auto":
            return self.backend
        if self.direct_datastreams_ready():
            return "datastreams_ws"
        return None

    def availability_issue(self) -> str | None:
        if self.backend in {"public_page", "public_stream_page"}:
            return "public_delayed_moved_to_separate_service"
        if self.backend == "auto":
            if not self.direct_datastreams_requested():
                return "direct_datastreams_not_configured"
            missing = self.missing_datastreams_fields()
            if missing:
                return ",".join(f"missing_{field}" for field in missing)
            return None
        if self.selected_backend not in {"datastreams_api", "datastreams_ws"}:
            return None
        missing = self.missing_datastreams_fields()
        if missing:
            return ",".join(f"missing_{field}" for field in missing)
        return None

    def availability_message(self) -> str | None:
        issue = self.availability_issue()
        if issue is None:
            return None
        if issue == "direct_datastreams_not_configured":
            return (
                "Direct Chainlink Data Streams WebSocket is not configured. "
                "Set CHAINLINK_DATASTREAMS_API_KEY, CHAINLINK_DATASTREAMS_API_SECRET, "
                "and CHAINLINK_DATASTREAMS_FEED_ID_BTCUSD to enable the authoritative live feed. "
                "Use the Polymarket RTDS Chainlink BTC/USD stream as the public fallback."
            )
        if issue == "public_delayed_moved_to_separate_service":
            return (
                "The public delayed Chainlink BTC/USD page is no longer exposed through "
                "CHAINLINK_BACKEND. Use the separate chainlink_public_delayed recorder for "
                "comparison-only delayed page capture."
            )
        missing = [part.removeprefix("missing_") for part in issue.split(",") if part.startswith("missing_")]
        if missing:
            human_missing = ", ".join(missing)
            return (
                "Direct Chainlink Data Streams configuration is incomplete; missing "
                f"{human_missing}. "
                "The recorder will not silently fall back to onchain/page live equivalence. "
                "Use the Polymarket RTDS Chainlink BTC/USD stream as the public fallback."
            )
        return issue


@dataclass(slots=True)
class ChainlinkOnchainConfig:
    out: Path
    symbol: str = "BTCUSD"
    network: str = "ethereum-mainnet"
    rpc_url: str = "https://ethereum.publicnode.com"
    proxy_address: str = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"
    poll_interval_seconds: float = 15.0
    flush_interval_seconds: float = 2.0
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 2
    http_retry_backoff_seconds: float = 0.5
    http_max_connections: int = 10
    http_max_keepalive_connections: int = 2
    http_keepalive_expiry_seconds: float = 5.0
    http_trust_env: bool = False

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("chainlink", "onchain", self.symbol)


@dataclass(slots=True)
class ChainlinkPublicDelayedConfig:
    out: Path
    symbol: str = "BTCUSD"
    page_url: str = "https://data.chain.link/streams/btc-usd-cexprice-streams"
    # NOTE: the HTTP 429 that produced 0 valid records for days was NOT CDN
    # volume rate-limiting (it persisted at a gentle 30s/120s cadence).  The
    # real cause is data.chain.link's Vercel "Security Checkpoint" JS
    # challenge (response header `x-vercel-mitigated: challenge`, body title
    # "Vercel Security Checkpoint").  Plain httpx / curl_cffi cannot solve a
    # JS challenge, so every request was answered with the 429 challenge
    # page.  The recorder now drives a headless Chromium (Playwright) that
    # executes the challenge JS, obtains the clearance, and then calls the
    # SAME live-data-engine benchmark endpoint through the cleared browser
    # context — identical price semantics, just a transport that passes the
    # checkpoint.
    poll_interval_seconds: float = 30.0
    # If the data API ever answers with a challenge/HTTP error again, wait
    # this long before re-navigating (which re-solves the checkpoint).
    rate_limit_backoff_seconds: float = 120.0
    metadata_refresh_interval_seconds: float = 300.0
    http_timeout_seconds: float = 15.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_seconds: float = 0.5
    flush_interval_seconds: float = 2.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    debug_include_html: bool = False
    live_reports_query: str = "LIVE_STREAM_REPORTS_QUERY"
    # Headless Chromium controls for the Vercel-challenge transport.
    headless: bool = True
    nav_timeout_seconds: float = 45.0
    challenge_max_wait_seconds: float = 30.0

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("chainlink_public_delayed", "public_stream_page", self.symbol)


def _pick(keys: dict[str, str], *aliases: str) -> str | None:
    for alias in aliases:
        if alias.lower() in keys:
            return keys[alias.lower()]
    return None


def load_chainlink_config(*, out: Path, key_path: Path | None = None) -> ChainlinkRecorderConfig:
    key_path = key_path or Path("API_Keys")
    keys = load_key_value_file(key_path)
    backend = os.getenv("CHAINLINK_BACKEND", "auto")
    return ChainlinkRecorderConfig(
        out=out,
        symbol=os.getenv("CHAINLINK_SYMBOL", "BTCUSD"),
        backend=backend,
        page_url=os.getenv(
            "CHAINLINK_PAGE_URL", "https://data.chain.link/streams/btc-usd-cexprice-streams"
        ),
        poll_interval_seconds=float(os.getenv("CHAINLINK_POLL_INTERVAL_SECONDS", "2.0")),
        http_timeout_seconds=float(os.getenv("CHAINLINK_HTTP_TIMEOUT_SECONDS", "15.0")),
        reconnect_min_seconds=float(os.getenv("CHAINLINK_RECONNECT_MIN_SECONDS", "1.0")),
        reconnect_max_seconds=float(os.getenv("CHAINLINK_RECONNECT_MAX_SECONDS", "30.0")),
        reconnect_jitter_seconds=float(os.getenv("CHAINLINK_RECONNECT_JITTER_SECONDS", "0.5")),
        flush_interval_seconds=float(os.getenv("CHAINLINK_FLUSH_INTERVAL_SECONDS", "2.0")),
        user_agent=os.getenv(
            "CHAINLINK_USER_AGENT", "Mozilla/5.0 (compatible; market-recorders/0.1)"
        ),
        store_raw_html=os.getenv("CHAINLINK_STORE_RAW_HTML", "false").lower() == "true",
        datastreams_rest_url=os.getenv(
            "CHAINLINK_DATASTREAMS_REST_URL",
            _pick(keys, "datastreams rest url", "chainlink_datastreams_rest_url")
            or "https://api.dataengine.chain.link",
        ),
        datastreams_ws_url=os.getenv(
            "CHAINLINK_DATASTREAMS_WS_URL",
            _pick(keys, "datastreams ws url", "chainlink_datastreams_ws_url")
            or "wss://ws.dataengine.chain.link",
        ),
        datastreams_api_key=os.getenv(
            "CHAINLINK_DATASTREAMS_API_KEY",
            _pick(keys, "datastreams api key", "chainlink_datastreams_api_key"),
        ),
        datastreams_api_secret=os.getenv(
            "CHAINLINK_DATASTREAMS_API_SECRET",
            _pick(
                keys,
                "datastreams api secret",
                "chainlink_datastreams_api_secret",
            ),
        ),
        datastreams_feed_id_btcusd=os.getenv(
            "CHAINLINK_DATASTREAMS_FEED_ID_BTCUSD",
            _pick(
                keys,
                "datastreams feed id btcusd",
                "chainlink_datastreams_feed_id_btcusd",
            ),
        ),
    )


def load_chainlink_onchain_config(*, out: Path) -> ChainlinkOnchainConfig:
    return ChainlinkOnchainConfig(
        out=out,
        symbol=os.getenv("CHAINLINK_ONCHAIN_SYMBOL", "BTCUSD"),
        network=os.getenv("CHAINLINK_ONCHAIN_NETWORK", "ethereum-mainnet"),
        rpc_url=os.getenv("CHAINLINK_ONCHAIN_RPC_URL", "https://ethereum.publicnode.com"),
        proxy_address=os.getenv(
            "CHAINLINK_ONCHAIN_PROXY_ADDRESS", "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"
        ),
        poll_interval_seconds=float(os.getenv("CHAINLINK_ONCHAIN_POLL_INTERVAL_SECONDS", "15.0")),
        flush_interval_seconds=float(os.getenv("CHAINLINK_ONCHAIN_FLUSH_INTERVAL_SECONDS", "2.0")),
        http_timeout_seconds=float(os.getenv("CHAINLINK_ONCHAIN_HTTP_TIMEOUT_SECONDS", "15.0")),
        http_max_retries=int(os.getenv("CHAINLINK_ONCHAIN_HTTP_MAX_RETRIES", "2")),
        http_retry_backoff_seconds=float(
            os.getenv("CHAINLINK_ONCHAIN_HTTP_RETRY_BACKOFF_SECONDS", "0.5")
        ),
        http_max_connections=int(os.getenv("CHAINLINK_ONCHAIN_HTTP_MAX_CONNECTIONS", "10")),
        http_max_keepalive_connections=int(
            os.getenv("CHAINLINK_ONCHAIN_HTTP_MAX_KEEPALIVE_CONNECTIONS", "2")
        ),
        http_keepalive_expiry_seconds=float(
            os.getenv("CHAINLINK_ONCHAIN_HTTP_KEEPALIVE_EXPIRY_SECONDS", "5.0")
        ),
        http_trust_env=os.getenv("CHAINLINK_ONCHAIN_HTTP_TRUST_ENV", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )


def load_chainlink_public_delayed_config(*, out: Path) -> ChainlinkPublicDelayedConfig:
    return ChainlinkPublicDelayedConfig(
        out=out,
        symbol=os.getenv("CHAINLINK_PUBLIC_DELAYED_SYMBOL", "BTCUSD"),
        page_url=os.getenv(
            "CHAINLINK_PUBLIC_DELAYED_PAGE_URL",
            "https://data.chain.link/streams/btc-usd-cexprice-streams",
        ),
        poll_interval_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_POLL_INTERVAL_SECONDS", "30.0")
        ),
        rate_limit_backoff_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_RATE_LIMIT_BACKOFF_SECONDS", "120.0")
        ),
        metadata_refresh_interval_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_METADATA_REFRESH_INTERVAL_SECONDS", "300.0")
        ),
        http_timeout_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_HTTP_TIMEOUT_SECONDS", "15.0")
        ),
        reconnect_min_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_RECONNECT_MIN_SECONDS", "1.0")
        ),
        reconnect_max_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_RECONNECT_MAX_SECONDS", "30.0")
        ),
        reconnect_jitter_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_RECONNECT_JITTER_SECONDS", "0.5")
        ),
        flush_interval_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_FLUSH_INTERVAL_SECONDS", "2.0")
        ),
        user_agent=os.getenv(
            "CHAINLINK_PUBLIC_DELAYED_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ),
        debug_include_html=os.getenv(
            "CHAINLINK_PUBLIC_DELAYED_DEBUG_INCLUDE_HTML", "false"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        live_reports_query=os.getenv(
            "CHAINLINK_PUBLIC_DELAYED_LIVE_REPORTS_QUERY",
            "LIVE_STREAM_REPORTS_QUERY",
        ),
        headless=os.getenv("CHAINLINK_PUBLIC_DELAYED_HEADLESS", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        nav_timeout_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_NAV_TIMEOUT_SECONDS", "45.0")
        ),
        challenge_max_wait_seconds=float(
            os.getenv("CHAINLINK_PUBLIC_DELAYED_CHALLENGE_MAX_WAIT_SECONDS", "30.0")
        ),
    )
