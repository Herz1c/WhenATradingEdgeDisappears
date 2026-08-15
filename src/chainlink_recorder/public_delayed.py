from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import orjson
from playwright.async_api import async_playwright

from binance_recorder.utils import ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import (
    build_capture_timing,
    parse_iso8601_to_ns,
    recv_timestamp_source,
    utc_now_ns,
)

from .config import ChainlinkPublicDelayedConfig

_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(?P<json>.*?)</script>', re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)


def _lifecycle(
    event: str,
    config: ChainlinkPublicDelayedConfig,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "chainlink_public_delayed",
        "backend": "public_stream_page",
        "symbol": config.symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


@dataclass(slots=True)
class _PublicPageMetadata:
    feed_id: str
    multiply: Decimal
    schema: str | None
    page_hash: str
    page_title: str | None
    fetched_at_ns: int
    html: str | None


def _parse_page_metadata(html: str, *, fetched_at_ns: int, debug_include_html: bool) -> _PublicPageMetadata:
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        raise ValueError("next_data_not_found")
    try:
        payload = json.loads(match.group("json"))
    except json.JSONDecodeError as exc:
        raise ValueError("next_data_invalid_json") from exc
    stream_metadata = payload.get("props", {}).get("pageProps", {}).get("streamData", {}).get("streamMetadata", {})
    if not isinstance(stream_metadata, dict):
        raise ValueError("stream_metadata_not_found")
    feed_id = stream_metadata.get("feedId")
    if not isinstance(feed_id, str) or not feed_id:
        raise ValueError("feed_id_not_found")
    docs = stream_metadata.get("docs", {})
    schema = docs.get("schema") if isinstance(docs, dict) else None
    if schema is not None and not isinstance(schema, str):
        schema = None
    multiply_raw = stream_metadata.get("multiply")
    if multiply_raw in (None, ""):
        decimals = stream_metadata.get("decimals")
        multiply_raw = str(10 ** int(decimals)) if decimals is not None else "1"
    try:
        multiply = Decimal(str(multiply_raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid_multiply") from exc
    title_match = _TITLE_RE.search(html)
    return _PublicPageMetadata(
        feed_id=feed_id,
        multiply=multiply,
        schema=schema,
        page_hash=hashlib.sha1(html.encode("utf-8")).hexdigest(),
        page_title=title_match.group("title").strip() if title_match else None,
        fetched_at_ns=fetched_at_ns,
        html=html if debug_include_html else None,
    )


def _live_reports_data_url(config: ChainlinkPublicDelayedConfig) -> str:
    page_url = httpx.URL(config.page_url)
    return str(page_url.copy_with(path="/api/query-timescale", query=None))


def _live_data_engine_url(config: ChainlinkPublicDelayedConfig) -> str:
    page_url = httpx.URL(config.page_url)
    return str(page_url.copy_with(path="/api/live-data-engine-stream-data", query=None))


def _coerce_scaled_price(value: Any, *, multiply: Decimal) -> str | None:
    if value is None:
        return None
    try:
        scaled = Decimal(str(value)) / multiply
    except (InvalidOperation, ZeroDivisionError, TypeError, ValueError):
        return None
    return format(scaled.normalize() if scaled == scaled.to_integral() else scaled, "f")


def _coerce_decimal_price(value: Any) -> str | None:
    if value is None:
        return None
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return format(
        decimal_value.normalize() if decimal_value == decimal_value.to_integral() else decimal_value,
        "f",
    )


def _live_data_engine_query_params(metadata: _PublicPageMetadata) -> dict[str, str]:
    schema = metadata.schema or "v3"
    # The public BTC/USD Data Streams page currently exposes the v3 benchmark,
    # bid, and ask values through abiIndex=0 attributes. We capture benchmark
    # to preserve the old public-delayed mid-price semantics.
    if schema == "v3":
        return {
            "feedId": metadata.feed_id,
            "abiIndex": "0",
            "queryWindow": "1m",
            "attributeName": "benchmark",
        }
    return {
        "feedId": metadata.feed_id,
        "abiIndex": "0",
        "queryWindow": "1m",
    }


def _select_latest_live_data_engine_node(
    payload: dict[str, Any],
    *,
    attribute_name: str | None,
) -> tuple[dict[str, Any], int, str] | None:
    nodes = payload.get("data", {}).get("allStreamValuesGenerics", {}).get("nodes", [])
    if not isinstance(nodes, list):
        return None
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if attribute_name and node.get("attributeName") != attribute_name:
            continue
        value = _coerce_decimal_price(node.get("valueNumeric"))
        ts_ns = parse_iso8601_to_ns(str(node.get("validAfterTs") or ""))
        if value is None or ts_ns is None:
            continue
        candidates.append((ts_ns, node, value))
    if not candidates:
        return None
    ts_ns, node, value = max(candidates, key=lambda item: item[0])
    return node, ts_ns, value


def _build_failure_record(
    *,
    config: ChainlinkPublicDelayedConfig,
    page_url: str,
    recv_ts_ns: int,
    capture_timing: dict[str, Any],
    parse_status: str,
    parse_error: str,
    page_hash: str | None = None,
    content_hash: str | None = None,
    page_title: str | None = None,
    feed_id: str | None = None,
    data_url: str | None = None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "record_type": "chainlink_public_delayed_observation",
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
        "source": "chainlink_public_delayed",
        "backend": "public_stream_page",
        "symbol": config.symbol,
        "page_url": page_url,
        "feed_id": feed_id,
        "data_url": data_url,
        "chainlink_display_ts": None,
        "chainlink_display_ts_iso": None,
        "btc_usd_price": None,
        "capture_timing": capture_timing,
        "parse_status": parse_status,
        "parse_error": parse_error,
        "page_hash": page_hash,
        "content_hash": content_hash,
        "page_title": page_title,
    }
    if debug:
        record["debug"] = debug
    return record


class ChainlinkPublicDelayedRecorderService:
    def __init__(self, config: ChainlinkPublicDelayedConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="chainlink_public_delayed", config=config)
        static_fields = self._runtime.manifest_fields({
            "source": "chainlink_public_delayed",
            "backend": "public_stream_page",
            "symbol": config.symbol,
        })
        self._data_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="page",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
        )
        self._metadata: _PublicPageMetadata | None = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._data_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        # data.chain.link is fronted by a Vercel "Security Checkpoint" JS
        # challenge that answers every non-browser client with HTTP 429.
        # We therefore drive a real headless Chromium: it executes the
        # challenge JS, the browser context holds the resulting clearance,
        # and we then call the SAME benchmark data API through that context.
        self._lifecycle_writer.write(_lifecycle("service_start", self.config, {"page_url": self.config.page_url}))
        try:
            async with async_playwright() as pw:
                self._browser = await pw.chromium.launch(
                    headless=self.config.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                self._context = await self._browser.new_context(
                    user_agent=self.config.user_agent,
                    locale="en-US",
                    viewport={"width": 1280, "height": 800},
                )
                self._page = await self._context.new_page()
                self._lifecycle_writer.write(
                    _lifecycle("browser_started", self.config, {"headless": self.config.headless})
                )
                await self._poll_loop()
        finally:
            self._lifecycle_writer.write(_lifecycle("shutdown", self.config, {"page_url": self.config.page_url}))
            self._data_writer.close()
            self._lifecycle_writer.close()

    async def _poll_loop(self) -> None:
        while True:
            self._advance_writers()
            self._lifecycle_writer.write(_lifecycle("fetch_start", self.config, {"page_url": self.config.page_url}))
            sleep_seconds = self.config.poll_interval_seconds   # default: success path
            try:
                record = await self._fetch_record()
                self._data_writer.write(record)
                lifecycle_event = "fetch_success" if record["parse_status"] == "success" else "parse_failure"
                self._lifecycle_writer.write(
                    _lifecycle(
                        lifecycle_event,
                        self.config,
                        {
                            "page_url": self.config.page_url,
                            "feed_id": record.get("feed_id"),
                            "parse_status": record.get("parse_status"),
                            "parse_error": record.get("parse_error"),
                        },
                    )
                )
                # Back off harder when the Polymarket CDN rate-limits us.
                # The parse_error string contains "429 Too Many Requests"
                # when this happens (see _fetch_record). Without this
                # check we'd keep hammering at poll_interval and never
                # recover (see chainlink_public_delayed May 24-26: 0 ok
                # records across 10k requests, all 429).
                parse_err = (record.get("parse_error") or "")
                if "429" in parse_err or "Too Many Requests" in parse_err:
                    sleep_seconds = self.config.rate_limit_backoff_seconds
                    self._lifecycle_writer.write(_lifecycle(
                        "rate_limit_backoff", self.config,
                        {"page_url": self.config.page_url,
                         "sleep_seconds": sleep_seconds,
                         "trigger": "http_429"},
                    ))
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Network/transport-level exceptions (not 429 which is in-band)
                msg = repr(exc)
                if "429" in msg or "Too Many Requests" in msg:
                    sleep_seconds = self.config.rate_limit_backoff_seconds
                else:
                    sleep_seconds = self.config.reconnect_min_seconds + random.random() * self.config.reconnect_jitter_seconds
                self._lifecycle_writer.write(
                    _lifecycle(
                        "backoff",
                        self.config,
                        {
                            "page_url": self.config.page_url,
                            "error": msg,
                            "sleep_seconds": sleep_seconds,
                        },
                    )
                )
                await asyncio.sleep(sleep_seconds)

    async def _refresh_metadata_if_needed(self) -> _PublicPageMetadata:
        now_ns = utc_now_ns()
        refresh_due_ns = int(self.config.metadata_refresh_interval_seconds * 1_000_000_000)
        if self._metadata is not None and (now_ns - self._metadata.fetched_at_ns) < refresh_due_ns:
            return self._metadata
        # Navigating with a real browser executes the Vercel challenge JS and
        # leaves the clearance on the context. We then wait until the real
        # page (its __NEXT_DATA__ blob) replaces the "Security Checkpoint".
        await self._page.goto(
            self.config.page_url,
            wait_until="domcontentloaded",
            timeout=int(self.config.nav_timeout_seconds * 1000),
        )
        deadline = time.monotonic() + self.config.challenge_max_wait_seconds
        html = await self._page.content()
        while ("__NEXT_DATA__" not in html or "Security Checkpoint" in html) and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            html = await self._page.content()
        if "__NEXT_DATA__" not in html or "Security Checkpoint" in html:
            raise ValueError("vercel_challenge_unsolved")
        fetched_at_ns = utc_now_ns()
        self._metadata = _parse_page_metadata(
            html,
            fetched_at_ns=fetched_at_ns,
            debug_include_html=self.config.debug_include_html,
        )
        self._lifecycle_writer.write(
            _lifecycle(
                "challenge_cleared",
                self.config,
                {"feed_id": self._metadata.feed_id, "page_title": self._metadata.page_title},
            )
        )
        return self._metadata

    async def _fetch_record(self) -> dict[str, Any]:
        request_start_ns = utc_now_ns()
        metadata: _PublicPageMetadata | None = None
        try:
            metadata = await self._refresh_metadata_if_needed()
        except Exception as exc:
            response_end_ns = utc_now_ns()
            recv_ts_ns = utc_now_ns()
            # Force a fresh navigation (re-solve the challenge) next loop.
            self._metadata = None
            return _build_failure_record(
                config=self.config,
                page_url=self.config.page_url,
                recv_ts_ns=recv_ts_ns,
                capture_timing=build_capture_timing(
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                ),
                parse_status="page_metadata_error",
                parse_error=repr(exc),
            )

        data_url = _live_data_engine_url(self.config)
        query_params = _live_data_engine_query_params(metadata)
        try:
            # Issue the data request through the cleared browser context so it
            # carries the Vercel challenge clearance the page navigation set.
            response = await self._context.request.get(
                data_url,
                params=query_params,
                timeout=self.config.http_timeout_seconds * 1000,
            )
            response_end_ns = utc_now_ns()
            recv_ts_ns = utc_now_ns()
            status = response.status
            body = await response.body()
            if status >= 400:
                # 429 / challenge again -> drop metadata so we re-navigate.
                self._metadata = None
                raise RuntimeError(
                    f"HTTPStatusError status={status} (Too Many Requests / challenge)"
                    if status == 429
                    else f"HTTPStatusError status={status}"
                )
            payload = orjson.loads(body)
        except Exception as exc:
            response_end_ns = utc_now_ns()
            recv_ts_ns = utc_now_ns()
            return _build_failure_record(
                config=self.config,
                page_url=self.config.page_url,
                recv_ts_ns=recv_ts_ns,
                capture_timing=build_capture_timing(
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                ),
                parse_status="http_error",
                parse_error=repr(exc),
                page_hash=metadata.page_hash,
                page_title=metadata.page_title,
                feed_id=metadata.feed_id,
                data_url=data_url,
            )

        content_hash = hashlib.sha1(body).hexdigest()
        selected = _select_latest_live_data_engine_node(
            payload,
            attribute_name=query_params.get("attributeName"),
        )
        if selected is None:
            return _build_failure_record(
                config=self.config,
                page_url=self.config.page_url,
                recv_ts_ns=recv_ts_ns,
                capture_timing=build_capture_timing(
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                ),
                parse_status="missing_live_data_engine_node",
                parse_error="missing_live_data_engine_node",
                page_hash=metadata.page_hash,
                content_hash=content_hash,
                page_title=metadata.page_title,
                feed_id=metadata.feed_id,
                data_url=data_url,
                debug={"live_data_engine_payload": payload} if self.config.debug_include_html else None,
            )

        latest, display_ts_ns, price = selected
        record = {
            "record_type": "chainlink_public_delayed_observation",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "source": "chainlink_public_delayed",
            "backend": "public_stream_page",
            "symbol": self.config.symbol,
            "page_url": self.config.page_url,
            "feed_id": metadata.feed_id,
            "data_url": data_url,
            "data_api": "live_data_engine_stream_data",
            "data_query_params": query_params,
            "chainlink_display_ts": display_ts_ns // 1_000_000_000,
            "chainlink_display_ts_iso": ns_to_iso(display_ts_ns),
            "btc_usd_price": price,
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "parse_status": "success",
            "parse_error": None,
            "page_hash": metadata.page_hash,
            "content_hash": content_hash,
            "page_title": metadata.page_title,
        }
        if self.config.debug_include_html:
            record["debug"] = {
                "html": metadata.html,
                "live_data_engine_node": latest,
            }
        return record
