from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from html import unescape
from typing import Any

import httpx

from binance_recorder.utils import ns_to_iso
from market_recorders.http_utils import (
    HttpEndpointBackoffRegistry,
    build_async_http_client,
    request_with_retries,
)
from market_recorders.keyfile import load_key_value_file
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_capture_timing, recv_timestamp_source, utc_now_ns

from .config import PolymarketRecorderConfig
from .discovery import GammaDiscoveryClient
from .lifecycle import lifecycle_event
from .utils import (
    SelectedMarket,
    detection_metadata,
    market_active_window_fields,
    market_has_expired,
    market_is_active,
    market_start_time_iso,
    newest_market,
    session_common_fields,
    sort_markets_by_open,
)
from .writers import PolymarketSessionWriters

_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>', re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PRICE_TO_BEAT_RE = re.compile(
    r"price\s+to\s+beat(?:\s+of)?[^0-9]{0,80}(?P<price>\d{4,}(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
_OPENING_PRICE_RE = re.compile(
    r"opening(?:\s+bitcoin)?\s+price(?:\s+of)?[^0-9]{0,80}(?P<price>\d{4,}(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
_INITIAL_PARSE_RETRY_ATTEMPTS = 3
_INITIAL_PARSE_RETRY_DELAY_SECONDS = 0.35
_POST_SUCCESS_FAILURE_EMIT_THRESHOLD = 3


@dataclass(slots=True)
class _ParsedStrike:
    price_to_beat: str | None
    parse_status: str
    parse_error: str | None = None
    extraction_method: str | None = None


@dataclass(slots=True)
class _StrikeMarketState:
    market: SelectedMarket
    session: PolymarketSessionWriters
    last_emitted_signature: tuple[str, str | None, str | None] | None = None
    consecutive_parse_failures: int = 0
    has_emitted_label_safe_for_market: bool = False


class PolymarketStrikeRecorderService:
    def __init__(self, config: PolymarketRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="polymarket_strike", config=config)
        self._selected_market: SelectedMarket | None = None
        self._session: PolymarketSessionWriters | None = None
        self._last_emitted_signature: tuple[str, str | None, str | None] | None = None
        self._consecutive_parse_failures = 0
        self._has_emitted_label_safe_for_market = False
        self._known_markets: dict[str, SelectedMarket] = {}
        self._market_states: dict[str, _StrikeMarketState] = {}
        self._active_market_ids: set[str] = set()
        self._buffered_lifecycle: list[dict[str, Any]] = []
        self._http_backoff = HttpEndpointBackoffRegistry(
            degraded_failure_threshold=self.config.http_degraded_failure_threshold,
            suspended_failure_threshold=self.config.http_suspended_failure_threshold,
            degraded_backoff_seconds=self.config.http_degraded_backoff_seconds,
            suspended_backoff_seconds=self.config.http_suspended_backoff_seconds,
        )

    async def run(self) -> None:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; market-recorders/0.1)"}
        async with (
            build_async_http_client(
                base_url=self.config.gamma_base_url,
                timeout_seconds=self.config.http_timeout_seconds,
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections,
                keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
                trust_env=self.config.http_trust_env,
            ) as gamma_client,
            build_async_http_client(
                timeout_seconds=self.config.http_timeout_seconds,
                headers=headers,
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections,
                keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
                trust_env=self.config.http_trust_env,
            ) as page_client,
        ):
            discovery = GammaDiscoveryClient(gamma_client, self.config)
            self._write_lifecycle("service_start", {"mode": "public_market_strike"})
            self._write_lifecycle("api_keys_detected", detection_metadata(load_key_value_file(self.config.api_keys_path)))
            last_discovery_ns = 0
            last_observe_ns = 0
            loop_sleep_seconds = min(
                self.config.strike_poll_interval_seconds,
                self.config.market_boundary_check_interval_seconds,
            )
            try:
                while True:
                    now_ns = utc_now_ns()
                    if now_ns - last_discovery_ns >= int(self.config.gamma_poll_interval_seconds * 1_000_000_000):
                        await self._discover(discovery)
                        last_discovery_ns = now_ns
                    self._refresh_active_markets(reason="boundary_tick")
                    if self._iter_active_states() and now_ns - last_observe_ns >= int(
                        self.config.strike_poll_interval_seconds * 1_000_000_000
                    ):
                        await self._observe_strike(page_client)
                        last_observe_ns = now_ns
                    await asyncio.sleep(loop_sleep_seconds)
            finally:
                self._write_lifecycle("service_stop", {"mode": "public_market_strike"})
                for state in list(self._market_states.values()):
                    state.session.close()

    async def _discover(self, discovery: GammaDiscoveryClient) -> None:
        if not self._before_http_request("gamma_discovery"):
            return
        try:
            selected, candidates, _, _, _ = await discovery.discover_market_set()
        except httpx.HTTPError as exc:
            self._record_http_error(
                endpoint="gamma_discovery",
                request_method="GET",
                request_path="/markets",
                error=exc,
            )
            return
        self._observe_http_success("gamma_discovery")
        self._remember_markets(candidates)
        self._refresh_active_markets(reason="gamma_discovery_poll", selected_market=selected)

    async def _observe_strike(self, client: httpx.AsyncClient) -> None:
        if not self._before_http_request("market_page_strike"):
            return
        for state in self._iter_active_states():
            market = state.market
            url = f"{self.config.market_page_base_url}/{market.market_slug}"
            request_start_ns = utc_now_ns()
            request_attempt_count = 0
            try:
                response: httpx.Response | None = None
                parsed: _ParsedStrike | None = None
                html_hash: str | None = None
                html: str | None = None
                response_end_ns = request_start_ns
                for attempt in range(_INITIAL_PARSE_RETRY_ATTEMPTS):
                    outcome = await request_with_retries(
                        client,
                        "GET",
                        url,
                        max_retries=self.config.http_max_retries,
                        retry_backoff_seconds=self.config.http_retry_backoff_seconds,
                    )
                    request_attempt_count += outcome.attempt_count
                    response = outcome.response
                    response_end_ns = utc_now_ns()
                    html = response.text
                    html_hash = hashlib.sha1(html.encode("utf-8")).hexdigest()
                    parsed = self._parse_strike(html, market=market)
                    if parsed.parse_status == "success":
                        break
                    if parsed.parse_status != "price_to_beat_not_found" or attempt + 1 >= _INITIAL_PARSE_RETRY_ATTEMPTS:
                        break
                    await asyncio.sleep(_INITIAL_PARSE_RETRY_DELAY_SECONDS)
            except Exception as exc:
                response_end_ns = utc_now_ns()
                state.session.writer("lifecycle").write(
                    lifecycle_event(
                        event="http_error",
                        details={
                            **session_common_fields(market),
                            "page_url": url,
                            "error": repr(exc),
                        },
                    )
                )
                self._emit_strike_record(
                    state=state,
                    page_url=url,
                    parsed=_ParsedStrike(
                        price_to_beat=None,
                        parse_status="http_error",
                        parse_error=repr(exc),
                        extraction_method=None,
                    ),
                    capture_timing=build_capture_timing(
                        request_start_ns=request_start_ns,
                        response_end_ns=response_end_ns,
                    ),
                    html_hash=None,
                    html=None,
                    )
                continue
            assert parsed is not None
            self._observe_http_success("market_page_strike")
            capture_timing = build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            )
            capture_timing["request_attempt_count"] = request_attempt_count
            if parsed.parse_status == "success":
                state.consecutive_parse_failures = 0
                state.has_emitted_label_safe_for_market = True
            else:
                state.consecutive_parse_failures += 1
                if state.has_emitted_label_safe_for_market and (
                    state.consecutive_parse_failures < _POST_SUCCESS_FAILURE_EMIT_THRESHOLD
                ):
                    state.session.writer("lifecycle").write(
                        lifecycle_event(
                            event="strike_parse_failure_suppressed",
                            details={
                                **session_common_fields(market),
                                "page_url": url,
                                "parse_status": parsed.parse_status,
                                "parse_error": parsed.parse_error,
                                "consecutive_parse_failures": state.consecutive_parse_failures,
                            },
                        )
                    )
                    self._sync_compat_state(state)
                    continue
            self._emit_strike_record(
                state=state,
                page_url=url,
                parsed=parsed,
                capture_timing=capture_timing,
                html_hash=html_hash,
                html=html,
            )
            self._sync_compat_state(state)

    def _parse_strike(self, html: str, *, market: SelectedMarket | None = None) -> _ParsedStrike:
        selected_market = market or self._selected_market
        structured_price = self._parse_price_from_next_data(html, market=selected_market)
        if structured_price is not None:
            price_to_beat, extraction_method = structured_price
            return _ParsedStrike(
                price_to_beat=price_to_beat,
                parse_status="success",
                parse_error=None,
                extraction_method=extraction_method,
            )

        text = self._html_to_text(html)
        match = _PRICE_TO_BEAT_RE.search(text) or _OPENING_PRICE_RE.search(text)
        if match is None:
            return _ParsedStrike(
                price_to_beat=None,
                parse_status="price_to_beat_not_found",
                parse_error="price_to_beat_not_found",
                extraction_method="text_price_to_beat_fallback",
            )
        return _ParsedStrike(
            price_to_beat=match.group("price"),
            parse_status="success",
            parse_error=None,
            extraction_method="text_price_to_beat_fallback",
        )

    def _parse_price_from_next_data(self, html: str, *, market: SelectedMarket | None = None) -> tuple[str, str] | None:
        payload = self._load_next_data_payload(html)
        if payload is None:
            return None

        query_price = self._extract_price_from_crypto_prices_query(payload, market=market)
        if query_price is not None:
            return query_price, "next_data_crypto_prices_open"

        metadata_price = self._extract_price_from_event_metadata(payload, market=market)
        if metadata_price is not None:
            return metadata_price, "next_data_event_metadata_price_to_beat"

        return None

    def _load_next_data_payload(self, html: str) -> dict[str, Any] | None:
        match = _NEXT_DATA_RE.search(html)
        if match is None:
            return None
        try:
            payload = json.loads(match.group("json"), parse_float=Decimal)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_price_from_crypto_prices_query(
        self,
        payload: dict[str, Any],
        *,
        market: SelectedMarket | None = None,
    ) -> str | None:
        queries = payload.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        if not isinstance(queries, list):
            return None

        target_market = market or self._selected_market
        target_start = market_start_time_iso(target_market) if target_market is not None else None
        target_end = target_market.end_time_iso if target_market is not None else None
        fallback_price: str | None = None

        for query in queries:
            if not isinstance(query, dict):
                continue
            query_key = query.get("queryKey")
            if not self._query_key_matches_crypto_prices(query_key):
                continue
            data = query.get("state", {}).get("data", {})
            time_match = self._query_key_matches_market_window(query_key, target_start=target_start, target_end=target_end)
            for item in self._iter_nested_dicts(data):
                open_price = self._coerce_numeric_string(item.get("openPrice"))
                if open_price is None:
                    continue
                if fallback_price is None:
                    fallback_price = open_price
                if time_match:
                    return open_price
        return fallback_price

    def _extract_price_from_event_metadata(
        self,
        payload: dict[str, Any],
        *,
        market: SelectedMarket | None = None,
    ) -> str | None:
        target_market = market or self._selected_market
        if target_market is None:
            return None
        target_identifiers = {
            str(value)
            for value in (
                target_market.market_slug,
                target_market.market_id,
                target_market.condition_id,
            )
            if value is not None and str(value)
        }
        for item in self._iter_nested_dicts(payload):
            if not self._dict_matches_selected_market(item, target_identifiers):
                continue
            for candidate in (
                item,
                item.get("eventMetadata"),
                item.get("metadata"),
                item.get("market"),
                item.get("marketData"),
            ):
                if not isinstance(candidate, dict):
                    continue
                price = self._extract_price_from_metadata_dict(candidate)
                if price is not None:
                    return price
        return None

    @staticmethod
    def _query_key_matches_crypto_prices(query_key: Any) -> bool:
        if isinstance(query_key, list):
            key_text = " ".join(str(item) for item in query_key if item is not None).lower()
        else:
            key_text = str(query_key).lower()
        return "crypto-prices" in key_text and "price" in key_text

    @staticmethod
    def _query_key_matches_market_window(
        query_key: Any,
        *,
        target_start: str | None,
        target_end: str | None,
    ) -> bool:
        if target_start is None or not isinstance(query_key, list):
            return False
        query_values = {str(item) for item in query_key if item is not None}
        if target_start not in query_values:
            return False
        return target_end is None or target_end in query_values

    @staticmethod
    def _extract_price_from_metadata_dict(candidate: dict[str, Any]) -> str | None:
        for field_name in ("priceToBeat", "price_to_beat", "priceToBeatUsd", "price_to_beat_usd"):
            value = candidate.get(field_name)
            if value is not None:
                try:
                    return format(Decimal(str(value)), "f")
                except (InvalidOperation, ValueError):
                    return None
        return None

    @staticmethod
    def _dict_matches_selected_market(candidate: dict[str, Any], target_identifiers: set[str]) -> bool:
        for field_name in ("slug", "marketSlug", "market_slug", "marketId", "conditionId", "id"):
            value = candidate.get(field_name)
            if value is not None and str(value) in target_identifiers:
                return True
        return False

    def _html_to_text(self, html: str) -> str:
        without_tags = _HTML_TAG_RE.sub(" ", html)
        return re.sub(r"\s+", " ", unescape(without_tags))

    def _coerce_numeric_string(self, value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return format(Decimal(str(value)), "f")
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return format(Decimal(stripped), "f")
            except InvalidOperation:
                return None
        return None

    def _iter_nested_dicts(self, value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        stack: list[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                found.append(current)
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
        return found

    def _emit_strike_record(
        self,
        *,
        state: _StrikeMarketState,
        page_url: str,
        parsed: _ParsedStrike,
        capture_timing: dict[str, Any],
        html_hash: str | None,
        html: str | None,
    ) -> None:
        signature = (parsed.parse_status, parsed.price_to_beat, parsed.parse_error)
        if not self._should_emit(state, signature):
            return
        recv_ts_ns = utc_now_ns()
        label_safe = (
            parsed.parse_status == "success"
            and parsed.price_to_beat is not None
            and self._coerce_numeric_string(parsed.price_to_beat) is not None
        )
        record = {
            "record_type": "strike_observation",
            "source": "polymarket",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            **session_common_fields(state.market),
            "page_url": page_url,
            "price_to_beat": parsed.price_to_beat,
            "parse_status": parsed.parse_status,
            "parse_error": parsed.parse_error,
            "label_safe": label_safe,
            "extraction_method": parsed.extraction_method,
            "html_hash": html_hash,
            "capture_timing": capture_timing,
        }
        if self.config.strike_debug_include_html and html is not None:
            record["debug"] = {"html": html}
        state.session.writer("strike").write(record)
        state.last_emitted_signature = signature
        self._sync_compat_state(state)

    def _should_emit(self, state: _StrikeMarketState, signature: tuple[str, str | None, str | None]) -> bool:
        if self.config.strike_mode == "always":
            return True
        if state.last_emitted_signature is None:
            return True
        return state.last_emitted_signature != signature

    def _remember_markets(self, markets: list[SelectedMarket]) -> None:
        for market in markets:
            self._known_markets[market.identity] = market
            state = self._market_states.get(market.identity)
            if state is not None:
                state.market = market
                state.session.market = market

    def _refresh_active_markets(self, *, reason: str, selected_market: SelectedMarket | None = None) -> None:
        now = datetime.now(tz=UTC)
        if selected_market is not None:
            self._remember_markets([selected_market])
        previous_active_ids = set(self._active_market_ids)
        previous_active_markets = self._markets_for_ids(previous_active_ids)
        active_markets = sort_markets_by_open(
            [
                market
                for market in self._known_markets.values()
                if market_is_active(
                    market,
                    now=now,
                    interval_seconds=self.config.market_interval_seconds,
                    preopen_seconds=self.config.market_active_preopen_seconds,
                )
            ]
        )
        active_ids = {market.identity for market in active_markets}
        added_ids = [market.identity for market in active_markets if market.identity not in previous_active_ids]
        removed_ids = [market_id for market_id in previous_active_ids if market_id not in active_ids]
        for market in active_markets:
            self._ensure_state(market)
        self._active_market_ids = active_ids
        active_context = self._active_market_context(active_markets=active_markets, reason=reason)
        if len(previous_active_ids) < 2 <= len(active_ids):
            self._broadcast_market_lifecycle(
                list(active_ids),
                "dual_market_overlap_started",
                {
                    **active_context,
                    "previous_active_market_ids": list(previous_active_ids),
                    "previous_active_market_slugs": [market.market_slug for market in previous_active_markets],
                },
            )
        if len(previous_active_ids) >= 2 and len(active_ids) < 2:
            self._broadcast_market_lifecycle(
                list(previous_active_ids | active_ids),
                "dual_market_overlap_ended",
                {
                    **active_context,
                    "previous_active_market_ids": list(previous_active_ids),
                    "previous_active_market_slugs": [market.market_slug for market in previous_active_markets],
                },
            )
        for market_id in added_ids:
            state = self._market_states.get(market_id)
            if state is None:
                continue
            state.session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_active",
                    details={
                        **session_common_fields(state.market),
                        **self._market_boundary_fields(state.market),
                        **active_context,
                    },
                )
            )
        primary_market = newest_market(active_markets)
        if primary_market is not None:
            previous_primary = self._selected_market
            self._selected_market = primary_market
            primary_state = self._market_states.get(primary_market.identity)
            self._session = primary_state.session if primary_state is not None else None
            if self._session is not None and (
                previous_primary is None or previous_primary.identity != primary_market.identity
            ):
                self._session.writer("lifecycle").write(
                    lifecycle_event(
                        event="market_selected" if previous_primary is None else "market_switched",
                        details={
                            **session_common_fields(primary_market),
                            **self._market_boundary_fields(primary_market),
                            **active_context,
                        },
                    )
                )
        elif not self._active_market_ids:
            self._selected_market = None
            self._session = None
        for market_id in removed_ids:
            state = self._market_states.get(market_id)
            if state is None:
                continue
            state.session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_inactive",
                    details={
                        **session_common_fields(state.market),
                        **self._market_boundary_fields(state.market),
                        **active_context,
                    },
                )
            )
            state.session.close()
            del self._market_states[market_id]
        self._prune_expired_known_markets(now=now)

    def _ensure_state(self, market: SelectedMarket) -> _StrikeMarketState:
        state = self._market_states.get(market.identity)
        if state is not None:
            state.market = market
            state.session.market = market
            return state
        session = PolymarketSessionWriters(
            root=self.config.out,
            market=market,
            started_at=datetime.now(tz=UTC),
            kinds=("strike", "lifecycle"),
            flush_interval_seconds=self.config.flush_interval_seconds,
            manifest_enabled=self.config.manifest_enabled,
            static_fields_extra=self._runtime.static_fields(),
        )
        for buffered in self._buffered_lifecycle:
            session.writer("lifecycle").write({**buffered, **session_common_fields(market)})
        if self._buffered_lifecycle:
            self._buffered_lifecycle.clear()
        state = _StrikeMarketState(market=market, session=session)
        self._market_states[market.identity] = state
        return state

    def _iter_active_states(self) -> list[_StrikeMarketState]:
        markets = self._markets_for_ids(self._active_market_ids)
        states = [self._market_states[market.identity] for market in markets if market.identity in self._market_states]
        if states:
            return states
        if self._selected_market is not None and self._session is not None:
            return [
                _StrikeMarketState(
                    market=self._selected_market,
                    session=self._session,
                    last_emitted_signature=self._last_emitted_signature,
                    consecutive_parse_failures=self._consecutive_parse_failures,
                    has_emitted_label_safe_for_market=self._has_emitted_label_safe_for_market,
                )
            ]
        return []

    def _markets_for_ids(self, market_ids: set[str] | list[str]) -> list[SelectedMarket]:
        values: list[SelectedMarket] = []
        for market_id in market_ids:
            state = self._market_states.get(market_id)
            if state is not None:
                values.append(state.market)
                continue
            market = self._known_markets.get(market_id)
            if market is not None:
                values.append(market)
        return sort_markets_by_open(values)

    def _market_boundary_fields(self, market: SelectedMarket) -> dict[str, Any]:
        return market_active_window_fields(
            market,
            interval_seconds=self.config.market_interval_seconds,
            preopen_seconds=self.config.market_active_preopen_seconds,
        )

    def _active_market_context(
        self,
        *,
        active_markets: list[SelectedMarket] | None = None,
        reason: str,
    ) -> dict[str, Any]:
        markets = active_markets if active_markets is not None else self._markets_for_ids(self._active_market_ids)
        return {
            "reason": reason,
            "active_market_ids": [market.identity for market in markets],
            "active_market_slugs": [market.market_slug for market in markets],
            "active_market_boundaries": [self._market_boundary_fields(market) for market in markets],
        }

    def _broadcast_market_lifecycle(
        self,
        market_ids: list[str],
        event: str,
        details: dict[str, Any],
    ) -> None:
        seen: set[str] = set()
        for market_id in market_ids:
            if market_id in seen:
                continue
            seen.add(market_id)
            state = self._market_states.get(market_id)
            if state is None:
                continue
            state.session.writer("lifecycle").write(
                lifecycle_event(event=event, details={**session_common_fields(state.market), **details})
            )

    def _prune_expired_known_markets(self, *, now: datetime) -> None:
        expired_ids = [
            market_id
            for market_id, market in self._known_markets.items()
            if market_id not in self._active_market_ids
            and market_id not in self._market_states
            and market_has_expired(
                market,
                now=now,
                interval_seconds=self.config.market_interval_seconds,
            )
        ]
        for market_id in expired_ids:
            del self._known_markets[market_id]

    def _sync_compat_state(self, state: _StrikeMarketState) -> None:
        if state.market is self._selected_market and state.session is self._session:
            self._last_emitted_signature = state.last_emitted_signature
            self._consecutive_parse_failures = state.consecutive_parse_failures
            self._has_emitted_label_safe_for_market = state.has_emitted_label_safe_for_market

    def _record_http_error(
        self,
        *,
        endpoint: str,
        request_method: str,
        request_path: str,
        error: Exception,
    ) -> None:
        details = {
            "endpoint": endpoint,
            "request_method": request_method,
            "request_path": request_path,
            "error": repr(error),
        }
        status_code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
        health = self._http_backoff.observe_failure(
            endpoint,
            now_ns=utc_now_ns(),
            status_code=status_code,
            reason=type(error).__name__,
        )
        details["http_endpoint_state"] = health.current_state
        if status_code is not None:
            details["status_code"] = status_code
        self.logger.warning("Polymarket HTTP request failed", extra=details)
        self._write_lifecycle("http_error", details)
        for event in health.lifecycle_events:
            self._write_lifecycle(str(event["event"]), dict(event))

    def _before_http_request(self, endpoint: str) -> bool:
        decision = self._http_backoff.before_request(endpoint, now_ns=utc_now_ns())
        for event in decision.lifecycle_events:
            self._write_lifecycle(str(event["event"]), dict(event))
        return not decision.skip_request

    def _observe_http_success(self, endpoint: str) -> None:
        decision = self._http_backoff.observe_success(endpoint)
        for event in decision.lifecycle_events:
            self._write_lifecycle(str(event["event"]), dict(event))

    def _write_lifecycle(self, event: str, details: dict[str, Any] | None = None) -> None:
        record = lifecycle_event(
            event=event,
            details=self._runtime.lifecycle_fields(
                include_config_snapshot=event == "service_start",
                extra_fields=details,
            ),
        )
        if self._session is None:
            self._buffered_lifecycle.append(record)
        else:
            self._session.writer("lifecycle").write(
                {**record, **session_common_fields(self._session.market), **(details or {})}
            )
