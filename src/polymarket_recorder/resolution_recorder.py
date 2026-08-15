from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import orjson
from websockets import connect

from binance_recorder.compression import iter_zstd_jsonl_with_options
from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.asyncio_utils import cancel_and_await
from market_recorders.http_utils import (
    HttpEndpointBackoffRegistry,
    build_async_http_client,
    request_with_retries,
)
from market_recorders.keyfile import load_key_value_file
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import (
    epoch_value_to_ns,
    normalize_epoch_value,
    parse_iso8601_to_ns,
    recv_timestamp_source,
    resolve_epoch_unit,
    utc_now_ns,
)

from .config import PolymarketRecorderConfig
from .discovery import GammaDiscoveryClient
from .lifecycle import lifecycle_event
from .utils import (
    SelectedMarket,
    candidate_target_markets,
    detection_metadata,
    generate_target_slugs,
    market_active_window_fields,
    market_close_epoch,
    market_open_epoch,
    market_start_time_iso,
    newest_market,
    normalize_outcome_side,
    sort_markets_by_open,
)

_RESOLUTION_NAMESPACE = ("polymarket", "resolution", "btc_updown_5m")
_RESOLUTION_SOURCE = "polymarket_market_resolved_ws"
_REST_STATE_SOURCE = "polymarket_gamma_market_rest"
_REST_RESOLUTION_SOURCE = "polymarket_gamma_market_rest_winner"
_PAGE_RESOLUTION_SOURCE = "polymarket_market_page_crypto_prices"
_PAGE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; market-recorders/0.1)"}
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>', re.DOTALL)


def _iter_payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _timestamp_value_to_ns(value: Any) -> int | None:
    normalized_epoch = normalize_epoch_value(value)
    if normalized_epoch is not None:
        unit = resolve_epoch_unit(normalized_epoch)
        return epoch_value_to_ns(normalized_epoch, unit=unit)
    if isinstance(value, str):
        return parse_iso8601_to_ns(value)
    return None


def _coerce_decimal_string(value: Any) -> str | None:
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
    try:
        return format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return None


def _load_next_data_payload(html: str) -> dict[str, Any] | None:
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        return None
    try:
        payload = orjson.loads(match.group("json"))
    except orjson.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _iter_nested_dicts(value: Any) -> list[dict[str, Any]]:
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


def _query_key_matches_crypto_prices(query_key: Any) -> bool:
    if isinstance(query_key, list):
        key_text = " ".join(str(item) for item in query_key if item is not None).lower()
    else:
        key_text = str(query_key).lower()
    return "crypto-prices" in key_text and "price" in key_text


def _query_key_matches_market_window(
    query_key: Any,
    *,
    target_start: str | None,
    target_end: str | None,
) -> bool:
    if target_start is None or target_end is None or not isinstance(query_key, list):
        return False
    target_start_ns = _timestamp_value_to_ns(target_start)
    target_end_ns = _timestamp_value_to_ns(target_end)
    if target_start_ns is None or target_end_ns is None:
        query_values = {str(item) for item in query_key if item is not None}
        return target_start in query_values and target_end in query_values
    query_timestamp_values = {
        value
        for item in query_key
        for value in [_timestamp_value_to_ns(item)]
        if value is not None
    }
    return target_start_ns in query_timestamp_values and target_end_ns in query_timestamp_values


def _resolution_from_open_close_prices(*, open_price: Any, close_price: Any) -> dict[str, str] | None:
    normalized_open = _coerce_decimal_string(open_price)
    normalized_close = _coerce_decimal_string(close_price)
    if normalized_open is None or normalized_close is None:
        return None
    open_decimal = Decimal(normalized_open)
    close_decimal = Decimal(normalized_close)
    resolved_side = "up" if close_decimal >= open_decimal else "down"
    return {
        "open_price": normalized_open,
        "close_price": normalized_close,
        "resolved_side": resolved_side,
        "winning_outcome": "Up" if resolved_side == "up" else "Down",
    }


def _extract_page_resolution(market: SelectedMarket, *, html: str) -> dict[str, Any] | None:
    payload = _load_next_data_payload(html)
    if payload is None:
        return None
    target_start = market_start_time_iso(market)
    target_end = market.end_time_iso
    if target_end is None:
        close_epoch = market_close_epoch(market, interval_seconds=300)
        if close_epoch is not None:
            target_end = datetime.fromtimestamp(close_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = payload.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
    if isinstance(queries, list):
        for query in queries:
            if not isinstance(query, dict):
                continue
            query_key = query.get("queryKey")
            if not _query_key_matches_crypto_prices(query_key):
                continue
            if not _query_key_matches_market_window(query_key, target_start=target_start, target_end=target_end):
                continue
            data = query.get("state", {}).get("data", {})
            for item in _iter_nested_dicts(data):
                price_resolution = _resolution_from_open_close_prices(
                    open_price=item.get("openPrice"),
                    close_price=item.get("closePrice"),
                )
                if price_resolution is None:
                    continue
                return {
                    **price_resolution,
                    "source_detail": "crypto_prices_query",
                }
    open_ts_ns = _timestamp_value_to_ns(target_start)
    close_ts_ns = _timestamp_value_to_ns(target_end)
    if open_ts_ns is None or close_ts_ns is None:
        return None
    for item in _iter_nested_dicts(payload):
        results = item.get("results") if isinstance(item, dict) else None
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            if _timestamp_value_to_ns(result.get("startTime")) != open_ts_ns:
                continue
            if _timestamp_value_to_ns(result.get("endTime")) != close_ts_ns:
                continue
            price_resolution = _resolution_from_open_close_prices(
                open_price=result.get("openPrice"),
                close_price=result.get("closePrice"),
            )
            if price_resolution is None:
                continue
            return {
                **price_resolution,
                "source_detail": "past_results_window",
                "page_result_outcome": result.get("outcome"),
            }
    return None


def _dedupe_keys(
    *,
    market: SelectedMarket | None = None,
    market_id: str | None = None,
    condition_id: str | None = None,
    market_slug: str | None = None,
) -> list[str]:
    keys: list[str] = []
    resolved_market_id = market_id or (market.market_id if market is not None else None)
    resolved_condition_id = condition_id or (market.condition_id if market is not None else None)
    resolved_market_slug = market_slug or (market.market_slug if market is not None else None)
    for prefix, value in (
        ("market_id", resolved_market_id),
        ("condition_id", resolved_condition_id),
        ("market_slug", resolved_market_slug),
    ):
        if value:
            keys.append(f"{prefix}:{value}")
    return keys


def _top_level_market_fields(market: SelectedMarket) -> dict[str, Any]:
    open_epoch = market_open_epoch(market)
    close_epoch = market_close_epoch(market, interval_seconds=300)
    open_ns = open_epoch * 1_000_000_000 if open_epoch is not None else None
    close_ns = close_epoch * 1_000_000_000 if close_epoch is not None else None
    return {
        "market_id": market.market_id,
        "market_slug": market.market_slug,
        "condition_id": market.condition_id,
        "market_open_ts_ns": open_ns,
        "market_open_ts_iso": ns_to_iso(open_ns) if open_ns is not None else None,
        "market_close_ts_ns": close_ns,
        "market_close_ts_iso": ns_to_iso(close_ns) if close_ns is not None else None,
        "market_open_s": open_epoch,
        "market_close_s": close_epoch,
        "asset_ids": list(market.asset_ids),
        "outcomes": list(market.outcomes),
        "token_mapping": [
            {
                "asset_id": asset_id,
                "outcome": outcome,
                "normalized_side": normalize_outcome_side(outcome),
            }
            for asset_id, outcome in market.outcome_map.items()
        ],
        **market_active_window_fields(market, interval_seconds=300),
    }


@dataclass(slots=True)
class _ResolutionSubscriptionChange:
    added_asset_ids: list[str]
    removed_asset_ids: list[str]
    reason: str
    tracked_market_ids: list[str]
    tracked_market_slugs: list[str]


@dataclass(slots=True)
class _ResolvedMarketOutcome:
    winning_asset_id: str
    winning_outcome: str
    winning_outcome_raw: str | None
    winning_asset_id_in_market_map: bool
    resolved_side: str
    resolved_side_mapping_source: str
    resolved_side_authoritative: bool


class PolymarketResolutionRecorderService:
    def __init__(self, config: PolymarketRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="polymarket_resolution", config=config)
        static_fields = {
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "component": "resolution",
            **self._runtime.static_fields(),
        }
        self._history_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=_RESOLUTION_NAMESPACE,
            kind="history",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_counter_fields=("record_type", "signal_source", "canonical_action", "resolved_side"),
            logger=self.logger,
        )
        self._resolution_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=_RESOLUTION_NAMESPACE,
            kind="resolution",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
            manifest_counter_fields=("record_type", "resolution_source", "resolved_side"),
            logger=self.logger,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=_RESOLUTION_NAMESPACE,
            kind="lifecycle",
            flush_interval_seconds=max(1.0, min(config.flush_interval_seconds, 5.0)),
            manifest_static_fields=static_fields,
            manifest_counter_fields=("record_type", "event"),
            logger=self.logger,
        )
        self._known_markets: dict[str, SelectedMarket] = {}
        self._tracked_market_ids: set[str] = set()
        self._pending_rest_reconcile_market_ids: set[str] = set()
        self._pending_page_backfill_market_ids: set[str] = set()
        self._connected_asset_ids: list[str] = []
        self._pending_subscription_change: _ResolutionSubscriptionChange | None = None
        self._last_activity_ns = utc_now_ns()
        self._canonical_resolution_index: dict[str, dict[str, Any]] = {}
        self._latest_rest_state_by_market_identity: dict[str, dict[str, Any]] = {}
        self._rest_resolved_without_canonical_market_ids: set[str] = set()
        self._http_backoff = HttpEndpointBackoffRegistry(
            degraded_failure_threshold=self.config.http_degraded_failure_threshold,
            suspended_failure_threshold=self.config.http_suspended_failure_threshold,
            degraded_backoff_seconds=self.config.http_degraded_backoff_seconds,
            suspended_backoff_seconds=self.config.http_suspended_backoff_seconds,
        )

    async def run(self) -> None:
        self._bootstrap_existing_resolutions()
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
                headers=_PAGE_HEADERS,
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections,
                keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
                trust_env=self.config.http_trust_env,
            ) as page_client,
        ):
            discovery_client = GammaDiscoveryClient(gamma_client, self.config)
            discovery_task: asyncio.Task[None] | None = None
            rest_task: asyncio.Task[None] | None = None
            try:
                self._write_lifecycle("service_start", {"mode": "polymarket_market_resolution"})
                self._write_lifecycle("api_keys_detected", detection_metadata(load_key_value_file(self.config.api_keys_path)))
                await self._discover_once(discovery_client, poll_seq=1)
                await self._poll_rest_once(discovery_client, page_client=page_client)
                discovery_task = asyncio.create_task(
                    self._discovery_loop(discovery_client),
                    name="polymarket:resolution:discovery",
                )
                rest_task = asyncio.create_task(
                    self._rest_state_loop(discovery_client, page_client=page_client),
                    name="polymarket:resolution:rest",
                )
                await self._websocket_loop()
            finally:
                cancelled_during_cleanup = await cancel_and_await(discovery_task, rest_task)
                self._write_lifecycle("service_stop", {"mode": "polymarket_market_resolution"})
                self._history_writer.close()
                self._resolution_writer.close()
                self._lifecycle_writer.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    def _bootstrap_existing_resolutions(self) -> None:
        base = self.config.out / "raw" / Path(*_RESOLUTION_NAMESPACE)
        loaded = 0
        if base.exists():
            for path in sorted(base.rglob("*.resolution.jsonl.zst")):
                for record in iter_zstd_jsonl_with_options(
                    path,
                    allow_truncated_stream=True,
                    allow_partial_final_line=True,
                ):
                    if str(record.get("record_type")) != "market_resolution":
                        continue
                    self._index_canonical_resolution(record)
                    loaded += 1
        self._write_lifecycle(
            "bootstrap_existing_resolutions_loaded",
            {
                "loaded_resolution_count": loaded,
                "dedupe_key_count": len(self._canonical_resolution_index),
            },
        )

    async def _discovery_loop(self, discovery_client: GammaDiscoveryClient) -> None:
        poll_seq = 1
        while True:
            poll_seq += 1
            await self._discover_once(discovery_client, poll_seq=poll_seq)
            await asyncio.sleep(self.config.gamma_poll_interval_seconds)

    async def _discover_once(self, discovery_client: GammaDiscoveryClient, *, poll_seq: int) -> None:
        if not self._before_http_request("gamma_resolution_discovery"):
            return
        try:
            now = datetime.now(tz=UTC)
            active_target_slugs = generate_target_slugs(
                now=now,
                prefix=self.config.market_slug_prefix,
                interval_seconds=self.config.market_interval_seconds,
                lookback_intervals=self.config.market_discovery_lookback_intervals,
                lookahead_intervals=self.config.market_discovery_lookahead_intervals,
            )
            closed_target_slugs = generate_target_slugs(
                now=now,
                prefix=self.config.market_slug_prefix,
                interval_seconds=self.config.market_interval_seconds,
                lookback_intervals=self.config.resolution_closed_lookback_intervals,
                lookahead_intervals=0,
            )
            active_items = await discovery_client.fetch_markets_for_slugs(active_target_slugs, closed=False)
            closed_items = await discovery_client.fetch_markets_for_slugs(closed_target_slugs, closed=True)
        except httpx.HTTPError as exc:
            self._record_http_error(
                endpoint="gamma_resolution_discovery",
                request_method="GET",
                request_path="/markets",
                error=exc,
            )
            return
        self._observe_http_success("gamma_resolution_discovery")
        discovered = sort_markets_by_open(
            candidate_target_markets(
                active_items,
                target_slugs=active_target_slugs,
                now=now,
                stale_grace_seconds=self.config.market_stale_grace_seconds,
            )
            + candidate_target_markets(
                closed_items,
                target_slugs=closed_target_slugs,
                now=now,
                stale_grace_seconds=self.config.market_stale_grace_seconds,
                allow_stale=True,
            )
        )
        self._remember_markets(discovered)
        self._refresh_tracked_markets(reason="gamma_discovery_poll", poll_seq=poll_seq)

    async def _rest_state_loop(
        self,
        discovery_client: GammaDiscoveryClient,
        *,
        page_client: httpx.AsyncClient,
    ) -> None:
        while True:
            await self._poll_rest_once(discovery_client, page_client=page_client)
            await asyncio.sleep(self.config.resolution_rest_poll_interval_seconds)

    async def _poll_rest_once(
        self,
        discovery_client: GammaDiscoveryClient,
        *,
        page_client: httpx.AsyncClient,
    ) -> None:
        self._refresh_tracked_markets(reason="boundary_tick")
        targets = self._rest_poll_targets()
        if not targets:
            self._advance_writers()
            return
        for market in targets:
            if not self._before_http_request("gamma_resolution_market_state"):
                continue
            try:
                payload = await discovery_client.fetch_market_by_id(market.market_id)
            except httpx.HTTPError as exc:
                self._record_http_error(
                    endpoint="gamma_resolution_market_state",
                    request_method="GET",
                    request_path=f"/markets/{market.market_id}",
                    error=exc,
                )
                continue
            self._observe_http_success("gamma_resolution_market_state")
            if payload is not None:
                self._handle_rest_market_state(market, payload)
                if self._lookup_canonical_resolution(market=market) is not None:
                    self._pending_page_backfill_market_ids.discard(market.identity)
                    continue
                rest_record = self._build_rest_canonical_resolution_record(market, payload=payload)
                if rest_record is not None:
                    self._record_canonical_resolution(
                        market,
                        rest_record,
                        add_rest_reconcile=False,
                        lifecycle_event="market_resolution_recorded_from_rest",
                    )
                    continue
                if str(payload.get("umaResolutionStatus")).lower() == "resolved":
                    self._pending_page_backfill_market_ids.add(market.identity)
                    await self._attempt_page_resolution_backfill(
                        page_client,
                        market,
                        rest_payload=payload,
                    )

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            self._refresh_tracked_markets(reason="boundary_tick")
            if not self._tracked_market_ids:
                self._advance_writers()
                await asyncio.sleep(self.config.market_boundary_check_interval_seconds)
                continue
            connection_id = generate_connection_id("poly-resolution")
            try:
                await self._consume_websocket(connection_id)
                delay = self.config.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._write_lifecycle("websocket_error", {"connection_id": connection_id, "error": repr(exc)})
                await asyncio.sleep(delay + random.random() * self.config.reconnect_jitter_seconds)
                delay = min(self.config.reconnect_max_seconds, max(self.config.reconnect_min_seconds, delay * 2))

    async def _consume_websocket(self, connection_id: str) -> None:
        tracked_asset_ids = self._tracked_asset_ids()
        if not tracked_asset_ids:
            return
        self._write_lifecycle("websocket_connecting", {"connection_id": connection_id, "ws_url": self.config.websocket_url})
        async with connect(
            self.config.websocket_url,
            open_timeout=self.config.connect_open_timeout_seconds,
            close_timeout=self.config.connect_close_timeout_seconds,
            max_queue=self.config.websocket_max_queue,
            ping_interval=None,
        ) as ws:
            await ws.send(
                orjson.dumps(
                    {
                        "assets_ids": tracked_asset_ids,
                        "type": "market",
                        "custom_feature_enabled": True,
                        "initial_dump": True,
                    }
                ).decode("utf-8")
            )
            self._connected_asset_ids = list(tracked_asset_ids)
            self._pending_subscription_change = None
            self._last_activity_ns = utc_now_ns()
            self._write_lifecycle(
                "reconnect_rebuilt_active_market_set",
                {
                    "connection_id": connection_id,
                    "tracked_market_ids": [market.identity for market in self._tracked_markets()],
                    "tracked_market_slugs": [market.market_slug for market in self._tracked_markets()],
                    "tracked_asset_ids": tracked_asset_ids,
                },
            )
            self._write_lifecycle(
                "websocket_connected",
                {
                    "connection_id": connection_id,
                    "ws_url": self.config.websocket_url,
                    "tracked_market_ids": [market.identity for market in self._tracked_markets()],
                    "tracked_market_slugs": [market.market_slug for market in self._tracked_markets()],
                },
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws, connection_id), name="polymarket:resolution:heartbeat")
            local_seq = 0
            try:
                while True:
                    self._refresh_tracked_markets(reason="boundary_tick")
                    await self._apply_pending_change(ws, connection_id)
                    self._advance_writers()
                    try:
                        message = await asyncio.wait_for(
                            ws.recv(),
                            timeout=max(0.25, self.config.market_boundary_check_interval_seconds),
                        )
                    except TimeoutError:
                        continue
                    self._last_activity_ns = utc_now_ns()
                    if message in {"PONG", "{}", b"{}"}:
                        continue
                    payload = orjson.loads(message)
                    for item in _iter_payload_items(payload):
                        local_seq += 1
                        self._handle_ws_payload(item, connection_id=connection_id, local_seq=local_seq)
            finally:
                cancelled_during_cleanup = await cancel_and_await(heartbeat_task)
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError
        self._connected_asset_ids = []
        self._write_lifecycle("websocket_closed", {"connection_id": connection_id, "ws_url": self.config.websocket_url})

    async def _heartbeat_loop(self, ws: Any, connection_id: str) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            self._advance_writers()
            if utc_now_ns() - self._last_activity_ns > int(
                self.config.heartbeat_health_timeout_seconds * 1_000_000_000
            ):
                self._write_lifecycle(
                    "connection_health_timeout",
                    {"connection_id": connection_id, "health_timeout_seconds": self.config.heartbeat_health_timeout_seconds},
                )
                await ws.close()
                return
            await ws.send("PING")

    async def _apply_pending_change(self, ws: Any, connection_id: str) -> None:
        if self._pending_subscription_change is None:
            return
        change = self._pending_subscription_change
        if change.added_asset_ids:
            await ws.send(
                orjson.dumps(
                    {
                        "operation": "subscribe",
                        "assets_ids": change.added_asset_ids,
                        "custom_feature_enabled": True,
                        "initial_dump": True,
                    }
                ).decode("utf-8")
            )
        if change.removed_asset_ids:
            await ws.send(
                orjson.dumps({"operation": "unsubscribe", "assets_ids": change.removed_asset_ids}).decode("utf-8")
            )
        new_asset_ids = [asset_id for asset_id in self._connected_asset_ids if asset_id not in set(change.removed_asset_ids)]
        for asset_id in change.added_asset_ids:
            if asset_id not in new_asset_ids:
                new_asset_ids.append(asset_id)
        self._connected_asset_ids = new_asset_ids
        self._pending_subscription_change = None
        self._write_lifecycle(
            "subscription_updated_with_active_market_set",
            {
                "connection_id": connection_id,
                "reason": change.reason,
                "asset_ids_added": change.added_asset_ids,
                "asset_ids_removed": change.removed_asset_ids,
                "tracked_market_ids": change.tracked_market_ids,
                "tracked_market_slugs": change.tracked_market_slugs,
            },
        )

    def _handle_ws_payload(self, payload: dict[str, Any], *, connection_id: str, local_seq: int) -> None:
        event_type = str(payload.get("event_type") or "")
        if event_type != "market_resolved":
            return
        market = self._match_market_for_resolution_payload(payload)
        canonical_record: dict[str, Any] | None = None
        existing_canonical: dict[str, Any] | None = None
        canonical_action = "unmatched_market"
        if market is not None:
            canonical_record = self._build_canonical_resolution_record(
                market,
                payload=payload,
                connection_id=connection_id,
                local_seq=local_seq,
            )
            if canonical_record is None:
                canonical_action = "invalid_resolution_signal"
            else:
                canonical_action = self._record_canonical_resolution(
                    market,
                    canonical_record,
                    add_rest_reconcile=True,
                    lifecycle_event="market_resolution_recorded",
                )
                existing_canonical = self._lookup_canonical_resolution(market=market)
        history_recv_ts_ns = utc_now_ns()
        resolved_ts_ns = _timestamp_value_to_ns(payload.get("timestamp"))
        self._history_writer.write(
            {
                "record_type": "market_resolution_signal",
                "signal_source": _RESOLUTION_SOURCE,
                "source": "polymarket",
                "channel": "market",
                "connection_id": connection_id,
                "local_msg_seq": local_seq,
                "ws_url": self.config.websocket_url,
                "recv_ts_ns": history_recv_ts_ns,
                "recv_ts_iso": ns_to_iso(history_recv_ts_ns),
                "recv_ts_source": recv_timestamp_source(),
                "event_type": event_type,
                "canonical_action": canonical_action,
                "matched_market_id": market.market_id if market is not None else None,
                "matched_market_slug": market.market_slug if market is not None else None,
                "matched_condition_id": market.condition_id if market is not None else None,
                "winning_asset_id": str(payload.get("winning_asset_id") or ""),
                "winning_outcome": payload.get("winning_outcome"),
                "resolved_ts_ns": resolved_ts_ns,
                "resolved_ts_iso": ns_to_iso(resolved_ts_ns)
                if resolved_ts_ns is not None
                else None,
                "raw_resolution_payload": payload,
                "existing_canonical_summary": self._canonical_summary(existing_canonical),
                **(_top_level_market_fields(market) if market is not None else {}),
                **self._runtime.static_fields(),
            }
        )

    def _handle_rest_market_state(self, market: SelectedMarket, payload: dict[str, Any]) -> None:
        recv_ts_ns = utc_now_ns()
        winner_candidate_asset_id = self._rest_winner_asset_id(payload)
        winner_candidate_outcome = self._rest_winner_outcome(payload)
        winner_candidate_side = normalize_outcome_side(winner_candidate_outcome)
        closed = payload.get("closed")
        active = payload.get("active")
        uma_resolution_status = payload.get("umaResolutionStatus")
        rest_record = {
            "record_type": "market_resolution_state_observation",
            "signal_source": _REST_STATE_SOURCE,
            "source": "polymarket",
            "api": "gamma",
            "endpoint": "market_state",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "market_id": market.market_id,
            "market_slug": market.market_slug,
            "condition_id": market.condition_id,
            "closed": closed,
            "active": active,
            "closed_time": payload.get("closedTime"),
            "resolved_by": payload.get("resolvedBy"),
            "uma_resolution_status": uma_resolution_status,
            "resolution_source_url": payload.get("resolutionSource"),
            "winner_candidate_asset_id": winner_candidate_asset_id,
            "winner_candidate_outcome": winner_candidate_outcome,
            "winner_candidate_side": winner_candidate_side,
            "winner_candidate_is_definitive": winner_candidate_asset_id is not None or winner_candidate_outcome is not None,
            "raw_market_state_payload": payload,
            "canonical_resolution_recorded": self._lookup_canonical_resolution(market=market) is not None,
            **_top_level_market_fields(market),
            **self._runtime.static_fields(),
        }
        self._latest_rest_state_by_market_identity[market.identity] = rest_record
        self._history_writer.write(rest_record)
        if (
            self._lookup_canonical_resolution(market=market) is None
            and str(uma_resolution_status).lower() == "resolved"
            and winner_candidate_asset_id is None
            and winner_candidate_outcome is None
            and market.identity not in self._rest_resolved_without_canonical_market_ids
        ):
            self._rest_resolved_without_canonical_market_ids.add(market.identity)
            self._write_lifecycle(
                "market_resolution_rest_resolved_without_canonical_winner",
                {
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "uma_resolution_status": uma_resolution_status,
                    "closed": closed,
                },
            )
        if market.identity in self._pending_rest_reconcile_market_ids and str(uma_resolution_status).lower() == "resolved":
            self._pending_rest_reconcile_market_ids.remove(market.identity)
            self._write_lifecycle(
                "market_resolution_rest_reconciled",
                {
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "uma_resolution_status": uma_resolution_status,
                    "closed": closed,
                },
            )

    def _resolve_market_outcome(
        self,
        market: SelectedMarket,
        *,
        winning_asset_id: str | None = None,
        winning_outcome: str | None = None,
        resolved_side: str | None = None,
        mapping_source: str,
        authoritative: bool,
    ) -> _ResolvedMarketOutcome | None:
        normalized_winning_asset_id = str(winning_asset_id or "").strip() or None
        normalized_winning_outcome = None if winning_outcome is None else str(winning_outcome).strip() or None
        normalized_resolved_side = normalize_outcome_side(resolved_side)
        mapped_outcome = market.outcome_map.get(normalized_winning_asset_id or "")
        mapped_side = normalize_outcome_side(mapped_outcome)
        if mapped_side is not None and normalized_winning_asset_id is not None and mapped_outcome is not None:
            return _ResolvedMarketOutcome(
                winning_asset_id=normalized_winning_asset_id,
                winning_outcome=mapped_outcome,
                winning_outcome_raw=normalized_winning_outcome,
                winning_asset_id_in_market_map=True,
                resolved_side=mapped_side,
                resolved_side_mapping_source="winning_asset_id_mapping",
                resolved_side_authoritative=True,
            )
        side = normalized_resolved_side or normalize_outcome_side(normalized_winning_outcome)
        if side is None:
            return None
        for asset_id, outcome in market.outcome_map.items():
            if normalize_outcome_side(outcome) != side:
                continue
            return _ResolvedMarketOutcome(
                winning_asset_id=asset_id,
                winning_outcome=outcome,
                winning_outcome_raw=normalized_winning_outcome,
                winning_asset_id_in_market_map=normalized_winning_asset_id is not None and asset_id == normalized_winning_asset_id,
                resolved_side=side,
                resolved_side_mapping_source=mapping_source,
                resolved_side_authoritative=authoritative,
            )
        return None

    def _match_market_for_resolution_payload(self, payload: dict[str, Any]) -> SelectedMarket | None:
        payload_market = str(payload.get("market") or "")
        if payload_market:
            for market in self._known_markets.values():
                if payload_market in {market.condition_id, market.market_id, market.market_slug}:
                    return market
        winning_asset_id = str(payload.get("winning_asset_id") or "")
        if winning_asset_id:
            matches = [market for market in self._known_markets.values() if winning_asset_id in market.outcome_map]
            if len(matches) == 1:
                return matches[0]
            if matches:
                return newest_market(matches)
        asset_ids = {
            str(value)
            for value in payload.get("assets_ids", [])
            if value is not None
        }
        if asset_ids:
            matches = [
                market
                for market in self._known_markets.values()
                if asset_ids <= set(market.asset_ids)
            ]
            if len(matches) == 1:
                return matches[0]
            if matches:
                return newest_market(matches)
        return None

    def _build_canonical_resolution_record(
        self,
        market: SelectedMarket,
        *,
        payload: dict[str, Any],
        connection_id: str,
        local_seq: int,
    ) -> dict[str, Any] | None:
        recv_ts_ns = utc_now_ns()
        winning_outcome = None if payload.get("winning_outcome") is None else str(payload.get("winning_outcome"))
        outcome = self._resolve_market_outcome(
            market,
            winning_asset_id=str(payload.get("winning_asset_id") or ""),
            winning_outcome=winning_outcome,
            mapping_source="winning_outcome_fallback",
            authoritative=False,
        )
        if outcome is None:
            return None
        resolved_ts_ns = _timestamp_value_to_ns(payload.get("timestamp")) or recv_ts_ns
        return self._build_resolution_record(
            market,
            recv_ts_ns=recv_ts_ns,
            resolved_ts_ns=resolved_ts_ns,
            resolution_source=_RESOLUTION_SOURCE,
            winning=outcome,
            raw_resolution_payload=payload,
            latest_rest_state=self._latest_rest_state_by_market_identity.get(market.identity),
            extra_fields={
                "channel": "market",
                "connection_id": connection_id,
                "local_msg_seq": local_seq,
            },
        )

    def _build_rest_canonical_resolution_record(
        self,
        market: SelectedMarket,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        winning = self._resolve_market_outcome(
            market,
            winning_asset_id=self._rest_winner_asset_id(payload),
            winning_outcome=self._rest_winner_outcome(payload),
            mapping_source="winning_outcome_fallback",
            authoritative=True,
        )
        if winning is None:
            return None
        recv_ts_ns = utc_now_ns()
        resolved_ts_ns = _timestamp_value_to_ns(payload.get("closedTime")) or recv_ts_ns
        return self._build_resolution_record(
            market,
            recv_ts_ns=recv_ts_ns,
            resolved_ts_ns=resolved_ts_ns,
            resolution_source=_REST_RESOLUTION_SOURCE,
            winning=winning,
            raw_resolution_payload=payload,
            latest_rest_state=self._latest_rest_state_by_market_identity.get(market.identity),
            extra_fields={
                "channel": "rest",
                "api": "gamma",
                "endpoint": "market_state",
            },
        )

    def _build_page_canonical_resolution_record(
        self,
        market: SelectedMarket,
        *,
        page_resolution: dict[str, Any],
        resolved_ts_ns: int,
        recv_ts_ns: int,
    ) -> dict[str, Any] | None:
        winning = self._resolve_market_outcome(
            market,
            winning_outcome=str(page_resolution.get("winning_outcome") or ""),
            resolved_side=str(page_resolution.get("resolved_side") or ""),
            mapping_source="page_resolution_side",
            authoritative=True,
        )
        if winning is None:
            return None
        return self._build_resolution_record(
            market,
            recv_ts_ns=recv_ts_ns,
            resolved_ts_ns=resolved_ts_ns,
            resolution_source=_PAGE_RESOLUTION_SOURCE,
            winning=winning,
            raw_resolution_payload=page_resolution,
            latest_rest_state=self._latest_rest_state_by_market_identity.get(market.identity),
            extra_fields={
                "channel": "page",
                "page_resolution_open_price": page_resolution.get("open_price"),
                "page_resolution_close_price": page_resolution.get("close_price"),
                "page_resolution_source_detail": page_resolution.get("source_detail"),
                "page_result_outcome": page_resolution.get("page_result_outcome"),
            },
        )

    def _build_resolution_record(
        self,
        market: SelectedMarket,
        *,
        recv_ts_ns: int,
        resolved_ts_ns: int,
        resolution_source: str,
        winning: _ResolvedMarketOutcome,
        raw_resolution_payload: Any,
        latest_rest_state: dict[str, Any] | None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "record_type": "market_resolution",
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "resolved_ts_ns": resolved_ts_ns,
            "resolved_ts_iso": ns_to_iso(resolved_ts_ns),
            "resolution_source": resolution_source,
            "market_id": market.market_id,
            "market_slug": market.market_slug,
            "condition_id": market.condition_id,
            "winning_asset_id": winning.winning_asset_id,
            "winning_outcome": winning.winning_outcome,
            "winning_outcome_raw": winning.winning_outcome_raw,
            "winning_asset_id_in_market_map": winning.winning_asset_id_in_market_map,
            "resolved_side": winning.resolved_side,
            "resolved_side_mapping_source": winning.resolved_side_mapping_source,
            "resolved_side_authoritative": winning.resolved_side_authoritative,
            "raw_resolution_payload": raw_resolution_payload,
            "rest_state_reconciled": latest_rest_state is not None
            and str(latest_rest_state.get("uma_resolution_status")).lower() == "resolved",
            "latest_rest_state": latest_rest_state,
            **_top_level_market_fields(market),
            **self._runtime.static_fields(),
        }
        if extra_fields:
            record.update(extra_fields)
        return record

    def _record_canonical_resolution(
        self,
        market: SelectedMarket,
        record: dict[str, Any],
        *,
        add_rest_reconcile: bool,
        lifecycle_event: str,
    ) -> str:
        existing_canonical = self._lookup_canonical_resolution(market=market)
        if existing_canonical is None:
            self._resolution_writer.write(record)
            self._index_canonical_resolution(record)
            if add_rest_reconcile:
                self._pending_rest_reconcile_market_ids.add(market.identity)
            self._pending_page_backfill_market_ids.discard(market.identity)
            self._write_lifecycle(
                lifecycle_event,
                {
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "winning_asset_id": record.get("winning_asset_id"),
                    "winning_outcome": record.get("winning_outcome"),
                    "resolved_side": record.get("resolved_side"),
                    "resolution_source": record.get("resolution_source"),
                },
            )
            self._refresh_tracked_markets(reason="resolution_recorded")
            return "recorded"
        if self._canonical_matches(existing_canonical, record):
            self._pending_page_backfill_market_ids.discard(market.identity)
            self._write_lifecycle(
                "market_resolution_duplicate_observed",
                {
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "winning_asset_id": record.get("winning_asset_id"),
                    "winning_outcome": record.get("winning_outcome"),
                    "resolution_source": record.get("resolution_source"),
                },
            )
            return "duplicate_ignored"
        self._write_lifecycle(
            "market_resolution_conflict_observed",
            {
                "market_id": market.market_id,
                "market_slug": market.market_slug,
                "existing_winning_asset_id": existing_canonical.get("winning_asset_id"),
                "incoming_winning_asset_id": record.get("winning_asset_id"),
                "existing_resolution_source": existing_canonical.get("resolution_source"),
                "incoming_resolution_source": record.get("resolution_source"),
            },
        )
        return "conflict_ignored"

    async def _attempt_page_resolution_backfill(
        self,
        client: httpx.AsyncClient,
        market: SelectedMarket,
        *,
        rest_payload: dict[str, Any] | None,
        historical_recv_ts_ns: int | None = None,
        historical_resolved_ts_ns: int | None = None,
    ) -> bool:
        if self._lookup_canonical_resolution(market=market) is not None:
            self._pending_page_backfill_market_ids.discard(market.identity)
            return True
        if not self._before_http_request("market_page_resolution_backfill"):
            return False
        page_url = f"{self.config.market_page_base_url}/{market.market_slug}"
        try:
            outcome = await request_with_retries(
                client,
                "GET",
                page_url,
                max_retries=self.config.http_max_retries,
                retry_backoff_seconds=self.config.http_retry_backoff_seconds,
            )
        except httpx.HTTPError as exc:
            self._record_http_error(
                endpoint="market_page_resolution_backfill",
                request_method="GET",
                request_path=f"/event/{market.market_slug}",
                error=exc,
            )
            return False
        self._observe_http_success("market_page_resolution_backfill")
        response = outcome.response
        page_resolution = _extract_page_resolution(market, html=response.text)
        if page_resolution is None:
            self._write_lifecycle(
                "market_resolution_page_window_not_found",
                {
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "page_url": page_url,
                },
            )
            return False
        recv_ts_ns = historical_recv_ts_ns or utc_now_ns()
        resolved_ts_ns = historical_resolved_ts_ns or historical_recv_ts_ns or utc_now_ns()
        record = self._build_page_canonical_resolution_record(
            market,
            page_resolution=page_resolution,
            recv_ts_ns=recv_ts_ns,
            resolved_ts_ns=resolved_ts_ns,
        )
        if record is None:
            return False
        self._record_canonical_resolution(
            market,
            record,
            add_rest_reconcile=False,
            lifecycle_event="market_resolution_recorded_from_page",
        )
        return True

    def _remember_markets(self, markets: list[SelectedMarket]) -> None:
        for market in markets:
            self._known_markets[market.identity] = market

    def _refresh_tracked_markets(self, *, reason: str, poll_seq: int | None = None) -> None:
        now = datetime.now(tz=UTC)
        previous_ids = set(self._tracked_market_ids)
        tracked_markets = sort_markets_by_open(
            [
                market
                for market in self._known_markets.values()
                if self._market_should_be_tracked(market, now=now)
            ]
        )
        tracked_ids = {market.identity for market in tracked_markets}
        self._tracked_market_ids = tracked_ids
        added_markets = [market for market in tracked_markets if market.identity not in previous_ids]
        removed_markets = [market for market in self._markets_for_ids(previous_ids) if market.identity not in tracked_ids]
        for market in added_markets:
            self._write_lifecycle(
                "market_resolution_tracking_started",
                {
                    "reason": reason,
                    "poll_seq": poll_seq,
                    **_top_level_market_fields(market),
                },
            )
        for market in removed_markets:
            self._write_lifecycle(
                "market_resolution_tracking_stopped",
                {
                    "reason": reason,
                    "poll_seq": poll_seq,
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "condition_id": market.condition_id,
                    "resolution_recorded": self._lookup_canonical_resolution(market=market) is not None,
                },
            )
        connected_assets = set(self._connected_asset_ids)
        tracked_asset_ids = set(self._tracked_asset_ids())
        if connected_assets != tracked_asset_ids:
            self._pending_subscription_change = _ResolutionSubscriptionChange(
                added_asset_ids=sorted(tracked_asset_ids - connected_assets),
                removed_asset_ids=sorted(connected_assets - tracked_asset_ids),
                reason=reason,
                tracked_market_ids=[market.identity for market in tracked_markets],
                tracked_market_slugs=[market.market_slug for market in tracked_markets],
            )

    def _market_should_be_tracked(self, market: SelectedMarket, *, now: datetime) -> bool:
        if self._lookup_canonical_resolution(market=market) is not None:
            return False
        close_epoch = market_close_epoch(market, interval_seconds=self.config.market_interval_seconds)
        if close_epoch is None:
            return False
        latest_rest_state = self._latest_rest_state_by_market_identity.get(market.identity)
        if latest_rest_state is not None and str(latest_rest_state.get("uma_resolution_status")).lower() == "resolved":
            resolved_observed_ns = int(latest_rest_state.get("recv_ts_ns") or 0)
            grace_ns = int(self.config.resolution_rest_resolved_grace_seconds * 1_000_000_000)
            close_age_ns = utc_now_ns() - (close_epoch * 1_000_000_000)
            if close_age_ns > grace_ns or (resolved_observed_ns and utc_now_ns() - resolved_observed_ns > grace_ns):
                return False
        return now.astimezone(UTC).timestamp() >= close_epoch - self.config.resolution_watch_preclose_seconds

    def _tracked_markets(self) -> list[SelectedMarket]:
        return self._markets_for_ids(self._tracked_market_ids)

    def _tracked_asset_ids(self) -> list[str]:
        asset_ids: list[str] = []
        for market in self._tracked_markets():
            for asset_id in market.asset_ids:
                if asset_id not in asset_ids:
                    asset_ids.append(asset_id)
        return asset_ids

    def _rest_poll_targets(self) -> list[SelectedMarket]:
        market_ids = (
            self._tracked_market_ids
            | self._pending_rest_reconcile_market_ids
            | self._pending_page_backfill_market_ids
        )
        return self._markets_for_ids(market_ids)

    def _markets_for_ids(self, market_ids: set[str] | list[str]) -> list[SelectedMarket]:
        return sort_markets_by_open(
            [self._known_markets[market_id] for market_id in market_ids if market_id in self._known_markets]
        )

    def _lookup_canonical_resolution(
        self,
        *,
        market: SelectedMarket | None = None,
        market_id: str | None = None,
        condition_id: str | None = None,
        market_slug: str | None = None,
    ) -> dict[str, Any] | None:
        for key in _dedupe_keys(
            market=market,
            market_id=market_id,
            condition_id=condition_id,
            market_slug=market_slug,
        ):
            record = self._canonical_resolution_index.get(key)
            if record is not None:
                return record
        return None

    def _index_canonical_resolution(self, record: dict[str, Any]) -> None:
        for key in _dedupe_keys(
            market_id=str(record.get("market_id") or ""),
            condition_id=str(record.get("condition_id") or ""),
            market_slug=str(record.get("market_slug") or ""),
        ):
            self._canonical_resolution_index[key] = record

    @staticmethod
    def _canonical_matches(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
        return (
            str(existing.get("winning_asset_id") or "") == str(incoming.get("winning_asset_id") or "")
            and str(existing.get("winning_outcome") or "") == str(incoming.get("winning_outcome") or "")
            and str(existing.get("resolved_side") or "") == str(incoming.get("resolved_side") or "")
        )

    @staticmethod
    def _canonical_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "market_id": record.get("market_id"),
            "market_slug": record.get("market_slug"),
            "condition_id": record.get("condition_id"),
            "winning_asset_id": record.get("winning_asset_id"),
            "winning_outcome": record.get("winning_outcome"),
            "resolved_side": record.get("resolved_side"),
            "resolved_ts_iso": record.get("resolved_ts_iso"),
            "resolution_source": record.get("resolution_source"),
        }

    @staticmethod
    def _rest_winner_asset_id(payload: dict[str, Any]) -> str | None:
        for field_name in ("winningAssetId", "winning_asset_id", "winner"):
            value = payload.get(field_name)
            if value is not None and str(value).strip():
                return str(value)
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                winner_flag = token.get("winner")
                if winner_flag is True or str(winner_flag).strip().lower() in {"true", "1", "yes"}:
                    token_id = token.get("tokenId") or token.get("asset_id") or token.get("id")
                    if token_id is not None:
                        return str(token_id)
        return None

    @staticmethod
    def _rest_winner_outcome(payload: dict[str, Any]) -> str | None:
        for field_name in ("winningOutcome", "winning_outcome"):
            value = payload.get(field_name)
            if value is not None and str(value).strip():
                return str(value)
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                winner_flag = token.get("winner")
                if winner_flag is True or str(winner_flag).strip().lower() in {"true", "1", "yes"}:
                    outcome = token.get("outcome")
                    if outcome is not None and str(outcome).strip():
                        return str(outcome)
        return None

    def _advance_writers(self) -> None:
        now_ns = utc_now_ns()
        self._history_writer.advance_time(now_ns=now_ns)
        self._resolution_writer.advance_time(now_ns=now_ns)
        self._lifecycle_writer.advance_time(now_ns=now_ns)

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
        self.logger.warning("Polymarket resolution HTTP request failed", extra=details)
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
        self._lifecycle_writer.write(
            lifecycle_event(
                event=event,
                details=self._runtime.lifecycle_fields(
                    include_config_snapshot=event == "service_start",
                    extra_fields=details,
                ),
            )
        )


def _first_zstd_jsonl_record(path: Path) -> dict[str, Any] | None:
    for record in iter_zstd_jsonl_with_options(
        path,
        allow_truncated_stream=True,
        allow_partial_final_line=True,
    ):
        if isinstance(record, dict):
            return record
    return None


def _selected_market_from_l2_record(record: dict[str, Any]) -> SelectedMarket | None:
    market_slug = str(record.get("selected_market_slug") or "").strip()
    if not market_slug:
        return None
    asset_ids = [str(value) for value in record.get("selected_asset_ids", []) if value is not None]
    outcomes = [str(value) for value in record.get("selected_outcomes", []) if value is not None]
    if len(asset_ids) != 2 or len(outcomes) != 2:
        return None
    open_epoch = record.get("market_open_s")
    try:
        start_time_epoch = int(open_epoch) if open_epoch is not None else None
    except Exception:
        start_time_epoch = None
    market_open_iso = record.get("market_open_iso")
    market_close_iso = record.get("market_close_iso")
    return SelectedMarket(
        market_id=str(record.get("selected_market_id") or record.get("selected_condition_id") or market_slug),
        condition_id=str(record.get("selected_condition_id") or record.get("selected_market_id") or ""),
        market_slug=market_slug,
        question=str(record.get("question") or market_slug),
        asset_ids=asset_ids,
        outcomes=outcomes,
        series_slug="btc-up-or-down-5m",
        end_time_iso=str(market_close_iso) if market_close_iso is not None else None,
        start_time_epoch=start_time_epoch,
        selection_reason="l2_backfill",
        raw_market={
            "eventStartTime": market_open_iso,
            "endDate": market_close_iso,
        },
    )


def load_recorded_markets_from_finalized_l2(*, root: Path, target_date: str) -> list[SelectedMarket]:
    manifest_dir = root / "manifests" / "polymarket" / "btc_updown_5m" / target_date
    markets_by_slug: dict[str, SelectedMarket] = {}
    if not manifest_dir.exists():
        return []
    for manifest_path in sorted(manifest_dir.glob("*.l2.manifest.json")):
        try:
            manifest = orjson.loads(manifest_path.read_bytes())
        except orjson.JSONDecodeError:
            continue
        if not isinstance(manifest, dict):
            continue
        if str(manifest.get("quality_class")) != "finalized_clean":
            continue
        raw_file = Path(str(manifest.get("file_path") or ""))
        if raw_file.is_absolute():
            candidate_paths = [raw_file]
        else:
            candidate_paths = [root / raw_file, root.parent / raw_file]
            if raw_file.parts and root.name == raw_file.parts[0]:
                candidate_paths.insert(0, root.parent / raw_file)
        resolved_raw_file = next((path for path in candidate_paths if path.exists()), None)
        if resolved_raw_file is None:
            continue
        first_record = _first_zstd_jsonl_record(resolved_raw_file)
        if first_record is None:
            continue
        market = _selected_market_from_l2_record(first_record)
        if market is None:
            continue
        markets_by_slug.setdefault(market.market_slug, market)
    return sort_markets_by_open(list(markets_by_slug.values()))


def load_latest_rest_states_by_slug(
    *,
    root: Path,
    target_date: str,
    finalized_only: bool = False,
    tail_tolerant: bool = True,
) -> dict[str, dict[str, Any]]:
    from .resolution_reader import PolymarketResolutionReader

    reader = PolymarketResolutionReader(root=root)
    latest_by_slug: dict[str, dict[str, Any]] = {}
    for record in reader.iter_history(
        target_date=target_date,
        finalized_only=finalized_only,
        tail_tolerant=tail_tolerant,
    ):
        if str(record.get("record_type")) != "market_resolution_state_observation":
            continue
        slug = str(record.get("market_slug") or record.get("matched_market_slug") or "").strip()
        if not slug:
            continue
        existing = latest_by_slug.get(slug)
        if existing is None or int(record.get("recv_ts_ns") or 0) >= int(existing.get("recv_ts_ns") or 0):
            latest_by_slug[slug] = record
    return latest_by_slug


async def backfill_missing_resolutions_for_date(
    *,
    config: PolymarketRecorderConfig,
    target_date: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    service = PolymarketResolutionRecorderService(config, logger=logger)
    service._write_lifecycle = lambda event, details=None: None  # type: ignore[method-assign]
    service._bootstrap_existing_resolutions()
    recorded_markets = load_recorded_markets_from_finalized_l2(root=config.out, target_date=target_date)
    service._remember_markets(recorded_markets)
    latest_rest_by_slug = load_latest_rest_states_by_slug(root=config.out, target_date=target_date)
    for market in recorded_markets:
        latest_rest_state = latest_rest_by_slug.get(market.market_slug)
        if latest_rest_state is not None:
            service._latest_rest_state_by_market_identity[market.identity] = latest_rest_state

    attempted = 0
    recorded = 0
    already_present = 0
    skipped_current_hour = 0
    unresolved: list[str] = []
    current_hour_start_ns = int(datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0).timestamp() * 1_000_000_000)

    async with build_async_http_client(
        timeout_seconds=config.http_timeout_seconds,
        headers=_PAGE_HEADERS,
        max_connections=config.http_max_connections,
        max_keepalive_connections=config.http_max_keepalive_connections,
        keepalive_expiry_seconds=config.http_keepalive_expiry_seconds,
        trust_env=config.http_trust_env,
    ) as page_client:
        for market in recorded_markets:
            if service._lookup_canonical_resolution(market=market) is not None:
                already_present += 1
                continue
            latest_rest_state = latest_rest_by_slug.get(market.market_slug)
            resolved_history_ns = int(latest_rest_state.get("recv_ts_ns") or 0) if latest_rest_state is not None else 0
            if resolved_history_ns >= current_hour_start_ns and resolved_history_ns != 0:
                skipped_current_hour += 1
                unresolved.append(market.market_slug)
                continue
            close_epoch = market_close_epoch(market, interval_seconds=config.market_interval_seconds)
            close_ts_ns = close_epoch * 1_000_000_000 if close_epoch is not None else 0
            if close_ts_ns >= current_hour_start_ns and close_ts_ns != 0:
                skipped_current_hour += 1
                unresolved.append(market.market_slug)
                continue
            attempted += 1
            success = await service._attempt_page_resolution_backfill(
                page_client,
                market,
                rest_payload=latest_rest_state,
                historical_recv_ts_ns=resolved_history_ns or close_ts_ns or None,
                historical_resolved_ts_ns=resolved_history_ns or close_ts_ns or None,
            )
            if success and service._lookup_canonical_resolution(market=market) is not None:
                recorded += 1
            else:
                unresolved.append(market.market_slug)

    service._history_writer.close()
    service._resolution_writer.close()
    service._lifecycle_writer.close()
    return {
        "target_date": target_date,
        "recorded_market_count": len(recorded_markets),
        "already_present": already_present,
        "attempted_backfill": attempted,
        "recorded_backfill": recorded,
        "skipped_current_hour": skipped_current_hour,
        "remaining_unresolved": len(unresolved),
        "remaining_unresolved_slugs": unresolved,
    }
