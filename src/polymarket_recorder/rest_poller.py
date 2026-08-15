from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
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
from market_recorders.time_utils import (
    build_capture_timing,
    build_clock_probe,
    build_epoch_timestamp_fields,
    epoch_value_to_ns,
    normalize_epoch_value,
    parse_iso8601_to_ns,
    resolve_epoch_unit,
    recv_timestamp_source,
    utc_now_ns,
)

from .config import PolymarketRecorderConfig
from .discovery import GammaDiscoveryClient
from .lifecycle import lifecycle_event
from .utils import (
    SelectedMarket,
    build_polymarket_source_timestamps,
    detection_metadata,
    market_active_window_fields,
    market_has_expired,
    market_is_active,
    newest_market,
    session_common_fields,
    sort_markets_by_open,
)
from .writers import PolymarketSessionWriters

_METADATA_ONLY_ENDPOINTS = {
    "gamma_discovery",
    "gamma_selected_market",
    "gamma_selected_event",
    "gamma_event_detail",
    "holders",
    "market_positions",
    "market_open_interest",
    "event_live_volume",
    "market_rewards",
    "clob_fee_rate",
    "clob_tick_size",
}
_COLLECTION_SNAPSHOT_ENDPOINTS = {
    "clob_order_books_batch",
    "clob_midpoints_batch",
    "clob_spreads_batch",
    "clob_last_trade_prices_batch",
    "clob_market_prices_batch",
    "price_history",
    "price_history_batch_recent",
    "trade_history",
}
_SOURCE_EVENT_ENDPOINTS = {
    "clob_server_time",
    "clob_order_book",
    "clob_midpoint",
    "clob_spread",
    "clob_last_trade_price",
}
_TIMESTAMP_FIELD_CANDIDATES = ("timestamp", "ts", "time", "serverTime", "server_time")


def _extract_top_level_timestamp(payload: Any) -> tuple[int | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    for field_name in _TIMESTAMP_FIELD_CANDIDATES:
        if payload.get(field_name) is None:
            continue
        event_ts_ns = _timestamp_value_to_ns(payload[field_name])
        if event_ts_ns is not None:
            return event_ts_ns, field_name
    return None, None


def _timestamp_value_to_ns(value: Any, *, unit_hint: str | None = None) -> int | None:
    normalized_epoch = normalize_epoch_value(value)
    if normalized_epoch is not None:
        unit = resolve_epoch_unit(normalized_epoch, unit_hint=unit_hint)
        return epoch_value_to_ns(normalized_epoch, unit=unit)
    if isinstance(value, str):
        return parse_iso8601_to_ns(value)
    return None


def _rest_timestamp_fields(*, endpoint: str, payload: Any, recv_ts_ns: int, recv_ts_iso: str) -> dict[str, Any]:
    extracted_timestamps = build_polymarket_source_timestamps(payload) if isinstance(payload, dict) else {}
    if endpoint == "clob_server_time":
        if isinstance(payload, int):
            server_fields = build_epoch_timestamp_fields("server_time", payload)
            server_time_ns = _timestamp_value_to_ns(payload, unit_hint="second")
            if server_time_ns is not None:
                return {
                    "event_ts_ns": server_time_ns,
                    "event_ts_iso": server_fields.get("server_time_iso"),
                    "event_ts_kind": "source_event",
                    "event_ts_field": "server_time",
                    "timestamp_semantics": "source_event",
                    "has_source_timestamp": True,
                    "source_event_timestamps": server_fields,
                }
        if isinstance(payload, dict):
            for candidate in ("serverTime", "server_time", "time", "timestamp"):
                if payload.get(candidate) is None:
                    continue
                server_fields = build_epoch_timestamp_fields("server_time", payload[candidate])
                server_time_ns = _timestamp_value_to_ns(payload[candidate], unit_hint="second")
                if server_time_ns is not None:
                    return {
                        "event_ts_ns": server_time_ns,
                        "event_ts_iso": server_fields.get("server_time_iso"),
                        "event_ts_kind": "source_event",
                        "event_ts_field": candidate,
                        "timestamp_semantics": "source_event",
                        "has_source_timestamp": True,
                        "source_event_timestamps": server_fields,
                    }

    if endpoint in _SOURCE_EVENT_ENDPOINTS:
        event_ts_ns, event_ts_field = _extract_top_level_timestamp(payload)
        if event_ts_ns is not None:
            return {
                "event_ts_ns": event_ts_ns,
                "event_ts_iso": ns_to_iso(event_ts_ns),
                "event_ts_kind": "source_event",
                "event_ts_field": event_ts_field,
                "timestamp_semantics": "source_event",
                "has_source_timestamp": True,
                "source_event_timestamps": extracted_timestamps,
            }

    if endpoint in _METADATA_ONLY_ENDPOINTS:
        return {
            "event_ts_ns": recv_ts_ns,
            "event_ts_iso": recv_ts_iso,
            "event_ts_kind": "recv_fallback",
            "event_ts_field": "recv_ts_ns",
            "timestamp_semantics": "source_metadata",
            "has_source_timestamp": bool(extracted_timestamps),
            "source_metadata_timestamps": extracted_timestamps or None,
        }

    if endpoint in _COLLECTION_SNAPSHOT_ENDPOINTS:
        return {
            "event_ts_ns": recv_ts_ns,
            "event_ts_iso": recv_ts_iso,
            "event_ts_kind": "recv_fallback",
            "event_ts_field": "recv_ts_ns",
            "timestamp_semantics": "collection_snapshot",
            "has_source_timestamp": bool(extracted_timestamps),
        }

    return {
        "event_ts_ns": recv_ts_ns,
        "event_ts_iso": recv_ts_iso,
        "event_ts_kind": "recv_fallback",
        "event_ts_field": "recv_ts_ns",
        "timestamp_semantics": "recv_only",
        "has_source_timestamp": False,
    }


class PolymarketRestPollerService:
    def __init__(self, config: PolymarketRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="polymarket_rest", config=config)
        self._selected_market: SelectedMarket | None = None
        self._session: PolymarketSessionWriters | None = None
        self._known_markets: dict[str, SelectedMarket] = {}
        self._market_sessions: dict[str, PolymarketSessionWriters] = {}
        self._active_market_ids: set[str] = set()
        self._buffered_lifecycle: list[dict[str, Any]] = []
        self._http_backoff = HttpEndpointBackoffRegistry(
            degraded_failure_threshold=self.config.http_degraded_failure_threshold,
            suspended_failure_threshold=self.config.http_suspended_failure_threshold,
            degraded_backoff_seconds=self.config.http_degraded_backoff_seconds,
            suspended_backoff_seconds=self.config.http_suspended_backoff_seconds,
        )

    async def run(self) -> None:
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
                base_url=self.config.clob_base_url,
                timeout_seconds=self.config.http_timeout_seconds,
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections,
                keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
                trust_env=self.config.http_trust_env,
            ) as clob_client,
            build_async_http_client(
                base_url=self.config.data_api_base_url,
                timeout_seconds=self.config.http_timeout_seconds,
                max_connections=self.config.http_max_connections,
                max_keepalive_connections=self.config.http_max_keepalive_connections,
                keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
                trust_env=self.config.http_trust_env,
            ) as data_client,
        ):
            discovery = GammaDiscoveryClient(gamma_client, self.config)
            self._write_lifecycle("service_start", {"mode": "public_rest_polling"})
            self._write_lifecycle("api_keys_detected", detection_metadata(load_key_value_file(self.config.api_keys_path)))
            last_discovery_ns = 0
            last_quote_ns = 0
            last_batch_quote_ns = 0
            last_history_ns = 0
            last_trade_history_ns = 0
            last_metadata_ns = 0
            last_metrics_ns = 0
            last_rewards_ns = 0
            last_event_detail_ns = 0
            loop_sleep_seconds = min(
                self.config.rest_loop_sleep_seconds,
                self.config.market_boundary_check_interval_seconds,
            )
            try:
                while True:
                    now_ns = utc_now_ns()
                    if now_ns - last_discovery_ns >= int(self.config.gamma_poll_interval_seconds * 1_000_000_000):
                        await self._discover(discovery)
                        last_discovery_ns = now_ns
                    self._refresh_active_markets(reason="boundary_tick")
                    if self._iter_poll_targets():
                        if now_ns - last_quote_ns >= int(
                            self.config.clob_quote_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_quotes(clob_client)
                            last_quote_ns = now_ns
                        if now_ns - last_batch_quote_ns >= int(
                            self.config.clob_batch_quote_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_batch_quotes(clob_client)
                            last_batch_quote_ns = now_ns
                        if now_ns - last_history_ns >= int(
                            self.config.history_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_price_history(clob_client)
                            last_history_ns = now_ns
                        if now_ns - last_trade_history_ns >= int(
                            self.config.trade_history_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_trade_history(data_client)
                            last_trade_history_ns = now_ns
                        if now_ns - last_metadata_ns >= int(
                            self.config.metadata_poll_interval_seconds * 1_000_000_000
                        ):
                            self._write_selected_metadata()
                            await self._poll_static_quote_metadata(clob_client)
                            last_metadata_ns = now_ns
                        if now_ns - last_metrics_ns >= int(
                            self.config.metrics_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_metrics(data_client)
                            last_metrics_ns = now_ns
                        if now_ns - last_event_detail_ns >= int(
                            self.config.event_detail_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_event_detail(gamma_client)
                            last_event_detail_ns = now_ns
                        if now_ns - last_rewards_ns >= int(
                            self.config.rewards_poll_interval_seconds * 1_000_000_000
                        ):
                            await self._poll_rewards(clob_client)
                            last_rewards_ns = now_ns
                    await asyncio.sleep(loop_sleep_seconds)
            finally:
                self._write_lifecycle("service_stop", {"mode": "public_rest_polling"})
                for session in list(self._market_sessions.values()):
                    session.close()

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
        targets = self._iter_poll_targets()
        for market, session in targets:
            recv_ts_ns = utc_now_ns()
            recv_ts_iso = ns_to_iso(recv_ts_ns)
            session.writer("rest").write(
                {
                    "record_type": "rest_observation",
                    "source": "polymarket",
                    "endpoint": "gamma_discovery",
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": recv_ts_iso,
                    "recv_ts_source": recv_timestamp_source(),
                    **session_common_fields(market),
                    "active_market_ids": [active_market.identity for active_market, _ in targets],
                    "active_market_slugs": [active_market.market_slug for active_market, _ in targets],
                    "payload": {"selected_market": market.raw_market, "candidate_count": len(candidates)},
                    **_rest_timestamp_fields(
                        endpoint="gamma_discovery",
                        payload=market.raw_market,
                        recv_ts_ns=recv_ts_ns,
                        recv_ts_iso=recv_ts_iso,
                    ),
                }
            )

    async def _poll_quotes(self, clob_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            await self._fetch_rest(
                clob_client,
                endpoint="clob_server_time",
                path="/time",
                params={},
                market=market,
                session=session,
            )
            for asset_id in market.asset_ids:
                for endpoint, path in (
                    ("clob_order_book", "/book"),
                    ("clob_midpoint", "/midpoint"),
                    ("clob_spread", "/spread"),
                    ("clob_last_trade_price", "/last-trade-price"),
                ):
                    await self._fetch_rest(
                        clob_client,
                        endpoint=endpoint,
                        path=path,
                        params={"token_id": asset_id},
                        asset_id=asset_id,
                        market=market,
                        session=session,
                    )

    async def _poll_static_quote_metadata(self, clob_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            for asset_id in market.asset_ids:
                for endpoint, path in (
                    ("clob_fee_rate", "/fee-rate"),
                    ("clob_tick_size", "/tick-size"),
                ):
                    await self._fetch_rest(
                        clob_client,
                        endpoint=endpoint,
                        path=path,
                        params={"token_id": asset_id},
                        asset_id=asset_id,
                        market=market,
                        session=session,
                    )

    async def _poll_batch_quotes(self, clob_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            token_body = [{"token_id": asset_id} for asset_id in market.asset_ids]
            await self._post_rest(
                clob_client,
                endpoint="clob_order_books_batch",
                path="/books",
                json_body=token_body,
                extra={"asset_ids": list(market.asset_ids)},
                market=market,
                session=session,
            )
            await self._post_rest(
                clob_client,
                endpoint="clob_midpoints_batch",
                path="/midpoints",
                json_body=token_body,
                extra={"asset_ids": list(market.asset_ids)},
                market=market,
                session=session,
            )
            await self._post_rest(
                clob_client,
                endpoint="clob_spreads_batch",
                path="/spreads",
                json_body=token_body,
                extra={"asset_ids": list(market.asset_ids)},
                market=market,
                session=session,
            )
            await self._post_rest(
                clob_client,
                endpoint="clob_last_trade_prices_batch",
                path="/last-trades-prices",
                json_body=token_body,
                extra={"asset_ids": list(market.asset_ids)},
                market=market,
                session=session,
            )
            price_body = [
                {"token_id": asset_id, "side": side}
                for asset_id in market.asset_ids
                for side in ("BUY", "SELL")
            ]
            await self._post_rest(
                clob_client,
                endpoint="clob_market_prices_batch",
                path="/prices",
                json_body=price_body,
                extra={"asset_ids": list(market.asset_ids), "sides": ["BUY", "SELL"]},
                market=market,
                session=session,
            )

    async def _poll_price_history(self, clob_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            for asset_id in market.asset_ids:
                await self._fetch_rest(
                    clob_client,
                    endpoint="price_history",
                    path="/prices-history",
                    params={"market": asset_id, "interval": "max"},
                    asset_id=asset_id,
                    market=market,
                    session=session,
                )
            now_s = int(datetime.now(tz=UTC).timestamp())
            await self._post_rest(
                clob_client,
                endpoint="price_history_batch_recent",
                path="/batch-prices-history",
                json_body={
                    "markets": list(market.asset_ids),
                    "start_ts": now_s - 300,
                    "end_ts": now_s,
                    "interval": "all",
                    "fidelity": 1,
                },
                extra={"asset_ids": list(market.asset_ids)},
                market=market,
                session=session,
            )

    async def _poll_trade_history(self, data_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            await self._fetch_rest(
                data_client,
                endpoint="trade_history",
                path="/trades",
                params={"market": market.condition_id, "limit": 500},
                market=market,
                session=session,
            )

    async def _poll_metrics(self, data_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            await self._fetch_rest(
                data_client,
                endpoint="holders",
                path="/holders",
                params={"market": market.condition_id, "limit": 200},
                market=market,
                session=session,
            )
            await self._fetch_rest(
                data_client,
                endpoint="market_positions",
                path="/v1/market-positions",
                params={"market": market.condition_id, "limit": 50, "status": "ALL"},
                market=market,
                session=session,
            )
            await self._fetch_rest(
                data_client,
                endpoint="market_open_interest",
                path="/oi",
                params={"market": market.condition_id},
                market=market,
                session=session,
            )
            if market.event_id is not None:
                await self._fetch_rest(
                    data_client,
                    endpoint="event_live_volume",
                    path="/live-volume",
                    params={"id": market.event_id},
                    market=market,
                    session=session,
                )

    async def _poll_event_detail(self, gamma_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            if market.event_id is None:
                continue
            await self._fetch_rest(
                gamma_client,
                endpoint="gamma_event_detail",
                path=f"/events/{market.event_id}",
                params={},
                market=market,
                session=session,
            )

    async def _poll_rewards(self, clob_client: httpx.AsyncClient) -> None:
        for market, session in self._iter_poll_targets():
            await self._fetch_rest(
                clob_client,
                endpoint="market_rewards",
                path=f"/rewards/markets/{market.condition_id}",
                params={"sponsored": "true"},
                market=market,
                session=session,
            )

    def _write_selected_metadata(self) -> None:
        for market, session in self._iter_poll_targets():
            recv_ts_ns = utc_now_ns()
            recv_ts_iso = ns_to_iso(recv_ts_ns)
            base = {
                "record_type": "rest_observation",
                "source": "polymarket",
                "recv_ts_ns": recv_ts_ns,
                "recv_ts_iso": recv_ts_iso,
                "recv_ts_source": recv_timestamp_source(),
                **session_common_fields(market),
            }
            session.writer("rest").write(
                {
                    **base,
                    "endpoint": "gamma_selected_market",
                    "payload": market.raw_market,
                    **_rest_timestamp_fields(
                        endpoint="gamma_selected_market",
                        payload=market.raw_market,
                        recv_ts_ns=recv_ts_ns,
                        recv_ts_iso=recv_ts_iso,
                    ),
                }
            )
            event_payload = market.raw_market.get("events", [None])[0]
            session.writer("rest").write(
                {
                    **base,
                    "endpoint": "gamma_selected_event",
                    "payload": event_payload,
                    **_rest_timestamp_fields(
                        endpoint="gamma_selected_event",
                        payload=event_payload or {},
                        recv_ts_ns=recv_ts_ns,
                        recv_ts_iso=recv_ts_iso,
                    ),
                }
            )

    async def _fetch_rest(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str,
        path: str,
        params: dict[str, Any],
        asset_id: str | None = None,
        market: SelectedMarket | None = None,
        session: PolymarketSessionWriters | None = None,
    ) -> None:
        market = market or self._selected_market
        session = session or self._session
        if market is None or session is None:
            return
        if not self._before_http_request(endpoint):
            return
        request_start_ns = utc_now_ns()
        try:
            outcome = await request_with_retries(
                client,
                "GET",
                path,
                max_retries=self.config.http_max_retries,
                retry_backoff_seconds=self.config.http_retry_backoff_seconds,
                params=params,
            )
        except httpx.HTTPError as exc:
            self._record_http_error(
                endpoint=endpoint,
                request_method="GET",
                request_path=path,
                error=exc,
                extra={"params": dict(params), "asset_id": asset_id},
            )
            return
        response = outcome.response
        response_end_ns = utc_now_ns()
        payload = response.json()
        self._observe_http_success(endpoint)
        recv_ts_ns = utc_now_ns()
        recv_ts_iso = ns_to_iso(recv_ts_ns)
        capture_timing = build_capture_timing(
            request_start_ns=request_start_ns,
            response_end_ns=response_end_ns,
        )
        capture_timing["request_attempt_count"] = outcome.attempt_count
        record = {
            "record_type": "rest_observation",
            "source": "polymarket",
            "endpoint": endpoint,
            "url": str(response.request.url),
            "http_status": response.status_code,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": recv_ts_iso,
            "recv_ts_source": recv_timestamp_source(),
            **session_common_fields(market),
            "payload": payload,
            **_rest_timestamp_fields(
                endpoint=endpoint,
                payload=payload,
                recv_ts_ns=recv_ts_ns,
                recv_ts_iso=recv_ts_iso,
            ),
            "capture_timing": capture_timing,
            "extra": {"asset_id": asset_id} if asset_id else {},
        }
        if asset_id is not None:
            record["token_asset_id"] = asset_id
            record["token_outcome"] = market.outcome_map.get(asset_id)
        if endpoint == "clob_server_time":
            server_time_value: Any
            if isinstance(payload, int):
                server_time_value = payload
            elif isinstance(payload, dict):
                server_time_value = next(
                    (payload.get(candidate) for candidate in ("serverTime", "server_time", "time", "timestamp") if payload.get(candidate) is not None),
                    None,
                )
            else:
                server_time_value = None
            if server_time_value is not None:
                record["clock_probe"] = build_clock_probe(
                    "server_time",
                    server_time_value,
                    request_start_ns=request_start_ns,
                    response_end_ns=response_end_ns,
                )
        if endpoint == "clob_order_book" and isinstance(payload, dict):
            bids = payload.get("bids", [])
            asks = payload.get("asks", [])
            record["extra"]["derived"] = {
                "best_bid": bids[0] if bids else None,
                "best_ask": asks[0] if asks else None,
            }
        session.writer("rest").write(record)
        if asset_id is not None:
            token_writer = session.token_writers.get(asset_id, {}).get("token_rest")
            if token_writer is not None:
                token_writer.write(
                    {
                        **record,
                        "record_type": "token_rest_observation",
                    }
                )

    async def _post_rest(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str,
        path: str,
        json_body: Any,
        extra: dict[str, Any] | None = None,
        market: SelectedMarket | None = None,
        session: PolymarketSessionWriters | None = None,
    ) -> None:
        market = market or self._selected_market
        session = session or self._session
        if market is None or session is None:
            return
        if not self._before_http_request(endpoint):
            return
        request_start_ns = utc_now_ns()
        try:
            outcome = await request_with_retries(
                client,
                "POST",
                path,
                max_retries=self.config.http_max_retries,
                retry_backoff_seconds=self.config.http_retry_backoff_seconds,
                json=json_body,
            )
        except httpx.HTTPError as exc:
            self._record_http_error(
                endpoint=endpoint,
                request_method="POST",
                request_path=path,
                error=exc,
                extra=extra,
            )
            return
        response = outcome.response
        response_end_ns = utc_now_ns()
        payload = response.json()
        self._observe_http_success(endpoint)
        recv_ts_ns = utc_now_ns()
        recv_ts_iso = ns_to_iso(recv_ts_ns)
        capture_timing = build_capture_timing(
            request_start_ns=request_start_ns,
            response_end_ns=response_end_ns,
        )
        capture_timing["request_attempt_count"] = outcome.attempt_count
        record = {
            "record_type": "rest_observation",
            "source": "polymarket",
            "endpoint": endpoint,
            "url": str(response.request.url),
            "http_status": response.status_code,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": recv_ts_iso,
            "recv_ts_source": recv_timestamp_source(),
            **session_common_fields(market),
            "payload": payload,
            "request_payload": json_body,
            **_rest_timestamp_fields(
                endpoint=endpoint,
                payload=payload,
                recv_ts_ns=recv_ts_ns,
                recv_ts_iso=recv_ts_iso,
            ),
            "capture_timing": capture_timing,
            "extra": extra or {},
        }
        session.writer("rest").write(record)

    def _record_http_error(
        self,
        *,
        endpoint: str,
        request_method: str,
        request_path: str,
        error: Exception,
        extra: dict[str, Any] | None = None,
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
        if extra:
            details.update({key: value for key, value in extra.items() if value is not None})
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

    def _remember_markets(self, markets: list[SelectedMarket]) -> None:
        for market in markets:
            self._known_markets[market.identity] = market
            session = self._market_sessions.get(market.identity)
            if session is not None:
                session.market = market

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
            self._ensure_session(market)
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
            session = self._market_sessions.get(market_id)
            market = self._known_markets.get(market_id)
            if session is None or market is None:
                continue
            session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_active",
                    details={
                        **session_common_fields(market),
                        **self._market_boundary_fields(market),
                        **active_context,
                    },
                )
            )
        primary_market = newest_market(active_markets)
        if primary_market is not None:
            previous_primary = self._selected_market
            self._selected_market = primary_market
            self._session = self._market_sessions.get(primary_market.identity)
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
            session = self._market_sessions.get(market_id)
            market = self._known_markets.get(market_id)
            if session is None or market is None:
                continue
            session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_inactive",
                    details={
                        **session_common_fields(market),
                        **self._market_boundary_fields(market),
                        **active_context,
                    },
                )
            )
            session.close()
            del self._market_sessions[market_id]
        self._prune_expired_known_markets(now=now)

    def _ensure_session(self, market: SelectedMarket) -> PolymarketSessionWriters:
        session = self._market_sessions.get(market.identity)
        if session is not None:
            session.market = market
            return session
        session = PolymarketSessionWriters(
            root=self.config.out,
            market=market,
            started_at=datetime.now(tz=UTC),
            kinds=("rest", "lifecycle"),
            token_kinds=("token_rest",) if self.config.enable_token_rest_split else (),
            flush_interval_seconds=self.config.flush_interval_seconds,
            manifest_enabled=self.config.manifest_enabled,
            static_fields_extra=self._runtime.static_fields(),
        )
        for buffered in self._buffered_lifecycle:
            session.writer("lifecycle").write({**buffered, **session_common_fields(market)})
        if self._buffered_lifecycle:
            self._buffered_lifecycle.clear()
        self._market_sessions[market.identity] = session
        return session

    def _iter_poll_targets(self) -> list[tuple[SelectedMarket, PolymarketSessionWriters]]:
        markets = self._markets_for_ids(self._active_market_ids)
        targets = [
            (market, self._market_sessions[market.identity])
            for market in markets
            if market.identity in self._market_sessions
        ]
        if targets:
            return targets
        if self._selected_market is not None and self._session is not None:
            return [(self._selected_market, self._session)]
        return []

    def _markets_for_ids(self, market_ids: set[str] | list[str]) -> list[SelectedMarket]:
        values: list[SelectedMarket] = []
        for market_id in market_ids:
            session = self._market_sessions.get(market_id)
            if session is not None:
                values.append(session.market)
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
            session = self._market_sessions.get(market_id)
            market = self._known_markets.get(market_id)
            if session is None or market is None:
                continue
            session.writer("lifecycle").write(
                lifecycle_event(event=event, details={**session_common_fields(market), **details})
            )

    def _prune_expired_known_markets(self, *, now: datetime) -> None:
        expired_ids = [
            market_id
            for market_id, market in self._known_markets.items()
            if market_id not in self._active_market_ids
            and market_id not in self._market_sessions
            and market_has_expired(
                market,
                now=now,
                interval_seconds=self.config.market_interval_seconds,
            )
        ]
        for market_id in expired_ids:
            del self._known_markets[market_id]
