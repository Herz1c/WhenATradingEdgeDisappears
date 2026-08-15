from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import orjson
from websockets import connect

from binance_recorder.utils import generate_connection_id, ns_to_iso
from market_recorders.asyncio_utils import cancel_and_await
from market_recorders.http_utils import HttpEndpointBackoffRegistry, build_async_http_client
from market_recorders.keyfile import load_key_value_file
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import recv_timestamp_source, utc_now_ns

from .config import PolymarketRecorderConfig
from .discovery import GammaDiscoveryClient
from .l2 import PolymarketL2StateTracker
from .lifecycle import lifecycle_event
from .utils import (
    SelectedMarket,
    build_polymarket_source_timestamps,
    detection_metadata,
    market_active_window_fields,
    market_has_expired,
    market_close_epoch,
    market_is_active,
    newest_market,
    session_common_fields,
    sort_markets_by_open,
)
from .writers import PolymarketSessionWriters


def _iter_payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


@dataclass(slots=True)
class _SubscriptionChange:
    added_asset_ids: list[str]
    removed_asset_ids: list[str]
    reason: str
    active_market_ids: list[str]
    active_market_slugs: list[str]
    boundary_times: list[dict[str, Any]]


@dataclass(slots=True)
class _WsMarketRuntime:
    market: SelectedMarket
    session: PolymarketSessionWriters
    tracker: PolymarketL2StateTracker
    # session_common_fields(market) is static per market but was being rebuilt
    # (3 datetime->iso conversions included) for every record on a 300-600
    # msg/s stream; cache it here.
    common_fields: dict[str, Any] = field(default_factory=dict)


class PolymarketRecorderService:
    def __init__(self, config: PolymarketRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="polymarket_ws", config=config)
        self._selected_market: SelectedMarket | None = None
        self._session: PolymarketSessionWriters | None = None
        self._session_tracker: PolymarketL2StateTracker | None = None
        self._known_markets: dict[str, SelectedMarket] = {}
        self._market_runtimes: dict[str, _WsMarketRuntime] = {}
        self._active_market_ids: set[str] = set()
        self._pending_subscription_change: _SubscriptionChange | None = None
        self._connected_asset_ids: list[str] = []
        self._buffered_lifecycle: list[dict[str, Any]] = []
        self._last_activity_ns = utc_now_ns()
        self._prewindow_resynced_market_ids: set[str] = set()
        # market_id -> ns deadline after which its (already inactive) session
        # is really closed; lets in-flight/queued events land in the file.
        self._retiring_runtime_deadline_ns: dict[str, int] = {}
        self._http_backoff = HttpEndpointBackoffRegistry(
            degraded_failure_threshold=self.config.http_degraded_failure_threshold,
            suspended_failure_threshold=self.config.http_suspended_failure_threshold,
            degraded_backoff_seconds=self.config.http_degraded_backoff_seconds,
            suspended_backoff_seconds=self.config.http_suspended_backoff_seconds,
        )

    async def run(self) -> None:
        async with build_async_http_client(
            base_url=self.config.gamma_base_url,
            timeout_seconds=self.config.http_timeout_seconds,
            max_connections=self.config.http_max_connections,
            max_keepalive_connections=self.config.http_max_keepalive_connections,
            keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
            trust_env=self.config.http_trust_env,
        ) as gamma_client:
            discovery_client = GammaDiscoveryClient(gamma_client, self.config)
            self._write_lifecycle("service_start", {"mode": "public_market_ws"})
            self._write_lifecycle("api_keys_detected", detection_metadata(load_key_value_file(self.config.api_keys_path)))
            discovery_task = asyncio.create_task(self._discovery_loop(discovery_client), name="polymarket:discovery")
            try:
                await self._websocket_loop()
            finally:
                cancelled_during_cleanup = await cancel_and_await(discovery_task)
                self._write_lifecycle("service_stop", {"mode": "public_market_ws"})
                for runtime in list(self._market_runtimes.values()):
                    runtime.session.close()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def _discovery_loop(self, discovery_client: GammaDiscoveryClient) -> None:
        seq = 0
        while True:
            seq += 1
            if not self._before_http_request("gamma_discovery"):
                await asyncio.sleep(self.config.gamma_poll_interval_seconds)
                continue
            try:
                selected, candidates, _, items, target_slugs = await discovery_client.discover_market_set()
            except httpx.HTTPError as exc:
                self._record_http_error(
                    endpoint="gamma_discovery",
                    request_method="GET",
                    request_path="/markets",
                    error=exc,
                )
                await asyncio.sleep(self.config.gamma_poll_interval_seconds)
                continue
            self._observe_http_success("gamma_discovery")
            self._remember_markets(candidates)
            self._refresh_active_markets(
                reason="gamma_discovery_poll",
                selected_market=selected,
                target_slugs=target_slugs,
                raw_items=items,
                poll_seq=seq,
            )
            await asyncio.sleep(self.config.gamma_poll_interval_seconds)

    async def _switch_market(
        self,
        selected: SelectedMarket,
        *,
        target_slugs: list[str],
        raw_items: list[dict[str, Any]],
        poll_seq: int,
    ) -> None:
        self._remember_markets([selected])
        runtime = self._ensure_runtime(selected)
        old_asset_ids = self._active_asset_ids()
        self._active_market_ids = {selected.identity}
        self._selected_market = selected
        self._session = runtime.session
        self._session_tracker = runtime.tracker
        runtime.session.writer("discovery").write(
            self._discovery_record(
                selected,
                target_slugs=target_slugs,
                raw_items=raw_items,
                poll_seq=poll_seq,
                event="market_selected",
                active_markets=[selected],
            )
        )
        runtime.session.writer("lifecycle").write(
            lifecycle_event(
                event="market_selected",
                details={**session_common_fields(selected), "poll_seq": poll_seq, **self._market_boundary_fields(selected)},
            )
        )
        new_asset_ids = self._active_asset_ids()
        self._pending_subscription_change = _SubscriptionChange(
            added_asset_ids=[asset_id for asset_id in new_asset_ids if asset_id not in old_asset_ids],
            removed_asset_ids=[asset_id for asset_id in old_asset_ids if asset_id not in new_asset_ids],
            reason="test_switch_market",
            active_market_ids=[selected.identity],
            active_market_slugs=[selected.market_slug],
            boundary_times=[self._market_boundary_fields(selected)],
        )

    async def _websocket_loop(self) -> None:
        delay = self.config.reconnect_min_seconds
        while True:
            self._refresh_active_markets(reason="boundary_tick")
            if not self._active_market_ids:
                await asyncio.sleep(self.config.market_boundary_check_interval_seconds)
                continue
            connection_id = generate_connection_id("poly-conn")
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
        active_market_ids = self._active_market_ids_snapshot()
        if not active_market_ids:
            return
        active_asset_ids = self._active_asset_ids()
        if not active_asset_ids:
            return
        self._write_lifecycle("websocket_connecting", {"connection_id": connection_id, "ws_url": self.config.websocket_url})
        async with connect(
            self.config.websocket_url,
            open_timeout=self.config.connect_open_timeout_seconds,
            close_timeout=self.config.connect_close_timeout_seconds,
            max_queue=self.config.websocket_max_queue,
            ping_interval=None,
        ) as ws:
            initial_payload = {
                "assets_ids": active_asset_ids,
                "type": "market",
                "custom_feature_enabled": True,
                "initial_dump": True,
            }
            await ws.send(orjson.dumps(initial_payload).decode("utf-8"))
            self._connected_asset_ids = list(active_asset_ids)
            self._pending_subscription_change = None
            self._last_activity_ns = utc_now_ns()
            self._broadcast_market_lifecycle(
                active_market_ids,
                "reconnect_rebuilt_active_market_set",
                {
                    "connection_id": connection_id,
                    "ws_url": self.config.websocket_url,
                    **self._active_market_context(reason="websocket_connected"),
                },
            )
            self._write_lifecycle(
                "websocket_connected",
                {
                    "connection_id": connection_id,
                    "ws_url": self.config.websocket_url,
                    **self._active_market_context(reason="websocket_connected"),
                },
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws, connection_id), name="polymarket:heartbeat")
            # READER/WORKER SPLIT. The single-loop design (recv -> track ->
            # serialize -> zstd-write per message) could not keep up with the
            # 300-600 msg/s market channel: the websockets library queue
            # absorbed minutes of frames (recv_ts fresh, payload source time
            # 60-180s stale), TCP backpressure eventually made the server drop
            # the connection, and the decision-window tail of every market was
            # lost. The reader task below only drains the socket and stamps
            # arrival time; this loop does the heavy per-record work from an
            # in-process queue that is observable and never blocks the socket.
            ingest_queue: deque[tuple[int, str | bytes]] = deque()
            queue_event = asyncio.Event()
            reader_task = asyncio.create_task(
                self._reader_loop(ws, ingest_queue, queue_event), name="polymarket:ws_reader"
            )
            local_seq = 0
            boundary_interval_s = max(0.05, self.config.market_boundary_check_interval_seconds)
            boundary_interval_ns = int(boundary_interval_s * 1_000_000_000)
            stats_interval_ns = int(max(1.0, self.config.ingest_stats_interval_seconds) * 1_000_000_000)
            last_boundary_ns = 0
            last_stats_ns = utc_now_ns()
            stats_processed = 0
            stats_max_ingest_lag_s = 0.0
            try:
                while True:
                    now_ns = utc_now_ns()
                    if now_ns - last_boundary_ns >= boundary_interval_ns:
                        last_boundary_ns = now_ns
                        self._refresh_active_markets(reason="boundary_tick")
                        due_resync = self._prewindow_resync_due()
                        if due_resync:
                            self._write_lifecycle(
                                "prewindow_hard_resync",
                                {
                                    "connection_id": connection_id,
                                    "reason": "fresh_pm_book_before_decision_window",
                                    "queue_depth": len(ingest_queue),
                                    "market_slugs": [market.market_slug for market in due_resync],
                                    "ttc_seconds": [
                                        round(
                                            market_close_epoch(
                                                market,
                                                interval_seconds=self.config.market_interval_seconds,
                                            )
                                            - utc_now_ns() / 1_000_000_000,
                                            3,
                                        )
                                        for market in due_resync
                                    ],
                                },
                            )
                            local_seq = self._drain_ingest_queue(ingest_queue, connection_id, local_seq)
                            await ws.close()
                            return
                        await self._apply_pending_change(ws, connection_id)
                    if now_ns - last_stats_ns >= stats_interval_ns:
                        elapsed_s = (now_ns - last_stats_ns) / 1_000_000_000
                        self._write_lifecycle(
                            "ws_ingest_stats",
                            {
                                "connection_id": connection_id,
                                "queue_depth": len(ingest_queue),
                                "processed_messages": stats_processed,
                                "processed_per_s": round(stats_processed / elapsed_s, 1),
                                "max_ingest_lag_s": round(stats_max_ingest_lag_s, 3),
                            },
                        )
                        last_stats_ns = now_ns
                        stats_processed = 0
                        stats_max_ingest_lag_s = 0.0
                    if len(ingest_queue) > self.config.ingest_queue_hard_limit:
                        self._write_lifecycle(
                            "ingest_queue_overflow_resync",
                            {
                                "connection_id": connection_id,
                                "queue_depth": len(ingest_queue),
                                "hard_limit": self.config.ingest_queue_hard_limit,
                            },
                        )
                        ingest_queue.clear()
                        await ws.close()
                        return
                    if not ingest_queue:
                        if reader_task.done():
                            exc = reader_task.exception()
                            if exc is not None:
                                raise exc
                            return
                        queue_event.clear()
                        try:
                            await asyncio.wait_for(queue_event.wait(), timeout=boundary_interval_s)
                        except TimeoutError:
                            pass
                        continue
                    # Process a small batch, then yield so the reader (and the
                    # other recorder services sharing this event loop) run.
                    for _ in range(min(len(ingest_queue), 20)):
                        recv_ts_ns, message = ingest_queue.popleft()
                        stats_processed += 1
                        ingest_lag_s = (utc_now_ns() - recv_ts_ns) / 1_000_000_000
                        if ingest_lag_s > stats_max_ingest_lag_s:
                            stats_max_ingest_lag_s = ingest_lag_s
                        try:
                            payload = orjson.loads(message)
                        except orjson.JSONDecodeError:
                            continue
                        for item in _iter_payload_items(payload):
                            local_seq += 1
                            self._handle_payload(
                                item,
                                connection_id=connection_id,
                                local_seq=local_seq,
                                recv_ts_ns=recv_ts_ns,
                            )
                    await asyncio.sleep(0)
            finally:
                reader_cancelled = await cancel_and_await(reader_task)
                cancelled_during_cleanup = await cancel_and_await(heartbeat_task)
                if reader_cancelled or cancelled_during_cleanup:
                    raise asyncio.CancelledError
        self._connected_asset_ids = []
        self._write_lifecycle("websocket_closed", {"connection_id": connection_id, "ws_url": self.config.websocket_url})

    async def _reader_loop(
        self,
        ws: Any,
        ingest_queue: deque[tuple[int, str | bytes]],
        queue_event: asyncio.Event,
    ) -> None:
        """Drain the websocket as fast as messages arrive.

        recv_ts_ns is stamped HERE, at true wire arrival, so recorded
        recv-vs-source lag reflects the network, not our processing backlog.
        Keeping this task trivially cheap is what prevents TCP backpressure
        (and the server-side slow-consumer disconnects it used to cause).
        """
        while True:
            message = await ws.recv()
            self._last_activity_ns = utc_now_ns()
            if message in {"PONG", "{}", b"{}"}:
                continue
            ingest_queue.append((utc_now_ns(), message))
            queue_event.set()

    def _drain_ingest_queue(
        self,
        ingest_queue: deque[tuple[int, str | bytes]],
        connection_id: str,
        local_seq: int,
    ) -> int:
        """Flush already-received messages before an intentional reconnect so
        the pre-window resync never discards data. Synchronous on purpose --
        the queue is expected to be near-empty when the writer keeps up."""
        while ingest_queue:
            recv_ts_ns, message = ingest_queue.popleft()
            try:
                payload = orjson.loads(message)
            except orjson.JSONDecodeError:
                continue
            for item in _iter_payload_items(payload):
                local_seq += 1
                self._handle_payload(
                    item,
                    connection_id=connection_id,
                    local_seq=local_seq,
                    recv_ts_ns=recv_ts_ns,
                )
        return local_seq

    async def _heartbeat_loop(self, ws: Any, connection_id: str) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
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
        target_asset_ids = self._active_asset_ids()
        if change.added_asset_ids:
            await ws.send(
                orjson.dumps(
                    {
                        "operation": "subscribe",
                        "assets_ids": target_asset_ids,
                        "custom_feature_enabled": True,
                        "initial_dump": True,
                    }
                ).decode("utf-8")
            )
        if change.removed_asset_ids:
            await ws.send(
                orjson.dumps({"operation": "unsubscribe", "assets_ids": change.removed_asset_ids}).decode("utf-8")
            )
        self._connected_asset_ids = list(target_asset_ids)
        self._pending_subscription_change = None
        self._broadcast_market_lifecycle(
            change.active_market_ids,
            "subscription_updated_with_active_market_set",
            {
                "connection_id": connection_id,
                "reason": change.reason,
                "asset_ids_added": change.added_asset_ids,
                "asset_ids_removed": change.removed_asset_ids,
                "active_market_ids": change.active_market_ids,
                "active_market_slugs": change.active_market_slugs,
                "active_market_boundaries": change.boundary_times,
            },
        )
        self._write_lifecycle(
            "subscription_updated",
            {
                "connection_id": connection_id,
                "reason": change.reason,
                "old_asset_ids": change.removed_asset_ids,
                "new_asset_ids": change.added_asset_ids,
                "active_market_ids": change.active_market_ids,
                "active_market_slugs": change.active_market_slugs,
            },
        )

    def _handle_payload(
        self,
        payload: dict[str, Any],
        *,
        connection_id: str,
        local_seq: int,
        recv_ts_ns: int | None = None,
    ) -> None:
        matched_asset_ids = self._extract_asset_ids(payload)
        runtimes = self._target_runtimes_for_payload(matched_asset_ids)
        if not runtimes:
            return
        if recv_ts_ns is None:
            recv_ts_ns = utc_now_ns()
        recv_ts_iso = ns_to_iso(recv_ts_ns)
        recv_source = recv_timestamp_source()
        payload_source_timestamps = build_polymarket_source_timestamps(payload)
        for runtime in runtimes:
            market = runtime.market
            matched_outcomes = [
                market.outcome_map.get(asset_id)
                for asset_id in matched_asset_ids
                if asset_id in market.outcome_map
            ]
            event_type = str(payload.get("event_type")) if payload.get("event_type") is not None else None
            market_scope = self._market_scope(market.asset_ids, matched_asset_ids)
            token_modeling_safe = market_scope == "selected_only" and event_type != "new_market"
            record = {
                "record_type": "ws_event",
                "source": "polymarket",
                "channel": "market",
                "connection_id": connection_id,
                "local_msg_seq": local_seq,
                "ws_url": self.config.websocket_url,
                "recv_ts_ns": recv_ts_ns,
                "recv_ts_iso": recv_ts_iso,
                "recv_ts_source": recv_source,
                **runtime.common_fields,
                "event_type": event_type,
                "market_scope": market_scope,
                "token_modeling_safe": token_modeling_safe,
                "matched_asset_ids": matched_asset_ids,
                "matched_outcomes": matched_outcomes,
                "payload": payload,
                "timestamp_semantics": "source_metadata",
                "source_timestamps": payload_source_timestamps,
            }
            runtime.session.writer("ws").write(record)
            for asset_id in [asset_id for asset_id in matched_asset_ids if asset_id in market.outcome_map]:
                token_writer = runtime.session.token_writers.get(asset_id, {}).get("token_ws")
                if token_writer is None:
                    continue
                token_writer.write(
                    {
                        **record,
                        "record_type": "token_ws_event",
                        "token_asset_id": asset_id,
                        "token_outcome": market.outcome_map.get(asset_id),
                    }
                )
            filtered_payload = self._filter_payload_for_market(payload, market)
            if filtered_payload is None:
                continue
            filtered_source_timestamps = build_polymarket_source_timestamps(filtered_payload)
            for l2_record in runtime.tracker.process_event(
                filtered_payload, source_timestamps=filtered_source_timestamps
            ):
                token_asset_id = str(l2_record.get("token_asset_id") or "")
                if not self._selected_l2_record_is_safe(market, l2_record, token_asset_id=token_asset_id):
                    continue
                l2_output = {
                    "source": "polymarket",
                    "channel": "market",
                    "connection_id": connection_id,
                    "local_msg_seq": local_seq,
                    "ws_url": self.config.websocket_url,
                    "recv_ts_ns": recv_ts_ns,
                    "recv_ts_iso": recv_ts_iso,
                    "recv_ts_source": recv_source,
                    **runtime.common_fields,
                    **l2_record,
                    "token_outcome": market.outcome_map.get(token_asset_id),
                    "token_modeling_safe": True,
                    "timestamp_semantics": "source_metadata",
                    "source_timestamps": filtered_source_timestamps,
                }
                runtime.session.writer("l2").write(l2_output)

    @staticmethod
    def _selected_l2_record_is_safe(
        market: SelectedMarket,
        l2_record: dict[str, Any],
        *,
        token_asset_id: str,
    ) -> bool:
        del l2_record
        return token_asset_id in market.outcome_map

    @staticmethod
    def _market_scope(selected_asset_ids: list[str], matched_asset_ids: list[str]) -> str:
        if not matched_asset_ids:
            return "no_asset_match"
        selected = set(selected_asset_ids)
        matched = set(matched_asset_ids)
        if matched <= selected:
            return "selected_only"
        if matched & selected:
            return "contains_external_assets"
        return "off_market"

    def _extract_asset_ids(self, payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        if payload.get("asset_id") is not None:
            values.append(str(payload["asset_id"]))
        for key in ("asset_ids", "assets_ids"):
            if isinstance(payload.get(key), list):
                values.extend(str(item) for item in payload[key])
        if isinstance(payload.get("price_changes"), list):
            values.extend(str(item.get("asset_id")) for item in payload["price_changes"] if item.get("asset_id") is not None)
        if isinstance(payload.get("orders"), list):
            values.extend(str(item.get("asset_id")) for item in payload["orders"] if item.get("asset_id") is not None)
        return list(dict.fromkeys(value for value in values if value))

    def _filter_payload_for_market(self, payload: dict[str, Any], market: SelectedMarket) -> dict[str, Any] | None:
        relevant_asset_ids = set(market.asset_ids)
        filtered = dict(payload)
        has_relevant = False
        if filtered.get("asset_id") is not None:
            asset_id = str(filtered["asset_id"])
            if asset_id not in relevant_asset_ids:
                return None
            filtered["asset_id"] = asset_id
            has_relevant = True
        for key in ("asset_ids", "assets_ids"):
            if isinstance(filtered.get(key), list):
                filtered_values = [str(item) for item in filtered[key] if str(item) in relevant_asset_ids]
                filtered[key] = filtered_values
                has_relevant = has_relevant or bool(filtered_values)
        if isinstance(filtered.get("price_changes"), list):
            filtered_changes = [
                item
                for item in filtered["price_changes"]
                if isinstance(item, dict) and str(item.get("asset_id") or "") in relevant_asset_ids
            ]
            filtered["price_changes"] = filtered_changes
            has_relevant = has_relevant or bool(filtered_changes)
        if isinstance(filtered.get("orders"), list):
            filtered_orders = [
                item
                for item in filtered["orders"]
                if isinstance(item, dict) and str(item.get("asset_id") or "") in relevant_asset_ids
            ]
            filtered["orders"] = filtered_orders
            has_relevant = has_relevant or bool(filtered_orders)
        return filtered if has_relevant else None

    def _target_runtimes_for_payload(self, asset_ids: list[str]) -> list[_WsMarketRuntime]:
        # Match against every open runtime (active AND retiring/lingering) so
        # events that arrive just after a market's close still land in that
        # market's files; unmatched payloads fall back to the newest active.
        fallback = self._runtime_values()
        all_runtimes = self._all_runtimes()
        if not all_runtimes:
            return []
        if not fallback:
            fallback = all_runtimes
        if not asset_ids:
            return [fallback[-1]]
        matched = [
            runtime
            for runtime in all_runtimes
            if any(asset_id in runtime.market.asset_ids for asset_id in asset_ids)
        ]
        return matched or [fallback[-1]]

    def _all_runtimes(self) -> list[_WsMarketRuntime]:
        ordered = self._markets_for_ids(set(self._market_runtimes))
        return [
            self._market_runtimes[market.identity]
            for market in ordered
            if market.identity in self._market_runtimes
        ]

    def _remember_markets(self, markets: list[SelectedMarket]) -> None:
        for market in markets:
            self._known_markets[market.identity] = market
            runtime = self._market_runtimes.get(market.identity)
            if runtime is not None:
                runtime.market = market
                runtime.session.market = market
                runtime.common_fields = dict(session_common_fields(market))

    def _refresh_active_markets(
        self,
        *,
        reason: str,
        selected_market: SelectedMarket | None = None,
        target_slugs: list[str] | None = None,
        raw_items: list[dict[str, Any]] | None = None,
        poll_seq: int | None = None,
    ) -> None:
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
            self._ensure_runtime(market)
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
            runtime = self._market_runtimes.get(market_id)
            if runtime is None:
                continue
            runtime.session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_active",
                    details={
                        **session_common_fields(runtime.market),
                        **self._market_boundary_fields(runtime.market),
                        **active_context,
                    },
                )
            )
        if target_slugs is not None and raw_items is not None and poll_seq is not None:
            for market in active_markets:
                runtime = self._market_runtimes.get(market.identity)
                if runtime is None:
                    continue
                event_name = (
                    "market_became_active"
                    if market.identity in added_ids
                    else "market_confirmed"
                )
                runtime.session.writer("discovery").write(
                    self._discovery_record(
                        market,
                        target_slugs=target_slugs,
                        raw_items=raw_items,
                        poll_seq=poll_seq,
                        event=event_name,
                        active_markets=active_markets,
                    )
                )
        primary_market = newest_market(active_markets)
        if primary_market is None and self._active_market_ids:
            primary_market = newest_market(self._markets_for_ids(self._active_market_ids))
        if primary_market is not None:
            runtime = self._market_runtimes.get(primary_market.identity)
            if runtime is not None:
                previous_primary = self._selected_market
                self._selected_market = runtime.market
                self._session = runtime.session
                self._session_tracker = runtime.tracker
                if previous_primary is None or previous_primary.identity != primary_market.identity:
                    runtime.session.writer("lifecycle").write(
                        lifecycle_event(
                            event="market_selected" if previous_primary is None else "market_switched",
                            details={
                                **session_common_fields(runtime.market),
                                **self._market_boundary_fields(runtime.market),
                                "reason": reason,
                                **active_context,
                                **({"poll_seq": poll_seq} if poll_seq is not None else {}),
                            },
                        )
                    )
        else:
            self._selected_market = None
            self._session = None
            self._session_tracker = None
        connected_assets = set(self._connected_asset_ids)
        new_active_assets = set(self._active_asset_ids())
        if connected_assets != new_active_assets:
            self._pending_subscription_change = _SubscriptionChange(
                added_asset_ids=sorted(new_active_assets - connected_assets),
                removed_asset_ids=sorted(connected_assets - new_active_assets),
                reason=reason,
                active_market_ids=[market.identity for market in active_markets],
                active_market_slugs=[market.market_slug for market in active_markets],
                boundary_times=[self._market_boundary_fields(market) for market in active_markets],
            )
        linger_ns = int(max(0.0, self.config.market_close_linger_seconds) * 1_000_000_000)
        now_ns = utc_now_ns()
        for market_id in removed_ids:
            runtime = self._market_runtimes.get(market_id)
            if runtime is None:
                continue
            runtime.session.writer("lifecycle").write(
                lifecycle_event(
                    event="market_became_inactive",
                    details={
                        **session_common_fields(runtime.market),
                        **self._market_boundary_fields(runtime.market),
                        **active_context,
                    },
                )
            )
            # Do NOT close the session yet: events for this market received
            # around the close boundary may still be queued/in flight, and the
            # decision-window tail must land in the file. Close after linger.
            self._retiring_runtime_deadline_ns[market_id] = now_ns + linger_ns
        for market_id, deadline_ns in list(self._retiring_runtime_deadline_ns.items()):
            if market_id in self._active_market_ids:
                del self._retiring_runtime_deadline_ns[market_id]
                continue
            if now_ns < deadline_ns:
                continue
            runtime = self._market_runtimes.get(market_id)
            if runtime is not None:
                runtime.session.close()
                del self._market_runtimes[market_id]
            del self._retiring_runtime_deadline_ns[market_id]
            self._prewindow_resynced_market_ids.discard(market_id)
        self._prune_expired_known_markets(now=now)

    def _ensure_runtime(self, market: SelectedMarket) -> _WsMarketRuntime:
        self._retiring_runtime_deadline_ns.pop(market.identity, None)
        existing = self._market_runtimes.get(market.identity)
        if existing is not None:
            existing.market = market
            existing.session.market = market
            existing.common_fields = dict(session_common_fields(market))
            return existing
        session = PolymarketSessionWriters(
            root=self.config.out,
            market=market,
            started_at=datetime.now(tz=UTC),
            kinds=("ws", "l2", "discovery", "lifecycle"),
            token_kinds=("token_ws",) if self.config.enable_token_ws_split else (),
            flush_interval_seconds=self.config.flush_interval_seconds,
            manifest_enabled=self.config.manifest_enabled,
            static_fields_extra=self._runtime.static_fields(),
        )
        tracker = PolymarketL2StateTracker()
        for buffered in self._buffered_lifecycle:
            session.writer("lifecycle").write({**buffered, **session_common_fields(market)})
        if self._buffered_lifecycle:
            self._buffered_lifecycle.clear()
        runtime = _WsMarketRuntime(
            market=market,
            session=session,
            tracker=tracker,
            common_fields=dict(session_common_fields(market)),
        )
        self._market_runtimes[market.identity] = runtime
        return runtime

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

    def _markets_for_ids(self, market_ids: set[str] | list[str]) -> list[SelectedMarket]:
        values: list[SelectedMarket] = []
        for market_id in market_ids:
            runtime = self._market_runtimes.get(market_id)
            if runtime is not None:
                values.append(runtime.market)
                continue
            market = self._known_markets.get(market_id)
            if market is not None:
                values.append(market)
        return sort_markets_by_open(values)

    def _active_market_ids_snapshot(self) -> list[str]:
        return [market.identity for market in self._markets_for_ids(self._active_market_ids)]

    def _active_asset_ids(self) -> list[str]:
        asset_ids: list[str] = []
        for market in self._markets_for_ids(self._active_market_ids):
            for asset_id in market.asset_ids:
                if asset_id not in asset_ids:
                    asset_ids.append(asset_id)
        if asset_ids:
            return asset_ids
        if self._selected_market is not None:
            return list(self._selected_market.asset_ids)
        return []

    def _runtime_values(self) -> list[_WsMarketRuntime]:
        if self._market_runtimes:
            ordered = self._markets_for_ids(self._active_market_ids or set(self._market_runtimes))
            runtimes = [self._market_runtimes[market.identity] for market in ordered if market.identity in self._market_runtimes]
            if runtimes:
                return runtimes
        if self._session is not None and self._selected_market is not None:
            if self._session_tracker is None:
                self._session_tracker = PolymarketL2StateTracker()
            return [
                _WsMarketRuntime(
                    market=self._selected_market,
                    session=self._session,
                    tracker=self._session_tracker,
                )
            ]
        return []

    def _prewindow_resync_due(self) -> list[SelectedMarket]:
        """Reconnect once shortly before each active market's decision window.

        The CLOB market websocket can spend minutes draining an initial backlog.
        Source-time datasets/replay need fresh source timestamps near the
        trading window, so mirror the live bot's pre-window hard resync.
        """
        target = float(self.config.market_prewindow_resync_ttc_seconds)
        width = max(0.0, float(self.config.market_prewindow_resync_width_seconds))
        if target <= 0.0 or width <= 0.0:
            return []
        now_s = utc_now_ns() / 1_000_000_000
        due: list[SelectedMarket] = []
        for market in self._markets_for_ids(self._active_market_ids):
            if market.identity in self._prewindow_resynced_market_ids:
                continue
            close_s = market_close_epoch(market, interval_seconds=self.config.market_interval_seconds)
            if close_s is None:
                continue
            ttc = close_s - now_s
            if target - width <= ttc <= target:
                due.append(market)
                self._prewindow_resynced_market_ids.add(market.identity)
        return due

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
            runtime = self._market_runtimes.get(market_id)
            if runtime is None:
                continue
            runtime.session.writer("lifecycle").write(
                lifecycle_event(event=event, details={**session_common_fields(runtime.market), **details})
            )

    def _prune_expired_known_markets(self, *, now: datetime) -> None:
        expired_ids = [
            market_id
            for market_id, market in self._known_markets.items()
            if market_id not in self._active_market_ids
            and market_id not in self._market_runtimes
            and market_has_expired(
                market,
                now=now,
                interval_seconds=self.config.market_interval_seconds,
            )
        ]
        for market_id in expired_ids:
            del self._known_markets[market_id]
            self._prewindow_resynced_market_ids.discard(market_id)

    def _discovery_record(
        self,
        market: SelectedMarket | None,
        *,
        target_slugs: list[str],
        raw_items: list[dict[str, Any]],
        poll_seq: int,
        event: str = "market_selected",
        active_markets: list[SelectedMarket] | None = None,
    ) -> dict[str, Any]:
        recv_ts_ns = utc_now_ns()
        payload = {
            "record_type": "discovery_event",
            "source": "polymarket",
            "api": "gamma",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "poll_seq": poll_seq,
            "event": event,
            "target_slugs": target_slugs,
            "candidate_count": len(raw_items),
            "active_market_ids": [item.identity for item in active_markets or []],
            "active_market_slugs": [item.market_slug for item in active_markets or []],
            "raw": {
                "selected_market": market.raw_market if market is not None else None,
            },
        }
        if market is not None:
            payload.update(session_common_fields(market))
            payload.update(self._market_boundary_fields(market))
        return payload

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
            self._session.writer("lifecycle").write({**record, **session_common_fields(self._session.market), **(details or {})})
