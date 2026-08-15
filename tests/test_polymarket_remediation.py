from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import polymarket_recorder.rest_poller as rest_poller_module
import polymarket_recorder.strike_recorder as strike_recorder_module
import polymarket_recorder.ws_client as ws_client_module

from binance_recorder.compression import ZstdJsonlAppender
from binance_recorder.compression import iter_zstd_jsonl
from polymarket_recorder.config import PolymarketRecorderConfig
from polymarket_recorder.l2 import PolymarketL2StateTracker
from polymarket_recorder.reconstruction import (
    PolymarketTrainingViewReader,
    reconstruct_token_rest_records,
    reconstruct_token_ws_records,
)
from polymarket_recorder.rest_poller import PolymarketRestPollerService
from polymarket_recorder.strike_recorder import PolymarketStrikeRecorderService
from polymarket_recorder.utils import SelectedMarket, session_common_fields
from polymarket_recorder.ws_client import PolymarketRecorderService, _WsMarketRuntime
from polymarket_recorder.writers import PolymarketSessionWriters


def _sample_market() -> SelectedMarket:
    return SelectedMarket(
        market_id="market-1",
        condition_id="condition-1",
        market_slug="btc-updown-5m-1776359700",
        question="Bitcoin Up or Down - sample",
        asset_ids=["111", "222"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso="2026-04-16T17:20:00Z",
        start_time_epoch=1776359700,
        selection_reason="slug_exact",
        raw_market={
            "eventStartTime": "2026-04-16T17:15:00Z",
            "endDate": "2026-04-16T17:20:00Z",
            "events": [{"id": "evt-1", "slug": "event-1"}],
        },
    )


def _overlap_market(*, start: datetime, suffix: str, asset_ids: list[str]) -> SelectedMarket:
    end = datetime.fromtimestamp(int(start.timestamp()) + 300, tz=UTC)
    return SelectedMarket(
        market_id=f"market-{suffix}",
        condition_id=f"condition-{suffix}",
        market_slug=f"btc-updown-5m-{int(start.timestamp())}",
        question="Bitcoin Up or Down - overlap",
        asset_ids=asset_ids,
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        start_time_epoch=int(start.timestamp()),
        selection_reason="slug_exact",
        raw_market={
            "eventStartTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDate": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "events": [{"id": f"evt-{suffix}", "slug": f"event-{suffix}"}],
        },
    )


class _CollectingWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class _FakeSession:
    def __init__(self, market: SelectedMarket, *kinds: str) -> None:
        self.market = market
        self._writers = {kind: _CollectingWriter() for kind in kinds}
        self.token_writers = {asset_id: {} for asset_id in market.asset_ids}

    def writer(self, kind: str) -> _CollectingWriter:
        return self._writers[kind]


class _FakeManagedSession(_FakeSession):
    def __init__(
        self,
        *,
        root: Path | None = None,
        market: SelectedMarket,
        started_at: datetime | None = None,
        kinds: tuple[str, ...] = ("ws", "lifecycle"),
        token_kinds: tuple[str, ...] = (),
        flush_interval_seconds: float = 2.0,
        manifest_enabled: bool = True,
        static_fields_extra: dict | None = None,
    ) -> None:
        del root, started_at, flush_interval_seconds, manifest_enabled, static_fields_extra
        super().__init__(market, *kinds)
        self.closed = False
        self.token_writers = {
            asset_id: {token_kind: _CollectingWriter() for token_kind in token_kinds}
            for asset_id in market.asset_ids
        }

    def close(self) -> None:
        self.closed = True


def _register_runtime(
    service: PolymarketRecorderService,
    market: SelectedMarket,
    session: _FakeSession,
    *,
    tracker: PolymarketL2StateTracker | None = None,
) -> _WsMarketRuntime:
    """Wire a fake session into the service the way `_ensure_runtime` does.

    The recorder tracks one `_WsMarketRuntime` per open market (two are open
    at once across a close boundary), so payload routing goes through
    `_market_runtimes` rather than a single `_session` attribute.
    """
    runtime = _WsMarketRuntime(
        market=market,
        session=session,  # type: ignore[arg-type]
        tracker=tracker if tracker is not None else PolymarketL2StateTracker(),
        common_fields=dict(session_common_fields(market)),
    )
    service._market_runtimes[market.identity] = runtime
    service._selected_market = market
    service._session = session  # type: ignore[assignment]
    return runtime


class _MutableDateTime(datetime):
    current = datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        value = cls.current
        if tz is None:
            return value
        return value.astimezone(tz)


def _next_data_html(*, open_price: float, close_price: float = 74130.0) -> str:
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": [
                                "crypto-prices",
                                "price",
                                "BTC",
                                "2026-04-16T17:15:00Z",
                                "fiveminute",
                                "2026-04-16T17:20:00Z",
                            ],
                            "state": {
                                "data": {
                                    "openPrice": open_price,
                                    "closePrice": close_price,
                                }
                            },
                        }
                    ]
                }
            }
        }
    }
    return (
        '<html><body>'
        'To trade this market, decide whether Bitcoin finishes above the opening "Price to Beat" by 6:45PM ET.'
        f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
        "</body></html>"
    )


def _next_data_html_nested_open_price(*, open_price: float) -> str:
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["crypto-prices", "price", "BTC", "fiveminute", "2026-04-16T17:15:00Z"],
                            "state": {
                                "data": {
                                    "series": {
                                        "window": {
                                            "openPrice": open_price,
                                        }
                                    }
                                }
                            },
                        }
                    ]
                }
            }
        }
    }
    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></body></html>'


def _next_data_html_market_metadata(*, price_to_beat: float) -> str:
    payload = {
        "props": {
            "pageProps": {
                "market": {
                    "marketSlug": "btc-updown-5m-1776359700",
                    "metadata": {"price_to_beat": price_to_beat},
                }
            }
        }
    }
    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></body></html>'


def test_polymarket_config_defaults_disable_token_sidecars(tmp_path) -> None:
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    assert config.enable_token_ws_split is False
    assert config.enable_token_rest_split is False
    assert config.strike_mode == "first_success_or_on_change"
    assert config.strike_debug_include_html is False


async def test_polymarket_ws_service_does_not_create_token_sidecar_writers_by_default(tmp_path) -> None:
    market = _sample_market()
    service = PolymarketRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    await service._switch_market(market, target_slugs=[market.market_slug], raw_items=[market.raw_market], poll_seq=1)
    assert service._session is not None
    assert all(not writer_group for writer_group in service._session.token_writers.values())
    service._session.close()


def test_reconstruct_token_ws_records_splits_multi_token_price_changes() -> None:
    record = {
        "record_type": "ws_event",
        "selected_asset_ids": ["111", "222"],
        "selected_outcomes": ["Up", "Down"],
        "matched_asset_ids": ["111", "222"],
        "matched_outcomes": ["Up", "Down"],
        "payload": {
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "111", "side": "BUY", "price": "0.51", "size": "10"},
                {"asset_id": "222", "side": "SELL", "price": "0.49", "size": "12"},
            ],
        },
    }
    reconstructed = reconstruct_token_ws_records(record)
    assert [item["token_outcome"] for item in reconstructed] == ["Up", "Down"]
    assert reconstructed[0]["payload"]["price_changes"] == [
        {"asset_id": "111", "side": "BUY", "price": "0.51", "size": "10"}
    ]
    assert reconstructed[1]["payload"]["price_changes"] == [
        {"asset_id": "222", "side": "SELL", "price": "0.49", "size": "12"}
    ]


def test_reconstruct_token_rest_records_uses_market_level_token_fields() -> None:
    record = {
        "record_type": "rest_observation",
        "selected_asset_ids": ["111", "222"],
        "selected_outcomes": ["Up", "Down"],
        "endpoint": "clob_order_book",
        "token_asset_id": "111",
        "token_outcome": "Up",
        "payload": {"bids": [], "asks": []},
    }
    reconstructed = reconstruct_token_rest_records(record)
    assert len(reconstructed) == 1
    assert reconstructed[0]["record_type"] == "reconstructed_token_rest_observation"
    assert reconstructed[0]["token_asset_id"] == "111"
    assert reconstructed[0]["token_outcome"] == "Up"


def test_training_view_reader_rebuilds_token_views_from_market_level_files(tmp_path) -> None:
    date_folder = tmp_path / "raw" / "polymarket" / "btc_updown_5m" / "2026-04-16"
    ws_path = date_folder / "00-00-00__sample__abc.ws.jsonl.zst"
    l2_path = date_folder / "00-00-00__sample__abc.l2.jsonl.zst"
    rest_path = date_folder / "00-00-00__sample__abc.rest.jsonl.zst"

    ws_appender = ZstdJsonlAppender(ws_path)
    ws_appender.write_json(
        {
            "record_type": "ws_event",
            "recv_ts_ns": 1,
            "selected_asset_ids": ["111", "222"],
            "selected_outcomes": ["Up", "Down"],
            "matched_asset_ids": ["111", "222"],
            "matched_outcomes": ["Up", "Down"],
            "payload": {
                "event_type": "price_change",
                "price_changes": [
                    {"asset_id": "111", "side": "BUY", "price": "0.51", "size": "10"},
                    {"asset_id": "222", "side": "SELL", "price": "0.49", "size": "12"},
                ],
            },
        }
    )
    ws_appender.close()

    l2_appender = ZstdJsonlAppender(l2_path)
    l2_appender.write_json(
        {
            "record_type": "l2_book_state",
            "recv_ts_ns": 2,
            "token_asset_id": "111",
            "token_outcome": "Up",
        }
    )
    l2_appender.close()

    rest_appender = ZstdJsonlAppender(rest_path)
    rest_appender.write_json(
        {
            "record_type": "rest_observation",
            "recv_ts_ns": 3,
            "selected_asset_ids": ["111", "222"],
            "selected_outcomes": ["Up", "Down"],
            "endpoint": "clob_order_book",
            "token_asset_id": "111",
            "token_outcome": "Up",
            "payload": {"bids": [], "asks": []},
        }
    )
    rest_appender.close()

    reader = PolymarketTrainingViewReader(root=tmp_path)
    reconstructed_ws = list(reader.iter_token_ws(target_date="2026-04-16", token_outcome="Up", finalized_only=False))
    reconstructed_l2 = list(reader.iter_token_l2(target_date="2026-04-16", token_outcome="Up", finalized_only=False))
    reconstructed_rest = list(
        reader.iter_token_rest(target_date="2026-04-16", token_outcome="Up", finalized_only=False)
    )
    assert len(reconstructed_ws) == 1
    assert reconstructed_ws[0]["token_asset_id"] == "111"
    assert len(reconstructed_l2) == 1
    assert reconstructed_l2[0]["token_asset_id"] == "111"
    assert len(reconstructed_rest) == 1
    assert reconstructed_rest[0]["token_asset_id"] == "111"


def test_ws_service_market_level_l2_records_keep_token_outcome() -> None:
    market = _sample_market()
    service = PolymarketRecorderService(PolymarketRecorderConfig(out=Path(".")))
    fake_session = _FakeSession(market, "ws", "l2", "discovery", "lifecycle")
    _register_runtime(service, market, fake_session)
    service._handle_payload(
        {
            "event_type": "book",
            "asset_id": "111",
            "market": "mkt-1",
            "hash": "0xabc",
            "bids": [{"price": "0.45", "size": "10"}],
            "asks": [{"price": "0.55", "size": "8"}],
            "timestamp": 1_776_358_506_123,
        },
        connection_id="conn-1",
        local_seq=1,
    )
    l2_records = fake_session.writer("l2").records
    assert len(l2_records) == 1
    assert l2_records[0]["token_asset_id"] == "111"
    assert l2_records[0]["token_outcome"] == "Up"
    assert l2_records[0]["token_modeling_safe"] is True


def test_ws_service_excludes_off_market_rows_from_selected_market_l2() -> None:
    market = _sample_market()
    service = PolymarketRecorderService(PolymarketRecorderConfig(out=Path(".")))
    fake_session = _FakeSession(market, "ws", "l2", "discovery", "lifecycle")
    _register_runtime(service, market, fake_session)

    service._handle_payload(
        {
            "event_type": "book",
            "asset_id": "111",
            "market": "condition-1",
            "hash": "0xselected",
            "bids": [{"price": "0.45", "size": "10"}],
            "asks": [{"price": "0.55", "size": "8"}],
            "timestamp": 1_776_358_506_123,
        },
        connection_id="conn-1",
        local_seq=1,
    )
    service._handle_payload(
        {
            "event_type": "book",
            "asset_id": "999",
            "market": "external-market",
            "hash": "0xexternal",
            "bids": [{"price": "0.25", "size": "5"}],
            "asks": [{"price": "0.75", "size": "7"}],
            "timestamp": 1_776_358_506_456,
        },
        connection_id="conn-1",
        local_seq=2,
    )

    l2_records = fake_session.writer("l2").records
    assert len(l2_records) == 1
    assert {record["token_asset_id"] for record in l2_records} == {"111"}
    assert {record["token_outcome"] for record in l2_records} == {"Up"}


async def test_strike_service_writes_lean_records_and_emits_only_on_change(tmp_path) -> None:
    market = _sample_market()
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    service = PolymarketStrikeRecorderService(config)
    fake_session = _FakeSession(market, "strike", "lifecycle")
    service._selected_market = market
    service._session = fake_session  # type: ignore[assignment]

    responses = iter(
        [
            "<html><body>Price to beat 74123.5</body></html>",
            "<html><body>Price to beat 74123.5</body></html>",
            "<html><body>Price to beat 74150.0</body></html>",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text=next(responses))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await service._observe_strike(client)
        await service._observe_strike(client)
        await service._observe_strike(client)

    strike_records = fake_session.writer("strike").records
    assert len(strike_records) == 2
    assert strike_records[0]["record_type"] == "strike_observation"
    assert strike_records[0]["price_to_beat"] == "74123.5"
    assert strike_records[0]["parse_status"] == "success"
    assert strike_records[0]["label_safe"] is True
    assert "html_hash" in strike_records[0]
    assert "debug" not in strike_records[0]
    assert "raw" not in strike_records[0]
    assert strike_records[1]["price_to_beat"] == "74150.0"
    assert strike_records[1]["label_safe"] is True


async def test_strike_service_retries_initial_parse_miss_before_emitting_failure(tmp_path) -> None:
    market = _sample_market()
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    service = PolymarketStrikeRecorderService(config)
    fake_session = _FakeSession(market, "strike", "lifecycle")
    service._selected_market = market
    service._session = fake_session  # type: ignore[assignment]

    responses = iter(
        [
            '<html><body>Bitcoin finishes above the opening "Price to Beat" by 6:45PM ET.</body></html>',
            _next_data_html(open_price=74123.5),
        ]
    )

    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request, text=next(responses)))
    async with httpx.AsyncClient(transport=transport) as client:
        await service._observe_strike(client)

    strike_records = fake_session.writer("strike").records
    assert len(strike_records) == 1
    assert strike_records[0]["parse_status"] == "success"
    assert strike_records[0]["price_to_beat"] == "74123.5"
    assert strike_records[0]["capture_timing"]["request_attempt_count"] == 2


async def test_strike_service_prefers_next_data_open_price_over_faq_text(tmp_path) -> None:
    market = _sample_market()
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    service = PolymarketStrikeRecorderService(config)
    fake_session = _FakeSession(market, "strike", "lifecycle")
    service._selected_market = market
    service._session = fake_session  # type: ignore[assignment]

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, request=request, text=_next_data_html(open_price=74123.5))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await service._observe_strike(client)

    strike_records = fake_session.writer("strike").records
    assert len(strike_records) == 1
    assert strike_records[0]["price_to_beat"] == "74123.5"
    assert strike_records[0]["extraction_method"] == "next_data_crypto_prices_open"


def test_strike_service_parses_nested_open_price_variant(tmp_path) -> None:
    service = PolymarketStrikeRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._selected_market = _sample_market()
    parsed = service._parse_strike(_next_data_html_nested_open_price(open_price=74123.5))
    assert parsed.price_to_beat == "74123.5"
    assert parsed.parse_status == "success"


def test_strike_service_parses_market_metadata_variant(tmp_path) -> None:
    service = PolymarketStrikeRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._selected_market = _sample_market()
    parsed = service._parse_strike(_next_data_html_market_metadata(price_to_beat=74123.5))
    assert parsed.price_to_beat == "74123.5"
    assert parsed.extraction_method == "next_data_event_metadata_price_to_beat"


def test_strike_service_ignores_faq_time_false_positive(tmp_path) -> None:
    service = PolymarketStrikeRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    parsed = service._parse_strike(
        (
            '<html><body>'
            'To trade this market, decide whether Bitcoin finishes above the opening "Price to Beat" by 6:45PM ET.'
            "</body></html>"
        )
    )
    assert parsed.price_to_beat is None
    assert parsed.parse_status == "price_to_beat_not_found"


async def test_strike_service_suppresses_transient_post_success_parse_failures(tmp_path) -> None:
    market = _sample_market()
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    service = PolymarketStrikeRecorderService(config)
    fake_session = _FakeSession(market, "strike", "lifecycle")
    service._selected_market = market
    service._session = fake_session  # type: ignore[assignment]

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, request=request, text=_next_data_html(open_price=74123.5))
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await service._observe_strike(client)

    suppressed_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            text='<html><body>Bitcoin finishes above the opening "Price to Beat" by 6:45PM ET.</body></html>',
        )
    )
    async with httpx.AsyncClient(transport=suppressed_transport) as client:
        await service._observe_strike(client)

    strike_records = fake_session.writer("strike").records
    lifecycle_records = fake_session.writer("lifecycle").records
    assert len(strike_records) == 1
    assert strike_records[0]["parse_status"] == "success"
    assert any(record.get("event") == "strike_parse_failure_suppressed" for record in lifecycle_records)


async def test_strike_service_marks_invalid_rows_as_not_label_safe(tmp_path) -> None:
    market = _sample_market()
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    service = PolymarketStrikeRecorderService(config)
    fake_session = _FakeSession(market, "strike", "lifecycle")
    service._selected_market = market
    service._session = fake_session  # type: ignore[assignment]

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            request=request,
            text='<html><body>Bitcoin finishes above the opening "Price to Beat" by 6:45PM ET.</body></html>',
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await service._observe_strike(client)

    strike_record = fake_session.writer("strike").records[0]
    assert strike_record["parse_status"] == "price_to_beat_not_found"
    assert strike_record["price_to_beat"] is None
    assert strike_record["label_safe"] is False


def test_strike_service_default_mode_does_not_store_debug_html(tmp_path) -> None:
    config = PolymarketRecorderConfig.from_env(out=tmp_path)
    assert config.strike_debug_include_html is False


def test_ws_service_marks_new_market_rows_as_not_token_modeling_safe() -> None:
    market = _sample_market()
    service = PolymarketRecorderService(PolymarketRecorderConfig(out=Path(".")))
    fake_session = _FakeSession(market, "ws", "l2", "discovery", "lifecycle")
    _register_runtime(service, market, fake_session)
    service._handle_payload(
        {
            "event_type": "new_market",
            "asset_ids": ["999", "111"],
        },
        connection_id="conn-1",
        local_seq=1,
    )
    record = fake_session.writer("ws").records[0]
    assert record["event_type"] == "new_market"
    assert record["market_scope"] == "contains_external_assets"
    assert record["token_modeling_safe"] is False
    assert record["timestamp_semantics"] == "source_metadata"


def test_ws_service_overlap_mode_adds_next_market_before_close_and_drops_old_at_exact_close(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(ws_client_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(ws_client_module, "PolymarketSessionWriters", _FakeManagedSession)
    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    # Close the retiring market's writers at the boundary instead of lingering,
    # so this test can assert the drop deterministically; the linger path has
    # its own test below.
    config = dataclasses.replace(
        PolymarketRecorderConfig.from_env(out=tmp_path),
        market_close_linger_seconds=0.0,
    )
    service = PolymarketRecorderService(config)
    service._remember_markets([market_a, market_b])

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert service._active_market_ids == {market_a.identity}

    service._connected_asset_ids = list(market_a.asset_ids)
    service._pending_subscription_change = None

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert service._active_market_ids == {market_a.identity, market_b.identity}
    assert service._pending_subscription_change is not None
    assert service._pending_subscription_change.added_asset_ids == ["333", "444"]
    assert service._pending_subscription_change.removed_asset_ids == []
    assert any(
        record.get("event") == "dual_market_overlap_started"
        for record in service._market_runtimes[market_a.identity].session.writer("lifecycle").records
    )
    assert any(
        record.get("event") == "dual_market_overlap_started"
        for record in service._market_runtimes[market_b.identity].session.writer("lifecycle").records
    )

    old_session = service._market_runtimes[market_a.identity].session
    service._connected_asset_ids = [*market_a.asset_ids, *market_b.asset_ids]
    service._pending_subscription_change = None

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 39, 59, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert service._active_market_ids == {market_a.identity, market_b.identity}

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 40, 0, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert service._active_market_ids == {market_b.identity}
    assert old_session.closed is True
    assert service._pending_subscription_change is not None
    assert service._pending_subscription_change.added_asset_ids == []
    assert service._pending_subscription_change.removed_asset_ids == ["111", "222"]


def test_ws_service_lingers_retiring_market_writers_past_close(monkeypatch, tmp_path) -> None:
    """Book events in flight at close must still land in the closing market's files.

    The decision window ends exactly at close, so dropping the writers at the
    boundary truncated the most valuable part of every episode. The runtime is
    therefore retired on a deadline rather than closed inline.
    """
    monkeypatch.setattr(ws_client_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(ws_client_module, "PolymarketSessionWriters", _FakeManagedSession)
    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    config = dataclasses.replace(
        PolymarketRecorderConfig.from_env(out=tmp_path),
        market_close_linger_seconds=30.0,
    )
    service = PolymarketRecorderService(config)
    service._remember_markets([market_a, market_b])

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    old_session = service._market_runtimes[market_a.identity].session

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 40, 0, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")

    # Dropped from the active set, but the writers stay open for the linger.
    assert service._active_market_ids == {market_b.identity}
    assert old_session.closed is False
    assert market_a.identity in service._market_runtimes
    assert market_a.identity in service._retiring_runtime_deadline_ns

    # An event arriving after close still routes to the retiring market.
    service._handle_payload(
        {
            "event_type": "book",
            "asset_id": "111",
            "market": market_a.condition_id,
            "hash": "0xlate",
            "bids": [{"price": "0.45", "size": "10"}],
            "asks": [{"price": "0.55", "size": "8"}],
            "timestamp": 1_776_358_506_123,
        },
        connection_id="conn-1",
        local_seq=1,
    )
    assert len(old_session.writer("l2").records) == 1

    # Once the deadline passes, the runtime is retired and the writers close.
    service._retiring_runtime_deadline_ns[market_a.identity] = 0
    service._refresh_active_markets(reason="test_boundary")
    assert old_session.closed is True
    assert market_a.identity not in service._market_runtimes


@pytest.mark.asyncio
async def test_ws_service_reconnect_rebuilds_full_active_market_set(monkeypatch, tmp_path) -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def recv(self) -> str:
            raise asyncio.CancelledError()

        async def close(self) -> None:
            return None

    class _FakeConnect:
        def __init__(self, ws: _FakeWebSocket) -> None:
            self._ws = ws

        async def __aenter__(self) -> _FakeWebSocket:
            return self._ws

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(ws_client_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(ws_client_module, "PolymarketSessionWriters", _FakeManagedSession)
    fake_ws = _FakeWebSocket()
    monkeypatch.setattr(ws_client_module, "connect", lambda *args, **kwargs: _FakeConnect(fake_ws))

    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    service = PolymarketRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market_a, market_b])
    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")

    with pytest.raises(asyncio.CancelledError):
        await service._consume_websocket("conn-test")

    sent_payload = json.loads(fake_ws.sent[0])
    assert sent_payload["assets_ids"] == ["111", "222", "333", "444"]
    assert any(
        record.get("event") == "reconnect_rebuilt_active_market_set"
        for record in service._market_runtimes[market_a.identity].session.writer("lifecycle").records
    )
    assert any(
        record.get("event") == "reconnect_rebuilt_active_market_set"
        for record in service._market_runtimes[market_b.identity].session.writer("lifecycle").records
    )


@pytest.mark.asyncio
async def test_ws_service_overlap_subscription_change_resends_full_active_asset_set(monkeypatch, tmp_path) -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

    monkeypatch.setattr(ws_client_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(ws_client_module, "PolymarketSessionWriters", _FakeManagedSession)

    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    service = PolymarketRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market_a, market_b])

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    service._connected_asset_ids = list(market_a.asset_ids)

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")

    fake_ws = _FakeWebSocket()
    await service._apply_pending_change(fake_ws, "conn-test")

    subscribe_payload = json.loads(fake_ws.sent[0])
    assert subscribe_payload == {
        "operation": "subscribe",
        "assets_ids": ["111", "222", "333", "444"],
        "custom_feature_enabled": True,
        "initial_dump": True,
    }
    assert service._connected_asset_ids == ["111", "222", "333", "444"]


def test_rest_poller_overlap_mode_keeps_previous_market_alive_until_close(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(rest_poller_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(rest_poller_module, "PolymarketSessionWriters", _FakeManagedSession)
    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market_a, market_b])

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_sessions) == {market_a.identity}

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_sessions) == {market_a.identity, market_b.identity}
    assert any(
        record.get("event") == "dual_market_overlap_started"
        for record in service._market_sessions[market_a.identity].writer("lifecycle").records
    )

    old_session = service._market_sessions[market_a.identity]
    _MutableDateTime.current = datetime(2026, 4, 16, 17, 40, 0, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_sessions) == {market_b.identity}
    assert old_session.closed is True


def test_strike_recorder_overlap_mode_keeps_previous_market_alive_until_close(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(strike_recorder_module, "datetime", _MutableDateTime)
    monkeypatch.setattr(strike_recorder_module, "PolymarketSessionWriters", _FakeManagedSession)
    market_a = _overlap_market(
        start=datetime(2026, 4, 16, 17, 35, tzinfo=UTC),
        suffix="a",
        asset_ids=["111", "222"],
    )
    market_b = _overlap_market(
        start=datetime(2026, 4, 16, 17, 40, tzinfo=UTC),
        suffix="b",
        asset_ids=["333", "444"],
    )
    service = PolymarketStrikeRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market_a, market_b])

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_states) == {market_a.identity}

    _MutableDateTime.current = datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_states) == {market_a.identity, market_b.identity}
    assert any(
        record.get("event") == "dual_market_overlap_started"
        for record in service._market_states[market_b.identity].session.writer("lifecycle").records
    )

    old_session = service._market_states[market_a.identity].session
    _MutableDateTime.current = datetime(2026, 4, 16, 17, 40, 0, tzinfo=UTC)
    service._refresh_active_markets(reason="test_boundary")
    assert set(service._market_states) == {market_b.identity}
    assert old_session.closed is True


def test_lifecycle_writer_flushes_readable_frames_before_close(tmp_path) -> None:
    session = PolymarketSessionWriters(
        root=tmp_path,
        market=_sample_market(),
        started_at=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        kinds=("lifecycle",),
    )
    writer = session.writer("lifecycle")
    writer.write({"record_type": "lifecycle_event", "event": "service_start", "recv_ts_ns": 1})
    writer.write({"record_type": "lifecycle_event", "event": "service_stop", "recv_ts_ns": 2})

    raw_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    records = list(iter_zstd_jsonl(raw_path))
    assert [record["event"] for record in records] == ["service_start", "service_stop"]
    session.close()
