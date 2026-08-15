from pathlib import Path

import httpx

from polymarket_recorder.config import PolymarketRecorderConfig
from polymarket_recorder.discovery import GammaDiscoveryClient
from polymarket_recorder.rest_poller import PolymarketRestPollerService
from polymarket_recorder.utils import SelectedMarket


class _CollectingWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class _FakeSession:
    def __init__(self, market: SelectedMarket) -> None:
        self.market = market
        self._rest_writer = _CollectingWriter()
        self._lifecycle_writer = _CollectingWriter()
        self.token_writers = {
            asset_id: {"token_rest": _CollectingWriter()}
            for asset_id in market.asset_ids
        }

    def writer(self, kind: str) -> _CollectingWriter:
        if kind == "rest":
            return self._rest_writer
        if kind == "lifecycle":
            return self._lifecycle_writer
        raise KeyError(kind)


async def test_poll_quotes_writes_clob_order_book_for_both_asset_ids() -> None:
    market = SelectedMarket(
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
        raw_market={},
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=Path("./data")))
    service._selected_market = market
    fake_session = _FakeSession(market)
    service._session = fake_session  # type: ignore[assignment]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/time":
            return httpx.Response(200, request=request, json=1776358506)
        if path == "/book":
            token_id = request.url.params["token_id"]
            return httpx.Response(
                200,
                request=request,
                json={
                    "bids": [{"price": f"0.{token_id}", "size": "10"}],
                    "asks": [{"price": f"0.9{token_id}", "size": "9"}],
                },
            )
        if path in {"/midpoint", "/spread", "/last-trade-price", "/fee-rate", "/tick-size"}:
            return httpx.Response(200, request=request, json={"ok": True})
        raise AssertionError(f"Unexpected path: {path}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://clob.polymarket.com", transport=transport) as client:
        await service._poll_quotes(client)

    book_records = [
        record for record in fake_session.writer("rest").records if record.get("endpoint") == "clob_order_book"
    ]
    assert len(book_records) == 2
    assert [record["extra"]["asset_id"] for record in book_records] == ["111", "222"]
    assert [record["url"] for record in book_records] == [
        "https://clob.polymarket.com/book?token_id=111",
        "https://clob.polymarket.com/book?token_id=222",
    ]
    server_time_records = [
        record for record in fake_session.writer("rest").records if record.get("endpoint") == "clob_server_time"
    ]
    assert len(server_time_records) == 1
    assert server_time_records[0]["event_ts_kind"] == "source_event"
    assert server_time_records[0]["timestamp_semantics"] == "source_event"
    assert server_time_records[0]["source_event_timestamps"]["server_time_s"] == 1776358506


async def test_gamma_discovery_requests_future_markets_without_active_true_filter() -> None:
    config = PolymarketRecorderConfig.from_env(out=Path("./data"))
    captured_params: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(request.url.params.multi_items())
        return httpx.Response(200, request=request, json=[])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://gamma-api.polymarket.com", transport=transport) as client:
        await GammaDiscoveryClient(client, config).fetch_target_market_pages()

    assert "active" not in captured_params
    assert captured_params["closed"] == "false"


async def test_discover_timeout_keeps_current_market_and_writes_http_error() -> None:
    market = SelectedMarket(
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
        raw_market={},
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=Path("./data")))
    service._selected_market = market
    fake_session = _FakeSession(market)
    service._session = fake_session  # type: ignore[assignment]

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://gamma-api.polymarket.com", transport=transport) as client:
        await service._discover(GammaDiscoveryClient(client, service.config))

    assert service._selected_market is market
    lifecycle_records = fake_session.writer("lifecycle").records
    assert len(lifecycle_records) == 1
    assert lifecycle_records[0]["event"] == "http_error"
    assert lifecycle_records[0]["endpoint"] == "gamma_discovery"
    assert "ReadTimeout" in lifecycle_records[0]["error"]


async def test_poll_quotes_timeout_logs_http_error_and_continues() -> None:
    market = SelectedMarket(
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
        raw_market={},
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=Path("./data")))
    service._selected_market = market
    fake_session = _FakeSession(market)
    service._session = fake_session  # type: ignore[assignment]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/time":
            raise httpx.ReadTimeout("timed out", request=request)
        if path == "/book":
            token_id = request.url.params["token_id"]
            return httpx.Response(
                200,
                request=request,
                json={
                    "bids": [{"price": f"0.{token_id}", "size": "10"}],
                    "asks": [{"price": f"0.9{token_id}", "size": "9"}],
                },
            )
        if path in {"/midpoint", "/spread", "/last-trade-price", "/fee-rate", "/tick-size"}:
            return httpx.Response(200, request=request, json={"ok": True})
        raise AssertionError(f"Unexpected path: {path}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://clob.polymarket.com", transport=transport) as client:
        await service._poll_quotes(client)

    lifecycle_records = fake_session.writer("lifecycle").records
    assert len(lifecycle_records) == 1
    assert lifecycle_records[0]["event"] == "http_error"
    assert lifecycle_records[0]["endpoint"] == "clob_server_time"
    book_records = [
        record for record in fake_session.writer("rest").records if record.get("endpoint") == "clob_order_book"
    ]
    assert len(book_records) == 2


def test_gamma_selected_metadata_uses_recv_fallback_event_time() -> None:
    market = SelectedMarket(
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
            "updatedAt": "2026-04-15T17:20:00Z",
            "events": [{"updatedAt": "2026-04-15T17:21:00Z"}],
        },
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=Path("./data")))
    service._selected_market = market
    fake_session = _FakeSession(market)
    service._session = fake_session  # type: ignore[assignment]

    service._write_selected_metadata()

    market_record = next(
        record for record in fake_session.writer("rest").records if record.get("endpoint") == "gamma_selected_market"
    )
    event_record = next(
        record for record in fake_session.writer("rest").records if record.get("endpoint") == "gamma_selected_event"
    )
    assert market_record["event_ts_ns"] == market_record["recv_ts_ns"]
    assert market_record["event_ts_kind"] == "recv_fallback"
    assert market_record["timestamp_semantics"] == "source_metadata"
    assert market_record["source_metadata_timestamps"]["message_timestamp_ns"] < market_record["recv_ts_ns"]
    assert event_record["event_ts_ns"] == event_record["recv_ts_ns"]
    assert event_record["timestamp_semantics"] == "source_metadata"


async def test_price_history_uses_recv_fallback_collection_semantics() -> None:
    market = SelectedMarket(
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
        raw_market={},
    )
    service = PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=Path("./data")))
    service._selected_market = market
    fake_session = _FakeSession(market)
    service._session = fake_session  # type: ignore[assignment]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"history": [{"timestamp": 1776358506, "p": "0.52"}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://clob.polymarket.com", transport=transport) as client:
        await service._fetch_rest(
            client,
            endpoint="price_history",
            path="/prices-history",
            params={"market": "111", "interval": "max"},
            asset_id="111",
        )

    record = fake_session.writer("rest").records[0]
    assert record["event_ts_ns"] == record["recv_ts_ns"]
    assert record["event_ts_kind"] == "recv_fallback"
    assert record["timestamp_semantics"] == "collection_snapshot"
