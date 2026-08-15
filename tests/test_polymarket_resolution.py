from __future__ import annotations

import json
from datetime import UTC, datetime

import polymarket_recorder.resolution_recorder as resolution_module

from polymarket_recorder.config import PolymarketRecorderConfig
from polymarket_recorder.resolution_reader import PolymarketResolutionReader
from polymarket_recorder.resolution_recorder import PolymarketResolutionRecorderService
from polymarket_recorder.utils import SelectedMarket


def _fixed_ns() -> int:
    return 1_776_358_506_000_000_000


def _sample_market() -> SelectedMarket:
    return SelectedMarket(
        market_id="2015872",
        condition_id="0xfe89485a7fc1452048840a589c0f792ef622d6cf5fa34053de0e18bc6b85a983",
        market_slug="btc-updown-5m-1776622200",
        question="Bitcoin Up or Down - April 19, 2:10PM-2:15PM ET",
        asset_ids=["111", "222"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso="2026-04-19T18:15:00Z",
        start_time_epoch=1776622200,
        selection_reason="slug_exact",
        raw_market={"eventStartTime": "2026-04-19T18:10:00Z"},
    )


def _market_page_html(*, start_iso: str, end_iso: str, open_price: float, close_price: float) -> str:
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
                                start_iso,
                                "fiveminute",
                                end_iso,
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
    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></body></html>'


def _close_service(service: PolymarketResolutionRecorderService) -> None:
    service._history_writer.close()
    service._resolution_writer.close()
    service._lifecycle_writer.close()


def test_resolution_service_records_canonical_market_resolution_and_normalized_side(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resolution_module, "utc_now_ns", _fixed_ns)
    market = _sample_market()
    service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market])

    service._handle_ws_payload(
        {
            "event_type": "market_resolved",
            "market": market.condition_id,
            "assets_ids": list(market.asset_ids),
            "winning_asset_id": "111",
            "winning_outcome": "Up",
            "timestamp": "1776622526000",
        },
        connection_id="conn-1",
        local_seq=1,
    )
    _close_service(service)

    reader = PolymarketResolutionReader(root=tmp_path)
    resolutions = list(reader.iter_resolutions(target_date="2026-04-16"))
    history = list(reader.iter_history(target_date="2026-04-16"))

    assert len(resolutions) == 1
    assert len(history) == 1
    assert resolutions[0]["market_id"] == market.market_id
    assert resolutions[0]["market_slug"] == market.market_slug
    assert resolutions[0]["winning_asset_id"] == "111"
    assert resolutions[0]["winning_outcome"] == "Up"
    assert resolutions[0]["resolved_side"] == "up"
    assert resolutions[0]["resolved_side_authoritative"] is True
    assert resolutions[0]["resolution_source"] == "polymarket_market_resolved_ws"
    assert history[0]["canonical_action"] == "recorded"


def test_resolution_service_preserves_duplicate_history_but_only_one_canonical_record(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resolution_module, "utc_now_ns", _fixed_ns)
    market = _sample_market()
    service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market])
    payload = {
        "event_type": "market_resolved",
        "market": market.condition_id,
        "assets_ids": list(market.asset_ids),
        "winning_asset_id": "222",
        "winning_outcome": "Down",
        "timestamp": "1776622528000",
    }

    service._handle_ws_payload(payload, connection_id="conn-1", local_seq=1)
    service._handle_ws_payload(payload, connection_id="conn-1", local_seq=2)
    _close_service(service)

    reader = PolymarketResolutionReader(root=tmp_path)
    resolutions = list(reader.iter_resolutions(target_date="2026-04-16"))
    history = list(reader.iter_history(target_date="2026-04-16"))

    assert len(resolutions) == 1
    assert [record["canonical_action"] for record in history] == ["recorded", "duplicate_ignored"]
    assert resolutions[0]["winning_asset_id"] == "222"
    assert resolutions[0]["resolved_side"] == "down"


def test_resolution_service_tracks_rest_state_until_reconciled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resolution_module, "utc_now_ns", _fixed_ns)
    market = _sample_market()
    service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market])
    service._pending_rest_reconcile_market_ids.add(market.identity)

    service._handle_rest_market_state(
        market,
        {
            "id": market.market_id,
            "slug": market.market_slug,
            "closed": True,
            "active": True,
            "closedTime": "2026-04-19 18:15:26+00",
            "resolvedBy": None,
            "umaResolutionStatus": "resolved",
            "resolutionSource": "https://data.chain.link/streams/btc-usd",
            "clobTokenIds": json.dumps(market.asset_ids),
            "outcomes": json.dumps(market.outcomes),
        },
    )
    _close_service(service)

    reader = PolymarketResolutionReader(root=tmp_path)
    history = list(reader.iter_history(target_date="2026-04-16"))
    assert len(history) == 1
    assert history[0]["record_type"] == "market_resolution_state_observation"
    assert history[0]["uma_resolution_status"] == "resolved"
    assert history[0]["winner_candidate_is_definitive"] is False
    assert service._pending_rest_reconcile_market_ids == set()


def test_resolution_service_drops_old_rest_resolved_backfill_from_ws_tracking(monkeypatch, tmp_path) -> None:
    current_ns = int(datetime(2026, 4, 19, 18, 16, 0, tzinfo=UTC).timestamp() * 1_000_000_000)

    def _current_ns() -> int:
        return current_ns

    monkeypatch.setattr(resolution_module, "utc_now_ns", _current_ns)
    market = _sample_market()
    service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market])

    assert service._market_should_be_tracked(market, now=datetime(2026, 4, 19, 18, 14, 0, tzinfo=UTC)) is True

    service._handle_rest_market_state(
        market,
        {
            "id": market.market_id,
            "slug": market.market_slug,
            "closed": True,
            "active": True,
            "closedTime": "2026-04-19 18:15:26+00",
            "resolvedBy": None,
            "umaResolutionStatus": "resolved",
            "resolutionSource": "https://data.chain.link/streams/btc-usd",
        },
    )

    assert service._market_should_be_tracked(market, now=datetime(2026, 4, 19, 18, 16, 0, tzinfo=UTC)) is True

    current_ns += 121_000_000_000
    assert service._market_should_be_tracked(market, now=datetime(2026, 4, 19, 18, 18, 5, tzinfo=UTC)) is False
    _close_service(service)


def test_resolution_service_builds_canonical_record_from_rest_winner_outcome(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(resolution_module, "utc_now_ns", _fixed_ns)
    market = _sample_market()
    service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
    service._remember_markets([market])
    service._handle_rest_market_state(
        market,
        {
            "id": market.market_id,
            "slug": market.market_slug,
            "closed": True,
            "active": True,
            "closedTime": "2026-04-19 18:15:26+00",
            "umaResolutionStatus": "resolved",
            "winningOutcome": "Down",
        },
    )

    record = service._build_rest_canonical_resolution_record(
        market,
        payload={
            "closedTime": "2026-04-19 18:15:26+00",
            "winningOutcome": "Down",
        },
    )
    assert record is not None
    assert record["winning_asset_id"] == "222"
    assert record["winning_outcome"] == "Down"
    assert record["resolved_side"] == "down"
    _close_service(service)


def test_extract_page_resolution_uses_exact_market_window_query() -> None:
    market = _sample_market()
    html = _market_page_html(
        start_iso="2026-04-19T18:10:00Z",
        end_iso="2026-04-19T18:15:00Z",
        open_price=74125.25,
        close_price=74150.50,
    )
    resolution = resolution_module._extract_page_resolution(market, html=html)
    assert resolution is not None
    assert resolution["open_price"] == "74125.25"
    assert resolution["close_price"] == "74150.5"
    assert resolution["resolved_side"] == "up"
    assert resolution["winning_outcome"] == "Up"
