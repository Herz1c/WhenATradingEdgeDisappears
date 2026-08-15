from datetime import UTC, datetime

from polymarket_recorder.utils import (
    candidate_target_markets,
    generate_target_slugs,
    select_active_markets,
    select_target_market,
)


def test_generate_target_slugs_window() -> None:
    slugs = generate_target_slugs(
        now=datetime(2026, 4, 16, 17, 0, 30, tzinfo=UTC),
        prefix="btc-updown-5m",
        interval_seconds=300,
        lookback_intervals=1,
        lookahead_intervals=1,
    )
    assert len(slugs) == 3
    assert slugs[1].startswith("btc-updown-5m-")


def test_select_target_market_prefers_exact_slug() -> None:
    now = datetime(2026, 4, 16, 17, 0, 30, tzinfo=UTC)
    target_slugs = ["btc-updown-5m-1776358800"]
    items = [
        {
            "id": "1",
            "conditionId": "0xabc",
            "slug": "btc-updown-5m-1776358800",
            "question": "BTC Up or Down 5m?",
            "clobTokenIds": "[\"111\",\"222\"]",
            "outcomes": "[\"Up\",\"Down\"]",
            "active": True,
            "closed": False,
            "endDate": "2026-04-16T17:10:00Z",
        }
    ]
    selected = select_target_market(items, target_slugs=target_slugs, now=now, stale_grace_seconds=15.0)
    assert selected is not None
    assert selected.market_slug == "btc-updown-5m-1776358800"
    assert selected.asset_ids == ["111", "222"]
    assert selected.outcomes == ["Up", "Down"]


def test_select_active_markets_uses_preopen_overlap_and_exact_close_boundary() -> None:
    market_a_start = int(datetime(2026, 4, 16, 17, 35, tzinfo=UTC).timestamp())
    market_b_start = int(datetime(2026, 4, 16, 17, 40, tzinfo=UTC).timestamp())
    target_slugs = [
        f"btc-updown-5m-{market_a_start}",
        f"btc-updown-5m-{market_b_start}",
    ]
    items = [
        {
            "id": "1",
            "conditionId": "condition-a",
            "slug": target_slugs[0],
            "question": "BTC Up or Down 5m?",
            "clobTokenIds": "[\"111\",\"222\"]",
            "outcomes": "[\"Up\",\"Down\"]",
            "active": True,
            "closed": False,
            "endDate": "2026-04-16T17:40:00Z",
        },
        {
            "id": "2",
            "conditionId": "condition-b",
            "slug": target_slugs[1],
            "question": "BTC Up or Down 5m?",
            "clobTokenIds": "[\"333\",\"444\"]",
            "outcomes": "[\"Up\",\"Down\"]",
            "active": True,
            "closed": False,
            "endDate": "2026-04-16T17:45:00Z",
        },
    ]

    before_overlap = select_active_markets(
        items,
        target_slugs=target_slugs,
        now=datetime(2026, 4, 16, 17, 37, 29, tzinfo=UTC),
        stale_grace_seconds=15.0,
        interval_seconds=300,
        preopen_seconds=150,
    )
    assert [market.condition_id for market in before_overlap] == ["condition-a"]

    in_overlap = select_active_markets(
        items,
        target_slugs=target_slugs,
        now=datetime(2026, 4, 16, 17, 37, 30, tzinfo=UTC),
        stale_grace_seconds=15.0,
        interval_seconds=300,
        preopen_seconds=150,
    )
    assert [market.condition_id for market in in_overlap] == ["condition-a", "condition-b"]

    exact_close = select_active_markets(
        items,
        target_slugs=target_slugs,
        now=datetime(2026, 4, 16, 17, 40, 0, tzinfo=UTC),
        stale_grace_seconds=15.0,
        interval_seconds=300,
        preopen_seconds=150,
    )
    assert [market.condition_id for market in exact_close] == ["condition-b"]


def test_candidate_target_markets_can_include_stale_closed_markets_for_resolution_tracking() -> None:
    now = datetime(2026, 4, 19, 22, 0, tzinfo=UTC)
    target_slugs = ["btc-updown-5m-1776625200"]
    items = [
        {
            "id": "2016138",
            "conditionId": "0xclosed",
            "slug": "btc-updown-5m-1776625200",
            "question": "Bitcoin Up or Down - April 19, 3:00PM-3:05PM ET",
            "clobTokenIds": "[\"111\",\"222\"]",
            "outcomes": "[\"Up\",\"Down\"]",
            "active": True,
            "closed": True,
            "endDate": "2026-04-19T19:05:00Z",
        }
    ]

    assert candidate_target_markets(
        items,
        target_slugs=target_slugs,
        now=now,
        stale_grace_seconds=15.0,
    ) == []

    stale_candidates = candidate_target_markets(
        items,
        target_slugs=target_slugs,
        now=now,
        stale_grace_seconds=15.0,
        allow_stale=True,
    )
    assert len(stale_candidates) == 1
    assert stale_candidates[0].market_id == "2016138"
