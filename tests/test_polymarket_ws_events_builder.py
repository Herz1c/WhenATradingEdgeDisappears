from canonical.polymarket_ws_events_builder import (
    market_scope,
    normalize_event_type,
    normalize_record,
    token_modeling_safe,
)


def test_event_type_normalization_covers_known_types():
    assert normalize_event_type("price_change") == "price_change"
    assert normalize_event_type("last_trade_price") == "trade"
    assert normalize_event_type("tick_size_change") == "tick_update"
    assert normalize_event_type("open") == "market_open"
    assert normalize_event_type("closed") == "market_close"
    assert normalize_event_type("subscribed") == "system"
    assert normalize_event_type("surprise") == "other"


def test_token_modeling_safe_requires_specific_token_event():
    assert token_modeling_safe({"event_type": "price_change", "asset_id": "token-a"}) is True
    assert token_modeling_safe({"event_type": "market_open", "asset_ids": ["a", "b"]}) is False
    assert token_modeling_safe({"event_type": "trade", "asset_ids": ["a", "b"]}) is False


def test_market_scope_values():
    assert market_scope({"event_type": "trade", "asset_id": "a"}) == "token-level"
    assert market_scope({"market_slug": "btc-updown"}) == "market-level"
    assert market_scope({}) == "system"


def test_normalize_record_no_future_clock_and_summary():
    row = normalize_record(
        {
            "recv_ts_ns": 1_000,
            "recv_ts_iso": "2026-04-28T00:00:00Z",
            "market_slug": "btc",
            "event_type": "trade",
            "asset_id": "asset",
            "price": 0.51,
        }
    )
    assert row["recv_ts_ns"] == 1_000
    assert row["event_type"] == "trade"
    assert row["token_modeling_safe"] is True
    assert row["matched_asset_ids"] == ["asset"]

