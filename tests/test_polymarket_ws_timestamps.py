from polymarket_recorder.utils import build_polymarket_source_timestamps
from polymarket_recorder.ws_client import _iter_payload_items


def test_polymarket_ws_nested_timestamp_fields_are_lifted() -> None:
    payload = {
        "timestamp": 1_776_358_506_123,
        "book": {"timestamp": 1_776_358_506_456},
        "trade": {"lastTradedAt": 1_776_358_506_789},
        "event": {"createdAt": 1_776_358_507_111},
        "non_timestamp_counter": 42,
    }
    result = build_polymarket_source_timestamps(payload)
    assert result["message_timestamp_ms"] == 1_776_358_506_123
    assert result["book_timestamp_ms"] == 1_776_358_506_456
    assert result["trade_last_traded_at_ms"] == 1_776_358_506_789
    assert result["event_created_at_ms"] == 1_776_358_507_111
    assert "non_timestamp_counter_s" not in result


def test_polymarket_ws_initial_dump_list_is_split_into_dict_events() -> None:
    payload = [
        {"event_type": "book", "asset_id": "111"},
        {"event_type": "book", "asset_id": "222"},
        "ignored",
    ]
    result = _iter_payload_items(payload)
    assert [item["asset_id"] for item in result] == ["111", "222"]
