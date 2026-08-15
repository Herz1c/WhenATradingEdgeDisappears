from market_recorders.time_utils import build_timestamp_fields


def test_build_timestamp_fields_microsecond_epoch() -> None:
    result = build_timestamp_fields("event_time", 1_776_279_001_123_456)
    assert result["event_time_us"] == 1_776_279_001_123_456
    assert result["event_time_iso"].endswith("Z")


def test_build_timestamp_fields_millisecond_epoch() -> None:
    result = build_timestamp_fields("event_time", 1_776_279_001_123)
    assert result["event_time_ms"] == 1_776_279_001_123
    assert result["event_time_iso"].endswith("Z")
