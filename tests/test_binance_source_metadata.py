from binance_recorder.source_metadata import (
    extract_binance_source_timestamps,
    infer_binance_event_type,
    normalize_binance_payload,
)


def test_extract_binance_microsecond_timestamps() -> None:
    payload = {"e": "trade", "E": 1_776_279_001_123_456, "T": 1_776_279_001_223_456}
    result = extract_binance_source_timestamps(payload, unit_hint="microsecond")
    assert result["event_time_us"] == 1_776_279_001_123_456
    assert result["trade_time_us"] == 1_776_279_001_223_456
    assert result["event_time_iso"].endswith("Z")


def test_normalize_binance_mark_price_fields() -> None:
    payload = {"e": "markPriceUpdate", "p": "74100.1", "i": "74105.2", "P": "74120.3", "r": "0.0001", "T": 1776279000000}
    result = normalize_binance_payload(payload)
    assert result["mark_price"] == "74100.1"
    assert result["index_price"] == "74105.2"
    assert result["funding_rate"] == "0.0001"
    assert result["next_funding_time_ms"] == 1776279000000


def test_extract_binance_mark_price_uses_next_funding_time_not_trade_time() -> None:
    payload = {"e": "markPriceUpdate", "E": 1_776_279_001_123, "T": 1_776_279_600_000}
    result = extract_binance_source_timestamps(payload, unit_hint="millisecond")
    assert result["event_time_ms"] == 1_776_279_001_123
    assert result["next_funding_time_ms"] == 1_776_279_600_000
    assert "trade_time_ms" not in result


def test_infer_binance_bookticker_event_type_from_stream() -> None:
    payload = {"s": "BTCUSDT", "u": 123, "b": "74100.1", "B": "1.2", "a": "74101.3", "A": "0.8"}
    assert infer_binance_event_type(payload, stream="btcusdt@bookTicker") == "bookTicker"
    result = normalize_binance_payload(payload, stream="btcusdt@bookTicker")
    assert result["normalized_event_type"] == "bookTicker"
    assert result["best_bid_price"] == "74100.1"
