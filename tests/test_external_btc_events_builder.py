from canonical.external_btc_events_builder import (
    BINANCE_USDM_QUARANTINE_START_NS,
    continuity_state,
    is_binance_usdm_quarantined,
    normalize_record,
)


def test_quarantine_correctly_applied_for_20260420_0700_utc():
    assert is_binance_usdm_quarantined(BINANCE_USDM_QUARANTINE_START_NS) is True
    row = normalize_record(
        {"recv_ts_ns": BINANCE_USDM_QUARANTINE_START_NS, "best_bid": 100.0, "best_ask": 102.0},
        source="binance_usdm",
        venue="binance_usdm",
        symbol="BTCUSDT",
    )
    assert row["quarantine_flag"] is True


def test_forbidden_source_rejected():
    import pytest

    with pytest.raises(ValueError):
        normalize_record({"recv_ts_ns": 1}, source="bybit", venue="bybit", symbol="BTCUSDT")


def test_mid_price_computation_correct():
    row = normalize_record(
        {"recv_ts_ns": 1, "best_bid": 100.0, "best_ask": 102.0},
        source="binance_spot",
        venue="binance",
        symbol="BTCUSDT",
    )
    assert row["mid_price"] == 101.0


def test_continuity_state_set_correctly():
    assert continuity_state({"continuity_state": "gap"}) == "gap"
    assert continuity_state({"event": "reconnect"}) == "reconnect"
    assert continuity_state({}, {"quality_flags": ["sequence_gap"]}) == "gap"
    assert continuity_state({}) == "ok"


def test_no_forbidden_source_columns_in_output_row():
    row = normalize_record(
        {"recv_ts_ns": 1, "best_bid": 100.0, "best_ask": 101.0},
        source="hyperliquid",
        venue="hyperliquid",
        symbol="BTC",
    )
    assert not {"bybit", "coinbase", "deribit", "fng", "rtds"} & set(row)

