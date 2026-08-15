from canonical.secondary_context_events_builder import CAUSAL_FLOOR_NS, normalize_record


def test_causal_available_ts_ns_exactly_1800s_after_recv():
    row = normalize_record({"recv_ts_ns": 100, "chainlink_display_ts": 1, "btc_usd_price": 50000.0})
    assert row["causal_available_ts_ns"] == 100 + CAUSAL_FLOOR_NS


def test_no_forbidden_sources():
    row = normalize_record({"recv_ts_ns": 100, "chainlink_display_ts": 1, "btc_usd_price": 50000.0})
    assert row["source"] == "chainlink_public_delayed"
    assert row["source"] not in {"onchain", "rtds", "fng", "bybit", "coinbase", "deribit"}


def test_1800s_floor_enforced():
    row = normalize_record({"recv_ts_ns": 5_000, "chainlink_display_ts": 1, "btc_usd_price": 50000.0})
    assert row["causal_available_ts_ns"] - row["recv_ts_ns"] == 1_800_000_000_000

