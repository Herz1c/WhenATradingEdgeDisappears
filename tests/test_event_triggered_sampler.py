import polars as pl

from polymarket_recorder.event_triggered_sampler import EventTriggeredSampler


def _base_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_slug": ["m"] * 8,
            "snapshot_ts_ns": [0, 100_000_000, 200_000_000, 260_000_000, 600_000_000, 1_000_000_000, 1_400_000_000, 1_800_000_000],
            "delta_to_strike": [-1.0, 1.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            "up_token_spread": [1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0],
            "up_token_trade_count_1s": [1, 1, 1, 1, 1, 10, 1, 1],
            "imbalance": [0.0, 0.1, 0.2, 0.25, 0.26, 0.27, 0.8, 0.8],
            "tick_size": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02],
            "binance_spot_mid": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 130.0],
            "live_btc_usd": [100.0] * 8,
            "training_feature_eligible": [True] * 8,
        }
    )


def test_each_trigger_type_fires_correctly():
    out = EventTriggeredSampler().detect_triggers(_base_frame())
    triggers = set(out["trigger_type"].to_list())
    assert {
        "regular",
        "strike_cross",
        "spread_burst",
        "trade_burst",
        "imbalance_change",
        "tick_size_update",
        "cross_source_divergence",
    } <= triggers


def test_250ms_dedup_enforced():
    out = EventTriggeredSampler().detect_triggers(_base_frame())
    strike_rows = out.filter(pl.col("trigger_type") == "strike_cross")
    assert strike_rows.height == 1


def test_no_future_data_in_triggered_snapshots():
    frame = _base_frame()
    out = EventTriggeredSampler().detect_triggers(frame)
    assert out["snapshot_ts_ns"].max() <= frame["snapshot_ts_ns"].max()


def test_schema_identical_to_regular_snapshots_plus_trigger_columns():
    frame = _base_frame()
    out = EventTriggeredSampler().detect_triggers(frame)
    assert set(frame.columns) <= set(out.columns)
    assert {"trigger_type", "trigger_magnitude"} <= set(out.columns)


def test_generate_additional_snapshots_filters_regular_rows():
    frame = _base_frame()
    out = EventTriggeredSampler().generate_additional_snapshots("m", pl.DataFrame(), pl.DataFrame(), frame)
    assert out.height > 0
    assert "regular" not in set(out["trigger_type"].to_list())

