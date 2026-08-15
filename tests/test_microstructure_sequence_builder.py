from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

from polymarket_recorder.microstructure_sequence_builder import (
    AnchorRow,
    SCHEMA_COLUMNS,
    TokenEventIndex,
    _row_from_anchor,
    _rows_from_anchor,
    assert_schema_columns,
    write_microstructure_schema,
)


def _anchor() -> AnchorRow:
    return AnchorRow(
        market_slug="btc-updown-5m-1",
        token_asset_id="up-token",
        token_side_normalized="UP",
        anchor_ts_ns=100_000_000_000,
        anchor_source="coarse",
        market_close_ts_ns=200_000_000_000,
        resolved_side="UP",
        price_to_beat=100.0,
        live_btc_usd=101.0,
        delta_to_strike=1.0,
        abs_delta_to_strike=1.0,
        delta_sign=1,
        t_since_open_s=10.0,
        t_to_close_s=100.0,
        phase_bucket="early",
        live_reference_trusted=True,
        live_price_valid=True,
        live_bias_mode="active_window",
        live_applied_bias_age_seconds=1800.0,
        training_feature_eligible=True,
        exclusion_reason="",
        terminal_margin_to_strike=5.0,
        flip_happened_after_t=False,
    )


def test_schema_columns_exact(tmp_path) -> None:
    schema = write_microstructure_schema(tmp_path / "schema.yaml")
    assert_schema_columns(SCHEMA_COLUMNS, schema)


def test_sequence_row_is_causal_and_targets_are_labels() -> None:
    events = pd.DataFrame(
        [
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 90_000_000_000 + i * 1_000_000_000,
                "event_type": "book_state",
                "best_bid": 0.40 + i * 0.001,
                "best_ask": 0.42 + i * 0.001,
                "mid": 0.41 + i * 0.001,
                "spread": 0.02,
                "microprice": 0.41 + i * 0.001,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.41 + i * 0.001,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            }
            for i in range(20)
        ]
    )
    index = TokenEventIndex(events)
    row = _row_from_anchor(_anchor(), index, 12)
    assert all(ts < row["anchor_ts_ns"] for ts in row["events_recv_ts_ns"] if ts)
    assert row["sequence_length_actual"] == 10
    assert row["sequence_feature_eligible"] is True
    assert row["target_1s_available"] is True
    assert row["mid_move_1s"] > 0


def test_insufficient_history_excludes_sequence() -> None:
    events = pd.DataFrame(
        [
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 99_000_000_000,
                "event_type": "book_state",
                "best_bid": 0.4,
                "best_ask": 0.5,
                "mid": 0.45,
                "spread": 0.1,
                "microprice": 0.45,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.45,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            }
        ]
    )
    row = _row_from_anchor(_anchor(), TokenEventIndex(events), 64)
    assert row["sequence_feature_eligible"] is False
    assert row["microstructure_exclusion_reason"] == "insufficient_event_history"


def test_extreme_mid_return_remains_training_eligible() -> None:
    old_events = [
        {
            "market_slug": "btc-updown-5m-1",
            "token_asset_id": "up-token",
            "token_side_normalized": "UP",
            "recv_ts_ns": (90 + i) * 1_000_000_000,
            "event_type": "book_state",
            "best_bid": 0.09,
            "best_ask": 0.11,
            "mid": 0.10,
            "spread": 0.02,
            "microprice": 0.10,
            "bid_depth_total": 100.0,
            "ask_depth_total": 100.0,
            "imbalance": 0.0,
            "last_trade_price": 0.10,
            "trade_size": 0.0,
            "trade_side": "none",
            "tick_size": 0.01,
        }
        for i in range(9)
    ]
    events = pd.DataFrame(
        [
            *old_events,
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 99_900_000_000,
                "event_type": "book_state",
                "best_bid": 0.69,
                "best_ask": 0.71,
                "mid": 0.70,
                "spread": 0.02,
                "microprice": 0.70,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.70,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            },
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 101_000_000_000,
                "event_type": "book_state",
                "best_bid": 0.70,
                "best_ask": 0.72,
                "mid": 0.71,
                "spread": 0.02,
                "microprice": 0.71,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.71,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            },
        ]
    )
    row = _row_from_anchor(_anchor(), TokenEventIndex(events), 64)
    assert row["sequence_feature_eligible"] is True
    assert row["microstructure_exclusion_reason"] == ""
    assert row["mid_return_1s"] == 0.6


def test_target_availability_respects_close_boundary() -> None:
    events = pd.DataFrame(
        [
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 99_000_000_000 + i * 100_000_000,
                "event_type": "book_state",
                "best_bid": 0.4,
                "best_ask": 0.5,
                "mid": 0.45 + i * 0.001,
                "spread": 0.1,
                "microprice": 0.45,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.45,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            }
            for i in range(20)
        ]
    )
    anchor = replace(_anchor(), t_to_close_s=1.0, market_close_ts_ns=101_000_000_000)
    row = _row_from_anchor(anchor, TokenEventIndex(events), 12)
    assert row["target_1s_available"] is False
    assert pd.isna(row["mid_move_1s"])
    assert row["mid_move_sign_1s"] == 0


def test_excursions_are_non_negative_and_sequence_id_is_shared() -> None:
    events = pd.DataFrame(
        [
            {
                "market_slug": "btc-updown-5m-1",
                "token_asset_id": "up-token",
                "token_side_normalized": "UP",
                "recv_ts_ns": 90_000_000_000 + i * 1_000_000_000,
                "event_type": "book_state",
                "best_bid": 0.7 - i * 0.01,
                "best_ask": 0.72 - i * 0.01,
                "mid": 0.71 - i * 0.01,
                "spread": 0.02,
                "microprice": 0.71 - i * 0.01,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "imbalance": 0.0,
                "last_trade_price": 0.71 - i * 0.01,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            }
            for i in range(20)
        ]
    )
    row64, row128 = _rows_from_anchor(_anchor(), TokenEventIndex(events))
    assert row64["sequence_id"] == row128["sequence_id"]
    assert row64["max_favorable_excursion_5s"] >= 0.0
    assert row64["max_adverse_excursion_5s"] >= 0.0
