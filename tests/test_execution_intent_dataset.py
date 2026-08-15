from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from polymarket_recorder.execution_intent_audit import audit_execution_intent_frame
from polymarket_recorder.execution_intent_dataset import (
    DATASET_COLUMNS,
    PHASE8_TARGET_COLUMNS,
    materialize_execution_intent_frame,
    write_execution_intent_schema,
)
from polymarket_recorder.execution_replay import ExecutionReplayIndex
from polymarket_recorder.execution_simulator import ExecutionSimulator, ExecutionSimulatorConfig, OrderIntent


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_slug": "m",
                "token_asset_id": "up",
                "token_side_normalized": "UP",
                "recv_ts_ns": 1_000,
                "event_type": "book_state",
                "best_bid": 0.49,
                "best_ask": 0.51,
                "mid": 0.50,
                "spread": 0.02,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "last_trade_price": None,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            },
            {
                "market_slug": "m",
                "token_asset_id": "up",
                "token_side_normalized": "UP",
                "recv_ts_ns": 2_000,
                "event_type": "trade",
                "best_bid": 0.49,
                "best_ask": 0.51,
                "mid": 0.50,
                "spread": 0.02,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "last_trade_price": 0.49,
                "trade_size": 1.0,
                "trade_side": "sell",
                "tick_size": 0.01,
            },
            {
                "market_slug": "m",
                "token_asset_id": "up",
                "token_side_normalized": "UP",
                "recv_ts_ns": 3_000,
                "event_type": "book_state",
                "best_bid": 0.54,
                "best_ask": 0.56,
                "mid": 0.55,
                "spread": 0.02,
                "bid_depth_total": 100.0,
                "ask_depth_total": 100.0,
                "last_trade_price": None,
                "trade_size": 0.0,
                "trade_side": "none",
                "tick_size": 0.01,
            },
        ]
    )


class _LiveReference:
    def __init__(self, *, degraded: bool = False) -> None:
        self.degraded = degraded

    def latest_asof(self, *, asof_ts_ns: int):
        return {
            "recv_ts_ns": asof_ts_ns - 1,
            "btc_usd_price": 100_000.0,
            "price_valid": True,
            "degraded": self.degraded,
            "source_count": 2 if self.degraded else 3,
            "bias_mode": "active_window",
            "bias_active": True,
            "bias_carried_forward": False,
            "applied_bias_age_seconds": 1800.0,
        }


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anchor_sequence_id": "seq1",
                "anchor_source": "test_anchor",
                "anchor_feature_eligible": True,
                "anchor_sequence_complete_eligible": True,
                "phase8_microstructure_exclusion_reason": "",
                "token_registry_match": True,
                "market_registry_trainable": True,
                "price_to_beat": 99_990.0,
                "delta_to_strike": 0.0,
                "abs_delta_to_strike": 0.0,
                "delta_sign": 1,
                "t_since_open_s": 1.5,
                "t_to_close_s": 0.0000085,
                "phase_bucket": "test",
                "event_count_1s": 1,
                "event_count_3s": 3,
                "event_count_5s": 3,
                "event_count_15s": 3,
                "quote_update_count_1s": 1,
                "quote_update_count_3s": 2,
                "quote_update_count_5s": 2,
                "trade_count_1s": 1,
                "trade_count_3s": 1,
                "trade_count_5s": 1,
                "signed_trade_size_1s": -1.0,
                "signed_trade_size_3s": -1.0,
                "signed_trade_size_5s": -1.0,
                "absolute_trade_size_1s": 1.0,
                "absolute_trade_size_3s": 1.0,
                "absolute_trade_size_5s": 1.0,
                "spread_mean_5s": 0.02,
                "spread_min_5s": 0.02,
                "spread_max_5s": 0.02,
                "imbalance_mean_5s": 0.0,
                "imbalance_last": 0.0,
                "imbalance_slope_5s": 0.0,
                "mid_return_1s": 0.0,
                "mid_return_3s": 0.05,
                "mid_return_5s": 0.05,
                "quote_churn_rate_5s": 0.0,
                "depth_change_rate_5s": 0.0,
                "sequence_completeness_rate": 1.0,
            }
        ]
    )


def _dataset_frame(tmp_path: Path, *, degraded: bool = False) -> tuple[pd.DataFrame, Path]:
    schema_path = write_execution_intent_schema(tmp_path / "schema.yaml")
    simulator = ExecutionSimulator(
        ExecutionReplayIndex.from_events(_events()),
        ExecutionSimulatorConfig(require_live_reference=True),
        live_reference_reader=_LiveReference(degraded=degraded),
    )
    intent = OrderIntent.create(
        market_slug="m",
        token_asset_id="up",
        token_side_normalized="UP",
        order_ts_ns=1_500,
        market_close_ts_ns=10_000,
        order_side="buy",
        limit_price=0.49,
        order_size=1.0,
        price_level="join_best_bid",
        source_sequence_id="seq1",
        resolved_side="UP",
    )
    results = simulator.simulate_many([intent])
    frame = materialize_execution_intent_frame(
        results=results,
        anchors=_anchors(),
        target_date=date(2026, 4, 26),
        build_id="test-build",
        build_ts_utc="2026-04-26T00:00:00Z",
        policy_path=Path("config/dataset_policy_v1.yaml"),
        quality_registry_path=Path("data/canonical/quality_registry_v1/quality_registry_v1.parquet"),
    )
    return frame, schema_path


def test_execution_intent_materialization_separates_features_and_outcomes(tmp_path) -> None:
    frame, schema_path = _dataset_frame(tmp_path)
    assert list(frame.columns) == DATASET_COLUMNS
    assert not (set(frame.columns) & PHASE8_TARGET_COLUMNS)
    assert bool(frame.loc[0, "execution_training_eligible"]) is True
    assert frame.loc[0, "delta_to_strike"] == 10.0
    audit = audit_execution_intent_frame(frame=frame, schema_path=schema_path)
    assert audit["critical_count"] == 0


def test_execution_intent_audit_flags_future_live_reference(tmp_path) -> None:
    frame, schema_path = _dataset_frame(tmp_path)
    frame.loc[0, "live_reference_recv_ts_ns"] = frame.loc[0, "order_ts_ns"] + 1
    audit = audit_execution_intent_frame(frame=frame, schema_path=schema_path)
    assert audit["critical_count"] >= 1
    assert any(finding["check"] == "live_reference_causality" for finding in audit["findings"])


def test_degraded_live_reference_is_not_training_eligible(tmp_path) -> None:
    frame, _ = _dataset_frame(tmp_path, degraded=True)
    assert bool(frame.loc[0, "live_price_degraded"]) is True
    assert bool(frame.loc[0, "execution_training_eligible"]) is False


def test_execution_intent_audit_allows_schema_valid_empty_shard(tmp_path) -> None:
    schema_path = write_execution_intent_schema(tmp_path / "schema.yaml")
    frame = pd.DataFrame(columns=DATASET_COLUMNS)
    audit = audit_execution_intent_frame(frame=frame, schema_path=schema_path)
    assert audit["critical_count"] == 0
    assert audit["warning_count"] == 1
    assert audit["ledger"]["balances"] is True
    assert any(finding["check"] == "empty_dataset" for finding in audit["findings"])


def test_execution_intent_audit_flags_phase8_target_leakage(tmp_path) -> None:
    frame, schema_path = _dataset_frame(tmp_path)
    frame["mid_move_1s"] = 0.01
    audit = audit_execution_intent_frame(frame=frame, schema_path=schema_path)
    assert audit["critical_count"] >= 1
    assert any(finding["check"] == "phase8_target_leakage" for finding in audit["findings"])
