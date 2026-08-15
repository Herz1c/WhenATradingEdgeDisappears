from __future__ import annotations

import pandas as pd

from polymarket_recorder.execution_audit import audit_execution_results
from polymarket_recorder.execution_ledger import InventoryLedger
from polymarket_recorder.execution_replay import ExecutionReplayIndex, SECOND_NS
from polymarket_recorder.execution_simulator import ExecutionSimulator, ExecutionSimulatorConfig, FeeSchedule, OrderIntent, RESULT_COLUMNS
from polymarket_recorder.execution_validation import run_synthetic_scenario_suite


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
                "trade_size": 5.0,
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


def test_execution_replay_state_is_strictly_before_order_ts() -> None:
    replay = ExecutionReplayIndex.from_events(_events()).require("m", "up")
    state = replay.state_before(1_000)
    assert not state.valid
    state = replay.state_before(2_000)
    assert state.recv_ts_ns == 1_000
    assert state.mid == 0.50


def test_execution_replay_tie_order_is_content_deterministic() -> None:
    rows = [
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
            "recv_ts_ns": 1_000,
            "event_type": "book_state",
            "best_bid": 0.59,
            "best_ask": 0.61,
            "mid": 0.60,
            "spread": 0.02,
            "bid_depth_total": 100.0,
            "ask_depth_total": 100.0,
            "last_trade_price": None,
            "trade_size": 0.0,
            "trade_side": "none",
            "tick_size": 0.01,
        },
    ]
    replay_a = ExecutionReplayIndex.from_events(pd.DataFrame(rows)).require("m", "up")
    replay_b = ExecutionReplayIndex.from_events(pd.DataFrame(list(reversed(rows)))).require("m", "up")
    assert replay_a.state_before(2_000).mid == replay_b.state_before(2_000).mid


def test_post_only_crossing_order_is_rejected() -> None:
    index = ExecutionReplayIndex.from_events(_events())
    simulator = ExecutionSimulator(index)
    intent = OrderIntent.create(
        market_slug="m",
        token_asset_id="up",
        token_side_normalized="UP",
        order_ts_ns=1_500,
        market_close_ts_ns=10_000,
        order_side="buy",
        limit_price=0.51,
        order_size=1.0,
        price_level="crossing_bid",
    )
    result = simulator.simulate_intent(intent)
    assert result["post_only_rejected"] is True
    assert result["filled"] is False


def test_cancel_before_trade_blocks_fill() -> None:
    index = ExecutionReplayIndex.from_events(_events())
    simulator = ExecutionSimulator(index)
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
        cancel_after_ns=100,
    )
    result = simulator.simulate_intent(intent)
    assert result["filled"] is False
    assert result["cancelled"] is True
    assert result["cancel_effective_ts_ns"] < 2_000


def test_partial_fill_accumulates_until_cancel() -> None:
    frame = pd.concat(
        [
            _events(),
            pd.DataFrame(
                [
                    {
                        **_events().iloc[1].to_dict(),
                        "recv_ts_ns": 2_500,
                        "trade_size": 4.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    simulator = ExecutionSimulator(ExecutionReplayIndex.from_events(frame))
    intent = OrderIntent.create(
        market_slug="m",
        token_asset_id="up",
        token_side_normalized="UP",
        order_ts_ns=1_500,
        market_close_ts_ns=10_000,
        order_side="buy",
        limit_price=0.49,
        order_size=10.0,
        price_level="join_best_bid",
        cancel_after_ns=2_000,
    )
    result = simulator.simulate_intent(intent)
    assert result["partial_fill"] is True
    assert result["fill_size"] == 9.0
    assert result["fill_count"] == 2


def test_replace_order_fills_only_after_replace_effective() -> None:
    simulator = ExecutionSimulator(ExecutionReplayIndex.from_events(_events()))
    intent = OrderIntent.create(
        market_slug="m",
        token_asset_id="up",
        token_side_normalized="UP",
        order_ts_ns=1_500,
        market_close_ts_ns=10_000,
        order_side="buy",
        limit_price=0.40,
        order_size=1.0,
        price_level="far_bid",
        replace_after_ns=250,
        replace_limit_price=0.49,
        replace_order_size=1.0,
        replace_price_level="join_best_bid",
        cancel_after_ns=5_000,
    )
    result = simulator.simulate_intent(intent)
    assert result["filled"] is True
    assert result["replaced"] is True
    assert result["fill_ts_ns"] >= result["replace_effective_ts_ns"]


class _LiveReference:
    def latest_asof(self, *, asof_ts_ns: int):
        return {
            "recv_ts_ns": asof_ts_ns - 1,
            "btc_usd_price": 100_000.0,
            "price_valid": True,
            "degraded": False,
            "source_count": 3,
            "bias_mode": "active_window",
            "bias_active": True,
            "bias_carried_forward": False,
            "applied_bias_age_seconds": 1800.0,
        }


def test_live_reference_context_and_fee_schedule_are_applied() -> None:
    simulator = ExecutionSimulator(
        ExecutionReplayIndex.from_events(_events()),
        ExecutionSimulatorConfig(
            fee_schedule=FeeSchedule(schedule_id="test_fee"),
            require_live_reference=True,
        ),
        live_reference_reader=_LiveReference(),
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
    )
    result = simulator.simulate_intent(intent)
    assert result["live_reference_trusted"] is True
    assert result["live_reference_recv_ts_ns"] < result["order_ts_ns"]
    assert result["fees_paid"] == 0.0
    assert result["rebate_earned"] > 0


def test_synthetic_suite_passes_core_invariants() -> None:
    report = run_synthetic_scenario_suite()
    assert report["failed_checks"] == []
    assert report["audit"]["critical_count"] == 0
    assert list(report["results"].columns) == RESULT_COLUMNS


def test_audit_flags_fill_after_cancel() -> None:
    report = run_synthetic_scenario_suite()
    frame = report["results"].copy()
    frame.loc[0, "cancel_effective_ts_ns"] = frame.loc[0, "order_ts_ns"]
    audit = audit_execution_results(frame)
    assert audit["critical_count"] >= 1
    assert any(finding["check"] == "cancel_causality" for finding in audit["findings"])


def test_inventory_ledger_reconciles_buy_cash() -> None:
    ledger = InventoryLedger()
    ledger.apply_result(
        {
            "filled": True,
            "token_asset_id": "up",
            "order_side": "buy",
            "fill_size": 10.0,
            "fill_price": 0.40,
            "fees_paid": 0.01,
            "rebate_earned": 0.0,
        }
    )
    summary = ledger.summary()
    assert summary["fills"] == 1
    assert summary["gross_abs_position"] == 10.0
    assert ledger.cash == -4.01
