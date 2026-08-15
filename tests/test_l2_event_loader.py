from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_recorders.archive import HourlyRotatingJsonlWriter
from polymarket_recorder.l2_event_loader import load_l2_events, normalize_l2_record


def _base_record(**overrides):
    record = {
        "selected_market_slug": "btc-updown-5m-1",
        "selected_asset_ids": ["up-token", "down-token"],
        "selected_outcomes": ["Up", "Down"],
        "token_asset_id": "up-token",
        "token_outcome": "Up",
        "token_modeling_safe": True,
        "recv_ts_ns": 1_000_000_000,
    }
    record.update(overrides)
    return record


def test_normalize_book_state_microprice_zero_depth_falls_back_to_mid() -> None:
    normalized = normalize_l2_record(
        _base_record(
            record_type="l2_book_state",
            event_type="book",
            best_bid="0.40",
            best_ask="0.60",
            spread="0.20",
            depth_totals={"bid_size_total": "0", "ask_size_total": "0"},
        )
    )
    assert normalized is not None
    assert normalized["event_type"] == "book_state"
    assert normalized["microprice"] == 0.5
    assert normalized["imbalance"] == 0.0


def test_normalize_trade_event() -> None:
    normalized = normalize_l2_record(
        _base_record(
            record_type="trade_execution",
            event_type="last_trade_price",
            price="0.55",
            size="12.5",
            side="BUY",
        )
    )
    assert normalized is not None
    assert normalized["event_type"] == "trade"
    assert normalized["trade_size"] == 12.5
    assert normalized["trade_side"] == "buy"
    assert normalized["last_trade_price"] == 0.55


def test_load_l2_events_uses_quality_registry_and_skips_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "data"
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("polymarket", "btc_updown_5m"),
        kind="l2",
    )
    recv_dt = datetime(2026, 4, 19, tzinfo=UTC)
    writer.write(
        _base_record(
            recv_ts_ns=int(recv_dt.timestamp() * 1_000_000_000),
            record_type="l2_book_state",
            event_type="book",
            best_bid="0.49",
            best_ask="0.51",
            depth_totals={"bid_size_total": "10", "ask_size_total": "10"},
        )
    )
    writer.close()
    raw_path = next((root / "raw" / "polymarket" / "btc_updown_5m" / "2026-04-19").glob("*.l2.jsonl.zst"))
    registry_path = root / "canonical" / "quality_registry_v1" / "quality_registry_v1.parquet"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "file_path": str(raw_path).replace("\\", "/"),
                "source_name": "polymarket_l2",
                "source_tier": "primary_training",
                "quality_class": "clean",
                "manifest_quality_class": "finalized_clean",
                "finalized": True,
                "file_state": "finalized",
                "tail_status": "tail_clean",
                "first_recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
                "last_recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
                "quarantine_flag": True,
                "quarantine_reason": "test",
            }
        ]
    ).to_parquet(registry_path, index=False)
    result = load_l2_events(
        root=root,
        target_date=recv_dt.date(),
        policy_path=Path("config/dataset_policy_v1.yaml"),
        quality_registry_path=registry_path,
    )
    assert result.events.empty
    assert result.exclusion_summary["polymarket_l2"]["quality_registry_excluded"] == 1
