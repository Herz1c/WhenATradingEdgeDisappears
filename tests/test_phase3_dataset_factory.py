from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_recorders.archive import HourlyRotatingJsonlWriter
from polymarket_recorder.dataset_factory import build_phase3_datasets, _load_market_labels
from polymarket_recorder.utils import SelectedMarket, session_common_fields
from polymarket_recorder.writers import PolymarketSessionWriters


def _market(*, open_dt: datetime, resolved_side: str, suffix: str) -> SelectedMarket:
    open_epoch = int(open_dt.timestamp())
    return SelectedMarket(
        market_id=f"market-{suffix}",
        condition_id=f"condition-{suffix}",
        market_slug=f"btc-updown-5m-{open_epoch}",
        question=f"Bitcoin Up or Down - {suffix}",
        asset_ids=[f"up-{suffix}", f"down-{suffix}"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso=(open_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        start_time_epoch=open_epoch,
        selection_reason="test_fixture",
        raw_market={},
    )


def _write_market_session_bundle(
    root: Path,
    *,
    market: SelectedMarket,
    started_at: datetime,
    strike_price: float,
) -> None:
    common = session_common_fields(market)
    session = PolymarketSessionWriters(
        root=root,
        market=market,
        started_at=started_at,
        kinds=("l2", "rest", "strike"),
    )
    strike_recv_ns = int(started_at.timestamp() * 1_000_000_000)
    rest_recv_ns = strike_recv_ns + 5_000_000_000
    open_ns = market.start_time_epoch * 1_000_000_000

    session.writer("strike").write(
        {
            **common,
            "record_type": "strike_observation",
            "recv_ts_ns": strike_recv_ns,
            "recv_ts_iso": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selected_condition_id": market.condition_id,
            "price_to_beat": strike_price,
            "label_safe": True,
        }
    )
    session.writer("rest").write(
        {
            **common,
            "record_type": "rest_observation",
            "endpoint": "gamma_discovery",
            "recv_ts_ns": rest_recv_ns,
            "recv_ts_iso": datetime.fromtimestamp(rest_recv_ns / 1_000_000_000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {
                "selected_market": {
                    "makerBaseFee": 1000,
                    "takerBaseFee": 1000,
                    "rewardsMinSize": 50,
                    "rewardsMaxSpread": 4.5,
                    "orderMinSize": 5,
                    "spread": 0.01,
                    "bestBid": 0.49,
                    "bestAsk": 0.51,
                    "lastTradePrice": 0.50,
                }
            },
        }
    )

    for token_outcome, best_bid, best_ask in (("up", 0.49, 0.51), ("down", 0.49, 0.51)):
        session.writer("l2").write(
            {
                **common,
                "record_type": "l2_book_state",
                "event_type": "book",
                "token_modeling_safe": True,
                "recv_ts_ns": open_ns - 1_000_000_000,
                "recv_ts_iso": datetime.fromtimestamp((open_ns - 1_000_000_000) / 1_000_000_000, tz=UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "token_outcome": token_outcome,
                "token_asset_id": f"{token_outcome}-{market.market_slug}",
                "best_bid": f"{best_bid:.2f}",
                "best_ask": f"{best_ask:.2f}",
                "spread": "0.02",
                "depth_totals": {"bid_size_total": "120.0", "ask_size_total": "120.0"},
                "tick_size": "0.01",
                "min_order_size": "5",
                "last_trade_price": "0.50",
            }
        )
        session.writer("l2").write(
            {
                **common,
                "record_type": "trade_execution",
                "event_type": "last_trade_price",
                "token_modeling_safe": True,
                "recv_ts_ns": open_ns + 5_000_000_000,
                "recv_ts_iso": datetime.fromtimestamp((open_ns + 5_000_000_000) / 1_000_000_000, tz=UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "token_outcome": token_outcome,
                "token_asset_id": f"{token_outcome}-{market.market_slug}",
                "price": "0.50",
                "size": "10",
                "side": "BUY",
            }
        )
    session.close()


def _write_resolution(root: Path, *, market: SelectedMarket, resolved_side: str, recv_dt: datetime) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("polymarket", "resolution", "btc_updown_5m"),
        kind="resolution",
    )
    writer.write(
        {
            "record_type": "market_resolution",
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "market_slug": market.market_slug,
            "resolved_side": resolved_side,
            "recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
            "resolved_ts_ns": int((market.start_time_epoch + (5 * 60)) * 1_000_000_000),
        }
    )
    writer.close()


def _write_fng(root: Path, *, target_dt: datetime, value: int, classification: str) -> None:
    writer = HourlyRotatingJsonlWriter(root=root, namespace=("fng",), kind="fng")
    writer.write(
        {
            "record_type": "rest_observation",
            "source": "alternative_me",
            "endpoint": "fear_and_greed",
            "recv_ts_ns": int(target_dt.timestamp() * 1_000_000_000),
            "recv_ts_iso": target_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extracted": {
                "value": value,
                "value_classification": classification,
            },
        }
    )
    writer.close()


def _write_live_reference_shard(root: Path, *, target_day: date, start_dt: datetime, seconds: int) -> None:
    rows: list[dict[str, float | bool | int | str]] = []
    for offset in range(seconds):
        ts = start_dt + timedelta(seconds=offset)
        if offset < 600:
            price = 100.0 + (0.01 * offset)
        else:
            price = 106.0 - (0.015 * (offset - 600))
        rows.append(
            {
                "ts_seconds": int(ts.timestamp()),
                "synthetic_corrected": price,
                "rolling_bias": 0.0,
                "active_window_bias": 0.0,
                "bias_state_age_seconds": 1_800.0 + offset,
                "applied_bias_age_seconds": 1_800.0 + offset,
                "last_bias_observation_age_seconds": 1_800.0 + offset,
                "bias_mode": "active_window",
                "bias_bootstrapped": True,
                "bias_active": True,
                "bias_carried_forward": False,
                "bias_state_stale": False,
                "bias_neutralized": False,
                "bias_observation_count": 3,
                "price_valid": True,
                "degraded": False,
                "source_count": 3,
                "binance_spot_mid": price,
                "binance_usdm_mid": price + 0.5,
                "hyperliquid_mid": price - 0.25,
                "max_cross_source_spread": 0.75,
            }
        )
    out_dir = root / "canonical" / "live_reference_events_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"{target_day.isoformat()}.parquet", index=False)


def _write_uncalibrated_live_reference_shard(root: Path, *, target_day: date, start_dt: datetime, seconds: int) -> None:
    rows: list[dict[str, float | bool | int | str]] = []
    for offset in range(seconds):
        ts = start_dt + timedelta(seconds=offset)
        price = 100.0 + (0.01 * offset)
        rows.append(
            {
                "ts_seconds": int(ts.timestamp()),
                "synthetic_corrected": price,
                "rolling_bias": 0.0,
                "active_window_bias": float("nan"),
                "bias_state_age_seconds": float("nan"),
                "applied_bias_age_seconds": float("nan"),
                "last_bias_observation_age_seconds": 86_400.0 + offset,
                "bias_mode": "unavailable",
                "bias_bootstrapped": True,
                "bias_active": False,
                "bias_carried_forward": False,
                "bias_state_stale": True,
                "bias_neutralized": True,
                "bias_observation_count": 0,
                "price_valid": True,
                "degraded": False,
                "source_count": 3,
                "binance_spot_mid": price,
                "binance_usdm_mid": price + 0.5,
                "hyperliquid_mid": price - 0.25,
                "max_cross_source_spread": 0.75,
            }
        )
    out_dir = root / "canonical" / "live_reference_events_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"{target_day.isoformat()}.parquet", index=False)


def _prepare_phase3_fixture(tmp_path: Path) -> tuple[SelectedMarket, SelectedMarket]:
    day = datetime(2026, 4, 19, tzinfo=UTC)
    market_1 = _market(open_dt=day + timedelta(minutes=5), resolved_side="up", suffix="one")
    market_2 = _market(open_dt=day + timedelta(minutes=10), resolved_side="down", suffix="two")

    _write_market_session_bundle(
        tmp_path,
        market=market_1,
        started_at=day + timedelta(minutes=4, seconds=30),
        strike_price=103.0,
    )
    _write_market_session_bundle(
        tmp_path,
        market=market_2,
        started_at=day + timedelta(minutes=9, seconds=30),
        strike_price=106.0,
    )
    _write_resolution(tmp_path, market=market_1, resolved_side="up", recv_dt=day + timedelta(minutes=10, seconds=5))
    _write_resolution(tmp_path, market=market_2, resolved_side="down", recv_dt=day + timedelta(minutes=15, seconds=5))
    _write_fng(tmp_path, target_dt=day, value=29, classification="Fear")
    _write_live_reference_shard(
        tmp_path,
        target_day=date(2026, 4, 19),
        start_dt=day,
        seconds=1200,
    )
    return market_1, market_2


def test_load_market_labels_scans_adjacent_strike_and_resolution_dates(tmp_path) -> None:
    target_day = date(2026, 4, 19)
    start_of_day = datetime(2026, 4, 19, tzinfo=UTC)
    market_preopen_prev_day = _market(open_dt=start_of_day, resolved_side="up", suffix="prev-day-strike")
    market_resolution_next_day = _market(
        open_dt=datetime(2026, 4, 19, 23, 55, tzinfo=UTC),
        resolved_side="down",
        suffix="next-day-resolution",
    )

    _write_market_session_bundle(
        tmp_path,
        market=market_preopen_prev_day,
        started_at=datetime(2026, 4, 18, 23, 59, tzinfo=UTC),
        strike_price=100.0,
    )
    _write_market_session_bundle(
        tmp_path,
        market=market_resolution_next_day,
        started_at=datetime(2026, 4, 19, 23, 54, tzinfo=UTC),
        strike_price=200.0,
    )
    _write_resolution(
        tmp_path,
        market=market_preopen_prev_day,
        resolved_side="up",
        recv_dt=datetime(2026, 4, 19, 0, 6, tzinfo=UTC),
    )
    _write_resolution(
        tmp_path,
        market=market_resolution_next_day,
        resolved_side="down",
        recv_dt=datetime(2026, 4, 20, 0, 1, tzinfo=UTC),
    )

    labels_by_day, summary = _load_market_labels(
        root=tmp_path,
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=target_day,
        date_to=target_day,
    )

    assert summary["matched_markets"] == 2
    assert summary["missing_resolution_market_count"] == 0
    assert len(labels_by_day[target_day]) == 2


def test_build_phase3_datasets_writes_artifacts_and_market_interval_context(tmp_path) -> None:
    _, market_2 = _prepare_phase3_fixture(tmp_path)

    result = build_phase3_datasets(
        root=tmp_path,
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=date(2026, 4, 19),
        date_to=date(2026, 4, 19),
        dataset_output_dir=tmp_path / "datasets",
        report_path=tmp_path / "docs" / "phase3_dataset_factory_report.md",
        schema_dir=tmp_path / "config" / "schemas",
        quality_registry_path=tmp_path / "canonical" / "quality_registry_v1" / "phase3_market_exclusions.parquet",
        decision_log_path=tmp_path / "docs" / "decision_log.md",
        max_workers=1,
    )

    assert result["matched_markets"] == 2
    assert result["missing_resolution_market_count"] == 0
    assert result["total_rows_by_dataset"]["resolution_snapshot_dataset_v1_coarse"] == 482
    assert result["total_rows_by_dataset"]["resolution_snapshot_dataset_v1_dense_close"] == 600
    assert result["total_rows_by_dataset"]["market_interval_dataset_v1_preopen"] == 2
    assert Path(result["report_path"]).exists()
    assert Path(result["sampling_policy_path"]).exists()
    assert Path(result["quality_registry_path"]).exists()
    assert (tmp_path / "docs" / "decision_log.md").exists()
    assert Path(tmp_path / "config" / "schemas" / "resolution_snapshot_dataset_v1_coarse_schema.yaml").exists()
    assert Path(tmp_path / "config" / "schemas" / "resolution_snapshot_dataset_v1_dense_close_schema.yaml").exists()
    assert Path(tmp_path / "config" / "schemas" / "market_interval_dataset_v1_schema.yaml").exists()

    interval_frame = pd.read_parquet(tmp_path / "datasets" / "market_interval_dataset_v1_first30s" / "2026-04-19.parquet")
    assert not any("fng" in column.lower() for column in interval_frame.columns)
    assert "previous_market_resolved_side_label" in interval_frame.columns
    assert "btc_return_60s" in interval_frame.columns
    assert "prev_interval_btc_open" in interval_frame.columns
    assert "early_direction_sign" in interval_frame.columns
    assert "first_30s_drift_sign" in interval_frame.columns
    assert "training_feature_eligible" in interval_frame.columns
    assert "complete_feature_matrix_eligible" in interval_frame.columns
    assert "exclusion_reason" in interval_frame.columns
    assert "feature_matrix_exclusion_reason" in interval_frame.columns
    assert "live_reference_trusted" in interval_frame.columns
    assert "live_reference_strict_trusted" in interval_frame.columns
    assert "live_reference_carried_trusted" in interval_frame.columns
    assert "live_bias_mode" in interval_frame.columns
    assert "live_bias_carried_forward" in interval_frame.columns
    assert "market_5m_complete_rate" in interval_frame.columns
    assert "market_full_matrix_trainable" in interval_frame.columns
    assert "full_matrix_training_eligible" in interval_frame.columns
    assert "full_matrix_exclusion_reason" in interval_frame.columns
    assert "token_mid_complete" in interval_frame.columns
    assert "coupling_anomaly" in interval_frame.columns
    assert "crossed_book" in interval_frame.columns

    row = interval_frame.loc[interval_frame["market_slug"] == market_2.market_slug].iloc[0]
    assert int(row["prior_market_count"]) == 1
    assert int(row["previous_market_resolved_side_label"]) == 1
    assert int(row["close_to_open_btc_sign"]) == 0
    assert int(row["first_30s_drift_sign"]) == 0
    assert bool(row["training_feature_eligible"]) is True
    assert bool(row["live_reference_trusted"]) is True
    assert bool(row["live_reference_strict_trusted"]) is True
    assert bool(row["live_reference_carried_trusted"]) is True
    assert bool(row["live_bias_carried_forward"]) is False
    assert str(row["live_bias_mode"]) == "active_window"
    assert bool(row["token_mid_complete"]) is True
    assert bool(row["market_full_matrix_trainable"]) is True
    assert bool(row["full_matrix_training_eligible"]) is True
    assert float(row["market_5m_complete_rate"]) == 1.0
    assert str(row["exclusion_reason"]) == ""
    assert str(row["full_matrix_exclusion_reason"]) == ""

    report_text = (tmp_path / "docs" / "phase3_dataset_factory_report.md").read_text(encoding="utf-8")
    assert "Phase 3 Dataset Factory Report" in report_text
    assert "missing_resolution_market_count: `0`" in report_text
    assert "Validation Summary" in report_text


def test_build_phase3_datasets_marks_base_proxy_fallback_untrusted_for_training(tmp_path) -> None:
    _prepare_phase3_fixture(tmp_path)
    _write_uncalibrated_live_reference_shard(
        tmp_path,
        target_day=date(2026, 4, 19),
        start_dt=datetime(2026, 4, 19, tzinfo=UTC),
        seconds=1200,
    )

    build_phase3_datasets(
        root=tmp_path,
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=date(2026, 4, 19),
        date_to=date(2026, 4, 19),
        dataset_output_dir=tmp_path / "datasets",
        report_path=tmp_path / "docs" / "phase3_dataset_factory_report.md",
        schema_dir=tmp_path / "config" / "schemas",
        quality_registry_path=tmp_path / "canonical" / "quality_registry_v1" / "phase3_market_exclusions.parquet",
        decision_log_path=tmp_path / "docs" / "decision_log.md",
        max_workers=1,
    )

    frame = pd.read_parquet(tmp_path / "datasets" / "resolution_snapshot_dataset_v1_dense_close" / "2026-04-19.parquet")
    assert bool(frame["live_price_valid"].all()) is True
    assert not bool(frame["live_reference_trusted"].any())
    assert not bool(frame["live_reference_strict_trusted"].any())
    assert not bool(frame["live_reference_carried_trusted"].any())
    assert not bool(frame["full_matrix_training_eligible"].any())
    assert not bool(frame["training_feature_eligible"].any())
    assert set(frame["live_bias_mode"]) == {"unavailable"}
    assert set(frame["exclusion_reason"]) == {"live_reference_untrusted"}
    assert frame["delta_to_strike"].notna().all()


def test_build_phase3_datasets_is_deterministic_and_decision_log_is_idempotent(tmp_path) -> None:
    _prepare_phase3_fixture(tmp_path)
    build_kwargs = {
        "root": tmp_path,
        "policy_path": Path("config/dataset_policy_v1.yaml"),
        "date_from": date(2026, 4, 19),
        "date_to": date(2026, 4, 19),
        "dataset_output_dir": tmp_path / "datasets",
        "report_path": tmp_path / "docs" / "phase3_dataset_factory_report.md",
        "schema_dir": tmp_path / "config" / "schemas",
        "quality_registry_path": tmp_path / "canonical" / "quality_registry_v1" / "phase3_market_exclusions.parquet",
        "decision_log_path": tmp_path / "docs" / "decision_log.md",
        "max_workers": 1,
    }

    first = build_phase3_datasets(**build_kwargs)
    second = build_phase3_datasets(**build_kwargs)

    assert first["hashes"] == second["hashes"]
    decision_log = (tmp_path / "docs" / "decision_log.md").read_text(encoding="utf-8")
    assert decision_log.count("## 2026-04-23 - Phase 3 Sampling Policy Freeze") == 1


def test_build_phase3_datasets_preserves_existing_out_of_range_shards(tmp_path) -> None:
    _prepare_phase3_fixture(tmp_path)
    build_kwargs = {
        "root": tmp_path,
        "policy_path": Path("config/dataset_policy_v1.yaml"),
        "dataset_output_dir": tmp_path / "datasets",
        "report_path": tmp_path / "docs" / "phase3_dataset_factory_report.md",
        "schema_dir": tmp_path / "config" / "schemas",
        "quality_registry_path": tmp_path / "canonical" / "quality_registry_v1" / "phase3_market_exclusions.parquet",
        "decision_log_path": tmp_path / "docs" / "decision_log.md",
        "max_workers": 1,
    }

    build_phase3_datasets(
        **build_kwargs,
        date_from=date(2026, 4, 19),
        date_to=date(2026, 4, 19),
    )
    existing_path = tmp_path / "datasets" / "resolution_snapshot_dataset_v1_coarse" / "2026-04-19.parquet"
    existing_bytes = existing_path.read_bytes()

    build_phase3_datasets(
        **build_kwargs,
        date_from=date(2026, 4, 20),
        date_to=date(2026, 4, 20),
    )

    assert existing_path.exists()
    assert existing_path.read_bytes() == existing_bytes
