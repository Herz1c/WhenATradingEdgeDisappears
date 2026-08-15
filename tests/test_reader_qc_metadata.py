from __future__ import annotations

import json
from datetime import UTC, date, datetime

from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.qc import build_daily_qc_report
from market_recorders.transport_qc import TransportTelemetryAccumulator
from market_recorders.unified_reader import UnifiedRawReader
from polymarket_recorder.utils import SelectedMarket
from polymarket_recorder.writers import PolymarketSessionWriters


def test_manifest_and_reader_metadata_include_source_tier_and_quality_class(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("binance", "spot", "BTCUSDT"),
        kind="ws",
    )
    writer.write(
        {
            "record_type": "ws_event",
            "source": "binance",
            "venue": "spot",
            "symbol": "BTCUSDT",
            "recv_ts_ns": 1_776_358_506_000_000_000,
        }
    )
    writer.close()

    manifest_path = next((tmp_path / "manifests").rglob("*.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reader = UnifiedRawReader(root=tmp_path)
    metadata = reader.file_metadata(next((tmp_path / "raw").rglob("*.jsonl.zst")))

    assert manifest["source_tier"] == "primary_training"
    assert manifest["quality_class"] == "finalized_clean"
    assert metadata["source_tier"] == "primary_training"
    assert metadata["quality_class"] == "finalized_clean"


def test_polymarket_manifests_persist_source_tier_on_disk(tmp_path) -> None:
    market = SelectedMarket(
        market_id="market-1",
        condition_id="condition-1",
        market_slug="btc-updown-5m-1776359700",
        question="Bitcoin Up or Down - sample",
        asset_ids=["111", "222"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso="2026-04-16T17:20:00Z",
        start_time_epoch=1776359700,
        selection_reason="slug_exact",
        raw_market={},
    )
    session = PolymarketSessionWriters(
        root=tmp_path,
        market=market,
        started_at=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        kinds=("l2",),
    )
    session.writer("l2").write(
        {
            "record_type": "l2_book_state",
            "recv_ts_ns": 1_776_358_506_000_000_000,
            "token_asset_id": "111",
            "token_outcome": "Up",
        }
    )
    session.close()

    manifest_path = next((tmp_path / "manifests").rglob("*.l2.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_tier"] == "primary_training"


def test_polymarket_manifests_persist_market_window_and_runtime_metadata(tmp_path) -> None:
    market = SelectedMarket(
        market_id="market-1",
        condition_id="condition-1",
        market_slug="btc-updown-5m-1776359700",
        question="Bitcoin Up or Down - sample",
        asset_ids=["111", "222"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso="2026-04-16T17:20:00Z",
        start_time_epoch=1776359700,
        selection_reason="slug_exact",
        raw_market={"eventStartTime": "2026-04-16T17:15:00Z"},
    )
    session = PolymarketSessionWriters(
        root=tmp_path,
        market=market,
        started_at=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        kinds=("lifecycle",),
        static_fields_extra={
            "runtime_session_id": "runtime-1",
            "recorder_service": "polymarket_ws",
        },
    )
    session.writer("lifecycle").write(
        {
            "record_type": "lifecycle_event",
            "event": "service_start",
            "recv_ts_ns": 1_776_358_506_000_000_000,
        }
    )
    session.close()

    manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_session_id"] == "runtime-1"
    assert manifest["recorder_service"] == "polymarket_ws"
    assert manifest["market_open_iso"] == "2026-04-16T17:15:00.000000Z"
    assert manifest["market_close_iso"] == "2026-04-16T17:20:00.000000Z"


def test_polymarket_resolution_manifests_are_primary_training(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("polymarket", "resolution", "btc_updown_5m"),
        kind="resolution",
    )
    writer.write(
        {
            "record_type": "market_resolution",
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "market_id": "2015872",
            "market_slug": "btc-updown-5m-1776622200",
            "resolved_side": "up",
            "recv_ts_ns": 1_776_358_506_000_000_000,
        }
    )
    writer.close()

    manifest_path = next((tmp_path / "manifests").rglob("*.resolution.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_tier"] == "primary_training"


def test_reader_wrappers_default_to_finalized_only_behavior(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("coinbase", "advanced", "BTC-USD"),
        kind="ws",
    )
    writer.write(
        {
            "record_type": "ws_event",
            "source": "coinbase",
            "venue": "advanced",
            "product_id": "BTC-USD",
            "recv_ts_ns": 1_776_358_506_000_000_000,
        }
    )

    from coinbase_recorder.reader import CoinbaseAdvancedRawReader

    reader = CoinbaseAdvancedRawReader(root=tmp_path)
    assert list(reader.iter_kind(target_date="2026-04-16", kind="ws")) == []
    assert list(reader.iter_kind(target_date="2026-04-16", kind="ws", finalized_only=False, tail_tolerant=True)) != []


def test_transport_qc_computes_lag_and_gap_summary() -> None:
    accumulator = TransportTelemetryAccumulator(source_key="coinbase/advanced/BTC-USD")
    accumulator.observe(
        {
            "recv_ts_ns": 2_000_000_000,
            "source_timestamps": {"message_timestamp_ns": 1_500_000_000},
        }
    )
    accumulator.observe(
        {
            "recv_ts_ns": 8_500_000_000,
            "source_timestamps": {"message_timestamp_ns": 8_600_000_000},
        }
    )
    summary = accumulator.summary()
    assert summary["lag_sample_count"] == 2
    assert summary["negative_lag_count"] == 1
    assert summary["silent_gap_count"] == 1
    assert "negative_lag_observed" in summary["warnings"]


def test_transport_qc_skips_metadata_only_timestamp_semantics() -> None:
    accumulator = TransportTelemetryAccumulator(source_key="deribit/options/BTC")
    accumulator.observe(
        {
            "recv_ts_ns": 2_000_000_000,
            "timestamp_semantics": "metadata_only_sparse",
            "source_timestamps": {"timestamp_ms": 1_500},
        }
    )
    summary = accumulator.summary()
    assert summary["sample_count"] == 1
    assert summary["lag_sample_count"] == 0


def test_qc_report_includes_chainlink_validation_and_source_tiers(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink_public_delayed", "public_stream_page", "BTCUSD"),
        kind="page",
    )
    writer.write(
        {
            "record_type": "chainlink_public_delayed_observation",
            "source": "chainlink_public_delayed",
            "backend": "public_stream_page",
            "symbol": "BTCUSD",
            "recv_ts_ns": 1_776_358_506_000_000_000,
            "recv_ts_iso": "2026-04-16T17:15:06Z",
            "recv_ts_source": "time.time_ns",
            "chainlink_display_ts": 1_776_358_446,
            "btc_usd_price": "74100.0",
            "capture_timing": {},
            "parse_status": "success",
            "parse_error": None,
        }
    )
    writer.close()
    report = build_daily_qc_report(tmp_path, date(2026, 4, 16))
    source_bucket = report["sources"]["chainlink_public_delayed/public_stream_page/BTCUSD"]
    assert source_bucket["source_tier_counts"]["verification_only"] == 1
    assert "chainlink_proxy_validation" in report
