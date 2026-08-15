from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pytest
import zstandard as zstd

from binance_recorder.compression import ZstdJsonlAppender
from market_recorders.archive import HourlyRotatingJsonlWriter, rebuild_missing_manifests
from market_recorders.qc import build_daily_qc_report
from market_recorders.unified_reader import ActiveFileReadRequiresTailTolerantError, UnifiedRawReader
from polymarket_recorder.utils import SelectedMarket
from polymarket_recorder.writers import PolymarketSessionWriters


def _sample_market() -> SelectedMarket:
    return SelectedMarket(
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


def test_hourly_writer_close_without_records_creates_no_placeholder_files(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "empty"),
        kind="ws",
    )
    writer.close()
    assert not any((tmp_path / "raw").rglob("*.jsonl.zst"))
    assert not any((tmp_path / "manifests").rglob("*.manifest.json"))


def test_polymarket_session_close_without_records_creates_no_placeholder_files(tmp_path) -> None:
    session = PolymarketSessionWriters(
        root=tmp_path,
        market=_sample_market(),
        started_at=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        kinds=("ws", "l2", "rest", "strike", "discovery", "lifecycle"),
    )
    session.close()
    assert not any((tmp_path / "raw").rglob("*.jsonl.zst"))
    assert not any((tmp_path / "manifests").rglob("*.manifest.json"))


def test_rebuild_missing_manifests_marks_recovered_and_non_finalized(tmp_path) -> None:
    raw_path = tmp_path / "raw" / "tests" / "recovery" / "2026-04-16" / "17.ws.jsonl.zst"
    appender = ZstdJsonlAppender(raw_path)
    appender.write_json({"record_type": "ws_event", "source": "tests", "recv_ts_ns": 1_776_358_506_000_000_000})
    appender.close()

    report = rebuild_missing_manifests(root=tmp_path)
    manifest_path = tmp_path / "manifests" / "tests" / "recovery" / "2026-04-16" / "17.ws.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report["rebuilt_manifest_files"] == ["tests/recovery/2026-04-16/17.ws.manifest.json"]
    assert manifest["recovered_from_raw"] is True
    assert manifest["finalized"] is False
    assert manifest["file_state"] == "recovered"
    assert manifest["tail_status"] == "recovered_requires_review"
    assert manifest["manifest_record_count_source"] == "readable_rows"


def test_qc_reports_non_finalized_and_recovered_manifests(tmp_path) -> None:
    raw_path = tmp_path / "raw" / "tests" / "qc" / "2026-04-16" / "17.ws.jsonl.zst"
    appender = ZstdJsonlAppender(raw_path)
    appender.write_json({"record_type": "ws_event", "source": "tests", "recv_ts_ns": 1_776_358_506_000_000_000})
    appender.close()
    rebuild_missing_manifests(root=tmp_path)

    report = build_daily_qc_report(tmp_path, datetime(2026, 4, 16, tzinfo=UTC).date())
    assert report["non_finalized_manifest_files"] == ["tests/qc/2026-04-16/17.ws.manifest.json"]
    assert report["recovered_manifest_files"] == ["tests/qc/2026-04-16/17.ws.manifest.json"]
    assert report["sources"]["tests/qc/2026-04-16"]["manifest_file_state_counts"]["recovered"] == 1
    assert report["summary"]["sources_with_recovered_files"] == ["tests/qc/2026-04-16"]


def test_hourly_writer_active_manifest_marks_tail_as_non_finalized(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "active"),
        kind="ws",
    )
    writer.write({"record_type": "ws_event", "recv_ts_ns": 1_776_358_506_000_000_000})

    manifest_path = (
        tmp_path
        / "manifests"
        / "tests"
        / "active"
        / "2026-04-16"
        / "16.ws.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["finalized"] is False
    assert manifest["file_state"] == "active"
    assert manifest["tail_status"] == "tail_may_be_partial"
    writer.close()


def test_hourly_writer_advance_time_finalizes_prior_hour_without_new_rows(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "advance"),
        kind="lifecycle",
    )
    writer.write({"record_type": "lifecycle_event", "event": "service_start", "recv_ts_ns": 1_776_358_506_000_000_000})
    changed = writer.advance_time(now_ns=1_776_362_401_000_000_000)
    assert changed is True

    manifest_path = (
        tmp_path
        / "manifests"
        / "tests"
        / "advance"
        / "2026-04-16"
        / "16.lifecycle.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["finalized"] is True
    assert manifest["file_state"] == "finalized"


def test_hourly_writer_resumes_same_hour_manifest_counts_after_restart(tmp_path) -> None:
    first = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink", "onchain", "BTCUSD"),
        kind="onchain",
    )
    first.write({"record_type": "onchain_record", "recv_ts_ns": 1_776_358_506_000_000_000})
    assert first._state is not None
    first._state.appender.close()
    first._state = None

    second = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink", "onchain", "BTCUSD"),
        kind="onchain",
    )
    second.write({"record_type": "onchain_record", "recv_ts_ns": 1_776_358_507_000_000_000})
    second.close()

    manifest_path = (
        tmp_path
        / "manifests"
        / "chainlink"
        / "onchain"
        / "BTCUSD"
        / "2026-04-16"
        / "16.onchain.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["finalized"] is True
    assert manifest["record_count"] == 2
    assert manifest["readable_record_count"] == 2
    assert manifest["first_recv_ts_ns"] == 1_776_358_506_000_000_000
    assert manifest["last_recv_ts_ns"] == 1_776_358_507_000_000_000

    first_lifecycle = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink", "onchain", "BTCUSD"),
        kind="lifecycle",
    )
    first_lifecycle.write({"record_type": "lifecycle_event", "event": "service_start", "recv_ts_ns": 1_776_358_506_500_000_000})
    assert first_lifecycle._state is not None
    first_lifecycle._state.appender.close()
    first_lifecycle._state = None

    second_lifecycle = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink", "onchain", "BTCUSD"),
        kind="lifecycle",
    )
    second_lifecycle.write({"record_type": "lifecycle_event", "event": "service_stop", "recv_ts_ns": 1_776_358_507_500_000_000})
    second_lifecycle.close()

    lifecycle_manifest_path = (
        tmp_path
        / "manifests"
        / "chainlink"
        / "onchain"
        / "BTCUSD"
        / "2026-04-16"
        / "16.lifecycle.manifest.json"
    )
    lifecycle_manifest = json.loads(lifecycle_manifest_path.read_text(encoding="utf-8"))
    assert lifecycle_manifest["record_count"] == 2
    assert lifecycle_manifest["readable_record_count"] == 2


def test_unified_reader_finalized_only_skips_active_files(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "finalized_only"),
        kind="ws",
    )
    writer.write({"record_type": "ws_event", "recv_ts_ns": 1_776_358_506_000_000_000})

    reader = UnifiedRawReader(root=tmp_path)
    files = reader.list_files_for_date(
        relative_prefix=Path("tests") / "finalized_only",
        target_date="2026-04-16",
        kind="ws",
        finalized_only=True,
    )
    assert files == []
    writer.close()


def test_tail_tolerant_reader_drops_partial_final_line(tmp_path) -> None:
    raw_path = tmp_path / "raw" / "tests" / "partial_tail" / "2026-04-16" / "17.ws.jsonl.zst"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("wb") as file_handle:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(file_handle, closefd=False) as writer:
            writer.write(orjson.dumps({"record_type": "ws_event", "recv_ts_ns": 1}) + b"\n")
            writer.write(b'{"record_type":"ws_event","recv_ts_ns":')
            writer.flush(zstd.FLUSH_FRAME)

    reader = UnifiedRawReader(root=tmp_path)
    with pytest.raises(orjson.JSONDecodeError):
        list(reader.iter_file(raw_path))
    records = list(reader.iter_file(raw_path, tail_tolerant=True))
    assert records == [{"record_type": "ws_event", "recv_ts_ns": 1}]


def test_active_manifest_requires_explicit_tail_tolerant_read(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "guarded_active"),
        kind="ws",
    )
    writer.write({"record_type": "ws_event", "recv_ts_ns": 1_776_358_506_000_000_000})
    raw_path = next((tmp_path / "raw").rglob("*.ws.jsonl.zst"))

    reader = UnifiedRawReader(root=tmp_path)
    with pytest.raises(ActiveFileReadRequiresTailTolerantError):
        list(reader.iter_file(raw_path, finalized_only=False, tail_tolerant=False))

    records = list(reader.iter_file(raw_path, finalized_only=False, tail_tolerant=True))
    assert records == [{"record_type": "ws_event", "recv_ts_ns": 1_776_358_506_000_000_000}]
    writer.close()


def test_polymarket_lifecycle_manifest_uses_readable_row_count_on_finalize(tmp_path) -> None:
    session = PolymarketSessionWriters(
        root=tmp_path,
        market=_sample_market(),
        started_at=datetime(2026, 4, 16, 17, 0, tzinfo=UTC),
        kinds=("lifecycle",),
    )
    writer = session.writer("lifecycle")
    writer.write({"record_type": "lifecycle_event", "event": "service_start", "recv_ts_ns": 1})
    writer.write({"record_type": "lifecycle_event", "event": "service_stop", "recv_ts_ns": 2})
    writer._accumulator.record_count = 1

    session.close()

    manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 2
    assert manifest["readable_record_count"] == 2
    assert manifest["observed_record_count"] == 1
    assert manifest["manifest_record_count_source"] == "readable_rows"


def test_qc_flags_sources_with_only_lifecycle_output(tmp_path) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("tests", "lifecycle_only"),
        kind="lifecycle",
    )
    writer.write({"record_type": "lifecycle_event", "event": "service_start", "recv_ts_ns": 1_776_358_506_000_000_000})
    writer.close()
    report = build_daily_qc_report(tmp_path, datetime(2026, 4, 16, tzinfo=UTC).date())
    assert report["summary"]["sources_with_only_lifecycle_output"] == ["tests/lifecycle_only/2026-04-16"]


def test_audit_document_matches_post_remediation_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    # The document lives under docs/reports/; the repo-root location is the
    # historical one and is still accepted so the test works in either layout.
    candidates = [
        root / "docs" / "reports" / "RECORDER_FULL_AUDIT_DESCRIPTION.txt",
        root / "RECORDER_FULL_AUDIT_DESCRIPTION.txt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    assert path is not None, f"audit document not found in any of {candidates}"
    audit_text = path.read_text(encoding="utf-8")
    assert ".strike.jsonl.zst" in audit_text
    assert "token_ws sidecars are disabled by default" in audit_text
    assert "token_rest sidecars are disabled by default" in audit_text
    assert "token-level L2 is carried inside the market-level .l2 stream via token_asset_id and token_outcome" in audit_text
    assert "`.strike` replaced dense `.page`" in audit_text
    assert "ENABLE_POLYMARKET_STRIKE" in audit_text
    assert "direct Chainlink Data Streams WebSocket is the preferred live-reference path" in audit_text
    assert "Polymarket RTDS `crypto_prices_chainlink` BTC/USD is the public proxy fallback layer" in audit_text
    assert "`auto` no longer falls back to `public_page`" in audit_text
    assert "fails clearly instead of silently degrading to onchain/page" in audit_text
    assert "ENABLE_CHAINLINK_LIVE" in audit_text
    assert "source=chainlink_public_delayed" in audit_text
    assert "public_stream_page" in audit_text
    assert "compare-chainlink-rtds" in audit_text
    assert "file_state=active" in audit_text
    assert "tail_status=tail_may_be_partial" in audit_text
    assert "source_tier" in audit_text
    assert "quality_class" in audit_text
    assert "recv_minus_chainlink_ms" in audit_text
    assert "backlog_burst_detected" in audit_text
    assert "l2Book" in audit_text
    assert "sequence_gap_detected" in audit_text
    assert "channel_resubscribe_triggered" in audit_text
    assert "order_book_status" in audit_text
    assert "continuity_state" in audit_text
    assert "rest_health_state" in audit_text
    assert "recv_minus_source" in audit_text
    assert "strictly causal on `recv_ts_ns`" in audit_text
    assert "label_safe" in audit_text
    assert "Coinbase `l2_data` is now normalized correctly" in audit_text
    assert "token_modeling_safe=false" in audit_text
    assert "repair-manifests" in audit_text
