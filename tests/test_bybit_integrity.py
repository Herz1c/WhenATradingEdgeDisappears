from datetime import UTC, datetime

import json

from binance_recorder.compression import iter_zstd_jsonl
from bybit_recorder.config import BybitRecorderConfig
from bybit_recorder.recorder import BybitRecorderService
from bybit_recorder.integrity import BybitContinuityMonitor


def test_bybit_control_rows_are_classified_explicitly() -> None:
    monitor = BybitContinuityMonitor()
    observation = monitor.observe({"op": "subscribe", "success": True})
    assert observation.continuity_state == "control_row"
    assert observation.lifecycle_events[0]["event"] == "bybit_control_row_observed"


def test_bybit_delta_before_snapshot_is_continuity_uncertain() -> None:
    monitor = BybitContinuityMonitor()
    observation = monitor.observe(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "data": {"u": 10, "b": [], "a": []},
            "ts": 1_776_358_506_000,
        }
    )
    assert observation.continuity_state == "continuity_uncertain"
    assert observation.quality_class == "continuity_compromised"
    assert observation.lifecycle_events[0]["event"] == "bybit_sequence_continuity_break"


def test_bybit_sequence_regression_is_reported() -> None:
    monitor = BybitContinuityMonitor()
    first = monitor.observe(
        {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "cs": 100,
            "ts": 1_776_358_506_000,
            "data": {"symbol": "BTCUSDT"},
        }
    )
    second = monitor.observe(
        {
            "topic": "tickers.BTCUSDT",
            "type": "delta",
            "cs": 99,
            "ts": 1_776_358_506_100,
            "data": {"symbol": "BTCUSDT"},
        }
    )
    assert first.continuity_state == "reset_snapshot_transition"
    assert second.continuity_state == "continuity_uncertain"
    assert second.quality_class == "continuity_compromised"
    assert second.lifecycle_events[0]["event"] == "bybit_sequence_continuity_break"


def test_bybit_public_trade_snapshot_is_not_treated_as_reset_noise() -> None:
    monitor = BybitContinuityMonitor()
    observation = monitor.observe(
        {
            "topic": "publicTrade.BTCUSDT",
            "type": "snapshot",
            "ts": 1_776_358_506_000,
            "data": [{"i": "1", "p": "74100"}],
        }
    )
    assert observation.continuity_state == "healthy"
    assert observation.lifecycle_events == []


def test_bybit_service_heartbeat_creates_lifecycle_files_with_runtime_metadata(tmp_path) -> None:
    service = BybitRecorderService(BybitRecorderConfig.from_env(out=tmp_path))
    service._advance_writers(now_ns=1_000_000_000)
    service._advance_writers(now_ns=30_000_000_000)
    service._advance_writers(now_ns=61_000_000_000)
    service._ws_writer.close()
    service._rest_writer.close()
    service._lifecycle_writer.close()

    lifecycle_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    events = [record["event"] for record in iter_zstd_jsonl(lifecycle_path)]
    assert events == ["service_heartbeat", "service_heartbeat"]

    manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recorder_service"] == "bybit"
    assert manifest["runtime_session_id"]
    assert manifest["config_snapshot"]["symbol"] == "BTCUSDT"
