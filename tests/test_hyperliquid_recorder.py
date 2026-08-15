import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from binance_recorder.compression import iter_zstd_jsonl
from binance_recorder.compression import ZstdJsonlAppender
from hyperliquid_recorder.config import HyperliquidRecorderConfig
from hyperliquid_recorder.health import HyperliquidRestCircuitBreaker
from hyperliquid_recorder.recorder import HyperliquidRecorderService
from hyperliquid_recorder.reader import HyperliquidRawReader
from market_recorders.archive import HourlyArchivePathBuilder
from market_recorders.cli import _build_supervisor
from market_recorders.qc import build_daily_qc_report


def test_hyperliquid_config_defaults(tmp_path) -> None:
    config = HyperliquidRecorderConfig.from_env(out=tmp_path)
    assert config.websocket_url == "wss://api.hyperliquid.xyz/ws"
    assert config.rest_base_url == "https://api.hyperliquid.xyz"
    assert config.symbol == "BTC"


def test_hyperliquid_path_builder(tmp_path) -> None:
    config = HyperliquidRecorderConfig.from_env(out=tmp_path)
    builder = HourlyArchivePathBuilder(root=tmp_path, namespace=config.namespace)
    ts = datetime(2026, 4, 16, 17, 5, tzinfo=UTC)
    assert builder.raw_file_path(ts, "ws") == (
        tmp_path / "raw" / "hyperliquid" / "linear" / "BTC" / "2026-04-16" / "17.ws.jsonl.zst"
    )


def test_hyperliquid_reader_filters_endpoint(tmp_path) -> None:
    path = tmp_path / "raw" / "hyperliquid" / "linear" / "BTC" / "2026-04-16" / "17.rest.jsonl.zst"
    appender = ZstdJsonlAppender(path)
    appender.write_json({"record_type": "rest_observation", "endpoint": "l2Book", "recv_ts_ns": 1})
    appender.write_json({"record_type": "rest_observation", "endpoint": "fundingHistory", "recv_ts_ns": 2})
    appender.close()
    reader = HyperliquidRawReader(root=tmp_path)
    records = list(
        reader.iter_date(
            relative_prefix=reader.relative_prefix,
            target_date="2026-04-16",
            kind="rest",
            endpoint="l2Book",
        )
    )
    assert len(records) == 1
    assert records[0]["endpoint"] == "l2Book"


def test_supervisor_registers_hyperliquid(tmp_path) -> None:
    supervisor = _build_supervisor(
        out=tmp_path,
        log_level="INFO",
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=True,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )
    assert [service.name for service in supervisor._services] == ["hyperliquid"]


@pytest.mark.asyncio
async def test_hyperliquid_rest_loop_recovers_after_error(tmp_path, monkeypatch) -> None:
    config = HyperliquidRecorderConfig.from_env(out=tmp_path)
    service = HyperliquidRecorderService(config)
    calls = {"meta": 0, "l2book": 0, "sleep": 0}

    async def fake_post_info(_client, *, endpoint, body, extracted=None):
        del body, extracted
        if endpoint == "metaAndAssetCtxs":
            calls["meta"] += 1
            if calls["meta"] == 1:
                raise httpx.ReadTimeout("temporary")
        if endpoint == "l2Book":
            calls["l2book"] += 1

    async def fake_sleep(_seconds: float) -> None:
        calls["sleep"] += 1
        if calls["sleep"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(service, "_post_info", fake_post_info)
    monkeypatch.setattr("hyperliquid_recorder.recorder.asyncio.sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await service._rest_loop(client=None)  # type: ignore[arg-type]

    service._ws_writer.close()
    service._rest_writer.close()
    service._lifecycle_writer.close()

    lifecycle_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    events = [record["event"] for record in iter_zstd_jsonl(lifecycle_path)]
    assert "rest_loop_started" in events
    assert "rest_error" in events
    assert calls["meta"] >= 2
    assert calls["l2book"] >= 1


@pytest.mark.asyncio
async def test_hyperliquid_rest_cycle_uses_documented_l2book_request(tmp_path, monkeypatch) -> None:
    config = HyperliquidRecorderConfig.from_env(out=tmp_path)
    service = HyperliquidRecorderService(config)
    calls: list[tuple[str, dict]] = []

    async def fake_post_info(_client, *, endpoint, body, extracted=None):
        del extracted
        calls.append((endpoint, body))

    monkeypatch.setattr(service, "_post_info", fake_post_info)

    last_metrics_ns = await service._rest_cycle(client=None, last_metrics_ns=0)  # type: ignore[arg-type]

    assert last_metrics_ns > 0
    assert ("metaAndAssetCtxs", {"type": "metaAndAssetCtxs"}) in calls
    assert ("l2Book", {"type": "l2Book", "coin": "BTC"}) in calls


def test_hyperliquid_rest_circuit_breaker_transitions_through_degraded_and_suspended() -> None:
    breaker = HyperliquidRestCircuitBreaker(
        degraded_failure_threshold=2,
        suspended_failure_threshold=3,
        degraded_backoff_seconds=5.0,
        suspended_backoff_seconds=30.0,
    )
    first = breaker.observe_failure(now_ns=1, status_code=500, reason="HTTPStatusError")
    second = breaker.observe_failure(now_ns=2, status_code=500, reason="HTTPStatusError")
    third = breaker.observe_failure(now_ns=3, status_code=429, reason="HTTPStatusError")
    gate = breaker.before_cycle(now_ns=10)
    recovery = breaker.observe_success()

    assert first.current_state == "healthy"
    assert second.current_state == "degraded"
    assert second.lifecycle_events[0]["new_state"] == "degraded"
    assert third.current_state == "suspended"
    assert third.lifecycle_events[0]["new_state"] == "suspended"
    assert gate.skip_cycle is True
    assert recovery.current_state == "healthy"
    assert [event["new_state"] for event in recovery.lifecycle_events] == ["recovered", "healthy"]


def test_hyperliquid_qc_splits_rest_transport_by_endpoint(tmp_path) -> None:
    service = HyperliquidRecorderService(HyperliquidRecorderConfig.from_env(out=tmp_path))
    writer = service._rest_writer
    writer.write(
        {
            "record_type": "rest_observation",
            "source": "hyperliquid",
            "market_type": "linear",
            "symbol": "BTC",
            "endpoint": "l2Book",
            "timestamp_semantics": "source_event",
            "recv_ts_ns": 2_000_000_000,
            "source_timestamps": {"time_ms": 1_500},
        }
    )
    writer.write(
        {
            "record_type": "rest_observation",
            "source": "hyperliquid",
            "market_type": "linear",
            "symbol": "BTC",
            "endpoint": "fundingHistory",
            "timestamp_semantics": "historical_series",
            "recv_ts_ns": 3_000_000_000,
            "source_timestamps": {"time_ms": 1_000},
        }
    )
    writer.close()
    service._ws_writer.close()
    service._lifecycle_writer.close()

    report = build_daily_qc_report(tmp_path, datetime(1970, 1, 1, tzinfo=UTC).date())
    source_bucket = report["sources"]["hyperliquid/linear/BTC"]
    assert source_bucket["transport_qc"]["lag_sample_count"] == 1
    assert source_bucket["transport_qc_by_endpoint"]["l2Book"]["lag_sample_count"] == 1
    assert source_bucket["transport_qc_by_endpoint"]["fundingHistory"]["lag_sample_count"] == 0
    assert source_bucket["timestamp_semantics_counts"]["historical_series"] == 1


def test_hyperliquid_service_heartbeat_creates_lifecycle_files_with_runtime_metadata(tmp_path) -> None:
    service = HyperliquidRecorderService(HyperliquidRecorderConfig.from_env(out=tmp_path))
    service._advance_writers(now_ns=1_000_000_000)
    service._advance_writers(now_ns=61_000_000_000)
    service._ws_writer.close()
    service._rest_writer.close()
    service._lifecycle_writer.close()

    lifecycle_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    events = [record["event"] for record in iter_zstd_jsonl(lifecycle_path)]
    assert events == ["service_heartbeat", "service_heartbeat"]

    manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recorder_service"] == "hyperliquid"
    assert manifest["runtime_session_id"]
    assert manifest["config_snapshot"]["symbol"] == "BTC"
