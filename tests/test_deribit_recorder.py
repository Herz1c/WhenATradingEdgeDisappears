import json
from datetime import UTC, datetime

from binance_recorder.compression import iter_zstd_jsonl
from deribit_recorder.recorder import _timestamp_semantics
from deribit_recorder.config import DeribitRecorderConfig
from deribit_recorder.recorder import DeribitRecorderService
from market_recorders.archive import HourlyArchivePathBuilder
from market_recorders.cli import _build_supervisor


def test_deribit_config_defaults(tmp_path) -> None:
    config = DeribitRecorderConfig.from_env(out=tmp_path)
    assert config.websocket_url == "wss://www.deribit.com/ws/api/v2"
    assert config.rest_base_url == "https://www.deribit.com"
    assert config.currency == "BTC"


def test_deribit_path_builder(tmp_path) -> None:
    config = DeribitRecorderConfig.from_env(out=tmp_path)
    builder = HourlyArchivePathBuilder(root=tmp_path, namespace=config.namespace)
    ts = datetime(2026, 4, 16, 17, 5, tzinfo=UTC)
    assert builder.raw_file_path(ts, "rest") == (
        tmp_path / "raw" / "deribit" / "options" / "BTC" / "2026-04-16" / "17.rest.jsonl.zst"
    )


def test_supervisor_registers_deribit(tmp_path) -> None:
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
        enable_hyperliquid=False,
        enable_deribit=True,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )
    assert [service.name for service in supervisor._services] == ["deribit"]


def test_deribit_timestamp_semantics_are_metadata_only_sparse() -> None:
    assert _timestamp_semantics() == "metadata_only_sparse"
    assert _timestamp_semantics("index_price") == "metadata_only_sparse"


def test_deribit_service_heartbeat_creates_lifecycle_files_with_runtime_metadata(tmp_path) -> None:
    service = DeribitRecorderService(DeribitRecorderConfig.from_env(out=tmp_path))
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
    assert manifest["recorder_service"] == "deribit"
    assert manifest["runtime_session_id"]
    assert manifest["config_snapshot"]["currency"] == "BTC"
