from __future__ import annotations

from datetime import UTC, datetime

import orjson

from binance_recorder.compression import ZstdJsonlAppender
from binance_recorder.utils import ns_to_iso
from market_recorders.archive import HourlyArchivePathBuilder
from market_recorders.cli import _build_supervisor
from market_recorders.time_utils import recv_timestamp_source
from polymarket_recorder.rtds_config import PolymarketRtdsConfig
from polymarket_recorder.rtds_qc import summarize_rtds_stream
from polymarket_recorder.rtds_reader import PolymarketRtdsRawReader
from polymarket_recorder.rtds_recorder import (
    PolymarketRtdsRecorderService,
    build_subscription_payload,
)


SAMPLE_RTDS_UPDATE = {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": 1_776_358_506_123,
    "connection_id": "vendor-conn-1",
    "payload": {
        "symbol": "btc/usd",
        "timestamp": 1_776_358_506_100,
        "value": 74_123.45,
        "full_accuracy_value": "74123450000000000000000",
    },
}


def test_rtds_config_defaults(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    assert config.websocket_url == "wss://ws-live-data.polymarket.com"
    assert config.topic == "crypto_prices_chainlink"
    assert config.symbol == "btc/usd"
    assert config.subscription_type == "*"


def test_rtds_path_builder(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    builder = HourlyArchivePathBuilder(root=tmp_path, namespace=config.namespace)
    path = builder.raw_file_path(datetime(2026, 4, 16, 17, 5, tzinfo=UTC), "ws")
    assert path == (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "17.ws.jsonl.zst"
    )


def test_rtds_subscription_payload_uses_json_filter_string(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    payload = build_subscription_payload(config)
    assert payload == {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": '{"symbol":"btc/usd"}',
            }
        ],
    }


def test_rtds_build_record_extracts_timestamps_and_value(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    service = PolymarketRtdsRecorderService(config)
    record = service.build_record(
        SAMPLE_RTDS_UPDATE,
        connection_id="conn-1",
        local_msg_seq=7,
        recv_ts_ns=1_776_358_506_500_000_000,
    )
    assert record == {
        "record_type": "chainlink_live_reference",
        "recv_ts_ns": 1_776_358_506_500_000_000,
        "recv_ts_iso": ns_to_iso(1_776_358_506_500_000_000),
        "recv_ts_source": recv_timestamp_source(),
        "chainlink_ts_ms": 1_776_358_506_100,
        "chainlink_ts_iso": ns_to_iso(1_776_358_506_100_000_000),
        "btc_usd_price": "74123.45",
        "source": "polymarket_rtds_chainlink",
        "recv_minus_chainlink_ms": 400.0,
    }
    service._ws_writer.close()
    service._lifecycle_writer.close()


def test_rtds_writer_manifest_counts_topic_message_type_and_symbol(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    service = PolymarketRtdsRecorderService(config)
    service._ws_writer.write(
        service.build_record(
            SAMPLE_RTDS_UPDATE,
            connection_id="conn-1",
            local_msg_seq=1,
            recv_ts_ns=1_776_358_506_500_000_000,
        )
    )
    service._ws_writer.close()
    manifest_path = (
        tmp_path
        / "manifests"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "16.ws.manifest.json"
    )
    manifest = orjson.loads(manifest_path.read_bytes())
    assert manifest["record_count"] == 1
    assert manifest["field_counts"]["source"]["polymarket_rtds_chainlink"] == 1
    service._lifecycle_writer.close()


def test_rtds_reader_filters_and_normalizes(tmp_path) -> None:
    path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-15"
        / "17.ws.jsonl.zst"
    )
    appender = ZstdJsonlAppender(path)
    appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1,
            "recv_ts_iso": "1970-01-01T00:00:00.000000001Z",
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_100,
            "chainlink_ts_iso": "2026-04-16T00:15:06.100000000Z",
            "btc_usd_price": "74123.45",
            "source": "polymarket_rtds_chainlink",
        }
    )
    appender.close()
    reader = PolymarketRtdsRawReader(root=tmp_path)
    filtered = list(
        reader.iter_kind(
            target_date="2026-04-15",
            kind="ws",
            topic="crypto_prices_chainlink",
            message_type="update",
            symbol="btc/usd",
            finalized_only=False,
        )
    )
    normalized = list(
        reader.iter_normalized(
            target_date="2026-04-15",
            kind="ws",
            message_type="update",
            finalized_only=False,
        )
    )
    assert len(filtered) == 1
    assert normalized[0]["record_type"] == "chainlink_live_reference"
    assert normalized[0]["btc_usd_price"] == "74123.45"
    assert normalized[0]["chainlink_ts_ms"] == 1_776_358_506_100


def test_supervisor_registers_polymarket_rtds(tmp_path) -> None:
    supervisor = _build_supervisor(
        out=tmp_path,
        log_level="INFO",
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )
    assert [service.name for service in supervisor._services] == ["polymarket_rtds"]


def test_rtds_freshness_qc_detects_backlog_burst_pattern() -> None:
    records = [
        {
            "recv_ts_ns": 20_000_000_000,
            "chainlink_ts_ms": 1_000,
            "source": "polymarket_rtds_chainlink",
        },
        {
            "recv_ts_ns": 39_000_000_000,
            "chainlink_ts_ms": 20_000,
            "source": "polymarket_rtds_chainlink",
        },
        {
            "recv_ts_ns": 39_500_000_000,
            "chainlink_ts_ms": 30_000,
            "source": "polymarket_rtds_chainlink",
        },
        {
            "recv_ts_ns": 40_000_000_000,
            "chainlink_ts_ms": 35_000,
            "source": "polymarket_rtds_chainlink",
        },
    ]
    summary = summarize_rtds_stream(records)
    assert summary["status"] == "warn"
    assert summary["warning_event_counts"]["silent_gap_detected"] == 1
    assert summary["warning_event_counts"]["backlog_burst_detected"] == 1


def test_rtds_service_builds_fresh_qc_analyzer_per_connection(tmp_path) -> None:
    config = PolymarketRtdsConfig.from_env(out=tmp_path)
    service = PolymarketRtdsRecorderService(config)

    first_connection_analyzer = service._build_qc_analyzer()
    _, first_events = first_connection_analyzer.observe(
        {
            "recv_ts_ns": 1_500_000_000,
            "chainlink_ts_ms": 1_000,
        }
    )
    second_connection_analyzer = service._build_qc_analyzer()
    _, second_events = second_connection_analyzer.observe(
        {
            "recv_ts_ns": 20_500_000_000,
            "chainlink_ts_ms": 20_000,
        }
    )

    assert first_events == []
    assert second_events == []
