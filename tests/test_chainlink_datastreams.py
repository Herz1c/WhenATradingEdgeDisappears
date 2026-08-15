import asyncio

import pytest

from binance_recorder.compression import iter_zstd_jsonl
from binance_recorder.utils import ns_to_iso
from chainlink_recorder.config import ChainlinkRecorderConfig
from chainlink_recorder.datastreams_decode import decode_datastreams_report
from chainlink_recorder.datastreams_ws_client import DatastreamsWsClient
from chainlink_recorder.recorder import ChainlinkRecorderService
from market_recorders.time_utils import recv_timestamp_source


def test_decode_datastreams_report_prefers_benchmark_price() -> None:
    payload = {
        "report": {
            "feedID": "0xfeed",
            "schemaVersion": 3,
            "benchmarkPrice": "74238965204948560000000",
            "bid": "74233631042778840000000",
            "ask": "74247916971490340000000",
            "observationsTimestamp": 1776276348,
            "baseAsset": "BTC",
            "quoteAsset": "USD",
            "reportContext": ["0x00", "0x11"],
        }
    }
    result = decode_datastreams_report(payload)
    assert result["feed_id"] == "0xfeed"
    assert result["schema_version"] == 3
    assert result["mid_price_raw"] == "74238965204948560000000"
    assert result["bid_raw"] == "74233631042778840000000"
    assert result["ask_raw"] == "74247916971490340000000"
    assert result["primary_price_source"] == "reported_primary_price"


def test_chainlink_recorder_selects_datastreams_ws_backend(tmp_path) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="datastreams_ws",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
        datastreams_feed_id_btcusd="0xfeed",
    )
    service = ChainlinkRecorderService(config)
    backend_client = service.datastreams_backend_client()
    assert isinstance(backend_client, DatastreamsWsClient)


def test_datastreams_ws_build_record_decodes_mid_price(tmp_path) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="datastreams_ws",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
        datastreams_feed_id_btcusd="0xfeed",
    )
    service = ChainlinkRecorderService(config)
    client = service.datastreams_backend_client()
    payload = {
        "report": {
            "feedID": "0xfeed",
            "validFromTimestamp": 1_776_276_340,
            "benchmarkPrice": "74238965204948560000000",
            "bid": "74233631042778840000000",
            "ask": "74247916971490340000000",
            "observationsTimestamp": 1776276348,
            "expiresAt": 1_776_276_378,
        }
    }
    record = client.build_record(payload, recv_ts_ns=1_776_276_348_000_000_000)
    assert record == {
        "record_type": "chainlink_live_reference",
        "recv_ts_ns": 1_776_276_348_000_000_000,
        "recv_ts_iso": ns_to_iso(1_776_276_348_000_000_000),
        "recv_ts_source": recv_timestamp_source(),
        "chainlink_ts": 1_776_276_348,
        "chainlink_ts_iso": ns_to_iso(1_776_276_348_000_000_000),
        "btc_usd_price": "74238.96520494856",
        "source": "chainlink_datastreams_ws",
    }


def test_chainlink_config_auto_does_not_silently_fall_back_to_public_page(tmp_path) -> None:
    config = ChainlinkRecorderConfig(out=tmp_path, backend="auto")
    assert config.selected_backend is None
    assert config.availability_issue() == "direct_datastreams_not_configured"
    assert "Polymarket RTDS Chainlink BTC/USD stream as the public fallback" in (
        config.availability_message() or ""
    )


def test_chainlink_config_defaults_to_mainnet_datastreams_hosts(tmp_path) -> None:
    config = ChainlinkRecorderConfig(out=tmp_path, backend="auto")
    assert config.datastreams_ws_url == "wss://ws.dataengine.chain.link"
    assert config.datastreams_rest_url == "https://api.dataengine.chain.link"


def test_chainlink_config_reports_missing_feed_id_clearly(tmp_path) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="datastreams_ws",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
    )
    assert config.selected_backend == "datastreams_ws"
    assert config.availability_issue() == "missing_datastreams_feed_id_btcusd"
    assert "datastreams_feed_id_btcusd" in (config.availability_message() or "")


def test_chainlink_recorder_service_raises_clear_error_when_auto_direct_not_configured(tmp_path) -> None:
    config = ChainlinkRecorderConfig(out=tmp_path, backend="auto")
    with pytest.raises(RuntimeError, match="Direct Chainlink Data Streams WebSocket is not configured"):
        ChainlinkRecorderService(config)


def test_chainlink_public_page_backend_is_rejected_in_favor_of_public_delayed_service(tmp_path) -> None:
    config = ChainlinkRecorderConfig(out=tmp_path, backend="public_page")
    assert config.selected_backend is None
    assert config.availability_issue() == "public_delayed_moved_to_separate_service"
    with pytest.raises(RuntimeError, match="separate chainlink_public_delayed recorder"):
        ChainlinkRecorderService(config)


def test_datastreams_ws_client_classifies_auth_error_from_status_code(tmp_path) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="datastreams_ws",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
        datastreams_feed_id_btcusd="0xfeed",
    )
    service = ChainlinkRecorderService(config)
    client = service.datastreams_backend_client()

    class AuthFailure(Exception):
        status_code = 401

    assert client.classify_exception_event(AuthFailure("bad auth")) == "auth_error"
    service._data_writer.close()
    service._lifecycle_writer.close()


@pytest.mark.asyncio
async def test_datastreams_ws_run_writes_auth_error_and_reconnect_events(tmp_path, monkeypatch) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="datastreams_ws",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
        datastreams_feed_id_btcusd="0xfeed",
    )
    service = ChainlinkRecorderService(config)
    client: DatastreamsWsClient = service.datastreams_backend_client()

    class AuthFailure(Exception):
        status_code = 401

    async def fake_consume(_connection_id: str) -> None:
        raise AuthFailure("unauthorized")

    async def fake_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(client, "_consume", fake_consume)
    monkeypatch.setattr("chainlink_recorder.datastreams_ws_client.asyncio.sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await client.run()

    service._data_writer.close()
    service._lifecycle_writer.close()

    lifecycle_paths = sorted((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    assert lifecycle_paths
    events = [record["event"] for record in iter_zstd_jsonl(lifecycle_paths[0])]
    assert events == [
        "service_start",
        "websocket_connecting",
        "auth_error",
        "reconnect",
        "shutdown",
    ]
