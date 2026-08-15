from __future__ import annotations

import hashlib
import hmac

from eth_abi import encode as abi_encode

from binance_recorder.compression import ZstdJsonlAppender
from binance_recorder.utils import ns_to_iso
from chainlink_recorder.config import ChainlinkRecorderConfig
from chainlink_recorder.datastreams_auth import build_auth_headers, datastreams_websocket_url
from chainlink_recorder.datastreams_decode import decode_datastreams_report
from chainlink_recorder.live_reference import ChainlinkLiveReferenceReader


def _encode_full_report_v3(*, feed_id: str, observations_timestamp: int, price: int, bid: int, ask: int) -> str:
    report_blob = abi_encode(
        ["bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "int192", "int192"],
        [
            bytes.fromhex(feed_id[2:]),
            observations_timestamp - 1,
            observations_timestamp,
            1,
            2,
            observations_timestamp + 30,
            price,
            bid,
            ask,
        ],
    )
    full_report = abi_encode(
        ["bytes32[3]", "bytes", "bytes32[]", "bytes32[]", "bytes32"],
        [
            [b"\x00" * 32, b"\x01" * 32, b"\x02" * 32],
            report_blob,
            [b"\x03" * 32],
            [b"\x04" * 32],
            b"\x05" * 32,
        ],
    )
    return "0x" + full_report.hex()


def test_decode_datastreams_full_report_v3_extracts_live_price_and_freshness() -> None:
    observations_timestamp = 1_776_358_506
    full_report = _encode_full_report_v3(
        feed_id="0x000359843a543ee2fe414dc14c7e7920ef10f4372990b79d6361cdc0dd1ba782",
        observations_timestamp=observations_timestamp,
        price=75_000 * 10**18,
        bid=74_990 * 10**18,
        ask=75_010 * 10**18,
    )
    extracted = decode_datastreams_report(
        {
            "report": {
                "feedID": "0x000359843a543ee2fe414dc14c7e7920ef10f4372990b79d6361cdc0dd1ba782",
                "fullReport": full_report,
            }
        },
        recv_ts_ns=(observations_timestamp * 1_000_000_000) + 500_000_000,
    )
    assert extracted["schema_version"] == 3
    assert extracted["live_price"] == "75000"
    assert extracted["bid"] == "74990"
    assert extracted["ask"] == "75010"
    assert extracted["reference_ts_ns"] == observations_timestamp * 1_000_000_000
    assert extracted["source_age_ms"] == 500.0
    assert extracted["full_report_decode_status"] == "decoded"
    assert extracted["primary_price_source"] == "decoded_v3_benchmark_price"


def test_chainlink_config_auto_prefers_direct_datastreams_when_credentials_exist(tmp_path) -> None:
    config = ChainlinkRecorderConfig(
        out=tmp_path,
        backend="auto",
        datastreams_api_key="key",
        datastreams_api_secret="secret",
        datastreams_feed_id_btcusd="0xfeed",
    )
    assert config.selected_backend == "datastreams_ws"
    assert config.namespace == ("chainlink", "datastreams_ws", "BTCUSD")


def test_datastreams_websocket_auth_uses_full_path_and_empty_body_hash() -> None:
    ws_url, full_path = datastreams_websocket_url(
        "wss://ws.testnet-dataengine.chain.link",
        feed_ids=["0xfeed1", "0xfeed2"],
    )
    headers = build_auth_headers(
        api_key="test-key",
        api_secret="test-secret",
        method="GET",
        full_path=full_path,
        timestamp_ms=1_716_211_845_123,
    )
    expected_signature = hmac.new(
        b"test-secret",
        (
            "GET /api/v1/ws?feedIDs=0xfeed1,0xfeed2 "
            f"{hashlib.sha256(b'').hexdigest()} test-key 1716211845123"
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert ws_url == "wss://ws.testnet-dataengine.chain.link/api/v1/ws?feedIDs=0xfeed1,0xfeed2"
    assert headers["Authorization"] == "test-key"
    assert headers["X-Authorization-Signature-SHA256"] == expected_signature


def test_chainlink_live_reference_reader_prefers_direct_records_over_rtds(tmp_path) -> None:
    direct_path = tmp_path / "raw" / "chainlink" / "datastreams_ws" / "BTCUSD" / "2026-04-16" / "17.live.jsonl.zst"
    direct_appender = ZstdJsonlAppender(direct_path)
    direct_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_600_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_600_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts": 1_776_358_506,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_000_000_000),
            "btc_usd_price": "74123.45",
            "source": "chainlink_datastreams_ws",
        }
    )
    direct_appender.close()

    rtds_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "17.ws.jsonl.zst"
    )
    rtds_appender = ZstdJsonlAppender(rtds_path)
    rtds_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_700_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_700_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_650,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_650_000_000),
            "btc_usd_price": "74124.00",
            "source": "polymarket_rtds_chainlink",
        }
    )
    rtds_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    records = list(reader.iter_best_available(target_date="2026-04-16", finalized_only=False))
    assert len(records) == 1
    assert records[0]["record_type"] == "chainlink_live_reference"
    assert records[0]["source"] == "chainlink_datastreams_ws"
    assert records[0]["btc_usd_price"] == "74123.45"
    assert records[0]["chainlink_ts_ns"] == 1_776_358_506_000_000_000


def test_chainlink_live_reference_latest_asof_uses_rtds_if_direct_record_is_only_later(tmp_path) -> None:
    direct_path = tmp_path / "raw" / "chainlink" / "datastreams_ws" / "BTCUSD" / "2026-04-16" / "17.live.jsonl.zst"
    direct_appender = ZstdJsonlAppender(direct_path)
    direct_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_507_100_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_507_100_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts": 1_776_358_507,
            "chainlink_ts_iso": ns_to_iso(1_776_358_507_000_000_000),
            "btc_usd_price": "74130.00",
            "source": "chainlink_datastreams_ws",
        }
    )
    direct_appender.close()

    rtds_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "17.ws.jsonl.zst"
    )
    rtds_appender = ZstdJsonlAppender(rtds_path)
    rtds_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_900_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_900_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_800,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_800_000_000),
            "btc_usd_price": "74120.00",
            "source": "polymarket_rtds_chainlink",
        }
    )
    rtds_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    best = reader.latest_asof(asof_ts_ns=1_776_358_506_950_000_000, finalized_only=False)
    assert best is not None
    assert best["source"] == "polymarket_rtds_chainlink"
    assert best["btc_usd_price"] == "74120.00"


def test_chainlink_live_reference_reader_prefers_rtds_over_datastreams_api_when_ws_missing(tmp_path) -> None:
    direct_rest_path = (
        tmp_path / "raw" / "chainlink" / "datastreams_api" / "BTCUSD" / "2026-04-16" / "17.live.jsonl.zst"
    )
    direct_rest_appender = ZstdJsonlAppender(direct_rest_path)
    direct_rest_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_650_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_650_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts": 1_776_358_506,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_000_000_000),
            "btc_usd_price": "74122.50",
            "source": "chainlink_datastreams_api",
        }
    )
    direct_rest_appender.close()

    rtds_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "17.ws.jsonl.zst"
    )
    rtds_appender = ZstdJsonlAppender(rtds_path)
    rtds_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_700_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_700_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_650,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_650_000_000),
            "btc_usd_price": "74124.00",
            "source": "polymarket_rtds_chainlink",
        }
    )
    rtds_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    records = list(reader.iter_best_available(target_date="2026-04-16", finalized_only=False))
    assert len(records) == 1
    assert records[0]["source"] == "polymarket_rtds_chainlink"
    assert records[0]["btc_usd_price"] == "74124.00"


def test_chainlink_live_reference_does_not_silently_use_verification_layers(tmp_path) -> None:
    onchain_path = tmp_path / "raw" / "chainlink" / "onchain" / "BTCUSD" / "2026-04-16" / "17.onchain.jsonl.zst"
    onchain_appender = ZstdJsonlAppender(onchain_path)
    onchain_appender.write_json(
        {
            "record_type": "onchain_observation",
            "source": "chainlink",
            "symbol": "BTCUSD",
            "recv_ts_ns": 1_776_358_506_900_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_900_000_000),
            "round_data": {"answer": "74110.00", "updated_at_s": 1_776_358_506},
        }
    )
    onchain_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    assert list(reader.iter_best_available(target_date="2026-04-16", finalized_only=False)) == []
    assert reader.latest_asof(asof_ts_ns=1_776_358_506_950_000_000, finalized_only=False) is None


def test_chainlink_live_reference_latest_asof_never_selects_future_recv_timestamp(tmp_path) -> None:
    rtds_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-16"
        / "17.ws.jsonl.zst"
    )
    rtds_appender = ZstdJsonlAppender(rtds_path)
    rtds_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_507_050_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_507_050_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_900,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_900_000_000),
            "btc_usd_price": "74124.50",
            "source": "polymarket_rtds_chainlink",
        }
    )
    rtds_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_506_990_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_506_990_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts_ms": 1_776_358_506_850,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_850_000_000),
            "btc_usd_price": "74123.90",
            "source": "polymarket_rtds_chainlink",
        }
    )
    rtds_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    best = reader.latest_asof(asof_ts_ns=1_776_358_507_000_000_000, finalized_only=False)
    assert best is not None
    assert best["recv_ts_ns"] == 1_776_358_506_990_000_000
    assert best["btc_usd_price"] == "74123.90"


def test_chainlink_live_reference_latest_asof_uses_same_recv_time_then_latest_chainlink_time(tmp_path) -> None:
    direct_path = tmp_path / "raw" / "chainlink" / "datastreams_ws" / "BTCUSD" / "2026-04-16" / "17.live.jsonl.zst"
    direct_appender = ZstdJsonlAppender(direct_path)
    direct_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_507_000_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_507_000_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts": 1_776_358_506,
            "chainlink_ts_iso": ns_to_iso(1_776_358_506_000_000_000),
            "btc_usd_price": "74122.10",
            "source": "chainlink_datastreams_ws",
        }
    )
    direct_appender.write_json(
        {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": 1_776_358_507_000_000_000,
            "recv_ts_iso": ns_to_iso(1_776_358_507_000_000_000),
            "recv_ts_source": "time.time_ns",
            "chainlink_ts": 1_776_358_507,
            "chainlink_ts_iso": ns_to_iso(1_776_358_507_000_000_000),
            "btc_usd_price": "74122.20",
            "source": "chainlink_datastreams_ws",
        }
    )
    direct_appender.close()

    reader = ChainlinkLiveReferenceReader(root=tmp_path)
    best = reader.latest_asof(asof_ts_ns=1_776_358_507_000_000_000, finalized_only=False)
    assert best is not None
    assert best["recv_ts_ns"] == 1_776_358_507_000_000_000
    assert best["chainlink_ts_ns"] == 1_776_358_507_000_000_000
    assert best["btc_usd_price"] == "74122.20"
