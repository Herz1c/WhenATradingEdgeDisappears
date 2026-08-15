from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from market_recorders.archive import HourlyRotatingJsonlWriter
from chainlink_recorder.live_reference_events import (
    LiveReferenceEventsReader,
    build_live_reference_events,
)


def _update_manifest_quality(manifest_path: Path, *, quality_class: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["quality_class"] = quality_class
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_live_reference_events_respects_policy_quarantine(tmp_path) -> None:
    spot_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("binance", "spot", "BTCUSDT"),
        kind="ws",
    )
    spot_writer.write(
        {
            "record_type": "ws_event",
            "recv_ts_ns": 1_776_636_001_100_000_000,
            "normalized": {
                "best_bid_price": "100.0",
                "best_ask_price": "102.0",
            },
        }
    )
    spot_writer.close()

    usdm_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("binance", "usdm", "BTCUSDT"),
        kind="ws",
    )
    usdm_writer.write(
        {
            "record_type": "ws_event",
            "recv_ts_ns": 1_776_636_001_100_000_000,
            "normalized": {
                "best_bid_price": "99.0",
                "best_ask_price": "101.0",
            },
        }
    )
    usdm_writer.close()

    usdm_manifest = next((tmp_path / "manifests" / "binance" / "usdm" / "BTCUSDT").rglob("*.ws.manifest.json"))
    _update_manifest_quality(usdm_manifest, quality_class="continuity_compromised")

    hyper_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("hyperliquid", "linear", "BTC"),
        kind="ws",
    )
    hyper_writer.write(
        {
            "record_type": "ws_event",
            "topic": "l2Book",
            "recv_ts_ns": 1_776_636_001_200_000_000,
            "payload": {
                "data": {
                    "levels": [
                        [{"px": "200.5", "sz": "1"}],
                        [{"px": "201.5", "sz": "1"}],
                    ]
                }
            },
        }
    )
    hyper_writer.close()

    delayed_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink_public_delayed", "public_stream_page", "BTCUSD"),
        kind="page",
    )
    delayed_writer.write(
        {
            "record_type": "chainlink_public_delayed_observation",
            "recv_ts_ns": 1_776_636_005_000_000_000,
            "chainlink_display_ts": 1_776_636_002,
            "btc_usd_price": "101.25",
        }
    )
    delayed_writer.close()

    report = build_live_reference_events(
        root=tmp_path,
        output_path=Path("canonical/live_reference_events_v1"),
        validation_path=Path("live_reference_events_v1_validation.md"),
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=pd.Timestamp("2026-04-19", tz="UTC").date(),
        date_to=pd.Timestamp("2026-04-19", tz="UTC").date(),
        max_workers=1,
    )

    assert report["row_count"] == 1
    assert report["daily_parquets"] == [str(tmp_path / "canonical" / "live_reference_events_v1" / "2026-04-19.parquet")]
    assert report["schema_path"] == str(tmp_path / "config" / "schemas" / "live_reference_events_v1_schema.yaml")
    frame = pd.read_parquet(tmp_path / "canonical" / "live_reference_events_v1" / "2026-04-19.parquet")
    assert list(frame.columns) == [
        "recv_ts_ns",
        "ts_seconds",
        "synthetic_raw",
        "synthetic_corrected",
        "rolling_bias",
        "active_window_bias",
        "bias_state_age_seconds",
        "applied_bias_age_seconds",
        "last_bias_observation_age_seconds",
        "bias_mode",
        "bias_bootstrapped",
        "bias_active",
        "bias_carried_forward",
        "bias_state_stale",
        "bias_neutralized",
        "bias_observation_count",
        "source_count",
        "price_valid",
        "degraded",
        "binance_spot_mid",
        "binance_spot_staleness_s",
        "binance_usdm_mid",
        "binance_usdm_staleness_s",
        "hyperliquid_mid",
        "hyperliquid_staleness_s",
        "max_cross_source_spread",
    ]
    row = frame.iloc[0]
    assert row["source_count"] == 2
    assert bool(row["price_valid"]) is True
    assert bool(row["degraded"]) is True
    assert int(row["recv_ts_ns"]) == int(row["ts_seconds"]) * 1_000_000_000
    assert math.isnan(float(row["binance_usdm_mid"]))
    expected_base = 101.0 * 1.00029
    assert math.isclose(float(row["synthetic_raw"]), expected_base)
    assert math.isclose(float(row["synthetic_corrected"]), expected_base)
    assert math.isnan(float(row["active_window_bias"]))
    assert math.isnan(float(row["applied_bias_age_seconds"]))
    assert str(row["bias_mode"]) == "unavailable"
    assert bool(row["bias_bootstrapped"]) is False
    assert bool(row["bias_active"]) is False
    assert bool(row["bias_carried_forward"]) is False
    assert bool(row["bias_neutralized"]) is False
    assert int(row["bias_observation_count"]) == 0
    assert report["quarantine_summary"]["binance_usdm"]["manifest_quality_class:continuity_compromised"] == 1


def test_build_live_reference_events_is_spot_only_when_nonspot_sources_missing(tmp_path) -> None:
    spot_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("binance", "spot", "BTCUSDT"),
        kind="ws",
    )
    spot_writer.write(
        {
            "record_type": "ws_event",
            "recv_ts_ns": 1_776_636_001_100_000_000,
            "normalized": {
                "best_bid_price": "100.0",
                "best_ask_price": "102.0",
            },
        }
    )
    spot_writer.close()

    report = build_live_reference_events(
        root=tmp_path,
        output_path=Path("canonical/live_reference_events_v1"),
        validation_path=Path("live_reference_events_v1_validation.md"),
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=pd.Timestamp("2026-04-19", tz="UTC").date(),
        date_to=pd.Timestamp("2026-04-19", tz="UTC").date(),
        max_workers=1,
    )

    frame = pd.read_parquet(tmp_path / "canonical" / "live_reference_events_v1" / "2026-04-19.parquet")
    row = frame.iloc[0]
    assert report["row_count"] == 1
    assert row["source_count"] == 1
    assert bool(row["price_valid"]) is True
    assert bool(row["degraded"]) is True
    assert math.isclose(float(row["synthetic_raw"]), 101.0 * 1.00029)


def test_live_reference_events_reader_returns_causal_latest_row(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "recv_ts_ns": [100_000_000_000, 101_000_000_000, 102_000_000_000],
            "ts_seconds": [100, 101, 102],
            "synthetic_raw": [1.0, 2.0, math.nan],
            "synthetic_corrected": [1.1, 2.1, math.nan],
            "rolling_bias": [0.1, 0.1, 0.1],
            "bias_state_age_seconds": [1800.0, 1801.0, 1802.0],
            "last_bias_observation_age_seconds": [1800.0, 1801.0, 1802.0],
            "bias_bootstrapped": [True, True, True],
            "bias_active": [True, True, True],
            "bias_state_stale": [False, False, False],
            "bias_neutralized": [False, False, False],
            "bias_observation_count": [2, 2, 2],
            "source_count": [2, 1, 0],
            "price_valid": [True, True, False],
            "degraded": [False, True, False],
            "binance_spot_mid": [1.0, 2.0, math.nan],
            "binance_spot_staleness_s": [0.0, 0.0, math.nan],
            "binance_usdm_mid": [1.2, math.nan, math.nan],
            "binance_usdm_staleness_s": [0.0, math.nan, math.nan],
            "hyperliquid_mid": [1.1, math.nan, math.nan],
            "hyperliquid_staleness_s": [0.0, math.nan, math.nan],
            "max_cross_source_spread": [0.2, math.nan, math.nan],
        }
    )
    out = tmp_path / "canonical"
    shard_dir = out / "live_reference_events_v1"
    shard_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard_dir / "2026-04-19.parquet", index=False)

    reader = LiveReferenceEventsReader(root=tmp_path)
    record = reader.latest_asof(asof_ts_ns=101_500_000_000)
    assert record is not None
    assert record["recv_ts_ns"] == 101_000_000_000
    assert record["btc_usd_price"] == 2.1
    assert record["degraded"] is True
    assert record["price_unavailable"] is False

    record = reader.latest_asof(asof_ts_ns=102_500_000_000)
    assert record is not None
    assert record["recv_ts_ns"] == 102_000_000_000
    assert record["btc_usd_price"] is None
    assert record["price_unavailable"] is True


def test_build_live_reference_events_carries_last_valid_delayed_residual_bias(tmp_path) -> None:
    start_ts = 1_776_636_000
    row_count = 3_605
    for namespace, bid_offset, ask_offset in (
        (("binance", "spot", "BTCUSDT"), 0.0, 2.0),
        (("binance", "usdm", "BTCUSDT"), 0.2, 2.2),
    ):
        writer = HourlyRotatingJsonlWriter(root=tmp_path, namespace=namespace, kind="ws")
        for offset in range(row_count):
            writer.write(
                {
                    "record_type": "ws_event",
                    "recv_ts_ns": (start_ts + offset) * 1_000_000_000,
                    "normalized": {
                        "best_bid_price": str(100.0 + bid_offset),
                        "best_ask_price": str(100.0 + ask_offset),
                    },
                }
            )
        writer.close()

    hyper_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("hyperliquid", "linear", "BTC"),
        kind="ws",
    )
    for offset in range(row_count):
        hyper_writer.write(
            {
                "record_type": "ws_event",
                "topic": "l2Book",
                "recv_ts_ns": (start_ts + offset) * 1_000_000_000,
                "payload": {
                    "data": {
                        "levels": [
                            [{"px": "100.4", "sz": "1"}],
                            [{"px": "101.4", "sz": "1"}],
                        ]
                    }
                },
            }
        )
    hyper_writer.close()

    delayed_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink_public_delayed", "public_stream_page", "BTCUSD"),
        kind="page",
    )
    delayed_writer.write(
        {
            "record_type": "chainlink_public_delayed_observation",
            "recv_ts_ns": (start_ts + 1) * 1_000_000_000,
            "chainlink_display_ts": start_ts + 1,
            "btc_usd_price": str((101.0 * 1.00029) + 10.0),
        }
    )
    delayed_writer.close()

    report = build_live_reference_events(
        root=tmp_path,
        output_path=Path("canonical/live_reference_events_v1"),
        validation_path=Path("live_reference_events_v1_validation.md"),
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=pd.Timestamp("2026-04-19", tz="UTC").date(),
        date_to=pd.Timestamp("2026-04-19", tz="UTC").date(),
        max_workers=1,
    )

    frame = pd.read_parquet(tmp_path / "canonical" / "live_reference_events_v1" / "2026-04-19.parquet")
    before = frame.loc[frame["ts_seconds"] < start_ts + 1_801]
    active = frame.loc[(frame["ts_seconds"] >= start_ts + 1_801) & (frame["ts_seconds"] <= start_ts + 3_601)]
    after = frame.loc[frame["ts_seconds"] > start_ts + 3_601]
    assert report["bias_anomaly_events"] == 0
    assert float(before["rolling_bias"].abs().max()) == 0.0
    assert math.isclose(float(active["rolling_bias"].median()), 10.0)
    assert math.isclose(float(active["active_window_bias"].median()), 10.0)
    assert bool(active["bias_active"].all())
    assert not bool(active["bias_carried_forward"].any())
    assert int(active["bias_observation_count"].max()) == 1
    assert math.isclose(float(after["rolling_bias"].median()), 10.0)
    assert bool(after["bias_carried_forward"].all())
    assert not bool(after["bias_active"].any())
    assert set(after["bias_mode"]) == {"carried_forward"}


def test_build_live_reference_events_bootstraps_carried_bias_for_independent_window(tmp_path) -> None:
    prior_ts = int(pd.Timestamp("2026-04-19T21:00:00Z").timestamp())
    requested_ts = int(pd.Timestamp("2026-04-20T00:00:00Z").timestamp())

    for namespace, bid_offset, ask_offset in (
        (("binance", "spot", "BTCUSDT"), 0.0, 2.0),
        (("binance", "usdm", "BTCUSDT"), 0.2, 2.2),
    ):
        writer = HourlyRotatingJsonlWriter(root=tmp_path, namespace=namespace, kind="ws")
        for ts in (prior_ts, requested_ts, requested_ts + 1):
            writer.write(
                {
                    "record_type": "ws_event",
                    "recv_ts_ns": ts * 1_000_000_000,
                    "normalized": {
                        "best_bid_price": str(100.0 + bid_offset),
                        "best_ask_price": str(100.0 + ask_offset),
                    },
                }
            )
        writer.close()

    hyper_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("hyperliquid", "linear", "BTC"),
        kind="ws",
    )
    for ts in (prior_ts, requested_ts, requested_ts + 1):
        hyper_writer.write(
            {
                "record_type": "ws_event",
                "topic": "l2Book",
                "recv_ts_ns": ts * 1_000_000_000,
                "payload": {
                    "data": {
                        "levels": [
                            [{"px": "100.4", "sz": "1"}],
                            [{"px": "101.4", "sz": "1"}],
                        ]
                    }
                },
            }
        )
    hyper_writer.close()

    delayed_writer = HourlyRotatingJsonlWriter(
        root=tmp_path,
        namespace=("chainlink_public_delayed", "public_stream_page", "BTCUSD"),
        kind="page",
    )
    delayed_writer.write(
        {
            "record_type": "chainlink_public_delayed_observation",
            "recv_ts_ns": (prior_ts + 1) * 1_000_000_000,
            "chainlink_display_ts": prior_ts + 1,
            "btc_usd_price": str((101.0 * 1.00029) + 10.0),
        }
    )
    delayed_writer.close()

    build_live_reference_events(
        root=tmp_path,
        output_path=Path("canonical/live_reference_events_v1"),
        validation_path=Path("live_reference_events_v1_validation.md"),
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=pd.Timestamp("2026-04-20", tz="UTC").date(),
        date_to=pd.Timestamp("2026-04-20", tz="UTC").date(),
        max_workers=1,
    )

    frame = pd.read_parquet(tmp_path / "canonical" / "live_reference_events_v1" / "2026-04-20.parquet")
    valid = frame.loc[frame["price_valid"].astype(bool)]
    assert not valid.empty
    assert set(valid["bias_mode"]) == {"carried_forward"}
    assert bool(valid["bias_carried_forward"].all())
    assert math.isclose(float(valid["rolling_bias"].median()), 10.0)
