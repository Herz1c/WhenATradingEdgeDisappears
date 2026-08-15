from __future__ import annotations

import math

from chainlink_recorder.synthetic_chainlink import (
    DelayedChainlinkBiasCalibrator,
    build_synthetic_chainlink,
    compute_candidate_snapshot,
)


def test_build_synthetic_chainlink_uses_spot_only_premium_builder() -> None:
    recv_ts_ns = 1_776_636_100_000_000_000
    sources = {
        "binance_spot": {
            "mid": 74_200.0,
            "raw_mid": 74_200.0,
            "weight": 2.0,
            "last_recv_ts_ns": recv_ts_ns - 500_000_000,
            "data_present": True,
        },
        "binance_usdm": {
            "mid": 73_500.0,
            "raw_mid": 73_500.0,
            "weight": 3.0,
            "last_recv_ts_ns": recv_ts_ns - 400_000_000,
            "data_present": True,
        },
        "hyperliquid": {
            "mid": 75_000.0,
            "raw_mid": 75_000.0,
            "weight": 1.5,
            "last_recv_ts_ns": recv_ts_ns - 300_000_000,
            "data_present": True,
        },
    }
    expected = 74_200.0 * 1.00029
    assert build_synthetic_chainlink(sources, recv_ts_ns) == expected


def test_build_synthetic_chainlink_applies_delayed_bias_metadata() -> None:
    recv_ts_ns = 1_776_636_100_000_000_000
    sources = {
        "binance_spot": {
            "mid": 74_200.0,
            "raw_mid": 74_200.0,
            "weight": 2.0,
            "last_recv_ts_ns": recv_ts_ns - 500_000_000,
            "data_present": True,
        },
        "binance_usdm": {
            "mid": 74_198.0,
            "raw_mid": 74_198.0,
            "weight": 3.0,
            "last_recv_ts_ns": recv_ts_ns - 400_000_000,
            "data_present": True,
        },
        "hyperliquid": {
            "mid": 74_201.0,
            "raw_mid": 74_201.0,
            "weight": 1.5,
            "last_recv_ts_ns": recv_ts_ns - 300_000_000,
            "data_present": True,
        },
        "_delayed_chainlink_calibration": {
            "bias_usd": 2.75,
        },
    }
    base = 74_200.0 * 1.00029
    assert build_synthetic_chainlink(sources, recv_ts_ns) == base + 2.75


def test_build_synthetic_chainlink_treats_stale_sources_as_missing() -> None:
    recv_ts_ns = 1_776_636_100_000_000_000
    sources = {
        "binance_spot": {
            "mid": 74_200.0,
            "last_recv_ts_ns": recv_ts_ns - 31_000_000_000,
            "data_present": False,
        },
        "binance_usdm": {
            "mid": 74_198.0,
            "last_recv_ts_ns": recv_ts_ns - 500_000_000,
            "data_present": True,
        },
        "hyperliquid": {
            "mid": 74_201.0,
            "last_recv_ts_ns": recv_ts_ns - 500_000_000,
            "data_present": True,
        },
    }
    value = build_synthetic_chainlink(sources, recv_ts_ns)
    assert math.isnan(value)


def test_build_synthetic_chainlink_returns_spot_only_value_with_one_fresh_source() -> None:
    recv_ts_ns = 1_776_636_100_000_000_000
    value = build_synthetic_chainlink(
        {
            "binance_spot": {
                "mid": 74_200.0,
                "last_recv_ts_ns": recv_ts_ns - 100_000_000,
                "data_present": True,
            },
            "binance_usdm": {
                "mid": 74_198.0,
                "last_recv_ts_ns": recv_ts_ns - 45_000_000_000,
                "data_present": False,
            },
            "hyperliquid": {
                "mid": None,
                "last_recv_ts_ns": None,
                "data_present": False,
            },
        },
        recv_ts_ns,
    )
    assert value == 74_200.0 * 1.00029


def test_compute_candidate_snapshot_reports_spread_and_outlier() -> None:
    recv_ts_ns = 1_776_636_100_000_000_000
    snapshot = compute_candidate_snapshot(
        {
            "binance_spot": {
                "mid": 74_200.0,
                "weight": 2.0,
                "last_recv_ts_ns": recv_ts_ns - 100_000_000,
                "data_present": True,
            },
            "binance_usdm": {
                "mid": 74_199.0,
                "weight": 2.0,
                "last_recv_ts_ns": recv_ts_ns - 100_000_000,
                "data_present": True,
            },
            "hyperliquid": {
                "mid": 74_210.0,
                "weight": 2.0,
                "last_recv_ts_ns": recv_ts_ns - 100_000_000,
                "data_present": True,
            },
        },
        recv_ts_ns=recv_ts_ns,
    )
    assert snapshot.source_count == 3
    assert snapshot.max_spread_usd == 11.0
    assert snapshot.outlier_source == "hyperliquid"
    premium = snapshot.candidate_values["spot_only_premium_v1"]
    assert premium is not None
    assert premium > snapshot.candidate_values["binance_spot_raw"]


def test_delayed_chainlink_bias_calibrator_updates_with_clipped_ewma() -> None:
    calibrator = DelayedChainlinkBiasCalibrator(ewma_alpha=0.2, residual_clip_usd=10.0)
    assert calibrator.observe(base_synthetic_price=100.0, observed_chainlink_price=112.0) == 10.0
    updated = calibrator.observe(base_synthetic_price=100.0, observed_chainlink_price=95.0)
    assert math.isclose(updated, 7.0)
    assert calibrator.sample_count == 2
