from __future__ import annotations

import math

import pandas as pd

from binance_recorder.compression import ZstdJsonlAppender
from binance_recorder.utils import ns_to_iso

from polymarket_recorder.strike_delta import PolymarketStrikeDeltaReader


def _write_live_reference_parquet(tmp_path, frame: pd.DataFrame) -> None:
    target = tmp_path / "canonical" / "live_reference_events_v1"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "2026-04-16.parquet", index=False)


def _write_strike(tmp_path, *, recv_ts_ns: int, price_to_beat: str = "74120") -> None:
    strike_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "btc_updown_5m"
        / "2026-04-16"
        / "17-00-00__sample__abc.strike.jsonl.zst"
    )
    strike_appender = ZstdJsonlAppender(strike_path)
    strike_appender.write_json(
        {
            "record_type": "strike_observation",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "market_family": "btc_updown_5m",
            "selected_market_slug": "btc-updown-5m-1776359700",
            "selected_market_id": "market-1",
            "selected_condition_id": "condition-1",
            "price_to_beat": price_to_beat,
            "parse_status": "success",
            "label_safe": True,
        }
    )
    strike_appender.close()


def test_strike_delta_reader_builds_historical_asof_delta_from_live_reference_events(tmp_path) -> None:
    _write_strike(tmp_path, recv_ts_ns=1_776_358_507_000_000_000)
    _write_live_reference_parquet(
        tmp_path,
        pd.DataFrame(
            {
                "ts_seconds": [1_776_358_506, 1_776_358_507],
                "synthetic_raw": [74_123.40, 74_123.45],
                "synthetic_corrected": [74_123.40, 74_123.45],
                "rolling_bias": [0.0, 0.0],
                "bias_state_age_seconds": [1800.0, 1801.0],
                "bias_bootstrapped": [True, True],
                "source_count": [2, 2],
                "price_valid": [True, True],
                "degraded": [False, False],
                "binance_spot_mid": [74_123.30, 74_123.35],
                "binance_spot_staleness_s": [0.0, 0.0],
                "binance_usdm_mid": [74_123.50, 74_123.55],
                "binance_usdm_staleness_s": [0.0, 0.0],
                "hyperliquid_mid": [74_123.40, 74_123.45],
                "hyperliquid_staleness_s": [0.0, 0.0],
                "max_cross_source_spread": [0.2, 0.2],
            }
        ),
    )

    reader = PolymarketStrikeDeltaReader(root=tmp_path)
    deltas = list(reader.iter_historical_delta(target_date="2026-04-16", finalized_only=False))
    assert len(deltas) == 1
    assert deltas[0]["live_source"] == "synthetic_chainlink_v1"
    assert deltas[0]["price_unavailable"] is False
    assert deltas[0]["delta_to_strike"] == "3.45"
    assert deltas[0]["absolute_delta_to_strike"] == "3.45"
    assert round(deltas[0]["delta_percent"], 6) == round((3.45 / 74120) * 100, 6)
    assert round(deltas[0]["delta_bps"], 3) == 0.465


def test_current_live_delta_reads_latest_causal_live_reference_row(tmp_path) -> None:
    _write_strike(tmp_path, recv_ts_ns=1_776_358_507_000_000_000)
    _write_live_reference_parquet(
        tmp_path,
        pd.DataFrame(
            {
                "ts_seconds": [1_776_358_507, 1_776_358_508],
                "synthetic_raw": [74_118.25, 74_119.00],
                "synthetic_corrected": [74_118.25, 74_119.00],
                "rolling_bias": [0.0, 0.0],
                "bias_state_age_seconds": [1800.0, 1801.0],
                "bias_bootstrapped": [True, True],
                "source_count": [2, 2],
                "price_valid": [True, True],
                "degraded": [False, False],
                "binance_spot_mid": [74_118.25, 74_119.00],
                "binance_spot_staleness_s": [0.0, 0.0],
                "binance_usdm_mid": [74_118.35, 74_119.10],
                "binance_usdm_staleness_s": [0.0, 0.0],
                "hyperliquid_mid": [74_118.20, 74_118.95],
                "hyperliquid_staleness_s": [0.0, 0.0],
                "max_cross_source_spread": [0.15, 0.15],
            }
        ),
    )

    reader = PolymarketStrikeDeltaReader(root=tmp_path)
    delta = reader.current_live_delta(
        target_date="2026-04-16",
        now_ts_ns=1_776_358_507_500_000_000,
        finalized_only=False,
    )
    assert delta is not None
    assert delta["live_source"] == "synthetic_chainlink_v1"
    assert delta["delta_to_strike"] == "-1.75"


def test_historical_delta_uses_latest_non_future_row_and_propagates_unavailable_price(tmp_path) -> None:
    _write_strike(tmp_path, recv_ts_ns=1_776_358_507_000_000_000)
    _write_live_reference_parquet(
        tmp_path,
        pd.DataFrame(
            {
                "ts_seconds": [1_776_358_506, 1_776_358_507, 1_776_358_508],
                "synthetic_raw": [74_123.90, math.nan, 74_124.50],
                "synthetic_corrected": [74_123.90, math.nan, 74_124.50],
                "rolling_bias": [0.0, 0.0, 0.0],
                "bias_state_age_seconds": [1800.0, 1801.0, 1802.0],
                "bias_bootstrapped": [True, True, True],
                "source_count": [2, 0, 2],
                "price_valid": [True, False, True],
                "degraded": [False, False, False],
                "binance_spot_mid": [74_123.90, math.nan, 74_124.50],
                "binance_spot_staleness_s": [0.0, math.nan, 0.0],
                "binance_usdm_mid": [74_124.00, math.nan, 74_124.60],
                "binance_usdm_staleness_s": [0.0, math.nan, 0.0],
                "hyperliquid_mid": [74_123.80, math.nan, 74_124.40],
                "hyperliquid_staleness_s": [0.0, math.nan, 0.0],
                "max_cross_source_spread": [0.2, math.nan, 0.2],
            }
        ),
    )

    reader = PolymarketStrikeDeltaReader(root=tmp_path)
    deltas = list(reader.iter_historical_delta(target_date="2026-04-16", finalized_only=False))
    assert len(deltas) == 1
    assert deltas[0]["live_recv_ts_ns"] == 1_776_358_507_000_000_000
    assert deltas[0]["price_unavailable"] is True
    assert math.isnan(deltas[0]["delta_to_strike"])
    assert math.isnan(deltas[0]["absolute_delta_to_strike"])
