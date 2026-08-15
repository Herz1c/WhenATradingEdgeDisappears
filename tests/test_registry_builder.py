from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_recorders.archive import HourlyRotatingJsonlWriter
from polymarket_recorder.registry_builder import build_registries
from polymarket_recorder.utils import SelectedMarket, session_common_fields


def _market(*, open_dt: datetime, suffix: str) -> SelectedMarket:
    open_epoch = int(open_dt.timestamp())
    return SelectedMarket(
        market_id=f"market-{suffix}",
        condition_id=f"condition-{suffix}",
        market_slug=f"btc-updown-5m-{open_epoch}",
        question=f"Bitcoin Up or Down - {suffix}",
        asset_ids=[f"up-{suffix}", f"down-{suffix}"],
        outcomes=["Up", "Down"],
        series_slug="btc-up-or-down-5m",
        end_time_iso=(open_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        start_time_epoch=open_epoch,
        selection_reason="test_fixture",
        raw_market={
            "events": [
                {
                    "id": f"event-{suffix}",
                    "slug": f"event-slug-{suffix}",
                }
            ]
        },
    )


def _write_strike(root: Path, *, market: SelectedMarket, strike_dt: datetime, price_to_beat: float) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("polymarket", "btc_updown_5m"),
        kind="strike",
    )
    writer.write(
        {
            **session_common_fields(market),
            "record_type": "strike_observation",
            "source": "polymarket",
            "recv_ts_ns": int(strike_dt.timestamp() * 1_000_000_000),
            "recv_ts_iso": strike_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price_to_beat": price_to_beat,
            "label_safe": True,
        }
    )
    writer.close()


def _write_resolution(root: Path, *, market: SelectedMarket, resolved_side: str, recv_dt: datetime) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("polymarket", "resolution", "btc_updown_5m"),
        kind="resolution",
    )
    writer.write(
        {
            "record_type": "market_resolution",
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "market_id": market.market_id,
            "condition_id": market.condition_id,
            "market_slug": market.market_slug,
            "resolved_side": resolved_side,
            "recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
            "resolved_ts_ns": int((market.start_time_epoch + (5 * 60)) * 1_000_000_000),
        }
    )
    writer.close()


def _write_binance_ws(root: Path, *, venue: str, recv_dt: datetime, bid: float, ask: float) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("binance", venue, "BTCUSDT"),
        kind="ws",
    )
    writer.write(
        {
            "record_type": "ws_event",
            "recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
            "normalized": {
                "best_bid_price": f"{bid:.1f}",
                "best_ask_price": f"{ask:.1f}",
            },
        }
    )
    writer.close()


def _write_hyperliquid_ws(root: Path, *, recv_dt: datetime, bid: float, ask: float) -> None:
    writer = HourlyRotatingJsonlWriter(
        root=root,
        namespace=("hyperliquid", "linear", "BTC"),
        kind="ws",
    )
    writer.write(
        {
            "record_type": "ws_event",
            "topic": "l2Book",
            "recv_ts_ns": int(recv_dt.timestamp() * 1_000_000_000),
            "payload": {
                "data": {
                    "levels": [
                        [{"px": f"{bid:.1f}", "sz": "1"}],
                        [{"px": f"{ask:.1f}", "sz": "1"}],
                    ]
                }
            },
        }
    )
    writer.close()


def test_build_registries_writes_phase1_truth_tables(tmp_path) -> None:
    market = _market(open_dt=datetime(2026, 4, 19, 0, 5, tzinfo=UTC), suffix="one")
    _write_strike(
        tmp_path,
        market=market,
        strike_dt=datetime(2026, 4, 19, 0, 4, 30, tzinfo=UTC),
        price_to_beat=103_000.0,
    )
    _write_resolution(
        tmp_path,
        market=market,
        resolved_side="up",
        recv_dt=datetime(2026, 4, 19, 0, 10, 5, tzinfo=UTC),
    )
    _write_binance_ws(
        tmp_path,
        venue="spot",
        recv_dt=datetime(2026, 4, 20, 6, 59, 59, tzinfo=UTC),
        bid=100.0,
        ask=101.0,
    )
    _write_binance_ws(
        tmp_path,
        venue="usdm",
        recv_dt=datetime(2026, 4, 20, 7, 0, 1, tzinfo=UTC),
        bid=100.0,
        ask=101.0,
    )
    _write_hyperliquid_ws(
        tmp_path,
        recv_dt=datetime(2026, 4, 20, 6, 59, 59, tzinfo=UTC),
        bid=100.0,
        ask=101.0,
    )

    report = build_registries(
        root=tmp_path,
        policy_path=Path("config/dataset_policy_v1.yaml"),
        date_from=date(2026, 4, 19),
        date_to=date(2026, 4, 20),
    )

    assert Path(report["market_registry_path"]).exists()
    assert Path(report["token_registry_path"]).exists()
    assert Path(report["label_registry_path"]).exists()
    assert Path(report["quality_registry_path"]).exists()
    assert Path(report["source_registry_path"]).exists()

    market_registry = pd.read_parquet(report["market_registry_path"])
    token_registry = pd.read_parquet(report["token_registry_path"])
    label_registry = pd.read_parquet(report["label_registry_path"])
    quality_registry = pd.read_parquet(report["quality_registry_path"])
    source_registry = pd.read_parquet(report["source_registry_path"])

    assert len(market_registry) == 1
    assert bool(market_registry.iloc[0]["is_trainable"]) is True
    assert bool(market_registry.iloc[0]["has_canonical_resolution"]) is True
    assert market_registry.iloc[0]["resolved_side"] == "Up"

    token_counts = token_registry.groupby("market_slug").size()
    assert token_counts.iloc[0] == 2
    assert set(token_registry["token_side_normalized"].tolist()) == {"UP", "DOWN"}

    assert len(label_registry) == 1
    assert label_registry.iloc[0]["label_join_quality"] == "market_slug_exact"
    assert bool(label_registry.iloc[0]["resolved_side_authoritative"]) is True

    usdm_quarantine = quality_registry.loc[
        (quality_registry["source_name"] == "binance_usdm")
        & (quality_registry["quarantine_reason"] == "source_specific_integrity_check")
    ]
    assert len(usdm_quarantine) == 1

    source_lookup = source_registry.set_index("source_name")
    assert bool(source_lookup.loc["binance_spot", "default_include"]) is True
    assert bool(source_lookup.loc["hyperliquid", "default_include"]) is True
    assert bool(source_lookup.loc["bybit", "default_include"]) is False
    assert bool(source_lookup.loc["fng", "default_include"]) is False
