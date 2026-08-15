from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymarket_recorder.dataset_audit import audit_phase3_dataset, render_phase3_audit_markdown


def _write_audit_fixture(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "snapshot_ts_ns": 1776884640000000000,
                "snapshot_ts_iso": "2026-04-22T19:04:00.000000Z",
                "sample_family": "dense_close",
                "sample_bucket_ms": 250,
                "market_slug": "btc-updown-5m-1776884400",
                "resolved_side": "up",
                "price_to_beat": 90000.0,
                "live_btc_usd": float("nan"),
                "live_price_valid": False,
                "live_price_degraded": False,
                "live_source_count": 0,
                "binance_spot_mid": float("nan"),
                "binance_usdm_mid": float("nan"),
                "hyperliquid_mid": float("nan"),
                "spot_perp_divergence_usd": float("nan"),
                "delta_to_strike": float("nan"),
                "abs_delta_to_strike": float("nan"),
                "btc_realized_vol_1s": float("nan"),
                "up_token_min_order_size": float("nan"),
                "down_token_min_order_size": float("nan"),
                "up_token_best_bid": 0.40,
                "up_token_best_ask": 0.41,
                "up_token_mid": 0.405,
                "down_token_best_bid": 0.59,
                "down_token_best_ask": 0.60,
                "down_token_mid": 0.595,
                "yes_no_coupling_residual": 0.0,
                "cross_source_spread_usd": float("nan"),
                "flip_happened_after_t": True,
                "terminal_delta_to_strike": 10.0,
                "terminal_margin_to_strike": 10.0,
                "hold_to_close_edge_vs_mid": 0.2,
            },
            {
                "snapshot_ts_ns": 1776884640250000000,
                "snapshot_ts_iso": "2026-04-22T19:04:00.250000Z",
                "sample_family": "dense_close",
                "sample_bucket_ms": 250,
                "market_slug": "btc-updown-5m-1776884400",
                "resolved_side": "up",
                "price_to_beat": 90000.0,
                "live_btc_usd": float("nan"),
                "live_price_valid": False,
                "live_price_degraded": False,
                "live_source_count": 0,
                "binance_spot_mid": float("nan"),
                "binance_usdm_mid": float("nan"),
                "hyperliquid_mid": float("nan"),
                "spot_perp_divergence_usd": float("nan"),
                "delta_to_strike": float("nan"),
                "abs_delta_to_strike": float("nan"),
                "btc_realized_vol_1s": float("nan"),
                "up_token_min_order_size": float("nan"),
                "down_token_min_order_size": float("nan"),
                "up_token_best_bid": 0.41,
                "up_token_best_ask": 0.42,
                "up_token_mid": 0.415,
                "down_token_best_bid": 0.58,
                "down_token_best_ask": 0.59,
                "down_token_mid": 0.585,
                "yes_no_coupling_residual": 0.0,
                "cross_source_spread_usd": float("nan"),
                "flip_happened_after_t": False,
                "terminal_delta_to_strike": 9.0,
                "terminal_margin_to_strike": 9.0,
                "hold_to_close_edge_vs_mid": 0.25,
            },
            {
                "snapshot_ts_ns": 1776879881750000000,
                "snapshot_ts_iso": "2026-04-22T17:44:41.750000Z",
                "sample_family": "dense_close",
                "sample_bucket_ms": 250,
                "market_slug": "btc-updown-5m-1776879600",
                "resolved_side": "down",
                "price_to_beat": 90500.0,
                "live_btc_usd": 90450.0,
                "live_price_valid": True,
                "live_price_degraded": False,
                "live_source_count": 3,
                "binance_spot_mid": 90448.0,
                "binance_usdm_mid": 90447.0,
                "hyperliquid_mid": 90446.0,
                "spot_perp_divergence_usd": 1.0,
                "delta_to_strike": -50.0,
                "abs_delta_to_strike": 50.0,
                "btc_realized_vol_1s": float("nan"),
                "up_token_min_order_size": float("nan"),
                "down_token_min_order_size": float("nan"),
                "up_token_best_bid": 0.85,
                "up_token_best_ask": 0.84,
                "up_token_mid": 0.845,
                "down_token_best_bid": 0.50,
                "down_token_best_ask": 0.49,
                "down_token_mid": 0.495,
                "yes_no_coupling_residual": 0.34,
                "cross_source_spread_usd": 140.0,
                "flip_happened_after_t": True,
                "terminal_delta_to_strike": -12.0,
                "terminal_margin_to_strike": 12.0,
                "hold_to_close_edge_vs_mid": 0.5,
            },
            {
                "snapshot_ts_ns": 1776879600000000000,
                "snapshot_ts_iso": "2026-04-22T17:40:00.000000Z",
                "sample_family": "dense_close",
                "sample_bucket_ms": 100,
                "market_slug": "btc-updown-5m-1776879300",
                "resolved_side": "up",
                "price_to_beat": 90550.0,
                "live_btc_usd": 90560.0,
                "live_price_valid": True,
                "live_price_degraded": False,
                "live_source_count": 2,
                "binance_spot_mid": 90561.0,
                "binance_usdm_mid": 90559.0,
                "hyperliquid_mid": 90558.0,
                "spot_perp_divergence_usd": 2.0,
                "delta_to_strike": 10.0,
                "abs_delta_to_strike": 10.0,
                "btc_realized_vol_1s": float("nan"),
                "up_token_min_order_size": float("nan"),
                "down_token_min_order_size": float("nan"),
                "up_token_best_bid": 0.60,
                "up_token_best_ask": 0.61,
                "up_token_mid": 0.605,
                "down_token_best_bid": 0.39,
                "down_token_best_ask": 0.40,
                "down_token_mid": 0.395,
                "yes_no_coupling_residual": 0.0,
                "cross_source_spread_usd": 8.0,
                "flip_happened_after_t": False,
                "terminal_delta_to_strike": 15.0,
                "terminal_margin_to_strike": 15.0,
                "hold_to_close_edge_vs_mid": 0.1,
            },
        ]
    )
    frame.to_parquet(path, index=False)


def _write_clean_audit_fixture(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "snapshot_ts_ns": 1776879881750000000,
                "snapshot_ts_iso": "2026-04-22T17:44:41.750000Z",
                "sample_family": "dense_close",
                "sample_bucket_ms": 250,
                "market_slug": "btc-updown-5m-1776879600",
                "market_id": "m1",
                "condition_id": "c1",
                "resolved_side": "down",
                "resolved_side_label": 0,
                "price_to_beat": 90500.0,
                "live_btc_usd": 90450.0,
                "live_price_valid": True,
                "live_price_degraded": False,
                "live_source_count": 3,
                "binance_spot_mid": 90448.0,
                "binance_usdm_mid": 90447.0,
                "hyperliquid_mid": 90446.0,
                "spot_perp_divergence_usd": 1.0,
                "delta_to_strike": -50.0,
                "abs_delta_to_strike": 50.0,
                "btc_realized_vol_5s": 0.01,
                "btc_realized_vol_15s": 0.02,
                "up_token_best_bid": 0.85,
                "up_token_best_ask": 0.84,
                "up_token_mid": 0.845,
                "down_token_best_bid": 0.50,
                "down_token_best_ask": 0.49,
                "down_token_mid": 0.495,
                "yes_no_coupling_residual": 0.34,
                "cross_source_spread_usd": 140.0,
                "flip_happened_after_t": True,
                "terminal_delta_to_strike": -12.0,
                "terminal_margin_to_strike": 12.0,
                "hold_to_close_edge_vs_mid": 0.5,
                "coupling_anomaly": True,
                "crossed_book": True,
                "has_canonical_resolution": True,
                "is_trainable": True,
                "training_feature_eligible": False,
                "exclusion_reason": "coupling_anomaly",
            }
        ]
    )
    frame.to_parquet(path, index=False)


def test_audit_phase3_dataset_detects_expected_findings(tmp_path: Path) -> None:
    dataset_path = tmp_path / "audit_fixture.parquet"
    _write_audit_fixture(dataset_path)

    report = audit_phase3_dataset(dataset_path)

    assert report["summary"]["rows"] == 4
    assert report["summary"]["unique_markets"] == 3
    assert report["policy"]["dead_columns_present"] == [
        "btc_realized_vol_1s",
        "up_token_min_order_size",
        "down_token_min_order_size",
    ]
    assert len(report["live_reference"]["dead_hours"]) == 1
    assert report["live_reference"]["dead_hours"][0]["hour_utc"] == "2026-04-22 19:00:00+00:00"
    assert len(report["live_reference"]["all_invalid_markets"]) == 1
    assert report["microstructure"]["coupling_anomaly_count"] == 1
    assert sum(item["count"] for item in report["microstructure"]["crossed_book_rows"]) == 2
    assert report["leakage"]["leakage_columns_present"] == [
        "flip_happened_after_t",
        "terminal_delta_to_strike",
        "terminal_margin_to_strike",
        "hold_to_close_edge_vs_mid",
    ]
    problems = {item["problem"] for item in report["findings"]}
    assert "Dead columns that should have been removed are still present in the parquet" in problems
    assert "Entire live-reference-dead hour(s) reached the modeling dataset" in problems
    assert "Future-derived diagnostic columns are present without an explicit feature-matrix exclusion field" in problems


def test_render_phase3_audit_markdown_includes_manual_checklist(tmp_path: Path) -> None:
    dataset_path = tmp_path / "audit_fixture.parquet"
    _write_audit_fixture(dataset_path)

    markdown = render_phase3_audit_markdown(audit_phase3_dataset(dataset_path))

    assert "# Phase 3 Dataset Audit" in markdown
    assert "## Manual Upstream Checks" in markdown
    assert "### Policy compliance" in markdown
    assert "`btc_realized_vol_1s`" in markdown
    assert "`2026-04-22 19:00:00+00:00`" in markdown


def test_audit_phase3_dataset_accepts_flagged_anomalies_with_schema_contract(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "2026-04-22.parquet"
    _write_clean_audit_fixture(dataset_path)

    schema_dir = tmp_path / "config" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "resolution_snapshot_dataset_v1_dense_close_schema.yaml").write_text(
        """
dataset_name: resolution_snapshot_dataset_v1_dense_close
columns:
  resolved_side:
    leakage_only: true
    training_eligible: false
  flip_happened_after_t:
    leakage_only: true
    training_eligible: false
  terminal_delta_to_strike:
    leakage_only: true
    training_eligible: false
  terminal_margin_to_strike:
    leakage_only: true
    training_eligible: false
  hold_to_close_edge_vs_mid:
    leakage_only: true
    training_eligible: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = audit_phase3_dataset(dataset_path)

    assert report["findings"] == []
    assert "Leakage-sensitive columns remain present only as diagnostic fields outside the training feature contract." in report["passes"]
