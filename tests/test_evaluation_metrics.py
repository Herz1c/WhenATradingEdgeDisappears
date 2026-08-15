import numpy as np
import pytest

from model_factory.evaluation.metrics import (
    compute_adverse_flag_precision_recall,
    compute_brier_score,
    compute_calibration_curve,
    compute_ece,
    compute_edge_by_time_to_close,
    compute_fee_adjusted_edge,
    compute_fill_adjusted_edge,
    compute_fill_rate_by_offset,
    compute_flip_precision_recall,
    compute_flip_rate_by_delta_bucket,
    compute_implied_edge,
    compute_inventory_stress,
    compute_log_loss,
    compute_markout_stats,
    compute_pnl_stats,
    compute_stability_by_day,
)


def test_probabilistic_metrics_known_inputs():
    y = np.array([0, 1])
    p = np.array([0.25, 0.75])
    assert compute_log_loss(y, p) == pytest.approx(0.287682, rel=1e-5)
    assert compute_brier_score(y, p) == pytest.approx(0.0625)


def test_ece_perfectly_calibrated_zero():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert compute_ece(y, p, n_bins=2) == 0.0


def test_ece_constant_predictor():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_ece(y, p, n_bins=2) == pytest.approx(0.25)


def test_calibration_curve_bins():
    curve = compute_calibration_curve(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=2)
    assert curve["bin_sizes"] == [1, 1]
    assert curve["fraction_positive"] == [0.0, 1.0]


def test_fee_adjusted_edge_negative_for_zero_edge_predictor_at_max_fee():
    assert compute_fee_adjusted_edge(np.array([0.5]), np.array([0.5]), fee_rate=0.018) < 0


def test_market_and_trading_metrics():
    assert compute_implied_edge(np.array([0.6, 0.4]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert "0_10" in compute_edge_by_time_to_close(np.array([0, 1]), np.array([0.2, 0.8]), np.array([5, 20]))
    assert compute_fill_adjusted_edge(np.array([0.6]), np.array([0.5]), np.array([0.5])) == pytest.approx(0.05)
    markouts = compute_markout_stats(np.array([1, 2]), np.array([2, 3]), np.array([3, 4]), np.array([4, 5]))
    assert markouts["1s"]["mean"] == pytest.approx(1.5)
    pnl = compute_pnl_stats(np.array([1, 2]), np.array([0.5, 1.5]), np.array([True, False]))
    assert pnl["filled_rows"] == 1
    stress = compute_inventory_stress(np.array([10, -20]), np.array([1, -1]), max_position=15)
    assert stress["breach_rate"] == pytest.approx(0.5)


def test_stability_by_day_manual_computation():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 0.5, 0.5])
    dates = np.array(["a", "a", "b", "b"])
    out = compute_stability_by_day(y, p, dates, compute_brier_score)
    assert out["by_day"] == [0.0, 0.25]
    assert out["mean"] == pytest.approx(0.125)


def test_execution_specific_metrics():
    pr = compute_adverse_flag_precision_recall(np.array([True, True, False]), np.array([True, False, True]))
    assert pr["precision"] == pytest.approx(0.5)
    flip = compute_flip_precision_recall(np.array([0.6, 0.2]), np.array([True, True]))
    assert flip["recall"] == pytest.approx(0.5)
    rates = compute_fill_rate_by_offset(np.array([True, False]), np.array([1, 7]))
    assert rates["0_5"] == pytest.approx(1.0)
    bucket = compute_flip_rate_by_delta_bucket(np.array([True, False]), np.array([2, 20]))
    assert bucket["0_5"] == pytest.approx(1.0)

