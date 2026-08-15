from datetime import datetime, timezone

import pytest

from execution_simulator.fee_schedule_registry import FeeScheduleRegistry


def test_registry_loads_without_error():
    registry = FeeScheduleRegistry("config/fee_schedules/fee_schedule_registry.yaml")
    assert "polymarket_crypto_post_20260330" in registry.schedules


def test_schedule_lookup_by_date_returns_correct_schedule():
    registry = FeeScheduleRegistry("config/fee_schedules/fee_schedule_registry.yaml")
    pre = registry.lookup(datetime(2026, 3, 29, 12, tzinfo=timezone.utc))
    post = registry.lookup(datetime(2026, 4, 19, 12, tzinfo=timezone.utc))
    assert pre.schedule_id == "polymarket_crypto_pre_20260330"
    assert post.schedule_id == "polymarket_crypto_post_20260330"


def test_no_overlapping_date_ranges():
    registry = FeeScheduleRegistry("config/fee_schedules/fee_schedule_registry.yaml")
    assert len(registry.schedules) == 2


def test_all_markets_20260419_to_20260423_return_post_schedule():
    registry = FeeScheduleRegistry("config/fee_schedules/fee_schedule_registry.yaml")
    for day in range(19, 24):
        schedule = registry.lookup(datetime(2026, 4, day, 12, tzinfo=timezone.utc))
        assert schedule.schedule_id == "polymarket_crypto_post_20260330"


def test_fee_formula_matches_documented_peak_cases():
    registry = FeeScheduleRegistry("config/fee_schedules/fee_schedule_registry.yaml")
    post = registry.lookup(datetime(2026, 4, 19, 12, tzinfo=timezone.utc))
    pre = registry.lookup(datetime(2026, 3, 29, 12, tzinfo=timezone.utc))
    assert post.compute_fee(size=1.0, price=0.5) == pytest.approx(0.018)
    assert pre.compute_fee(size=1.0, price=0.5) == pytest.approx(0.015625)

